import pandas as pd
import numpy as np
import os
from datetime import timedelta


# ==========================================
# 1. 数据解析与合成模块
# ==========================================
def load_and_preprocess_data(file_list):
    print("⏳ 正在解析并合并数据...")
    dfs = []
    for file in file_list:
        basename = os.path.basename(file).split('_')
        coin_main, coin_sub = basename[0], basename[1]

        df = pd.read_csv(file)
        df['open_time'] = pd.to_datetime(df['open_time'])
        df.set_index('open_time', inplace=True)

        df_main = df[['close_main']].rename(columns={'close_main': coin_main})
        df_sub = df[['close_sub']].rename(columns={'close_sub': coin_sub})

        dfs.extend([df_main, df_sub])

    price_df_1m = pd.concat(dfs, axis=1).sort_index().ffill()
    price_df_4h = price_df_1m.resample('4h').last().dropna()
    print(f"✅ 数据合并完成！共有 {len(price_df_4h)} 根 4H K线。包含币种: {list(price_df_4h.columns)}")
    return price_df_4h


# ==========================================
# 2. 策略引擎与回测逻辑 (支持动态参数注入)
# ==========================================
def run_backtest(df, param_name="默认基准参数", custom_params=None):
    print(f"\n🚀 启动截面动量(做空)回测引擎... [{param_name}]")

    # --- 策略参数 (动态支持) ---
    if custom_params is None:
        custom_params = {
            'MOM_WINDOW': 20 * 6,
            'VOL_WINDOW': 20 * 6,
            'BTC_TREND_WINDOW': 60 * 6,
            'MAX_WEIGHT': 0.30
        }

    MOM_WINDOW = custom_params['MOM_WINDOW']
    VOL_WINDOW = custom_params['VOL_WINDOW']
    BTC_TREND_WINDOW = custom_params['BTC_TREND_WINDOW']
    MAX_WEIGHT = custom_params['MAX_WEIGHT']
    # MAX_WEIGHT = 1
    TOP_K = 2  # 每次做空排名前 2 的跌幅币种
    FEE_RATE = 0.0005  # 统一修改为真实的万分之五 Taker 手续费
    INITIAL_CAPITAL = 10000.0  # 初始本金 ($)

    coins = list(df.columns)
    if 'BTC' not in coins:
        raise ValueError("数据中必须包含 BTC 作为宏观开关！")

    returns = df.pct_change(MOM_WINDOW)
    log_returns = np.log(df / df.shift(1))
    volatility = log_returns.rolling(window=VOL_WINDOW).std() * np.sqrt(365 * 6)
    adj_mom = returns / (volatility + 1e-8)

    btc_ma = df['BTC'].rolling(window=BTC_TREND_WINDOW).mean()
    btc_trend_off = df['BTC'] < btc_ma

    cash = INITIAL_CAPITAL
    positions = {coin: 0.0 for coin in coins}
    trade_logs = []
    equity_curve = []

    start_idx = max(MOM_WINDOW, VOL_WINDOW, BTC_TREND_WINDOW)

    for i in range(start_idx, len(df)):
        current_time = df.index[i]
        prices = df.iloc[i]

        current_equity = cash + sum(positions[c] * prices[c] for c in coins)
        equity_curve.append({'time': current_time, 'equity': current_equity})

        if i % 6 != 0:
            continue

        target_weights = {coin: 0.0 for coin in coins}

        # 🔴 零延迟：直接使用当前的 i 获取信号，并在同一刻用 prices 执行
        if btc_trend_off.iloc[i]:
            current_mom = adj_mom.iloc[i].dropna()

            # 寻找动量为负的币种
            negative_mom = current_mom[current_mom < 0]

            if not negative_mom.empty:
                # 选取跌得最惨的 TOP_K 个币种
                top_coins = negative_mom.nsmallest(TOP_K).index.tolist()

                inv_vol = {}
                for c in top_coins:
                    c_vol = volatility[c].iloc[i]  # 零延迟
                    inv_vol[c] = 1.0 / c_vol if c_vol > 0 else 0

                total_inv_vol = sum(inv_vol.values())

                for c in top_coins:
                    if total_inv_vol > 0:
                        raw_weight = inv_vol[c] / total_inv_vol
                        target_weights[c] = -min(raw_weight, MAX_WEIGHT)  # 🔴 设置为负数，代表做空目标

        target_values = {c: current_equity * w for c, w in target_weights.items()}

        # [执行卖出：开空 (Open Short) 或 减仓做多]
        for c in coins:
            current_val = positions[c] * prices[c]
            diff_val = target_values.get(c, 0) - current_val

            if diff_val < -1.0:
                sell_amount = abs(diff_val) / prices[c]
                actual_sell_val = sell_amount * prices[c]
                fee = actual_sell_val * FEE_RATE

                positions[c] -= sell_amount
                cash += (actual_sell_val - fee)

                trade_logs.append({
                    "time": current_time, "action": "SELL (SHORT)", "coin": c,
                    "price": prices[c], "amount": sell_amount, "value": actual_sell_val,
                    "fee": fee, "reason": "Open Short / Macro Off"
                })

        # [执行买入：平空 (Cover Short) 或 开多]
        for c in coins:
            current_val = positions[c] * prices[c]
            diff_val = target_values.get(c, 0) - current_val

            if diff_val > 1.0:
                buy_val = diff_val
                fee = buy_val * FEE_RATE
                buy_amount = buy_val / prices[c]

                positions[c] += buy_amount
                cash -= (buy_val + fee)

                trade_logs.append({
                    "time": current_time, "action": "BUY (COVER)", "coin": c,
                    "price": prices[c], "amount": buy_amount, "value": buy_val,
                    "fee": fee, "reason": "Cover Short"
                })

    # --- 结果与指标统计 ---
    final_equity = cash + sum(positions[c] * df.iloc[-1][c] for c in coins)
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # 转换为 DataFrame 方便计算高级指标
    curve_df = pd.DataFrame(equity_curve).set_index('time')

    # 计算最大回撤 (Max Drawdown)
    curve_df['cum_max'] = curve_df['equity'].cummax()
    curve_df['drawdown'] = (curve_df['equity'] - curve_df['cum_max']) / curve_df['cum_max']
    max_drawdown = curve_df['drawdown'].min()

    # 计算年化收益率 (Annualized Return)
    days_passed = (curve_df.index[-1] - curve_df.index[0]).days
    if days_passed > 0:
        annual_return = (final_equity / INITIAL_CAPITAL) ** (365 / days_passed) - 1
    else:
        annual_return = 0.0

    # 计算夏普比率 (Sharpe Ratio)，按 4H 频率调整为年化 (365天 * 每天6根K线)
    curve_df['returns'] = curve_df['equity'].pct_change()
    mean_return = curve_df['returns'].mean()
    std_return = curve_df['returns'].std()
    sharpe_ratio = (mean_return / std_return * np.sqrt(365 * 6)) if std_return > 0 else 0

    # ==========================================
    # 🔴 新增模块：高级交易统计 (做空适配版)
    # ==========================================
    win_trades = 0
    loss_trades = 0
    total_profit = 0.0
    total_loss = 0.0
    holding_times = []

    # 动态追踪各币种的开仓成本和首次建仓时间
    # 注意：做空逻辑下，记录的 qty 为负数或取绝对值计算
    coin_states = {c: {'qty': 0.0, 'cost': 0.0, 'entry_time': None} for c in coins}

    for log in trade_logs:
        c = log['coin']
        action = log['action']
        amt = log['amount']
        price = log['price']
        fee = log['fee']
        time = log['time']

        if action == 'SELL (SHORT)':
            # 开空/加空：计算加权均价 (使用仓位绝对值处理)
            old_qty = abs(coin_states[c]['qty'])
            old_cost = coin_states[c]['cost']
            new_qty = old_qty + amt

            if new_qty > 0:
                coin_states[c]['cost'] = (old_qty * old_cost + amt * price) / new_qty

            # 做空状态下，内部实际 qty 记为负数以追踪余额
            coin_states[c]['qty'] -= amt

            # 记录此轮空头持仓的首次开仓时间
            if coin_states[c]['entry_time'] is None:
                coin_states[c]['entry_time'] = time

        elif action == 'BUY (COVER)':
            # 平空/减空：计算已实现盈亏
            cost_price = coin_states[c]['cost']
            if cost_price > 0:
                # 🔴 核心反转：做空利润 = (开仓均价 - 当前平仓价) * 数量 - 手续费
                pnl = amt * (cost_price - price) - fee

                if pnl > 0:
                    win_trades += 1
                    total_profit += pnl
                else:
                    loss_trades += 1
                    total_loss += abs(pnl)

                # 计算持仓时间
                if coin_states[c]['entry_time'] is not None:
                    duration = time - coin_states[c]['entry_time']
                    holding_times.append(duration)

            # 更新剩余空头数量 (加上 amt 使其趋向于 0)
            coin_states[c]['qty'] += amt

            # 清理微小浮点误差，视为完全平仓
            if abs(coin_states[c]['qty']) < 1e-6:
                coin_states[c]['qty'] = 0.0
                coin_states[c]['cost'] = 0.0
                coin_states[c]['entry_time'] = None

    # 统计结算
    total_closed_trades = win_trades + loss_trades
    win_rate = win_trades / total_closed_trades if total_closed_trades > 0 else 0.0
    avg_profit = total_profit / win_trades if win_trades > 0 else 0.0
    avg_loss = total_loss / loss_trades if loss_trades > 0 else 0.0
    profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else float('inf')

    if holding_times:
        avg_holding_time = sum(holding_times, timedelta()) / len(holding_times)
    else:
        avg_holding_time = timedelta(0)

    print("\n📊 === 回测结果核心指标 ===")
    print(f"初始资金:     ${INITIAL_CAPITAL:.2f}")
    print(f"最终资金:     ${final_equity:.2f}")
    print(f"总收益率:     {total_return * 100:.2f}%")
    print(f"年化收益率:   {annual_return * 100:.2f}%")
    print(f"最大回撤:     {max_drawdown * 100:.2f}%")
    print(f"夏普比率:     {sharpe_ratio:.2f}")
    print("-" * 25)
    print(f"总触发操作:   {len(trade_logs)} 次")
    print(f"有效平仓笔数: {total_closed_trades} 笔")
    print(f"胜率 (Win%):  {win_rate * 100:.2f}%")
    print(f"盈亏比 (P/L): {profit_loss_ratio:.2f}")
    print(f"平均持仓时间: {avg_holding_time}")

    return pd.DataFrame(trade_logs), curve_df


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":

    file_list = ["kline_data/BTC_ETH_1m.csv", "kline_data/DOGE_SOL_1m.csv", "kline_data/TON_XRP_1m.csv"]
    df_4h = load_and_preprocess_data(file_list)

    # ==========================================
    # 🔴 循环测试参数敏感性 (做空组)
    # ==========================================
    test_scenarios = [
        {
            "name": "基准参数 (最初设定)",
            "params": {'MOM_WINDOW': 20 * 6, 'VOL_WINDOW': 20 * 6, 'BTC_TREND_WINDOW': 60 * 6, 'MAX_WEIGHT': 0.30}
        },
        {
            "name": "挑战组 1：短期敏捷神经质 (减半周期)",
            "params": {'MOM_WINDOW': 10 * 6, 'VOL_WINDOW': 10 * 6, 'BTC_TREND_WINDOW': 30 * 6, 'MAX_WEIGHT': 0.30}
        },
        {
            "name": "挑战组 2：长期宏观迟钝 (拉长周期)",
            "params": {'MOM_WINDOW': 30 * 6, 'VOL_WINDOW': 30 * 6, 'BTC_TREND_WINDOW': 90 * 6, 'MAX_WEIGHT': 0.30}
        },
        {
            "name": "挑战组 3：风控资本极度受限 (减半仓位)",
            "params": {'MOM_WINDOW': 20 * 6, 'VOL_WINDOW': 20 * 6, 'BTC_TREND_WINDOW': 60 * 6, 'MAX_WEIGHT': 0.15}
        }
    ]

    # 依次执行各组参数
    for scenario in test_scenarios:
        logs_df, curve_df = run_backtest(df_4h, param_name=scenario["name"], custom_params=scenario["params"])

    print("\n✅ 所有做空参数组敏感性测试执行完毕。")