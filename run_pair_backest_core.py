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
        # 从文件名提取币种名称，例如 BTC_ETH_1m.csv -> main: BTC, sub: ETH
        basename = os.path.basename(file).split('_')
        coin_main, coin_sub = basename[0], basename[1]

        df = pd.read_csv(file)
        df['open_time'] = pd.to_datetime(df['open_time'])
        df.set_index('open_time', inplace=True)

        # 拆分并重命名列
        df_main = df[['close_main']].rename(columns={'close_main': coin_main})
        df_sub = df[['close_sub']].rename(columns={'close_sub': coin_sub})

        dfs.extend([df_main, df_sub])

    # 横向合并所有币种，向前填充缺失值 (处理停机或不同步)
    price_df_1m = pd.concat(dfs, axis=1).sort_index().ffill()

    # 【降采样】：将 1m 数据合成为 4H 级别，取每个 4H 窗口的最后一个收盘价
    price_df_4h = price_df_1m.resample('4h').last().dropna()
    print(f"✅ 数据合并完成！共有 {len(price_df_4h)} 根 4H K线。包含币种: {list(price_df_4h.columns)}")
    return price_df_4h


# ==========================================
# 2. 策略引擎与回测逻辑
# ==========================================
def run_backtest(df, param_name="默认基准参数", custom_params=None):
    print(f"\n🚀 启动截面动量回测引擎... [{param_name}]")

    # --- 策略参数 (支持动态传入) ---
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

    TOP_K = 2  # 每次做多排名前 2 的币种
    FEE_RATE = 0.0005  # 固定的单边交易手续费 (0.05%)
    INITIAL_CAPITAL = 10000.0  # 初始本金 ($)

    coins = list(df.columns)
    if 'BTC' not in coins:
        raise ValueError("数据中必须包含 BTC 作为宏观开关！")

    # --- 预计算向量化指标 (Alpha 层) ---
    # 1. 累计收益率 (分子)
    returns = df.pct_change(MOM_WINDOW)
    # 2. 年化波动率 (分母)，这里用 4H 收益率的滚动标准差代替
    log_returns = np.log(df / df.shift(1))
    volatility = log_returns.rolling(window=VOL_WINDOW).std() * np.sqrt(365 * 6)
    # 3. 风险调整后动量
    adj_mom = returns / (volatility + 1e-8)

    # --- 预计算宏观开关 (Beta 层) ---
    btc_ma = df['BTC'].rolling(window=BTC_TREND_WINDOW).mean()
    btc_trend_on = df['BTC'] > btc_ma

    # --- 逐根 K 线步进模拟 ---
    # 从指标计算需要的最大窗口后开始遍历
    start_idx = max(MOM_WINDOW, VOL_WINDOW, BTC_TREND_WINDOW)

    # ==========================================
    # 🔴 新增模块：回测区间与标的基准涨跌幅统计
    # ==========================================
    if start_idx < len(df):
        actual_start_time = df.index[start_idx]
        actual_end_time = df.index[-1]
        print(f"\n📅 【实际交易区间】: {actual_start_time} 至 {actual_end_time}")
        print(f"📈 【区间内各标的基准涨跌幅 (Buy & Hold)】:")
        for c in coins:
            start_price = df[c].iloc[start_idx]
            end_price = df[c].iloc[-1]
            pct_change = (end_price - start_price) / start_price * 100
            print(f"   - {c}: {pct_change:+.2f}% (起点: ${start_price:.6f} -> 终点: ${end_price:.6f})")
    else:
        print("\n⚠️ 警告：数据量不足以支撑当前参数的预热窗口。")
    # ==========================================

    # --- 初始化账户 ---
    cash = INITIAL_CAPITAL
    positions = {coin: 0.0 for coin in coins}  # 持币数量
    trade_logs = []
    equity_curve = []

    for i in range(start_idx, len(df)):
        current_time = df.index[i]
        prices = df.iloc[i]

        # 计算当前总资产 (现金 + 持仓市值)
        current_equity = cash + sum(positions[c] * prices[c] for c in coins)
        equity_curve.append({'time': current_time, 'equity': current_equity})

        # 每隔 1 天 (即 6 根 4H K线) 触发一次换仓评估，避免换手过高
        if i % 6 != 0:
            continue

        target_weights = {coin: 0.0 for coin in coins}

        # 1. 判断宏观开关 (当前代码已经是 iloc[i]，零延迟逻辑)
        if btc_trend_on.iloc[i]:
            # 获取当前时刻的截面动量得分
            current_mom = adj_mom.iloc[i].dropna()
            # 筛选动量 > 0 的币种
            positive_mom = current_mom[current_mom > 0]

            if not positive_mom.empty:
                # 选取排名前 TOP_K 的币种
                top_coins = positive_mom.nlargest(TOP_K).index.tolist()

                # 2. Risk 层：逆波动率分配权重
                inv_vol = {}
                for c in top_coins:
                    c_vol = volatility[c].iloc[i]
                    inv_vol[c] = 1.0 / c_vol if c_vol > 0 else 0

                total_inv_vol = sum(inv_vol.values())

                for c in top_coins:
                    if total_inv_vol > 0:
                        raw_weight = inv_vol[c] / total_inv_vol
                        target_weights[c] = min(raw_weight, MAX_WEIGHT)  # 强制限制最大仓位

        # 3. Execution 层：执行交易 (先卖后买，释放现金)
        target_values = {c: current_equity * w for c, w in target_weights.items()}

        # [执行卖出]
        for c in coins:
            current_val = positions[c] * prices[c]
            diff_val = target_values.get(c, 0) - current_val

            if diff_val < -1.0:  # 需要卖出 (阈值 1 美金避免浮点误差导致的微小交易)
                sell_amount = abs(diff_val) / prices[c]
                # 确保持仓足够
                sell_amount = min(sell_amount, positions[c])
                actual_sell_val = sell_amount * prices[c]
                fee = actual_sell_val * FEE_RATE

                positions[c] -= sell_amount
                cash += (actual_sell_val - fee)

                trade_logs.append({
                    "time": current_time, "action": "SELL", "coin": c,
                    "price": prices[c], "amount": sell_amount, "value": actual_sell_val,
                    "fee": fee, "reason": "Target rebalance or Macro turn off"
                })

        # [执行买入]
        for c in coins:
            current_val = positions[c] * prices[c]
            diff_val = target_values.get(c, 0) - current_val

            if diff_val > 1.0:  # 需要买入
                # 考虑手续费后的可买金额
                available_to_spend = diff_val / (1 + FEE_RATE)
                if cash >= available_to_spend:
                    buy_val = available_to_spend
                else:
                    buy_val = cash / (1 + FEE_RATE)  # 现金不足时 All-in

                if buy_val > 1.0:
                    fee = buy_val * FEE_RATE
                    buy_amount = buy_val / prices[c]

                    positions[c] += buy_amount
                    cash -= (buy_val + fee)

                    trade_logs.append({
                        "time": current_time, "action": "BUY", "coin": c,
                        "price": prices[c], "amount": buy_amount, "value": buy_val,
                        "fee": fee, "reason": "Target rebalance"
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

    # 计算夏普比率 (Sharpe Ratio)，按 4H 频率调整为年化 (365天 * 每天 6 根 K线)
    curve_df['returns'] = curve_df['equity'].pct_change()
    mean_return = curve_df['returns'].mean()
    std_return = curve_df['returns'].std()
    sharpe_ratio = (mean_return / std_return * np.sqrt(365 * 6)) if std_return > 0 else 0

    # ==========================================
    # 🔴 高级交易统计 (胜率、盈亏比、持仓时间)
    # ==========================================
    win_trades = 0
    loss_trades = 0
    total_profit = 0.0
    total_loss = 0.0
    holding_times = []

    # 动态追踪各币种的开仓成本和首次建仓时间
    coin_states = {c: {'qty': 0.0, 'cost': 0.0, 'entry_time': None} for c in coins}

    for log in trade_logs:
        c = log['coin']
        action = log['action']
        amt = log['amount']
        price = log['price']
        fee = log['fee']
        time = log['time']

        if action == 'BUY':
            # 均价计算 (加权平均成本)
            old_qty = coin_states[c]['qty']
            old_cost = coin_states[c]['cost']
            new_qty = old_qty + amt
            if new_qty > 0:
                coin_states[c]['cost'] = (old_qty * old_cost + amt * price) / new_qty
            coin_states[c]['qty'] = new_qty

            # 记录此轮持仓的首次开仓时间
            if coin_states[c]['entry_time'] is None:
                coin_states[c]['entry_time'] = time

        elif action == 'SELL':
            # 计算已实现盈亏 (Realized PnL)
            cost_price = coin_states[c]['cost']
            if cost_price > 0:
                pnl = amt * (price - cost_price) - fee
                if pnl > 0:
                    win_trades += 1
                    total_profit += pnl
                else:
                    loss_trades += 1
                    total_loss += abs(pnl)

                # 计算并记录此次卖出份额的持仓时间
                if coin_states[c]['entry_time'] is not None:
                    duration = time - coin_states[c]['entry_time']
                    holding_times.append(duration)

            # 更新剩余数量
            coin_states[c]['qty'] -= amt

            # 避免浮点数精度问题，仓位极小时视为清仓，重置开仓时间和成本
            if coin_states[c]['qty'] < 1e-6:
                coin_states[c]['qty'] = 0.0
                coin_states[c]['cost'] = 0.0
                coin_states[c]['entry_time'] = None

    # 计算统计结果
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

    # 2. 你的输入列表
    file_list = ["kline_data/BTC_ETH_1m.csv", "kline_data/DOGE_SOL_1m.csv", "kline_data/TON_XRP_1m.csv"]

    # 3. 解析为 4H 数据
    df_4h = load_and_preprocess_data(file_list)

    # ==========================================
    # 循环测试参数敏感性
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

    print("\n✅ 所有参数组敏感性测试执行完毕。")