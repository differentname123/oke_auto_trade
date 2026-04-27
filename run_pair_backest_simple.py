import pandas as pd
import numpy as np
import os
from datetime import timedelta

from data_utils import load_and_preprocess_data



# ==========================================
# 核心：策略引擎与回测逻辑 (做空版)
# ==========================================
def run_backtest(df, param_name="默认基准参数", custom_params=None):
    print(f"\n🚀 启动截面动量回测引擎... [做空组 - {param_name}]")

    # --- 策略参数 ---
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

    TOP_K = 2
    FEE_RATE = 0.0005
    INITIAL_CAPITAL = 10000.0

    coins = list(df.columns)
    if 'BTC' not in coins:
        raise ValueError("数据中必须包含 BTC 作为宏观开关！")

    # --- 预计算向量化指标 (Alpha 层) ---
    returns = df.pct_change(MOM_WINDOW)
    log_returns = np.log(df / df.shift(1))
    volatility = log_returns.rolling(window=VOL_WINDOW).std() * np.sqrt(365 * 6)
    adj_mom = returns / (volatility + 1e-8)

    # --- 预计算宏观开关 (Beta 层 - 做空看跌开关) ---
    btc_ma = df['BTC'].rolling(window=BTC_TREND_WINDOW).mean()
    btc_trend_off = df['BTC'] < btc_ma

    # --- 初始化账户 ---
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

        # 1. 截面动量打分 (寻找跌幅最大的币种)
        if btc_trend_off.iloc[i]:
            current_mom = adj_mom.iloc[i].dropna()
            negative_mom = current_mom[current_mom < 0]

            if not negative_mom.empty:
                top_coins = negative_mom.nsmallest(TOP_K).index.tolist()

                # 2. 逆波动率分配权重
                inv_vol = {}
                for c in top_coins:
                    c_vol = volatility[c].iloc[i]
                    inv_vol[c] = 1.0 / c_vol if c_vol > 0 else 0

                total_inv_vol = sum(inv_vol.values())

                for c in top_coins:
                    if total_inv_vol > 0:
                        raw_weight = inv_vol[c] / total_inv_vol
                        target_weights[c] = -min(raw_weight, MAX_WEIGHT)  # 🔴 负数代表做空权重

        # 3. 执行交易 (先买后卖：先平空释放保证金，再开新空单)
        target_values = {c: current_equity * w for c, w in target_weights.items()}

        # [买入：平空 (Cover Short) 或 减空]
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

        # [卖出：开空 (Open Short) 或 加空]
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
                    "fee": fee, "reason": "Open Short"
                })

    # ==========================================
    # 🔴 核心指标与高级统计计算 (严格对齐做多代码)
    # ==========================================
    final_equity = cash + sum(positions[c] * df.iloc[-1][c] for c in coins)
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL

    curve_df = pd.DataFrame(equity_curve).set_index('time')
    curve_df['cum_max'] = curve_df['equity'].cummax()
    curve_df['drawdown'] = (curve_df['equity'] - curve_df['cum_max']) / curve_df['cum_max']
    max_drawdown = curve_df['drawdown'].min()

    days_passed = (curve_df.index[-1] - curve_df.index[0]).days
    annual_return = ((final_equity / INITIAL_CAPITAL) ** (365 / days_passed) - 1) if days_passed > 0 else 0.0

    curve_df['returns'] = curve_df['equity'].pct_change()
    mean_return = curve_df['returns'].mean()
    std_return = curve_df['returns'].std()
    sharpe_ratio = (mean_return / std_return * np.sqrt(365 * 6)) if std_return > 0 else 0

    # 新增：卡玛比率 (年化收益/最大回撤)
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')

    win_trades, loss_trades = 0, 0
    total_profit, total_loss = 0.0, 0.0
    holding_times = []
    coin_states = {c: {'qty': 0.0, 'cost': 0.0, 'entry_time': None} for c in coins}

    for log in trade_logs:
        c, action, amt, price, fee, time = log['coin'], log['action'], log['amount'], log['price'], log['fee'], log['time']

        if action == 'SELL (SHORT)':
            old_qty = abs(coin_states[c]['qty'])
            old_cost = coin_states[c]['cost']
            new_qty = old_qty + amt

            if new_qty > 0:
                coin_states[c]['cost'] = (old_qty * old_cost + amt * price) / new_qty

            coin_states[c]['qty'] -= amt
            if coin_states[c]['entry_time'] is None:
                coin_states[c]['entry_time'] = time

        elif action == 'BUY (COVER)':
            cost_price = coin_states[c]['cost']
            if cost_price > 0:
                # 🔴 做空利润 = (开仓均价 - 当前平仓价) * 数量 - 手续费
                pnl = amt * (cost_price - price) - fee
                if pnl > 0:
                    win_trades += 1
                    total_profit += pnl
                else:
                    loss_trades += 1
                    total_loss += abs(pnl)

                if coin_states[c]['entry_time'] is not None:
                    holding_times.append(time - coin_states[c]['entry_time'])

            coin_states[c]['qty'] += amt
            if abs(coin_states[c]['qty']) < 1e-6:
                coin_states[c]['qty'], coin_states[c]['cost'], coin_states[c]['entry_time'] = 0.0, 0.0, None

    total_closed_trades = win_trades + loss_trades
    win_rate = win_trades / total_closed_trades if total_closed_trades > 0 else 0.0
    avg_profit = total_profit / win_trades if win_trades > 0 else 0.0
    avg_loss = total_loss / loss_trades if loss_trades > 0 else 0.0
    profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else float('inf')
    avg_holding_time = sum(holding_times, timedelta()) / len(holding_times) if holding_times else timedelta(0)

    # 🔴 恢复并增强输出面板 (与做多代码完全一致)
    print("\n" + "=" * 45)
    print(f"📊 【测试结果面板】: {param_name}")
    print("-" * 45)
    print(f"💸 [资金与收益]")
    print(f"  初始资金:     ${INITIAL_CAPITAL:.2f}")
    print(f"  最终资金:     ${final_equity:.2f}")
    print(f"  总收益率:     {total_return * 100:.2f}%")
    print(f"  年化收益率:   {annual_return * 100:.2f}%")
    print(f"🛡️ [风险与绩效指标]")
    print(f"  最大回撤:     {max_drawdown * 100:.2f}%")
    print(f"  夏普比率:     {sharpe_ratio:.2f}")
    print(f"  卡玛比率:     {calmar_ratio:.2f}")
    print(f"⚖️ [交易统计]")
    print(f"  总触发动作:   {len(trade_logs)} 次")
    print(f"  有效平仓笔数: {total_closed_trades} 笔")
    if total_closed_trades > 0:
        print(f"  胜率 (Win%):  {win_rate * 100:.2f}%")
        print(f"  盈亏比 (P/L): {profit_loss_ratio:.2f}")
        print(f"  单笔均盈:     ${avg_profit:.2f}")
        print(f"  单笔均亏:     ${avg_loss:.2f}")
        print(f"  平均持仓时间: {avg_holding_time}")
    else:
        print("  (无有效平仓记录)")
    print("=" * 45)

    return pd.DataFrame(trade_logs), curve_df


# ==========================================
# 主程序执行入口
# ==========================================
if __name__ == "__main__":

    file_list = ["kline_data/BTC_ETH_1m.csv", "kline_data/DOGE_SOL_1m.csv", "kline_data/TON_XRP_1m.csv"]
    df_4h = load_and_preprocess_data(file_list)

    test_scenarios = [
        {"name": "基准参数",
         "params": {'MOM_WINDOW': 20 * 6, 'VOL_WINDOW': 20 * 6, 'BTC_TREND_WINDOW': 60 * 6, 'MAX_WEIGHT': 0.30}},
        {"name": "挑战组 1 (短期)",
         "params": {'MOM_WINDOW': 10 * 6, 'VOL_WINDOW': 10 * 6, 'BTC_TREND_WINDOW': 30 * 6, 'MAX_WEIGHT': 0.30}},
        {"name": "挑战组 2 (长期)",
         "params": {'MOM_WINDOW': 30 * 6, 'VOL_WINDOW': 30 * 6, 'BTC_TREND_WINDOW': 90 * 6, 'MAX_WEIGHT': 0.30}},
        {"name": "挑战组 3 (低仓)",
         "params": {'MOM_WINDOW': 20 * 6, 'VOL_WINDOW': 20 * 6, 'BTC_TREND_WINDOW': 60 * 6, 'MAX_WEIGHT': 0.15}}
    ]

    for scenario in test_scenarios:
        logs_df, curve_df = run_backtest(df_4h, param_name=scenario["name"], custom_params=scenario["params"])

    print("\n✅ 所有做空参数组敏感性测试执行完毕。")