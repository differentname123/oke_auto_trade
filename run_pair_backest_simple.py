import pandas as pd
import numpy as np
import os
from datetime import timedelta


# ==========================================
# 0. 准备测试数据 (为了让你拿到就能跑，实盘请删掉这部分)
# ==========================================
def generate_mock_data():
    os.makedirs("kline_data", exist_ok=True)
    dates = pd.date_range("2023-01-01", "2023-06-01", freq="1min")
    np.random.seed(42)

    # 模拟 BTC 和 ETH
    btc_prices = 20000 * np.exp(np.cumsum(np.random.normal(0.00001, 0.001, len(dates))))
    eth_prices = 1500 * np.exp(np.cumsum(np.random.normal(0.000012, 0.0015, len(dates))))
    df_btc_eth = pd.DataFrame({"open_time": dates, "close_main": btc_prices, "close_sub": eth_prices})
    df_btc_eth.to_csv("kline_data/BTC_ETH_1m.csv", index=False)

    # 模拟 DOGE 和 SOL
    doge_prices = 0.08 * np.exp(np.cumsum(np.random.normal(0.000005, 0.002, len(dates))))
    sol_prices = 20 * np.exp(np.cumsum(np.random.normal(0.000015, 0.0025, len(dates))))
    df_doge_sol = pd.DataFrame({"open_time": dates, "close_main": doge_prices, "close_sub": sol_prices})
    df_doge_sol.to_csv("kline_data/DOGE_SOL_1m.csv", index=False)

    # 模拟 TON 和 XRP
    ton_prices = 2 * np.exp(np.cumsum(np.random.normal(-0.00001, 0.002, len(dates))))
    xrp_prices = 0.5 * np.exp(np.cumsum(np.random.normal(-0.00002, 0.0018, len(dates))))
    df_ton_xrp = pd.DataFrame({"open_time": dates, "close_main": ton_prices, "close_sub": xrp_prices})
    df_ton_xrp.to_csv("kline_data/TON_XRP_1m.csv", index=False)

    print("✅ 模拟 1分钟 K 线数据生成完毕！")


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
# 2. 策略引擎与回测逻辑 (零延迟 + 详尽统计版)
# ==========================================
def run_backtest(df):
    print("\n🚀 启动截面动量(做空)回测引擎...")

    MOM_WINDOW = 20 * 6
    VOL_WINDOW = 20 * 6
    BTC_TREND_WINDOW = 60 * 6
    MAX_WEIGHT = 0.30
    TOP_K = 2
    FEE_RATE = 0.0005
    INITIAL_CAPITAL = 10000.0

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

        # 🔴 改回零延迟：直接使用当前的 i 获取信号，并在同一刻用 prices 执行
        if btc_trend_off.iloc[i]:
            current_mom = adj_mom.iloc[i].dropna()

            negative_mom = current_mom[current_mom < 0]

            if not negative_mom.empty:
                top_coins = negative_mom.nsmallest(TOP_K).index.tolist()

                inv_vol = {}
                for c in top_coins:
                    c_vol = volatility[c].iloc[i]  # 零延迟
                    inv_vol[c] = 1.0 / c_vol if c_vol > 0 else 0

                total_inv_vol = sum(inv_vol.values())

                for c in top_coins:
                    if total_inv_vol > 0:
                        raw_weight = inv_vol[c] / total_inv_vol
                        target_weights[c] = -min(raw_weight, MAX_WEIGHT)

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

    print("\n📊 === 回测结果核心指标 ===")
    print(f"初始资金:     ${INITIAL_CAPITAL:.2f}")
    print(f"最终资金:     ${final_equity:.2f}")
    print(f"总收益率:     {total_return * 100:.2f}%")
    print(f"年化收益率:   {annual_return * 100:.2f}%")
    print(f"最大回撤:     {max_drawdown * 100:.2f}%")
    print(f"夏普比率:     {sharpe_ratio:.2f}")
    print(f"总交易笔数:   {len(trade_logs)} 笔")

    return pd.DataFrame(trade_logs), curve_df


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":

    file_list = ["kline_data/BTC_ETH_1m.csv", "kline_data/DOGE_SOL_1m.csv", "kline_data/TON_XRP_1m.csv"]
    df_4h = load_and_preprocess_data(file_list)
    logs_df, curve_df = run_backtest(df_4h)

    print("\n📝 === 最近 10 笔交易日志 ===")
    if not logs_df.empty:
        display_logs = logs_df.tail(10).copy()
        display_logs['price'] = display_logs['price'].apply(lambda x: f"${x:.4f}")
        display_logs['value'] = display_logs['value'].apply(lambda x: f"${x:.2f}")
        display_logs['fee'] = display_logs['fee'].apply(lambda x: f"${x:.2f}")
        print(display_logs.to_string(index=False))
    else:
        print("未发生任何交易 (可能是因为宏观开关一直未触发，或动量全为负)。")