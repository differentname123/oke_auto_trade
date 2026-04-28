import os
import pandas as pd
import numpy as np
from datetime import timedelta


# ==========================================
# 1. 数据解析与特征构建
# ==========================================
def load_and_preprocess_data(file_list):
    print("⏳ 正在解析并合并数据...")
    coin_dfs = {}
    for file in file_list:
        basename = os.path.basename(file)
        coin_name = basename.split('_')[0].replace('USDT', '') if 'USDT' in basename else basename.split('_')[0]

        df = pd.read_csv(file)

        if pd.api.types.is_numeric_dtype(df['open_time']):
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        else:
            df['open_time'] = pd.to_datetime(df['open_time'])

        df.set_index('open_time', inplace=True)
        cols_to_keep = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        available_cols = [c for c in cols_to_keep if c in df.columns]
        df = df[available_cols].copy().sort_index()
        df = df[~df.index.duplicated(keep='last')]
        coin_dfs[coin_name] = df

    common_start = max(df.index.min() for df in coin_dfs.values())
    common_end = min(df.index.max() for df in coin_dfs.values())
    print(f"✅ 公共时间窗口: {common_start} 至 {common_end}")

    for coin in coin_dfs:
        coin_dfs[coin] = coin_dfs[coin].loc[common_start:common_end]

    aggregated = {}
    for coin, df in coin_dfs.items():
        df = df.ffill()
        agg = pd.DataFrame()
        agg['open'] = df['open'].resample('4h').first()
        agg['high'] = df['high'].resample('4h').max()
        agg['low'] = df['low'].resample('4h').min()
        agg['close'] = df['close'].resample('4h').last()
        aggregated[coin] = agg

    coins = list(aggregated.keys())
    feature_panels = {}
    for feat in ['open', 'high', 'low', 'close']:
        feature_panels[feat] = pd.concat(
            [aggregated[c][feat].rename(c) for c in coins], axis=1
        ).ffill()

    return feature_panels


# ==========================================
# 2. 核心：理想成交策略引擎 (消除延迟, 绑定 Close)
# ==========================================
def run_backtest_ideal(panels, param_name="理论极限版", custom_params=None, verbose=True):
    if verbose:
        print(f"\n🚀 启动理想成交回测引擎 (收盘价秒进秒出)... [{param_name}]")

    if custom_params is None:
        custom_params = {
            'MOM_WINDOW': 20 * 6,
            'BTC_TREND_WINDOW': 90 * 6,
            'MAX_WEIGHT': 0.30,
            'ATR_WINDOW': 14,
            'ATR_MULTI': 2.5,
            'DECAY_TOLERANCE': 2,
            'DEGRAD_WINDOW': 3
        }

    MOM_WINDOW = custom_params['MOM_WINDOW']
    BTC_TREND_WINDOW = custom_params['BTC_TREND_WINDOW']
    MAX_WEIGHT = custom_params['MAX_WEIGHT']
    ATR_W = custom_params['ATR_WINDOW']
    ATR_MULTI = custom_params['ATR_MULTI']
    DECAY_TOL = custom_params['DECAY_TOLERANCE']
    DEGRAD_W = custom_params['DEGRAD_WINDOW']

    TOP_K = 2
    FEE_RATE = 0.0005
    INITIAL_CAPITAL = 10000.0

    close_df = panels['close']
    high_df = panels['high']
    low_df = panels['low']

    coins = list(close_df.columns)
    if 'BTC' not in coins:
        raise ValueError("数据中必须包含 BTC 作为宏观开关！")

    # --- 因子计算 ---
    abs_change = (close_df - close_df.shift(MOM_WINDOW)).abs()
    sum_abs_diff = (close_df - close_df.shift(1)).abs().rolling(MOM_WINDOW).sum()
    ER = abs_change / (sum_abs_diff + 1e-8)

    tr1 = high_df - low_df
    tr2 = (high_df - close_df.shift(1)).abs()
    tr3 = (low_df - close_df.shift(1)).abs()
    TR = pd.DataFrame(np.maximum(np.maximum(tr1.values, tr2.values), tr3.values),
                      index=close_df.index, columns=coins)
    ATR = TR.ewm(alpha=1 / ATR_W, adjust=False).mean()

    returns = close_df.pct_change(MOM_WINDOW)
    ATR_pct = ATR / close_df
    adj_mom = returns / (ATR_pct + 1e-8)

    btc_ma = close_df['BTC'].rolling(window=BTC_TREND_WINDOW).mean()
    btc_trend_on = close_df['BTC'] > btc_ma

    cross_median = returns.median(axis=1)
    macro_degrad = cross_median.rolling(DEGRAD_W).max() < 0

    # ==================================================
    # ⚠️ 关键修改：取消 shift(1)，直接使用当期信号测试极限表现
    # ==================================================
    sig_ER = ER
    sig_adj_mom = adj_mom
    sig_ATR = ATR
    sig_btc_trend = btc_trend_on
    sig_macro_degrad = macro_degrad

    cash = INITIAL_CAPITAL
    positions = {c: 0.0 for c in coins}
    trade_logs = []
    equity_curve = []

    highest_high = {c: 0.0 for c in coins}
    out_of_top_k_count = {c: 0 for c in coins}

    start_idx = max(MOM_WINDOW, BTC_TREND_WINDOW, ATR_W)

    for i in range(start_idx, len(close_df)):
        current_time = close_df.index[i]

        # ⚠️ 关键修改：提取当期 Close 作为唯一的执行价格
        curr_close = close_df.iloc[i]
        curr_high = high_df.iloc[i]

        s_er = sig_ER.iloc[i]
        s_mom = sig_adj_mom.iloc[i]
        s_atr = sig_ATR.iloc[i]
        s_btc = sig_btc_trend.iloc[i]
        s_degrad = sig_macro_degrad.iloc[i]

        current_equity = cash + sum(positions[c] * curr_close[c] for c in coins)
        equity_curve.append({'time': current_time, 'equity': current_equity})

        for c in coins:
            if positions[c] > 0:
                highest_high[c] = max(highest_high[c], curr_high[c])

        top_coins = []
        if s_btc and not s_degrad:
            valid_mask = (s_er >= 0.2) & (s_mom > 0)
            if valid_mask.any():
                valid_mom = s_mom[valid_mask]
                top_coins = valid_mom.nlargest(TOP_K).index.tolist()

        for c in coins:
            if positions[c] > 0:
                if c not in top_coins:
                    out_of_top_k_count[c] += 1
                else:
                    out_of_top_k_count[c] = 0
            else:
                out_of_top_k_count[c] = 0

        # [SELL 卖出逻辑]
        for c in coins:
            if positions[c] > 0:
                # ⚠️ 关键修改：直接以产生信号的收盘价成交
                exec_price = curr_close[c]
                sell_flag = False
                reason = ""

                if s_degrad:
                    sell_flag = True
                    reason = "宏观退化"
                elif out_of_top_k_count[c] >= DECAY_TOL:
                    sell_flag = True
                    reason = f"动量衰减退出"
                elif curr_close[c] < (highest_high[c] - ATR_MULTI * s_atr[c]):
                    sell_flag = True
                    reason = f"ATR 跟踪止损"
                    # 理论极限下，止损也强行设定为当期收盘价（最乐观假设）
                    exec_price = curr_close[c]

                if sell_flag:
                    sell_val = positions[c] * exec_price
                    fee = sell_val * FEE_RATE
                    cash += (sell_val - fee)
                    trade_logs.append({
                        "time": current_time, "action": "SELL", "coin": c,
                        "price": exec_price, "amount": positions[c], "value": sell_val,
                        "fee": fee, "reason": reason
                    })
                    positions[c] = 0.0
                    highest_high[c] = 0.0
                    out_of_top_k_count[c] = 0

        # [BUY 买入逻辑]
        if s_btc and not s_degrad:
            inv_atr = {c: 1.0 / s_atr[c] for c in top_coins if s_atr[c] > 0}
            total_inv_atr = sum(inv_atr.values())

            for c in top_coins:
                if total_inv_atr > 0:
                    raw_w = inv_atr[c] / total_inv_atr
                    target_w = min(raw_w, MAX_WEIGHT)
                    curr_w = (positions[c] * curr_close[c]) / current_equity if current_equity > 0 else 0.0

                    if curr_w < 0.5 * target_w:
                        target_value = current_equity * target_w
                        diff_value = target_value - (positions[c] * curr_close[c])

                        if diff_value > 1.0:
                            avail_cash = diff_value / (1 + FEE_RATE)
                            buy_val = min(avail_cash, cash / (1 + FEE_RATE))

                            if buy_val > 1.0:
                                # ⚠️ 关键修改：直接以产生信号的收盘价成交
                                exec_price = curr_close[c]
                                fee = buy_val * FEE_RATE
                                buy_amt = buy_val / exec_price

                                positions[c] += buy_amt
                                cash -= (buy_val + fee)
                                trade_logs.append({
                                    "time": current_time, "action": "BUY", "coin": c,
                                    "price": exec_price, "amount": buy_amt, "value": buy_val,
                                    "fee": fee, "reason": "目标调仓"
                                })
                                if highest_high[c] == 0:
                                    highest_high[c] = curr_high[c]

    final_equity = cash + sum(positions[c] * close_df.iloc[-1][c] for c in coins)
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL

    if not equity_curve:
        return pd.DataFrame(), pd.DataFrame()

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
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')

    # 交易统计计算
    win_trades, loss_trades = 0, 0
    total_profit, total_loss = 0.0, 0.0
    holding_times = []
    coin_states = {c: {'qty': 0.0, 'cost': 0.0, 'entry_time': None} for c in coins}

    for log in trade_logs:
        c, action, amt, price, fee, time = log['coin'], log['action'], log['amount'], log['price'], log['fee'], log[
            'time']

        if action == 'BUY':
            old_qty, old_cost = coin_states[c]['qty'], coin_states[c]['cost']
            new_qty = old_qty + amt
            if new_qty > 0:
                coin_states[c]['cost'] = (old_qty * old_cost + amt * price) / new_qty
            coin_states[c]['qty'] = new_qty
            if coin_states[c]['entry_time'] is None:
                coin_states[c]['entry_time'] = time

        elif action == 'SELL':
            cost_price = coin_states[c]['cost']
            if cost_price > 0:
                pnl = amt * (price - cost_price) - fee
                if pnl > 0:
                    win_trades += 1
                    total_profit += pnl
                else:
                    loss_trades += 1
                    total_loss += abs(pnl)

                if coin_states[c]['entry_time'] is not None:
                    holding_times.append(time - coin_states[c]['entry_time'])

            coin_states[c]['qty'] -= amt
            if coin_states[c]['qty'] < 1e-6:
                coin_states[c]['qty'], coin_states[c]['cost'], coin_states[c]['entry_time'] = 0.0, 0.0, None

    total_closed_trades = win_trades + loss_trades
    win_rate = win_trades / total_closed_trades if total_closed_trades > 0 else 0.0
    avg_profit = total_profit / win_trades if win_trades > 0 else 0.0
    avg_loss = total_loss / loss_trades if loss_trades > 0 else 0.0
    profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else float('inf')
    avg_holding_time = sum(holding_times, timedelta()) / len(holding_times) if holding_times else timedelta(0)

    if verbose:
        print("\n" + "=" * 45)
        print(f"📊 【理论天花板面板】: {param_name}")
        print("-" * 45)
        print(f"💸 [资金与收益]")
        print(f"  最终资金:     ${final_equity:.2f}")
        print(f"  年化收益率:   {annual_return * 100:.2f}%")
        print(f"🛡️ [风险与绩效指标]")
        print(f"  最大回撤:     {max_drawdown * 100:.2f}%")
        print(f"  夏普比率:     {sharpe_ratio:.2f}")
        print(f"  卡玛比率:     {calmar_ratio:.2f}")
        print(f"⚖️ [交易统计]")
        print(f"  有效平仓笔数: {total_closed_trades} 笔")
        if total_closed_trades > 0:
            print(f"  胜率 (Win%):  {win_rate * 100:.2f}%")
            print(f"  盈亏比 (P/L): {profit_loss_ratio:.2f}")
            print(f"  平均持仓时间: {avg_holding_time}")
        print("=" * 45)

    return pd.DataFrame(trade_logs), curve_df


# ==========================================
# 主程序执行入口
# ==========================================
if __name__ == "__main__":
    file_list = [
        "kline_data/BTCUSDT_1m_merged.csv",
        "kline_data/ETHUSDT_1m_merged.csv",
        "kline_data/DOGEUSDT_1m_merged.csv",
        "kline_data/SOLUSDT_1m_merged.csv",
        "kline_data/TONUSDT_1m_merged.csv",
        "kline_data/XRPUSDT_1m_merged.csv"
    ]

    feature_panels = load_and_preprocess_data(file_list)

    # 测试最优参数在“理论极限”下的表现
    optimal_params = {
        'MOM_WINDOW': 20 * 6,
        'BTC_TREND_WINDOW': 90 * 6,
        'MAX_WEIGHT': 0.30,
        'ATR_WINDOW': 14,
        'ATR_MULTI': 2.5,
        'DECAY_TOLERANCE': 2,
        'DEGRAD_WINDOW': 3
    }

    print("\n" + "*" * 60)
    print(" ⚠️ 警告：当前正在运行【无延迟/零滑点】的理想化回测")
    print(" 此结果仅用于验证 Alpha 信号的理论上限，不可直接用于实盘预期！")
    print("*" * 60)

    logs_df, curve_df = run_backtest_ideal(
        feature_panels,
        param_name="无延迟收盘价成交 (理论最高值)",
        custom_params=optimal_params
    )