import os
import pandas as pd
import numpy as np
from datetime import timedelta


# ==========================================
# 数据加载与预处理 (严格保持原样)
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

    # 1. 基础合并
    price_df_1m_raw = pd.concat(dfs, axis=1).sort_index()
    price_df_1m_raw = price_df_1m_raw.loc[:, ~price_df_1m_raw.columns.duplicated()]

    # ==========================================
    # 🎯 核心修正：锁定全局共有区间 (Intersection)
    # ==========================================
    common_start = max([price_df_1m_raw[c].first_valid_index() for c in price_df_1m_raw.columns])
    common_end = min([price_df_1m_raw[c].last_valid_index() for c in price_df_1m_raw.columns])

    price_df_1m = price_df_1m_raw.loc[common_start:common_end].copy()

    print(f"✅ 成功锁定公共时间窗口: {common_start} 至 {common_end}")

    # ==========================================
    # 🛠️ 数据质量与填充统计 (共有区间内)
    # ==========================================
    total_1m_rows = len(price_df_1m)
    missing_counts = price_df_1m.isna().sum()

    print(f"\n🔍 【数据质量检测：共有区间内填充统计】 (总 1m K线数: {total_1m_rows})")
    for c in price_df_1m.columns:
        missing = missing_counts[c]
        fill_ratio = (missing / total_1m_rows) * 100
        alert_flag = " ⚠️ [流动性差/频繁断档]" if fill_ratio > 5.0 else ""
        print(f"   - {c:8s}: 真实缺失/需填充 {missing:>8d} 条 | 填充率 {fill_ratio:>6.2f}%{alert_flag}")
    print("-" * 50)

    # 2. 填充与降频操作
    price_df_1m = price_df_1m.ffill()
    price_df_4h = price_df_1m.resample('4h').last()

    # ==========================================
    # 🔴 涨跌幅与风险（最大回撤）统计
    # ==========================================
    if not price_df_4h.empty:
        print(f"\n📈 【共有区间内各标的表现 (Buy & Hold)】:")

        roll_max = price_df_4h.cummax()
        drawdowns = (price_df_4h - roll_max) / roll_max
        max_drawdowns_pct = drawdowns.min() * 100

        total_pct_change = 0.0
        num_coins = len(price_df_4h.columns)

        for c in price_df_4h.columns:
            start_price = price_df_4h[c].iloc[0]
            end_price = price_df_4h[c].iloc[-1]

            pct_change = (end_price - start_price) / start_price * 100
            total_pct_change += pct_change
            mdd = max_drawdowns_pct[c]

            print(f"   - {c:8s}: 涨跌幅 {pct_change:>8.2f}%  |  最大回撤 {mdd:>8.2f}%")

        avg_pct_change = total_pct_change / num_coins if num_coins > 0 else 0.0
        avg_mdd = max_drawdowns_pct.mean() if num_coins > 0 else 0.0

        print("-" * 50)
        print(f"   >>> 📊 基准表现 (等权 Buy & Hold):")
        print(f"            平均涨跌幅: {avg_pct_change:+.2f}%")
        print(f"            平均最大回撤: {avg_mdd:.2f}%")
        print("=" * 50)

    return price_df_4h


# ==========================================
# 核心：策略引擎与回测逻辑 (多空整合版)
# ==========================================
def run_backtest(df, param_name="默认基准参数", custom_params=None, enable_long=True, enable_short=True):
    mode_str = "多空双向" if (enable_long and enable_short) else ("仅做多" if enable_long else "仅做空")
    print(f"\n🚀 启动截面动量回测引擎... [{mode_str} | {param_name}]")

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

    # --- 预计算宏观开关 (Beta 层) ---
    btc_ma = df['BTC'].rolling(window=BTC_TREND_WINDOW).mean()
    btc_trend_on = df['BTC'] > btc_ma
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

        # 1. 截面动量打分 (包含多空判断)
        if enable_long and btc_trend_on.iloc[i]:
            current_mom = adj_mom.iloc[i].dropna()
            positive_mom = current_mom[current_mom > 0]

            if not positive_mom.empty:
                top_coins = positive_mom.nlargest(TOP_K).index.tolist()
                inv_vol = {}
                for c in top_coins:
                    c_vol = volatility[c].iloc[i]
                    inv_vol[c] = 1.0 / c_vol if c_vol > 0 else 0
                total_inv_vol = sum(inv_vol.values())

                for c in top_coins:
                    if total_inv_vol > 0:
                        raw_weight = inv_vol[c] / total_inv_vol
                        target_weights[c] = min(raw_weight, MAX_WEIGHT)

        elif enable_short and btc_trend_off.iloc[i]:
            current_mom = adj_mom.iloc[i].dropna()
            negative_mom = current_mom[current_mom < 0]

            if not negative_mom.empty:
                top_coins = negative_mom.nsmallest(TOP_K).index.tolist()
                inv_vol = {}
                for c in top_coins:
                    c_vol = volatility[c].iloc[i]
                    inv_vol[c] = 1.0 / c_vol if c_vol > 0 else 0
                total_inv_vol = sum(inv_vol.values())

                for c in top_coins:
                    if total_inv_vol > 0:
                        raw_weight = inv_vol[c] / total_inv_vol
                        target_weights[c] = -min(raw_weight, MAX_WEIGHT)

        # 2. 执行交易 (先处理所有抛售释放流动性：减多/平多/开空/加空)
        target_values = {c: current_equity * w for c, w in target_weights.items()}

        # [卖出：SELL]
        for c in coins:
            current_val = positions[c] * prices[c]
            diff_val = target_values.get(c, 0) - current_val

            if diff_val < -1.0:
                sell_amount = abs(diff_val) / prices[c]
                # 如果不允许做空，卖出数量不能超过已有仓位（最多平多至0）
                if not enable_short:
                    sell_amount = min(sell_amount, max(0.0, positions[c]))

                if sell_amount * prices[c] > 1.0:
                    actual_sell_val = sell_amount * prices[c]
                    fee = actual_sell_val * FEE_RATE

                    positions[c] -= sell_amount
                    cash += (actual_sell_val - fee)

                    trade_logs.append({
                        "time": current_time, "action": "SELL", "coin": c,
                        "price": prices[c], "amount": sell_amount, "value": actual_sell_val,
                        "fee": fee, "reason": "Target rebalance"
                    })

        # [买入：BUY (平空/减空/开多/加多)]
        for c in coins:
            current_val = positions[c] * prices[c]
            diff_val = target_values.get(c, 0) - current_val

            if diff_val > 1.0:
                available_to_spend = diff_val / (1 + FEE_RATE)
                if cash >= available_to_spend:
                    buy_val = available_to_spend
                else:
                    buy_val = cash / (1 + FEE_RATE)

                if buy_val > 1.0:
                    buy_amount = buy_val / prices[c]

                    # 如果不允许做多，买入数量不能超过空仓抵补量（最多平空至0）
                    if not enable_long:
                        buy_amount = min(buy_amount, max(0.0, -positions[c]))
                        buy_val = buy_amount * prices[c]

                    if buy_amount * prices[c] > 1.0:
                        fee = buy_val * FEE_RATE
                        positions[c] += buy_amount
                        cash -= (buy_val + fee)

                        trade_logs.append({
                            "time": current_time, "action": "BUY", "coin": c,
                            "price": prices[c], "amount": buy_amount, "value": buy_val,
                            "fee": fee, "reason": "Target rebalance"
                        })

    # ==========================================
    # 🔴 核心指标与高级统计计算 (深度兼容多空混合日志)
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

    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')

    win_trades, loss_trades = 0, 0
    total_profit, total_loss = 0.0, 0.0
    holding_times = []
    coin_states = {c: {'qty': 0.0, 'cost': 0.0, 'entry_time': None} for c in coins}

    for log in trade_logs:
        c, action, amt, price, fee, time = log['coin'], log['action'], log['amount'], log['price'], log['fee'], log[
            'time']

        current_qty = coin_states[c]['qty']
        current_cost = coin_states[c]['cost']

        if action == 'BUY':
            if current_qty >= 0:  # 动作：开多或加多
                new_qty = current_qty + amt
                coin_states[c]['cost'] = (current_qty * current_cost + amt * price) / new_qty
                coin_states[c]['qty'] = new_qty
                if coin_states[c]['entry_time'] is None:
                    coin_states[c]['entry_time'] = time
            else:  # 动作：平空或减空
                cover_amt = min(amt, abs(current_qty))
                proportion = cover_amt / amt if amt > 0 else 1
                pnl = cover_amt * (current_cost - price) - (fee * proportion)  # 利润 = (做空均价 - 平空价)

                if pnl > 0:
                    win_trades += 1;
                    total_profit += pnl
                else:
                    loss_trades += 1;
                    total_loss += abs(pnl)

                if coin_states[c]['entry_time'] is not None:
                    holding_times.append(time - coin_states[c]['entry_time'])

                # 判定是否发生“平空并反手开多”
                remaining_amt = amt - cover_amt
                if remaining_amt > 1e-6:
                    coin_states[c]['qty'] = remaining_amt
                    coin_states[c]['cost'] = price
                    coin_states[c]['entry_time'] = time
                else:
                    coin_states[c]['qty'] += amt
                    if abs(coin_states[c]['qty']) < 1e-6:
                        coin_states[c]['qty'], coin_states[c]['cost'], coin_states[c]['entry_time'] = 0.0, 0.0, None

        elif action == 'SELL':
            if current_qty <= 0:  # 动作：开空或加空
                old_qty_abs = abs(current_qty)
                new_qty_abs = old_qty_abs + amt
                coin_states[c]['cost'] = (old_qty_abs * current_cost + amt * price) / new_qty_abs
                coin_states[c]['qty'] -= amt
                if coin_states[c]['entry_time'] is None:
                    coin_states[c]['entry_time'] = time
            else:  # 动作：平多或减多
                sell_amt = min(amt, current_qty)
                proportion = sell_amt / amt if amt > 0 else 1
                pnl = sell_amt * (price - current_cost) - (fee * proportion)  # 利润 = (平多价 - 做多均价)

                if pnl > 0:
                    win_trades += 1;
                    total_profit += pnl
                else:
                    loss_trades += 1;
                    total_loss += abs(pnl)

                if coin_states[c]['entry_time'] is not None:
                    holding_times.append(time - coin_states[c]['entry_time'])

                # 判定是否发生“平多并反手开空”
                remaining_amt = amt - sell_amt
                if remaining_amt > 1e-6:
                    coin_states[c]['qty'] = -remaining_amt
                    coin_states[c]['cost'] = price
                    coin_states[c]['entry_time'] = time
                else:
                    coin_states[c]['qty'] -= amt
                    if abs(coin_states[c]['qty']) < 1e-6:
                        coin_states[c]['qty'], coin_states[c]['cost'], coin_states[c]['entry_time'] = 0.0, 0.0, None

    total_closed_trades = win_trades + loss_trades
    win_rate = win_trades / total_closed_trades if total_closed_trades > 0 else 0.0
    avg_profit = total_profit / win_trades if win_trades > 0 else 0.0
    avg_loss = total_loss / loss_trades if loss_trades > 0 else 0.0
    profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else float('inf')

    # 🎯 此处已修正：使用 Pandas 的 mean() 来规避内置 sum() 产生的 Cython 整型溢出问题
    avg_holding_time = pd.Series(holding_times).mean() if holding_times else timedelta(0)

    # 🔴 恢复并增强输出面板
    print("\n" + "=" * 45)
    print(f"📊 【测试结果面板】: {param_name} ({mode_str})")
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

    # 3 种模式的不同测试场景示例：
    test_scenarios = [
        {"name": "双向做多做空 (多空双开)", "enable_long": True, "enable_short": True,
         "params": {'MOM_WINDOW': 20 * 6, 'VOL_WINDOW': 20 * 6, 'BTC_TREND_WINDOW': 60 * 6, 'MAX_WEIGHT': 0.30}},
        {"name": "仅做多策略 (传统牛市多头)", "enable_long": True, "enable_short": False,
         "params": {'MOM_WINDOW': 20 * 6, 'VOL_WINDOW': 20 * 6, 'BTC_TREND_WINDOW': 60 * 6, 'MAX_WEIGHT': 0.30}},
        {"name": "仅做空策略 (单边熊市避险)", "enable_long": False, "enable_short": True,
         "params": {'MOM_WINDOW': 20 * 6, 'VOL_WINDOW': 20 * 6, 'BTC_TREND_WINDOW': 60 * 6, 'MAX_WEIGHT': 0.30}}
    ]

    for scenario in test_scenarios:
        logs_df, curve_df = run_backtest(
            df_4h,
            param_name=scenario["name"],
            custom_params=scenario["params"],
            enable_long=scenario["enable_long"],
            enable_short=scenario["enable_short"]
        )

    print("\n✅ 所有组合策略参数敏感性测试执行完毕。")