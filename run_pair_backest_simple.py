import os
import pandas as pd
import numpy as np
from datetime import timedelta


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
        alert_flag = " ⚠️[流动性差/频繁断档]" if fill_ratio > 5.0 else ""
        print(f"   - {c:8s}: 真实缺失/需填充 {missing:>8d} 条 | 填充率 {fill_ratio:>6.2f}%{alert_flag}")
    print("-" * 50)

    price_df_1m = price_df_1m.ffill()
    price_df_4h = price_df_1m.resample('4h').last()

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

        # =========================================================
        # ⚠️ 优化点 1：把大盘分年明细提出来，在开始只全局打印一次，避免冗余
        # =========================================================
        print("-" * 50)
        print(f"   >>> 📅 【全局基准：各年度等权大盘表现】")
        for year, group in price_df_4h.groupby(price_df_4h.index.year):
            start_prices = group.iloc[0]
            end_prices = group.iloc[-1]
            pct_changes = (end_prices - start_prices) / start_prices * 100
            avg_beta = pct_changes.mean()
            coin_details = ", ".join([f"{c}: {pct:+.1f}%" for c, pct in pct_changes.items()])
            print(f"            ► {year}年: {avg_beta:>+7.2f}% | [{coin_details}]")
        print("=" * 50)

    return price_df_4h


# ==========================================
# 核心：策略引擎与回测逻辑 (二元进出测试版)
# ==========================================
def run_backtest(df, param_name="默认基准参数", custom_params=None):
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

    # =========================================================
    # ⚠️ 优化点 2：明确打出本次回测的具体参数组合参数值
    # =========================================================
    print(f"\n🚀 启动截面动量回测引擎 (验证：信号驱动二元进出)... [{param_name}]")
    print(
        f"   ⚙️ 参数配置: MOM_WIN={MOM_WINDOW}, VOL_WIN={VOL_WINDOW}, BTC_TREND={BTC_TREND_WINDOW}, MAX_WT={MAX_WEIGHT}")

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
    btc_trend_on = df['BTC'] > btc_ma

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

        top_coins = []

        # 1. 截面动量打分（计算当前信号）
        if btc_trend_on.iloc[i]:
            current_mom = adj_mom.iloc[i].dropna()
            positive_mom = current_mom[current_mom > 0]
            if not positive_mom.empty:
                top_coins = positive_mom.nlargest(TOP_K).index.tolist()

        # [卖出] 只要持仓标的触发了卖出信号（不再属于 top_coins，或宏观关闭） -> 直接全仓清空
        for c in coins:
            if positions[c] > 0:
                if c not in top_coins:
                    sell_amount = positions[c]
                    actual_sell_val = sell_amount * prices[c]
                    fee = actual_sell_val * FEE_RATE

                    positions[c] -= sell_amount  # 清零
                    cash += (actual_sell_val - fee)

                    trade_logs.append({
                        "time": current_time, "action": "SELL", "coin": c,
                        "price": prices[c], "amount": sell_amount, "value": actual_sell_val,
                        "fee": fee, "reason": "Signal Exit (Not in Top K / Trend Off)"
                    })

        # [买入] 如果目前有入场信号，并且当前空仓 -> 才分配资金买入。死拿不补仓、不减仓。
        if top_coins:
            # 仅为计算初始买入权重提供参考
            inv_vol = {}
            for c in top_coins:
                c_vol = volatility[c].iloc[i]
                inv_vol[c] = 1.0 / c_vol if c_vol > 0 else 0

            total_inv_vol = sum(inv_vol.values())

            for c in top_coins:
                # 🔴 关键核心：只有完全空仓时，才进行买入。一旦买入，无论涨跌不再微调
                if positions[c] == 0:
                    if total_inv_vol > 0:
                        raw_weight = inv_vol[c] / total_inv_vol
                        target_weight = min(raw_weight, MAX_WEIGHT)
                        target_val = current_equity * target_weight

                        available_to_spend = target_val / (1 + FEE_RATE)
                        if cash >= available_to_spend:
                            buy_val = available_to_spend
                        else:
                            buy_val = cash / (1 + FEE_RATE)

                        if buy_val > 1.0:
                            fee = buy_val * FEE_RATE
                            buy_amount = buy_val / prices[c]

                            positions[c] += buy_amount
                            cash -= (buy_val + fee)

                            trade_logs.append({
                                "time": current_time, "action": "BUY", "coin": c,
                                "price": prices[c], "amount": buy_amount, "value": buy_val,
                                "fee": fee, "reason": "Signal Entry (Top K)"
                            })

    # ==========================================
    # 🔴 核心指标与高级统计计算 (保持完全不变)
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

                # =========================================================
                # ⚠️ 优化点 3：顺手将 pnl （每笔平仓盈亏）动态记入 log 中，方便后续年度穿透分析
                # =========================================================
                log['pnl'] = pnl

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

    print("\n" + "=" * 45)
    print(f"📊 【测试结果面板】: {param_name}")
    print("-" * 45)
    print(f"💸[资金与收益]")
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
# 🔴 升级版：深度验证分析模块 (包含年度 Beta 对齐)
# ==========================================
def deep_robustness_check(logs_df, curve_df, price_df, param_name=""):
    print("\n" + "🔥" * 20)
    print(f"🕵️ 【深度鲁棒性检验报告】: {param_name}")
    print("🔥" * 20)

    if curve_df.empty or logs_df.empty:
        print("无交易数据，无法分析。")
        return

    # --- 1. 分年度/分季度绩效拆解 (Alpha vs Beta) ---
    print("\n📅 [年度绩效拆解] (策略表现 vs 同期市场基准):")
    curve_df['year'] = curve_df.index.year

    # 提取有 pnl 的平仓记录，为了计算当年专属的胜率和盈亏比
    if 'time' in logs_df.columns:
        logs_df['year'] = logs_df['time'].dt.year
    else:
        logs_df['year'] = pd.to_datetime(logs_df['time']).dt.year

    sell_logs = logs_df[(logs_df['action'] == 'SELL') & (logs_df['pnl'].notna())]

    for year, group in curve_df.groupby('year'):
        # 1. 计算策略当年表现
        start_eq = group['equity'].iloc[0]
        end_eq = group['equity'].iloc[-1]
        y_ret = (end_eq - start_eq) / start_eq * 100

        roll_max = group['equity'].cummax()
        y_mdd = ((group['equity'] - roll_max) / roll_max).min() * 100

        # 2. 计算同期市场 Beta 表现 (使用传入的 price_df)
        year_mask = price_df.index.year == year
        year_prices = price_df[year_mask]

        if not year_prices.empty:
            start_prices = year_prices.iloc[0]
            end_prices = year_prices.iloc[-1]
            avg_beta = ((end_prices - start_prices) / start_prices * 100).mean()
        else:
            avg_beta = 0.0

        # =========================================================
        # ⚠️ 优化点 4：替换冗长的硬编码大盘币种明细，改为输出每年的交易质量剖析
        # =========================================================
        y_sells = sell_logs[sell_logs['year'] == year]
        trades_cnt = len(y_sells)
        if trades_cnt > 0:
            y_win = (y_sells['pnl'] > 0).sum()
            y_win_rate = y_win / trades_cnt * 100

            sum_win = y_sells[y_sells['pnl'] > 0]['pnl'].sum()
            avg_win = sum_win / y_win if y_win > 0 else 0.0

            y_loss_cnt = trades_cnt - y_win
            sum_loss = abs(y_sells[y_sells['pnl'] <= 0]['pnl'].sum())
            avg_loss = sum_loss / y_loss_cnt if y_loss_cnt > 0 else 0.0

            y_pl_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

            trade_stats = f"{trades_cnt:>3d} 笔平仓 | 胜率: {y_win_rate:>5.1f}% | 盈亏比: {y_pl_ratio:>4.2f}"
        else:
            trade_stats = "  0 笔平仓"

        excess_ret = y_ret - avg_beta

        # 3. 打印对比与策略当年剖析面板
        print(
            f"   ► 【{year}年】 策略收益: {y_ret:>+7.2f}% (最大回撤 {y_mdd:>7.2f}%) | 等权大盘: {avg_beta:>+7.2f}% | 超额: {excess_ret:>+7.2f}%")
        print(f"            交易统计: {trade_stats}")
        print("-" * 50)

    # --- 2. 摩擦成本极限压力测试 (Transaction Cost Stress Test) ---
    print("\n🌪️[滑点与手续费压力测试] (检验低胜率策略的生存力):")
    stress_fees = [0.0005, 0.0010, 0.0020, 0.0030]  # 万5, 千1, 千2, 千3
    base_fee_rate = 0.0005

    for test_fee in stress_fees:
        buy_volume = logs_df[logs_df['action'] == 'BUY']['value'].sum()
        sell_volume = logs_df[logs_df['action'] == 'SELL']['value'].sum()
        total_trading_volume = buy_volume + sell_volume

        extra_fee_rate = max(0, test_fee - base_fee_rate)
        extra_friction_loss = total_trading_volume * extra_fee_rate

        original_final_equity = curve_df['equity'].iloc[-1]
        stressed_equity = original_final_equity - extra_friction_loss
        stressed_return = (stressed_equity - 10000) / 10000

        status = "✅ 存活" if stressed_return > 0 else "💀 破产"
        print(f"   - 单边综合成本 {test_fee * 10000:>2.0f} bps: 最终收益率 {stressed_return * 100:>8.2f}%[{status}]")

    print("=" * 60)


# ==========================================
# 主程序执行入口修改
# ==========================================
if __name__ == "__main__":
    file_list = ["kline_data/BTC_ETH_1m.csv", "kline_data/DOGE_SOL_1m.csv", "kline_data/TON_XRP_1m.csv"]
    df_4h = load_and_preprocess_data(file_list)

    # 恢复多组参数测试，确保参数空间的多样性
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

        # ⚠️ 注意这里传入了 df_4h 以便计算市场基准
        deep_robustness_check(logs_df, curve_df, df_4h, param_name=scenario["name"])

    print("\n✅ 所有参数组敏感性及深度检验执行完毕。")