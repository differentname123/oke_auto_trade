import pandas as pd
import numpy as np
import os
import glob
from datetime import timedelta


# ==========================================
# 1. 数据解析模块 (新增个股期间涨跌幅与总和)
# ==========================================
def load_and_preprocess_data(folder_path, macro_ticker="沪深300ETF"):
    print(f"⏳ 正在读取并合并目录 [{folder_path}] 下的个股与指数数据...")
    file_list = glob.glob(os.path.join(folder_path, "*.csv"))
    if not file_list:
        raise ValueError(f"在 {folder_path} 目录下没有找到 CSV 文件！")

    dfs = []
    for file in file_list:
        basename = os.path.basename(file)
        parts = basename.split('_')
        stock_name = parts[1] if len(parts) >= 2 else basename.replace('.csv', '')

        df = pd.read_csv(file)
        # 【修改点一】：适配 CSV 中的中文列名 '日期' 和 '收盘'
        df['date'] = pd.to_datetime(df['日期'])
        df.set_index('date', inplace=True)
        df_close = df[['收盘']].rename(columns={'收盘': stock_name})
        dfs.append(df_close)

    # 横向合并，向前填充 (处理停牌)
    price_df = pd.concat(dfs, axis=1).sort_index().ffill()
    price_df = price_df.dropna()  # 对齐所有股票共同的上市时间

    start_date = price_df.index[0].strftime('%Y-%m-%d')
    end_date = price_df.index[-1].strftime('%Y-%m-%d')

    print("-" * 40)
    print(f"✅ 数据对齐完成！共有 {len(price_df)} 个交易日。")
    print(f"📅 回测区间: {start_date} 至 {end_date}")

    if macro_ticker in price_df.columns:
        idx_start = price_df[macro_ticker].iloc[0]
        idx_end = price_df[macro_ticker].iloc[-1]
        idx_return = (idx_end / idx_start - 1) * 100
        print(f"📈 基准指数 [{macro_ticker}] 期间涨跌幅: {idx_return:.2f}%")

    print(f"🎯 包含标的: {list(price_df.columns)}")

    # ✨ 新增：计算并打印每只股票的期间涨跌幅及其总和
    print("\n📊 各交易标的期间涨跌幅:")
    tradeable_stocks = [c for c in price_df.columns if c != macro_ticker]
    total_return_sum = 0.0
    for stock in tradeable_stocks:
        s_start = price_df[stock].iloc[0]
        s_end = price_df[stock].iloc[-1]
        s_ret = (s_end / s_start - 1) * 100
        total_return_sum += s_ret
        print(f"   - {stock:^8}: {s_ret:>7.2f}%")

    print(f"🔥 所有个股期间涨跌幅之和: {total_return_sum:.2f}%")
    print("-" * 40)

    return price_df


# ==========================================
# 2. 核心回测引擎 (新增平均开仓得分统计)
# ==========================================
def run_backtest(df, macro_ticker="沪深300ETF", param_name="参数组", custom_params=None):
    print(f"\n🚀 正在运行: [{param_name}]")

    p = {
        'MOM_WINDOW': 20, 'VOL_WINDOW': 20, 'MACRO_WINDOW': 60, 'MAX_WEIGHT': 0.50
    }
    if custom_params: p.update(custom_params)

    TOP_K = 2
    BUY_FEE = 0.0003  # 买入佣金
    SELL_FEE = 0.0008  # 卖出印花税+佣金
    INITIAL_CAPITAL = 200000.0

    tradeable_stocks = [c for c in df.columns if c != macro_ticker]

    # 指标计算
    returns = df[tradeable_stocks].pct_change(p['MOM_WINDOW'])
    log_ret = np.log(df[tradeable_stocks] / df[tradeable_stocks].shift(1))
    volatility = log_ret.rolling(p['VOL_WINDOW']).std() * np.sqrt(252)
    adj_mom = returns / (volatility + 1e-8)
    macro_ma = df[macro_ticker].rolling(p['MACRO_WINDOW']).mean()
    macro_on = df[macro_ticker] > macro_ma

    # 账户初始化
    cash = INITIAL_CAPITAL
    positions = {s: 0.0 for s in tradeable_stocks}
    trade_logs = []
    equity_curve = []
    coin_states = {s: {'qty': 0.0, 'cost': 0.0, 'entry_time': None} for s in tradeable_stocks}
    win_trades, loss_trades = 0, 0
    total_profit, total_loss = 0.0, 0.0
    holding_times = []

    # ✨ 新增：用于记录每次触发买入（开仓/加仓）时的动量得分
    entry_scores = []

    start_idx = max(p['MOM_WINDOW'], p['VOL_WINDOW'], p['MACRO_WINDOW'])

    for i in range(start_idx, len(df)):
        curr_time, prices, prev_prices = df.index[i], df.iloc[i], df.iloc[i - 1]

        # 提取当天的所有动量得分，方便后续买入时记录
        current_mom_scores = adj_mom.iloc[i]

        curr_equity = cash + sum(positions[s] * prices[s] for s in tradeable_stocks)
        equity_curve.append({'time': curr_time, 'equity': curr_equity})

        target_weights = {s: 0.0 for s in tradeable_stocks}
        if macro_on.iloc[i]:
            mom_scores = current_mom_scores.dropna()
            pos_mom = mom_scores[mom_scores > 0.0]
            if not pos_mom.empty:
                top_s = pos_mom.nlargest(TOP_K).index.tolist()
                inv_v = {s: 1.0 / volatility[s].iloc[i] for s in top_s if volatility[s].iloc[i] > 0}
                total_v = sum(inv_v.values())
                for s in top_s:
                    if total_v > 0: target_weights[s] = min(inv_v[s] / total_v, p['MAX_WEIGHT'])

        # 卖出
        target_vals = {s: curr_equity * w for s, w in target_weights.items()}
        for s in tradeable_stocks:
            diff = target_vals.get(s, 0) - positions[s] * prices[s]
            if diff < -100.0:
                if prices[s] <= prev_prices[s] * 0.905: continue  # 跌停锁死
                sell_amt = min(abs(diff) / prices[s], positions[s])
                val = sell_amt * prices[s]
                fee = val * SELL_FEE

                pnl = sell_amt * (prices[s] - coin_states[s]['cost']) - fee
                if pnl > 0:
                    win_trades += 1;
                    total_profit += pnl
                else:
                    loss_trades += 1;
                    total_loss += abs(pnl)
                if coin_states[s]['entry_time']: holding_times.append(curr_time - coin_states[s]['entry_time'])

                positions[s] -= sell_amt
                cash += (val - fee)
                if positions[s] < 1e-5: coin_states[s] = {'qty': 0, 'cost': 0, 'entry_time': None}

        # 买入
        for s in tradeable_stocks:
            diff = target_vals.get(s, 0) - positions[s] * prices[s]
            if diff > 100.0:
                if prices[s] >= prev_prices[s] * 1.095: continue  # 涨停锁死
                buy_val = min(diff / (1 + BUY_FEE), cash / (1 + BUY_FEE))
                if buy_val > 100.0:
                    fee = buy_val * BUY_FEE
                    amt = buy_val / prices[s]

                    old_q = coin_states[s]['qty']
                    coin_states[s]['cost'] = (old_q * coin_states[s]['cost'] + amt * prices[s]) / (old_q + amt)
                    coin_states[s]['qty'] += amt
                    if not coin_states[s]['entry_time']: coin_states[s]['entry_time'] = curr_time
                    positions[s] += amt
                    cash -= (buy_val + fee)

                    # ✨ 新增：记录有效买入时的该股动量得分
                    score_at_buy = current_mom_scores[s]
                    if pd.notna(score_at_buy):
                        entry_scores.append(score_at_buy)

    # 计算整体指标
    curve = pd.DataFrame(equity_curve).set_index('time')
    total_ret = (curr_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    max_dd = ((curve['equity'] - curve['equity'].cummax()) / curve['equity'].cummax()).min()
    ann_ret = (curr_equity / INITIAL_CAPITAL) ** (365 / (curve.index[-1] - curve.index[0]).days) - 1
    sr = (curve['equity'].pct_change().mean() / curve['equity'].pct_change().std()) * np.sqrt(252)

    # 同期基准与超额收益计算
    idx_start = df[macro_ticker].iloc[start_idx]
    idx_end = df[macro_ticker].iloc[-1]
    idx_ret = (idx_end / idx_start) - 1
    alpha_ret = total_ret - idx_ret

    win_rate = win_trades / (win_trades + loss_trades) if (win_trades + loss_trades) > 0 else 0
    pl_ratio = (total_profit / win_trades) / (total_loss / loss_trades) if win_trades > 0 and loss_trades > 0 else 0
    avg_hold = sum(holding_times, timedelta(0)) / len(holding_times) if holding_times else timedelta(0)

    # ✨ 新增：计算平均开仓得分
    avg_entry_score = sum(entry_scores) / len(entry_scores) if entry_scores else 0.0

    print("-" * 45)
    print(f"初始本金: ¥{INITIAL_CAPITAL:,.2f} | 最终净值: ¥{curr_equity:,.2f}")
    print(f"总收益率: {total_ret * 100:6.2f}% | 年化: {ann_ret * 100:6.2f}%")
    print(f"基准收益: {idx_ret * 100:6.2f}% | 超额 (Alpha): {alpha_ret * 100:6.2f}%")
    print(f"最大回撤: {max_dd * 100:6.2f}% | 夏普比率: {sr:.2f}")
    # ✨ 新增：打印输出中加入了平均开仓得分
    print(
        f"胜率:     {win_rate * 100:6.2f}% | 盈亏比: {pl_ratio:.2f} | 平均持仓: {avg_hold.days}天 | 平均开仓得分: {avg_entry_score:.4f}")
    return curve


# ==========================================
# 3. 多参数循环测试
# ==========================================
if __name__ == "__main__":
    # 传入 macro_ticker 以便数据合并完能直接计算基准涨幅
    df_all = load_and_preprocess_data("k_line_all_history", macro_ticker="沪深300ETF")

    scenarios = [
        {"name": "基准 (20d动量/60d均线)", "params": {'MOM_WINDOW': 20, 'MACRO_WINDOW': 60}},
        {"name": "短线 (10d动量/20d均线)", "params": {'MOM_WINDOW': 10, 'MACRO_WINDOW': 20}},
        {"name": "长线 (40d动量/120d均线)", "params": {'MOM_WINDOW': 40, 'MACRO_WINDOW': 120}},
        {"name": "低频 (60d动量/60d均线)", "params": {'MOM_WINDOW': 60, 'MACRO_WINDOW': 60}},
    ]

    for sc in scenarios:
        # 【修改点二】：这里的 macro_ticker 必须和上面加载时保持完全一致
        run_backtest(df_all, macro_ticker="沪深300ETF", param_name=sc['name'], custom_params=sc['params'])