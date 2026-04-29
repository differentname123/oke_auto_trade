import pandas as pd
import numpy as np
import os
import glob
from datetime import timedelta


# ==========================================
# 1. 数据解析模块 (修改：增加最高/最低价读取以计算ATR)
# ==========================================
def load_and_preprocess_data(folder_path, macro_ticker="沪深300ETF"):
    print(f"⏳ 正在读取并合并目录 [{folder_path}] 下的个股与指数数据...")
    file_list = glob.glob(os.path.join(folder_path, "*.csv"))
    if not file_list:
        raise ValueError(f"在 {folder_path} 目录下没有找到 CSV 文件！")

    dfs_close, dfs_high, dfs_low = [], [], []

    for file in file_list:
        basename = os.path.basename(file)
        parts = basename.split('_')
        stock_name = parts[1] if len(parts) >= 2 else basename.replace('.csv', '')

        df = pd.read_csv(file)
        df['date'] = pd.to_datetime(df['日期'])
        df.set_index('date', inplace=True)

        # 提取收盘、最高、最低
        dfs_close.append(df[['收盘']].rename(columns={'收盘': stock_name}))
        dfs_high.append(df[['最高']].rename(columns={'最高': stock_name}))
        dfs_low.append(df[['最低']].rename(columns={'最低': stock_name}))

    # 合并并对齐
    price_df = pd.concat(dfs_close, axis=1).sort_index().ffill()
    high_df = pd.concat(dfs_high, axis=1).sort_index().ffill()
    low_df = pd.concat(dfs_low, axis=1).sort_index().ffill()

    # 统一剔除缺失值
    common_index = price_df.dropna().index
    price_df = price_df.loc[common_index]
    high_df = high_df.loc[common_index]
    low_df = low_df.loc[common_index]

    start_date = price_df.index[0].strftime('%Y-%m-%d')
    end_date = price_df.index[-1].strftime('%Y-%m-%d')

    print("-" * 40)
    print(f"✅ 数据对齐完成！共有 {len(price_df)} 个交易日。")
    print(f"📅 回测区间: {start_date} 至 {end_date}")

    # 这里保留原有的打印逻辑
    if macro_ticker in price_df.columns:
        idx_start = price_df[macro_ticker].iloc[0]
        idx_end = price_df[macro_ticker].iloc[-1]
        idx_return = (idx_end / idx_start - 1) * 100
        print(f"📈 基准指数 [{macro_ticker}] 期间涨跌幅: {idx_return:.2f}%")

    tradeable_stocks = [c for c in price_df.columns if c != macro_ticker]
    total_return_sum = 0.0
    for stock in tradeable_stocks:
        s_start = price_df[stock].iloc[0]
        s_end = price_df[stock].iloc[-1]
        s_ret = (s_end / s_start - 1) * 100
        total_return_sum += s_ret
    print(f"🔥 所有个股期间涨跌幅之和: {total_return_sum:.2f}%")
    print("-" * 40)

    return price_df, high_df, low_df


# ==========================================
# 2. 核心回测引擎 (修改：引入ATR与严苛买卖逻辑)
# ==========================================
def run_backtest(df_dict, macro_ticker="沪深300ETF", param_name="参数组", custom_params=None):
    df, df_high, df_low = df_dict
    print(f"\n🚀 正在运行: [{param_name}]")

    p = {
        'MOM_WINDOW': 20, 'VOL_WINDOW': 20, 'MACRO_WINDOW': 60, 'MAX_WEIGHT': 0.50
    }
    if custom_params: p.update(custom_params)

    TOP_K = 2
    BUY_FEE = 0.0003
    SELL_FEE = 0.0008
    INITIAL_CAPITAL = 200000.0

    tradeable_stocks = [c for c in df.columns if c != macro_ticker]

    # --- 调整点 1: 引入 True Range 计算 ATR ---
    prev_close = df[tradeable_stocks].shift(1)
    tr1 = df_high[tradeable_stocks] - df_low[tradeable_stocks]
    tr2 = (df_high[tradeable_stocks] - prev_close).abs()
    tr3 = (df_low[tradeable_stocks] - prev_close).abs()
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = tr.rolling(p['VOL_WINDOW']).mean()
    atr_pct = atr / df[tradeable_stocks]  # 归一化风险分母

    # 动量指标计算
    returns = df[tradeable_stocks].pct_change(p['MOM_WINDOW'])
    # 计算得分时仍可用波动率或ATR，此处维持逻辑一致性，但分母改为ATR
    adj_mom = returns / (atr_pct + 1e-8)

    macro_ma = df[macro_ticker].rolling(p['MACRO_WINDOW']).mean()
    macro_on = df[macro_ticker] > macro_ma

    # 账户初始化
    cash = INITIAL_CAPITAL
    positions = {s: 0.0 for s in tradeable_stocks}
    equity_curve = []
    coin_states = {s: {'qty': 0.0, 'cost': 0.0, 'entry_time': None} for s in tradeable_stocks}
    win_trades, loss_trades = 0, 0
    total_profit, total_loss = 0.0, 0.0
    holding_times = []
    entry_scores = []

    start_idx = max(p['MOM_WINDOW'], p['VOL_WINDOW'], p['MACRO_WINDOW'])

    for i in range(start_idx, len(df)):
        curr_time, prices, prev_prices = df.index[i], df.iloc[i], df.iloc[i - 1]
        current_mom_scores = adj_mom.iloc[i]

        curr_equity = cash + sum(positions[s] * prices[s] for s in tradeable_stocks)
        equity_curve.append({'time': curr_time, 'equity': curr_equity})

        # 确定当前的 Top K 列表
        top_coins = []
        if macro_on.iloc[i]:
            mom_scores = current_mom_scores.dropna()
            pos_mom = mom_scores[mom_scores > 0.0]
            if not pos_mom.empty:
                top_coins = pos_mom.nlargest(TOP_K).index.tolist()

        # --- 调整点 2: 卖出逻辑 (跌出排名全清) ---
        for s in tradeable_stocks:
            if positions[s] > 0 and s not in top_coins:
                if prices[s] <= prev_prices[s] * 0.905: continue  # 跌停锁死无法卖出

                sell_amount = positions[s]
                val = sell_amount * prices[s]
                fee = val * SELL_FEE

                pnl = sell_amount * (prices[s] - coin_states[s]['cost']) - fee
                if pnl > 0:
                    win_trades += 1
                    total_profit += pnl
                else:
                    loss_trades += 1
                    total_loss += abs(pnl)

                if coin_states[s]['entry_time']:
                    holding_times.append(curr_time - coin_states[s]['entry_time'])

                cash += (val - fee)
                positions[s] = 0.0
                coin_states[s] = {'qty': 0, 'cost': 0, 'entry_time': None}

        # --- 调整点 2: 买入逻辑 (绝对空仓才买入) ---
        # 计算潜在的买入权重（基于ATR倒数）
        if top_coins:
            # 仅针对Top K中目前没持仓的币种计算权重，或者预先算出所有Top K的理论权重
            inv_v = {s: 1.0 / atr_pct[s].iloc[i] for s in top_coins if atr_pct[s].iloc[i] > 0}
            total_v = sum(inv_v.values())

            for s in top_coins:
                # 只有当前毫无持仓，才会按权重买入
                if positions[s] == 0 and total_v > 0:
                    if prices[s] >= prev_prices[s] * 1.095: continue  # 涨停锁死无法买入

                    target_w = min(inv_v[s] / total_v, p['MAX_WEIGHT'])
                    buy_val = min(curr_equity * target_w, cash) / (1 + BUY_FEE)

                    if buy_val > 100.0:
                        fee = buy_val * BUY_FEE
                        amt = buy_val / prices[s]

                        coin_states[s]['cost'] = prices[s]
                        coin_states[s]['qty'] = amt
                        coin_states[s]['entry_time'] = curr_time
                        positions[s] = amt
                        cash -= (buy_val + fee)

                        score_at_buy = current_mom_scores[s]
                        if pd.notna(score_at_buy):
                            entry_scores.append(score_at_buy)

    # --- 后续指标计算 (保持原样) ---
    curve = pd.DataFrame(equity_curve).set_index('time')
    total_ret = (curr_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    max_dd = ((curve['equity'] - curve['equity'].cummax()) / curve['equity'].cummax()).min()
    ann_ret = (curr_equity / INITIAL_CAPITAL) ** (365 / (curve.index[-1] - curve.index[0]).days) - 1
    sr = (curve['equity'].pct_change().mean() / curve['equity'].pct_change().std()) * np.sqrt(252)

    idx_start = df[macro_ticker].iloc[start_idx]
    idx_end = df[macro_ticker].iloc[-1]
    idx_ret = (idx_end / idx_start) - 1
    alpha_ret = total_ret - idx_ret

    win_rate = win_trades / (win_trades + loss_trades) if (win_trades + loss_trades) > 0 else 0
    pl_ratio = (total_profit / win_trades) / (total_loss / loss_trades) if win_trades > 0 and loss_trades > 0 else 0
    avg_hold = sum(holding_times, timedelta(0)) / len(holding_times) if holding_times else timedelta(0)
    avg_entry_score = sum(entry_scores) / len(entry_scores) if entry_scores else 0.0

    print("-" * 45)
    print(f"初始本金: ¥{INITIAL_CAPITAL:,.2f} | 最终净值: ¥{curr_equity:,.2f}")
    print(f"总收益率: {total_ret * 100:6.2f}% | 年化: {ann_ret * 100:6.2f}%")
    print(f"基准收益: {idx_ret * 100:6.2f}% | 超额 (Alpha): {alpha_ret * 100:6.2f}%")
    print(f"最大回撤: {max_dd * 100:6.2f}% | 夏普比率: {sr:.2f}")
    print(
        f"胜率:     {win_rate * 100:6.2f}% | 盈亏比: {pl_ratio:.2f} | 平均持仓: {avg_hold.days}天 | 平均开仓得分: {avg_entry_score:.4f}")
    return curve


# ==========================================
# 3. 多参数循环测试
# ==========================================
if __name__ == "__main__":
    # 修改：load_and_preprocess_data 现在返回三个 DataFrame
    df_all, df_high, df_low = load_and_preprocess_data("k_line_all_history", macro_ticker="沪深300ETF")
    data_bundle = (df_all, df_high, df_low)

    scenarios = [
        {"name": "基准 (20d动量/60d均线)", "params": {'MOM_WINDOW': 20, 'MACRO_WINDOW': 60}},
        {"name": "短线 (10d动量/20d均线)", "params": {'MOM_WINDOW': 10, 'MACRO_WINDOW': 20}},
        {"name": "长线 (40d动量/120d均线)", "params": {'MOM_WINDOW': 40, 'MACRO_WINDOW': 120}},
        {"name": "低频 (60d动量/60d均线)", "params": {'MOM_WINDOW': 60, 'MACRO_WINDOW': 60}},
    ]

    for sc in scenarios:
        run_backtest(data_bundle, macro_ticker="沪深300ETF", param_name=sc['name'], custom_params=sc['params'])