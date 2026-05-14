import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import pandas as pd
import numpy as np
from datetime import timedelta

# ==========================================
# 🌐 全局配置与交易模式控制
# ==========================================
GLOBAL_TRADE_MODE = 'SHORT_ONLY'
INITIAL_CAPITAL = 10000.0
FEE_RATE = 0.0005

# 🔴 [核心改动] 动态标的池配置：支持每年更换标的
YEARLY_POOL_CONFIG = {
  "2021": [
    "kline_data/BTCUSDT_1m_merged.csv",
    "kline_data/ETHUSDT_1m_merged.csv",
    "kline_data/XRPUSDT_1m_merged.csv",
    "kline_data/LTCUSDT_1m_merged.csv",
    "kline_data/BCHUSDT_1m_merged.csv",
    "kline_data/ADAUSDT_1m_merged.csv"
  ],
  "2022": [
    "kline_data/BTCUSDT_1m_merged.csv",
    "kline_data/ETHUSDT_1m_merged.csv",
    "kline_data/SOLUSDT_1m_merged.csv",
    "kline_data/BNBUSDT_1m_merged.csv",
    "kline_data/ADAUSDT_1m_merged.csv",
    "kline_data/XRPUSDT_1m_merged.csv"
  ],
  "2023": [
    "kline_data/BTCUSDT_1m_merged.csv",
    "kline_data/ETHUSDT_1m_merged.csv",
    "kline_data/BNBUSDT_1m_merged.csv",
    "kline_data/XRPUSDT_1m_merged.csv",
    "kline_data/DOGEUSDT_1m_merged.csv",
    "kline_data/ADAUSDT_1m_merged.csv"
  ],
  "2024": [
    "kline_data/BTCUSDT_1m_merged.csv",
    "kline_data/ETHUSDT_1m_merged.csv",
    "kline_data/SOLUSDT_1m_merged.csv",
    "kline_data/BNBUSDT_1m_merged.csv",
    "kline_data/XRPUSDT_1m_merged.csv",
    "kline_data/DOGEUSDT_1m_merged.csv"
  ],
  "2025": [
    "kline_data/BTCUSDT_1m_merged.csv",
    "kline_data/ETHUSDT_1m_merged.csv",
    "kline_data/SOLUSDT_1m_merged.csv",
    "kline_data/XRPUSDT_1m_merged.csv",
    "kline_data/BNBUSDT_1m_merged.csv",
    "kline_data/DOGEUSDT_1m_merged.csv"
  ],
  "2026": [
    "kline_data/BTCUSDT_1m_merged.csv",
    "kline_data/ETHUSDT_1m_merged.csv",
    "kline_data/SOLUSDT_1m_merged.csv",
    "kline_data/XRPUSDT_1m_merged.csv",
    "kline_data/BNBUSDT_1m_merged.csv",
    "kline_data/DOGEUSDT_1m_merged.csv"
  ]
}


# ==========================================
# 🔴 核心改动 1: 环境预热与数据组装层 (支持时间相位错位)
# ==========================================
def prepare_environment(pool_config, time_offset='0h'):
    print("\n" + "═" * 78)
    print(f"⏳ 正在构建全周期动态标的池环境 (预加载与预热推导 | 相位偏移: {time_offset})...")
    print("═" * 78)

    yearly_cache = {}
    min_warmup_bars = float('inf')
    trading_dfs = []

    for year_str, file_list in pool_config.items():
        year = int(year_str)
        print(f"   ► 解析 {year} 年标的池数据...")
        dfs = []

        for file in file_list:
            if not os.path.exists(file):
                print(f"     ⚠️ 文件未找到，请确保路径正确: {file}")
                continue

            coin_name = os.path.basename(file).split('_')[0].replace('USDT', '')
            df = pd.read_csv(file, usecols=['open_time', 'close'], engine='pyarrow')
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)

            # 🔴 [核心修改]: 在重采样中注入时间错位参数 offset
            df_4h_coin = df['close'].resample('4h', offset=time_offset).agg(
                open='first', high='max', low='min', close='last'
            )

            # 🔴 [安全补丁]: 因为时间错位，首尾可能会切出全为 NaN 的碎片 K 线，必须清理
            df_4h_coin.dropna(how='all', inplace=True)

            df_4h_coin.rename(columns={
                'open': f"{coin_name}_open",
                'high': f"{coin_name}_high",
                'low': f"{coin_name}_low",
                'close': coin_name
            }, inplace=True)
            dfs.append(df_4h_coin)

        df_raw = pd.concat(dfs, axis=1).sort_index()
        main_coins = [c for c in df_raw.columns if not any(x in c for x in ['_open', '_high', '_low'])]

        # 🎯 锁定当前年份数据的公共区间
        coin_starts = [df_raw[c].first_valid_index() for c in main_coins]
        coin_ends = [df_raw[c].last_valid_index() for c in main_coins]
        common_start = max(coin_starts)
        common_end = min(coin_ends)

        # 截断截止时间：不能超过该年的 12-31 23:59:59
        year_end_limit = pd.to_datetime(f"{year}-12-31 23:59:59")
        common_end = min(common_end, year_end_limit)

        df_year = df_raw.loc[common_start:common_end].ffill()

        # 计算预热 Bar 数（公共起点 -> 该年 1月1日 的 K线数）
        year_start_time = pd.to_datetime(f"{year}-01-01 00:00:00")
        warmup_data = df_year.loc[:year_start_time]
        warmup_bars = len(warmup_data) - 1 if len(warmup_data) > 0 else 0

        print(f"      - 公共最大起始: {common_start} | 跨年截止: {common_end}")
        print(f"      - 该年可用预热 Bar 数: {warmup_bars}")

        if warmup_bars < min_warmup_bars:
            min_warmup_bars = warmup_bars

        yearly_cache[year] = df_year

        # 只取交易期的数据组合成全局 price_df，用于后续评估 MFE/MAE
        trade_period = df_year.loc[year_start_time:common_end]
        if not trade_period.empty:
            trading_dfs.append(trade_period)

    print(f"\n✅ 成功锁定全局最小可用预热 Bar 数为: {min_warmup_bars}")

    print(f"🚀 正在合并全局指标分析专用 Price DataFrame...")
    # 使用 outer join 合并全局价格，因为不同年份标的会变
    global_price_df = pd.concat(trading_dfs, axis=0)
    global_price_df = global_price_df[~global_price_df.index.duplicated(keep='first')]

    return yearly_cache, global_price_df, min_warmup_bars

# ==========================================
# 评价指标辅助函数 (保留原逻辑)
# ==========================================
def _compute_gini(values):
    if values is None or len(values) == 0: return np.nan
    arr = np.abs(np.asarray(values, dtype=float))
    if arr.sum() == 0: return 0.0
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * sorted_arr)) / (n * np.sum(sorted_arr)) - (n + 1.0) / n)


def _compute_underwater_periods(curve_df):
    if curve_df.empty: return 0.0, 0.0, []
    cum_max = curve_df['equity'].cummax()
    drawdown = (curve_df['equity'] - cum_max) / cum_max
    underwater = drawdown < -1e-9
    if not underwater.any(): return 0.0, 0.0, []

    periods = []
    in_uw, start_uw = False, None
    for t, uw in underwater.items():
        if uw and not in_uw:
            start_uw, in_uw = t, True
        elif not uw and in_uw:
            periods.append((t - start_uw).total_seconds() / 86400.0)
            in_uw = False
    if in_uw:
        periods.append((underwater.index[-1] - start_uw).total_seconds() / 86400.0)
    if not periods: return 0.0, 0.0, []
    return float(max(periods)), float(np.mean(periods)), periods


def _compute_time_in_market(logs_df, time_index):
    if logs_df is None or logs_df.empty or len(time_index) == 0: return 0.0
    logs_sorted = logs_df.sort_values('time').reset_index(drop=True)
    events = [(row['time'], row['coin'], row['action'], row['amount']) for _, row in logs_sorted.iterrows()]
    holdings = {}
    in_market = np.zeros(len(time_index), dtype=bool)
    event_idx = 0
    for i, t in enumerate(time_index):
        while event_idx < len(events) and events[event_idx][0] <= t:
            _, coin, action, amt = events[event_idx]
            if action == 'BUY':
                holdings[coin] = holdings.get(coin, 0) + amt
            else:
                holdings[coin] = holdings.get(coin, 0) - amt
            if abs(holdings.get(coin, 0)) < 1e-9: holdings[coin] = 0
            event_idx += 1
        in_market[i] = any(abs(qty) > 1e-9 for qty in holdings.values())
    return float(in_market.sum() / len(in_market))


def _compute_mae_mfe(logs_df, price_df):
    out = {'mae': [], 'mfe': [], 'holding_h': []}
    if logs_df is None or logs_df.empty: return out
    logs_sorted = logs_df.sort_values('time').reset_index(drop=True)
    states = {}
    for _, log in logs_sorted.iterrows():
        c, t, p, amt, event, direction = log['coin'], log['time'], log['price'], log['amount'], log.get('event',
                                                                                                        None), log.get(
            'direction', 'LONG')
        if event == 'OPEN':
            states[c] = {'qty': amt, 'entry_p': p, 'entry': t, 'side': direction}
        elif event == 'CLOSE':
            if c in states and states[c].get('qty', 0) > 0:
                entry_t = states[c]['entry']
                entry_p = states[c]['entry_p']
                side = states[c]['side']
                out['holding_h'].append((t - entry_t).total_seconds() / 3600.0)
                if f"{c}_high" in price_df.columns and f"{c}_low" in price_df.columns:
                    period = price_df.loc[entry_t:t]
                    if not period.empty and entry_p > 0:
                        ph = float(period[f"{c}_high"].max())
                        pl = float(period[f"{c}_low"].min())
                        if side == 'LONG':
                            out['mfe'].append((ph - entry_p) / entry_p)
                            out['mae'].append((pl - entry_p) / entry_p)
                        else:
                            out['mfe'].append((entry_p - pl) / entry_p)
                            out['mae'].append((entry_p - ph) / entry_p)
                states[c] = {'qty': 0, 'entry_p': 0, 'entry': None, 'side': None}
    return out


# ==========================================
# 完整指标计算 (保留原逻辑, 但传入全局 df)
# ==========================================
def calculate_comprehensive_metrics(logs_df, curve_df, price_df, custom_params, param_name,
                                    initial_capital=10000.0, fee_rate=0.0005,
                                    final_equity_override=None):
    metrics = {'param_name': param_name}
    metrics.update({f'param_{k}': v for k, v in custom_params.items()})
    bars_per_year = 365 * 6

    if curve_df is None or curve_df.empty:
        return metrics

    # =============== A. 资金曲线层面 ===============
    final_equity = float(final_equity_override) if final_equity_override is not None else float(
        curve_df['equity'].iloc[-1])
    metrics['initial_capital'] = float(initial_capital)
    metrics['final_equity'] = final_equity
    metrics['total_return'] = (final_equity - initial_capital) / initial_capital

    days_passed = max(1, (curve_df.index[-1] - curve_df.index[0]).days)
    metrics['days_passed'] = days_passed
    metrics['annual_return'] = (
                (final_equity / initial_capital) ** (365 / days_passed) - 1) if final_equity > 0 else -1.0

    cum_max = curve_df['equity'].cummax()
    drawdown = (curve_df['equity'] - cum_max) / cum_max
    metrics['max_drawdown'] = float(drawdown.min())
    metrics['avg_drawdown'] = float(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0

    dd_pct = drawdown * 100
    metrics['ulcer_index'] = float(np.sqrt((dd_pct ** 2).mean()))
    metrics['pain_index'] = float(dd_pct.abs().mean())

    max_uw, avg_uw, _ = _compute_underwater_periods(curve_df)
    metrics['max_time_under_water_days'] = max_uw
    metrics['avg_recovery_time_days'] = avg_uw

    log_eq = np.log(curve_df['equity'].clip(lower=1e-9).values)
    x = np.arange(len(log_eq))
    if len(x) > 2 and np.std(log_eq) > 0:
        slope, intercept = np.polyfit(x, log_eq, 1)
        y_pred = slope * x + intercept
        ss_res = float(np.sum((log_eq - y_pred) ** 2))
        ss_tot = float(np.sum((log_eq - log_eq.mean()) ** 2))
        metrics['log_equity_r2'] = (1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        metrics['log_equity_r2'] = 0.0

    metrics['time_in_market_pct'] = _compute_time_in_market(logs_df, curve_df.index)

    # =============== E. 风险调整收益族 ===============
    rets_4h = curve_df['equity'].pct_change().dropna()
    if len(rets_4h) > 1:
        mr, sr_ = float(rets_4h.mean()), float(rets_4h.std())
        metrics['sharpe_ratio'] = (mr / sr_ * np.sqrt(bars_per_year)) if sr_ > 0 else 0.0
        downside = rets_4h[rets_4h < 0]
        d_std = float(downside.std()) if len(downside) > 1 else 0
        metrics['sortino_ratio'] = (mr / d_std * np.sqrt(bars_per_year)) if d_std and d_std > 0 else 0.0
        metrics['calmar_ratio'] = (metrics['annual_return'] / abs(metrics['max_drawdown'])) if metrics[
                                                                                                   'max_drawdown'] < 0 else float(
            'inf')
        metrics['mar_ratio'] = metrics['calmar_ratio']
        pos_r = float(rets_4h[rets_4h > 0].sum())
        neg_r = float(abs(rets_4h[rets_4h < 0].sum()))
        metrics['omega_ratio'] = pos_r / neg_r if neg_r > 0 else float('inf')
        if len(rets_4h) > 20:
            p95 = float(rets_4h.quantile(0.95))
            p5 = float(rets_4h.quantile(0.05))
            metrics['tail_ratio'] = p95 / abs(p5) if abs(p5) > 0 else float('inf')
        else:
            metrics['tail_ratio'] = np.nan
    else:
        for k in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'mar_ratio', 'omega_ratio', 'tail_ratio']:
            metrics[k] = 0.0

    monthly_eq = curve_df['equity'].resample('M').last()
    monthly_rets = monthly_eq.pct_change().dropna()

    if len(monthly_rets) >= 12:
        rolling_12m = monthly_eq.pct_change(12).dropna()
        metrics['rolling_12m_return_mean'] = float(rolling_12m.mean())
        metrics['rolling_12m_return_std'] = float(rolling_12m.std())
        metrics['rolling_12m_return_min'] = float(rolling_12m.min())
        roll_sharpe = monthly_rets.rolling(12).apply(
            lambda x: (x.mean() / x.std() * np.sqrt(12)) if x.std() > 0 else 0).dropna()
        metrics['rolling_12m_sharpe_mean'] = float(roll_sharpe.mean())
        metrics['rolling_12m_sharpe_std'] = float(roll_sharpe.std())
        metrics['rolling_12m_sharpe_cv'] = float(abs(roll_sharpe.std() / roll_sharpe.mean())) if abs(
            roll_sharpe.mean()) > 1e-9 else np.nan

        def _r_sortino(x):
            if not (x < 0).any(): return 0
            d = x[x < 0].std()
            return (x.mean() / d * np.sqrt(12)) if d > 0 else 0

        roll_sortino = monthly_rets.rolling(12).apply(_r_sortino).dropna()
        metrics['rolling_12m_sortino_mean'] = float(roll_sortino.mean())
        metrics['rolling_12m_sortino_std'] = float(roll_sortino.std())
        metrics['rolling_12m_sortino_cv'] = float(abs(roll_sortino.std() / roll_sortino.mean())) if abs(
            roll_sortino.mean()) > 1e-9 else np.nan
    else:
        for k in ['rolling_12m_return_mean', 'rolling_12m_return_std', 'rolling_12m_return_min',
                  'rolling_12m_sharpe_mean', 'rolling_12m_sharpe_std', 'rolling_12m_sharpe_cv',
                  'rolling_12m_sortino_mean', 'rolling_12m_sortino_std', 'rolling_12m_sortino_cv']:
            metrics[k] = np.nan

    # =============== B. 交易层面 ===============
    sell_logs = pd.DataFrame()
    if logs_df is not None and not logs_df.empty and 'pnl' in logs_df.columns:
        sell_logs = logs_df[logs_df['pnl'].notna()].copy()

    metrics['total_actions'] = int(len(logs_df)) if logs_df is not None else 0
    metrics['total_closed_trades'] = int(len(sell_logs))

    if len(sell_logs) > 0:
        pnls = sell_logs['pnl'].values
        metrics['pnl_mean'] = float(np.mean(pnls))
        metrics['pnl_std'] = float(np.std(pnls))
        metrics['pnl_skew'] = float(pd.Series(pnls).skew()) if len(pnls) > 2 else 0.0
        metrics['pnl_kurtosis'] = float(pd.Series(pnls).kurtosis()) if len(pnls) > 3 else 0.0
        metrics['pnl_gini'] = _compute_gini(pnls)

        sorted_desc = np.sort(pnls)[::-1]
        total_pnl = float(sorted_desc.sum())

        if total_pnl > 0:
            metrics['top1_pnl_ratio'] = float(sorted_desc[0]) / total_pnl
            metrics['top3_pnl_ratio'] = float(sorted_desc[:min(3, len(sorted_desc))].sum()) / total_pnl
            metrics['top5_pnl_ratio'] = float(sorted_desc[:min(5, len(sorted_desc))].sum()) / total_pnl
        else:
            metrics['top1_pnl_ratio'] = metrics['top3_pnl_ratio'] = metrics['top5_pnl_ratio'] = np.nan

        wins = sell_logs[sell_logs['pnl'] > 0]
        losses = sell_logs[sell_logs['pnl'] <= 0]
        metrics['win_rate'] = float(len(wins) / len(sell_logs))
        metrics['avg_win'] = float(wins['pnl'].mean()) if len(wins) > 0 else 0.0
        metrics['avg_loss'] = float(abs(losses['pnl'].mean())) if len(losses) > 0 else 0.0
        metrics['profit_loss_ratio'] = (metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] > 0 else float(
            'inf')

        metrics['expectancy'] = metrics['pnl_mean']
        if len(pnls) > 1:
            se = float(np.std(pnls)) / np.sqrt(len(pnls))
            metrics['expectancy_ci_low'] = metrics['expectancy'] - 1.96 * se
            metrics['expectancy_ci_high'] = metrics['expectancy'] + 1.96 * se
        else:
            metrics['expectancy_ci_low'] = metrics['expectancy_ci_high'] = metrics['expectancy']

        for k in [1, 3, 5]:
            if len(sorted_desc) > k:
                after_drop = float(sorted_desc[k:].sum())
                metrics[f'drop_top{k}_pnl_decay'] = (total_pnl - after_drop) / total_pnl if total_pnl > 0 else 0.0
            else:
                metrics[f'drop_top{k}_pnl_decay'] = 1.0

        if len(pnls) > 5:
            n_boot = 1000
            rng = np.random.default_rng(42)
            boots = np.array([rng.choice(pnls, size=len(pnls), replace=True).mean() for _ in range(n_boot)])
            metrics['bootstrap_pnl_mean_mean'] = float(boots.mean())
            metrics['bootstrap_pnl_mean_std'] = float(boots.std())
            metrics['bootstrap_pnl_mean_p5'] = float(np.percentile(boots, 5))
            metrics['bootstrap_pnl_mean_p95'] = float(np.percentile(boots, 95))
            metrics['bootstrap_positive_prob'] = float((boots > 0).mean())
        else:
            for k in ['bootstrap_pnl_mean_mean', 'bootstrap_pnl_mean_std', 'bootstrap_pnl_mean_p5',
                      'bootstrap_pnl_mean_p95', 'bootstrap_positive_prob']:
                metrics[k] = np.nan
    else:
        for k in ['pnl_mean', 'pnl_std', 'pnl_skew', 'pnl_kurtosis', 'pnl_gini',
                  'top1_pnl_ratio', 'top3_pnl_ratio', 'top5_pnl_ratio', 'win_rate', 'avg_win', 'avg_loss',
                  'profit_loss_ratio',
                  'expectancy', 'expectancy_ci_low', 'expectancy_ci_high', 'drop_top1_pnl_decay', 'drop_top3_pnl_decay',
                  'drop_top5_pnl_decay',
                  'bootstrap_pnl_mean_mean', 'bootstrap_pnl_mean_std', 'bootstrap_pnl_mean_p5',
                  'bootstrap_pnl_mean_p95', 'bootstrap_positive_prob']:
            metrics[k] = np.nan

    mm = _compute_mae_mfe(logs_df, price_df)
    if mm['mae']:
        metrics['mae_pct_mean'], metrics['mae_pct_median'], metrics['mae_pct_worst'] = float(np.mean(mm['mae'])), float(
            np.median(mm['mae'])), float(np.min(mm['mae']))
        metrics['mfe_pct_mean'], metrics['mfe_pct_median'], metrics['mfe_pct_best'] = float(np.mean(mm['mfe'])), float(
            np.median(mm['mfe'])), float(np.max(mm['mfe']))
    else:
        for k in ['mae_pct_mean', 'mae_pct_median', 'mae_pct_worst', 'mfe_pct_mean', 'mfe_pct_median', 'mfe_pct_best']:
            metrics[k] = np.nan

    metrics['avg_holding_hours'] = float(np.mean(mm['holding_h'])) if mm['holding_h'] else 0.0
    metrics['median_holding_hours'] = float(np.median(mm['holding_h'])) if mm['holding_h'] else 0.0

    # =============== C. 标的层面 ===============
    coins = [c for c in price_df.columns if not any(s in c for s in ['_open', '_high', '_low'])]
    asset_records = {}
    total_pnl_all = float(sell_logs['pnl'].sum()) if len(sell_logs) > 0 else 0.0

    for c in coins:
        c_sells = sell_logs[sell_logs['coin'] == c] if len(sell_logs) > 0 else pd.DataFrame()
        if len(c_sells) > 0:
            cp = c_sells['pnl'].values
            cw, cl = c_sells[c_sells['pnl'] > 0], c_sells[c_sells['pnl'] <= 0]
            avg_w = float(cw['pnl'].mean()) if len(cw) > 0 else 0.0
            avg_l = float(abs(cl['pnl'].mean())) if len(cl) > 0 else 0.0
            expect = float(np.mean(cp))
            ci_low = expect - 1.96 * float(np.std(cp)) / np.sqrt(len(cp)) if len(cp) > 1 else expect
            net = float(cp.sum())
            share = net / total_pnl_all if total_pnl_all > 0 else 0.0
            asset_records[c] = {
                'trades': int(len(c_sells)), 'win_rate': float(len(cw) / len(c_sells)),
                'avg_win': avg_w, 'avg_loss': avg_l, 'profit_loss_ratio': avg_w / avg_l if avg_l > 0 else float('inf'),
                'expectancy': expect, 'expectancy_ci_low': ci_low, 'net_pnl': net, 'pnl_share': share
            }
        else:
            asset_records[c] = {k: 0.0 for k in
                                ['trades', 'win_rate', 'avg_win', 'avg_loss', 'profit_loss_ratio', 'expectancy',
                                 'expectancy_ci_low', 'net_pnl', 'pnl_share']}

    for c, m in asset_records.items():
        for k, v in m.items(): metrics[f'asset_{c}_{k}'] = v

    pos_pnls = [m['net_pnl'] for m in asset_records.values() if m['net_pnl'] > 0]
    if pos_pnls:
        tp = sum(pos_pnls)
        shares = [p / tp for p in pos_pnls]
        metrics['asset_hhi'] = float(sum(s ** 2 for s in shares))
        metrics['asset_top1_share'] = float(max(shares))
    else:
        metrics['asset_hhi'] = metrics['asset_top1_share'] = np.nan

    metrics['active_assets'] = int(sum(1 for m in asset_records.values() if m['trades'] > 0))
    metrics['negative_expectancy_assets'] = int(
        sum(1 for m in asset_records.values() if m['expectancy'] < 0 and m['trades'] > 0))

    # =============== D. 时间维度 ===============
    if len(monthly_rets) > 0:
        metrics['monthly_positive_ratio'] = float((monthly_rets > 0).sum() / len(monthly_rets))
        ls, mls = 0, 0
        for r in monthly_rets:
            if r < 0:
                ls += 1; mls = max(mls, ls)
            else:
                ls = 0
        metrics['max_consecutive_losing_months'] = int(mls)
        metrics['monthly_return_mean'] = float(monthly_rets.mean())
        metrics['monthly_return_std'] = float(monthly_rets.std())
        m_pnl_dollar = monthly_eq.diff().dropna()
        pos_m_pnl = m_pnl_dollar[m_pnl_dollar > 0]
        if pos_m_pnl.sum() > 0:
            metrics['top1_month_pnl_ratio'] = float(pos_m_pnl.max() / pos_m_pnl.sum())
            metrics['top3_months_pnl_ratio'] = float(pos_m_pnl.nlargest(min(3, len(pos_m_pnl))).sum() / pos_m_pnl.sum())
        else:
            metrics['top1_month_pnl_ratio'] = metrics['top3_months_pnl_ratio'] = np.nan
    else:
        for k in ['monthly_positive_ratio', 'max_consecutive_losing_months', 'monthly_return_mean',
                  'monthly_return_std']: metrics[k] = 0.0
        metrics['top1_month_pnl_ratio'] = metrics['top3_months_pnl_ratio'] = np.nan

    annual_records = []
    for year in sorted(set(curve_df.index.year)):
        yd = curve_df[curve_df.index.year == year]
        if len(yd) > 1:
            se_, ee_ = float(yd['equity'].iloc[0]), float(yd['equity'].iloc[-1])
            yr = (ee_ - se_) / se_ if se_ > 0 else 0.0
            ycm = yd['equity'].cummax()
            ydd = float(((yd['equity'] - ycm) / ycm).min())
            yrr = yd['equity'].pct_change().dropna()
            ys = (yrr.mean() / yrr.std() * np.sqrt(bars_per_year)) if len(yrr) > 0 and yrr.std() > 0 else 0.0
            yds = yrr[yrr < 0]
            yo = (yrr.mean() / yds.std() * np.sqrt(bars_per_year)) if len(yds) > 1 and yds.std() > 0 else 0.0
            yc = (yr / abs(ydd)) if ydd < 0 else (float('inf') if yr > 0 else 0.0)
            annual_records.append(
                {'year': int(year), 'return': yr, 'max_dd': ydd, 'sharpe': ys, 'sortino': yo, 'calmar': yc})

    if annual_records:
        adf = pd.DataFrame(annual_records)
        metrics['annual_return_mean'] = float(adf['return'].mean())
        metrics['annual_return_std'] = float(adf['return'].std()) if len(adf) > 1 else 0.0
        metrics['annual_dd_mean'] = float(adf['max_dd'].mean())
        metrics['annual_dd_worst'] = float(adf['max_dd'].min())
        finite_sortino = adf['sortino'][np.isfinite(adf['sortino'])]
        metrics['annual_sortino_mean'] = float(finite_sortino.mean()) if len(finite_sortino) > 0 else 0.0
        finite_calmar = adf['calmar'][np.isfinite(adf['calmar'])]
        metrics['annual_calmar_mean'] = float(finite_calmar.mean()) if len(finite_calmar) > 0 else 0.0
        metrics['profitable_years'] = int((adf['return'] > 0).sum())
        metrics['total_years'] = int(len(adf))
        metrics['profitable_years_ratio'] = metrics['profitable_years'] / metrics['total_years']
        for _, row in adf.iterrows():
            y = int(row['year'])
            metrics[f'year_{y}_return'] = float(row['return'])
            metrics[f'year_{y}_max_dd'] = float(row['max_dd'])
            metrics[f'year_{y}_sortino'] = float(row['sortino']) if np.isfinite(row['sortino']) else np.nan
    else:
        for k in ['annual_return_mean', 'annual_return_std', 'annual_dd_mean', 'annual_dd_worst', 'annual_sortino_mean',
                  'annual_calmar_mean', 'profitable_years_ratio']: metrics[k] = 0.0
        metrics['profitable_years'] = metrics['total_years'] = 0

    if 'BTC_TREND_WINDOW' in custom_params and 'btc_trend_on' in curve_df.columns:
        btc_on = curve_df['btc_trend_on'].fillna(False)
        btc_on_lag = btc_on.shift(1).fillna(False)

        bull_rets = rets_4h[btc_on_lag]
        bear_rets = rets_4h[~btc_on_lag]

        metrics['bull_regime_total_return'] = float((1 + bull_rets).prod() - 1) if len(bull_rets) > 0 else 0.0
        metrics['bull_regime_bars'] = int(len(bull_rets))
        metrics['bear_regime_total_return'] = float((1 + bear_rets).prod() - 1) if len(bear_rets) > 0 else 0.0
        metrics['bear_regime_bars'] = int(len(bear_rets))

    # =============== F. 鲁棒性 ===============
    if logs_df is not None and not logs_df.empty:
        total_vol = float(logs_df['value'].sum())
        ofe = final_equity
        for tf in [0.0010, 0.0020, 0.0030]:
            ef = max(0, tf - fee_rate)
            seq = ofe - total_vol * ef
            metrics[f'cost_stress_{int(tf * 10000)}bps_return'] = float((seq - initial_capital) / initial_capital)
            metrics[f'cost_stress_{int(tf * 10000)}bps_annual'] = float(
                ((seq / initial_capital) ** (365 / days_passed) - 1)) if days_passed > 0 and seq > 0 else -1.0
        metrics['total_trading_volume'] = total_vol
        metrics['total_fee_paid'] = float(logs_df['fee'].sum())
        metrics['fee_to_pnl_ratio'] = float(metrics['total_fee_paid'] / abs(total_pnl_all)) if abs(
            total_pnl_all) > 0 else np.nan
    else:
        for tf in [10, 20, 30]: metrics[f'cost_stress_{tf}bps_return'] = metrics[f'cost_stress_{tf}bps_annual'] = np.nan
        metrics['total_trading_volume'] = metrics['total_fee_paid'] = metrics['fee_to_pnl_ratio'] = np.nan

    return metrics


# ==========================================
# 🔴 核心改动 2: 单年状态流转推演模块 (仅增加 target_weight 记录)
# ==========================================
def run_backtest_single_year(df, year, prev_state, custom_params, next_year_coins=None):
    MOM_WINDOW = custom_params['MOM_WINDOW']
    VOL_WINDOW = custom_params['VOL_WINDOW']
    BTC_TREND_WINDOW = custom_params['BTC_TREND_WINDOW']
    MAX_WEIGHT = custom_params['MAX_WEIGHT']
    TOP_K = int(custom_params.get('TOP_K', 2))

    coins = [c for c in df.columns if not any(suffix in c for suffix in ['_open', '_high', '_low'])]
    if 'BTC' not in coins:
        raise ValueError("数据中必须包含 BTC 作为宏观开关！")

    df_close = df[coins]

    # 在全局 DataFrame 上计算指标 (天然利用预热期)
    returns = df_close.pct_change(MOM_WINDOW)
    high_df = df[[f"{c}_high" for c in coins]].copy()
    high_df.columns = coins
    low_df = df[[f"{c}_low" for c in coins]].copy()
    low_df.columns = coins
    prev_close = df_close.shift(1)

    tr_arr = np.fmax.reduce(
        [(high_df - low_df).values, (high_df - prev_close).abs().values, (low_df - prev_close).abs().values])
    atr = pd.DataFrame(tr_arr, index=df.index, columns=coins).rolling(window=VOL_WINDOW).mean()
    atr_pct = atr / df_close

    adj_mom = returns / (atr_pct + 1e-8)
    volatility = atr_pct

    btc_ma = df['BTC'].rolling(window=BTC_TREND_WINDOW).mean()
    btc_trend_on = df['BTC'] > btc_ma

    # 状态继承
    cash = prev_state['cash']
    positions_dict = prev_state['positions']
    coin_states = prev_state['coin_states']
    trade_logs = []

    for c in coins:
        if c not in coin_states:
            coin_states[c] = {'qty': 0.0, 'cost': 0.0, 'entry_time': None, 'side': None}

    n_coins = len(coins)
    coin_to_idx = {c: idx for idx, c in enumerate(coins)}

    positions_arr = np.zeros(n_coins, dtype=float)
    for idx, c in enumerate(coins):
        if c in positions_dict:
            positions_arr[idx] = positions_dict[c]

    close_arr = df_close.values
    vol_arr = volatility[coins].values
    mom_arr = adj_mom[coins].values
    btc_trend_arr = btc_trend_on.values
    time_index = df.index

    # 核心：精确定位该年实际可交易的起点 (1月1日)
    year_start_time = pd.to_datetime(f"{year}-01-01 00:00:00")
    trading_mask = time_index >= year_start_time

    if not trading_mask.any():
        return prev_state, [], pd.DataFrame()

    start_trade_idx = np.where(trading_mask)[0][0]
    n_steps = len(df) - start_trade_idx
    equity_values = np.empty(n_steps, dtype=float)
    equity_times = time_index[start_trade_idx:]

    for j, i in enumerate(range(start_trade_idx, len(df))):
        current_time = time_index[i]
        prices_row = close_arr[i]

        current_equity = cash + np.dot(positions_arr, prices_row)
        equity_values[j] = current_equity

        top_long_coins = []
        top_short_coins = []
        current_mom = mom_arr[i]

        if btc_trend_arr[i]:
            if GLOBAL_TRADE_MODE in ['BOTH', 'LONG_ONLY']:
                mask = ~np.isnan(current_mom) & (current_mom > 0)
                if mask.any():
                    valid_idx = np.where(mask)[0]
                    valid_vals = current_mom[valid_idx]
                    order = np.argsort(-valid_vals, kind='stable')
                    top_long_coins = [coins[idx] for idx in valid_idx[order[:TOP_K]]]
        else:
            if GLOBAL_TRADE_MODE in ['BOTH', 'SHORT_ONLY']:
                mask = ~np.isnan(current_mom) & (current_mom < 0)
                if mask.any():
                    valid_idx = np.where(mask)[0]
                    valid_vals = current_mom[valid_idx]
                    order = np.argsort(valid_vals, kind='stable')
                    top_short_coins = [coins[idx] for idx in valid_idx[order[:TOP_K]]]

        # 平仓逻辑
        for idx_c in range(n_coins):
            if positions_arr[idx_c] > 0:
                c = coins[idx_c]
                if c not in top_long_coins:
                    sell_amount = positions_arr[idx_c]
                    actual_sell_val = sell_amount * prices_row[idx_c]
                    fee = actual_sell_val * FEE_RATE
                    positions_arr[idx_c] = 0
                    cash += (actual_sell_val - fee)
                    trade_logs.append({
                        "time": current_time, "action": "SELL", "coin": c, "direction": "LONG", "event": "CLOSE",
                        "price": prices_row[idx_c], "amount": sell_amount, "value": actual_sell_val, "fee": fee,
                        "reason": "Signal Exit Long",
                        "target_weight": 0.0  # 🟢【新增】平仓记录为 0
                    })

            elif positions_arr[idx_c] < 0:
                c = coins[idx_c]
                if c not in top_short_coins:
                    buy_amount = abs(positions_arr[idx_c])
                    actual_buy_val = buy_amount * prices_row[idx_c]
                    fee = actual_buy_val * FEE_RATE
                    positions_arr[idx_c] = 0
                    cash -= (actual_buy_val + fee)
                    trade_logs.append({
                        "time": current_time, "action": "BUY", "coin": c, "direction": "SHORT", "event": "CLOSE",
                        "price": prices_row[idx_c], "amount": buy_amount, "value": actual_buy_val, "fee": fee,
                        "reason": "Signal Exit Short",
                        "target_weight": 0.0  # 🟢【新增】平仓记录为 0
                    })

        # 判断是否为当前年份最后一条记录 (修复 BUG 3)
        is_last_bar_of_year = (i == len(df) - 1)

        # 开仓逻辑 (多)
        if top_long_coins and not (is_last_bar_of_year and next_year_coins is not None):
            inv_vols = [1.0 / vol_arr[i, coin_to_idx[c]] if vol_arr[i, coin_to_idx[c]] > 0 else 0 for c in
                        top_long_coins]
            total_inv_vol = sum(inv_vols)
            for k_, c in enumerate(top_long_coins):
                idx_c = coin_to_idx[c]
                if positions_arr[idx_c] == 0 and total_inv_vol > 0:
                    target_weight = min(inv_vols[k_] / total_inv_vol, MAX_WEIGHT)
                    target_val = current_equity * target_weight
                    buy_val = target_val / (1 + FEE_RATE) if cash >= target_val / (1 + FEE_RATE) else cash / (
                            1 + FEE_RATE)
                    if buy_val > 1.0:
                        fee = buy_val * FEE_RATE
                        buy_amount = buy_val / prices_row[idx_c]
                        positions_arr[idx_c] += buy_amount
                        cash -= (buy_val + fee)
                        trade_logs.append({
                            "time": current_time, "action": "BUY", "coin": c, "direction": "LONG", "event": "OPEN",
                            "price": prices_row[idx_c], "amount": buy_amount, "value": buy_val, "fee": fee,
                            "reason": "Signal Entry Long",
                            "target_weight": target_weight  # 🟢【核心新增】真实目标权重
                        })

        # 开仓逻辑 (空)
        if top_short_coins and not (is_last_bar_of_year and next_year_coins is not None):
            inv_vols = [1.0 / vol_arr[i, coin_to_idx[c]] if vol_arr[i, coin_to_idx[c]] > 0 else 0 for c in
                        top_short_coins]
            total_inv_vol = sum(inv_vols)
            for k_, c in enumerate(top_short_coins):
                idx_c = coin_to_idx[c]
                if positions_arr[idx_c] == 0 and total_inv_vol > 0:
                    target_weight = min(inv_vols[k_] / total_inv_vol, MAX_WEIGHT)
                    sell_val = current_equity * target_weight / (1 + FEE_RATE)
                    if sell_val > 1.0:
                        fee = sell_val * FEE_RATE
                        sell_amount = sell_val / prices_row[idx_c]
                        positions_arr[idx_c] -= sell_amount
                        cash += (sell_val - fee)
                        trade_logs.append({
                            "time": current_time, "action": "SELL", "coin": c, "direction": "SHORT", "event": "OPEN",
                            "price": prices_row[idx_c], "amount": sell_amount, "value": sell_val, "fee": fee,
                            "reason": "Signal Entry Short",
                            "target_weight": target_weight  # 🟢【核心新增】真实目标权重
                        })

    # 核心：年底清退不在下一年标的池的资产 (强制平仓)
    if next_year_coins is not None:
        last_time = time_index[-1]
        last_prices_row = close_arr[-1]
        for idx_c in range(n_coins):
            c = coins[idx_c]
            if c not in next_year_coins and abs(positions_arr[idx_c]) > 1e-9:
                amt = abs(positions_arr[idx_c])
                val = amt * last_prices_row[idx_c]
                fee = val * FEE_RATE

                if positions_arr[idx_c] > 0:  # 平多
                    cash += (val - fee)
                    action, direction = "SELL", "LONG"
                else:  # 平空
                    cash -= (val + fee)
                    action, direction = "BUY", "SHORT"

                trade_logs.append({
                    "time": last_time, "action": action, "coin": c, "direction": direction, "event": "CLOSE",
                    "price": last_prices_row[idx_c], "amount": amt, "value": val, "fee": fee,
                    "reason": f"End of Year {year} Delisting",
                    "target_weight": 0.0  # 🟢【新增】退市平仓记录为 0
                })
                positions_arr[idx_c] = 0

        # 修复 BUG 2：清退完成后，补记最后一根 Bar 的真实 equity
        if n_steps > 0:
            equity_values[-1] = cash + np.dot(positions_arr, last_prices_row)

    # 更新持仓与 PNL 追踪
    for log in trade_logs:
        c, event, direction, amt, price, fee, time_ = log['coin'], log['event'], log['direction'], log['amount'], log[
            'price'], log['fee'], log['time']
        if event == 'OPEN':
            coin_states[c]['qty'] = amt
            # 修复 BUG 4：防止除零报错
            if amt > 0:
                coin_states[c]['cost'] = price + (fee / amt) if direction == 'LONG' else price - (fee / amt)
            else:
                coin_states[c]['cost'] = price
            coin_states[c]['side'] = direction
            coin_states[c]['entry_time'] = time_
        elif event == 'CLOSE':
            cost_price = coin_states[c]['cost']
            side = coin_states[c]['side']
            if cost_price > 0 and side is not None:
                pnl = amt * (price - cost_price) - fee if side == 'LONG' else amt * (cost_price - price) - fee
                log['pnl'] = pnl
            coin_states[c] = {'qty': 0.0, 'cost': 0.0, 'entry_time': None, 'side': None}

    new_positions_dict = {coins[i]: positions_arr[i] for i in range(n_coins) if abs(positions_arr[i]) > 1e-9}

    new_state = {
        'cash': cash,
        'positions': new_positions_dict,
        'coin_states': coin_states
    }

    curve_df = pd.DataFrame({'equity': equity_values}, index=equity_times)
    curve_df.index.name = 'time'
    curve_df['btc_trend_on'] = btc_trend_arr[start_trade_idx:]

    return new_state, trade_logs, curve_df


# ==========================================
# 🔴 核心改动 3: 参数动态过滤
# ==========================================
def build_clean_param_grid(min_warmup_bars):
    base_grid = {
        'MOM_WINDOW': sorted([6, 12, 18, 24, 30, 36, 42, 48, 60, 72, 84, 90, 120, 168, 180, 240, 360, 480, 540, 720, 1080, 1440, 2190]),
        'VOL_WINDOW': sorted([6, 12, 18, 24, 36, 42, 48, 60, 84, 90, 120, 180, 240, 360, 480, 540, 720]),
        'BTC_TREND_WINDOW': sorted([30, 60, 90, 120, 180, 270, 300, 360, 540, 720, 1080, 1200, 1440, 2160]),
        'MAX_WEIGHT': sorted([0.05, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00]),
        'TOP_K': [1, 2, 3]
    }

    keys, values = zip(*base_grid.items())
    raw_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    def is_valid(p):
        tk, mw = p['TOP_K'], p['MAX_WEIGHT']
        mom, vol, btc = p['MOM_WINDOW'], p['VOL_WINDOW'], p['BTC_TREND_WINDOW']
        if tk * mw > 1.0 + 1e-9:
            return False
        # ✅ 修复: ATR 需要 VOL_WINDOW + 1 根 K 线，这里补上 +1 保证发车时绝对没有 NaN
        if max(mom, vol, btc) + 1 > min_warmup_bars:
            return False
        return True

    clean_combos = [c for c in raw_combos if is_valid(c)]
    return clean_combos


# ==========================================
# 🔴 核心改动 4: 串行连接执行器 (缝合资金曲线与事件流持久化)
# ==========================================
def _parallel_worker_task(i, params, yearly_data_cache, global_price_df, pool_years, offset):
    param_name = f"Grid_No.{i + 1}_{offset}"
    try:
        # 初始资金状态
        state = {
            'cash': INITIAL_CAPITAL,
            'positions': {},
            'coin_states': {}
        }

        all_logs = []
        all_curves = []

        # 按年份依次串行执行
        for idx, year in enumerate(pool_years):
            df_year = yearly_data_cache[year]
            next_year_coins = None

            if idx < len(pool_years) - 1:
                next_year = pool_years[idx + 1]
                next_df = yearly_data_cache[next_year]
                next_year_coins = [c for c in next_df.columns if not any(s in c for s in ['_open', '_high', '_low'])]

            state, year_logs, curve_df = run_backtest_single_year(
                df=df_year, year=year, prev_state=state, custom_params=params, next_year_coins=next_year_coins
            )

            all_logs.extend(year_logs)
            if not curve_df.empty:
                all_curves.append(curve_df)

        # 修复 BUG 1：判断 all_curves 是否为空
        if not all_curves:
            return False, None, params, param_name, "所有年份均无可交易数据(受预热等约束)"

        # 完美缝合所有年份记录 (修复 BUG 5: 显式 sort_index)
        full_curve_df = pd.concat(all_curves).sort_index()
        full_curve_df = full_curve_df[~full_curve_df.index.duplicated(keep='last')]
        logs_df = pd.DataFrame(all_logs)

        # 🟢【新增核心逻辑：提取纯净事件流并持久化保存】
        if all_logs:
            pure_events = []
            for log in all_logs:
                pure_events.append({
                    "time": log["time"],
                    "coin": log["coin"],
                    "event": log["event"],
                    "direction": log["direction"],
                    "action": log["action"],
                    "price": log["price"],
                    "target_weight": log.get("target_weight", 0.0)
                })

            events_df = pd.DataFrame(pure_events)
            # 建立用于保存事件流的专门子目录
            save_dir = r'W:\project\python_project\oke_auto_trade\param_search_results\event_streams'
            os.makedirs(save_dir, exist_ok=True)

            # 使用参数组合ID和交易模式命名文件，方便后续组合
            event_save_path = os.path.join(save_dir, f"{GLOBAL_TRADE_MODE}_{param_name}_events.csv")
            events_df.to_csv(event_save_path, index=False, encoding='utf-8-sig')
        # 🟢 ========================================

        # 核心修复: 在所有年份跑完后，获取最后一年的最后一根收盘价，精确计算包含持仓市值的最终资金
        last_year = pool_years[-1]
        last_df = yearly_data_cache[last_year]
        last_close_row = last_df.iloc[-1]

        final_equity_computed = state['cash']
        for c, qty in state['positions'].items():
            if c in last_close_row.index:
                final_equity_computed += qty * last_close_row[c]

        # 全局重新计算一次综合评价所需的资金指标
        metrics = calculate_comprehensive_metrics(
            logs_df=logs_df,
            curve_df=full_curve_df,
            price_df=global_price_df,  # 全局价格表
            custom_params=params,
            param_name=param_name,
            initial_capital=INITIAL_CAPITAL,
            fee_rate=FEE_RATE,
            final_equity_override=final_equity_computed  # 强制传入精确算好的最终资金
        )
        return True, pd.DataFrame([metrics]), params, param_name, None

    except Exception as e:
        import traceback
        return False, None, params, param_name, traceback.format_exc()


def append_benchmarks_to_existing_csv(result_csv_path, global_price_df):
    print("\n" + "═" * 78)
    print(f"🛠️ 启动后处理：构建【动态合成大盘指数】并注入基准指标")
    print("═" * 78)

    if not os.path.exists(result_csv_path): return
    results_df = pd.read_csv(result_csv_path)

    # 1. 初始化大盘指数资金曲线
    bench_equity = pd.Series(index=global_price_df.index, dtype=float)
    current_capital = 1.0  # 基准指数的初始净值设为 1.0

    annual_benchmarks = {}

    # 2. 按年生成动态等权指数并拼接
    pool_years = sorted(list(set(global_price_df.index.year)))

    for year in pool_years:
        # 切出当年的数据
        year_mask = global_price_df.index.year == year
        year_df = global_price_df.loc[year_mask]
        main_cols = [c for c in year_df.columns if not any(s in c for s in ['_open', '_high', '_low'])]
        year_df = year_df[main_cols]
        # 🎯 核心逻辑：剔除当年全为 NaN 的币（即当年不在标的池的币种，如 2021 年的 SOL）
        year_coins_df = year_df.dropna(axis=1, how='all')

        if year_coins_df.empty:
            continue

        # 将年初第一根 K 线价格作为基准 1.0（归一化）
        start_prices = year_coins_df.iloc[0]
        normalized_df = year_coins_df / start_prices

        # 当年等权组合的走势曲线
        year_curve = normalized_df.mean(axis=1)

        # 按照上一年年末的资金量，进行等比例缩放（资金继承）
        bench_equity.loc[year_mask] = year_curve * current_capital

        # 更新跨年继承资金
        current_capital = bench_equity.loc[year_mask].iloc[-1]

        # 记录当年的大盘收益率
        annual_benchmarks[year] = year_curve.iloc[-1] - 1.0

    # 3. 补全可能的 NaN（用前值填充）
    bench_equity = bench_equity.ffill()

    # 4. 计算真正的全局大盘指标 (基于合成指数曲线)
    global_bench_return = bench_equity.iloc[-1] - 1.0

    # 计算全局大盘最大回撤
    roll_max = bench_equity.cummax()
    drawdowns = (bench_equity - roll_max) / roll_max
    global_bench_max_dd = drawdowns.min()

    # 5. 写入 CSV 报表
    results_df['benchmark_global_return'] = global_bench_return
    results_df['benchmark_global_max_dd'] = global_bench_max_dd

    for year, b_ret in annual_benchmarks.items():
        results_df[f'benchmark_year_{year}_return'] = b_ret

    # 6. 计算极其严谨的超额收益 (Alpha)
    if 'total_return' in results_df.columns:
        results_df['excess_global_return'] = results_df['total_return'] - results_df['benchmark_global_return']

    for year, b_ret in annual_benchmarks.items():
        if f'year_{year}_return' in results_df.columns:
            results_df[f'year_{year}_excess_return'] = results_df[f'year_{year}_return'] - b_ret

    output_path = result_csv_path.replace('.csv', '_with_Benchmark.csv')
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(
        f"📊 动态大盘指数 (等权复利) 终极表现: 收益率 {global_bench_return * 100:+.2f}% | 最大回撤 {global_bench_max_dd * 100:.2f}%")
    print(f"✅ 科学基准注入完成！新文件已保存至: {output_path}")


# ==========================================
# 🔴 核心改动 5: 组装引擎执行搜索 (支持动态相位错位)
# ==========================================
def run_grid_search(time_offset='0h', max_workers=15):
    print("\n" + "═" * 70)
    print(f"🚀 启动参数暴力搜索引擎 (动态标的池/状态流转版 | 相位错位: {time_offset})")
    print("═" * 70)

    # 1. 组装缓存层，严格获取全局最小可用 warmup (🔴 传入 time_offset)
    yearly_data_cache, global_price_df, min_warmup_bars = prepare_environment(YEARLY_POOL_CONFIG,
                                                                              time_offset=time_offset)
    pool_years = sorted(list(yearly_data_cache.keys()))

    # 2. 生成过滤参数 (利用 min_warmup_bars)
    param_combinations = build_clean_param_grid(min_warmup_bars)
    total_combos = len(param_combinations)

    if total_combos == 0:
        print("❌ 警告：所有参数组合都被严格预热约束拦截过滤掉了！请缩小参数步长或放宽标的池时间！")
        return pd.DataFrame()

    keys = list(param_combinations[0].keys())
    save_dir = r'W:\project\python_project\oke_auto_trade\param_search_results'

    # 🔴 [核心修改]: 保存的 CSV 文件名称动态体现 time_offset
    filename = f"grid_search_{total_combos}_{GLOBAL_TRADE_MODE}_dynamic_pool_offset_{time_offset}.csv"
    save_path = os.path.join(save_dir, filename)
    # if os.path.exists(save_path):
    #     print(f"⚠️ 注意：已存在同名结果文件 {filename}，即将覆盖！如需保留请先备份或改名。")
    #     return

    print(f"\n   ► 包含交易年份: {pool_years}")
    print(f"   ► 参数维度: {len(keys)} 维 ({', '.join(keys)})")
    print(f"   ► 总搜索量: {total_combos} 组组合")
    print("═" * 70)

    if not os.path.exists(save_dir): os.makedirs(save_dir)

    all_metrics_df = []
    start_time = time.time()
    completed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_parallel_worker_task, i, params, yearly_data_cache, global_price_df, pool_years, time_offset): (
                i, params)
            for i, params in enumerate(param_combinations)
        }

        for future in as_completed(future_to_task):
            i, params = future_to_task[future]
            success, metrics_df, res_params, param_name, err_msg = future.result()

            completed_count += 1
            progress_pct = completed_count / total_combos * 100
            elapsed = time.time() - start_time
            avg_time = elapsed / completed_count if completed_count > 0 else 0
            eta = avg_time * (total_combos - completed_count)

            print(
                f"\r⏳ 进度: [{completed_count}/{total_combos}] {progress_pct:5.1f}% | 耗时: {elapsed:.1f}s | 预估剩余: {eta:.1f}s | 最新完成: {res_params}",
                end="")

            if success:
                all_metrics_df.append(metrics_df)
            else:
                print(f"\n❌ [异常捕获] {param_name} 测试失败 | 错误信息:\n{err_msg}")

    print("\n\n✅ 搜索任务执行完毕！正在聚合保存数据...")
    if not all_metrics_df: return pd.DataFrame()

    aggregated_df = pd.concat(all_metrics_df, ignore_index=True)
    aggregated_df.to_csv(save_path, index=False, encoding='utf-8-sig')

    append_benchmarks_to_existing_csv(save_path, global_price_df)

    print(f"⏱️ 总耗时: {(time.time() - start_time) / 60:.2f} 分钟")
    print("═" * 70)

    return aggregated_df


def fast_combined_replay(long_events_file, short_events_file, global_price_df, initial_capital=10000.0,
                         fee_rate=0.0005):
    """
    极速多空信号缝合与复利推演引擎 (V2 修复版：精细化成本追踪与防幽灵仓位)
    """
    print(f"🚀 开始极速推演 BOTH 模式 (引擎 V2版)...")

    # 1. 读取并合并事件流
    df_long = pd.read_csv(long_events_file) if long_events_file else pd.DataFrame()
    df_short = pd.read_csv(short_events_file) if short_events_file else pd.DataFrame()

    df_all = pd.concat([df_long, df_short], ignore_index=True)
    if df_all.empty:
        print("⚠️ 警告：多空事件流均为空！")
        return pd.DataFrame(), pd.DataFrame()

    df_all['time'] = pd.to_datetime(df_all['time'])
    df_all.sort_values('time', inplace=True)

    # 2. 虚拟账户状态初始化
    cash = float(initial_capital)
    # 🟢 修复1：加强状态机的记忆，记录真实带有正负号的 amount 和 cost_price
    positions = {}  # 格式: {coin: {'amount': 0.0, 'cost_price': 0.0, 'side': None}}

    replay_logs = []

    # 用于直接快照，摒弃二次循环
    pos_records = []
    cash_records = []

    # 3. 按时间截面分组重演
    grouped_events = df_all.groupby('time')

    for t, group in grouped_events:
        if t in global_price_df.index:
            prices_row = global_price_df.loc[t]
        else:
            prices_row = global_price_df.asof(t)

        # ==========================================
        # 计算发信号这一刻的【期初总净值】
        # ==========================================
        current_equity = cash
        for c, pos_info in positions.items():
            amt = pos_info['amount']  # 空头是负数，直接相加
            if abs(amt) > 1e-9 and c in prices_row:
                current_equity += amt * prices_row[c]

        closes = group[group['event'] == 'CLOSE']
        opens = group[group['event'] == 'OPEN']

        # ==========================================
        # 执行平仓 (释放资金并计算精准 PNL)
        # ==========================================
        for _, row in closes.iterrows():
            c = row['coin']
            if c not in positions or abs(positions[c]['amount']) < 1e-9:
                continue

            # 🟢 调出记忆的开仓数据
            amt = abs(positions[c]['amount'])
            cost_price = positions[c]['cost_price']
            side = positions[c]['side']
            price = row['price']

            val = amt * price
            fee = val * fee_rate

            if side == 'LONG':
                cash += (val - fee)
                pnl = amt * (price - cost_price) - fee
            else:
                cash -= (val + fee)
                pnl = amt * (cost_price - price) - fee

            # 彻底清空仓位状态
            positions[c] = {'amount': 0.0, 'cost_price': 0.0, 'side': None}

            replay_logs.append({
                "time": t, "action": row['action'], "coin": c,
                "direction": row['direction'], "event": "CLOSE",
                "price": price, "amount": amt, "value": val, "fee": fee,
                "pnl": pnl  # 🟢 修复1：录入真实的盈亏数字
            })

        # ==========================================
        # 执行开仓 (应用组合后的资金池)
        # ==========================================
        for _, row in opens.iterrows():
            c = row['coin']
            weight = row['target_weight']
            price = row['price']
            direction = row['direction']

            target_val = current_equity * weight

            if direction == 'LONG':
                buy_val = target_val / (1 + fee_rate) if cash >= target_val / (1 + fee_rate) else cash / (1 + fee_rate)
                if buy_val > 1.0:
                    fee = buy_val * fee_rate
                    buy_amount = buy_val / price
                    cost_price = price + (fee / buy_amount)

                    # 记忆多头仓位
                    positions[c] = {'amount': buy_amount, 'cost_price': cost_price, 'side': direction}
                    cash -= (buy_val + fee)

                    replay_logs.append({
                        "time": t, "action": row['action'], "coin": c,
                        "direction": direction, "event": "OPEN",
                        "price": price, "amount": buy_amount, "value": buy_val, "fee": fee
                    })
            else:  # SHORT
                sell_val = target_val / (1 + fee_rate)
                if sell_val > 1.0:
                    fee = sell_val * fee_rate
                    sell_amount = sell_val / price
                    cost_price = price - (fee / sell_amount)

                    # 记忆空头仓位 (录入负数 amount 方便算市值)
                    positions[c] = {'amount': -sell_amount, 'cost_price': cost_price, 'side': direction}
                    cash += (sell_val - fee)

                    replay_logs.append({
                        "time": t, "action": row['action'], "coin": c,
                        "direction": direction, "event": "OPEN",
                        "price": price, "amount": sell_amount, "value": sell_val, "fee": fee
                    })

        # 🟢 修复2：截面计算完，直接在这里打“确定性快照”
        current_pos_snapshot = {c: info['amount'] for c, info in positions.items() if abs(info['amount']) > 1e-9}
        pos_records.append((t, current_pos_snapshot))
        cash_records.append((t, cash))

    # ==========================================
    # 生成绝对严谨的时间轴资金曲线
    # ==========================================
    logs_df = pd.DataFrame(replay_logs)
    print(f"   ► 正在重构合并后的资金曲线 (修复防幽灵BUG)...")

    if pos_records and cash_records:
        df_pos = pd.DataFrame([p[1] for p in pos_records], index=[p[0] for p in pos_records])
        df_cash = pd.Series([c[1] for c in cash_records], index=[c[0] for c in cash_records])

        # 利用 reindex 纯净地向未来填充状态（0.0 就是 0.0，完美解决幽灵仓位）
        df_pos = df_pos.reindex(global_price_df.index, method='ffill').fillna(0.0)
        df_cash = df_cash.reindex(global_price_df.index, method='ffill').fillna(initial_capital)

        coins_in_matrix = [c for c in df_pos.columns if c in global_price_df.columns]
        equity_values = df_cash + (df_pos[coins_in_matrix] * global_price_df[coins_in_matrix]).sum(axis=1)
    else:
        equity_values = pd.Series(initial_capital, index=global_price_df.index)

    curve_df = pd.DataFrame({'equity': equity_values}, index=global_price_df.index)
    curve_df.index.name = 'time'

    print(f"✅ BOTH 模式重演完成！最终资金: {curve_df['equity'].iloc[-1]:.2f}")

    return logs_df, curve_df


def print_performance_report(metrics):
    """
    将 metrics 字典转化为高逼格的终端控制台输出
    """

    # 安全获取并处理潜在的 NaN 值，防止报错
    def safe_get(key, default=0.0):
        val = metrics.get(key, default)
        return default if val is None or np.isnan(val) else val

    # 提取并计算格式化所需的变量
    trades = int(safe_get('total_closed_trades', 0))
    win_rate = safe_get('win_rate', 0.0) * 100
    pl_ratio = safe_get('profit_loss_ratio', 0.0)

    total_ret = safe_get('total_return', 0.0) * 100
    annual_ret = safe_get('annual_return', 0.0) * 100

    max_dd = safe_get('max_drawdown', 0.0) * 100
    max_uw = safe_get('max_time_under_water_days', 0.0)
    avg_uw = safe_get('avg_recovery_time_days', 0.0)

    # 计算最长水下期占总回测时间的百分比
    days_passed = safe_get('days_passed', 1)
    uw_ratio = (max_uw / days_passed) * 100 if days_passed > 0 else 0.0

    sortino = safe_get('sortino_ratio', 0.0)
    monthly_win_ratio = safe_get('monthly_positive_ratio', 0.0) * 100

    hhi = safe_get('asset_hhi', 0.0)
    top1_pnl = safe_get('top1_pnl_ratio', 0.0) * 100

    # 开始打印酷炫日志
    print("\n [📊 核心基础绩效]")
    print(f"   ├─ 交易统计: {trades} 笔平仓 | 胜率: {win_rate:.1f}% | 盈亏比: {pl_ratio:.2f}")

    # 收益区分正负颜色标志（这里用 +/- 直观显示）
    t_sign = "+" if total_ret > 0 else ""
    a_sign = "+" if annual_ret > 0 else ""
    print(f"   ├─ 收益状况: 总收益: {t_sign}{total_ret:.1f}% | 年化收益: {a_sign}{annual_ret:.1f}%")

    print(
        f"   ├─ 回撤体验: 最大回撤: {max_dd:.1f}% | 最长水下期: {max_uw:.1f}天 (占回测总时长 {uw_ratio:.1f}%) | 平均恢复: {avg_uw:.1f}天")
    print(f"   ├─ 风险调整: sortino_ratio(主目标): {sortino:.3f} | 盈利月占比: {monthly_win_ratio:.1f}%")
    print(f"   └─ 集中度险: 币种HHI: {hhi:.3f} | 最赚1笔占比: {top1_pnl:.1f}%")
    print("-" * 78)

# ==========================================
# 🔴 底部入口：全自动遍历你指定的错位列表
# ==========================================
if __name__ == "__main__":

    # # 1. 正常获取你的全局价格表 (这一步你只需要调用你原有的 prepare_environment 获取)
    # yearly_data_cache, global_price_df, _ = prepare_environment(YEARLY_POOL_CONFIG)
    #
    # # 2. 假设你找到了做多的天选参数日志 和 做空的天选参数日志
    # long_file = r"W:\project\python_project\oke_auto_trade\param_search_results\event_streams\SHORT_ONLY_Grid_No.8_0h_events.csv"
    # short_file = r"W:\project\python_project\oke_auto_trade\param_search_results\event_streams\SHORT_ONLY_Grid_No.8_0h_events.csv"
    #
    # # 3. 极速缝合并重演！
    # combined_logs, combined_curve = fast_combined_replay(
    #     long_events_file=long_file,
    #     short_events_file=short_file,
    #     global_price_df=global_price_df,
    #     initial_capital=10000.0,
    #     fee_rate=0.0005
    # )
    #
    # # 4. 把合并后的结果直接喂给你的指标计算函数
    # metrics = calculate_comprehensive_metrics(
    #     logs_df=combined_logs,
    #     curve_df=combined_curve,
    #     price_df=global_price_df,
    #     custom_params={'COMBINED': 'Long_15_Short_82'},
    #     param_name="Super_Both_Strategy",
    #     final_equity_override=combined_curve['equity'].iloc[-1]
    # )
    #
    # # 🔴 调用打印函数，享受视觉上的极致体验！
    # print_performance_report(metrics)



    # 定义需要依次回测的时间错位列表
    offsets = ['30min', '1h', '2h', '3h','0h']

    for offset in offsets:
        print(f"\n\n{'=' * 80}")
        print(f"🌟 正在启动全局相位测试，当前测试阶段: [ 错位 {offset} ]")
        print(f"{'=' * 80}")

        # 依次回测，并传入当前的 offset
        run_grid_search(time_offset=offset, max_workers=15)

    print("\n🎉 所有时间错位测试已全部执行完毕！请前往 results 文件夹对比各相位的 CSV 表现。")