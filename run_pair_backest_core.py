import itertools
import os
import time

import pandas as pd
import numpy as np
from datetime import timedelta, datetime


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

    # ==========================================
    # ⚠️ 优化点 1：基于1m数据提取 4h 的 OHLC 数据
    # ==========================================
    df_4h_close = price_df_1m.resample('4h').last()
    df_4h_open = price_df_1m.resample('4h').first()
    df_4h_high = price_df_1m.resample('4h').max()
    df_4h_low = price_df_1m.resample('4h').min()

    # 组合为包含所有 OHLC 字段的大 DataFrame
    price_df_4h = df_4h_close.copy()
    for c in df_4h_close.columns:
        price_df_4h[f"{c}_open"] = df_4h_open[c]
        price_df_4h[f"{c}_high"] = df_4h_high[c]
        price_df_4h[f"{c}_low"] = df_4h_low[c]

    if not df_4h_close.empty:
        print(f"\n📈 【共有区间内各标的表现 (Buy & Hold)】:")
        roll_max = df_4h_close.cummax()
        drawdowns = (df_4h_close - roll_max) / roll_max
        max_drawdowns_pct = drawdowns.min() * 100

        total_pct_change = 0.0
        num_coins = len(df_4h_close.columns)

        for c in df_4h_close.columns:
            start_price = df_4h_close[c].iloc[0]
            end_price = df_4h_close[c].iloc[-1]
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

        print("-" * 50)
        print(f"   >>> 📅 【全局基准：各年度等权大盘表现】")
        for year, group in df_4h_close.groupby(df_4h_close.index.year):
            start_prices = group.iloc[0]
            end_prices = group.iloc[-1]
            pct_changes = (end_prices - start_prices) / start_prices * 100
            avg_beta = pct_changes.mean()
            coin_details = ", ".join([f"{c}: {pct:+.1f}%" for c, pct in pct_changes.items()])
            print(f"            ► {year}年: {avg_beta:>+7.2f}% | [{coin_details}]")
        print("=" * 50)

    return price_df_4h


# ==========================================
# 🆕 评价指标辅助函数 (Helper Functions)
# ==========================================
def _compute_gini(values):
    """计算 Gini 系数（衡量利润集中度）"""
    if values is None or len(values) == 0:
        return np.nan
    arr = np.abs(np.asarray(values, dtype=float))
    if arr.sum() == 0:
        return 0.0
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * sorted_arr)) / (n * np.sum(sorted_arr)) - (n + 1.0) / n)


def _compute_underwater_periods(curve_df):
    """计算所有水下时段（单位：天），返回 (max, mean, list)"""
    if curve_df.empty:
        return 0.0, 0.0, []
    cum_max = curve_df['equity'].cummax()
    drawdown = (curve_df['equity'] - cum_max) / cum_max
    underwater = drawdown < -1e-9
    if not underwater.any():
        return 0.0, 0.0, []

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

    if not periods:
        return 0.0, 0.0, []
    return float(max(periods)), float(np.mean(periods)), periods


def _compute_time_in_market(logs_df, time_index):
    """根据交易日志重建持仓时间线，计算 Time in Market 占比"""
    if logs_df is None or logs_df.empty or len(time_index) == 0:
        return 0.0
    logs_sorted = logs_df.sort_values('time').reset_index(drop=True)
    events = [(row['time'], row['coin'], row['action'], row['amount'])
              for _, row in logs_sorted.iterrows()]

    holdings = {}
    in_market = np.zeros(len(time_index), dtype=bool)
    event_idx = 0

    for i, t in enumerate(time_index):
        while event_idx < len(events) and events[event_idx][0] <= t:
            _, coin, action, amt = events[event_idx]
            if action == 'BUY':
                holdings[coin] = holdings.get(coin, 0) + amt
            else:  # SELL
                holdings[coin] = holdings.get(coin, 0) - amt
                if abs(holdings[coin]) < 1e-9:
                    holdings[coin] = 0
            event_idx += 1
        in_market[i] = any(qty > 1e-9 for qty in holdings.values())

    return float(in_market.sum() / len(in_market))


def _compute_mae_mfe(logs_df, price_df):
    """根据交易日志和价格 OHLC 数据，计算每笔交易的 MAE / MFE / 持仓时长"""
    out = {'mae': [], 'mfe': [], 'holding_h': []}
    if logs_df is None or logs_df.empty:
        return out

    logs_sorted = logs_df.sort_values('time').reset_index(drop=True)
    states = {}

    for _, log in logs_sorted.iterrows():
        c, t, p = log['coin'], log['time'], log['price']
        action, amt = log['action'], log['amount']

        if action == 'BUY':
            if c not in states or states[c]['qty'] == 0:
                states[c] = {'qty': amt, 'cost': p, 'entry': t}
            else:
                old_qty = states[c]['qty']
                old_cost = states[c]['cost']
                new_qty = old_qty + amt
                states[c]['cost'] = (old_qty * old_cost + amt * p) / new_qty
                states[c]['qty'] = new_qty
        elif action == 'SELL':
            if c in states and states[c]['qty'] > 0:
                entry_t = states[c]['entry']
                entry_p = states[c]['cost']
                out['holding_h'].append((t - entry_t).total_seconds() / 3600.0)

                if f"{c}_high" in price_df.columns and f"{c}_low" in price_df.columns:
                    period = price_df.loc[entry_t:t]
                    if not period.empty and entry_p > 0:
                        ph = float(period[f"{c}_high"].max())
                        pl = float(period[f"{c}_low"].min())
                        out['mfe'].append((ph - entry_p) / entry_p)
                        out['mae'].append((pl - entry_p) / entry_p)

                states[c]['qty'] -= amt
                if states[c]['qty'] < 1e-6:
                    states[c] = {'qty': 0, 'cost': 0, 'entry': None}
    return out


# ==========================================
# 🆕 完整指标计算（按方案的 A/B/C/D/E/F 六大维度组织）
# ==========================================
def calculate_comprehensive_metrics(logs_df, curve_df, price_df, custom_params, param_name,
                                    initial_capital=10000.0, fee_rate=0.0005,
                                    final_equity_override=None):
    """根据回测结果计算完整的评价指标字典 (覆盖方案的 A/B/C/D/E/F 全部六大类)"""
    metrics = {'param_name': param_name}
    metrics.update({f'param_{k}': v for k, v in custom_params.items()})

    bars_per_year = 365 * 6  # 4h K线

    if curve_df is None or curve_df.empty:
        return metrics

    # =============== A. 资金曲线层面 ===============
    final_equity = float(final_equity_override) if final_equity_override is not None else float(curve_df['equity'].iloc[-1])
    metrics['initial_capital'] = float(initial_capital)
    metrics['final_equity'] = final_equity
    metrics['total_return'] = (final_equity - initial_capital) / initial_capital

    days_passed = max(1, (curve_df.index[-1] - curve_df.index[0]).days)
    metrics['days_passed'] = days_passed
    metrics['annual_return'] = ((final_equity / initial_capital) ** (365 / days_passed) - 1) if final_equity > 0 else -1.0

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

    # log(equity) 的 R² (用对数曲线避免后期资金规模扭曲)
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

        metrics['calmar_ratio'] = (metrics['annual_return'] / abs(metrics['max_drawdown'])) if metrics['max_drawdown'] < 0 else float('inf')
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

    # 滚动 12 个月
    monthly_eq = curve_df['equity'].resample('M').last()
    monthly_rets = monthly_eq.pct_change().dropna()

    if len(monthly_rets) >= 12:
        rolling_12m = monthly_eq.pct_change(12).dropna()
        metrics['rolling_12m_return_mean'] = float(rolling_12m.mean())
        metrics['rolling_12m_return_std'] = float(rolling_12m.std())
        metrics['rolling_12m_return_min'] = float(rolling_12m.min())

        roll_sharpe = monthly_rets.rolling(12).apply(
            lambda x: (x.mean() / x.std() * np.sqrt(12)) if x.std() > 0 else 0
        ).dropna()
        metrics['rolling_12m_sharpe_mean'] = float(roll_sharpe.mean())
        metrics['rolling_12m_sharpe_std'] = float(roll_sharpe.std())
        metrics['rolling_12m_sharpe_cv'] = (
            float(abs(roll_sharpe.std() / roll_sharpe.mean()))
            if abs(roll_sharpe.mean()) > 1e-9 else np.nan
        )

        def _r_sortino(x):
            if not (x < 0).any():
                return 0
            d = x[x < 0].std()
            return (x.mean() / d * np.sqrt(12)) if d > 0 else 0

        roll_sortino = monthly_rets.rolling(12).apply(_r_sortino).dropna()
        metrics['rolling_12m_sortino_mean'] = float(roll_sortino.mean())
        metrics['rolling_12m_sortino_std'] = float(roll_sortino.std())
        metrics['rolling_12m_sortino_cv'] = (
            float(abs(roll_sortino.std() / roll_sortino.mean()))
            if abs(roll_sortino.mean()) > 1e-9 else np.nan
        )
    else:
        for k in ['rolling_12m_return_mean', 'rolling_12m_return_std', 'rolling_12m_return_min',
                  'rolling_12m_sharpe_mean', 'rolling_12m_sharpe_std', 'rolling_12m_sharpe_cv',
                  'rolling_12m_sortino_mean', 'rolling_12m_sortino_std', 'rolling_12m_sortino_cv']:
            metrics[k] = np.nan

    # =============== B. 交易层面 ===============
    sell_logs = pd.DataFrame()
    if logs_df is not None and not logs_df.empty:
        sell_logs = logs_df[(logs_df['action'] == 'SELL') & (logs_df['pnl'].notna())].copy()

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
            metrics['top1_pnl_ratio'] = np.nan
            metrics['top3_pnl_ratio'] = np.nan
            metrics['top5_pnl_ratio'] = np.nan

        wins = sell_logs[sell_logs['pnl'] > 0]
        losses = sell_logs[sell_logs['pnl'] <= 0]
        metrics['win_rate'] = float(len(wins) / len(sell_logs))
        metrics['avg_win'] = float(wins['pnl'].mean()) if len(wins) > 0 else 0.0
        metrics['avg_loss'] = float(abs(losses['pnl'].mean())) if len(losses) > 0 else 0.0
        metrics['profit_loss_ratio'] = (metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] > 0 else float('inf')

        metrics['expectancy'] = metrics['pnl_mean']
        if len(pnls) > 1:
            se = float(np.std(pnls)) / np.sqrt(len(pnls))
            metrics['expectancy_ci_low'] = metrics['expectancy'] - 1.96 * se
            metrics['expectancy_ci_high'] = metrics['expectancy'] + 1.96 * se
        else:
            metrics['expectancy_ci_low'] = metrics['expectancy']
            metrics['expectancy_ci_high'] = metrics['expectancy']

        # Drop-Top-K 衰减
        for k in [1, 3, 5]:
            if len(sorted_desc) > k:
                after_drop = float(sorted_desc[k:].sum())
                metrics[f'drop_top{k}_pnl_decay'] = (total_pnl - after_drop) / total_pnl if total_pnl > 0 else 0.0
            else:
                metrics[f'drop_top{k}_pnl_decay'] = 1.0

        # Bootstrap 重采样
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
                  'top1_pnl_ratio', 'top3_pnl_ratio', 'top5_pnl_ratio',
                  'win_rate', 'avg_win', 'avg_loss', 'profit_loss_ratio',
                  'expectancy', 'expectancy_ci_low', 'expectancy_ci_high',
                  'drop_top1_pnl_decay', 'drop_top3_pnl_decay', 'drop_top5_pnl_decay',
                  'bootstrap_pnl_mean_mean', 'bootstrap_pnl_mean_std',
                  'bootstrap_pnl_mean_p5', 'bootstrap_pnl_mean_p95',
                  'bootstrap_positive_prob']:
            metrics[k] = np.nan

    # MAE/MFE/持仓时长
    mm = _compute_mae_mfe(logs_df, price_df)
    if mm['mae']:
        metrics['mae_pct_mean'] = float(np.mean(mm['mae']))
        metrics['mae_pct_median'] = float(np.median(mm['mae']))
        metrics['mae_pct_worst'] = float(np.min(mm['mae']))
        metrics['mfe_pct_mean'] = float(np.mean(mm['mfe']))
        metrics['mfe_pct_median'] = float(np.median(mm['mfe']))
        metrics['mfe_pct_best'] = float(np.max(mm['mfe']))
    else:
        for k in ['mae_pct_mean', 'mae_pct_median', 'mae_pct_worst',
                  'mfe_pct_mean', 'mfe_pct_median', 'mfe_pct_best']:
            metrics[k] = np.nan

    if mm['holding_h']:
        metrics['avg_holding_hours'] = float(np.mean(mm['holding_h']))
        metrics['median_holding_hours'] = float(np.median(mm['holding_h']))
    else:
        metrics['avg_holding_hours'] = 0.0
        metrics['median_holding_hours'] = 0.0

    # =============== C. 标的层面 ===============
    coins = [c for c in price_df.columns if not any(s in c for s in ['_open', '_high', '_low'])]
    asset_records = {}
    total_pnl_all = float(sell_logs['pnl'].sum()) if len(sell_logs) > 0 else 0.0

    for c in coins:
        c_sells = sell_logs[sell_logs['coin'] == c] if len(sell_logs) > 0 else pd.DataFrame()
        if len(c_sells) > 0:
            cp = c_sells['pnl'].values
            cw = c_sells[c_sells['pnl'] > 0]
            cl = c_sells[c_sells['pnl'] <= 0]
            avg_w = float(cw['pnl'].mean()) if len(cw) > 0 else 0.0
            avg_l = float(abs(cl['pnl'].mean())) if len(cl) > 0 else 0.0
            expect = float(np.mean(cp))
            ci_low = expect - 1.96 * float(np.std(cp)) / np.sqrt(len(cp)) if len(cp) > 1 else expect
            net = float(cp.sum())
            share = net / total_pnl_all if total_pnl_all > 0 else 0.0
            asset_records[c] = {
                'trades': int(len(c_sells)),
                'win_rate': float(len(cw) / len(c_sells)),
                'avg_win': avg_w, 'avg_loss': avg_l,
                'profit_loss_ratio': avg_w / avg_l if avg_l > 0 else float('inf'),
                'expectancy': expect, 'expectancy_ci_low': ci_low,
                'net_pnl': net, 'pnl_share': share
            }
        else:
            asset_records[c] = {
                'trades': 0, 'win_rate': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
                'profit_loss_ratio': 0.0, 'expectancy': 0.0, 'expectancy_ci_low': 0.0,
                'net_pnl': 0.0, 'pnl_share': 0.0
            }

    for c, m in asset_records.items():
        for k, v in m.items():
            metrics[f'asset_{c}_{k}'] = v

    # 标的级 HHI 与 Top-1 占比 (基于正利润标的)
    pos_pnls = [m['net_pnl'] for m in asset_records.values() if m['net_pnl'] > 0]
    if pos_pnls:
        tp = sum(pos_pnls)
        shares = [p / tp for p in pos_pnls]
        metrics['asset_hhi'] = float(sum(s ** 2 for s in shares))
        metrics['asset_top1_share'] = float(max(shares))
    else:
        metrics['asset_hhi'] = np.nan
        metrics['asset_top1_share'] = np.nan

    metrics['active_assets'] = int(sum(1 for m in asset_records.values() if m['trades'] > 0))
    metrics['negative_expectancy_assets'] = int(sum(
        1 for m in asset_records.values() if m['expectancy'] < 0 and m['trades'] > 0))

    # 简易 LOO（仅 PnL 减法的近似；完整 LOO 需重跑回测）
    for c, m in asset_records.items():
        metrics[f'asset_{c}_loo_pnl_change'] = -m['net_pnl']
        metrics[f'asset_{c}_loo_total_pnl'] = total_pnl_all - m['net_pnl']

    # =============== D. 时间维度 ===============
    if len(monthly_rets) > 0:
        metrics['monthly_positive_ratio'] = float((monthly_rets > 0).sum() / len(monthly_rets))

        ls, mls = 0, 0
        for r in monthly_rets:
            if r < 0:
                ls += 1
                mls = max(mls, ls)
            else:
                ls = 0
        metrics['max_consecutive_losing_months'] = int(mls)
        metrics['monthly_return_mean'] = float(monthly_rets.mean())
        metrics['monthly_return_std'] = float(monthly_rets.std())

        # 月度时间集中度
        m_pnl_dollar = monthly_eq.diff().dropna()
        pos_m_pnl = m_pnl_dollar[m_pnl_dollar > 0]
        if pos_m_pnl.sum() > 0:
            metrics['top1_month_pnl_ratio'] = float(pos_m_pnl.max() / pos_m_pnl.sum())
            metrics['top3_months_pnl_ratio'] = float(
                pos_m_pnl.nlargest(min(3, len(pos_m_pnl))).sum() / pos_m_pnl.sum()
            )
        else:
            metrics['top1_month_pnl_ratio'] = np.nan
            metrics['top3_months_pnl_ratio'] = np.nan
    else:
        metrics['monthly_positive_ratio'] = 0.0
        metrics['max_consecutive_losing_months'] = 0
        metrics['monthly_return_mean'] = 0.0
        metrics['monthly_return_std'] = 0.0
        metrics['top1_month_pnl_ratio'] = np.nan
        metrics['top3_months_pnl_ratio'] = np.nan

    # 年度统计
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
            annual_records.append({
                'year': int(year), 'return': yr, 'max_dd': ydd,
                'sharpe': ys, 'sortino': yo, 'calmar': yc
            })

    if annual_records:
        adf = pd.DataFrame(annual_records)
        metrics['annual_return_mean'] = float(adf['return'].mean())
        metrics['annual_return_std'] = float(adf['return'].std()) if len(adf) > 1 else 0.0
        metrics['annual_dd_mean'] = float(adf['max_dd'].mean())
        metrics['annual_dd_worst'] = float(adf['max_dd'].min())

        finite_sortino = adf['sortino'][np.isfinite(adf['sortino'])]
        metrics['annual_sortino_mean'] = float(finite_sortino.mean()) if len(finite_sortino) > 0 else 0.0
        metrics['annual_sortino_std'] = float(finite_sortino.std()) if len(finite_sortino) > 1 else 0.0

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
        for k in ['annual_return_mean', 'annual_return_std', 'annual_dd_mean', 'annual_dd_worst',
                  'annual_sortino_mean', 'annual_sortino_std', 'annual_calmar_mean',
                  'profitable_years_ratio']:
            metrics[k] = 0.0
        metrics['profitable_years'] = 0
        metrics['total_years'] = 0

    # 子样本一致性 (上半段 vs 下半段)
    mid = len(curve_df) // 2
    if mid > 10 and len(curve_df) - mid > 10:
        fh = curve_df.iloc[:mid]
        sh = curve_df.iloc[mid:]
        fhs, fhe = float(fh['equity'].iloc[0]), float(fh['equity'].iloc[-1])
        shs, she = float(sh['equity'].iloc[0]), float(sh['equity'].iloc[-1])
        metrics['first_half_return'] = (fhe - fhs) / fhs if fhs > 0 else 0.0
        metrics['second_half_return'] = (she - shs) / shs if shs > 0 else 0.0
        metrics['half_return_diff'] = metrics['second_half_return'] - metrics['first_half_return']
        metrics['first_half_max_dd'] = float(((fh['equity'] - fh['equity'].cummax()) / fh['equity'].cummax()).min())
        metrics['second_half_max_dd'] = float(((sh['equity'] - sh['equity'].cummax()) / sh['equity'].cummax()).min())
    else:
        for k in ['first_half_return', 'second_half_return', 'half_return_diff',
                  'first_half_max_dd', 'second_half_max_dd']:
            metrics[k] = np.nan

    # 不同 regime 下的收益拆分 (使用 BTC 趋势作为 regime 代理)
    if 'BTC' in price_df.columns and 'BTC_TREND_WINDOW' in custom_params:
        btw = custom_params['BTC_TREND_WINDOW']
        btc_ma = price_df['BTC'].rolling(window=btw).mean()
        btc_on = (price_df['BTC'] > btc_ma).reindex(curve_df.index).fillna(False)
        btc_on_lag = btc_on.shift(1).fillna(False)

        bull_mask = btc_on_lag.reindex(rets_4h.index).fillna(False)
        bull_rets = rets_4h[bull_mask]
        bear_rets = rets_4h[~bull_mask]

        if len(bull_rets) > 0:
            metrics['bull_regime_total_return'] = float((1 + bull_rets).prod() - 1)
            metrics['bull_regime_bars'] = int(len(bull_rets))
            metrics['bull_regime_sharpe'] = float(bull_rets.mean() / bull_rets.std() * np.sqrt(bars_per_year)) if bull_rets.std() > 0 else 0.0
        else:
            metrics['bull_regime_total_return'] = 0.0
            metrics['bull_regime_bars'] = 0
            metrics['bull_regime_sharpe'] = 0.0

        if len(bear_rets) > 0:
            metrics['bear_regime_total_return'] = float((1 + bear_rets).prod() - 1)
            metrics['bear_regime_bars'] = int(len(bear_rets))
            metrics['bear_regime_sharpe'] = float(bear_rets.mean() / bear_rets.std() * np.sqrt(bars_per_year)) if bear_rets.std() > 0 else 0.0
        else:
            metrics['bear_regime_total_return'] = 0.0
            metrics['bear_regime_bars'] = 0
            metrics['bear_regime_sharpe'] = 0.0

    # =============== F. 鲁棒性 (成本敏感性曲线) ===============
    if logs_df is not None and not logs_df.empty:
        bv = float(logs_df[logs_df['action'] == 'BUY']['value'].sum())
        sv = float(logs_df[logs_df['action'] == 'SELL']['value'].sum())
        total_vol = bv + sv
        ofe = final_equity

        for tf in [0.0010, 0.0020, 0.0030]:
            ef = max(0, tf - fee_rate)
            el = total_vol * ef
            seq = ofe - el
            sret = (seq - initial_capital) / initial_capital
            sann = ((seq / initial_capital) ** (365 / days_passed) - 1) if days_passed > 0 and seq > 0 else -1.0
            metrics[f'cost_stress_{int(tf * 10000)}bps_return'] = float(sret)
            metrics[f'cost_stress_{int(tf * 10000)}bps_annual'] = float(sann)

        metrics['total_trading_volume'] = total_vol
        metrics['total_fee_paid'] = float(logs_df['fee'].sum())
        metrics['fee_to_pnl_ratio'] = (
            float(metrics['total_fee_paid'] / abs(total_pnl_all))
            if abs(total_pnl_all) > 0 else np.nan
        )
    else:
        for tf in [10, 20, 30]:
            metrics[f'cost_stress_{tf}bps_return'] = np.nan
            metrics[f'cost_stress_{tf}bps_annual'] = np.nan
        metrics['total_trading_volume'] = 0.0
        metrics['total_fee_paid'] = 0.0
        metrics['fee_to_pnl_ratio'] = np.nan

    return metrics


def _print_comprehensive_panel(metrics, param_name=""):
    """打印完整指标面板 (按方案的 A/B/C/D/E/F 分维度展示)"""
    print("\n" + "═" * 78)
    print(f"📊 【完整评价指标面板】: {param_name}")
    print("═" * 78)

    g = metrics.get

    def _fmt_inf(v, fmt="{:>+8.3f}", inf_str="    +inf"):
        if v == float('inf'):
            return inf_str
        if v == float('-inf'):
            return "    -inf"
        if isinstance(v, float) and (np.isnan(v) or not np.isfinite(v)):
            return "     N/A"
        return fmt.format(v)

    # A. 资金曲线
    print("\n[A. 资金曲线 & 回撤]")
    print(f"  初始/最终资金:         ${g('initial_capital', 0):>12,.2f}  →  ${g('final_equity', 0):>12,.2f}")
    print(f"  总收益率/年化收益率:   {g('total_return', 0)*100:>+8.2f}%  /  {g('annual_return', 0)*100:>+8.2f}%")
    print(f"  Max DD / Avg DD:       {g('max_drawdown', 0)*100:>+8.2f}%  /  {g('avg_drawdown', 0)*100:>+8.2f}%")
    print(f"  Ulcer Index:           {g('ulcer_index', 0):>8.3f}")
    print(f"  Pain Index:            {g('pain_index', 0):>8.3f}")
    print(f"  最长水下时长:          {g('max_time_under_water_days', 0):>8.1f} 天")
    print(f"  平均回撤恢复时长:      {g('avg_recovery_time_days', 0):>8.1f} 天")
    print(f"  log(equity) R²:        {g('log_equity_r2', 0):>8.4f}")
    print(f"  Time in Market:        {g('time_in_market_pct', 0)*100:>8.2f}%")

    # E. 风险调整
    print("\n[E. 风险调整收益]")
    print(f"  Sharpe / Sortino:      {g('sharpe_ratio', 0):>+8.3f}  /  {g('sortino_ratio', 0):>+8.3f}")
    print(f"  Calmar / Omega:        {_fmt_inf(g('calmar_ratio', 0))}  /  {_fmt_inf(g('omega_ratio', 0))}")
    print(f"  Tail Ratio (95/5):     {_fmt_inf(g('tail_ratio', np.nan))}")
    rs_m = g('rolling_12m_sharpe_mean', np.nan)
    if not (isinstance(rs_m, float) and np.isnan(rs_m)):
        rs_cv = g('rolling_12m_sharpe_cv', np.nan)
        print(f"  滚动12M Sharpe (均值/CV):  {rs_m:>+7.3f}  /  {_fmt_inf(rs_cv, '{:>7.3f}', '   N/A')}")
        ro_m = g('rolling_12m_sortino_mean', np.nan)
        ro_cv = g('rolling_12m_sortino_cv', np.nan)
        print(f"  滚动12M Sortino (均值/CV): {ro_m:>+7.3f}  /  {_fmt_inf(ro_cv, '{:>7.3f}', '   N/A')}")
    else:
        print(f"  滚动12M Sharpe/Sortino:    N/A (样本不足12个月)")

    # B. 交易层面
    print("\n[B. 交易层面]")
    print(f"  总动作 / 平仓笔数:     {g('total_actions', 0):>5d}  /  {g('total_closed_trades', 0):>5d}")
    if g('total_closed_trades', 0) > 0:
        print(f"  胜率 / 盈亏比:         {g('win_rate', 0)*100:>6.2f}%  /  {_fmt_inf(g('profit_loss_ratio', 0), '{:>6.2f}', '   inf')}")
        print(f"  平均盈/亏:             ${g('avg_win', 0):>+9.2f}  /  ${g('avg_loss', 0):>9.2f}")
        print(f"  Expectancy:            ${g('expectancy', 0):>+9.2f}")
        print(f"  Expectancy 95% CI:     [${g('expectancy_ci_low', 0):>+8.2f}, ${g('expectancy_ci_high', 0):>+8.2f}]")
        bp = g('bootstrap_positive_prob', np.nan)
        if not np.isnan(bp):
            print(f"  Bootstrap 正期望概率:  {bp*100:>6.2f}%")
        print(f"  PnL Mean / Std:        ${g('pnl_mean', 0):>+9.2f}  /  ${g('pnl_std', 0):>9.2f}")
        print(f"  PnL Skew / Kurt:       {g('pnl_skew', 0):>+8.3f}  /  {g('pnl_kurtosis', 0):>+8.3f}")
        print(f"  PnL Gini系数:          {g('pnl_gini', 0):>8.4f}")

        t1 = g('top1_pnl_ratio', np.nan)
        if not (isinstance(t1, float) and np.isnan(t1)):
            print(f"  Top-1/3/5 利润占比:    "
                  f"{t1*100:>5.1f}% / {g('top3_pnl_ratio', 0)*100:>5.1f}% / {g('top5_pnl_ratio', 0)*100:>5.1f}%")
        print(f"  Drop-Top1/3/5 衰减:    "
              f"{g('drop_top1_pnl_decay', 0)*100:>5.1f}% / "
              f"{g('drop_top3_pnl_decay', 0)*100:>5.1f}% / "
              f"{g('drop_top5_pnl_decay', 0)*100:>5.1f}%")

        mae_m = g('mae_pct_mean', np.nan)
        if not (isinstance(mae_m, float) and np.isnan(mae_m)):
            print(f"  MAE 平均/最差:         {mae_m*100:>+6.2f}%  /  {g('mae_pct_worst', 0)*100:>+6.2f}%")
            print(f"  MFE 平均/最佳:         {g('mfe_pct_mean', 0)*100:>+6.2f}%  /  {g('mfe_pct_best', 0)*100:>+6.2f}%")
        print(f"  平均/中位持仓时长:     {g('avg_holding_hours', 0):>6.1f}h  /  {g('median_holding_hours', 0):>6.1f}h")

    # C. 标的集中度
    print("\n[C. 标的集中度]")
    print(f"  活跃标的 / 负期望标的: {g('active_assets', 0)} / {g('negative_expectancy_assets', 0)}")
    hhi = g('asset_hhi', np.nan)
    if not (isinstance(hhi, float) and np.isnan(hhi)):
        print(f"  Asset HHI:             {hhi:>8.4f}")
        print(f"  Top-1 标的利润占比:    {g('asset_top1_share', 0)*100:>6.2f}%")

    asset_keys = sorted([k for k in metrics.keys() if k.startswith('asset_') and k.endswith('_trades')])
    coins_in_metrics = [k[len('asset_'):-len('_trades')] for k in asset_keys]
    if coins_in_metrics:
        print("  各标的明细:")
        print(f"    {'Coin':<6} {'Trades':>6} {'WinRate':>8} {'P/L':>7} {'Expect':>10} {'NetPnL':>11} {'Share':>8}")
        for c in coins_in_metrics:
            tr_n = g(f'asset_{c}_trades', 0)
            if tr_n > 0:
                pl = g(f'asset_{c}_profit_loss_ratio', 0)
                pl_s = "  inf" if pl == float('inf') else f"{pl:>5.2f}"
                print(f"    {c:<6} {tr_n:>6d} {g(f'asset_{c}_win_rate', 0)*100:>7.2f}% "
                      f"{pl_s:>7} {g(f'asset_{c}_expectancy', 0):>+10.2f} "
                      f"{g(f'asset_{c}_net_pnl', 0):>+11.2f} {g(f'asset_{c}_pnl_share', 0)*100:>+7.2f}%")
            else:
                print(f"    {c:<6} {0:>6d} {'--':>8} {'--':>7} {'--':>10} {'--':>11} {'--':>8}")

    # D. 时间稳定性
    print("\n[D. 时间稳定性]")
    print(f"  月度正收益占比:        {g('monthly_positive_ratio', 0)*100:>6.2f}%")
    print(f"  最长连续亏损月:        {g('max_consecutive_losing_months', 0):>2d} 个月")
    t1m = g('top1_month_pnl_ratio', np.nan)
    if not (isinstance(t1m, float) and np.isnan(t1m)):
        print(f"  Top-1/Top-3 月利润占比: {t1m*100:>5.1f}% / {g('top3_months_pnl_ratio', 0)*100:>5.1f}%")
    print(f"  盈利年份占比:          {g('profitable_years_ratio', 0)*100:>6.2f}%  ({g('profitable_years', 0)}/{g('total_years', 0)})")
    print(f"  年度回撤(均值/最差):   {g('annual_dd_mean', 0)*100:>+7.2f}%  /  {g('annual_dd_worst', 0)*100:>+7.2f}%")
    fh_r = g('first_half_return', np.nan)
    if not (isinstance(fh_r, float) and np.isnan(fh_r)):
        print(f"  上半段/下半段 收益:    {fh_r*100:>+7.2f}%  /  {g('second_half_return', 0)*100:>+7.2f}%")
        print(f"  上半段/下半段 Max DD:  {g('first_half_max_dd', 0)*100:>+7.2f}%  /  {g('second_half_max_dd', 0)*100:>+7.2f}%")

    # F. 鲁棒性
    print("\n[F. 鲁棒性 - 成本敏感性曲线]")
    print(f"  原始 5bps (基准年化):  {g('annual_return', 0)*100:>+7.2f}%")
    for fee in [10, 20, 30]:
        ann = g(f'cost_stress_{fee}bps_annual', np.nan)
        if not (isinstance(ann, float) and np.isnan(ann)):
            status = "✅ 存活" if ann > 0 else "💀 破产"
            print(f"  压力 {fee:>2d}bps 年化:        {ann*100:>+7.2f}%  {status}")

    if 'bull_regime_total_return' in metrics:
        bull_r = g('bull_regime_total_return', 0)
        bull_b = g('bull_regime_bars', 0)
        bear_r = g('bear_regime_total_return', 0)
        bear_b = g('bear_regime_bars', 0)
        print(f"  Bull市表现 (BTC>MA):   收益={bull_r*100:>+7.2f}%  Bar数={bull_b}")
        print(f"  Bear市表现 (BTC≤MA):   收益={bear_r*100:>+7.2f}%  Bar数={bear_b}")

    print(f"  总交易额:              ${g('total_trading_volume', 0):>11,.0f}")
    print(f"  总手续费:              ${g('total_fee_paid', 0):>11,.2f}")
    f2p = g('fee_to_pnl_ratio', np.nan)
    if not (isinstance(f2p, float) and np.isnan(f2p)):
        print(f"  手续费/PnL 比率:       {f2p*100:>6.2f}%")

    print("═" * 78)


# ==========================================
# 核心：策略引擎与回测逻辑 (二元进出测试版)
# ==========================================
def run_backtest(df, param_name="默认基准参数", custom_params=None, verbose=True):
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

    if verbose:
        print(f"\n🚀 启动截面动量回测引擎 (验证：信号驱动二元进出)... [{param_name}]")
        print(
            f"   ⚙️ 参数配置: MOM_WIN={MOM_WINDOW}, VOL_WIN={VOL_WINDOW}, BTC_TREND={BTC_TREND_WINDOW}, MAX_WT={MAX_WEIGHT}")

    TOP_K = 2
    FEE_RATE = 0.0005
    INITIAL_CAPITAL = 10000.0

    # 过滤掉衍生出来的 _open, _high, _low 列，只保留基础币种作为交易对象 (即 close)
    coins = [c for c in df.columns if not any(suffix in c for suffix in ['_open', '_high', '_low'])]
    if 'BTC' not in coins:
        raise ValueError("数据中必须包含 BTC 作为宏观开关！")

    df_close = df[coins]
    returns = df_close.pct_change(MOM_WINDOW)

    # =========================================================
    # ⚠️ 优化点 2：计算 ATR_pct，并将 adj_mom 分母替换为 ATR_pct
    # =========================================================
    tr_df = pd.DataFrame(index=df.index, columns=coins)
    for c in coins:
        high = df[f"{c}_high"]
        low = df[f"{c}_low"]
        prev_close = df_close[c].shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr_df[c] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr_df.rolling(window=VOL_WINDOW).mean()
    atr_pct = atr / df_close  # 归一化为百分比形式

    # 【新增归因日志统计计算】：计算原版 std 波动率以对比
    log_returns = np.log(df_close / df_close.shift(1))
    old_volatility = log_returns.rolling(window=VOL_WINDOW).std() * np.sqrt(365 * 6)
    old_adj_mom = returns / (old_volatility + 1e-8)

    # 新版截面动量
    adj_mom = returns / (atr_pct + 1e-8)

    # 将 volatility 变量指向 atr_pct，保证后续仓位控制(inv_vol)同样基于 ATR 进行反向加权
    volatility = atr_pct

    # 打印归因验证统计日志
    if verbose:
        print(f"   [归因验证] 动量计算分母已从 原标准差(Std) 替换为 ATR_pct !")
        rank_corr = old_adj_mom.rank(axis=1).corrwith(adj_mom.rank(axis=1), axis=1).mean()
        print(f"             -> 截面排序平均相关系数 (Rank Corr): {rank_corr:.4f} (越低说明优化影响越大)")
        print(
            f"             -> 均值对比 | ATR_pct: {atr_pct.mean().mean():.4f}  vs  原波动率: {old_volatility.mean().mean():.4f}")

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

    if verbose:
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

    # ==========================================
    # 🆕 完整评价指标计算 + 返回 metrics_df 用于参数搜索聚合
    # ==========================================
    logs_df = pd.DataFrame(trade_logs)

    metrics = calculate_comprehensive_metrics(
        logs_df=logs_df,
        curve_df=curve_df,
        price_df=df,
        custom_params=custom_params,
        param_name=param_name,
        initial_capital=INITIAL_CAPITAL,
        fee_rate=FEE_RATE,
        final_equity_override=final_equity   # 与简易面板的 final_equity 保持一致
    )

    if verbose:
        _print_comprehensive_panel(metrics, param_name=param_name)

    metrics_df = pd.DataFrame([metrics])

    return logs_df, curve_df, metrics_df


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

    if 'time' in logs_df.columns:
        logs_df['year'] = logs_df['time'].dt.year
    else:
        logs_df['year'] = pd.to_datetime(logs_df['time']).dt.year

    sell_logs = logs_df[(logs_df['action'] == 'SELL') & (logs_df['pnl'].notna())]

    # 过滤衍生OHLC列，计算大盘时仅看Close价格
    coins = [c for c in price_df.columns if not any(sub in c for sub in ['_open', '_high', '_low'])]

    for year, group in curve_df.groupby('year'):
        start_eq = group['equity'].iloc[0]
        end_eq = group['equity'].iloc[-1]
        y_ret = (end_eq - start_eq) / start_eq * 100

        roll_max = group['equity'].cummax()
        y_mdd = ((group['equity'] - roll_max) / roll_max).min() * 100

        year_mask = price_df.index.year == year
        year_prices = price_df[year_mask][coins]

        if not year_prices.empty:
            start_prices = year_prices.iloc[0]
            end_prices = year_prices.iloc[-1]
            avg_beta = ((end_prices - start_prices) / start_prices * 100).mean()
        else:
            avg_beta = 0.0

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

        print(
            f"   ► 【{year}年】 策略收益: {y_ret:>+7.2f}% (最大回撤 {y_mdd:>7.2f}%) | 等权大盘: {avg_beta:>+7.2f}% | 超额: {excess_ret:>+7.2f}%")
        print(f"            交易统计: {trade_stats}")

        # === 🎯 核心修改点：新增单币种维度的表现透视 ===
        print(f"            ↳ [各标的归因明细]")
        for c in coins:
            # 1. 计算单币的大盘基准表现
            if not year_prices.empty and c in year_prices.columns:
                c_start = year_prices[c].iloc[0]
                c_end = year_prices[c].iloc[-1]
                c_beta = (c_end - c_start) / c_start * 100
            else:
                c_beta = 0.0

            # 2. 截取该币种当年的卖出平仓日志
            c_sells = y_sells[y_sells['coin'] == c]
            c_trades_cnt = len(c_sells)

            if c_trades_cnt > 0:
                c_win = (c_sells['pnl'] > 0).sum()
                c_win_rate = c_win / c_trades_cnt * 100

                c_sum_win = c_sells[c_sells['pnl'] > 0]['pnl'].sum()
                c_avg_win = c_sum_win / c_win if c_win > 0 else 0.0

                c_loss_cnt = c_trades_cnt - c_win
                c_sum_loss = abs(c_sells[c_sells['pnl'] <= 0]['pnl'].sum())
                c_avg_loss = c_sum_loss / c_loss_cnt if c_loss_cnt > 0 else 0.0

                c_pl_ratio = c_avg_win / c_avg_loss if c_avg_loss > 0 else float('inf')
                c_net_pnl = c_sells['pnl'].sum()

                print(
                    f"              - {c:4s}: 策略净盈亏 ${c_net_pnl:>+8.2f} | 基准: {c_beta:>+7.2f}% | 平仓: {c_trades_cnt:>3d}笔 | 胜率: {c_win_rate:>5.1f}% | 盈亏比: {c_pl_ratio:>4.2f}")
            else:
                print(f"              - {c:4s}: 策略净盈亏 $   +0.00 | 基准: {c_beta:>+7.2f}% | 平仓:   0笔")

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





def run_grid_search():
    """
    执行参数网格搜索并自动持久化保存结果
    （全内置、零参数、零全局变量依赖，真正的高内聚闭环）
    """
    # ==========================================
    # 1. 内部固定：数据源与加载逻辑
    # ==========================================
    print("\n" + "═" * 70)
    print(f"🚀 启动参数暴力搜索引擎 (完全独立闭环版)")
    print("═" * 70)

    # 彻底告别全局变量：把 file_list 和数据处理直接写在函数内部
    file_list = ["kline_data/BTC_ETH_1m.csv", "kline_data/DOGE_SOL_1m.csv", "kline_data/TON_XRP_1m.csv"]

    # ==========================================
    # 2. 内部固定：搜索空间与保存路径
    # ==========================================
    param_grid = {
        'MOM_WINDOW': [60, 120, 180],
        'VOL_WINDOW': [120],
        'BTC_TREND_WINDOW': [360],
        'MAX_WEIGHT': [0.15, 0.20, 0.25, 0.30]
    }

    save_dir = r'W:\project\python_project\oke_auto_trade\param_search_results'

    # ==========================================
    # 3. 解析参数空间
    # ==========================================
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total_combos = len(param_combinations)
    filename = f"grid_search_{total_combos}.csv"
    save_path = os.path.join(save_dir, filename)
    if os.path.exists(save_path):
        result_df = pd.read_csv(save_path)
    print(f"\n   ► 参数维度: {len(keys)} 维 ({', '.join(keys)})")
    print(f"   ► 总搜索量: {total_combos} 组组合")
    print(f"   ► 存储路径: {save_dir}")
    print("═" * 70)
    df_4h = load_and_preprocess_data(file_list)

    # 4. 确保输出目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"📁 已自动创建输出目录: {save_dir}")

    all_metrics_df = []
    start_time = time.time()

    # 5. 开始循环回测
    for i, params in enumerate(param_combinations):
        param_name = f"Grid_No.{i + 1}"

        # 实时打印进度 (覆盖上一行)
        progress_pct = (i + 1) / total_combos * 100
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1) if i > 0 else 0
        eta = avg_time * (total_combos - i - 1)

        print(f"\r⏳ 进度: [{i + 1}/{total_combos}] {progress_pct:5.1f}% | "
              f"耗时: {elapsed:.1f}s | 预估剩余: {eta:.1f}s | 当前测算: {params}", end="")

        try:
            # 这里的 df_4h 来自函数第一步的局部变量，彻底隔离
            _, _, metrics_df = run_backtest(
                df_4h,
                param_name=param_name,
                custom_params=params,
                verbose=False
            )
            all_metrics_df.append(metrics_df)

        except Exception as e:
            # 容错机制：跳过报错的异常参数
            print(f"\n❌ [异常捕获] {param_name} 测试失败: {params} | 错误信息: {e}")

    # 6. 聚合与保存
    print("\n\n✅ 搜索任务执行完毕！正在聚合保存数据...")

    if not all_metrics_df:
        print("⚠️ 未生成任何有效评价数据。")
        return pd.DataFrame()

    aggregated_df = pd.concat(all_metrics_df, ignore_index=True)



    # 导出到指定目录
    aggregated_df.to_csv(save_path, index=False, encoding='utf-8-sig')

    total_time = time.time() - start_time
    print("═" * 70)
    print(f"🎉 搜索圆满结束！")
    print(f"⏱️ 总耗时: {total_time / 60:.2f} 分钟")
    print(f"💾 结果已保存至:\n   {save_path}")
    print("═" * 70)

    return aggregated_df


# ==========================================
# 实际调用示例 (极简到极致)
# ==========================================
if __name__ == "__main__":
    # 现在外部干干净净，什么都不用准备，直接执行
    run_grid_search()