import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ======================================================================
# 数据加载 (原版 100% 保留)
# ======================================================================
def load_and_preprocess_data(file_list):
    print("⏳ 正在解析并合并数据...")
    coin_dfs = {}
    for file in file_list:
        coin_name = os.path.basename(file).split('USDT')[0]
        df = pd.read_csv(file)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume', 'quote_volume']].copy().sort_index()
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
        agg['close'] = df['close'].resample('4h').last()
        agg['volume'] = df['volume'].resample('4h').sum()
        agg['quote_volume'] = df['quote_volume'].resample('4h').sum()
        agg['vwap'] = np.where(agg['volume'] > 0,
                               agg['quote_volume'] / agg['volume'], agg['close'])
        aggregated[coin] = agg

    coins = list(aggregated.keys())
    feature_panels = {}
    for feat in ['close', 'vwap']:
        feature_panels[feat] = pd.concat(
            [aggregated[c][feat].rename(c) for c in coins], axis=1
        )
    return feature_panels


# ======================================================================
# 回测核心 (原版, 仅把 FEE_RATE 参数化)
# ======================================================================
def run_backtest(feature_panels, param_name="默认", custom_params=None,
                 trade_start_date=None, trade_end_date=None, verbose=True):
    default_params = {
        'MOM_WINDOW': 20 * 6,
        'VOL_WINDOW': 20 * 6,
        'BTC_TREND_WINDOW': 90 * 6,
        'MAX_WEIGHT': 0.30,
        'REBAL_BARS': 6,
        'TOP_K': 2,
        'BOTTOM_K': 2,
        'ENABLE_SHORT': False,
        'SHORT_BUDGET': 0.5,
        'SIGNAL_THRESHOLD': 0.0,
        'USE_VWAP': True,
        'USE_BTC_TREND': True,
        'FEE_RATE': 0.0005,         # ⭐ 改动: 仅这一行 — 把费率从硬编码改为参数
    }
    if custom_params is None:
        custom_params = {}
    p = {**default_params, **custom_params}

    FEE_RATE = p['FEE_RATE']        # ⭐ 改动: 仅这一行 — 从 p 读取
    INITIAL_CAPITAL = 10000.0

    close_df = feature_panels['close']
    coins = list(close_df.columns)
    signal_price = feature_panels['vwap'] if p['USE_VWAP'] else close_df

    log_returns = np.log(signal_price / signal_price.shift(1))
    volatility = log_returns.rolling(window=p['VOL_WINDOW']).std() * np.sqrt(365 * 6)
    returns = signal_price.pct_change(p['MOM_WINDOW'])
    adj_mom = returns / (volatility + 1e-8)

    mom_rolling_std = adj_mom.rolling(window=180 * 6, min_periods=60 * 6).std()

    btc_close = close_df['BTC']
    btc_ma = btc_close.rolling(window=p['BTC_TREND_WINDOW']).mean()
    btc_trend_on = btc_close > btc_ma

    cash = INITIAL_CAPITAL
    positions = {coin: 0.0 for coin in coins}
    equity_curve = []

    warmup_idx = max(p['MOM_WINDOW'], p['VOL_WINDOW'], p['BTC_TREND_WINDOW'])
    start_idx_user = (close_df.index.searchsorted(pd.Timestamp(trade_start_date))
                      if trade_start_date else 0)
    end_idx_user = (close_df.index.searchsorted(pd.Timestamp(trade_end_date))
                    if trade_end_date else len(close_df))
    start_idx = max(warmup_idx, start_idx_user)

    if start_idx >= end_idx_user:
        return None

    for i in range(start_idx, end_idx_user):
        current_time = close_df.index[i]
        prices = close_df.iloc[i]

        long_value = sum(max(positions[c], 0) * prices[c] for c in coins)
        short_value = sum(min(positions[c], 0) * prices[c] for c in coins)
        current_equity = cash + long_value + short_value
        equity_curve.append({'time': current_time, 'equity': current_equity})

        if i % p['REBAL_BARS'] != 0:
            continue

        target_weights = {c: 0.0 for c in coins}
        btc_filter = btc_trend_on.iloc[i] if p['USE_BTC_TREND'] else True

        current_mom = adj_mom.iloc[i].dropna()
        current_mom_std = mom_rolling_std.iloc[i]

        if p['SIGNAL_THRESHOLD'] > 0:
            filtered_mom = pd.Series(index=current_mom.index, dtype=float)
            for c in current_mom.index:
                threshold_val = current_mom_std.get(c, np.nan) * p['SIGNAL_THRESHOLD']
                if not np.isnan(threshold_val) and abs(current_mom[c]) > threshold_val:
                    filtered_mom[c] = current_mom[c]
                else:
                    filtered_mom[c] = 0.0
            current_mom = filtered_mom[filtered_mom != 0]

        if not current_mom.empty:
            sorted_mom = current_mom.sort_values(ascending=False)

            if btc_filter:
                long_candidates = sorted_mom[sorted_mom > 0].head(p['TOP_K'])
                if not long_candidates.empty:
                    inv_vol = {c: (1.0 / volatility[c].iloc[i]
                                   if volatility[c].iloc[i] > 0
                                      and not np.isnan(volatility[c].iloc[i]) else 0)
                               for c in long_candidates.index}
                    total = sum(inv_vol.values())
                    for c in long_candidates.index:
                        if total > 0:
                            target_weights[c] = min(inv_vol[c] / total, p['MAX_WEIGHT'])

            if p['ENABLE_SHORT']:
                short_candidates = sorted_mom[sorted_mom < 0].tail(p['BOTTOM_K'])
                if not short_candidates.empty:
                    inv_vol_s = {c: (1.0 / volatility[c].iloc[i]
                                     if volatility[c].iloc[i] > 0
                                        and not np.isnan(volatility[c].iloc[i]) else 0)
                                 for c in short_candidates.index}
                    total = sum(inv_vol_s.values())
                    for c in short_candidates.index:
                        if total > 0:
                            target_weights[c] = -min(
                                inv_vol_s[c] / total * p['SHORT_BUDGET'],
                                p['MAX_WEIGHT']
                            )

        target_values = {c: current_equity * w for c, w in target_weights.items()}

        # 减仓 / 平仓 / 开空
        for c in coins:
            current_value = positions[c] * prices[c]
            target = target_values.get(c, 0)
            diff = target - current_value
            if positions[c] > 0 and diff < -1.0:
                sell_amt = min(abs(diff) / prices[c], positions[c])
                actual = sell_amt * prices[c]
                fee = actual * FEE_RATE
                positions[c] -= sell_amt
                cash += (actual - fee)
                if abs(positions[c]) < 1e-8:
                    positions[c] = 0.0
            if target < -1.0 and positions[c] <= 0:
                add_short_value = abs(target) - abs(positions[c]) * prices[c]
                if add_short_value > 1.0:
                    short_amt = add_short_value / prices[c]
                    fee = add_short_value * FEE_RATE
                    positions[c] -= short_amt
                    cash += (add_short_value - fee)
            if positions[c] < 0 and target > positions[c] * prices[c] + 1.0:
                if target >= 0:
                    cover_amt = abs(positions[c])
                else:
                    cover_amt = (target - positions[c] * prices[c]) / prices[c]
                actual = cover_amt * prices[c]
                fee = actual * FEE_RATE
                positions[c] += cover_amt
                cash -= (actual + fee)
                if abs(positions[c]) < 1e-8:
                    positions[c] = 0.0

        # 加仓 (开多)
        for c in coins:
            current_value = positions[c] * prices[c]
            target = target_values.get(c, 0)
            diff = target - current_value
            if positions[c] >= 0 and target > 1.0 and diff > 1.0:
                buy_val = min(diff / (1 + FEE_RATE),
                              cash / (1 + FEE_RATE) if cash > 0 else 0)
                if buy_val > 1.0:
                    fee = buy_val * FEE_RATE
                    buy_amt = buy_val / prices[c]
                    positions[c] += buy_amt
                    cash -= (buy_val + fee)

    final_long = sum(max(positions[c], 0) * close_df.iloc[end_idx_user - 1][c] for c in coins)
    final_short = sum(min(positions[c], 0) * close_df.iloc[end_idx_user - 1][c] for c in coins)
    final_equity = cash + final_long + final_short
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL

    curve_df = pd.DataFrame(equity_curve).set_index('time')
    curve_df['cum_max'] = curve_df['equity'].cummax()
    curve_df['drawdown'] = (curve_df['equity'] - curve_df['cum_max']) / curve_df['cum_max']
    max_dd = curve_df['drawdown'].min()

    days = (curve_df.index[-1] - curve_df.index[0]).days
    annual = ((final_equity / INITIAL_CAPITAL) ** (365 / days) - 1) if days > 0 and final_equity > 0 else 0
    curve_df['returns'] = curve_df['equity'].pct_change()
    sharpe = (curve_df['returns'].mean() / curve_df['returns'].std() * np.sqrt(365 * 6)
              if curve_df['returns'].std() > 0 else 0)
    calmar = annual / abs(max_dd) if max_dd < -1e-6 else float('inf')

    if verbose:
        print(f"\n📊 {param_name}")
        print(f"   总收益 {total_return * 100:>7.2f}% | 年化 {annual * 100:>6.2f}% | "
              f"回撤 {max_dd * 100:>6.2f}% | 夏普 {sharpe:.2f} | 卡玛 {calmar:.2f}")

    return {'annual': annual, 'max_dd': max_dd, 'sharpe': sharpe,
            'calmar': calmar, 'total_return': total_return}


# ======================================================================
# 走步前向 (原版 100% 保留)
# ======================================================================
def walk_forward_analysis(feature_panels, n_windows=6, custom_params=None, label=""):
    full_index = feature_panels['close'].index
    total_bars = len(full_index)
    bars_per_window = total_bars // n_windows

    print(f"\n{'=' * 70}")
    print(f"🔬 走步前向: {label}")
    print(f"{'=' * 70}")

    results = []
    for w in range(n_windows):
        start_idx = w * bars_per_window
        end_idx = (w + 1) * bars_per_window if w < n_windows - 1 else total_bars
        start_date = full_index[start_idx]
        end_date = full_index[end_idx - 1]

        result = run_backtest(
            feature_panels,
            param_name=f"窗口 {w + 1}/{n_windows} ({start_date.strftime('%y-%m')} → "
                       f"{end_date.strftime('%y-%m')})",
            custom_params=custom_params,
            trade_start_date=start_date,
            trade_end_date=end_date,
            verbose=True
        )
        if result:
            results.append({'window': w + 1, **result})

    if results:
        df = pd.DataFrame(results)
        print(f"\n📊 {label} 稳定性统计:")
        print(f"   卡玛 中位数 {df['calmar'].median():.2f} | "
              f"均值 {df['calmar'].mean():.2f} | "
              f"std {df['calmar'].std():.2f}")
        print(f"   亏损窗口 {(df['total_return'] < 0).sum()}/{len(df)} | "
              f"卡玛<1 窗口 {(df['calmar'] < 1).sum()}/{len(df)}")
    return results


# ======================================================================
# ⭐ 新增: 单参数平原扫描
# ======================================================================
def _safe_calmar(c):
    """处理 inf / nan 卡玛"""
    if np.isinf(c) or np.isnan(c):
        return 999.0
    return min(c, 999.0)


def sweep_param(feature_panels, param_name, values, base_params):
    print(f"\n{'━' * 72}")
    print(f"🔬 单参数扫描: {param_name}")
    print(f"{'━' * 72}")
    print(f"   {'参数':>12}  {'年化':>8}  {'回撤':>8}  {'夏普':>6}  {'卡玛':>6}")
    print(f"   {'-' * 12}  {'-' * 8}  {'-' * 8}  {'-' * 6}  {'-' * 6}")

    results = []
    for v in values:
        p = base_params.copy()
        if param_name == 'K':                           # 同步设置 TOP_K = BOTTOM_K
            p['TOP_K'] = v
            p['BOTTOM_K'] = v
            champ_v = base_params['TOP_K']
        else:
            p[param_name] = v
            champ_v = base_params.get(param_name)

        m = run_backtest(feature_panels, custom_params=p, verbose=False)
        if m is None:
            continue

        results.append({'value': v, 'annual': m['annual'],
                        'dd': m['max_dd'], 'sharpe': m['sharpe'],
                        'calmar': m['calmar']})
        flag = '⭐' if v == champ_v else '  '
        # 时间窗类参数显示成 "120(20d)" 更直观
        if param_name in ['MOM_WINDOW', 'VOL_WINDOW', 'BTC_TREND_WINDOW']:
            v_disp = f"{v}({v // 6}d)"
        else:
            v_disp = str(v)
        print(f"   {flag}{v_disp:>10}  {m['annual'] * 100:>7.2f}%  "
              f"{m['max_dd'] * 100:>7.2f}%  {m['sharpe']:>5.2f}  {m['calmar']:>5.2f}")
    return results


def evaluate_plateau(results, param_name, champion_value):
    if len(results) < 3:
        print("   ⚠️ 样本太少, 跳过平原评估")
        return

    calmars = np.array([_safe_calmar(r['calmar']) for r in results])
    values = [r['value'] for r in results]
    max_c, min_c, med_c = calmars.max(), calmars.min(), np.median(calmars)

    threshold = max_c * 0.7
    n_good = (calmars >= threshold).sum()
    pct = n_good / len(calmars) * 100

    if champion_value in values:
        ci = values.index(champion_value)
        cc = calmars[ci]
        nbs = []
        if ci > 0:
            nbs.append(calmars[ci - 1])
        if ci < len(calmars) - 1:
            nbs.append(calmars[ci + 1])
        nb_avg = np.mean(nbs) if nbs else 0
        nb_ratio = nb_avg / cc if cc > 0 else 0
    else:
        cc, nb_avg, nb_ratio = 0, 0, 0

    print(f"\n   📐 平原评估:")
    print(f"      卡玛  最优 {max_c:.2f}  中位 {med_c:.2f}  最差 {min_c:.2f}")
    print(f"      在最优 70% 之上的参数: {n_good}/{len(calmars)} ({pct:.0f}%)")
    if cc > 0:
        print(f"      冠军({champion_value}) 卡玛 {cc:.2f} | 邻居均值 {nb_avg:.2f} "
              f"| 邻居/冠军 = {nb_ratio * 100:.0f}%")

    if pct >= 70 and nb_ratio >= 0.7:
        print(f"      ✅ 强平原 — 参数极稳健, 不是过拟合")
    elif pct >= 50 and nb_ratio >= 0.5:
        print(f"      ✅ 平原 — 参数稳健")
    elif pct >= 30:
        print(f"      ⚠️ 半平原 — 中等稳健, 邻居略弱")
    else:
        print(f"      ❌ 尖峰风险 — 警告: {param_name} 可能过拟合")


# ======================================================================
# ⭐ 新增: 二维联合扫描
# ======================================================================
def grid_2d_scan(feature_panels, p1_name, p1_vals, p2_name, p2_vals, base_params):
    print(f"\n{'━' * 72}")
    print(f"🔬 二维平原: {p1_name} × {p2_name}  (单元格 = 卡玛)")
    print(f"{'━' * 72}")

    grid = np.full((len(p1_vals), len(p2_vals)), np.nan)
    for i, v1 in enumerate(p1_vals):
        for j, v2 in enumerate(p2_vals):
            p = base_params.copy()
            if p1_name == 'K':
                p['TOP_K'] = v1
                p['BOTTOM_K'] = v1
            else:
                p[p1_name] = v1
            if p2_name == 'K':
                p['TOP_K'] = v2
                p['BOTTOM_K'] = v2
            else:
                p[p2_name] = v2

            m = run_backtest(feature_panels, custom_params=p, verbose=False)
            if m is not None:
                grid[i, j] = _safe_calmar(m['calmar'])

    def _fmt(v):
        if v in ['MOM_WINDOW', 'VOL_WINDOW', 'BTC_TREND_WINDOW']:
            return lambda x: f"{x // 6}d"
        return lambda x: str(x)

    f1, f2 = _fmt(p1_name), _fmt(p2_name)
    print(f"\n   {p2_name} →")
    print(f"   {p1_name:>16}↓  " + "  ".join(f"{f2(v):>7}" for v in p2_vals))
    for i, v1 in enumerate(p1_vals):
        cells = "  ".join(
            f"{grid[i, j]:>7.2f}" if not np.isnan(grid[i, j]) else f"{'NA':>7}"
            for j in range(len(p2_vals)))
        print(f"   {f1(v1):>16}    " + cells)

    valid = grid[~np.isnan(grid)]
    if len(valid) > 0:
        max_c = valid.max()
        pct = (valid >= max_c * 0.7).sum() / len(valid) * 100
        print(f"\n   📐 二维平原占比: {pct:.0f}% (>= {max_c * 0.7:.2f})")
        if pct >= 60:
            print(f"   ✅ 二维平原稳健 — 参数对联合稳健")
        elif pct >= 40:
            print(f"   ⚠️ 二维半平原 — 中等稳健")
        else:
            print(f"   ❌ 二维尖峰风险 — 联合参数可能过拟合")


# ======================================================================
# ⭐ 新增: 成本压力测试
# ======================================================================
def stress_fees(feature_panels, base_params):
    print(f"\n{'━' * 72}")
    print(f"🔬 实盘成本压力测试 (费率敏感性)")
    print(f"{'━' * 72}")
    print(f"   {'费率(bp)':>10}  {'年化':>8}  {'回撤':>8}  {'夏普':>6}  {'卡玛':>6}  {'判定':>6}")
    print(f"   {'-' * 10}  {'-' * 8}  {'-' * 8}  {'-' * 6}  {'-' * 6}  {'-' * 6}")

    base = run_backtest(feature_panels,
                        custom_params={**base_params, 'FEE_RATE': 0.0005},
                        verbose=False)
    base_calmar = _safe_calmar(base['calmar']) if base else 0

    for fee in [0.0002, 0.0005, 0.001, 0.0015, 0.002, 0.003]:
        p = {**base_params, 'FEE_RATE': fee}
        m = run_backtest(feature_panels, custom_params=p, verbose=False)
        if m is None:
            continue
        flag = '⭐' if abs(fee - 0.0005) < 1e-6 else '  '
        c = _safe_calmar(m['calmar'])
        if c >= base_calmar * 0.6:
            sig = '✅'
        elif c >= 1.0:
            sig = '⚠️'
        else:
            sig = '❌'
        print(f"   {flag} {fee * 10000:>7.1f}  {m['annual'] * 100:>7.2f}%  "
              f"{m['max_dd'] * 100:>7.2f}%  {m['sharpe']:>5.2f}  "
              f"{c:>5.2f}  {sig:>4}")


# ======================================================================
# 主流程
# ======================================================================
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

    # ============================================================
    # 冠军参数 (锁定, 不再调整)
    # ============================================================
    CHAMPION = {
        'MOM_WINDOW': 20 * 6,           # 20 天
        'VOL_WINDOW': 20 * 6,
        'BTC_TREND_WINDOW': 90 * 6,     # 90 天
        'MAX_WEIGHT': 0.30,
        'REBAL_BARS': 6,                # 1 天
        'TOP_K': 2,
        'BOTTOM_K': 2,
        'ENABLE_SHORT': True,
        'SHORT_BUDGET': 0.5,
        'SIGNAL_THRESHOLD': 0.0,
        'USE_VWAP': True,
        'USE_BTC_TREND': True,
    }

    print("\n" + "=" * 72)
    print("🏆 最终冠军策略 · 参数平原全面验证")
    print("=" * 72)
    print(f"冠军参数: {CHAMPION}")

    full_idx = feature_panels['close'].index
    mid_date = full_idx[len(full_idx) // 2]

    # ============================================================
    # 阶段 1: 全样本 + 时段切片基准
    # ============================================================
    print(f"\n\n{'#' * 72}")
    print(f"# 阶段 1: 全样本 + 时段切片基准")
    print(f"{'#' * 72}")

    run_backtest(feature_panels, "【全样本】冠军", custom_params=CHAMPION)
    run_backtest(feature_panels, "【前半段】冠军",
                 custom_params=CHAMPION, trade_end_date=mid_date)
    run_backtest(feature_panels, "【后半段】冠军",
                 custom_params=CHAMPION, trade_start_date=mid_date)

    # ============================================================
    # 阶段 2: 单维参数平原扫描 (6 个参数, 每个 4-6 个取值)
    # ============================================================
    print(f"\n\n{'#' * 72}")
    print(f"# 阶段 2: 单维参数平原扫描")
    print(f"{'#' * 72}")

    # 2.1 MOM_WINDOW (动量回看)
    mom_vals = [d * 6 for d in [10, 15, 20, 25, 30, 40]]
    r = sweep_param(feature_panels, 'MOM_WINDOW', mom_vals, CHAMPION)
    evaluate_plateau(r, 'MOM_WINDOW', CHAMPION['MOM_WINDOW'])

    # 2.2 K (TOP_K = BOTTOM_K 同步)
    r = sweep_param(feature_panels, 'K', [1, 2, 3, 4], CHAMPION)
    evaluate_plateau(r, 'K', CHAMPION['TOP_K'])

    # 2.3 MAX_WEIGHT
    r = sweep_param(feature_panels, 'MAX_WEIGHT',
                    [0.20, 0.25, 0.30, 0.35, 0.40, 0.50], CHAMPION)
    evaluate_plateau(r, 'MAX_WEIGHT', CHAMPION['MAX_WEIGHT'])

    # 2.4 SHORT_BUDGET
    r = sweep_param(feature_panels, 'SHORT_BUDGET',
                    [0.0, 0.25, 0.5, 0.75, 1.0], CHAMPION)
    evaluate_plateau(r, 'SHORT_BUDGET', CHAMPION['SHORT_BUDGET'])

    # 2.5 BTC_TREND_WINDOW
    btc_vals = [d * 6 for d in [60, 90, 120, 150, 200]]
    r = sweep_param(feature_panels, 'BTC_TREND_WINDOW', btc_vals, CHAMPION)
    evaluate_plateau(r, 'BTC_TREND_WINDOW', CHAMPION['BTC_TREND_WINDOW'])

    # 2.6 REBAL_BARS
    r = sweep_param(feature_panels, 'REBAL_BARS', [3, 6, 12, 24], CHAMPION)
    evaluate_plateau(r, 'REBAL_BARS', CHAMPION['REBAL_BARS'])

    # ============================================================
    # 阶段 3: 二维联合平原扫描 (3 对关键参数)
    # ============================================================
    print(f"\n\n{'#' * 72}")
    print(f"# 阶段 3: 二维联合平原扫描")
    print(f"{'#' * 72}")

    grid_2d_scan(feature_panels,
                 'K', [1, 2, 3],
                 'SHORT_BUDGET', [0.0, 0.25, 0.5, 0.75, 1.0], CHAMPION)

    grid_2d_scan(feature_panels,
                 'MOM_WINDOW', [10 * 6, 20 * 6, 30 * 6, 40 * 6],
                 'MAX_WEIGHT', [0.20, 0.30, 0.40, 0.50], CHAMPION)

    grid_2d_scan(feature_panels,
                 'BTC_TREND_WINDOW', [60 * 6, 90 * 6, 120 * 6, 150 * 6],
                 'REBAL_BARS', [3, 6, 12], CHAMPION)

    # ============================================================
    # 阶段 4: 走步前向 (冠军 + 4 个邻居参数, 检查邻居稳定性)
    # ============================================================
    print(f"\n\n{'#' * 72}")
    print(f"# 阶段 4: 走步前向 (冠军 + 邻居参数)")
    print(f"{'#' * 72}")

    walk_forward_analysis(feature_panels, n_windows=6,
                          custom_params=CHAMPION, label="冠军参数 (基准)")

    nb1 = {**CHAMPION, 'MOM_WINDOW': 30 * 6}
    walk_forward_analysis(feature_panels, n_windows=6,
                          custom_params=nb1, label="MOM=30d (vs 冠军 20d)")

    nb2 = {**CHAMPION, 'TOP_K': 3, 'BOTTOM_K': 3}
    walk_forward_analysis(feature_panels, n_windows=6,
                          custom_params=nb2, label="K=3/3 (vs 冠军 2/2)")

    nb3 = {**CHAMPION, 'MAX_WEIGHT': 0.40}
    walk_forward_analysis(feature_panels, n_windows=6,
                          custom_params=nb3, label="MAX_W=0.40 (vs 冠军 0.30)")

    nb4 = {**CHAMPION, 'SHORT_BUDGET': 0.75}
    walk_forward_analysis(feature_panels, n_windows=6,
                          custom_params=nb4, label="SHORT_BUDGET=0.75 (vs 冠军 0.5)")

    # ============================================================
    # 阶段 5: 实盘成本压力测试
    # ============================================================
    print(f"\n\n{'#' * 72}")
    print(f"# 阶段 5: 实盘成本压力测试")
    print(f"{'#' * 72}")

    stress_fees(feature_panels, CHAMPION)

    # ============================================================
    # 最终判定指南
    # ============================================================
    print(f"\n\n{'=' * 72}")
    print(f"🎯 最终判定指南 (回看上面 5 个阶段的输出)")
    print(f"{'=' * 72}")
    print("""
    通过条件 (全部满足 → ✅ 不是过拟合, 可上 paper trading):

      1. 阶段 1: 后半段卡玛 > 1.0           → 时序不退化
      2. 阶段 2: ≥ 5/6 个参数显示 ✅平原     → 单维稳健
      3. 阶段 3: ≥ 2/3 个二维网格平原占比 ≥ 50% → 联合稳健
      4. 阶段 4: 4 个邻居参数中, ≥ 3 个走步前向亏损 ≤ 2/6 → 邻居不崩
      5. 阶段 5: 费率翻倍 (10bp) 后卡玛 > 2 → 成本可承受

    场景判读:

      A — 全部 ✅:
        → 这是回测能给的最强证据, 立即进入 paper trading
        → 同时跑实盘对照 1 个月, 实盘信号必须与回测信号 1:1 对齐

      B — 阶段 2 出现 1-2 个 ❌尖峰:
        → 把那个参数从冠军值改为"次优但平原中间"的值
        → 例如若 MOM=20 是尖峰但 MOM=15/25 都还行, 改 MOM=15 或 25
        → 牺牲一点全样本卡玛, 换更强鲁棒性

      C — 阶段 4 邻居走步前向显著差于冠军:
        → 红色警报, 冠军可能是抽样幸运
        → 退回研究阶段 (考虑改用集成: 同时跑 MOM=15/20/25 三组并平均权重)

      D — 阶段 5 费率 10bp 时卡玛 < 1:
        → 策略对成本敏感, 必须先确认你交易所的真实成本 (taker + 滑点)
        → 若真实成本 ≥ 8bp, 把 REBAL_BARS 改为 12 (2 天再平衡), 重新走全套验证
    """)