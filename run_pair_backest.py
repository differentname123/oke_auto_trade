import pandas as pd
import numpy as np
import os
import time
import numba as nb

# =============================================================================
# 全局配置
# =============================================================================
SINGLE_LEG_FEE = 0.0006
USE_NEXT_BAR_EXEC = True
USE_1MIN_PATH = True
SL_MULT = 3.0
BETA_MIN = 0.1
BETA_MAX = 5.0
WF_MAX_TRAIN_MDD = -15.0


# =============================================================================
# 0. 时间单位工具
# =============================================================================
def get_tf_minutes(tf_str):
    tf = tf_str.lower().strip()
    if tf.endswith('min'):
        return int(tf.replace('min', ''))
    elif tf.endswith('h'):
        return int(tf.replace('h', '')) * 60
    elif tf.endswith('d'):
        return int(tf.replace('d', '')) * 1440
    return 1


def hours_to_bars(hours, tf_minutes):
    return max(1, int(round(hours * 60.0 / tf_minutes)))


def days_to_bars(days, tf_minutes):
    return max(1, int(round(days * 1440.0 / tf_minutes)))


def delta_per_day_to_bar(delta_per_day, tf_minutes):
    return delta_per_day / (1440.0 / tf_minutes)


def _get_segment_bounds_ns(df, tf_minutes):
    if df.empty:
        nat = np.datetime64('NaT')
        return nat, nat
    start = pd.to_datetime(df['open_time'].iloc[0]).to_datetime64()
    end = pd.to_datetime(df['open_time'].iloc[-1]).to_datetime64() + np.timedelta64(int(tf_minutes), 'm')
    return start, end


def _infer_bar_delta_ns(times, tf_minutes=None):
    if tf_minutes is not None and tf_minutes > 0:
        return np.timedelta64(int(tf_minutes), 'm')
    if len(times) >= 2:
        diffs = np.diff(times).astype('timedelta64[ns]').astype(np.int64)
        if len(diffs) > 0:
            median_ns = int(np.median(diffs))
            return np.timedelta64(max(median_ns, 60_000_000_000), 'ns')
    return np.timedelta64(1, 'm')


# =============================================================================
# 0b. 共享辅助函数
# =============================================================================
def _shift_or_copy(arr, shift=True, fill=0.0):
    if not shift:
        return arr.copy()
    out = np.full_like(arr, fill)
    if len(arr) > 1:
        out[1:] = arr[:-1]
    return out


def _get_param_grids():
    return {
        'z_entry': [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0],
        'z_exit': [0.0, 0.5, 1.0],
        'lookback_hours': [10, 15, 30],
        'delta_per_day': [0.01, 0.001, 0.0001],
        've': [1e-3],
        'min_hold_hours': [0.5, 0.75, 1.0, 1.25, 1.5, 3.0, 6.0, 12.0],
    }


def _backtest_and_pnls(full_df, min1_cache, tf_min, fee=None, exec_state=None):
    trades, eq, eq_times, st = compute_equity_and_trades(
        full_df, min1_cache=min1_cache, fee_override=fee,
        tf_minutes=tf_min, exec_state=exec_state)
    eff = _build_effective_pnls(trades, eq, st)
    return trades, eq, eq_times, st, eff


def _unpack_exec_state(exec_state):
    _NAT = np.datetime64('NaT')
    if exec_state is None:
        return dict(entry_main=0.0, entry_sub=0.0, entry_dir=0, entry_beta_val=0.0,
                    cum_realized=0.0, entry_time=_NAT, last_mark_main=0.0,
                    last_mark_sub=0.0, entry_frozen_alpha=0.0, entry_spread_val=0.0,
                    entry_std_val=0.0, last_signal_val=0.0)
    s, n = exec_state, len(exec_state)
    _g = lambda i, d=0.0: (s[i] if i < n else d)
    dir_raw = float(_g(2))
    return dict(
        entry_main=float(_g(0)), entry_sub=float(_g(1)), entry_dir=int(dir_raw),
        entry_beta_val=float(_g(3)), cum_realized=float(_g(4)),
        entry_time=_g(5, _NAT),
        last_mark_main=float(_g(6)), last_mark_sub=float(_g(7)),
        entry_frozen_alpha=float(_g(8)), entry_spread_val=float(_g(9)),
        entry_std_val=float(_g(10)), last_signal_val=float(_g(11, dir_raw)))


# =============================================================================
# 1. 数据重采样
# =============================================================================
def resample_data(df, timeframe='15min'):
    df = df.copy()
    df['open_time'] = pd.to_datetime(df['open_time'])
    df = df.set_index('open_time')
    resampled = df.resample(timeframe).agg({
        'close_main': 'last',
        'close_sub': 'last'
    }).dropna()
    return resampled.reset_index()


# =============================================================================
# 2. 卡尔曼滤波（有状态版本）
# =============================================================================
@nb.njit(cache=True)
def fast_kalman_filter(x_arr, y_arr, delta, ve,
                       init_alpha=0.0, init_beta=0.0,
                       init_P00=1.0, init_P01=0.0,
                       init_P10=0.0, init_P11=1.0):
    n_obs = len(x_arr)
    betas = np.zeros(n_obs)
    alphas = np.zeros(n_obs)
    spreads = np.zeros(n_obs)

    alpha_mean = init_alpha
    beta_mean = init_beta
    P00 = init_P00
    P01 = init_P01
    P10 = init_P10
    P11 = init_P11

    for t in range(n_obs):
        x = x_arr[t]
        y = y_arr[t]
        P00 += delta
        P11 += delta

        y_pred = alpha_mean + beta_mean * x
        error = y - y_pred
        S = P00 + P01 * x + P10 * x + P11 * x * x + ve

        K0 = (P00 + P01 * x) / S
        K1 = (P10 + P11 * x) / S

        alpha_mean += K0 * error
        beta_mean += K1 * error

        new_P00 = P00 - K0 * (P00 + P10 * x)
        new_P01 = P01 - K0 * (P01 + P11 * x)
        new_P10 = P10 - K1 * (P00 + P10 * x)
        new_P11 = P11 - K1 * (P01 + P11 * x)
        P00, P01, P10, P11 = new_P00, new_P01, new_P10, new_P11

        alphas[t] = alpha_mean
        betas[t] = beta_mean
        spreads[t] = error

    return betas, alphas, spreads, alpha_mean, beta_mean, P00, P01, P10, P11


# =============================================================================
# 3. 信号生成
# =============================================================================
@nb.njit(cache=True)
def fast_generate_signals(z_values, spreads, betas, alphas,
                          log_main, log_sub, roll_std,
                          lookback, z_entry, z_exit,
                          min_hold_bars, cooldown_bars, max_hold_bars,
                          sl_mult, single_leg_fee,
                          beta_min=0.1, beta_max=5.0,
                          init_pos=0, init_held=0, init_since_close=9999,
                          init_frozen_beta=0.0, init_frozen_alpha=0.0,
                          init_entry_spread=0.0, init_entry_std=0.0):
    n = len(z_values)
    signals = np.zeros(n)
    frozen_betas_arr = np.zeros(n)
    frozen_alphas_arr = np.zeros(n)
    entry_spreads_arr = np.zeros(n)
    entry_stds_arr = np.zeros(n)
    stop_flags = np.zeros(n)

    pos = init_pos
    held = init_held
    since_close = init_since_close
    frozen_beta = init_frozen_beta
    frozen_alpha = init_frozen_alpha
    entry_spread = init_entry_spread
    entry_std = init_entry_std

    start_idx = 0 if pos != 0 else lookback

    for i in range(start_idx, n):
        z = z_values[i]

        if pos == 0:
            since_close += 1
            if since_close >= cooldown_bars:
                cur_std = roll_std[i]
                cur_beta = betas[i]
                if cur_std > 0.0 and cur_beta >= beta_min and cur_beta <= beta_max:
                    expected_profit = (z_entry - abs(z_exit)) * cur_std
                    round_trip_cost = 2.0 * single_leg_fee * (1.0 + abs(cur_beta))
                    if expected_profit > round_trip_cost:
                        if z > z_entry:
                            pos = 1
                            held = 0
                            frozen_beta = cur_beta
                            frozen_alpha = alphas[i]
                            entry_spread = log_main[i] - frozen_alpha - frozen_beta * log_sub[i]
                            entry_std = cur_std
                        elif z < -z_entry:
                            pos = -1
                            held = 0
                            frozen_beta = cur_beta
                            frozen_alpha = alphas[i]
                            entry_spread = log_main[i] - frozen_alpha - frozen_beta * log_sub[i]
                            entry_std = cur_std
        else:
            held += 1
            frozen_spread = log_main[i] - frozen_alpha - frozen_beta * log_sub[i]

            stopped = False
            if entry_std > 0.0 and sl_mult > 0.0:
                if pos == 1:
                    if (frozen_spread - entry_spread) > sl_mult * entry_std:
                        stopped = True
                elif pos == -1:
                    if (entry_spread - frozen_spread) > sl_mult * entry_std:
                        stopped = True

            if stopped:
                pos = 0
                since_close = 0
                stop_flags[i] = 1.0
            elif max_hold_bars > 0 and held >= max_hold_bars:
                pos = 0
                since_close = 0
            elif held >= min_hold_bars:
                if pos == 1:
                    spread_improved = frozen_spread < entry_spread
                else:
                    spread_improved = frozen_spread > entry_spread

                if max_hold_bars > 0 and held > max_hold_bars // 2:
                    spread_improved = True

                if pos == 1 and z < z_exit and spread_improved:
                    pos = 0
                    since_close = 0
                elif pos == -1 and z > -z_exit and spread_improved:
                    pos = 0
                    since_close = 0

        signals[i] = pos
        if pos != 0:
            frozen_betas_arr[i] = frozen_beta
            frozen_alphas_arr[i] = frozen_alpha
            entry_spreads_arr[i] = entry_spread
            entry_stds_arr[i] = entry_std
        else:
            frozen_betas_arr[i] = betas[i]
            frozen_alphas_arr[i] = alphas[i]
            entry_spreads_arr[i] = 0.0
            entry_stds_arr[i] = 0.0

    return (signals, frozen_betas_arr, frozen_alphas_arr,
            entry_spreads_arr, entry_stds_arr,
            stop_flags,
            pos, held, since_close,
            frozen_beta, frozen_alpha, entry_spread, entry_std)


# =============================================================================
# 4. 信号管道（KF + 信号生成）
# =============================================================================
def generate_pair_trading_signals(merged_df, main_col='close_main', sub_col='close_sub',
                                  z_lookback=60, z_entry=2.0, z_exit=0.5,
                                  delta=1e-5, ve=1e-3,
                                  min_hold_bars=12, cooldown_bars=4, max_hold_bars=0,
                                  kf_state=None, pos_state=None):
    df = merged_df.copy()
    df['log_main'] = np.log(df[main_col])
    df['log_sub'] = np.log(df[sub_col])

    if kf_state is None:
        kf_state = (0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    betas, alphas, kf_spreads, ka, kb, kP00, kP01, kP10, kP11 = fast_kalman_filter(
        df['log_sub'].values, df['log_main'].values,
        delta=delta, ve=ve,
        init_alpha=kf_state[0], init_beta=kf_state[1],
        init_P00=kf_state[2], init_P01=kf_state[3],
        init_P10=kf_state[4], init_P11=kf_state[5]
    )
    final_kf_state = (ka, kb, kP00, kP01, kP10, kP11)

    df['beta'] = betas
    df['alpha'] = alphas
    df['spread'] = kf_spreads

    spread_s = pd.Series(kf_spreads)
    roll_mean = spread_s.rolling(z_lookback, min_periods=z_lookback).mean()
    roll_std = spread_s.rolling(z_lookback, min_periods=z_lookback).std()
    z_score = ((spread_s - roll_mean) / roll_std).values
    df['z_score'] = z_score
    df['roll_std'] = roll_std.values

    z_clean = np.nan_to_num(z_score, nan=0.0, posinf=0.0, neginf=0.0)
    rs_clean = np.nan_to_num(roll_std.values, nan=0.0, posinf=0.0, neginf=0.0)

    if pos_state is None:
        pos_state = (0, 0, 9999, 0.0, 0.0, 0.0, 0.0)

    (raw_signals, frozen_betas, frozen_alphas_state,
     entry_spreads_state, entry_stds_state,
     stop_flags_raw,
     fp, fh, fsc, ffb, ffa, fes, fest) = fast_generate_signals(
        z_clean, kf_spreads, betas, alphas,
        df['log_main'].values, df['log_sub'].values, rs_clean,
        z_lookback, z_entry, z_exit,
        min_hold_bars, cooldown_bars, max_hold_bars,
        SL_MULT, SINGLE_LEG_FEE,
        BETA_MIN, BETA_MAX,
        init_pos=pos_state[0], init_held=pos_state[1],
        init_since_close=pos_state[2],
        init_frozen_beta=pos_state[3], init_frozen_alpha=pos_state[4],
        init_entry_spread=pos_state[5], init_entry_std=pos_state[6]
    )
    final_pos_state = (fp, fh, fsc, ffb, ffa, fes, fest)

    do_shift = USE_NEXT_BAR_EXEC
    df['signal'] = _shift_or_copy(raw_signals, do_shift)
    df['signal_frozen_beta'] = _shift_or_copy(frozen_betas, do_shift)
    df['signal_frozen_alpha'] = _shift_or_copy(frozen_alphas_state, do_shift)
    df['signal_entry_spread'] = _shift_or_copy(entry_spreads_state, do_shift)
    df['signal_entry_std'] = _shift_or_copy(entry_stds_state, do_shift)
    df['signal_entry_z'] = _shift_or_copy(z_score, do_shift, fill=np.nan)
    df['stop_flag'] = _shift_or_copy(stop_flags_raw, do_shift)

    _sig_ctx = {
        'z_entry': float(z_entry), 'z_exit': float(z_exit),
        'min_hold_bars': int(min_hold_bars), 'cooldown_bars': int(cooldown_bars),
        'max_hold_bars': int(max_hold_bars), 'signal_fee': float(SINGLE_LEG_FEE),
    }
    df.attrs['_pt_signal_ctx'] = _sig_ctx
    for _k, _v in _sig_ctx.items():
        df[f'__pt_{_k}'] = _v

    return df, final_kf_state, final_pos_state


# =============================================================================
# 5. 1min cache / 有效PnL / 时间三段
# =============================================================================
def _build_1min_cache(original_1min_df):
    odf = original_1min_df.copy()
    odf['open_time'] = pd.to_datetime(odf['open_time'])
    odf = odf.sort_values('open_time').reset_index(drop=True)
    cm = odf['close_main'].values.astype(float)
    cs = odf['close_sub'].values.astype(float)
    return {
        'times': odf['open_time'].values.astype('datetime64[ns]'),
        'close_main': cm, 'close_sub': cs,
        'log_main': np.log(cm), 'log_sub': np.log(cs),
    }


def _lookup_1min_point(min1_cache, target_time_ns):
    min1_times = min1_cache['times']
    idx = np.searchsorted(min1_times, target_time_ns, side='left')
    if idx >= len(min1_times):
        idx = len(min1_times) - 1
    return (min1_times[idx],
            min1_cache['close_main'][idx], min1_cache['close_sub'][idx],
            min1_cache['log_main'][idx], min1_cache['log_sub'][idx])


def _build_effective_pnls(trade_df, equity_curve, final_exec_state):
    if trade_df is not None and not trade_df.empty:
        pnls = trade_df['net_pnl'].values.astype(float).copy()
    else:
        pnls = np.array([], dtype=float)

    has_open = (final_exec_state is not None and len(final_exec_state) > 2
                and int(final_exec_state[2]) != 0)
    if has_open and equity_curve is not None and len(equity_curve) > 0:
        closed_sum = float(np.sum(pnls)) if len(pnls) > 0 else 0.0
        pnls = np.append(pnls, float(equity_curve[-1]) - closed_sum)
    return pnls


def _split_train_thirds(segment_start_time, segment_end_time, equity_times, equity_curve):
    if equity_curve is None or len(equity_curve) == 0:
        return [0.0, 0.0, 0.0]

    eq = np.asarray(equity_curve, dtype=float)
    etimes = np.asarray(equity_times).astype('datetime64[ns]')
    start = np.datetime64(segment_start_time, 'ns')
    end = np.datetime64(segment_end_time, 'ns')

    if str(start) == 'NaT' or str(end) == 'NaT':
        return [0.0, 0.0, 0.0]

    total_ns = int((end - start).astype('timedelta64[ns]').astype(np.int64))
    if total_ns <= 0:
        return [float(eq[-1]), 0.0, 0.0]

    b1 = start + np.timedelta64(total_ns // 3, 'ns')
    b2 = start + np.timedelta64((total_ns * 2) // 3, 'ns')

    def _eq_at(boundary_ns):
        idx = np.searchsorted(etimes, boundary_ns, side='right') - 1
        idx = max(0, min(idx, len(eq) - 1))
        return float(eq[idx])

    e1 = _eq_at(b1)
    e2 = _eq_at(b2)
    e3 = _eq_at(end)
    return [e1, e2 - e1, e3 - e2]


# =============================================================================
# 6. 逐 bar / 逐1min 权益曲线 + 交易提取
# =============================================================================
def compute_equity_and_trades(full_df, original_1min_df=None, fee_override=None,
                              exec_state=None, min1_cache=None, tf_minutes=None):
    fee = SINGLE_LEG_FEE if fee_override is None else fee_override

    df = full_df.reset_index(drop=True)
    if 'open_time' not in df.columns:
        if 'index' in df.columns:
            df.rename(columns={'index': 'open_time'}, inplace=True)
        elif 'timestamp' in df.columns:
            df.rename(columns={'timestamp': 'open_time'}, inplace=True)

    n = len(df)
    es = _unpack_exec_state(exec_state)

    if n == 0:
        final_exec_state = (
            es['entry_main'], es['entry_sub'], float(es['entry_dir']),
            es['entry_beta_val'], es['cum_realized'],
            es['entry_time'], es['last_mark_main'], es['last_mark_sub'],
            es['entry_frozen_alpha'], es['entry_spread_val'], es['entry_std_val'],
            es['last_signal_val']
        )
        return pd.DataFrame(), np.array([]), np.array([], dtype='datetime64[ns]'), final_exec_state

    signals = df['signal'].values.astype(int)
    times = pd.to_datetime(df['open_time']).values.astype('datetime64[ns]')
    close_main = df['close_main'].values.astype(float)
    close_sub = df['close_sub'].values.astype(float)

    def _col(col_name, default, dtype=float):
        if col_name in df.columns:
            return df[col_name].values.astype(dtype)
        return default() if callable(default) else default

    z_scores = _col('z_score', lambda: np.full(n, np.nan))
    spreads_arr = _col('spread', lambda: np.full(n, np.nan))
    betas_arr = _col('beta', lambda: np.zeros(n))
    alphas_arr = _col('alpha', lambda: np.zeros(n))
    roll_std_arr = _col('roll_std', lambda: np.zeros(n))
    signal_fb = _col('signal_frozen_beta', lambda: betas_arr.copy())
    signal_fa = _col('signal_frozen_alpha', lambda: alphas_arr.copy())
    signal_es = _col('signal_entry_spread', lambda: np.zeros(n))
    signal_estd = _col('signal_entry_std', lambda: np.zeros(n))
    signal_entry_z = _col('signal_entry_z', lambda: np.full(n, np.nan))
    stop_flags_col = _col('stop_flag', lambda: np.zeros(n, dtype=int), dtype=int)
    log_main_arr = _col('log_main', lambda: np.log(close_main))
    log_sub_arr = _col('log_sub', lambda: np.log(close_sub))

    z_clean = np.nan_to_num(z_scores, nan=0.0, posinf=0.0, neginf=0.0)
    rs_clean = np.nan_to_num(roll_std_arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _read_sig_ctx(key, default):
        if '_pt_signal_ctx' in getattr(df, 'attrs', {}):
            if key in df.attrs['_pt_signal_ctx']:
                return df.attrs['_pt_signal_ctx'][key]
        col = f'__pt_{key}'
        if col in df.columns and len(df) > 0:
            return df[col].iloc[0]
        return default

    sig_z_entry = float(_read_sig_ctx('z_entry', np.nan))
    sig_z_exit = float(_read_sig_ctx('z_exit', np.nan))
    sig_min_hold_bars = int(_read_sig_ctx('min_hold_bars', 0))
    sig_cooldown_bars = int(_read_sig_ctx('cooldown_bars', 0))
    sig_max_hold_bars = int(_read_sig_ctx('max_hold_bars', 0))
    sig_fee_assumption = float(_read_sig_ctx('signal_fee', SINGLE_LEG_FEE))

    can_rebuild_signal_path = (
        np.isfinite(sig_z_entry) and np.isfinite(sig_z_exit)
        and len(z_clean) == n and len(rs_clean) == n
        and len(spreads_arr) == n and len(betas_arr) == n and len(alphas_arr) == n
        and len(log_main_arr) == n and len(log_sub_arr) == n
    )

    need_min1 = (USE_NEXT_BAR_EXEC or USE_1MIN_PATH)
    if min1_cache is None and need_min1 and original_1min_df is not None:
        min1_cache = _build_1min_cache(original_1min_df)

    use_1m_exec = USE_NEXT_BAR_EXEC and (min1_cache is not None)
    use_1m_path = USE_1MIN_PATH and (min1_cache is not None)

    if use_1m_path:
        min1_times = min1_cache['times']
        min1_cm = min1_cache['close_main']
        min1_cs = min1_cache['close_sub']
        min1_lm = min1_cache['log_main']
        min1_ls = min1_cache['log_sub']

    bar_delta_ns = _infer_bar_delta_ns(times, tf_minutes=tf_minutes)
    segment_end_time = times[-1] + bar_delta_ns

    entry_main = es['entry_main']
    entry_sub = es['entry_sub']
    entry_dir = es['entry_dir']
    entry_beta_val = es['entry_beta_val']
    cum_realized = es['cum_realized']
    entry_time = es['entry_time']
    prev_last_mark_main = es['last_mark_main']
    prev_last_mark_sub = es['last_mark_sub']
    entry_frozen_alpha = es['entry_frozen_alpha']
    entry_spread_val = es['entry_spread_val']
    entry_std_val = es['entry_std_val']
    prev_sig_seed = int(es['last_signal_val'])
    in_trade = (entry_dir != 0)

    if in_trade:
        seg_entry_main = prev_last_mark_main if prev_last_mark_main > 0 else entry_main
        seg_entry_sub = prev_last_mark_sub if prev_last_mark_sub > 0 else entry_sub
        seg_open_time = times[0]
        seg_entry_fee = 0.0
        seg_inherited = True
        actual_open_time = entry_time
        seg_entry_z = np.nan
        seg_entry_spread_meta = np.nan
        seg_entry_alpha_meta = np.nan
    else:
        seg_entry_main = 0.0
        seg_entry_sub = 0.0
        seg_open_time = np.datetime64('NaT')
        seg_entry_fee = 0.0
        seg_inherited = False
        actual_open_time = np.datetime64('NaT')
        seg_entry_z = np.nan
        seg_entry_spread_meta = np.nan
        seg_entry_alpha_meta = np.nan

    trades = []
    equity_values = []
    equity_times = []

    last_mark_main = close_main[0]
    last_mark_sub = close_sub[0]

    def _append_equity_point(point_time, point_value):
        if len(equity_times) > 0 and equity_times[-1] == point_time:
            equity_values[-1] = point_value
        else:
            equity_times.append(point_time)
            equity_values.append(point_value)

    def _calc_gross(em, ess, cm, cs, ebeta, edir):
        br = (cm - em) / em if em != 0 else 0.0
        er = (cs - ess) / ess if ess != 0 else 0.0
        if edir == -1:
            return br - ebeta * er
        else:
            return ebeta * er - br

    def _calc_norm(em, ess, cm, cs, ebeta, edir):
        total_exp = 1.0 + abs(ebeta)
        if total_exp <= 0.0:
            return 0.0
        return _calc_gross(em, ess, cm, cs, ebeta, edir) / total_exp

    def _open_position(sig_dir, exec_time_val, exec_m, exec_s, bar_idx):
        nonlocal entry_main, entry_sub, entry_dir, entry_beta_val, cum_realized
        nonlocal entry_time, in_trade, entry_frozen_alpha, entry_spread_val, entry_std_val
        nonlocal seg_entry_main, seg_entry_sub, seg_open_time, seg_entry_fee, seg_inherited
        nonlocal actual_open_time, seg_entry_z, seg_entry_spread_meta, seg_entry_alpha_meta

        entry_main = exec_m
        entry_sub = exec_s
        entry_dir = int(sig_dir)
        entry_beta_val = signal_fb[bar_idx]
        entry_frozen_alpha = signal_fa[bar_idx]
        entry_spread_val = signal_es[bar_idx]
        entry_std_val = signal_estd[bar_idx]
        entry_time = exec_time_val
        in_trade = True
        cum_realized -= fee

        seg_entry_main = exec_m
        seg_entry_sub = exec_s
        seg_open_time = exec_time_val
        seg_entry_fee = fee
        seg_inherited = False
        actual_open_time = exec_time_val
        seg_entry_z = signal_entry_z[bar_idx] if bar_idx < n else np.nan
        seg_entry_spread_meta = signal_es[bar_idx] if bar_idx < n else np.nan
        seg_entry_alpha_meta = signal_fa[bar_idx] if bar_idx < n else np.nan

    def _close_position(exec_time_val, exec_m, exec_s, bar_idx, close_reason):
        nonlocal entry_main, entry_sub, entry_dir, entry_beta_val, cum_realized
        nonlocal entry_time, in_trade, entry_frozen_alpha, entry_spread_val, entry_std_val
        nonlocal seg_entry_main, seg_entry_sub, seg_open_time, seg_entry_fee, seg_inherited
        nonlocal actual_open_time, seg_entry_z, seg_entry_spread_meta, seg_entry_alpha_meta

        realized_add = _calc_norm(entry_main, entry_sub, exec_m, exec_s,
                                  entry_beta_val, entry_dir) - fee
        cum_realized += realized_add

        seg_net = (_calc_norm(seg_entry_main, seg_entry_sub, exec_m, exec_s,
                              entry_beta_val, entry_dir)
                   - fee - seg_entry_fee)

        direction_str = (f'做多价差 (Long BTC / Short {entry_beta_val:.3f} ETH)'
                         if entry_dir == -1
                         else f'做空价差 (Short BTC / Long {entry_beta_val:.3f} ETH)')

        if close_reason == 'hard_stop_1m':
            exit_z, exit_spread = np.nan, np.nan
            exit_alpha, exit_beta = entry_frozen_alpha, entry_beta_val
        else:
            exit_z = z_scores[bar_idx] if bar_idx < n else np.nan
            exit_spread = spreads_arr[bar_idx] if bar_idx < n else np.nan
            exit_alpha = alphas_arr[bar_idx] if bar_idx < n else np.nan
            exit_beta = betas_arr[bar_idx] if bar_idx < n else entry_beta_val

        duration_mins = 0.0
        if not pd.isna(seg_open_time):
            duration_mins = (pd.Timestamp(exec_time_val) - pd.Timestamp(seg_open_time)).total_seconds() / 60.0

        trades.append({
            'open_time': seg_open_time,
            'actual_open_time': actual_open_time,
            'close_time': exec_time_val,
            'duration_mins': duration_mins,
            'direction': direction_str,
            'btc_roi': f"{((exec_m - seg_entry_main) / seg_entry_main) * 100:.2f}" if seg_entry_main else "0",
            'eth_roi': f"{((exec_s - seg_entry_sub) / seg_entry_sub) * 100:.2f}" if seg_entry_sub else "0",
            'net_pnl': round(seg_net * 100, 4),
            'open_main': seg_entry_main, 'close_main': exec_m,
            'open_sub': seg_entry_sub, 'close_sub': exec_s,
            'entry_z': seg_entry_z if not seg_inherited else np.nan,
            'exit_z': exit_z,
            'entry_spread': seg_entry_spread_meta if not seg_inherited else np.nan,
            'exit_spread': exit_spread,
            'entry_beta': entry_beta_val, 'exit_beta': exit_beta,
            'entry_alpha': seg_entry_alpha_meta if not seg_inherited else np.nan,
            'exit_alpha': exit_alpha,
            'signal': entry_dir,
            'inherited_from_prev_window': seg_inherited,
            'close_reason': close_reason,
        })

        in_trade = False
        entry_main, entry_sub, entry_dir = 0.0, 0.0, 0
        entry_beta_val = 0.0
        entry_time = np.datetime64('NaT')
        entry_frozen_alpha = 0.0
        entry_spread_val = 0.0
        entry_std_val = 0.0
        seg_entry_main = 0.0
        seg_entry_sub = 0.0
        seg_open_time = np.datetime64('NaT')
        seg_entry_fee = 0.0
        seg_inherited = False
        actual_open_time = np.datetime64('NaT')
        seg_entry_z = np.nan
        seg_entry_spread_meta = np.nan
        seg_entry_alpha_meta = np.nan

    def _rebuild_future_signals_after_intrabar_stop(stop_bar_idx):
        if stop_bar_idx < 0 or stop_bar_idx >= n:
            return

        signals[stop_bar_idx] = 0
        signal_es[stop_bar_idx] = 0.0
        signal_estd[stop_bar_idx] = 0
        stop_flags_col[stop_bar_idx] = 0

        if not can_rebuild_signal_path:
            return

        suffix = stop_bar_idx + 1
        if suffix >= n:
            return

        (new_raw_sig, new_fb, new_fa,
         new_es, new_estd,
         new_stop_raw,
         _, _, _, _, _, _, _) = fast_generate_signals(
            z_clean[suffix:], spreads_arr[suffix:], betas_arr[suffix:], alphas_arr[suffix:],
            log_main_arr[suffix:], log_sub_arr[suffix:], rs_clean[suffix:],
            0, sig_z_entry, sig_z_exit,
            sig_min_hold_bars, sig_cooldown_bars, sig_max_hold_bars,
            SL_MULT, sig_fee_assumption,
            BETA_MIN, BETA_MAX,
            init_pos=0, init_held=0, init_since_close=0,
            init_frozen_beta=0.0, init_frozen_alpha=0.0,
            init_entry_spread=0.0, init_entry_std=0.0
        )

        signals[suffix:] = 0
        signal_fb[suffix:] = betas_arr[suffix:]
        signal_fa[suffix:] = alphas_arr[suffix:]
        signal_es[suffix:] = 0.0
        signal_estd[suffix:] = 0.0
        signal_entry_z[suffix:] = np.nan
        stop_flags_col[suffix:] = 0

        if USE_NEXT_BAR_EXEC:
            signal_fb[suffix] = betas_arr[stop_bar_idx]
            signal_fa[suffix] = alphas_arr[stop_bar_idx]
            signal_entry_z[suffix] = z_scores[stop_bar_idx] if stop_bar_idx < n else np.nan

            assign_start = suffix + 1
            if assign_start < n and len(new_raw_sig) > 0:
                signals[assign_start:] = new_raw_sig[:-1]
                signal_fb[assign_start:] = new_fb[:-1]
                signal_fa[assign_start:] = new_fa[:-1]
                signal_es[assign_start:] = new_es[:-1]
                signal_estd[assign_start:] = new_estd[:-1]
                signal_entry_z[assign_start:] = z_scores[suffix:-1]
                stop_flags_col[assign_start:] = new_stop_raw[:-1].astype(int)
        else:
            signals[suffix:] = new_raw_sig
            signal_fb[suffix:] = new_fb
            signal_fa[suffix:] = new_fa
            signal_es[suffix:] = new_es
            signal_estd[suffix:] = new_estd
            signal_entry_z[suffix:] = z_scores[suffix:]
            stop_flags_col[suffix:] = new_stop_raw.astype(int)

    def _check_stop_and_maybe_close(cur_time, cur_m, cur_s, cur_lm, cur_ls, bar_idx):
        if (not in_trade) or entry_dir == 0:
            return False
        if entry_std_val <= 0.0 or SL_MULT <= 0.0:
            return False

        frozen_spread = cur_lm - entry_frozen_alpha - entry_beta_val * cur_ls
        stopped = False
        if entry_dir == 1:
            if (frozen_spread - entry_spread_val) > SL_MULT * entry_std_val:
                stopped = True
        elif entry_dir == -1:
            if (entry_spread_val - frozen_spread) > SL_MULT * entry_std_val:
                stopped = True

        if stopped:
            _close_position(cur_time, cur_m, cur_s, bar_idx, close_reason='hard_stop_1m')
            _rebuild_future_signals_after_intrabar_stop(bar_idx)
            return True
        return False

    def _get_exec_point(bar_idx):
        if use_1m_exec:
            return _lookup_1min_point(min1_cache, times[bar_idx])
        return (times[bar_idx], close_main[bar_idx], close_sub[bar_idx],
                np.log(close_main[bar_idx]), np.log(close_sub[bar_idx]))

    if use_1m_path:
        for i in range(n):
            sig = int(signals[i])
            prev_sig = int(signals[i - 1]) if i > 0 else int(prev_sig_seed)

            start_ns = times[i]
            end_ns = times[i + 1] if i < n - 1 else (times[i] + bar_delta_ns)

            l = np.searchsorted(min1_times, start_ns, side='left')
            r = np.searchsorted(min1_times, end_ns, side='left')

            if r <= l:
                exec_time_val, exec_m, exec_s, _, _ = _get_exec_point(i)

                if use_1m_exec:
                    if sig != 0 and prev_sig == 0 and not in_trade:
                        _open_position(sig, exec_time_val, exec_m, exec_s, i)
                    elif in_trade and sig == 0 and prev_sig != 0:
                        _cr = 'hard_stop' if stop_flags_col[i] == 1 else 'signal'
                        _close_position(exec_time_val, exec_m, exec_s, i, close_reason=_cr)
                else:
                    if in_trade and sig == 0 and prev_sig != 0:
                        _cr = 'hard_stop' if stop_flags_col[i] == 1 else 'signal'
                        _close_position(exec_time_val, exec_m, exec_s, i, close_reason=_cr)
                    elif sig != 0 and prev_sig == 0 and not in_trade:
                        _open_position(sig, exec_time_val, exec_m, exec_s, i)

                if in_trade:
                    unrealized = _calc_norm(entry_main, entry_sub, close_main[i], close_sub[i],
                                            entry_beta_val, entry_dir)
                    _append_equity_point(times[i] + bar_delta_ns, (cum_realized + unrealized) * 100.0)
                else:
                    _append_equity_point(times[i] + bar_delta_ns, cum_realized * 100.0)

                last_mark_main = close_main[i]
                last_mark_sub = close_sub[i]
                continue

            if use_1m_exec:
                start_t = min1_times[l]
                start_m = min1_cm[l]
                start_s = min1_cs[l]

                if sig != 0 and prev_sig == 0 and not in_trade:
                    _open_position(sig, start_t, start_m, start_s, i)
                elif in_trade and sig == 0 and prev_sig != 0:
                    _cr = 'hard_stop' if stop_flags_col[i] == 1 else 'signal'
                    _close_position(start_t, start_m, start_s, i, close_reason=_cr)
                    _append_equity_point(start_t, cum_realized * 100.0)
                    last_mark_main = min1_cm[r - 1]
                    last_mark_sub = min1_cs[r - 1]
                    continue

                if in_trade:
                    for j in range(l, r):
                        cur_t = min1_times[j]
                        cur_m = min1_cm[j]
                        cur_s = min1_cs[j]
                        cur_lm = min1_lm[j]
                        cur_ls = min1_ls[j]

                        if _check_stop_and_maybe_close(cur_t, cur_m, cur_s, cur_lm, cur_ls, i):
                            _append_equity_point(cur_t, cum_realized * 100.0)
                            break
                        else:
                            unrealized = _calc_norm(entry_main, entry_sub, cur_m, cur_s,
                                                    entry_beta_val, entry_dir)
                            _append_equity_point(cur_t, (cum_realized + unrealized) * 100.0)

            else:
                event_idx = r - 1
                stopped = False

                if in_trade:
                    for j in range(l, event_idx):
                        cur_t = min1_times[j]
                        cur_m = min1_cm[j]
                        cur_s = min1_cs[j]
                        cur_lm = min1_lm[j]
                        cur_ls = min1_ls[j]

                        if _check_stop_and_maybe_close(cur_t, cur_m, cur_s, cur_lm, cur_ls, i):
                            _append_equity_point(cur_t, cum_realized * 100.0)
                            stopped = True
                            break
                        else:
                            unrealized = _calc_norm(entry_main, entry_sub, cur_m, cur_s,
                                                    entry_beta_val, entry_dir)
                            _append_equity_point(cur_t, (cum_realized + unrealized) * 100.0)

                if not stopped:
                    cur_t = min1_times[event_idx]
                    cur_m = min1_cm[event_idx]
                    cur_s = min1_cs[event_idx]
                    cur_lm = min1_lm[event_idx]
                    cur_ls = min1_ls[event_idx]

                    if in_trade and sig == 0 and prev_sig != 0:
                        _cr = 'hard_stop' if stop_flags_col[i] == 1 else 'signal'
                        _close_position(cur_t, cur_m, cur_s, i, close_reason=_cr)
                        _append_equity_point(cur_t, cum_realized * 100.0)
                    elif sig != 0 and prev_sig == 0 and not in_trade:
                        _open_position(sig, cur_t, cur_m, cur_s, i)
                        _append_equity_point(cur_t, cum_realized * 100.0)
                    elif in_trade:
                        if _check_stop_and_maybe_close(cur_t, cur_m, cur_s, cur_lm, cur_ls, i):
                            _append_equity_point(cur_t, cum_realized * 100.0)
                        else:
                            unrealized = _calc_norm(entry_main, entry_sub, cur_m, cur_s,
                                                    entry_beta_val, entry_dir)
                            _append_equity_point(cur_t, (cum_realized + unrealized) * 100.0)

            last_mark_main = min1_cm[r - 1]
            last_mark_sub = min1_cs[r - 1]

        if len(equity_values) == 0:
            _append_equity_point(segment_end_time, cum_realized * 100.0)

    else:
        for i in range(n):
            sig = int(signals[i])
            prev_sig = int(signals[i - 1]) if i > 0 else int(prev_sig_seed)

            exec_time_val, exec_m, exec_s, _, _ = _get_exec_point(i)

            if sig != 0 and prev_sig == 0 and not in_trade:
                _open_position(sig, exec_time_val, exec_m, exec_s, i)
            elif in_trade and sig == 0 and prev_sig != 0:
                _cr = 'hard_stop' if stop_flags_col[i] == 1 else 'signal'
                _close_position(exec_time_val, exec_m, exec_s, i, close_reason=_cr)

            if in_trade and entry_dir != 0:
                unrealized = _calc_norm(entry_main, entry_sub, close_main[i], close_sub[i],
                                        entry_beta_val, entry_dir)
                eq_val = (cum_realized + unrealized) * 100.0
            else:
                eq_val = cum_realized * 100.0

            _append_equity_point(times[i] + bar_delta_ns, eq_val)
            last_mark_main = close_main[i]
            last_mark_sub = close_sub[i]

    equity = np.asarray(equity_values, dtype=float)
    equity_times_arr = np.asarray(equity_times, dtype='datetime64[ns]')

    last_signal_val = float(signals[-1]) if n > 0 else float(prev_sig_seed)
    final_exec_state = (
        entry_main, entry_sub, float(entry_dir), entry_beta_val, cum_realized,
        entry_time, last_mark_main, last_mark_sub,
        entry_frozen_alpha, entry_spread_val, entry_std_val,
        last_signal_val
    )

    trade_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return trade_df, equity, equity_times_arr, final_exec_state


# =============================================================================
# 7. 统计函数
# =============================================================================
def print_backtest_stats(trade_df, equity_curve=None, effective_pnls=None):
    has_trade_rows = trade_df is not None and not trade_df.empty

    if effective_pnls is not None:
        pnl_arr = np.asarray(effective_pnls, dtype=float)
    elif has_trade_rows:
        pnl_arr = trade_df['net_pnl'].values.astype(float)
    else:
        pnl_arr = np.array([], dtype=float)

    total_trades = len(pnl_arr)
    if total_trades > 0:
        total_return_trade = float(np.sum(pnl_arr))
        avg_return = float(np.mean(pnl_arr))
        win_rate = float(np.sum(pnl_arr > 0)) / total_trades
        gross_profit = float(np.sum(pnl_arr[pnl_arr > 0]))
        gross_loss = float(abs(np.sum(pnl_arr[pnl_arr <= 0])))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)
    else:
        total_return_trade, avg_return, win_rate, profit_factor = 0.0, 0.0, 0.0, 0.0

    avg_duration = float(trade_df['duration_mins'].mean()) if has_trade_rows else 0.0
    max_duration = float(trade_df['duration_mins'].max()) if has_trade_rows else 0.0

    if equity_curve is not None and len(equity_curve) > 0:
        eq = np.asarray(equity_curve, dtype=float)
        total_return = float(eq[-1])
        eq_ext = np.concatenate(([0.0], eq))
    else:
        total_return = total_return_trade
        eq_ext = np.concatenate(([0.0], np.cumsum(pnl_arr))) if len(pnl_arr) > 0 else np.array([0.0])

    peak = np.maximum.accumulate(eq_ext)
    max_drawdown = float(np.min(eq_ext - peak))

    return {
        'Total Return': total_return,
        'total_trades': total_trades,
        'Win Rate': win_rate,
        'avg_profit_per_trade': avg_return,
        'Max Drawdown': max_drawdown,
        'Profit Factor': profit_factor,
        'avg_duration': avg_duration,
        '最长持仓时间': max_duration,
    }


# =============================================================================
# 8. 参数灵敏度扫描
# =============================================================================
def parameter_sensitivity_scan(df_file, timeframe='15min'):
    key_word = os.path.basename(df_file).replace('.csv', '')

    original_df = pd.read_csv(df_file)
    original_df['open_time'] = pd.to_datetime(original_df['open_time'])

    if timeframe != '1min':
        df = resample_data(original_df, timeframe)
        print(f"[{key_word}] 重采样: {len(original_df)} → {len(df)} 行 ({timeframe})")
    else:
        df = original_df.copy()
        print(f"[{key_word}] 原始1min: {len(df)} 行")

    tf_min = get_tf_minutes(timeframe)
    min1_cache = _build_1min_cache(original_df) if (USE_NEXT_BAR_EXEC or USE_1MIN_PATH) else None

    grids = _get_param_grids()
    cooldown_hours = 1.0
    max_hold_days = 4.0
    cooldown_bars = hours_to_bars(cooldown_hours, tf_min)
    max_hold_bars = days_to_bars(max_hold_days, tf_min)

    total_combos = (len(grids['z_entry']) * len(grids['z_exit']) * len(grids['lookback_hours'])
                    * len(grids['delta_per_day']) * len(grids['ve']) * len(grids['min_hold_hours']))
    print(f"总参数组合: {total_combos} | cooldown={cooldown_bars}bars, max_hold={max_hold_bars}bars")

    results = []
    count = 0
    start_time = time.time()

    for z_entry in grids['z_entry']:
        for z_exit in grids['z_exit']:
            if z_exit >= z_entry:
                continue
            for lb_h in grids['lookback_hours']:
                z_lb = hours_to_bars(lb_h, tf_min)
                for dpd in grids['delta_per_day']:
                    delta = delta_per_day_to_bar(dpd, tf_min)
                    for ve in grids['ve']:
                        for mh_h in grids['min_hold_hours']:
                            min_hold = hours_to_bars(mh_h, tf_min)
                            count += 1
                            try:
                                full_df, _, _ = generate_pair_trading_signals(
                                    df.copy(),
                                    z_lookback=z_lb, z_entry=z_entry, z_exit=z_exit,
                                    delta=delta, ve=ve,
                                    min_hold_bars=min_hold,
                                    cooldown_bars=cooldown_bars,
                                    max_hold_bars=max_hold_bars
                                )

                                trades_0, eq_0, _, st_0, eff_0 = _backtest_and_pnls(
                                    full_df, min1_cache, tf_min, fee=0.0)
                                trades_r, eq_r, _, st_r, eff_r = _backtest_and_pnls(
                                    full_df, min1_cache, tf_min, fee=SINGLE_LEG_FEE)

                                if len(eff_0) < 15:
                                    continue

                                stats_0 = print_backtest_stats(trades_0, eq_0, effective_pnls=eff_0)
                                stats_r = print_backtest_stats(trades_r, eq_r, effective_pnls=eff_r)

                                fee_per_trade = stats_0['avg_profit_per_trade'] - stats_r['avg_profit_per_trade']

                                results.append({
                                    'z_entry': z_entry, 'z_exit': z_exit,
                                    'lookback_hours': lb_h, 'z_lookback_bars': z_lb,
                                    'delta_per_day': dpd, 'delta_per_bar': delta,
                                    've': ve,
                                    'min_hold_hours': mh_h, 'min_hold_bars': min_hold,
                                    'trades': stats_0['total_trades'],
                                    'zero_total_ret': round(stats_0['Total Return'], 2),
                                    'zero_avg_pnl': round(stats_0['avg_profit_per_trade'], 4),
                                    'zero_win_rate': round(stats_0['Win Rate'], 4),
                                    'zero_pf': round(stats_0['Profit Factor'], 3),
                                    'zero_max_dd': round(stats_0['Max Drawdown'], 2),
                                    'real_total_ret': round(stats_r['Total Return'], 2),
                                    'real_avg_pnl': round(stats_r['avg_profit_per_trade'], 4),
                                    'real_win_rate': round(stats_r['Win Rate'], 4),
                                    'real_pf': round(stats_r['Profit Factor'], 3),
                                    'real_max_dd': round(stats_r['Max Drawdown'], 2),
                                    'fee_per_trade': round(fee_per_trade, 4),
                                    'avg_hold_hrs': round(stats_0['avg_duration'] / 60, 1),
                                    'max_hold_hrs': round(stats_0['最长持仓时间'] / 60, 1),
                                })
                            except Exception:
                                continue

                            if count % 100 == 0:
                                elapsed = time.time() - start_time
                                print(f"  进度: {count}/{total_combos} | "
                                      f"有效: {len(results)} | 耗时: {elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(f"\n扫描完成: {count} 组合, {len(results)} 有效, 耗时 {elapsed:.1f}s")

    if not results:
        print("没有有效结果！")
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    print(f"\n{'=' * 100}")
    print(f"TOP 20 参数组合（按真实手续费总收益排序）| {key_word} | {timeframe}")
    print(f"{'=' * 100}")
    top20 = result_df.sort_values('real_total_ret', ascending=False).head(20)
    display_cols = ['z_entry', 'z_exit', 'lookback_hours', 'delta_per_day', 'min_hold_hours',
                    'trades', 'zero_avg_pnl', 'real_avg_pnl', 'fee_per_trade',
                    'real_total_ret', 'real_win_rate', 'real_pf', 'real_max_dd', 'avg_hold_hrs']
    print(top20[display_cols].to_string(index=False))

    print(f"\n{'=' * 100}")
    print(f"各维度边际效应 | {key_word} | {timeframe}")
    print(f"{'=' * 100}")
    for dim in ['z_entry', 'z_exit', 'lookback_hours', 'delta_per_day', 'min_hold_hours']:
        grouped = result_df.groupby(dim).agg({
            'zero_avg_pnl': 'mean', 'real_avg_pnl': 'mean',
            'trades': 'mean', 'real_total_ret': 'mean',
        }).round(4)
        print(f"\n--- {dim} ---")
        print(grouped.to_string())

    profitable = result_df[result_df['real_total_ret'] > 0]
    print(f"\n{'=' * 80}\n核心判断 | {key_word} | {timeframe}\n{'=' * 80}")
    print(f"  总组合: {len(result_df)} | 盈利组合: {len(profitable)}")
    if len(profitable) > 0:
        print(f"  最高总收益: {profitable['real_total_ret'].max():.2f}%")
        print(f"  每笔利润: {profitable['real_avg_pnl'].min():.4f}% ~ {profitable['real_avg_pnl'].max():.4f}%")
        print(f"  持仓时间: {profitable['avg_hold_hrs'].min():.1f}h ~ {profitable['avg_hold_hrs'].max():.1f}h")
        print(f"  ✅ 存在可盈利参数区间，值得进一步做Walk-Forward验证")
    else:
        print(f"  ❌ 当前费率下无盈利组合")
        if not result_df.empty:
            zero_max = result_df['zero_avg_pnl'].max()
            fee_mean = result_df['fee_per_trade'].mean()
            print(f"  零费率最高每笔利润: {zero_max:.4f}% | "
                  f"每笔成本: {fee_mean:.4f}% | 成本/利润: {fee_mean / max(zero_max, 1e-4):.1f}x")

    os.makedirs('backtest_pair', exist_ok=True)
    out_path = f'backtest_pair/param_scan_{key_word}_{timeframe}.csv'
    result_df.to_csv(out_path, index=False)
    print(f"\n完整结果已保存到: {out_path}")
    return result_df


# =============================================================================
# 9. Walk-Forward
# =============================================================================
def _compute_param_neighbors(param_key, param_grids):
    neighbors = []
    for dim_idx in range(len(param_key)):
        grid = param_grids[dim_idx]
        cur_val = param_key[dim_idx]
        if cur_val in grid:
            pos = grid.index(cur_val)
            for offset in [-1, 1]:
                npos = pos + offset
                if 0 <= npos < len(grid):
                    nkey = list(param_key)
                    nkey[dim_idx] = grid[npos]
                    neighbors.append(tuple(nkey))
    return neighbors


def _wf_force_close_boundary(oos_trades, oos_equity_raw, test_exec_state,
                             boundary_equity, test_portion):
    if test_exec_state is None or float(test_exec_state[2]) == 0.0 or len(oos_equity_raw) == 0:
        return oos_trades, test_exec_state, False

    es = _unpack_exec_state(test_exec_state)
    oos_equity_raw[-1] -= SINGLE_LEG_FEE * 100.0

    closed_sum = float(oos_trades['net_pnl'].sum()) if not oos_trades.empty else 0.0
    fc_net_pnl = round(float(oos_equity_raw[-1]) - boundary_equity - closed_sum, 4)

    fc_open = pd.to_datetime(test_portion['open_time'].iloc[0]) if len(test_portion) > 0 else pd.NaT
    fc_close = pd.to_datetime(test_portion['open_time'].iloc[-1]) if len(test_portion) > 0 else pd.NaT
    fc_dur = 0.0
    if not pd.isna(es['entry_time']) and not pd.isna(fc_close):
        try:
            fc_dur = (pd.Timestamp(fc_close) - pd.Timestamp(es['entry_time'])).total_seconds() / 60.0
        except Exception:
            pass

    d, b = es['entry_dir'], es['entry_beta_val']
    em, esub = es['entry_main'], es['entry_sub']
    mm, ms = es['last_mark_main'], es['last_mark_sub']
    dir_str = (f'做多价差 (Long BTC / Short {b:.3f} ETH)' if d == -1
               else f'做空价差 (Short BTC / Long {b:.3f} ETH)')

    fc_trade = pd.DataFrame([{
        'open_time': fc_open, 'actual_open_time': es['entry_time'],
        'close_time': fc_close, 'duration_mins': fc_dur, 'direction': dir_str,
        'btc_roi': f"{((mm - em) / em * 100):.2f}" if em > 0 else "0.00",
        'eth_roi': f"{((ms - esub) / esub * 100):.2f}" if esub > 0 else "0.00",
        'net_pnl': fc_net_pnl, 'open_main': em, 'close_main': mm,
        'open_sub': esub, 'close_sub': ms,
        'entry_z': np.nan, 'exit_z': np.nan,
        'entry_spread': np.nan, 'exit_spread': np.nan,
        'entry_beta': b, 'exit_beta': b,
        'entry_alpha': np.nan, 'exit_alpha': np.nan,
        'signal': d, 'inherited_from_prev_window': True,
        'close_reason': 'wf_boundary_force_close',
    }])

    oos_trades = pd.concat([oos_trades, fc_trade], ignore_index=True) if not oos_trades.empty else fc_trade.copy()

    st = list(test_exec_state)
    st[2] = 0.0
    print(f"  [FIX-1] OOS期末强制平仓: dir={d}, net_pnl={fc_net_pnl:.4f}%")
    return oos_trades, tuple(st), True


def walk_forward_optimization(df_file, timeframe='15min',
                              train_months=6, test_months=1,
                              cooldown_hours=1.0, max_hold_days=4.0):
    key_word = os.path.basename(df_file).replace('.csv', '') + f'_{timeframe}'

    original_df = pd.read_csv(df_file)
    original_df['open_time'] = pd.to_datetime(original_df['open_time'])

    if timeframe != '1min':
        df = resample_data(original_df, timeframe)
        print(f"重采样: {len(original_df)} → {len(df)} 行 ({timeframe})")
    else:
        df = original_df.copy()

    tf_min = get_tf_minutes(timeframe)
    cooldown_bars = hours_to_bars(cooldown_hours, tf_min)
    max_hold_bars = days_to_bars(max_hold_days, tf_min)
    min1_cache = _build_1min_cache(original_df) if (USE_NEXT_BAR_EXEC or USE_1MIN_PATH) else None

    grids = _get_param_grids()
    param_grids = [grids['z_entry'], grids['z_exit'], grids['lookback_hours'],
                   grids['delta_per_day'], grids['ve'], grids['min_hold_hours']]

    min_trades_required = max(18, train_months * 3)

    all_oos_trades = []
    all_oos_stats = []
    all_oos_equity_segments = []
    all_oos_effective_pnls = []
    param_history = []
    oos_degradation_log = []

    start_date = df['open_time'].min()
    end_date = df['open_time'].max()
    train_delta = pd.DateOffset(months=train_months)
    test_delta = pd.DateOffset(months=test_months)

    current_train_start = start_date
    window_id = 0

    while current_train_start + train_delta + test_delta <= end_date:
        train_end = current_train_start + train_delta
        test_end = train_end + test_delta

        train_df = df[(df['open_time'] >= current_train_start) &
                      (df['open_time'] < train_end)].copy()
        test_df = df[(df['open_time'] >= train_end) &
                     (df['open_time'] < test_end)].copy()

        if len(train_df) < 500 or len(test_df) < 50:
            current_train_start += test_delta
            window_id += 1
            continue

        print(f"\n{'=' * 60}")
        print(f"Window {window_id}: Train [{current_train_start.date()} ~ {train_end.date()}] "
              f"→ Test [{train_end.date()} ~ {test_end.date()}]")
        print(f"Train: {len(train_df)} bars, Test: {len(test_df)} bars")

        all_candidates = {}
        all_evaluated_scores = {}
        combo_count = 0
        valid_count = 0

        for z_entry in grids['z_entry']:
            for z_exit in grids['z_exit']:
                if z_exit >= z_entry:
                    continue
                for lb_h in grids['lookback_hours']:
                    z_lb = hours_to_bars(lb_h, tf_min)
                    for dpd in grids['delta_per_day']:
                        delta = delta_per_day_to_bar(dpd, tf_min)
                        for ve in grids['ve']:
                            for mh_h in grids['min_hold_hours']:
                                min_hold = hours_to_bars(mh_h, tf_min)
                                combo_count += 1
                                param_key = (z_entry, z_exit, lb_h, dpd, ve, mh_h)
                                try:
                                    full_df, _, _ = generate_pair_trading_signals(
                                        train_df.copy(),
                                        z_lookback=z_lb, z_entry=z_entry, z_exit=z_exit,
                                        delta=delta, ve=ve,
                                        min_hold_bars=min_hold,
                                        cooldown_bars=cooldown_bars,
                                        max_hold_bars=max_hold_bars
                                    )

                                    train_start_ns, train_end_ns = _get_segment_bounds_ns(full_df, tf_min)

                                    trades, eq_curve, eq_times, exec_state_train, eff_pnls = \
                                        _backtest_and_pnls(full_df, min1_cache, tf_min)

                                    if len(eff_pnls) >= 2 and np.std(eff_pnls) > 0:
                                        raw_score = np.mean(eff_pnls) / np.std(eff_pnls) * np.sqrt(len(eff_pnls))
                                    elif len(eff_pnls) >= 2:
                                        raw_score = np.sign(np.mean(eff_pnls)) * 0.01
                                    else:
                                        raw_score = 0.0
                                    all_evaluated_scores[param_key] = raw_score

                                    if len(eff_pnls) < min_trades_required:
                                        continue
                                    if np.std(eff_pnls) == 0:
                                        continue
                                    if np.mean(eff_pnls) <= 0:
                                        continue

                                    _, _, _, _, eff_15x = _backtest_and_pnls(
                                        full_df, min1_cache, tf_min, fee=SINGLE_LEG_FEE * 1.5)
                                    if len(eff_15x) == 0 or np.mean(eff_15x) <= 0:
                                        continue

                                    seg_pnls = _split_train_thirds(train_start_ns, train_end_ns, eq_times, eq_curve)
                                    if sum(1 for s in seg_pnls if s > 0) < 2:
                                        continue

                                    if len(eq_curve) > 0:
                                        _eq_ext = np.concatenate(([0.0], eq_curve))
                                        _peak = np.maximum.accumulate(_eq_ext)
                                        _mdd = float(np.min(_eq_ext - _peak))
                                        if _mdd < WF_MAX_TRAIN_MDD:
                                            continue

                                    valid_count += 1
                                    train_total_ret = float(eq_curve[-1]) if len(eq_curve) > 0 else float(np.sum(eff_pnls))

                                    all_candidates[param_key] = {
                                        'score': raw_score,
                                        'params': {
                                            'z_entry': z_entry, 'z_exit': z_exit,
                                            'lookback_hours': lb_h, 'z_lookback': z_lb,
                                            'delta_per_day': dpd, 'delta': delta,
                                            've': ve,
                                            'min_hold_hours': mh_h, 'min_hold': min_hold,
                                        },
                                        'train_trades': len(eff_pnls),
                                        'train_closed_trades': len(trades),
                                        'train_ret': train_total_ret,
                                        'train_avg_pnl': float(np.mean(eff_pnls)),
                                    }
                                except Exception:
                                    all_evaluated_scores[param_key] = 0.0
                                    continue

        if not all_candidates:
            print(f"  Window {window_id}: 无通过硬过滤的参数 "
                  f"({combo_count} 组合, {valid_count} 有效), 跳过")
            current_train_start += test_delta
            window_id += 1
            continue

        # ===== 高原选择 =====
        sorted_keys = sorted(all_candidates.keys(),
                             key=lambda k: all_candidates[k]['score'], reverse=True)
        top_n = max(1, len(sorted_keys) // 5)
        top_keys = set(sorted_keys[:top_n])

        best_plateau_score = -np.inf
        best_key = sorted_keys[0]

        for pkey in top_keys:
            neighbors = _compute_param_neighbors(pkey, param_grids)
            neighbor_scores = [all_evaluated_scores[nk]
                               for nk in neighbors
                               if nk in all_evaluated_scores]
            avg_nb = float(np.median(neighbor_scores)) if neighbor_scores else 0.0
            plateau_score = all_candidates[pkey]['score'] * 0.5 + avg_nb * 0.5
            if plateau_score > best_plateau_score:
                best_plateau_score = plateau_score
                best_key = pkey

        chosen = all_candidates[best_key]
        best_params = chosen['params']

        param_history.append({
            'window_id': window_id, **best_params,
            'train_score': chosen['score'],
            'plateau_score': best_plateau_score,
            'train_trades': chosen['train_trades'],
            'train_closed_trades': chosen['train_closed_trades'],
            'train_total_ret': round(chosen['train_ret'], 4),
            'train_avg_pnl': round(chosen['train_avg_pnl'], 4),
            'combos_evaluated': combo_count,
            'combos_passed_filter': len(all_candidates),
        })
        print(f"  Best (plateau): {best_params}")
        print(f"  Train: t-stat={chosen['score']:.4f}, trades={chosen['train_trades']}, "
              f"closed={chosen['train_closed_trades']}, ret={chosen['train_ret']:.2f}%, "
              f"plateau={best_plateau_score:.4f} | passed: {len(all_candidates)}/{combo_count}")

        # ===== OOS 测试 =====
        combined = pd.concat([train_df, test_df], ignore_index=True)
        test_start_time = test_df['open_time'].iloc[0]

        test_full, _, _ = generate_pair_trading_signals(
            combined,
            z_lookback=best_params['z_lookback'],
            z_entry=best_params['z_entry'],
            z_exit=best_params['z_exit'],
            delta=best_params['delta'],
            ve=best_params['ve'],
            min_hold_bars=best_params['min_hold'],
            cooldown_bars=cooldown_bars,
            max_hold_bars=max_hold_bars
        )

        test_mask = test_full['open_time'] >= test_start_time
        train_portion = test_full[~test_mask].copy()
        test_portion = test_full[test_mask].copy()

        _, train_equity, _, train_exec_state = compute_equity_and_trades(
            train_portion, min1_cache=min1_cache, tf_minutes=tf_min
        )
        boundary_equity = train_equity[-1] if len(train_equity) > 0 else 0.0

        oos_trades, oos_equity_raw, oos_eq_times, test_exec_state = compute_equity_and_trades(
            test_portion, min1_cache=min1_cache,
            exec_state=train_exec_state, tf_minutes=tf_min
        )

        oos_trades, test_exec_state, _ = _wf_force_close_boundary(
            oos_trades, oos_equity_raw, test_exec_state, boundary_equity, test_portion)

        oos_equity = oos_equity_raw - boundary_equity if len(oos_equity_raw) > 0 else oos_equity_raw
        oos_effective_pnls = _build_effective_pnls(oos_trades, oos_equity, test_exec_state)

        if len(oos_equity) > 0:
            offset = all_oos_equity_segments[-1][-1] if all_oos_equity_segments else 0.0
            all_oos_equity_segments.append(oos_equity + offset)

        if len(oos_effective_pnls) > 0:
            all_oos_effective_pnls.append(oos_effective_pnls)

        if not oos_trades.empty:
            oos_trades['window_id'] = window_id
            oos_trades['best_params'] = str(best_params)
            all_oos_trades.append(oos_trades)

        stats = print_backtest_stats(oos_trades, oos_equity, effective_pnls=oos_effective_pnls)
        if stats:
            stats['window_id'] = window_id
            stats['best_params'] = str(best_params)
            all_oos_stats.append(stats)

            print(f"  OOS: trades={stats['total_trades']}, "
                  f"avg_pnl={stats['avg_profit_per_trade']:.4f}%, "
                  f"win={stats['Win Rate']:.2%}, "
                  f"ret={stats['Total Return']:.2f}%, "
                  f"MDD={stats['Max Drawdown']:.4f}%")

            train_avg = chosen['train_avg_pnl']
            oos_avg = stats['avg_profit_per_trade']
            deg_ratio = (train_avg / oos_avg) if oos_avg > 0 else float('inf')

            oos_degradation_log.append({
                'window_id': window_id,
                'train_avg_pnl': round(train_avg, 4),
                'oos_avg_pnl': round(oos_avg, 4),
                'degradation_ratio': round(deg_ratio, 2) if np.isfinite(deg_ratio) else float('inf'),
            })
            ratio_s = f"{deg_ratio:.2f}" if np.isfinite(deg_ratio) else "inf"
            print(f"  退化比 (Train/OOS): {ratio_s}")

        current_train_start += test_delta
        window_id += 1

    # ========== 汇总 ==========
    print(f"\n{'=' * 60}\nWalk-Forward 汇总 | {key_word}\n{'=' * 60}")
    os.makedirs('backtest_pair', exist_ok=True)

    combined_oos = pd.concat(all_oos_trades, ignore_index=True) if all_oos_trades else pd.DataFrame()
    combined_oos_equity = np.concatenate(all_oos_equity_segments) if all_oos_equity_segments else np.array([])
    combined_oos_eff = np.concatenate(all_oos_effective_pnls) if all_oos_effective_pnls else np.array([])

    if not combined_oos.empty:
        combined_oos.to_csv(f'backtest_pair/{key_word}_wf_oos_trades.csv', index=False)

    if len(combined_oos_equity) > 0 or not combined_oos.empty:
        final = print_backtest_stats(combined_oos, combined_oos_equity, effective_pnls=combined_oos_eff)
        if final:
            print(f"  总交易: {final['total_trades']} | 总收益: {final['Total Return']:.4f}%")
            print(f"  平均每笔: {final['avg_profit_per_trade']:.4f}% | "
                  f"胜率: {final['Win Rate']:.2%} | PF: {final['Profit Factor']:.4f}")
            print(f"  最大回撤: {final['Max Drawdown']:.4f}%")

        if all_oos_stats:
            oos_df = pd.DataFrame(all_oos_stats)
            print(f"\n  --- 各窗口OOS表现 ---")
            for _, row in oos_df.iterrows():
                print(f"  W{int(row['window_id'])}: trades={int(row['total_trades'])}, "
                      f"ret={row['Total Return']:.2f}%, win={row['Win Rate']:.1%}, "
                      f"pf={row['Profit Factor']:.2f}, MDD={row['Max Drawdown']:.4f}%")

            win_windows = len(oos_df[oos_df['Total Return'] > 0])
            total_windows = len(oos_df)
            if total_windows > 0:
                print(f"\n  窗口胜率: {win_windows}/{total_windows} = {win_windows / total_windows:.1%}")

        if oos_degradation_log:
            deg_df = pd.DataFrame(oos_degradation_log)
            print(f"\n  --- OOS退化比诊断 ---")
            for _, row in deg_df.iterrows():
                r = row['degradation_ratio']
                flag = " ⚠️" if (not np.isfinite(r) or r > 3.0) else ""
                r_s = f"{r:.2f}" if np.isfinite(r) else "inf"
                print(f"  W{int(row['window_id'])}: train={row['train_avg_pnl']:.4f}%, "
                      f"oos={row['oos_avg_pnl']:.4f}%, ratio={r_s}{flag}")

            finite_ratios = [r['degradation_ratio'] for r in oos_degradation_log
                             if np.isfinite(r['degradation_ratio'])]
            if finite_ratios:
                avg_r = np.mean(finite_ratios)
                hi_count = sum(1 for r in finite_ratios if r > 3.0)
                print(f"\n  平均退化比: {avg_r:.2f} | 退化>3的窗口: {hi_count}/{len(finite_ratios)}")
                if avg_r > 3.0:
                    print(f"  🔴 退化严重，过拟合风险高")
                elif avg_r > 2.0:
                    print(f"  🟡 有一定退化，建议增大训练窗口")
                else:
                    print(f"  🟢 退化比合理")

            deg_df.to_csv(f'backtest_pair/{key_word}_oos_degradation.csv', index=False)
    else:
        print("  所有窗口均无样本外权益变化")

    if param_history:
        ph = pd.DataFrame(param_history)
        ph.to_csv(f'backtest_pair/{key_word}_param_history.csv', index=False)
        print(f"\n  --- 参数收敛性 ---")
        for col in ['z_entry', 'z_exit', 'lookback_hours', 'delta_per_day', 've', 'min_hold_hours']:
            if col in ph.columns:
                vals = ph[col]
                print(f"  {col}: 最常选={vals.mode().values}, std={vals.std():.4f}")


# =============================================================================
# 10. 主入口
# =============================================================================
if __name__ == '__main__':
    target_dir = 'kline_data'
    suffix = '1m.csv'

    df_files = [
        os.path.join(target_dir, f)
        for f in os.listdir(target_dir)
        if os.path.isfile(os.path.join(target_dir, f))
           and f.endswith(suffix)
           and f.count('_') == 2
    ]
    timeframes = ['1min', '5min', '10min', '15min', '30min']

    for df_file in df_files:
        for tf in timeframes:
            print("\n" + "=" * 70)
            print(f"参数灵敏度扫描 | {os.path.basename(df_file)} | {tf}")
            print("=" * 70)
            parameter_sensitivity_scan(df_file, timeframe=tf)

    final_df_file = 'kline_data/DOGE_ETH_1m.csv'
    walk_forward_optimization(final_df_file, timeframe='30min',
                              train_months=6, test_months=1,
                              cooldown_hours=1.0, max_hold_days=4.0)