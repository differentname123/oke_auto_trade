import traceback

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

# --- KF Endogenous Diagnostics (方案1) ---
NI_ENTRY_GATE = 3.0
KF_KILL_TRACE_MULT = 5.0
KF_KILL_NI_CONSEC = 3
KF_KILL_NI_THRESHOLD = 3.0


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
# 2. 卡尔曼滤波（有状态版本 + 内生诊断输出）
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
    normalized_innovations = np.zeros(n_obs)
    trace_P_arr = np.zeros(n_obs)

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

        if S > 0.0:
            normalized_innovations[t] = error / np.sqrt(S)
        else:
            normalized_innovations[t] = 0.0

        K0 = (P00 + P01 * x) / S
        K1 = (P10 + P11 * x) / S

        alpha_mean += K0 * error
        beta_mean += K1 * error

        new_P00 = P00 - K0 * (P00 + P10 * x)
        new_P01 = P01 - K0 * (P01 + P11 * x)
        new_P10 = P10 - K1 * (P00 + P10 * x)
        new_P11 = P11 - K1 * (P01 + P11 * x)
        P00, P01, P10, P11 = new_P00, new_P01, new_P10, new_P11

        trace_P_arr[t] = P00 + P11

        alphas[t] = alpha_mean
        betas[t] = beta_mean
        spreads[t] = error

    return (betas, alphas, spreads, normalized_innovations, trace_P_arr,
            alpha_mean, beta_mean, P00, P01, P10, P11)


# =============================================================================
# 3. 信号生成（含KF内生诊断门控）
# =============================================================================
@nb.njit(cache=True)
def fast_generate_signals(z_values, spreads, betas, alphas,
                          log_main, log_sub, roll_std,
                          ni_arr, trace_P_arr,
                          lookback, z_entry, z_exit,
                          min_hold_bars, cooldown_bars, max_hold_bars,
                          sl_mult, single_leg_fee,
                          ni_entry_gate, kf_kill_trace_mult,
                          kf_kill_ni_consec, kf_kill_ni_threshold,
                          beta_min=0.1, beta_max=5.0,
                          init_pos=0, init_held=0, init_since_close=9999,
                          init_frozen_beta=0.0, init_frozen_alpha=0.0,
                          init_entry_spread=0.0, init_entry_std=0.0,
                          init_entry_trace_P=0.0, init_consec_ni_count=0):
    n = len(z_values)
    signals = np.zeros(n)
    frozen_betas_arr = np.zeros(n)
    frozen_alphas_arr = np.zeros(n)
    entry_spreads_arr = np.zeros(n)
    entry_stds_arr = np.zeros(n)
    stop_flags = np.zeros(n)
    kf_kill_flags = np.zeros(n)

    pos = init_pos
    held = init_held
    since_close = init_since_close
    frozen_beta = init_frozen_beta
    frozen_alpha = init_frozen_alpha
    entry_spread = init_entry_spread
    entry_std = init_entry_std
    entry_trace_P = init_entry_trace_P
    consec_ni_count = init_consec_ni_count

    start_idx = 0 if pos != 0 else lookback

    for i in range(start_idx, n):
        z = z_values[i]

        if pos == 0:
            since_close += 1
            if since_close >= cooldown_bars:
                cur_std = roll_std[i]
                cur_beta = betas[i]
                ni_ok = abs(ni_arr[i]) <= ni_entry_gate
                if cur_std > 0.0 and cur_beta >= beta_min and cur_beta <= beta_max and ni_ok:
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
                            entry_trace_P = trace_P_arr[i]
                            consec_ni_count = 0
                        elif z < -z_entry:
                            pos = -1
                            held = 0
                            frozen_beta = cur_beta
                            frozen_alpha = alphas[i]
                            entry_spread = log_main[i] - frozen_alpha - frozen_beta * log_sub[i]
                            entry_std = cur_std
                            entry_trace_P = trace_P_arr[i]
                            consec_ni_count = 0
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

            kf_killed = False
            if not stopped:
                if entry_trace_P > 0.0 and trace_P_arr[i] > entry_trace_P * kf_kill_trace_mult:
                    kf_killed = True
                if abs(ni_arr[i]) > kf_kill_ni_threshold:
                    consec_ni_count += 1
                else:
                    consec_ni_count = 0
                if consec_ni_count >= kf_kill_ni_consec:
                    kf_killed = True

            if stopped:
                pos = 0
                since_close = 0
                stop_flags[i] = 1.0
                consec_ni_count = 0
            elif kf_killed:
                pos = 0
                since_close = 0
                stop_flags[i] = 1.0
                kf_kill_flags[i] = 1.0
                consec_ni_count = 0
            elif max_hold_bars > 0 and held >= max_hold_bars:
                pos = 0
                since_close = 0
                consec_ni_count = 0
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
                    consec_ni_count = 0
                elif pos == -1 and z > -z_exit and spread_improved:
                    pos = 0
                    since_close = 0
                    consec_ni_count = 0

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
            stop_flags, kf_kill_flags,
            pos, held, since_close,
            frozen_beta, frozen_alpha, entry_spread, entry_std,
            entry_trace_P, consec_ni_count)


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
    (betas, alphas, kf_spreads, ni_values, trace_P_values,
     ka, kb, kP00, kP01, kP10, kP11) = fast_kalman_filter(
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
    df['kf_ni'] = ni_values
    df['kf_trace_P'] = trace_P_values

    spread_s = pd.Series(kf_spreads)
    roll_mean = spread_s.rolling(z_lookback, min_periods=z_lookback).mean()
    roll_std = spread_s.rolling(z_lookback, min_periods=z_lookback).std()
    z_score = ((spread_s - roll_mean) / roll_std).values
    df['z_score'] = z_score
    df['roll_std'] = roll_std.values

    z_clean = np.nan_to_num(z_score, nan=0.0, posinf=0.0, neginf=0.0)
    rs_clean = np.nan_to_num(roll_std.values, nan=0.0, posinf=0.0, neginf=0.0)
    ni_clean = np.nan_to_num(ni_values, nan=0.0, posinf=0.0, neginf=0.0)
    trP_clean = np.nan_to_num(trace_P_values, nan=0.0, posinf=0.0, neginf=0.0)

    if pos_state is None:
        pos_state = (0, 0, 9999, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    _ps = pos_state
    _psn = len(_ps)

    (raw_signals, frozen_betas, frozen_alphas_state,
     entry_spreads_state, entry_stds_state,
     stop_flags_raw, kf_kill_flags_raw,
     fp, fh, fsc, ffb, ffa, fes, fest,
     f_entry_trP, f_consec_ni) = fast_generate_signals(
        z_clean, kf_spreads, betas, alphas,
        df['log_main'].values, df['log_sub'].values, rs_clean,
        ni_clean, trP_clean,
        z_lookback, z_entry, z_exit,
        min_hold_bars, cooldown_bars, max_hold_bars,
        SL_MULT, SINGLE_LEG_FEE,
        NI_ENTRY_GATE, KF_KILL_TRACE_MULT,
        KF_KILL_NI_CONSEC, KF_KILL_NI_THRESHOLD,
        BETA_MIN, BETA_MAX,
        init_pos=int(_ps[0]), init_held=int(_ps[1]),
        init_since_close=int(_ps[2]),
        init_frozen_beta=float(_ps[3]), init_frozen_alpha=float(_ps[4]),
        init_entry_spread=float(_ps[5]), init_entry_std=float(_ps[6]),
        init_entry_trace_P=float(_ps[7]) if _psn > 7 else 0.0,
        init_consec_ni_count=int(_ps[8]) if _psn > 8 else 0
    )
    final_pos_state = (fp, fh, fsc, ffb, ffa, fes, fest, f_entry_trP, f_consec_ni)

    do_shift = USE_NEXT_BAR_EXEC
    df['signal'] = _shift_or_copy(raw_signals, do_shift)
    df['signal_frozen_beta'] = _shift_or_copy(frozen_betas, do_shift)
    df['signal_frozen_alpha'] = _shift_or_copy(frozen_alphas_state, do_shift)
    df['signal_entry_spread'] = _shift_or_copy(entry_spreads_state, do_shift)
    df['signal_entry_std'] = _shift_or_copy(entry_stds_state, do_shift)
    df['signal_entry_z'] = _shift_or_copy(z_score, do_shift, fill=np.nan)
    df['stop_flag'] = _shift_or_copy(stop_flags_raw, do_shift)
    df['kf_kill_flag'] = _shift_or_copy(kf_kill_flags_raw, do_shift)

    _sig_ctx = {
        'z_entry': float(z_entry), 'z_exit': float(z_exit),
        'min_hold_bars': int(min_hold_bars), 'cooldown_bars': int(cooldown_bars),
        'max_hold_bars': int(max_hold_bars), 'signal_fee': float(SINGLE_LEG_FEE),
        'ni_entry_gate': float(NI_ENTRY_GATE),
        'kf_kill_trace_mult': float(KF_KILL_TRACE_MULT),
        'kf_kill_ni_consec': int(KF_KILL_NI_CONSEC),
        'kf_kill_ni_threshold': float(KF_KILL_NI_THRESHOLD),
    }
    df.attrs['_pt_signal_ctx'] = _sig_ctx
    for _k, _v in _sig_ctx.items():
        df[f'__pt_{_k}'] = _v

    return df, final_kf_state, final_pos_state


# =============================================================================
# 5. 1min cache / 有效PnL / 时间分段
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


def _split_train_monthly_returns(segment_start_time, segment_end_time,
                                 equity_times, equity_curve, num_months):
    """将权益曲线按等分月切分，返回每月收益列表。"""
    if equity_curve is None or len(equity_curve) == 0:
        return []

    eq = np.asarray(equity_curve, dtype=float)
    etimes = np.asarray(equity_times).astype('datetime64[ns]')
    start = np.datetime64(segment_start_time, 'ns')
    end = np.datetime64(segment_end_time, 'ns')

    if str(start) == 'NaT' or str(end) == 'NaT':
        return []

    total_ns = int((end - start).astype('timedelta64[ns]').astype(np.int64))
    if total_ns <= 0 or num_months <= 0:
        return [float(eq[-1])]

    def _eq_at(boundary_ns):
        idx = np.searchsorted(etimes, boundary_ns, side='right') - 1
        idx = max(0, min(idx, len(eq) - 1))
        return float(eq[idx])

    monthly_returns = []
    prev_eq = 0.0
    for i in range(1, num_months + 1):
        boundary = start + np.timedelta64(int(total_ns * i / num_months), 'ns')
        cur_eq = _eq_at(boundary)
        monthly_returns.append(cur_eq - prev_eq)
        prev_eq = cur_eq

    return monthly_returns


@nb.njit(cache=True)
def _scan_1min_for_equity(
    min1_cm, min1_cs, min1_lm, min1_ls, min1_times_i64,
    scan_start, scan_end,
    entry_main, entry_sub, entry_beta_val, entry_dir,
    entry_frozen_alpha, entry_spread_val, entry_std_val,
    sl_mult, cum_realized, fee
):
    """
    扫描 [scan_start, scan_end) 范围内的1min K线:
    - 检测止损触发
    - 逐根计算权益曲线点

    返回:
        stop_found     : bool   是否触发了止损
        stop_j         : int64  止损触发的1min索引 (-1表示无)
        eq_vals        : float64[:]  权益值数组
        eq_times_i64   : int64[:]    权益时间戳数组(ns)
        count          : int64  有效权益点数量
        stop_close_m   : float64  止损时main价格
        stop_close_s   : float64  止损时sub价格
        new_cum_realized : float64  止损后的累计已实现收益
    """
    n_pts = scan_end - scan_start
    if n_pts <= 0:
        empty_f = np.empty(0, dtype=np.float64)
        empty_i = np.empty(0, dtype=np.int64)
        return (False, np.int64(-1), empty_f, empty_i, np.int64(0),
                0.0, 0.0, cum_realized)

    eq_vals = np.empty(n_pts, dtype=np.float64)
    eq_times_i64 = np.empty(n_pts, dtype=np.int64)
    count = np.int64(0)

    total_exp = 1.0 + abs(entry_beta_val)
    stop_found = False
    stop_j = np.int64(-1)
    stop_close_m = 0.0
    stop_close_s = 0.0
    new_cum_realized = cum_realized

    check_stop = (entry_std_val > 0.0 and sl_mult > 0.0)

    for j in range(scan_start, scan_end):
        cur_m = min1_cm[j]
        cur_s = min1_cs[j]
        cur_lm = min1_lm[j]
        cur_ls = min1_ls[j]
        cur_t = min1_times_i64[j]

        # ---------- 止损检测 ----------
        should_stop = False
        if check_stop:
            frozen_spread = cur_lm - entry_frozen_alpha - entry_beta_val * cur_ls
            if entry_dir == 1:
                if (frozen_spread - entry_spread_val) > sl_mult * entry_std_val:
                    should_stop = True
            elif entry_dir == -1:
                if (entry_spread_val - frozen_spread) > sl_mult * entry_std_val:
                    should_stop = True

        if should_stop:
            # 计算平仓已实现盈亏 (与 _calc_norm 完全一致)
            br = (cur_m - entry_main) / entry_main if entry_main != 0.0 else 0.0
            er = (cur_s - entry_sub) / entry_sub if entry_sub != 0.0 else 0.0
            if entry_dir == -1:
                gross = br - entry_beta_val * er
            else:
                gross = entry_beta_val * er - br
            norm = gross / total_exp if total_exp > 0.0 else 0.0
            realized_add = norm - fee
            new_cum_realized = cum_realized + realized_add

            eq_times_i64[count] = cur_t
            eq_vals[count] = new_cum_realized * 100.0
            count += 1

            stop_found = True
            stop_j = np.int64(j)
            stop_close_m = cur_m
            stop_close_s = cur_s
            break

        # ---------- 未实现盈亏权益 ----------
        br = (cur_m - entry_main) / entry_main if entry_main != 0.0 else 0.0
        er = (cur_s - entry_sub) / entry_sub if entry_sub != 0.0 else 0.0
        if entry_dir == -1:
            gross = br - entry_beta_val * er
        else:
            gross = entry_beta_val * er - br
        norm = gross / total_exp if total_exp > 0.0 else 0.0

        eq_times_i64[count] = cur_t
        eq_vals[count] = (cum_realized + norm) * 100.0
        count += 1

    return (stop_found, stop_j, eq_vals, eq_times_i64, count,
            stop_close_m, stop_close_s, new_cum_realized)


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

    signals = df['signal'].to_numpy(dtype=int)
    times = pd.to_datetime(df['open_time']).to_numpy(dtype='datetime64[ns]')
    close_main = df['close_main'].to_numpy(dtype=float)
    close_sub = df['close_sub'].to_numpy(dtype=float)

    def _col(col_name, default, dtype=float):
        if col_name in df.columns:
            return df[col_name].to_numpy(dtype=dtype)
        return default() if callable(default) else np.full(n, default, dtype=dtype)

    z_scores = _col('z_score', lambda: np.full(n, np.nan))
    spreads_arr = _col('spread', lambda: np.full(n, np.nan))
    betas_arr = _col('beta', lambda: np.zeros(n))
    alphas_arr = _col('alpha', lambda: np.zeros(n))
    roll_std_arr = _col('roll_std', lambda: np.zeros(n))

    signal_fb = _col('signal_frozen_beta', lambda: betas_arr.copy())
    signal_fa = _col('signal_frozen_alpha', lambda: alphas_arr.copy())
    signal_es = _col('signal_entry_spread', 0.0)
    signal_estd = _col('signal_entry_std', 0.0)
    signal_entry_z = _col('signal_entry_z', lambda: np.full(n, np.nan))
    stop_flags_col = _col('stop_flag', 0, dtype=int)
    kf_kill_flags_col = _col('kf_kill_flag', 0, dtype=int)
    log_main_arr = _col('log_main', lambda: np.log(close_main))
    log_sub_arr = _col('log_sub', lambda: np.log(close_sub))
    ni_arr = _col('kf_ni', 0.0)
    trace_P_arr = _col('kf_trace_P', 0.0)

    z_clean = np.nan_to_num(z_scores, nan=0.0, posinf=0.0, neginf=0.0)
    rs_clean = np.nan_to_num(roll_std_arr, nan=0.0, posinf=0.0, neginf=0.0)
    ni_clean = np.nan_to_num(ni_arr, nan=0.0, posinf=0.0, neginf=0.0)
    trP_clean = np.nan_to_num(trace_P_arr, nan=0.0, posinf=0.0, neginf=0.0)

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
    sig_ni_entry_gate = float(_read_sig_ctx('ni_entry_gate', NI_ENTRY_GATE))
    sig_kf_kill_trace_mult = float(_read_sig_ctx('kf_kill_trace_mult', KF_KILL_TRACE_MULT))
    sig_kf_kill_ni_consec = int(_read_sig_ctx('kf_kill_ni_consec', KF_KILL_NI_CONSEC))
    sig_kf_kill_ni_threshold = float(_read_sig_ctx('kf_kill_ni_threshold', KF_KILL_NI_THRESHOLD))

    can_rebuild_signal_path = (
            np.isfinite(sig_z_entry) and np.isfinite(sig_z_exit)
            and len(z_clean) == n and len(rs_clean) == n
            and len(spreads_arr) == n and len(betas_arr) == n and len(alphas_arr) == n
            and len(log_main_arr) == n and len(log_sub_arr) == n
            and len(ni_clean) == n and len(trP_clean) == n
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

    # =====================================================================
    # [优化] 预转换时间为int64，预计算bar到1min的索引映射
    # =====================================================================
    times_i64 = times.view(np.int64)
    # [修复] 更优雅安全的 Numpy 时间增量转 int64 写法
    bar_delta_ns_i64 = np.int64(bar_delta_ns.astype('timedelta64[ns]').astype(np.int64))

    if use_1m_path:
        min1_times_i64 = min1_times.view(np.int64)
        _bar_l_arr = np.searchsorted(min1_times_i64, times_i64, side='left')
        _end_times_i64 = np.empty(n, dtype=np.int64)
        _end_times_i64[:-1] = times_i64[1:]
        _end_times_i64[-1] = times_i64[-1] + bar_delta_ns_i64
        _bar_r_arr = np.searchsorted(min1_times_i64, _end_times_i64, side='left')

    # =====================================================================
    # [修复/优化] 预分配权益曲线缓冲区，带安全动态扩容机制防 OOM
    # =====================================================================
    _max_eq = (len(min1_times) + n + 1000) if use_1m_path else (n + 1000)
    _eq_vals_buf = np.empty(_max_eq, dtype=np.float64)
    _eq_times_buf = np.empty(_max_eq, dtype=np.int64)
    _eq_pos = 0

    def _ensure_capacity(add_size):
        nonlocal _max_eq, _eq_vals_buf, _eq_times_buf
        if _eq_pos + add_size > _max_eq:
            _max_eq = max(_max_eq * 2, _eq_pos + add_size + 10000)
            _eq_vals_buf = np.resize(_eq_vals_buf, _max_eq)
            _eq_times_buf = np.resize(_eq_times_buf, _max_eq)

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
    last_mark_main = close_main[0]
    last_mark_sub = close_sub[0]

    def _append_eq_point(point_time_i64, point_value):
        nonlocal _eq_pos
        if _eq_pos > 0 and _eq_times_buf[_eq_pos - 1] == point_time_i64:
            _eq_vals_buf[_eq_pos - 1] = point_value
        else:
            _ensure_capacity(1)
            _eq_times_buf[_eq_pos] = point_time_i64
            _eq_vals_buf[_eq_pos] = point_value
            _eq_pos += 1

    def _batch_append_eq(src_times_i64, src_vals, count):
        nonlocal _eq_pos
        if count <= 0:
            return
        start = 0
        if _eq_pos > 0 and _eq_times_buf[_eq_pos - 1] == src_times_i64[0]:
            _eq_vals_buf[_eq_pos - 1] = src_vals[0]
            start = 1
        nc = count - start
        if nc > 0:
            _ensure_capacity(nc)
            _eq_times_buf[_eq_pos:_eq_pos + nc] = src_times_i64[start:count]
            _eq_vals_buf[_eq_pos:_eq_pos + nc] = src_vals[start:count]
            _eq_pos += nc

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

        if close_reason in ('hard_stop_1m', 'kf_kill'):
            exit_z, exit_spread = np.nan, np.nan
            exit_alpha, exit_beta = entry_frozen_alpha, entry_beta_val
        else:
            exit_z = z_scores[bar_idx] if bar_idx < n else np.nan
            exit_spread = spreads_arr[bar_idx] if bar_idx < n else np.nan
            exit_alpha = alphas_arr[bar_idx] if bar_idx < n else np.nan
            exit_beta = betas_arr[bar_idx] if bar_idx < n else entry_beta_val

        duration_mins = 0.0
        if not np.isnat(np.datetime64(seg_open_time)):
            diff_ns = np.int64(exec_time_val) - np.int64(seg_open_time)
            duration_mins = diff_ns / 60_000_000_000.0

        trades.append({
            'open_time': seg_open_time,
            'actual_open_time': actual_open_time,
            'close_time': exec_time_val,
            'duration_mins': duration_mins,
            'entry_dir': entry_dir,
            'entry_beta': entry_beta_val,
            'raw_open_main': seg_entry_main,
            'raw_open_sub': seg_entry_sub,
            'raw_close_main': exec_m,
            'raw_close_sub': exec_s,
            'net_pnl': round(seg_net * 100, 4),
            'open_main': seg_entry_main, 'close_main': exec_m,
            'open_sub': seg_entry_sub, 'close_sub': exec_s,
            'entry_z': seg_entry_z if not seg_inherited else np.nan,
            'exit_z': exit_z,
            'entry_spread': seg_entry_spread_meta if not seg_inherited else np.nan,
            'exit_spread': exit_spread,
            'exit_beta': exit_beta,
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
        kf_kill_flags_col[stop_bar_idx] = 0

        if not can_rebuild_signal_path:
            return

        suffix = stop_bar_idx + 1
        if suffix >= n:
            return

        (new_raw_sig, new_fb, new_fa,
         new_es, new_estd,
         new_stop_raw, new_kf_kill_raw,
         _, _, _, _, _, _, _,
         _, _) = fast_generate_signals(
            z_clean[suffix:], spreads_arr[suffix:], betas_arr[suffix:], alphas_arr[suffix:],
            log_main_arr[suffix:], log_sub_arr[suffix:], rs_clean[suffix:],
            ni_clean[suffix:], trP_clean[suffix:],
            0, sig_z_entry, sig_z_exit,
            sig_min_hold_bars, sig_cooldown_bars, sig_max_hold_bars,
            SL_MULT, sig_fee_assumption,
            sig_ni_entry_gate, sig_kf_kill_trace_mult,
            sig_kf_kill_ni_consec, sig_kf_kill_ni_threshold,
            BETA_MIN, BETA_MAX,
            init_pos=0, init_held=0, init_since_close=0,
            init_frozen_beta=0.0, init_frozen_alpha=0.0,
            init_entry_spread=0.0, init_entry_std=0.0,
            init_entry_trace_P=0.0, init_consec_ni_count=0
        )

        signals[suffix:] = 0
        signal_fb[suffix:] = betas_arr[suffix:]
        signal_fa[suffix:] = alphas_arr[suffix:]
        signal_es[suffix:] = 0.0
        signal_estd[suffix:] = 0.0
        signal_entry_z[suffix:] = np.nan
        stop_flags_col[suffix:] = 0
        kf_kill_flags_col[suffix:] = 0

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
                kf_kill_flags_col[assign_start:] = new_kf_kill_raw[:-1].astype(int)
        else:
            signals[suffix:] = new_raw_sig
            signal_fb[suffix:] = new_fb
            signal_fa[suffix:] = new_fa
            signal_es[suffix:] = new_es
            signal_estd[suffix:] = new_estd
            signal_entry_z[suffix:] = z_scores[suffix:]
            stop_flags_col[suffix:] = new_stop_raw.astype(int)
            kf_kill_flags_col[suffix:] = new_kf_kill_raw.astype(int)

    def _determine_close_reason(bar_idx):
        if kf_kill_flags_col[bar_idx] == 1:
            return 'kf_kill'
        if stop_flags_col[bar_idx] == 1:
            return 'hard_stop'
        return 'signal'

    def _get_exec_point(bar_idx):
        if use_1m_exec:
            return _lookup_1min_point(min1_cache, times[bar_idx])
        return (times[bar_idx], close_main[bar_idx], close_sub[bar_idx],
                log_main_arr[bar_idx], log_sub_arr[bar_idx])

    def _check_stop_single(cur_time, cur_m, cur_s, cur_lm, cur_ls, bar_idx):
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

    if use_1m_path:
        for i in range(n):
            sig = int(signals[i])
            prev_sig = int(signals[i - 1]) if i > 0 else int(prev_sig_seed)

            l = int(_bar_l_arr[i])
            r = int(_bar_r_arr[i])

            if r <= l:
                exec_time_val, exec_m, exec_s, _, _ = _get_exec_point(i)

                if use_1m_exec:
                    if sig != 0 and prev_sig == 0 and not in_trade:
                        _open_position(sig, exec_time_val, exec_m, exec_s, i)
                    elif in_trade and sig == 0 and prev_sig != 0:
                        _cr = _determine_close_reason(i)
                        _close_position(exec_time_val, exec_m, exec_s, i, close_reason=_cr)
                else:
                    if in_trade and sig == 0 and prev_sig != 0:
                        _cr = _determine_close_reason(i)
                        _close_position(exec_time_val, exec_m, exec_s, i, close_reason=_cr)
                    elif sig != 0 and prev_sig == 0 and not in_trade:
                        _open_position(sig, exec_time_val, exec_m, exec_s, i)

                if in_trade:
                    unrealized = _calc_norm(entry_main, entry_sub, close_main[i], close_sub[i],
                                            entry_beta_val, entry_dir)
                    _append_eq_point(times_i64[i] + bar_delta_ns_i64,
                                     (cum_realized + unrealized) * 100.0)
                else:
                    _append_eq_point(times_i64[i] + bar_delta_ns_i64,
                                     cum_realized * 100.0)

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
                    _cr = _determine_close_reason(i)
                    _close_position(start_t, start_m, start_s, i, close_reason=_cr)
                    _append_eq_point(min1_times_i64[l], cum_realized * 100.0)
                    last_mark_main = min1_cm[r - 1]
                    last_mark_sub = min1_cs[r - 1]
                    continue

                if in_trade:
                    (stop_found, stop_j, eq_v, eq_t, cnt,
                     stop_m, stop_s, _new_cum) = _scan_1min_for_equity(
                        min1_cm, min1_cs, min1_lm, min1_ls, min1_times_i64,
                        l, r,
                        entry_main, entry_sub, entry_beta_val, entry_dir,
                        entry_frozen_alpha, entry_spread_val, entry_std_val,
                        SL_MULT, cum_realized, fee
                    )
                    _batch_append_eq(eq_t, eq_v, cnt)

                    if stop_found:
                        stop_time_dt = min1_times[stop_j]
                        _close_position(stop_time_dt, stop_m, stop_s, i,
                                        close_reason='hard_stop_1m')
                        _rebuild_future_signals_after_intrabar_stop(i)

            else:
                event_idx = r - 1
                stopped = False

                if in_trade:
                    (stop_found, stop_j, eq_v, eq_t, cnt,
                     stop_m, stop_s, _new_cum) = _scan_1min_for_equity(
                        min1_cm, min1_cs, min1_lm, min1_ls, min1_times_i64,
                        l, event_idx,
                        entry_main, entry_sub, entry_beta_val, entry_dir,
                        entry_frozen_alpha, entry_spread_val, entry_std_val,
                        SL_MULT, cum_realized, fee
                    )
                    _batch_append_eq(eq_t, eq_v, cnt)

                    if stop_found:
                        stop_time_dt = min1_times[stop_j]
                        _close_position(stop_time_dt, stop_m, stop_s, i,
                                        close_reason='hard_stop_1m')
                        _rebuild_future_signals_after_intrabar_stop(i)
                        stopped = True

                if not stopped:
                    cur_t = min1_times[event_idx]
                    cur_m = min1_cm[event_idx]
                    cur_s = min1_cs[event_idx]
                    cur_lm = min1_lm[event_idx]
                    cur_ls = min1_ls[event_idx]

                    if in_trade and sig == 0 and prev_sig != 0:
                        _cr = _determine_close_reason(i)
                        _close_position(cur_t, cur_m, cur_s, i, close_reason=_cr)
                        _append_eq_point(min1_times_i64[event_idx], cum_realized * 100.0)
                    elif sig != 0 and prev_sig == 0 and not in_trade:
                        _open_position(sig, cur_t, cur_m, cur_s, i)
                        _append_eq_point(min1_times_i64[event_idx], cum_realized * 100.0)
                    elif in_trade:
                        if _check_stop_single(cur_t, cur_m, cur_s, cur_lm, cur_ls, i):
                            _append_eq_point(min1_times_i64[event_idx], cum_realized * 100.0)
                        else:
                            unrealized = _calc_norm(entry_main, entry_sub, cur_m, cur_s,
                                                    entry_beta_val, entry_dir)
                            _append_eq_point(min1_times_i64[event_idx],
                                             (cum_realized + unrealized) * 100.0)

            last_mark_main = min1_cm[r - 1]
            last_mark_sub = min1_cs[r - 1]

        if _eq_pos == 0:
            seg_end_i64 = times_i64[-1] + bar_delta_ns_i64
            _append_eq_point(seg_end_i64, cum_realized * 100.0)

    else:
        for i in range(n):
            sig = int(signals[i])
            prev_sig = int(signals[i - 1]) if i > 0 else int(prev_sig_seed)

            exec_time_val, exec_m, exec_s, _, _ = _get_exec_point(i)

            if sig != 0 and prev_sig == 0 and not in_trade:
                _open_position(sig, exec_time_val, exec_m, exec_s, i)
            elif in_trade and sig == 0 and prev_sig != 0:
                _cr = _determine_close_reason(i)
                _close_position(exec_time_val, exec_m, exec_s, i, close_reason=_cr)

            if in_trade and entry_dir != 0:
                unrealized = _calc_norm(entry_main, entry_sub, close_main[i], close_sub[i],
                                        entry_beta_val, entry_dir)
                eq_val = (cum_realized + unrealized) * 100.0
            else:
                eq_val = cum_realized * 100.0

            _append_eq_point(times_i64[i] + bar_delta_ns_i64, eq_val)
            last_mark_main = close_main[i]
            last_mark_sub = close_sub[i]

    equity = _eq_vals_buf[:_eq_pos].copy()
    equity_times_arr = _eq_times_buf[:_eq_pos].copy().view('datetime64[ns]')

    last_signal_val = float(signals[-1]) if n > 0 else float(prev_sig_seed)
    final_exec_state = (
        entry_main, entry_sub, float(entry_dir), entry_beta_val, cum_realized,
        entry_time, last_mark_main, last_mark_sub,
        entry_frozen_alpha, entry_spread_val, entry_std_val,
        last_signal_val
    )

    if trades:
        trade_df = pd.DataFrame(trades)
        trade_df['direction'] = [
            f'做多价差 (Long BTC / Short {b:.3f} ETH)' if d == -1 else f'做空价差 (Short BTC / Long {b:.3f} ETH)'
            for d, b in zip(trade_df['entry_dir'], trade_df['entry_beta'])
        ]
        trade_df['btc_roi'] = [
            f"{((c - o) / o * 100):.2f}" if o else "0"
            for o, c in zip(trade_df['raw_open_main'], trade_df['raw_close_main'])
        ]
        trade_df['eth_roi'] = [
            f"{((c - o) / o * 100):.2f}" if o else "0"
            for o, c in zip(trade_df['raw_open_sub'], trade_df['raw_close_sub'])
        ]
        trade_df.drop(columns=[
            'entry_dir', 'raw_open_main', 'raw_open_sub', 'raw_close_main', 'raw_close_sub'
        ], inplace=True)
    else:
        trade_df = pd.DataFrame()

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
# 4b. 极速装配信号 DataFrame (为优化网格搜索专用)
# =============================================================================
def _assemble_signal_df_fast(base_df, log_main, log_sub, betas, alphas, kf_spreads,
                             ni_values, trace_P_values, ni_clean, trP_clean,
                             z_clean, rs_clean, z_score_raw, roll_std_raw,
                             z_lookback, z_entry, z_exit, min_hold_bars, cooldown_bars, max_hold_bars):
    """
    接受预先计算好的 KF 和 Rolling 数组，直接调用 JIT 函数并组装 DataFrame。
    完全去除了内部重复的 Pandas 开销。
    """
    df = base_df.copy()
    df['log_main'] = log_main
    df['log_sub'] = log_sub
    df['beta'] = betas
    df['alpha'] = alphas
    df['spread'] = kf_spreads
    df['kf_ni'] = ni_values
    df['kf_trace_P'] = trace_P_values
    df['z_score'] = z_score_raw
    df['roll_std'] = roll_std_raw

    # 默认初始状态 (与原始代码容错逻辑保持一致)
    pos_state = (0, 0, 9999, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    _ps = pos_state
    _psn = len(_ps)

    # 核心信号生成 (耗时极低)
    (raw_signals, frozen_betas, frozen_alphas_state,
     entry_spreads_state, entry_stds_state,
     stop_flags_raw, kf_kill_flags_raw,
     fp, fh, fsc, ffb, ffa, fes, fest,
     f_entry_trP, f_consec_ni) = fast_generate_signals(
        z_clean, kf_spreads, betas, alphas,
        log_main, log_sub, rs_clean,
        ni_clean, trP_clean,
        z_lookback, z_entry, z_exit,
        min_hold_bars, cooldown_bars, max_hold_bars,
        SL_MULT, SINGLE_LEG_FEE,
        NI_ENTRY_GATE, KF_KILL_TRACE_MULT,
        KF_KILL_NI_CONSEC, KF_KILL_NI_THRESHOLD,
        BETA_MIN, BETA_MAX,
        init_pos=int(_ps[0]), init_held=int(_ps[1]),
        init_since_close=int(_ps[2]),
        init_frozen_beta=float(_ps[3]), init_frozen_alpha=float(_ps[4]),
        init_entry_spread=float(_ps[5]), init_entry_std=float(_ps[6]),
        init_entry_trace_P=float(_ps[7]) if _psn > 7 else 0.0,
        init_consec_ni_count=int(_ps[8]) if _psn > 8 else 0
    )

    do_shift = USE_NEXT_BAR_EXEC
    df['signal'] = _shift_or_copy(raw_signals, do_shift)
    df['signal_frozen_beta'] = _shift_or_copy(frozen_betas, do_shift)
    df['signal_frozen_alpha'] = _shift_or_copy(frozen_alphas_state, do_shift)
    df['signal_entry_spread'] = _shift_or_copy(entry_spreads_state, do_shift)
    df['signal_entry_std'] = _shift_or_copy(entry_stds_state, do_shift)
    df['signal_entry_z'] = _shift_or_copy(z_score_raw, do_shift, fill=np.nan)
    df['stop_flag'] = _shift_or_copy(stop_flags_raw, do_shift)
    df['kf_kill_flag'] = _shift_or_copy(kf_kill_flags_raw, do_shift)

    # 写入上下文状态供撮合引擎读取
    _sig_ctx = {
        'z_entry': float(z_entry), 'z_exit': float(z_exit),
        'min_hold_bars': int(min_hold_bars), 'cooldown_bars': int(cooldown_bars),
        'max_hold_bars': int(max_hold_bars), 'signal_fee': float(SINGLE_LEG_FEE),
        'ni_entry_gate': float(NI_ENTRY_GATE),
        'kf_kill_trace_mult': float(KF_KILL_TRACE_MULT),
        'kf_kill_ni_consec': int(KF_KILL_NI_CONSEC),
        'kf_kill_ni_threshold': float(KF_KILL_NI_THRESHOLD),
    }
    df.attrs['_pt_signal_ctx'] = _sig_ctx
    for _k, _v in _sig_ctx.items():
        df[f'__pt_{_k}'] = _v

    return df


# =============================================================================
# 多进程全局变量（仅 Worker 进程使用，避免 IPC 传输大数组导致 OOM）
# =============================================================================
_GLOBAL_BASE_DF = None
_GLOBAL_MIN1_CACHE = None
_GLOBAL_LOG_MAIN = None
_GLOBAL_LOG_SUB = None


def _init_worker(base_df, min1_cache, log_main_arr, log_sub_arr):
    """初始化 Worker 进程的全局共享只读内存"""
    global _GLOBAL_BASE_DF, _GLOBAL_MIN1_CACHE, _GLOBAL_LOG_MAIN, _GLOBAL_LOG_SUB
    _GLOBAL_BASE_DF = base_df
    _GLOBAL_MIN1_CACHE = min1_cache
    _GLOBAL_LOG_MAIN = log_main_arr
    _GLOBAL_LOG_SUB = log_sub_arr


def _eval_group_worker(args):
    """
    按 (dpd, ve, lb_h) 聚类执行。
    Worker 自己算一次 KF 和 Rolling，然后在内部极速跑完 z_entry/z_exit 组合。
    """
    dpd, ve, lb_h, delta, tf_min, cooldown_bars, max_hold_bars, param_combinations = args
    pid = os.getpid()
    total_combos = len(param_combinations)

    # 打印任务包领受日志
    print(f"[Worker {pid}] 📥 收到任务: dpd={dpd}, ve={ve}, lb_h={lb_h} | 共 {total_combos} 个参数组合")

    try:
        global _GLOBAL_BASE_DF, _GLOBAL_MIN1_CACHE, _GLOBAL_LOG_MAIN, _GLOBAL_LOG_SUB

        # 1. 在 Worker 内部仅执行 1 次 Kalman Filter
        t0 = time.time()
        kf_state = (0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
        (betas, alphas, kf_spreads, ni_values, trace_P_values,
         _, _, _, _, _, _) = fast_kalman_filter(
            _GLOBAL_LOG_SUB, _GLOBAL_LOG_MAIN,
            delta=delta, ve=ve,
            init_alpha=kf_state[0], init_beta=kf_state[1],
            init_P00=kf_state[2], init_P01=kf_state[3],
            init_P10=kf_state[4], init_P11=kf_state[5]
        )
        print(f"[Worker {pid}] ⚡ KF滤波完成，耗时: {time.time() - t0:.2f}s")

        ni_clean = np.nan_to_num(ni_values, nan=0.0, posinf=0.0, neginf=0.0)
        trP_clean = np.nan_to_num(trace_P_values, nan=0.0, posinf=0.0, neginf=0.0)
        spread_s = pd.Series(kf_spreads)

        z_lb = hours_to_bars(lb_h, tf_min)

        # 2. 在 Worker 内部仅执行 1 次 Rolling
        t1 = time.time()
        roll_mean = spread_s.rolling(z_lb, min_periods=z_lb).mean()
        roll_std = spread_s.rolling(z_lb, min_periods=z_lb).std()
        z_score_raw = ((spread_s - roll_mean) / roll_std).values
        roll_std_raw = roll_std.values

        z_clean = np.nan_to_num(z_score_raw, nan=0.0, posinf=0.0, neginf=0.0)
        rs_clean = np.nan_to_num(roll_std_raw, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"[Worker {pid}] 🌊 Rolling统计完成，耗时: {time.time() - t1:.2f}s")

        group_results = []

        # 3. 极速遍历该类别下的所有参数
        print(f"[Worker {pid}] 🚀 开始遍历 {total_combos} 个参数组合...")

        for idx, (z_entry, z_exit, mh_h) in enumerate(param_combinations):
            t_combo_start = time.time()
            min_hold = hours_to_bars(mh_h, tf_min)

            # 监控：组装 DF 耗时
            t2 = time.time()
            full_df = _assemble_signal_df_fast(
                _GLOBAL_BASE_DF, _GLOBAL_LOG_MAIN, _GLOBAL_LOG_SUB,
                betas, alphas, kf_spreads,
                ni_values, trace_P_values, ni_clean, trP_clean,
                z_clean, rs_clean, z_score_raw, roll_std_raw,
                z_lb, z_entry, z_exit,
                min_hold, cooldown_bars, max_hold_bars
            )
            df_time = time.time() - t2

            # 监控：0手续费回测耗时
            t3 = time.time()
            trades_0, eq_0, _, _, eff_0 = _backtest_and_pnls(
                full_df, _GLOBAL_MIN1_CACHE, tf_min, fee=0.0)
            bt0_time = time.time() - t3

            # 监控：真实手续费回测耗时
            t4 = time.time()
            trades_r, eq_r, _, _, eff_r = _backtest_and_pnls(
                full_df, _GLOBAL_MIN1_CACHE, tf_min, fee=SINGLE_LEG_FEE)
            btr_time = time.time() - t4

            combo_total_time = time.time() - t_combo_start

            # 打印每个参数组合的明细耗时，让你知道大头在哪里
            print(f"[Worker {pid}]   [{idx + 1}/{total_combos}] z_e={z_entry}, z_x={z_exit} | "
                  f"总耗时:{combo_total_time:.2f}s (组装DF:{df_time:.2f}s, 测0费:{bt0_time:.2f}s, 测实费:{btr_time:.2f}s)")

            if len(eff_0) < 15:
                continue

            stats_0 = print_backtest_stats(trades_0, eq_0, effective_pnls=eff_0)
            stats_r = print_backtest_stats(trades_r, eq_r, effective_pnls=eff_r)

            fee_per_trade = stats_0['avg_profit_per_trade'] - stats_r['avg_profit_per_trade']

            group_results.append({
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

        print(f"[Worker {pid}] ✅ 任务包处理完毕，产出 {len(group_results)} 条有效结果。")
        return group_results

    except Exception as e:
        print(f"[Worker {pid}] ❌ 发生异常: {e}")
        traceback.print_exc()
        return []

def parameter_sensitivity_scan(df_file, timeframe='15min', n_workers=1):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

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
    cooldown_bars = hours_to_bars(1.0, tf_min)
    max_hold_bars = days_to_bars(4.0, tf_min)

    log_main_arr = np.log(df['close_main'].values.astype(float))
    log_sub_arr = np.log(df['close_sub'].values.astype(float))

    # 构建轻量级 DataFrame 用于初始化 Worker
    base_df = df[['open_time', 'close_main', 'close_sub']].copy()

    # ==========================================
    # 修改点 1：将 z_entry 提到任务包级别，细化任务粒度
    # ==========================================
    grouped_tasks = []
    for dpd in grids['delta_per_day']:
        delta = delta_per_day_to_bar(dpd, tf_min)
        for ve in grids['ve']:
            for lb_h in grids['lookback_hours']:
                for z_entry in grids['z_entry']:
                    fast_params = []
                    for z_exit in grids['z_exit']:
                        if z_exit >= z_entry: continue
                        for mh_h in grids['min_hold_hours']:
                            fast_params.append((z_entry, z_exit, mh_h))

                    if fast_params:
                        grouped_tasks.append((
                            dpd, ve, lb_h, delta, tf_min, cooldown_bars, max_hold_bars, fast_params
                        ))

    print(f"将组合划分为 {len(grouped_tasks)} 个任务包分发给 {n_workers} 个 Worker 进程...")

    # ==========================================
    # 修改点 2：主进程预热 Numba JIT 编译器，防止多进程缓存锁死
    # ==========================================
    print("正在预热 Numba JIT 编译器，防止多进程缓存冲突锁死...")
    _dummy_f = np.zeros(5, dtype=np.float64)
    _dummy_i = np.zeros(5, dtype=np.int64)
    try:
        fast_kalman_filter(_dummy_f, _dummy_f, 0.001, 0.001)
        fast_generate_signals(_dummy_f, _dummy_f, _dummy_f, _dummy_f, _dummy_f, _dummy_f, _dummy_f, _dummy_f, _dummy_f,
                              2, 2.0, 0.5, 1, 1, 1, 1.0, 0.001, 3.0, 5.0, 3, 3.0)
        _scan_1min_for_equity(_dummy_f, _dummy_f, _dummy_f, _dummy_f, _dummy_i, 0, 2, 1.0, 1.0, 1.0, 1, 0.0, 0.0, 0.0,
                              1.0, 0.0, 0.001)
    except Exception:
        pass
    print("预热完成，开始并行计算！")

    results = []
    start_time = time.time()

    # 使用 initializer 优雅地将巨型只读数据仅在进程创建时传输 1 次
    with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(base_df, min1_cache, log_main_arr, log_sub_arr)
    ) as executor:
        futures = {executor.submit(_eval_group_worker, task): task for task in grouped_tasks}

        for future in as_completed(futures):
            res_list = future.result()
            if res_list:
                results.extend(res_list)

            # 打印进度提示
            elapsed = time.time() - start_time
            print(f"  已扫描完成有效策略数: {len(results)} | 耗时: {elapsed:.1f}s", end='\r')

    elapsed = time.time() - start_time
    print(f"\n扫描完成: 共获取 {len(results)} 条有效结果, 总耗时 {elapsed:.1f}s")

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
# 9. Walk-Forward（含子样本稳定性评分 + 多样化集成报告）
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
    diverse_ensemble_history = []

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

        # --- 针对当前窗口，提前计算一次时间段和基础数组 ---
        train_start_ns, train_end_ns = _get_segment_bounds_ns(train_df, tf_min)
        log_main_arr = np.log(train_df['close_main'].values.astype(float))
        log_sub_arr = np.log(train_df['close_sub'].values.astype(float))
        base_train_df = train_df[['open_time', 'close_main', 'close_sub']].copy()

        # --- 重排循环拓扑，降维打击 ---
        for dpd in grids['delta_per_day']:
            delta = delta_per_day_to_bar(dpd, tf_min)
            for ve in grids['ve']:
                # 仅执行一次 KF
                kf_state = (0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
                (betas, alphas, kf_spreads, ni_values, trace_P_values,
                 _, _, _, _, _, _) = fast_kalman_filter(
                    log_sub_arr, log_main_arr, delta=delta, ve=ve,
                    init_alpha=kf_state[0], init_beta=kf_state[1],
                    init_P00=kf_state[2], init_P01=kf_state[3],
                    init_P10=kf_state[4], init_P11=kf_state[5]
                )

                ni_clean = np.nan_to_num(ni_values, nan=0.0, posinf=0.0, neginf=0.0)
                trP_clean = np.nan_to_num(trace_P_values, nan=0.0, posinf=0.0, neginf=0.0)
                spread_s = pd.Series(kf_spreads)

                for lb_h in grids['lookback_hours']:
                    z_lb = hours_to_bars(lb_h, tf_min)

                    # 仅执行一次 Pandas Rolling
                    roll_mean = spread_s.rolling(z_lb, min_periods=z_lb).mean()
                    roll_std = spread_s.rolling(z_lb, min_periods=z_lb).std()
                    z_score_raw = ((spread_s - roll_mean) / roll_std).values
                    roll_std_raw = roll_std.values

                    z_clean = np.nan_to_num(z_score_raw, nan=0.0, posinf=0.0, neginf=0.0)
                    rs_clean = np.nan_to_num(roll_std_raw, nan=0.0, posinf=0.0, neginf=0.0)

                    for z_entry in grids['z_entry']:
                        for z_exit in grids['z_exit']:
                            if z_exit >= z_entry:
                                continue
                            for mh_h in grids['min_hold_hours']:
                                min_hold = hours_to_bars(mh_h, tf_min)
                                combo_count += 1
                                param_key = (z_entry, z_exit, lb_h, dpd, ve, mh_h)
                                try:
                                    full_df = _assemble_signal_df_fast(
                                        base_train_df, log_main_arr, log_sub_arr,
                                        betas, alphas, kf_spreads,
                                        ni_values, trace_P_values, ni_clean, trP_clean,
                                        z_clean, rs_clean, z_score_raw, roll_std_raw,
                                        z_lb, z_entry, z_exit,
                                        min_hold, cooldown_bars, max_hold_bars
                                    )

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

                                    monthly_rets = _split_train_monthly_returns(
                                        train_start_ns, train_end_ns, eq_times, eq_curve, train_months)

                                    if len(monthly_rets) < 3:
                                        continue

                                    profitable_months = sum(1 for r in monthly_rets if r > 0)
                                    required_profitable = max(2, int(len(monthly_rets) * 0.6))
                                    if profitable_months < required_profitable:
                                        continue

                                    monthly_arr = np.array(monthly_rets)
                                    m_mean = np.mean(monthly_arr)
                                    m_std = np.std(monthly_arr)
                                    if m_std > 0:
                                        stability_score = m_mean / m_std
                                    else:
                                        stability_score = np.sign(m_mean) * 10.0 if m_mean > 0 else 0.0

                                    composite_score = raw_score * (1.0 + max(0.0, stability_score)) / 2.0

                                    if len(eq_curve) > 0:
                                        _eq_ext = np.concatenate(([0.0], eq_curve))
                                        _peak = np.maximum.accumulate(_eq_ext)
                                        _mdd = float(np.min(_eq_ext - _peak))
                                        if _mdd < WF_MAX_TRAIN_MDD:
                                            continue

                                    valid_count += 1
                                    train_total_ret = float(eq_curve[-1]) if len(eq_curve) > 0 else float(
                                        np.sum(eff_pnls))

                                    all_candidates[param_key] = {
                                        'score': composite_score,
                                        'raw_t_stat': raw_score,
                                        'stability_score': stability_score,
                                        'profitable_months': profitable_months,
                                        'total_months': len(monthly_rets),
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
            'raw_t_stat': chosen['raw_t_stat'],
            'stability_score': round(chosen['stability_score'], 4),
            'profitable_months': chosen['profitable_months'],
            'total_months': chosen['total_months'],
            'plateau_score': best_plateau_score,
            'train_trades': chosen['train_trades'],
            'train_closed_trades': chosen['train_closed_trades'],
            'train_total_ret': round(chosen['train_ret'], 4),
            'train_avg_pnl': round(chosen['train_avg_pnl'], 4),
            'combos_evaluated': combo_count,
            'combos_passed_filter': len(all_candidates),
        })
        print(f"  Best (plateau): {best_params}")
        print(f"  Train: t-stat={chosen['raw_t_stat']:.4f}, "
              f"stability={chosen['stability_score']:.4f} "
              f"({chosen['profitable_months']}/{chosen['total_months']}mo+), "
              f"composite={chosen['score']:.4f}, "
              f"trades={chosen['train_trades']}, "
              f"closed={chosen['train_closed_trades']}, ret={chosen['train_ret']:.2f}%, "
              f"plateau={best_plateau_score:.4f} | passed: {len(all_candidates)}/{combo_count}")

        # ===== 多样化集成候选 (方案3 - 报告层) =====
        lookback_groups = {}
        for pkey, info in all_candidates.items():
            lb = info['params']['lookback_hours']
            if lb not in lookback_groups:
                lookback_groups[lb] = []
            lookback_groups[lb].append((pkey, info))

        diverse_ensemble = []
        for lb in sorted(lookback_groups.keys()):
            group = lookback_groups[lb]
            group.sort(key=lambda x: x[1]['score'], reverse=True)
            if group:
                diverse_ensemble.append(group[0])

        if len(diverse_ensemble) >= 2:
            print(f"  --- Diverse Ensemble Candidates (by lookback, for live ref) ---")
            ensemble_info = []
            for rank, (pkey, info) in enumerate(diverse_ensemble[:5]):
                p = info['params']
                print(f"    #{rank + 1}: lb_h={p['lookback_hours']}, "
                      f"z_e={p['z_entry']}, z_x={p['z_exit']}, "
                      f"dpd={p['delta_per_day']}, mh={p['min_hold_hours']}, "
                      f"score={info['score']:.4f}, "
                      f"stab={info['stability_score']:.2f}, "
                      f"ret={info['train_ret']:.2f}%")
                ensemble_info.append({
                    'lookback_hours': p['lookback_hours'],
                    'z_entry': p['z_entry'], 'z_exit': p['z_exit'],
                    'delta_per_day': p['delta_per_day'],
                    'min_hold_hours': p['min_hold_hours'],
                    'score': round(info['score'], 4),
                    'stability': round(info['stability_score'], 4),
                })
            diverse_ensemble_history.append({
                'window_id': window_id,
                'candidates': ensemble_info,
            })

        # ===== OOS 测试 =====
        combined = pd.concat([train_df, test_df], ignore_index=True)
        test_start_time = test_df['open_time'].iloc[0]

        # 注意：这里的 OOS 测试仍然保留了原始逻辑，因为它每个窗口只调用1次，完全不影响速度，最大程度降低修改风险。
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

        if diverse_ensemble_history:
            print(f"\n  --- 多样化集成参数汇总 (供实盘参考) ---")
            print(f"  各窗口按lookback_hours分组的Top候选，实盘可用共识投票机制:")
            print(f"  (强共识→全仓, 分歧→空仓)")
            for entry in diverse_ensemble_history:
                wid = entry['window_id']
                cands = entry['candidates']
                lbs = [c['lookback_hours'] for c in cands]
                print(f"  W{wid}: {len(cands)}组 lookbacks={lbs}")

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

        if 'stability_score' in ph.columns:
            print(f"\n  --- 稳定性评分统计 ---")
            print(f"  stability_score: mean={ph['stability_score'].mean():.4f}, "
                  f"min={ph['stability_score'].min():.4f}, max={ph['stability_score'].max():.4f}")
            if 'profitable_months' in ph.columns and 'total_months' in ph.columns:
                avg_pct = (ph['profitable_months'] / ph['total_months']).mean()
                print(f"  平均月盈利率: {avg_pct:.1%}")


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