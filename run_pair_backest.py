import re
import traceback
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import numba as nb
from itertools import product

# =============================================================================
# 全局配置
# =============================================================================
SINGLE_LEG_FEE = 0.0006  # 单腿单次费率（含滑点）
USE_NEXT_BAR_EXEC = False  # True: 用信号下一根1min close成交; False: 用下一根重采样bar close
USE_1MIN_PATH = True  # True: 持仓期间MTM/硬止损走1min路径（若提供原始1min数据）
SL_MULT = 3.0  # spread-based 硬止损倍数（固定，不参与扫描）
BETA_MIN = 0.1   # [FIX-2] Beta合法下限：低于此值关系不稳定，禁止开仓
BETA_MAX = 5.0   # [FIX-2] Beta合法上限：超过此值杠杆失控，禁止开仓
WF_MAX_TRAIN_MDD = -15.0  # [FIX-3] WF训练集最大允许路径回撤(%)，超过则淘汰该参数


# =============================================================================
# 0. 时间单位工具
# =============================================================================
def get_tf_minutes(tf_str):
    """从timeframe字符串获取分钟数"""
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
    bars_per_day = 1440.0 / tf_minutes
    return delta_per_day / bars_per_day


def _get_segment_bounds_ns(df, tf_minutes):
    """返回一个重采样窗口的真实起止时间 [start, end)"""
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
# 3. 信号生成 V2（冻结beta、硬止损、成本感知开仓、价差改善平仓）
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
    stop_flags = np.zeros(n)  # [FIX-5] 标记每根bar是否因硬止损而平仓

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
                # [FIX-2] Beta合法区间检查：beta过小或过大时不开仓
                if cur_std > 0.0 and cur_beta >= beta_min and cur_beta <= beta_max:
                    # 成本感知：预期利润必须覆盖双边手续费
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
            # 用冻结的 alpha/beta 计算当前价差
            frozen_spread = log_main[i] - frozen_alpha - frozen_beta * log_sub[i]

            # --- bar级硬止损（execution层还会再做1min止损）---
            stopped = False
            if entry_std > 0.0 and sl_mult > 0.0:
                if pos == 1:  # 做空价差 → 价差上涨亏钱
                    if (frozen_spread - entry_spread) > sl_mult * entry_std:
                        stopped = True
                elif pos == -1:  # 做多价差 → 价差下跌亏钱
                    if (entry_spread - frozen_spread) > sl_mult * entry_std:
                        stopped = True

            if stopped:
                pos = 0
                since_close = 0
                stop_flags[i] = 1.0  # [FIX-5] 标记此bar为硬止损触发
            elif max_hold_bars > 0 and held >= max_hold_bars:
                pos = 0
                since_close = 0
            elif held >= min_hold_bars:
                # [FIX] 方向性判断：价差是否朝盈利方向移动
                # pos=1 做空价差→利润来自价差下降; pos=-1 做多价差→利润来自价差上升
                # 原 abs() 比较在价差穿越零点后会误判，改为方向性比较
                if pos == 1:
                    spread_improved = frozen_spread < entry_spread
                else:
                    spread_improved = frozen_spread > entry_spread

                # ★ FIX: spread_improved 时间衰减
                # 持仓超过 max_hold_bars 一半后，放弃 spread_improved 要求，
                # 避免仓位被困在"z已达标但spread未改善"的灰色地带
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

    # [FIX-5] 返回值新增 stop_flags
    return (signals, frozen_betas_arr, frozen_alphas_arr,
            entry_spreads_arr, entry_stds_arr,
            stop_flags,
            pos, held, since_close,
            frozen_beta, frozen_alpha, entry_spread, entry_std)


# =============================================================================
# 4. 信号管道（组装 KF + 信号生成）
# =============================================================================
def generate_pair_trading_signals(merged_df, main_col='close_main', sub_col='close_sub',
                                  z_lookback=60, z_entry=2.0, z_exit=0.5,
                                  delta=1e-5, ve=1e-3,
                                  min_hold_bars=12, cooldown_bars=4, max_hold_bars=0,
                                  kf_state=None, pos_state=None):
    """
    返回 df（含 raw_signal / signal / frozen_beta 等列），
    以及 final_kf_state, final_pos_state 用于跨窗口传递。
    """
    df = merged_df.copy()
    df['log_main'] = np.log(df[main_col])
    df['log_sub'] = np.log(df[sub_col])

    # --- KF ---
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

    # ★ FIX: inf 处理 — 防止 roll_std 极小时 z_score 变为巨大有限数触发幽灵交易
    z_clean = np.nan_to_num(z_score, nan=0.0, posinf=0.0, neginf=0.0)
    rs_clean = np.nan_to_num(roll_std.values, nan=0.0, posinf=0.0, neginf=0.0)

    # --- 信号 ---
    if pos_state is None:
        pos_state = (0, 0, 9999, 0.0, 0.0, 0.0, 0.0)

    # [FIX-2] 传入 BETA_MIN, BETA_MAX；[FIX-5] 接收新增的 stop_flags_raw
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

    df['raw_signal'] = raw_signals
    df['frozen_beta'] = frozen_betas
    df['frozen_alpha_state'] = frozen_alphas_state
    df['entry_spread_state'] = entry_spreads_state
    df['entry_std_state'] = entry_stds_state

    # [FIX 1] 移位：信号与元数据统一受 USE_NEXT_BAR_EXEC 控制
    if USE_NEXT_BAR_EXEC:
        # shift：信号 bar k → 交易检测在 bar k+1 → 元数据也取 bar k 的值
        shifted = np.zeros_like(raw_signals)
        shifted[1:] = raw_signals[:-1]
        df['signal'] = shifted

        shifted_fb = np.zeros_like(frozen_betas)
        shifted_fb[1:] = frozen_betas[:-1]
        df['signal_frozen_beta'] = shifted_fb

        shifted_fa = np.zeros_like(frozen_alphas_state)
        shifted_fa[1:] = frozen_alphas_state[:-1]
        df['signal_frozen_alpha'] = shifted_fa

        shifted_es = np.zeros_like(entry_spreads_state)
        shifted_es[1:] = entry_spreads_state[:-1]
        df['signal_entry_spread'] = shifted_es

        shifted_estd = np.zeros_like(entry_stds_state)
        shifted_estd[1:] = entry_stds_state[:-1]
        df['signal_entry_std'] = shifted_estd

        shifted_entry_z = np.full(len(df), np.nan)
        if len(df) > 1:
            shifted_entry_z[1:] = z_score[:-1]
        df['signal_entry_z'] = shifted_entry_z

        # [FIX-5] stop_flag 也需要同步 shift
        shifted_sf = np.zeros_like(stop_flags_raw)
        shifted_sf[1:] = stop_flags_raw[:-1]
        df['stop_flag'] = shifted_sf
    else:
        # 不 shift：信号 bar k → 交易检测就在 bar k → 元数据也取 bar k 的值
        df['signal'] = raw_signals.copy()
        df['signal_frozen_beta'] = frozen_betas.copy()
        df['signal_frozen_alpha'] = frozen_alphas_state.copy()
        df['signal_entry_spread'] = entry_spreads_state.copy()
        df['signal_entry_std'] = entry_stds_state.copy()
        df['signal_entry_z'] = z_score.copy()
        df['stop_flag'] = stop_flags_raw.copy()  # [FIX-5]

    return df, final_kf_state, final_pos_state


# =============================================================================
# 5. 1min cache / 有效PnL / 时间三段
# =============================================================================
def _build_1min_cache(original_1min_df):
    """构建1min价格/对数价格缓存，避免扫描时重复构建"""
    odf = original_1min_df.copy()
    odf['open_time'] = pd.to_datetime(odf['open_time'])
    odf = odf.sort_values('open_time').reset_index(drop=True)

    cm = odf['close_main'].values.astype(float)
    cs = odf['close_sub'].values.astype(float)
    return {
        'times': odf['open_time'].values.astype('datetime64[ns]'),
        'close_main': cm,
        'close_sub': cs,
        'log_main': np.log(cm),
        'log_sub': np.log(cs),
    }


def _lookup_1min_point(min1_cache, target_time_ns):
    """查找 >= target_time 的第一根1min close"""
    min1_times = min1_cache['times']
    idx = np.searchsorted(min1_times, target_time_ns, side='left')
    if idx >= len(min1_times):
        idx = len(min1_times) - 1
    return (min1_times[idx],
            min1_cache['close_main'][idx],
            min1_cache['close_sub'][idx],
            min1_cache['log_main'][idx],
            min1_cache['log_sub'][idx])


def _build_effective_pnls(trade_df, equity_curve, final_exec_state):
    """
    构造用于Train打分 / 硬过滤 / OOS退化比的"有效PnL单元"：
    - 已平仓交易的 net_pnl
    - 若窗口末仍有未平仓，则把窗口末MTM（不扣出场费）作为一个额外单元
    """
    if trade_df is not None and not trade_df.empty:
        pnls = trade_df['net_pnl'].values.astype(float).copy()
    else:
        pnls = np.array([], dtype=float)

    has_open = False
    if final_exec_state is not None and len(final_exec_state) > 2:
        has_open = (int(final_exec_state[2]) != 0)

    if has_open and equity_curve is not None and len(equity_curve) > 0:
        closed_sum = float(np.sum(pnls)) if len(pnls) > 0 else 0.0
        residual = float(equity_curve[-1]) - closed_sum
        pnls = np.append(pnls, residual)

    return pnls


def _split_train_thirds(segment_start_time, segment_end_time, equity_times, equity_curve):
    """
    按真实时间跨度把训练窗切成3段，返回每段收益（使用权益曲线增量）
    """
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

    # [FIX 2] 添加上界防御，防止极端情况下索引越界
    def _eq_at(boundary_ns):
        idx = np.searchsorted(etimes, boundary_ns, side='right') - 1
        if idx < 0:
            return 0.0
        if idx >= len(eq):
            idx = len(eq) - 1
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
    """
    计算权益曲线（支持1min path MTM）并提取交易列表。

    exec_state:
        (
            entry_main, entry_sub, entry_dir, entry_beta_val, cum_realized,
            entry_time, last_mark_main, last_mark_sub,
            entry_frozen_alpha, entry_spread_val, entry_std_val,
            last_signal_val
        )

    返回:
        trade_df, equity_array, equity_times_array, final_exec_state
    """
    fee = SINGLE_LEG_FEE if fee_override is None else fee_override

    df = full_df.reset_index(drop=True)
    if 'open_time' not in df.columns:
        if 'index' in df.columns:
            df.rename(columns={'index': 'open_time'}, inplace=True)
        elif 'timestamp' in df.columns:
            df.rename(columns={'timestamp': 'open_time'}, inplace=True)

    n = len(df)

    if n == 0:
        if exec_state is not None:
            entry_main = float(exec_state[0])
            entry_sub = float(exec_state[1])
            entry_dir = int(exec_state[2])
            entry_beta_val = float(exec_state[3])
            cum_realized = float(exec_state[4])
            entry_time = exec_state[5] if len(exec_state) > 5 else np.datetime64('NaT')
            last_mark_main = float(exec_state[6]) if len(exec_state) > 6 else 0.0
            last_mark_sub = float(exec_state[7]) if len(exec_state) > 7 else 0.0
            entry_frozen_alpha = float(exec_state[8]) if len(exec_state) > 8 else 0.0
            entry_spread_val = float(exec_state[9]) if len(exec_state) > 9 else 0.0
            entry_std_val = float(exec_state[10]) if len(exec_state) > 10 else 0.0
            last_signal_val = float(exec_state[11]) if len(exec_state) > 11 else float(entry_dir)
            final_exec_state = (
                entry_main, entry_sub, float(entry_dir), entry_beta_val, cum_realized,
                entry_time, last_mark_main, last_mark_sub,
                entry_frozen_alpha, entry_spread_val, entry_std_val,
                last_signal_val
            )
        else:
            final_exec_state = (
                0.0, 0.0, 0.0, 0.0, 0.0,
                np.datetime64('NaT'), 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0
            )
        return pd.DataFrame(), np.array([]), np.array([], dtype='datetime64[ns]'), final_exec_state

    signals = df['signal'].values.astype(int)
    times = pd.to_datetime(df['open_time']).values.astype('datetime64[ns]')
    close_main = df['close_main'].values.astype(float)
    close_sub = df['close_sub'].values.astype(float)
    z_scores = df['z_score'].values.astype(float) if 'z_score' in df.columns else np.full(n, np.nan)
    spreads_arr = df['spread'].values.astype(float) if 'spread' in df.columns else np.full(n, np.nan)
    betas_arr = df['beta'].values.astype(float) if 'beta' in df.columns else np.zeros(n)
    alphas_arr = df['alpha'].values.astype(float) if 'alpha' in df.columns else np.zeros(n)

    signal_fb = df['signal_frozen_beta'].values.astype(float) if 'signal_frozen_beta' in df.columns else betas_arr.copy()
    signal_fa = df['signal_frozen_alpha'].values.astype(float) if 'signal_frozen_alpha' in df.columns else alphas_arr.copy()
    signal_es = df['signal_entry_spread'].values.astype(float) if 'signal_entry_spread' in df.columns else np.zeros(n)
    signal_estd = df['signal_entry_std'].values.astype(float) if 'signal_entry_std' in df.columns else np.zeros(n)
    signal_entry_z = df['signal_entry_z'].values.astype(float) if 'signal_entry_z' in df.columns else np.full(n, np.nan)

    # [FIX-5] 读取止损标记列
    stop_flags_col = df['stop_flag'].values.astype(int) if 'stop_flag' in df.columns else np.zeros(n, dtype=int)

    # 1min缓存
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

    # --- 初始化真实持仓状态 ---
    if exec_state is not None:
        entry_main = float(exec_state[0])
        entry_sub = float(exec_state[1])
        entry_dir = int(exec_state[2])
        entry_beta_val = float(exec_state[3])
        cum_realized = float(exec_state[4])
        entry_time = exec_state[5] if len(exec_state) > 5 else np.datetime64('NaT')
        prev_last_mark_main = float(exec_state[6]) if len(exec_state) > 6 else 0.0
        prev_last_mark_sub = float(exec_state[7]) if len(exec_state) > 7 else 0.0
        entry_frozen_alpha = float(exec_state[8]) if len(exec_state) > 8 else 0.0
        entry_spread_val = float(exec_state[9]) if len(exec_state) > 9 else 0.0
        entry_std_val = float(exec_state[10]) if len(exec_state) > 10 else 0.0
        prev_sig_seed = int(exec_state[11]) if len(exec_state) > 11 else entry_dir
        in_trade = (entry_dir != 0)
    else:
        entry_main, entry_sub, entry_dir = 0.0, 0.0, 0
        entry_beta_val, cum_realized = 0.0, 0.0
        entry_time = np.datetime64('NaT')
        prev_last_mark_main, prev_last_mark_sub = 0.0, 0.0
        entry_frozen_alpha = 0.0
        entry_spread_val = 0.0
        entry_std_val = 0.0
        prev_sig_seed = 0
        in_trade = False

    # --- 本段收益归因状态 ---
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

    def _calc_gross(em, es, cm, cs, ebeta, edir):
        br = (cm - em) / em if em != 0 else 0.0
        er = (cs - es) / es if es != 0 else 0.0
        if edir == -1:
            return br - ebeta * er
        else:
            return ebeta * er - br

    def _calc_norm(em, es, cm, cs, ebeta, edir):
        total_exp = 1.0 + abs(ebeta)
        if total_exp <= 0.0:
            return 0.0
        return _calc_gross(em, es, cm, cs, ebeta, edir) / total_exp

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

        # 入场扣一次费
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

        # live真实持仓：到这次平仓点只扣出场费
        realized_add = _calc_norm(entry_main, entry_sub, exec_m, exec_s,
                                  entry_beta_val, entry_dir) - fee
        cum_realized += realized_add

        # 本段归因收益：
        # - 新开仓：gross - entry_fee - exit_fee
        # - 继承仓位：boundary->close - exit_fee
        seg_net = (_calc_norm(seg_entry_main, seg_entry_sub, exec_m, exec_s,
                              entry_beta_val, entry_dir)
                   - fee - seg_entry_fee)

        direction_str = (f'做多价差 (Long BTC / Short {entry_beta_val:.3f} ETH)'
                         if entry_dir == -1
                         else f'做空价差 (Short BTC / Long {entry_beta_val:.3f} ETH)')

        if close_reason == 'hard_stop_1m':
            exit_z = np.nan
            exit_spread = np.nan
            exit_alpha = entry_frozen_alpha
            exit_beta = entry_beta_val
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
            'entry_beta': entry_beta_val,
            'exit_beta': exit_beta,
            'entry_alpha': seg_entry_alpha_meta if not seg_inherited else np.nan,
            'exit_alpha': exit_alpha,
            'signal': entry_dir,
            'inherited_from_prev_window': seg_inherited,
            'close_reason': close_reason,
        })

        # reset
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
            return True
        return False

    def _get_exec_point(bar_idx):
        if use_1m_exec:
            return _lookup_1min_point(min1_cache, times[bar_idx])
        return (times[bar_idx], close_main[bar_idx], close_sub[bar_idx],
                np.log(close_main[bar_idx]), np.log(close_sub[bar_idx]))

    # =========================================================================
    # A. 严格1min路径
    # =========================================================================
    if use_1m_path:
        for i in range(n):
            sig = int(signals[i])
            prev_sig = int(signals[i - 1]) if i > 0 else int(prev_sig_seed)

            start_ns = times[i]
            end_ns = times[i + 1] if i < n - 1 else (times[i] + bar_delta_ns)

            l = np.searchsorted(min1_times, start_ns, side='left')
            r = np.searchsorted(min1_times, end_ns, side='left')

            # 极少数缺口情况 fallback 到bar级
            if r <= l:
                exec_time_val, exec_m, exec_s, _, _ = _get_exec_point(i)

                if use_1m_exec:
                    if sig != 0 and prev_sig == 0 and not in_trade:
                        _open_position(sig, exec_time_val, exec_m, exec_s, i)
                    elif in_trade and sig == 0 and prev_sig != 0:
                        # [FIX-5] 根据stop_flag区分平仓原因
                        _cr = 'hard_stop' if stop_flags_col[i] == 1 else 'signal'
                        _close_position(exec_time_val, exec_m, exec_s, i, close_reason=_cr)
                else:
                    if in_trade and sig == 0 and prev_sig != 0:
                        # [FIX-5] 根据stop_flag区分平仓原因
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

            # -------------------------
            # 执行在bar开始（下一根1min close）
            # -------------------------
            if use_1m_exec:
                start_t = min1_times[l]
                start_m = min1_cm[l]
                start_s = min1_cs[l]

                if sig != 0 and prev_sig == 0 and not in_trade:
                    _open_position(sig, start_t, start_m, start_s, i)
                elif in_trade and sig == 0 and prev_sig != 0:
                    # [FIX-5] 根据stop_flag区分平仓原因
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

            # -------------------------
            # 执行在bar结束（下一根重采样bar close）
            # -------------------------
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
                        # [FIX-5] 根据stop_flag区分平仓原因
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

        # 若整段完全没点（例如全程空仓），补一个期末平坦点
        if len(equity_values) == 0:
            _append_equity_point(segment_end_time, cum_realized * 100.0)

    # =========================================================================
    # B. 非1min路径：保留bar级
    # =========================================================================
    else:
        for i in range(n):
            sig = int(signals[i])
            prev_sig = int(signals[i - 1]) if i > 0 else int(prev_sig_seed)

            exec_time_val, exec_m, exec_s, _, _ = _get_exec_point(i)

            if sig != 0 and prev_sig == 0 and not in_trade:
                _open_position(sig, exec_time_val, exec_m, exec_s, i)
            elif in_trade and sig == 0 and prev_sig != 0:
                # [FIX-5] 根据stop_flag区分平仓原因（Path B核心修正）
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
# 7. 统计函数（优先使用逐bar/逐1min权益）
# =============================================================================
def print_backtest_stats(trade_df, equity_curve=None, effective_pnls=None):
    has_trade_rows = trade_df is not None and not trade_df.empty

    if effective_pnls is not None:
        pnl_arr = np.asarray(effective_pnls, dtype=float)
        total_trades = len(pnl_arr)
        if total_trades > 0:
            total_return_trade = float(np.sum(pnl_arr))
            avg_return = float(np.mean(pnl_arr))
            win_rate = float(np.sum(pnl_arr > 0)) / total_trades
            gross_profit = float(np.sum(pnl_arr[pnl_arr > 0]))
            gross_loss = float(abs(np.sum(pnl_arr[pnl_arr <= 0])))
            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
            else:
                profit_factor = float('inf') if gross_profit > 0 else 0.0
        else:
            total_return_trade = 0.0
            avg_return = 0.0
            win_rate = 0.0
            profit_factor = 0.0
    elif has_trade_rows:
        total_trades = len(trade_df)
        total_return_trade = float(trade_df['net_pnl'].sum())
        avg_return = float(trade_df['net_pnl'].mean())
        win_rate = float((trade_df['net_pnl'] > 0).sum()) / total_trades

        gross_profit = float(trade_df[trade_df['net_pnl'] > 0]['net_pnl'].sum())
        gross_loss = float(abs(trade_df[trade_df['net_pnl'] <= 0]['net_pnl'].sum()))
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float('inf') if gross_profit > 0 else 0.0
    else:
        total_trades = 0
        total_return_trade = 0.0
        avg_return = 0.0
        win_rate = 0.0
        profit_factor = 0.0

    if has_trade_rows:
        avg_duration = float(trade_df['duration_mins'].mean())
        max_duration = float(trade_df['duration_mins'].max())
    else:
        avg_duration = 0.0
        max_duration = 0.0

    # 优先用权益曲线算总收益和MDD
    if equity_curve is not None and len(equity_curve) > 0:
        eq = np.asarray(equity_curve, dtype=float)
        total_return = float(eq[-1])

        eq_ext = np.concatenate(([0.0], eq))
        peak = np.maximum.accumulate(eq_ext)
        dd = eq_ext - peak
        max_drawdown = float(np.min(dd))
    else:
        total_return = total_return_trade
        if effective_pnls is not None and len(effective_pnls) > 0:
            cumpnl = np.cumsum(np.asarray(effective_pnls, dtype=float))
            cumpnl_ext = np.concatenate(([0.0], cumpnl))
            peak = np.maximum.accumulate(cumpnl_ext)
            dd = cumpnl_ext - peak
            max_drawdown = float(np.min(dd))
        elif has_trade_rows:
            cumpnl = np.cumsum(trade_df['net_pnl'].values.astype(float))
            cumpnl_ext = np.concatenate(([0.0], cumpnl))
            peak = np.maximum.accumulate(cumpnl_ext)
            dd = cumpnl_ext - peak
            max_drawdown = float(np.min(dd))
        else:
            max_drawdown = 0.0

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
# 8. 参数灵敏度扫描（时间单位参数 + delta归一化）
# =============================================================================
def parameter_sensitivity_scan(df_file, timeframe='15min'):
    key_word = os.path.basename(df_file).replace('.csv', '')

    original_df = pd.read_csv(df_file)
    original_df['open_time'] = pd.to_datetime(original_df['open_time'])

    if timeframe != '1min':
        df = resample_data(original_df, timeframe)
        print(f"[{key_word}] 重采样: {len(original_df)} 行 → {len(df)} 行 ({timeframe})")
    else:
        df = original_df.copy()
        print(f"[{key_word}] 使用原始1min数据: {len(df)} 行")

    tf_min = get_tf_minutes(timeframe)
    min1_cache = _build_1min_cache(original_df) if (USE_NEXT_BAR_EXEC or USE_1MIN_PATH) else None

    # ===== 时间单位参数网格 =====
    z_entry_list = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
    z_exit_list = [0.0, 0.5, 1.0]
    lookback_hours_list = [10, 15, 30]
    delta_per_day_list = [0.01, 0.001, 0.0001]
    ve_list = [1e-3]
    min_hold_hours_list = [0.5, 0.75, 1.0, 1.25, 1.5, 3.0, 6.0, 12.0]
    cooldown_hours = 1.0
    max_hold_days = 4.0

    cooldown_bars = hours_to_bars(cooldown_hours, tf_min)
    max_hold_bars = days_to_bars(max_hold_days, tf_min)

    total_combos = (len(z_entry_list) * len(z_exit_list) * len(lookback_hours_list)
                    * len(delta_per_day_list) * len(ve_list) * len(min_hold_hours_list))
    print(f"总参数组合数: {total_combos} | cooldown={cooldown_bars}bars, "
          f"max_hold={max_hold_bars}bars")

    results = []
    count = 0
    start_time = time.time()

    for z_entry in z_entry_list:
        for z_exit in z_exit_list:
            if z_exit >= z_entry:
                continue
            for lb_h in lookback_hours_list:
                z_lb = hours_to_bars(lb_h, tf_min)
                for dpd in delta_per_day_list:
                    delta = delta_per_day_to_bar(dpd, tf_min)
                    for ve in ve_list:
                        for mh_h in min_hold_hours_list:
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

                                # 零手续费
                                trades_0, eq_0, eqt_0, st_0 = compute_equity_and_trades(
                                    full_df, min1_cache=min1_cache, fee_override=0.0, tf_minutes=tf_min)
                                eff_0 = _build_effective_pnls(trades_0, eq_0, st_0)

                                # 真实手续费
                                trades_r, eq_r, eqt_r, st_r = compute_equity_and_trades(
                                    full_df, min1_cache=min1_cache, fee_override=SINGLE_LEG_FEE, tf_minutes=tf_min)
                                eff_r = _build_effective_pnls(trades_r, eq_r, st_r)

                                # [FIX] 与WF对齐，scan门槛从10提升到15
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
                                      f"有效结果: {len(results)} | 耗时: {elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(f"\n扫描完成: {count} 组合, {len(results)} 有效结果, 耗时 {elapsed:.1f}s")

    if not results:
        print("没有有效结果！")
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # ---- 输出分析 ----
    print(f"\n{'=' * 100}")
    print(f"TOP 20 参数组合（按真实手续费总收益排序）| {key_word} | {timeframe}")
    print(f"{'=' * 100}")

    top20 = result_df.sort_values('real_total_ret', ascending=False).head(20)
    display_cols = ['z_entry', 'z_exit', 'lookback_hours', 'delta_per_day', 'min_hold_hours',
                    'trades', 'zero_avg_pnl', 'real_avg_pnl', 'fee_per_trade',
                    'real_total_ret', 'real_win_rate', 'real_pf', 'real_max_dd',
                    'avg_hold_hrs']
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
    print(f"\n{'=' * 100}")
    print(f"核心判断 | {key_word} | {timeframe}")
    print(f"{'=' * 100}")
    print(f"  扫描参数组合总数: {len(result_df)}")
    print(f"  真实手续费下盈利的组合数: {len(profitable)}")
    if len(profitable) > 0:
        print(f"  盈利组合中最高总收益: {profitable['real_total_ret'].max():.2f}%")
        print(f"  盈利组合中平均每笔利润范围: "
              f"{profitable['real_avg_pnl'].min():.4f}% ~ "
              f"{profitable['real_avg_pnl'].max():.4f}%")
        print(f"  盈利组合中平均持仓时间范围: "
              f"{profitable['avg_hold_hrs'].min():.1f}h ~ "
              f"{profitable['avg_hold_hrs'].max():.1f}h")
        print(f"\n  ✅ 存在可盈利参数区间，值得进一步做Walk-Forward验证")
    else:
        print(f"\n  ❌ 在当前费率下没有任何参数组合能盈利")
        if not result_df.empty:
            print(f"  零手续费下最高每笔利润: {result_df['zero_avg_pnl'].max():.4f}%")
            print(f"  每笔交易成本约: {result_df['fee_per_trade'].mean():.4f}%")
            ratio = result_df['fee_per_trade'].mean() / max(result_df['zero_avg_pnl'].max(), 0.0001)
            print(f"  成本是利润的 {ratio:.1f} 倍")

            if result_df['zero_avg_pnl'].max() > 0.05:
                print(f"\n  💡 零手续费下有不错的alpha({result_df['zero_avg_pnl'].max():.4f}%)，")
                print(f"     考虑用挂单(maker)降低手续费，或寻找费率更低的交易所")
            else:
                print(f"\n  💡 即使零手续费alpha也很薄，建议:")
                print(f"     1. 换其他交易对（BTC-SOL, ETH-SOL等相关性较低的配对）")
                print(f"     2. 在更长周期寻找alpha（4h, 1d）")

    os.makedirs('backtest_pair', exist_ok=True)
    out_path = f'backtest_pair/param_scan_{key_word}_{timeframe}.csv'
    result_df.to_csv(out_path, index=False)
    print(f"\n完整结果已保存到: {out_path}")
    return result_df


# =============================================================================
# 9. Walk-Forward
# =============================================================================
def _compute_param_neighbors(param_key, param_grids):
    """
    计算一个参数组合的所有理论网格邻居。邻居 = 只在一个维度上偏移±1步。
    [FIX] 不再过滤是否存在于某个通过集合中，返回所有理论上存在的网格邻居。
    """
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


def walk_forward_optimization(df_file, timeframe='15min',
                              train_months=6, test_months=1,
                              cooldown_hours=1.0, max_hold_days=4.0):
    key_word = os.path.basename(df_file).replace('.csv', '') + f'_{timeframe}'

    original_df = pd.read_csv(df_file)
    original_df['open_time'] = pd.to_datetime(original_df['open_time'])

    if timeframe != '1min':
        df = resample_data(original_df, timeframe)
        print(f"重采样: {len(original_df)} 行 → {len(df)} 行 ({timeframe})")
    else:
        df = original_df.copy()

    tf_min = get_tf_minutes(timeframe)
    cooldown_bars = hours_to_bars(cooldown_hours, tf_min)
    max_hold_bars = days_to_bars(max_hold_days, tf_min)
    min1_cache = _build_1min_cache(original_df) if (USE_NEXT_BAR_EXEC or USE_1MIN_PATH) else None

    # ===== 时间单位搜索空间 =====
    z_entry_list = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
    z_exit_list = [0.0, 0.5, 1.0]
    lookback_hours_list = [10, 15, 30]
    delta_per_day_list = [0.01, 0.001, 0.0001]
    ve_list = [1e-3]
    min_hold_hours_list = [0.5, 0.75, 1.0, 1.25, 1.5, 3.0, 6.0, 12.0]

    param_grids = [
        z_entry_list, z_exit_list, lookback_hours_list,
        delta_per_day_list, ve_list, min_hold_hours_list
    ]

    # 硬门槛：每月至少3笔 → train_months * 3
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

        # ===== 第一步：在训练集上扫描所有参数，收集(参数, score) =====
        all_candidates = {}
        # [FIX] 记录所有被评估组合的原始t-stat，用于高原选择中的邻域打分
        all_evaluated_scores = {}
        combo_count = 0
        valid_count = 0

        for z_entry in z_entry_list:
            for z_exit in z_exit_list:
                if z_exit >= z_entry:
                    continue
                for lb_h in lookback_hours_list:
                    z_lb = hours_to_bars(lb_h, tf_min)
                    for dpd in delta_per_day_list:
                        delta = delta_per_day_to_bar(dpd, tf_min)
                        for ve in ve_list:
                            for mh_h in min_hold_hours_list:
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

                                    trades, eq_curve, eq_times, exec_state_train = compute_equity_and_trades(
                                        full_df, min1_cache=min1_cache, tf_minutes=tf_min
                                    )
                                    eff_pnls = _build_effective_pnls(trades, eq_curve, exec_state_train)

                                    # ★ FIX: 高原选择 — 细化 raw_score 计算，区分 std=0 和数据不足
                                    if len(eff_pnls) >= 2 and np.std(eff_pnls) > 0:
                                        raw_score = np.mean(eff_pnls) / np.std(eff_pnls) * np.sqrt(len(eff_pnls))
                                    elif len(eff_pnls) >= 2:
                                        # std=0 说明每笔PnL完全相同，用均值符号给一个微弱方向信号
                                        raw_score = np.sign(np.mean(eff_pnls)) * 0.01
                                    else:
                                        raw_score = 0.0  # 数据不足，保持中性
                                    all_evaluated_scores[param_key] = raw_score

                                    if len(eff_pnls) < min_trades_required:
                                        continue
                                    if np.std(eff_pnls) == 0:
                                        continue

                                    # --- 硬过滤 1: 平均净收益为正（含期末MTM）---
                                    if np.mean(eff_pnls) <= 0:
                                        continue

                                    # --- 硬过滤 2: 1.5x费用压力下仍正 ---
                                    trades_15x, eq_15x, eqt_15x, st_15x = compute_equity_and_trades(
                                        full_df, min1_cache=min1_cache,
                                        fee_override=SINGLE_LEG_FEE * 1.5,
                                        tf_minutes=tf_min
                                    )
                                    eff_15x = _build_effective_pnls(trades_15x, eq_15x, st_15x)
                                    if len(eff_15x) == 0 or np.mean(eff_15x) <= 0:
                                        continue

                                    # --- 硬过滤 3: 训练数据按真实时间切3段，至少2段为正 ---
                                    seg_pnls = _split_train_thirds(train_start_ns, train_end_ns, eq_times, eq_curve)
                                    pos_segs = sum(1 for s in seg_pnls if s > 0)
                                    if pos_segs < 2:
                                        continue

                                    # --- [FIX-3] 硬过滤 4: 训练集路径MDD不超过阈值 ---
                                    if len(eq_curve) > 0:
                                        _eq_ext = np.concatenate(([0.0], eq_curve))
                                        _peak = np.maximum.accumulate(_eq_ext)
                                        _mdd = float(np.min(_eq_ext - _peak))
                                        if _mdd < WF_MAX_TRAIN_MDD:
                                            continue

                                    valid_count += 1
                                    score = raw_score
                                    train_total_ret = float(eq_curve[-1]) if len(eq_curve) > 0 else float(np.sum(eff_pnls))

                                    all_candidates[param_key] = {
                                        'score': score,
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

        # ===== 第二步：参数高原选择（而非尖峰） =====
        sorted_keys = sorted(all_candidates.keys(),
                             key=lambda k: all_candidates[k]['score'], reverse=True)
        top_n = max(1, len(sorted_keys) // 5)  # top 20%
        top_keys = set(sorted_keys[:top_n])

        best_plateau_score = -np.inf
        best_key = sorted_keys[0]  # fallback

        for pkey in top_keys:
            # [FIX] 使用 all_evaluated_scores（包含未通过过滤的组合）进行邻域打分
            # 这样孤立尖峰（周围邻居全部失败/分数低）会被显著惩罚
            neighbors = _compute_param_neighbors(pkey, param_grids)
            # ★ FIX: 高原选择 — 去掉 > 0 过滤，让失败/亏损的邻居拉低中位数，
            # 使孤立尖峰的 plateau_score 被正确惩罚而非保护
            neighbor_scores = [all_evaluated_scores[nk]
                               for nk in neighbors
                               if nk in all_evaluated_scores]
            if neighbor_scores:
                avg_nb = float(np.median(neighbor_scores))
            else:
                avg_nb = 0.0

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
        print(f"  Train: t-stat={chosen['score']:.4f}, "
              f"trades={chosen['train_trades']}, "
              f"closed={chosen['train_closed_trades']}, "
              f"total_ret={chosen['train_ret']:.2f}%, "
              f"plateau_score={best_plateau_score:.4f}")
        print(f"  通过硬过滤: {len(all_candidates)}/{combo_count}")

        # ===== 第三步：在 combined(train+test) 上用选定参数跑 OOS =====
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

        # 训练段：得到边界执行状态和边界净值
        _, train_equity, train_eq_times, train_exec_state = compute_equity_and_trades(
            train_portion, min1_cache=min1_cache, tf_minutes=tf_min
        )
        boundary_equity = train_equity[-1] if len(train_equity) > 0 else 0.0

        # 测试段：从train边界继续
        oos_trades, oos_equity_raw, oos_eq_times, test_exec_state = compute_equity_and_trades(
            test_portion, min1_cache=min1_cache,
            exec_state=train_exec_state, tf_minutes=tf_min
        )

        # =====================================================================
        # [FIX-1] WF窗口边界强制平仓：若OOS期末仍持仓，强制按末价结算并扣手续费
        # 原因：下一窗口将用新参数从零开始，遗留仓位会被"凭空消失"
        # =====================================================================
        if test_exec_state is not None and float(test_exec_state[2]) != 0.0 and len(oos_equity_raw) > 0:
            _fc_dir = int(float(test_exec_state[2]))
            _fc_entry_main = float(test_exec_state[0])
            _fc_entry_sub = float(test_exec_state[1])
            _fc_beta = float(test_exec_state[3])
            _fc_entry_time = test_exec_state[5]
            _fc_mark_main = float(test_exec_state[6])
            _fc_mark_sub = float(test_exec_state[7])

            # 1) 从权益曲线末点扣除出场手续费
            oos_equity_raw[-1] -= SINGLE_LEG_FEE * 100.0

            # 2) 计算强制平仓交易的 net_pnl（残差法：调整后总OOS权益 - 已平仓交易之和）
            _oos_eq_adj = float(oos_equity_raw[-1]) - boundary_equity
            _closed_sum = float(oos_trades['net_pnl'].sum()) if (not oos_trades.empty) else 0.0
            _fc_net_pnl = round(_oos_eq_adj - _closed_sum, 4)

            # 3) 构造强制平仓交易记录
            _fc_open_time = pd.to_datetime(test_portion['open_time'].iloc[0]) if len(test_portion) > 0 else pd.NaT
            _fc_close_time = pd.to_datetime(test_portion['open_time'].iloc[-1]) if len(test_portion) > 0 else pd.NaT

            _fc_duration = 0.0
            if not pd.isna(_fc_entry_time) and not pd.isna(_fc_close_time):
                try:
                    _fc_duration = (pd.Timestamp(_fc_close_time) - pd.Timestamp(_fc_entry_time)).total_seconds() / 60.0
                except Exception:
                    _fc_duration = 0.0

            _fc_direction_str = (f'做多价差 (Long BTC / Short {_fc_beta:.3f} ETH)'
                                 if _fc_dir == -1
                                 else f'做空价差 (Short BTC / Long {_fc_beta:.3f} ETH)')

            _fc_btc_roi = f"{((_fc_mark_main - _fc_entry_main) / _fc_entry_main * 100):.2f}" if _fc_entry_main > 0 else "0.00"
            _fc_eth_roi = f"{((_fc_mark_sub - _fc_entry_sub) / _fc_entry_sub * 100):.2f}" if _fc_entry_sub > 0 else "0.00"

            _fc_trade = pd.DataFrame([{
                'open_time': _fc_open_time,
                'actual_open_time': _fc_entry_time,
                'close_time': _fc_close_time,
                'duration_mins': _fc_duration,
                'direction': _fc_direction_str,
                'btc_roi': _fc_btc_roi,
                'eth_roi': _fc_eth_roi,
                'net_pnl': _fc_net_pnl,
                'open_main': _fc_entry_main, 'close_main': _fc_mark_main,
                'open_sub': _fc_entry_sub, 'close_sub': _fc_mark_sub,
                'entry_z': np.nan, 'exit_z': np.nan,
                'entry_spread': np.nan, 'exit_spread': np.nan,
                'entry_beta': _fc_beta, 'exit_beta': _fc_beta,
                'entry_alpha': np.nan, 'exit_alpha': np.nan,
                'signal': _fc_dir,
                'inherited_from_prev_window': True,
                'close_reason': 'wf_boundary_force_close',
            }])

            if not oos_trades.empty:
                oos_trades = pd.concat([oos_trades, _fc_trade], ignore_index=True)
            else:
                oos_trades = _fc_trade.copy()

            # 4) 更新 exec_state 为无持仓，使 _build_effective_pnls 不再追加残差
            _st = list(test_exec_state)
            _st[2] = 0.0  # entry_dir = 0
            test_exec_state = tuple(_st)

            print(f"  [FIX-1] OOS期末强制平仓: dir={_fc_dir}, net_pnl={_fc_net_pnl:.4f}%")
        # =====================================================================

        # 改成相对测试起点边界净值的增量曲线
        oos_equity = oos_equity_raw - boundary_equity if len(oos_equity_raw) > 0 else oos_equity_raw
        oos_effective_pnls = _build_effective_pnls(oos_trades, oos_equity, test_exec_state)

        # 用于最终汇总的连续OOS equity
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
                  f"win_rate={stats['Win Rate']:.2%}, "
                  f"total_ret={stats['Total Return']:.2f}%, "
                  f"MDD(path)={stats['Max Drawdown']:.4f}%")

            # --- OOS 退化比 ---
            train_avg = chosen['train_avg_pnl']
            oos_avg = stats['avg_profit_per_trade']
            if oos_avg <= 0:
                degradation_ratio = float('inf')
            else:
                degradation_ratio = train_avg / oos_avg

            oos_degradation_log.append({
                'window_id': window_id,
                'train_avg_pnl': round(train_avg, 4),
                'oos_avg_pnl': round(oos_avg, 4),
                'degradation_ratio': round(degradation_ratio, 2) if np.isfinite(degradation_ratio) else float('inf'),
            })
            print(f"  退化比 (Train/OOS): {degradation_ratio:.2f}" if np.isfinite(degradation_ratio)
                  else f"  退化比 (Train/OOS): inf")

        current_train_start += test_delta
        window_id += 1

    # ========== 汇总 ==========
    print(f"\n{'=' * 60}")
    print(f"Walk-Forward 汇总 | {key_word}")
    print(f"{'=' * 60}")

    os.makedirs('backtest_pair', exist_ok=True)

    combined_oos = pd.concat(all_oos_trades, ignore_index=True) if all_oos_trades else pd.DataFrame()
    combined_oos_equity = np.concatenate(all_oos_equity_segments) if all_oos_equity_segments else np.array([])
    combined_oos_effective_pnls = np.concatenate(all_oos_effective_pnls) if all_oos_effective_pnls else np.array([])

    if not combined_oos.empty:
        combined_oos.to_csv(f'backtest_pair/{key_word}_wf_oos_trades.csv', index=False)

    if len(combined_oos_equity) > 0 or not combined_oos.empty:
        final = print_backtest_stats(combined_oos, combined_oos_equity,
                                     effective_pnls=combined_oos_effective_pnls)
        if final:
            print(f"  总交易数:   {final['total_trades']}")
            print(f"  总收益:     {final['Total Return']:.4f}%")
            print(f"  平均每笔:   {final['avg_profit_per_trade']:.4f}%")
            print(f"  胜率:       {final['Win Rate']:.2%}")
            print(f"  盈亏比:     {final['Profit Factor']:.4f}")
            print(f"  最大回撤:   {final['Max Drawdown']:.4f}%")

        if all_oos_stats:
            oos_df = pd.DataFrame(all_oos_stats)
            print(f"\n  --- 各窗口OOS表现 ---")
            for _, row in oos_df.iterrows():
                print(f"  Window {int(row['window_id'])}: "
                      f"trades={int(row['total_trades'])}, "
                      f"ret={row['Total Return']:.2f}%, "
                      f"win={row['Win Rate']:.1%}, "
                      f"pf={row['Profit Factor']:.2f}, "
                      f"MDD={row['Max Drawdown']:.4f}%")

            win_windows = len(oos_df[oos_df['Total Return'] > 0])
            total_windows = len(oos_df)
            if total_windows > 0:
                print(f"\n  窗口胜率: {win_windows}/{total_windows} = "
                      f"{win_windows / total_windows:.1%}")

        # --- OOS退化比诊断 ---
        if oos_degradation_log:
            deg_df = pd.DataFrame(oos_degradation_log)
            print(f"\n  --- OOS退化比诊断 (Train_avg_pnl / OOS_avg_pnl) ---")
            for _, row in deg_df.iterrows():
                flag = " ⚠️ 过拟合风险" if ((np.isfinite(row['degradation_ratio']) and row['degradation_ratio'] > 3.0)
                                             or (not np.isfinite(row['degradation_ratio']))) else ""
                ratio_str = f"{row['degradation_ratio']:.2f}" if np.isfinite(row['degradation_ratio']) else "inf"
                print(f"  Window {int(row['window_id'])}: "
                      f"train={row['train_avg_pnl']:.4f}%, "
                      f"oos={row['oos_avg_pnl']:.4f}%, "
                      f"ratio={ratio_str}{flag}")

            finite_ratios = [r['degradation_ratio'] for r in oos_degradation_log
                             if np.isfinite(r['degradation_ratio'])]
            if finite_ratios:
                avg_ratio = np.mean(finite_ratios)
                high_ratio_count = sum(1 for r in finite_ratios if r > 3.0)
                print(f"\n  平均退化比: {avg_ratio:.2f}")
                print(f"  退化比>3的窗口: {high_ratio_count}/{len(finite_ratios)}")
                if avg_ratio > 3.0:
                    print(f"  🔴 整体退化严重 (均值>{3.0:.0f})，过拟合风险高，谨慎上线")
                elif avg_ratio > 2.0:
                    print(f"  🟡 有一定退化，建议增大训练窗口或放宽参数过滤")
                else:
                    print(f"  🟢 退化比在合理范围内")

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
# 10. 数据合并工具
# =============================================================================
def merge_df(main_df_file, sub_df_file, merged_file_path):
    if os.path.exists(merged_file_path):
        return
    try:
        main_df = pd.read_csv(main_df_file, usecols=['timestamp', 'close'],
                              parse_dates=['timestamp'])
        main_df = main_df.sort_values('timestamp').set_index('timestamp')
        sub_df = pd.read_csv(sub_df_file, usecols=['timestamp', 'close'],
                             parse_dates=['timestamp'])
        sub_df = sub_df.sort_values('timestamp').set_index('timestamp')
        merged = pd.merge(main_df[['close']], sub_df[['close']],
                          left_index=True, right_index=True,
                          suffixes=('_main', '_sub'), how='inner')
        if not merged.empty:
            merged = merged.rename_axis('open_time').reset_index()
            merged.to_csv(merged_file_path, index=False)
            print(f"Merged: {merged_file_path} ({len(merged)} rows)")
    except Exception as e:
        print(f"Error merging: {e}")


def analyze_generated_scan_results(scan_dir='backtest_pair'):
    """
    扫描读取指定目录下的 param_scan_*.csv 文件，提取核心判断数据并按统一格式打印
    """
    if not os.path.exists(scan_dir):
        print(f"提示: 目录 '{scan_dir}' 不存在，请先运行参数灵敏度扫描生成数据。")
        return

    scan_files = [f for f in os.listdir(scan_dir) if f.startswith('param_scan_') and f.endswith('.csv')]

    if not scan_files:
        print(f"提示: 在 '{scan_dir}' 目录下没有找到生成的 param_scan_*.csv 文件。")
        return

    print("\n\n=============== 批量分析已生成的扫描结果 ===============")

    for file in scan_files:
        file_path = os.path.join(scan_dir, file)

        match = re.search(r'param_scan_(.*?)_([0-9a-zA-Z]+)\.csv', file)
        if match:
            key_word = match.group(1)
            timeframe = match.group(2)
        else:
            key_word = file.replace('param_scan_', '').replace('.csv', '')
            timeframe = "未知"

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"读取文件出错: {file} | 错误信息: {e}")
            continue

        if df.empty:
            print(f"\n{'=' * 100}")
            print(f"核心判断 | {key_word} | {timeframe}")
            print(f"{'=' * 100}")
            print("  ❌ 警告: 该文件为空，无扫描结果\n")
            continue

        total_combos = len(df)
        profitable = df[df['real_total_ret'] > 0]
        profitable_count = len(profitable)

        print(f"{'=' * 100}")
        print(f"核心判断 | {key_word} | {timeframe}")
        print(f"{'=' * 100}")
        print(f"  扫描参数组合总数: {total_combos}")
        print(f"  真实手续费下盈利的组合数: {profitable_count}")

        if profitable_count > 0:
            max_ret = profitable['real_total_ret'].max()
            min_avg_pnl = profitable['real_avg_pnl'].min()
            max_avg_pnl = profitable['real_avg_pnl'].max()
            min_hold = profitable['avg_hold_hrs'].min()
            max_hold = profitable['avg_hold_hrs'].max()

            print(f"  盈利组合中最高总收益: {max_ret:.2f}%")
            print(f"  盈利组合中平均每笔利润范围: {min_avg_pnl:.4f}% ~ {max_avg_pnl:.4f}%")
            print(f"  盈利组合中平均持仓时间范围: {min_hold:.1f}h ~ {max_hold:.1f}h\n")
            print(f"  ✅ 存在可盈利参数区间，值得进一步做Walk-Forward验证\n")
        else:
            print(f"\n  ❌ 在当前费率下没有任何参数组合能盈利")
            if 'zero_avg_pnl' in df.columns and 'fee_per_trade' in df.columns:
                zero_pnl_max = df['zero_avg_pnl'].max()
                fee_mean = df['fee_per_trade'].mean()
                ratio = fee_mean / max(zero_pnl_max, 0.0001)
                print(f"  零手续费下最高每笔利润: {zero_pnl_max:.4f}%")
                print(f"  每笔交易成本约: {fee_mean:.4f}%")
                print(f"  成本是利润的 {ratio:.1f} 倍\n")
            else:
                print()


# =============================================================================
# 11. 主入口
# =============================================================================
if __name__ == '__main__':
    # analyze_generated_scan_results()

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

    # 如果扫描确认有盈利参数，取消下面的注释跑Walk-Forward
    final_df_file = 'kline_data/DOGE_ETH_1m.csv'
    walk_forward_optimization(final_df_file, timeframe='30min',
                              train_months=6, test_months=1,
                              cooldown_hours=1.0, max_hold_days=4.0)