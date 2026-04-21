"""
配对交易回测系统 (Refactored)
============================
§1  Config          全局配置 (dataclass)
§2  TimeUtils       时间工具
§3  DataUtils       数据处理
§4  KalmanFilter    卡尔曼滤波 (numba JIT)
§5  SignalEngine    信号生成 (numba JIT) + 管道
§6  Scan1Min        1分钟止损扫描 (numba JIT)
§7  TradeExecutor   交易执行引擎 (class)
§8  Statistics      统计分析
§9  ParamScan       参数灵敏度扫描 (并行)
§10 WalkForward     Walk-Forward 优化
§11 Main            主入口
"""

import traceback
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np
import os
import time
import numba as nb


# =============================================================================
# §1 全局配置
# =============================================================================
@dataclass
class Config:
    """策略与回测的全部可调参数"""
    single_leg_fee: float = 0.0006
    use_next_bar_exec: bool = True
    use_1min_path: bool = True
    sl_mult: float = 3.0
    beta_min: float = 0.1
    beta_max: float = 5.0
    wf_max_train_mdd: float = -15.0
    ni_entry_gate: float = 3.0
    kf_kill_trace_mult: float = 5.0
    kf_kill_ni_consec: int = 3
    kf_kill_ni_threshold: float = 3.0

    @staticmethod
    def param_grids():
        return {
            'z_entry': [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0],
            'z_exit': [0.0, 0.5, 1.0],
            'lookback_hours': [10, 15, 30],
            'delta_per_day': [0.01, 0.001, 0.0001],
            've': [1e-3],
            'min_hold_hours': [0.5, 0.75, 1.0, 1.25, 1.5, 3.0, 6.0, 12.0],
        }


CFG = Config()
_NAT = np.datetime64('NaT')


# =============================================================================
# §2 时间工具
# =============================================================================
def get_tf_minutes(tf_str):
    tf = tf_str.lower().strip()
    if tf.endswith('min'): return int(tf.replace('min', ''))
    if tf.endswith('h'):   return int(tf.replace('h', '')) * 60
    if tf.endswith('d'):   return int(tf.replace('d', '')) * 1440
    return 1


def hours_to_bars(hours, tf_min):
    return max(1, int(round(hours * 60.0 / tf_min)))


def days_to_bars(days, tf_min):
    return max(1, int(round(days * 1440.0 / tf_min)))


def delta_per_day_to_bar(dpd, tf_min):
    return dpd / (1440.0 / tf_min)


def _segment_bounds_ns(df, tf_min):
    if df.empty:
        return _NAT, _NAT
    s = pd.to_datetime(df['open_time'].iloc[0]).to_datetime64()
    e = pd.to_datetime(df['open_time'].iloc[-1]).to_datetime64() + np.timedelta64(int(tf_min), 'm')
    return s, e


def _bar_delta_ns(times, tf_min=None):
    if tf_min and tf_min > 0:
        return np.timedelta64(int(tf_min), 'm')
    if len(times) >= 2:
        d = np.diff(times).astype('timedelta64[ns]').astype(np.int64)
        if len(d):
            return np.timedelta64(max(int(np.median(d)), 60_000_000_000), 'ns')
    return np.timedelta64(1, 'm')


# =============================================================================
# §3 数据工具
# =============================================================================
def clean_nan(arr):
    """统一的 NaN/Inf → 0 清洗"""
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def shift_or_copy(arr, shift=True, fill=0.0):
    if not shift:
        return arr.copy()
    out = np.full_like(arr, fill)
    if len(arr) > 1:
        out[1:] = arr[:-1]
    return out


def resample_data(df, timeframe='15min'):
    df = df.copy()
    df['open_time'] = pd.to_datetime(df['open_time'])
    resampled = df.set_index('open_time').resample(timeframe).agg(
        {'close_main': 'last', 'close_sub': 'last'}
    ).dropna()
    return resampled.reset_index()


def build_1min_cache(odf):
    odf = odf.copy()
    odf['open_time'] = pd.to_datetime(odf['open_time'])
    odf = odf.sort_values('open_time').reset_index(drop=True)
    cm = odf['close_main'].values.astype(float)
    cs = odf['close_sub'].values.astype(float)
    return {
        'times': odf['open_time'].values.astype('datetime64[ns]'),
        'close_main': cm, 'close_sub': cs,
        'log_main': np.log(cm), 'log_sub': np.log(cs),
    }


def lookup_1min(cache, target_ns):
    idx = np.searchsorted(cache['times'], target_ns, side='left')
    idx = min(idx, len(cache['times']) - 1)
    return (cache['times'][idx],
            cache['close_main'][idx], cache['close_sub'][idx],
            cache['log_main'][idx], cache['log_sub'][idx])


# =============================================================================
# §4 卡尔曼滤波 (numba JIT, 不修改)
# =============================================================================
@nb.njit(cache=True)
def kalman_filter(x_arr, y_arr, delta, ve,
                  init_alpha=0.0, init_beta=0.0,
                  init_P00=1.0, init_P01=0.0, init_P10=0.0, init_P11=1.0):
    n = len(x_arr)
    betas = np.zeros(n); alphas = np.zeros(n); spreads = np.zeros(n)
    ni = np.zeros(n); trace_P = np.zeros(n)
    a, b = init_alpha, init_beta
    P00, P01, P10, P11 = init_P00, init_P01, init_P10, init_P11

    for t in range(n):
        x, y = x_arr[t], y_arr[t]
        P00 += delta; P11 += delta
        err = y - (a + b * x)
        S = P00 + P01 * x + P10 * x + P11 * x * x + ve
        ni[t] = err / np.sqrt(S) if S > 0 else 0.0
        K0 = (P00 + P01 * x) / S
        K1 = (P10 + P11 * x) / S
        a += K0 * err; b += K1 * err
        nP00 = P00 - K0 * (P00 + P10 * x)
        nP01 = P01 - K0 * (P01 + P11 * x)
        nP10 = P10 - K1 * (P00 + P10 * x)
        nP11 = P11 - K1 * (P01 + P11 * x)
        P00, P01, P10, P11 = nP00, nP01, nP10, nP11
        trace_P[t] = P00 + P11
        alphas[t] = a; betas[t] = b; spreads[t] = err

    return betas, alphas, spreads, ni, trace_P, a, b, P00, P01, P10, P11


# =============================================================================
# §5 信号引擎 (numba JIT + 管道)
# =============================================================================
@nb.njit(cache=True)
def generate_signals(z_values, spreads, betas, alphas,
                     log_main, log_sub, roll_std,
                     ni_arr, trace_P_arr,
                     lookback, z_entry, z_exit,
                     min_hold_bars, cooldown_bars, max_hold_bars,
                     sl_mult, fee, ni_gate, kill_trace, kill_ni_n, kill_ni_thr,
                     beta_min=0.1, beta_max=5.0,
                     init_pos=0, init_held=0, init_since_close=9999,
                     init_fb=0.0, init_fa=0.0, init_es=0.0, init_estd=0.0,
                     init_etrP=0.0, init_cni=0):
    n = len(z_values)
    sigs = np.zeros(n)
    fb_arr = np.zeros(n); fa_arr = np.zeros(n)
    es_arr = np.zeros(n); estd_arr = np.zeros(n)
    stop_f = np.zeros(n); kill_f = np.zeros(n)

    pos, held, sc = init_pos, init_held, init_since_close
    fb, fa, es, estd = init_fb, init_fa, init_es, init_estd
    etrP, cni = init_etrP, init_cni
    start = 0 if pos != 0 else lookback

    for i in range(start, n):
        z = z_values[i]
        if pos == 0:
            sc += 1
            if sc >= cooldown_bars:
                cur_std, cur_b = roll_std[i], betas[i]
                ni_ok = abs(ni_arr[i]) <= ni_gate
                if cur_std > 0 and beta_min <= cur_b <= beta_max and ni_ok:
                    exp_profit = (z_entry - abs(z_exit)) * cur_std
                    rt_cost = 2.0 * fee * (1.0 + abs(cur_b))
                    if exp_profit > rt_cost:
                        d = 0
                        if z > z_entry:   d = 1
                        elif z < -z_entry: d = -1
                        if d != 0:
                            pos, held = d, 0
                            fb, fa = cur_b, alphas[i]
                            es = log_main[i] - fa - fb * log_sub[i]
                            estd, etrP, cni = cur_std, trace_P_arr[i], 0
        else:
            held += 1
            fsp = log_main[i] - fa - fb * log_sub[i]

            stopped = False
            if estd > 0 and sl_mult > 0:
                dev = (fsp - es) if pos == 1 else (es - fsp)
                if dev > sl_mult * estd:
                    stopped = True

            killed = False
            if not stopped:
                if etrP > 0 and trace_P_arr[i] > etrP * kill_trace:
                    killed = True
                if abs(ni_arr[i]) > kill_ni_thr:
                    cni += 1
                else:
                    cni = 0
                if cni >= kill_ni_n:
                    killed = True

            if stopped:
                pos, sc, cni = 0, 0, 0; stop_f[i] = 1.0
            elif killed:
                pos, sc, cni = 0, 0, 0; stop_f[i] = 1.0; kill_f[i] = 1.0
            elif max_hold_bars > 0 and held >= max_hold_bars:
                pos, sc, cni = 0, 0, 0
            elif held >= min_hold_bars:
                improved = (fsp < es) if pos == 1 else (fsp > es)
                if max_hold_bars > 0 and held > max_hold_bars // 2:
                    improved = True
                if pos == 1 and z < z_exit and improved:
                    pos, sc, cni = 0, 0, 0
                elif pos == -1 and z > -z_exit and improved:
                    pos, sc, cni = 0, 0, 0

        sigs[i] = pos
        if pos != 0:
            fb_arr[i] = fb; fa_arr[i] = fa
            es_arr[i] = es; estd_arr[i] = estd
        else:
            fb_arr[i] = betas[i]; fa_arr[i] = alphas[i]

    return (sigs, fb_arr, fa_arr, es_arr, estd_arr, stop_f, kill_f,
            pos, held, sc, fb, fa, es, estd, etrP, cni)


def _write_signal_ctx(df, z_entry, z_exit, min_hold, cooldown, max_hold, cfg):
    """将信号上下文写入 DataFrame (attrs + 列), 供执行引擎读取"""
    ctx = {
        'z_entry': float(z_entry), 'z_exit': float(z_exit),
        'min_hold_bars': int(min_hold), 'cooldown_bars': int(cooldown),
        'max_hold_bars': int(max_hold), 'signal_fee': cfg.single_leg_fee,
        'ni_entry_gate': cfg.ni_entry_gate,
        'kf_kill_trace_mult': cfg.kf_kill_trace_mult,
        'kf_kill_ni_consec': cfg.kf_kill_ni_consec,
        'kf_kill_ni_threshold': cfg.kf_kill_ni_threshold,
    }
    df.attrs['_pt_signal_ctx'] = ctx
    for k, v in ctx.items():
        df[f'__pt_{k}'] = v


def _run_kf_and_rolling(log_sub, log_main, delta, ve, kf_state, z_lookback):
    """运行一次 KF + 一次 Rolling, 返回全部中间数组"""
    if kf_state is None:
        kf_state = (0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    betas, alphas, sp, ni, trP, ka, kb, kP00, kP01, kP10, kP11 = kalman_filter(
        log_sub, log_main, delta=delta, ve=ve,
        init_alpha=kf_state[0], init_beta=kf_state[1],
        init_P00=kf_state[2], init_P01=kf_state[3],
        init_P10=kf_state[4], init_P11=kf_state[5])
    final_kf = (ka, kb, kP00, kP01, kP10, kP11)

    ss = pd.Series(sp)
    rm = ss.rolling(z_lookback, min_periods=z_lookback).mean()
    rs = ss.rolling(z_lookback, min_periods=z_lookback).std()
    z_raw = ((ss - rm) / rs).values
    rs_raw = rs.values
    return (betas, alphas, sp, ni, trP, final_kf,
            z_raw, rs_raw, clean_nan(z_raw), clean_nan(rs_raw),
            clean_nan(ni), clean_nan(trP))


def _apply_signal_to_df(df, raw_sigs, fb, fa, es, estd, z_raw, stop_f, kill_f, do_shift):
    """将信号写入 DataFrame"""
    df['signal']              = shift_or_copy(raw_sigs, do_shift)
    df['signal_frozen_beta']  = shift_or_copy(fb, do_shift)
    df['signal_frozen_alpha'] = shift_or_copy(fa, do_shift)
    df['signal_entry_spread'] = shift_or_copy(es, do_shift)
    df['signal_entry_std']    = shift_or_copy(estd, do_shift)
    df['signal_entry_z']      = shift_or_copy(z_raw, do_shift, fill=np.nan)
    df['stop_flag']           = shift_or_copy(stop_f, do_shift)
    df['kf_kill_flag']        = shift_or_copy(kill_f, do_shift)


def build_signal_df(merged_df, cfg=None, z_lookback=60, z_entry=2.0, z_exit=0.5,
                    delta=1e-5, ve=1e-3,
                    min_hold_bars=12, cooldown_bars=4, max_hold_bars=0,
                    kf_state=None, pos_state=None,
                    precomputed=None):
    """
    统一的信号管道 (合并了原 generate_pair_trading_signals 与 _assemble_signal_df_fast)。
    """
    cfg = cfg or CFG
    df = merged_df.copy()

    if precomputed:
        pc = precomputed
        betas, alphas, sp = pc['betas'], pc['alphas'], pc['spreads']
        ni, trP = pc['ni'], pc['trP']
        z_raw, rs_raw = pc['z_raw'], pc['rs_raw']
        z_c, rs_c = pc['z_clean'], pc['rs_clean']
        ni_c, trP_c = pc['ni_clean'], pc['trP_clean']
        lm, ls = pc['log_main'], pc['log_sub']
        final_kf = None
    else:
        lm = np.log(df['close_main'].values.astype(float))
        ls = np.log(df['close_sub'].values.astype(float))
        betas, alphas, sp, ni, trP, final_kf, z_raw, rs_raw, z_c, rs_c, ni_c, trP_c = \
            _run_kf_and_rolling(ls, lm, delta, ve, kf_state, z_lookback)

    df['log_main'] = lm;  df['log_sub'] = ls
    df['beta'] = betas;   df['alpha'] = alphas
    df['spread'] = sp;    df['kf_ni'] = ni;  df['kf_trace_P'] = trP
    df['z_score'] = z_raw; df['roll_std'] = rs_raw

    if pos_state is None:
        pos_state = (0, 0, 9999, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    ps = pos_state
    pn = len(ps)

    result = generate_signals(
        z_c, sp, betas, alphas, lm, ls, rs_c, ni_c, trP_c,
        z_lookback, z_entry, z_exit,
        min_hold_bars, cooldown_bars, max_hold_bars,
        cfg.sl_mult, cfg.single_leg_fee,
        cfg.ni_entry_gate, cfg.kf_kill_trace_mult,
        cfg.kf_kill_ni_consec, cfg.kf_kill_ni_threshold,
        cfg.beta_min, cfg.beta_max,
        init_pos=int(ps[0]), init_held=int(ps[1]), init_since_close=int(ps[2]),
        init_fb=float(ps[3]), init_fa=float(ps[4]),
        init_es=float(ps[5]), init_estd=float(ps[6]),
        init_etrP=float(ps[7]) if pn > 7 else 0.0,
        init_cni=int(ps[8]) if pn > 8 else 0)

    # 【修复】显式解包，杜绝 Numba 函数签名改变导致的切片错位崩溃风险
    (raw_sigs, fb, fa, es, estd, sf, kf,
     fp, fh, fsc, ffb, ffa, fes, fest, fetrP, fcni) = result

    final_pos = (fp, fh, fsc, ffb, ffa, fes, fest, fetrP, fcni)

    _apply_signal_to_df(df, raw_sigs, fb, fa, es, estd, z_raw, sf, kf, cfg.use_next_bar_exec)
    _write_signal_ctx(df, z_entry, z_exit, min_hold_bars, cooldown_bars, max_hold_bars, cfg)

    return df, final_kf, final_pos


# =============================================================================
# §6 1分钟止损扫描 (numba JIT)
# =============================================================================
@nb.njit(cache=True)
def scan_1min_equity(min1_cm, min1_cs, min1_lm, min1_ls, min1_t_i64,
                     scan_start, scan_end,
                     e_main, e_sub, e_beta, e_dir,
                     e_alpha, e_spread, e_std,
                     sl_mult, cum_r, fee):
    n_pts = scan_end - scan_start
    if n_pts <= 0:
        ef = np.empty(0, dtype=np.float64); ei = np.empty(0, dtype=np.int64)
        return False, np.int64(-1), ef, ei, np.int64(0), 0.0, 0.0, cum_r

    eq_v = np.empty(n_pts, dtype=np.float64)
    eq_t = np.empty(n_pts, dtype=np.int64)
    cnt = np.int64(0)
    total_exp = 1.0 + abs(e_beta)
    check = (e_std > 0 and sl_mult > 0)
    stop_found = False; stop_j = np.int64(-1)
    stop_m = 0.0; stop_s = 0.0; new_cum = cum_r

    for j in range(scan_start, scan_end):
        cm, cs = min1_cm[j], min1_cs[j]
        lm, ls = min1_lm[j], min1_ls[j]
        t = min1_t_i64[j]

        should_stop = False
        if check:
            fsp = lm - e_alpha - e_beta * ls
            dev = (fsp - e_spread) if e_dir == 1 else (e_spread - fsp)
            if dev > sl_mult * e_std:
                should_stop = True

        br = (cm - e_main) / e_main if e_main != 0 else 0.0
        er = (cs - e_sub) / e_sub if e_sub != 0 else 0.0
        gross = (br - e_beta * er) if e_dir == -1 else (e_beta * er - br)
        norm = gross / total_exp if total_exp > 0 else 0.0

        if should_stop:
            new_cum = cum_r + norm - fee
            eq_t[cnt] = t; eq_v[cnt] = new_cum * 100.0; cnt += 1
            stop_found = True; stop_j = np.int64(j)
            stop_m = cm; stop_s = cs
            break

        eq_t[cnt] = t; eq_v[cnt] = (cum_r + norm) * 100.0; cnt += 1

    return stop_found, stop_j, eq_v, eq_t, cnt, stop_m, stop_s, new_cum


# =============================================================================
# §7 交易执行引擎
# =============================================================================
class TradeExecutor:
    """封装 开平仓 / 权益曲线 / 止损 / 信号重建 的完整逻辑"""

    def __init__(self, cfg, df, min1_cache, tf_minutes, fee=None, exec_state=None):
        self.cfg = cfg
        self.fee = fee if fee is not None else cfg.single_leg_fee
        self.trades = []
        self._orig_exec_state = exec_state  # 必须保存原始传入的状态

        self._prepare_arrays(df)
        self._init_state(exec_state)
        self._setup_1min(min1_cache, tf_minutes)
        self._setup_equity_buffer()
    # ---------- 初始化 ----------

    def _prepare_arrays(self, df):
        df = df.reset_index(drop=True)
        for old, new in [('index', 'open_time'), ('timestamp', 'open_time')]:
            if 'open_time' not in df.columns and old in df.columns:
                df.rename(columns={old: new}, inplace=True)
        self.n = len(df)
        if self.n == 0:
            return
        self.signals      = df['signal'].to_numpy(dtype=int)
        self.times         = pd.to_datetime(df['open_time']).to_numpy(dtype='datetime64[ns]')
        self.close_main    = df['close_main'].to_numpy(dtype=float)
        self.close_sub     = df['close_sub'].to_numpy(dtype=float)

        def col(c, d, dt=float):
            if c in df.columns: return df[c].to_numpy(dtype=dt)
            return d() if callable(d) else np.full(self.n, d, dtype=dt)

        n = self.n
        self.z_scores      = col('z_score', lambda: np.full(n, np.nan))
        self.spreads_arr   = col('spread', lambda: np.full(n, np.nan))
        self.betas_arr     = col('beta', lambda: np.zeros(n))
        self.alphas_arr    = col('alpha', lambda: np.zeros(n))
        self.roll_std_arr  = col('roll_std', lambda: np.zeros(n))
        self.sig_fb        = col('signal_frozen_beta', lambda: self.betas_arr.copy())
        self.sig_fa        = col('signal_frozen_alpha', lambda: self.alphas_arr.copy())
        self.sig_es        = col('signal_entry_spread', 0.0)
        self.sig_estd      = col('signal_entry_std', 0.0)
        self.sig_ez        = col('signal_entry_z', lambda: np.full(n, np.nan))
        self.stop_flags    = col('stop_flag', 0, dt=int)
        self.kf_kill_flags = col('kf_kill_flag', 0, dt=int)
        self.log_main      = col('log_main', lambda: np.log(self.close_main))
        self.log_sub       = col('log_sub', lambda: np.log(self.close_sub))
        self.ni_arr        = col('kf_ni', 0.0)
        self.trace_P_arr   = col('kf_trace_P', 0.0)

        self.z_c    = clean_nan(self.z_scores)
        self.rs_c   = clean_nan(self.roll_std_arr)
        self.ni_c   = clean_nan(self.ni_arr)
        self.trP_c  = clean_nan(self.trace_P_arr)

        self._read_sig_ctx(df)

    def _read_sig_ctx(self, df):
        def r(key, default):
            if '_pt_signal_ctx' in getattr(df, 'attrs', {}):
                if key in df.attrs['_pt_signal_ctx']:
                    return df.attrs['_pt_signal_ctx'][key]
            c = f'__pt_{key}'
            if c in df.columns and len(df) > 0:
                return df[c].iloc[0]
            return default
        self.sig_z_entry         = float(r('z_entry', np.nan))
        self.sig_z_exit          = float(r('z_exit', np.nan))
        self.sig_min_hold        = int(r('min_hold_bars', 0))
        self.sig_cooldown        = int(r('cooldown_bars', 0))
        self.sig_max_hold        = int(r('max_hold_bars', 0))
        self.sig_fee             = float(r('signal_fee', self.cfg.single_leg_fee))
        self.sig_ni_gate         = float(r('ni_entry_gate', self.cfg.ni_entry_gate))
        self.sig_kill_trace      = float(r('kf_kill_trace_mult', self.cfg.kf_kill_trace_mult))
        self.sig_kill_ni_n       = int(r('kf_kill_ni_consec', self.cfg.kf_kill_ni_consec))
        self.sig_kill_ni_thr     = float(r('kf_kill_ni_threshold', self.cfg.kf_kill_ni_threshold))
        n = self.n
        self.can_rebuild = (
            np.isfinite(self.sig_z_entry) and np.isfinite(self.sig_z_exit) and
            all(len(a) == n for a in [self.z_c, self.rs_c, self.spreads_arr,
                                       self.betas_arr, self.alphas_arr,
                                       self.log_main, self.log_sub, self.ni_c, self.trP_c]))

    def _init_state(self, es_tuple):
        if self.n == 0:
            return
        s = es_tuple
        if s is None:
            s = []
        sn = len(s) if s else 0
        g = lambda i, d=0.0: (s[i] if i < sn else d)

        self.e_main     = float(g(0))
        self.e_sub      = float(g(1))
        self.e_dir      = int(float(g(2)))
        self.e_beta     = float(g(3))
        self.cum_r      = float(g(4))
        self.e_time     = g(5, _NAT)
        self.mark_main  = float(g(6))
        self.mark_sub   = float(g(7))
        self.e_alpha    = float(g(8))
        self.e_spread   = float(g(9))
        self.e_std      = float(g(10))
        self.prev_sig   = int(float(g(11, float(g(2)))))
        self.in_trade   = (self.e_dir != 0)

        # 段级别开仓信息
        if self.in_trade:
            self.seg_m = self.mark_main if self.mark_main > 0 else self.e_main
            self.seg_s = self.mark_sub if self.mark_sub > 0 else self.e_sub
            self.seg_open  = self.times[0]
            self.seg_fee   = 0.0
            self.seg_inher = True
            self.seg_atime = self.e_time
            self.seg_ez = self.seg_esp = self.seg_ea = np.nan
        else:
            self._reset_seg()

    def _reset_seg(self):
        self.seg_m = self.seg_s = 0.0
        self.seg_open = _NAT
        self.seg_fee = 0.0
        self.seg_inher = False
        self.seg_atime = _NAT
        self.seg_ez = self.seg_esp = self.seg_ea = np.nan

    def _setup_1min(self, m1c, tf_min):
        need = self.cfg.use_next_bar_exec or self.cfg.use_1min_path
        self.m1c = m1c if need else None
        self.use_1m_exec = self.cfg.use_next_bar_exec and (self.m1c is not None)
        self.use_1m      = self.cfg.use_1min_path and (self.m1c is not None)

        if self.n == 0:
            return

        self.times_i64 = self.times.view(np.int64)
        bd = _bar_delta_ns(self.times, tf_min)
        self.bd_i64 = np.int64(bd.astype('timedelta64[ns]').astype(np.int64))

        if self.use_1m:
            m1t = self.m1c['times']
            self.m1t_i64 = m1t.view(np.int64)
            self.m1_cm = self.m1c['close_main']
            self.m1_cs = self.m1c['close_sub']
            self.m1_lm = self.m1c['log_main']
            self.m1_ls = self.m1c['log_sub']
            self.m1_times = m1t
            n = self.n
            self.bar_l = np.searchsorted(self.m1t_i64, self.times_i64, side='left')
            end_t = np.empty(n, dtype=np.int64)
            end_t[:-1] = self.times_i64[1:]
            end_t[-1] = self.times_i64[-1] + self.bd_i64
            self.bar_r = np.searchsorted(self.m1t_i64, end_t, side='left')

    def _setup_equity_buffer(self):
        if self.n == 0:
            return
        cap = (len(self.m1c['times']) + self.n + 1000) if self.use_1m else (self.n + 1000)
        self._eq_v = np.empty(cap, dtype=np.float64)
        self._eq_t = np.empty(cap, dtype=np.int64)
        self._eq_n = 0; self._eq_cap = cap

    def _ensure_cap(self, add):
        if self._eq_n + add > self._eq_cap:
            self._eq_cap = max(self._eq_cap * 2, self._eq_n + add + 10000)
            self._eq_v = np.resize(self._eq_v, self._eq_cap)
            self._eq_t = np.resize(self._eq_t, self._eq_cap)

    def _append_eq(self, t_i64, val):
        if self._eq_n > 0 and self._eq_t[self._eq_n - 1] == t_i64:
            self._eq_v[self._eq_n - 1] = val
        else:
            self._ensure_cap(1)
            self._eq_t[self._eq_n] = t_i64
            self._eq_v[self._eq_n] = val
            self._eq_n += 1

    def _batch_eq(self, src_t, src_v, cnt):
        if cnt <= 0: return
        start = 0
        if self._eq_n > 0 and self._eq_t[self._eq_n - 1] == src_t[0]:
            self._eq_v[self._eq_n - 1] = src_v[0]; start = 1
        nc = cnt - start
        if nc > 0:
            self._ensure_cap(nc)
            self._eq_t[self._eq_n:self._eq_n+nc] = src_t[start:cnt]
            self._eq_v[self._eq_n:self._eq_n+nc] = src_v[start:cnt]
            self._eq_n += nc

    # ---------- PnL 计算 ----------

    @staticmethod
    def _calc_norm(em, es, cm, cs, ebeta, edir):
        total = 1.0 + abs(ebeta)
        if total <= 0: return 0.0
        br = (cm - em) / em if em != 0 else 0.0
        er = (cs - es) / es if es != 0 else 0.0
        gross = (br - ebeta * er) if edir == -1 else (ebeta * er - br)
        return gross / total

    # ---------- 开平仓 ----------

    def _open(self, sig_dir, t_val, m, s, idx):
        self.e_main, self.e_sub = m, s
        self.e_dir  = int(sig_dir)
        self.e_beta = self.sig_fb[idx]
        self.e_alpha = self.sig_fa[idx]
        self.e_spread = self.sig_es[idx]
        self.e_std  = self.sig_estd[idx]
        self.e_time = t_val
        self.in_trade = True
        self.cum_r -= self.fee

        self.seg_m, self.seg_s = m, s
        self.seg_open = t_val; self.seg_fee = self.fee; self.seg_inher = False
        self.seg_atime = t_val
        self.seg_ez  = self.sig_ez[idx] if idx < self.n else np.nan
        self.seg_esp = self.sig_es[idx] if idx < self.n else np.nan
        self.seg_ea  = self.sig_fa[idx] if idx < self.n else np.nan

    def _close(self, t_val, m, s, idx, reason):
        r_add = self._calc_norm(self.e_main, self.e_sub, m, s,
                                self.e_beta, self.e_dir) - self.fee
        self.cum_r += r_add

        seg_net = (self._calc_norm(self.seg_m, self.seg_s, m, s,
                                   self.e_beta, self.e_dir)
                   - self.fee - self.seg_fee)

        if reason in ('hard_stop_1m', 'kf_kill'):
            exit_z = exit_sp = np.nan
            exit_a, exit_b = self.e_alpha, self.e_beta
        else:
            exit_z  = self.z_scores[idx] if idx < self.n else np.nan
            exit_sp = self.spreads_arr[idx] if idx < self.n else np.nan
            exit_a  = self.alphas_arr[idx] if idx < self.n else np.nan
            exit_b  = self.betas_arr[idx] if idx < self.n else self.e_beta

        dur = 0.0
        if not np.isnat(np.datetime64(self.seg_open)):
            dur = (np.int64(t_val) - np.int64(self.seg_open)) / 60e9

        d, b = self.e_dir, self.e_beta
        dir_str = (f'做多价差 (Long BTC / Short {b:.3f} ETH)' if d == -1
                   else f'做空价差 (Short BTC / Long {b:.3f} ETH)')

        self.trades.append({
            'open_time': self.seg_open, 'actual_open_time': self.seg_atime,
            'close_time': t_val, 'duration_mins': dur,
            'direction': dir_str, 'entry_beta': b,
            'open_main': self.seg_m, 'close_main': m,
            'open_sub': self.seg_s, 'close_sub': s,
            'net_pnl': round(seg_net * 100, 4),
            'btc_roi': f"{((m - self.seg_m) / self.seg_m * 100):.2f}" if self.seg_m else "0",
            'eth_roi': f"{((s - self.seg_s) / self.seg_s * 100):.2f}" if self.seg_s else "0",
            'entry_z':  self.seg_ez if not self.seg_inher else np.nan,
            'exit_z': exit_z,
            'entry_spread': self.seg_esp if not self.seg_inher else np.nan,
            'exit_spread': exit_sp,
            'exit_beta': exit_b,
            'entry_alpha': self.seg_ea if not self.seg_inher else np.nan,
            'exit_alpha': exit_a,
            'signal': d,
            'inherited_from_prev_window': self.seg_inher,
            'close_reason': reason,
        })
        self.in_trade = False
        self.e_main = self.e_sub = 0.0; self.e_dir = 0
        self.e_beta = 0.0; self.e_time = _NAT
        self.e_alpha = self.e_spread = self.e_std = 0.0
        self._reset_seg()

    def _close_reason(self, i):
        if self.kf_kill_flags[i] == 1: return 'kf_kill'
        if self.stop_flags[i] == 1: return 'hard_stop'
        return 'signal'

    def _exec_point(self, i):
        if self.use_1m_exec:
            return lookup_1min(self.m1c, self.times[i])
        return (self.times[i], self.close_main[i], self.close_sub[i],
                self.log_main[i], self.log_sub[i])

    # ---------- 信号重建 ----------

    def _rebuild_signals(self, stop_idx):
        if stop_idx < 0 or stop_idx >= self.n:
            return
        self.signals[stop_idx] = 0
        self.sig_es[stop_idx] = 0.0;
        self.sig_estd[stop_idx] = 0
        self.stop_flags[stop_idx] = 0;
        self.kf_kill_flags[stop_idx] = 0

        if not self.can_rebuild:
            return
        sfx = stop_idx + 1
        if sfx >= self.n:
            return

        res = generate_signals(
            self.z_c[sfx:], self.spreads_arr[sfx:], self.betas_arr[sfx:], self.alphas_arr[sfx:],
            self.log_main[sfx:], self.log_sub[sfx:], self.rs_c[sfx:],
            self.ni_c[sfx:], self.trP_c[sfx:],
            0, self.sig_z_entry, self.sig_z_exit,
            self.sig_min_hold, self.sig_cooldown, self.sig_max_hold,
            self.cfg.sl_mult, self.sig_fee,
            self.sig_ni_gate, self.sig_kill_trace,
            self.sig_kill_ni_n, self.sig_kill_ni_thr,
            self.cfg.beta_min, self.cfg.beta_max,
            init_pos=0, init_held=0, init_since_close=0)

        # 【修复】显式解包 16 个返回值，杜绝隐式切片 (res[:7]) 带来的错位崩溃风险
        (ns, nfb, nfa, nes, nestd, nsf, nkf,
         _, _, _, _, _, _, _, _, _) = res

        self.signals[sfx:] = 0
        self.sig_fb[sfx:] = self.betas_arr[sfx:]
        self.sig_fa[sfx:] = self.alphas_arr[sfx:]
        self.sig_es[sfx:] = 0.0;
        self.sig_estd[sfx:] = 0.0
        self.sig_ez[sfx:] = np.nan
        self.stop_flags[sfx:] = 0;
        self.kf_kill_flags[sfx:] = 0

        if self.cfg.use_next_bar_exec:
            self.sig_fb[sfx] = self.betas_arr[stop_idx]
            self.sig_fa[sfx] = self.alphas_arr[stop_idx]
            self.sig_ez[sfx] = self.z_scores[stop_idx] if stop_idx < self.n else np.nan
            a = sfx + 1
            if a < self.n and len(ns) > 0:
                self.signals[a:] = ns[:-1]
                self.sig_fb[a:] = nfb[:-1]
                self.sig_fa[a:] = nfa[:-1]
                self.sig_es[a:] = nes[:-1]
                self.sig_estd[a:] = nestd[:-1]
                self.sig_ez[a:] = self.z_scores[sfx:-1]
                self.stop_flags[a:] = nsf[:-1].astype(int)
                self.kf_kill_flags[a:] = nkf[:-1].astype(int)
        else:
            self.signals[sfx:] = ns
            self.sig_fb[sfx:] = nfb
            self.sig_fa[sfx:] = nfa
            self.sig_es[sfx:] = nes
            self.sig_estd[sfx:] = nestd
            self.sig_ez[sfx:] = self.z_scores[sfx:]
            self.stop_flags[sfx:] = nsf.astype(int)
            self.kf_kill_flags[sfx:] = nkf.astype(int)
    # ---------- 主循环 ----------

    def execute(self):
        """执行回测, 返回 (trade_df, equity, eq_times, final_state)"""
        if self.n == 0:
            return self._build_empty()

        if self.use_1m:
            self._loop_1min()
        else:
            self._loop_bars()
        return self._build_output()

    def _loop_1min(self):
        for i in range(self.n):
            sig = int(self.signals[i])
            psig = int(self.signals[i-1]) if i > 0 else self.prev_sig
            l, r = int(self.bar_l[i]), int(self.bar_r[i])

            if r <= l:
                self._handle_bar_no_1min(i, sig, psig)
                self.mark_main = self.close_main[i]; self.mark_sub = self.close_sub[i]
                continue

            if self.use_1m_exec:
                self._handle_1m_exec(i, sig, psig, l, r)
            else:
                self._handle_bar_exec_1m_path(i, sig, psig, l, r)

            self.mark_main = self.m1_cm[r-1]; self.mark_sub = self.m1_cs[r-1]

        if self._eq_n == 0:
            self._append_eq(self.times_i64[-1] + self.bd_i64, self.cum_r * 100)

    def _handle_bar_no_1min(self, i, sig, psig):
        t, m, s, _, _ = self._exec_point(i)
        if self.use_1m_exec:
            if sig != 0 and psig == 0 and not self.in_trade:
                self._open(sig, t, m, s, i)
            elif self.in_trade and sig == 0 and psig != 0:
                self._close(t, m, s, i, self._close_reason(i))
        else:
            if self.in_trade and sig == 0 and psig != 0:
                self._close(t, m, s, i, self._close_reason(i))
            elif sig != 0 and psig == 0 and not self.in_trade:
                self._open(sig, t, m, s, i)

        if self.in_trade:
            u = self._calc_norm(self.e_main, self.e_sub,
                                self.close_main[i], self.close_sub[i],
                                self.e_beta, self.e_dir)
            self._append_eq(self.times_i64[i] + self.bd_i64, (self.cum_r + u) * 100)
        else:
            self._append_eq(self.times_i64[i] + self.bd_i64, self.cum_r * 100)

    def _handle_1m_exec(self, i, sig, psig, l, r):
        st_t, st_m, st_s = self.m1_times[l], self.m1_cm[l], self.m1_cs[l]

        if sig != 0 and psig == 0 and not self.in_trade:
            self._open(sig, st_t, st_m, st_s, i)
        elif self.in_trade and sig == 0 and psig != 0:
            self._close(st_t, st_m, st_s, i, self._close_reason(i))
            self._append_eq(self.m1t_i64[l], self.cum_r * 100)
            return

        if self.in_trade:
            stop_found, sj, ev, et, cnt, sm, ss, _ = scan_1min_equity(
                self.m1_cm, self.m1_cs, self.m1_lm, self.m1_ls, self.m1t_i64,
                l, r, self.e_main, self.e_sub, self.e_beta, self.e_dir,
                self.e_alpha, self.e_spread, self.e_std,
                self.cfg.sl_mult, self.cum_r, self.fee)
            self._batch_eq(et, ev, cnt)
            if stop_found:
                self._close(self.m1_times[sj], sm, ss, i, 'hard_stop_1m')
                self._rebuild_signals(i)

    def _handle_bar_exec_1m_path(self, i, sig, psig, l, r):
        ev_idx = r - 1
        stopped = False

        if self.in_trade:
            sf, sj, ev, et, cnt, sm, ss, _ = scan_1min_equity(
                self.m1_cm, self.m1_cs, self.m1_lm, self.m1_ls, self.m1t_i64,
                l, ev_idx, self.e_main, self.e_sub, self.e_beta, self.e_dir,
                self.e_alpha, self.e_spread, self.e_std,
                self.cfg.sl_mult, self.cum_r, self.fee)
            self._batch_eq(et, ev, cnt)
            if sf:
                self._close(self.m1_times[sj], sm, ss, i, 'hard_stop_1m')
                self._rebuild_signals(i)
                stopped = True

        if not stopped:
            ct, cm, cs = self.m1_times[ev_idx], self.m1_cm[ev_idx], self.m1_cs[ev_idx]
            clm, cls_ = self.m1_lm[ev_idx], self.m1_ls[ev_idx]

            if self.in_trade and sig == 0 and psig != 0:
                self._close(ct, cm, cs, i, self._close_reason(i))
                self._append_eq(self.m1t_i64[ev_idx], self.cum_r * 100)
            elif sig != 0 and psig == 0 and not self.in_trade:
                self._open(sig, ct, cm, cs, i)
                self._append_eq(self.m1t_i64[ev_idx], self.cum_r * 100)
            elif self.in_trade:
                if self._check_stop_single(ct, cm, cs, clm, cls_, i):
                    self._append_eq(self.m1t_i64[ev_idx], self.cum_r * 100)
                else:
                    u = self._calc_norm(self.e_main, self.e_sub, cm, cs,
                                        self.e_beta, self.e_dir)
                    self._append_eq(self.m1t_i64[ev_idx], (self.cum_r + u) * 100)

    def _check_stop_single(self, t, m, s, lm, ls, idx):
        if not self.in_trade or self.e_dir == 0: return False
        if self.e_std <= 0 or self.cfg.sl_mult <= 0: return False
        fsp = lm - self.e_alpha - self.e_beta * ls
        dev = (fsp - self.e_spread) if self.e_dir == 1 else (self.e_spread - fsp)
        if dev > self.cfg.sl_mult * self.e_std:
            self._close(t, m, s, idx, 'hard_stop_1m')
            self._rebuild_signals(idx)
            return True
        return False

    def _loop_bars(self):
        for i in range(self.n):
            sig = int(self.signals[i])
            psig = int(self.signals[i-1]) if i > 0 else self.prev_sig
            t, m, s, _, _ = self._exec_point(i)

            if sig != 0 and psig == 0 and not self.in_trade:
                self._open(sig, t, m, s, i)
            elif self.in_trade and sig == 0 and psig != 0:
                self._close(t, m, s, i, self._close_reason(i))

            if self.in_trade and self.e_dir != 0:
                u = self._calc_norm(self.e_main, self.e_sub,
                                    self.close_main[i], self.close_sub[i],
                                    self.e_beta, self.e_dir)
                val = (self.cum_r + u) * 100
            else:
                val = self.cum_r * 100
            self._append_eq(self.times_i64[i] + self.bd_i64, val)
            self.mark_main = self.close_main[i]; self.mark_sub = self.close_sub[i]

    # ---------- 输出构建 ----------

    def _final_state(self):
        last_sig = float(self.signals[-1]) if self.n > 0 else float(self.prev_sig)
        return (self.e_main, self.e_sub, float(self.e_dir), self.e_beta, self.cum_r,
                self.e_time, self.mark_main, self.mark_sub,
                self.e_alpha, self.e_spread, self.e_std, last_sig)

    def _build_empty(self):
        # 如果当前窗口没数据，原样返回上一个窗口传进来的状态
        s = self._orig_exec_state or ()
        sn = len(s)
        g = lambda i, d=0.0: s[i] if i < sn else d

        fs = (
            float(g(0)), float(g(1)), float(g(2)), float(g(3)), float(g(4)),
            g(5, _NAT), float(g(6)), float(g(7)),
            float(g(8)), float(g(9)), float(g(10)),
            float(g(11, float(g(2))))
        )
        return pd.DataFrame(), np.array([]), np.array([], dtype='datetime64[ns]'), fs

    def _build_output(self):
        eq = self._eq_v[:self._eq_n].copy()
        eq_t = self._eq_t[:self._eq_n].copy().view('datetime64[ns]')
        fs = self._final_state()
        if self.trades:
            tdf = pd.DataFrame(self.trades)
        else:
            tdf = pd.DataFrame()
        return tdf, eq, eq_t, fs


# 修改 compute_equity_and_trades 的签名，增加 cfg 参数，并默认使用传入的 cfg
def compute_equity_and_trades(full_df, cfg=None, original_1min_df=None, fee_override=None,
                              exec_state=None, min1_cache=None, tf_minutes=None):
    """兼容旧调用接口的封装"""
    active_cfg = cfg or CFG  # 优先使用传入的 cfg
    fee = active_cfg.single_leg_fee if fee_override is None else fee_override

    if min1_cache is None and original_1min_df is not None and (
            active_cfg.use_next_bar_exec or active_cfg.use_1min_path):
        min1_cache = build_1min_cache(original_1min_df)

    ex = TradeExecutor(active_cfg, full_df, min1_cache, tf_minutes, fee, exec_state)
    return ex.execute()


# =============================================================================
# §8 统计分析
# =============================================================================
def build_effective_pnls(trade_df, equity_curve, final_exec_state):
    pnls = trade_df['net_pnl'].values.astype(float).copy() \
        if trade_df is not None and not trade_df.empty else np.array([], dtype=float)
    has_open = (final_exec_state is not None and len(final_exec_state) > 2
                and int(final_exec_state[2]) != 0)
    if has_open and equity_curve is not None and len(equity_curve) > 0:
        closed = float(np.sum(pnls)) if len(pnls) else 0.0
        pnls = np.append(pnls, float(equity_curve[-1]) - closed)
    return pnls


def backtest_stats(trade_df, equity_curve=None, effective_pnls=None):
    has_rows = trade_df is not None and not trade_df.empty
    pnl = (np.asarray(effective_pnls, dtype=float) if effective_pnls is not None
           else (trade_df['net_pnl'].values.astype(float) if has_rows
                 else np.array([], dtype=float)))
    nt = len(pnl)
    if nt > 0:
        tot_r = float(np.sum(pnl)); avg_r = float(np.mean(pnl))
        wr = float(np.sum(pnl > 0)) / nt
        gp = float(np.sum(pnl[pnl > 0])); gl = float(abs(np.sum(pnl[pnl <= 0])))
        pf = (gp / gl) if gl > 0 else (float('inf') if gp > 0 else 0.0)
    else:
        tot_r = avg_r = wr = pf = 0.0

    avg_dur = float(trade_df['duration_mins'].mean()) if has_rows else 0.0
    max_dur = float(trade_df['duration_mins'].max()) if has_rows else 0.0

    if equity_curve is not None and len(equity_curve) > 0:
        eq = np.asarray(equity_curve, dtype=float)
        total_ret = float(eq[-1])
        eq_ext = np.concatenate(([0.0], eq))
    else:
        total_ret = tot_r
        eq_ext = np.concatenate(([0.0], np.cumsum(pnl))) if len(pnl) else np.array([0.0])

    peak = np.maximum.accumulate(eq_ext)
    mdd = float(np.min(eq_ext - peak))
    return {
        'Total Return': total_ret, 'total_trades': nt, 'Win Rate': wr,
        'avg_profit_per_trade': avg_r, 'Max Drawdown': mdd,
        'Profit Factor': pf, 'avg_duration': avg_dur, '最长持仓时间': max_dur,
    }


def _split_monthly_returns(seg_start, seg_end, eq_times, eq_curve, n_months):
    if eq_curve is None or len(eq_curve) == 0 or n_months <= 0:
        return []
    eq = np.asarray(eq_curve, dtype=float)
    et = np.asarray(eq_times).astype('datetime64[ns]')
    s = np.datetime64(seg_start, 'ns')
    e = np.datetime64(seg_end, 'ns')

    # 【修复】使用原生 np.isnat，大幅提升底层性能
    if np.isnat(s) or np.isnat(e):
        return []

    total_ns = int((e - s).astype('timedelta64[ns]').astype(np.int64))
    if total_ns <= 0: return [float(eq[-1])]

    def eq_at(bnd):
        idx = max(0, min(np.searchsorted(et, bnd, side='right') - 1, len(eq) - 1))
        return float(eq[idx])

    rets, prev = [], 0.0
    for k in range(1, n_months + 1):
        cur = eq_at(s + np.timedelta64(int(total_ns * k / n_months), 'ns'))
        rets.append(cur - prev);
        prev = cur
    return rets

def _quick_bt(full_df, m1c, tf_min, fee=None, exec_state=None):
    tdf, eq, et, st = compute_equity_and_trades(
        full_df, min1_cache=m1c, fee_override=fee, tf_minutes=tf_min, exec_state=exec_state)
    eff = build_effective_pnls(tdf, eq, st)
    return tdf, eq, et, st, eff


# =============================================================================
# §6.5 快速回测引擎 (numba JIT) — 用于参数扫描的高速路径
# =============================================================================

@nb.njit(cache=True)
def _calc_norm_nb(em, es, cm, cs, ebeta, edir):
    """与 TradeExecutor._calc_norm 完全一致"""
    total = 1.0 + abs(ebeta)
    if total <= 0.0:
        return 0.0
    br = (cm - em) / em if em != 0.0 else 0.0
    er = (cs - es) / es if es != 0.0 else 0.0
    gross = (br - ebeta * er) if edir == -1 else (ebeta * er - br)
    return gross / total




@nb.njit(cache=True)
def _fast_bt_loop(
        n_bars,
        signals_in, sig_fb_in, sig_fa_in, sig_es_in, sig_estd_in,
        close_main, close_sub, log_main, log_sub,
        m1_cm, m1_cs, m1_lm, m1_ls, m1t_i64,
        bar_l, bar_r, times_i64, bd_i64,
        sl_mult,
        prev_sig_init,
):
    """
    极致优化的核心回测循环 (fee=0), 参数扫描专用.
    移除了所有 rebuild 相关的冗余数组拷贝。
    修复了止损后的“幽灵重入”Bug。
    """
    # 只有 signals 需要 copy，因为我们会就地修改它以抹除残余信号
    signals = signals_in.copy()
    # 其他 sig_xxx 数组全是只读的，直接使用 xxx_in，不 copy！

    m1_len = len(m1_cm)
    max_eq = m1_len + n_bars + 1000
    eq_v = np.empty(max_eq, dtype=np.float64)
    eq_t = np.empty(max_eq, dtype=np.int64)
    eq_fc = np.empty(max_eq, dtype=np.int64)
    eq_n = 0

    max_trades = n_bars // 2 + 100
    trade_pnls = np.empty(max_trades, dtype=np.float64)
    trade_durs = np.empty(max_trades, dtype=np.float64)
    n_trades = 0

    in_trade = False
    e_main = 0.0;
    e_sub = 0.0;
    e_dir = 0
    e_beta = 0.0;
    e_alpha = 0.0;
    e_spread = 0.0;
    e_std = 0.0
    cum_r = 0.0
    seg_m = 0.0;
    seg_s = 0.0
    seg_open_i64 = np.int64(0)
    mark_main = 0.0;
    mark_sub = 0.0
    prev_sig = prev_sig_init
    fc = np.int64(0)

    for i in range(n_bars):
        sig = int(signals[i])
        psig = int(signals[i - 1]) if i > 0 else prev_sig
        l = int(bar_l[i])
        r = int(bar_r[i])

        # ====== 无 1 分钟数据的 bar ======
        if r <= l:
            idx_1m = np.searchsorted(m1t_i64, times_i64[i])
            if idx_1m >= m1_len:
                idx_1m = m1_len - 1
            exec_m = m1_cm[idx_1m]
            exec_s = m1_cs[idx_1m]

            if sig != 0 and psig == 0 and not in_trade:
                e_main = exec_m;
                e_sub = exec_s;
                e_dir = sig
                e_beta = sig_fb_in[i];
                e_alpha = sig_fa_in[i]
                e_spread = sig_es_in[i];
                e_std = sig_estd_in[i]
                seg_m = exec_m;
                seg_s = exec_s
                seg_open_i64 = times_i64[i]
                in_trade = True
                fc += 1
            elif in_trade and sig == 0 and psig != 0:
                gross = _calc_norm_nb(e_main, e_sub, exec_m, exec_s, e_beta, e_dir)
                cum_r += gross
                seg_gross = _calc_norm_nb(seg_m, seg_s, exec_m, exec_s, e_beta, e_dir)
                if n_trades < max_trades:
                    trade_pnls[n_trades] = seg_gross * 100.0
                    trade_durs[n_trades] = (times_i64[i] - seg_open_i64) / 60000000000.0
                    n_trades += 1
                in_trade = False;
                e_dir = 0
                fc += 1

            t_i = times_i64[i] + bd_i64
            if in_trade:
                u = _calc_norm_nb(e_main, e_sub, close_main[i], close_sub[i], e_beta, e_dir)
                val = (cum_r + u) * 100.0
            else:
                val = cum_r * 100.0

            if eq_n > 0 and eq_t[eq_n - 1] == t_i:
                eq_v[eq_n - 1] = val;
                eq_fc[eq_n - 1] = fc
            else:
                eq_t[eq_n] = t_i;
                eq_v[eq_n] = val;
                eq_fc[eq_n] = fc
                eq_n += 1

            mark_main = close_main[i];
            mark_sub = close_sub[i]
            continue

        # ====== 有 1 分钟数据的 bar ======
        st_m = m1_cm[l];
        st_s = m1_cs[l]

        if sig != 0 and psig == 0 and not in_trade:
            e_main = st_m;
            e_sub = st_s;
            e_dir = sig
            e_beta = sig_fb_in[i];
            e_alpha = sig_fa_in[i]
            e_spread = sig_es_in[i];
            e_std = sig_estd_in[i]
            seg_m = st_m;
            seg_s = st_s
            seg_open_i64 = m1t_i64[l]
            in_trade = True
            fc += 1
        elif in_trade and sig == 0 and psig != 0:
            gross = _calc_norm_nb(e_main, e_sub, st_m, st_s, e_beta, e_dir)
            cum_r += gross
            seg_gross = _calc_norm_nb(seg_m, seg_s, st_m, st_s, e_beta, e_dir)
            if n_trades < max_trades:
                trade_pnls[n_trades] = seg_gross * 100.0
                trade_durs[n_trades] = (m1t_i64[l] - seg_open_i64) / 60000000000.0
                n_trades += 1
            in_trade = False;
            e_dir = 0
            fc += 1

            t_i = m1t_i64[l];
            val = cum_r * 100.0
            if eq_n > 0 and eq_t[eq_n - 1] == t_i:
                eq_v[eq_n - 1] = val;
                eq_fc[eq_n - 1] = fc
            else:
                eq_t[eq_n] = t_i;
                eq_v[eq_n] = val;
                eq_fc[eq_n] = fc
                eq_n += 1

            mark_main = m1_cm[r - 1];
            mark_sub = m1_cs[r - 1]
            continue

        if in_trade:
            total_exp = 1.0 + abs(e_beta)
            check_stop = (e_std > 0.0 and sl_mult > 0.0)

            for j in range(l, r):
                cm_j = m1_cm[j];
                cs_j = m1_cs[j]
                lm_j = m1_lm[j];
                ls_j = m1_ls[j]

                should_stop = False
                if check_stop:
                    fsp = lm_j - e_alpha - e_beta * ls_j
                    dev = (fsp - e_spread) if e_dir == 1 else (e_spread - fsp)
                    if dev > sl_mult * e_std:
                        should_stop = True

                br = (cm_j - e_main) / e_main if e_main != 0.0 else 0.0
                er = (cs_j - e_sub) / e_sub if e_sub != 0.0 else 0.0
                gross_r = (br - e_beta * er) if e_dir == -1 else (e_beta * er - br)
                norm = gross_r / total_exp if total_exp > 0.0 else 0.0

                if should_stop:
                    cum_r += norm
                    fc += 1
                    t_i = m1t_i64[j];
                    val = cum_r * 100.0
                    if eq_n > 0 and eq_t[eq_n - 1] == t_i:
                        eq_v[eq_n - 1] = val;
                        eq_fc[eq_n - 1] = fc
                    else:
                        eq_t[eq_n] = t_i;
                        eq_v[eq_n] = val;
                        eq_fc[eq_n] = fc
                        eq_n += 1

                    seg_gross = _calc_norm_nb(seg_m, seg_s, cm_j, cs_j, e_beta, e_dir)
                    if n_trades < max_trades:
                        trade_pnls[n_trades] = seg_gross * 100.0
                        trade_durs[n_trades] = (m1t_i64[j] - seg_open_i64) / 60000000000.0
                        n_trades += 1

                    # ★ 修复幽灵重入：轻量级抹除后续残余信号
                    signals[i] = 0
                    k = i + 1
                    # 一直往后清零，直到遇到平仓信号 (0) 或反向开仓信号 (-e_dir)
                    while k < n_bars and signals[k] == e_dir:
                        signals[k] = 0
                        k += 1

                    in_trade = False;
                    e_dir = 0
                    break
                else:
                    t_i = m1t_i64[j];
                    val = (cum_r + norm) * 100.0
                    if eq_n > 0 and eq_t[eq_n - 1] == t_i:
                        eq_v[eq_n - 1] = val;
                        eq_fc[eq_n - 1] = fc
                    else:
                        eq_t[eq_n] = t_i;
                        eq_v[eq_n] = val;
                        eq_fc[eq_n] = fc
                        eq_n += 1

        mark_main = m1_cm[r - 1];
        mark_sub = m1_cs[r - 1]

    if eq_n == 0 and n_bars > 0:
        eq_t[0] = times_i64[n_bars - 1] + bd_i64
        eq_v[0] = cum_r * 100.0;
        eq_fc[0] = fc
        eq_n = 1

    return (eq_v[:eq_n].copy(), eq_t[:eq_n].copy(), eq_fc[:eq_n].copy(),
            trade_pnls[:n_trades].copy(), trade_durs[:n_trades].copy(),
            n_trades, e_dir, cum_r, mark_main, mark_sub)


# ---------- Python 辅助 ----------

def _build_bar_mapping(times_i64, m1t_i64, bd_i64, n):
    bar_l = np.searchsorted(m1t_i64, times_i64, side='left')
    end_t = np.empty(n, dtype=np.int64)
    if n > 1:
        end_t[:-1] = times_i64[1:]
    if n > 0:
        end_t[-1] = times_i64[-1] + bd_i64
    bar_r = np.searchsorted(m1t_i64, end_t, side='left')
    return bar_l, bar_r


def _compute_signals_only(pc, cfg, z_lookback, z_entry, z_exit, delta, ve,
                           min_hold_bars, cooldown_bars, max_hold_bars):
    """
    只计算信号数组, 不创建 DataFrame.
    返回 (signals, sig_fb, sig_fa, sig_es, sig_estd, stop_f, kill_f)
    以及信号上下文参数 (供 rebuild 使用).
    """
    lm = pc['log_main']; ls = pc['log_sub']
    betas = pc['betas']; alphas = pc['alphas']; sp = pc['spreads']
    z_c = pc['z_clean']; rs_c = pc['rs_clean']
    ni_c = pc['ni_clean']; trP_c = pc['trP_clean']

    result = generate_signals(
        z_c, sp, betas, alphas, lm, ls, rs_c, ni_c, trP_c,
        z_lookback, z_entry, z_exit,
        min_hold_bars, cooldown_bars, max_hold_bars,
        cfg.sl_mult, cfg.single_leg_fee,
        cfg.ni_entry_gate, cfg.kf_kill_trace_mult,
        cfg.kf_kill_ni_consec, cfg.kf_kill_ni_threshold,
        cfg.beta_min, cfg.beta_max)

    raw_sigs, fb, fa, es, estd, sf, kf = result[0], result[1], result[2], result[3], result[4], result[5], result[6]

    do_shift = cfg.use_next_bar_exec
    signals = shift_or_copy(raw_sigs, do_shift).astype(np.int64)
    sig_fb  = shift_or_copy(fb, do_shift)
    sig_fa  = shift_or_copy(fa, do_shift)
    sig_es  = shift_or_copy(es, do_shift)
    sig_estd = shift_or_copy(estd, do_shift)
    stop_f  = shift_or_copy(sf, do_shift).astype(np.int64)
    kill_f  = shift_or_copy(kf, do_shift).astype(np.int64)

    return signals, sig_fb, sig_fa, sig_es, sig_estd, stop_f, kill_f


def _fast_scan_bt(pc, times_i64, close_main, close_sub, log_main, log_sub,
                  m1c, cfg, tf_min,
                  z_lookback, z_entry, z_exit, delta, ve,
                  min_hold_bars, cooldown_bars, max_hold_bars,
                  fee_real):
    n = len(times_i64)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        s = {'Total Return': 0, 'total_trades': 0, 'Win Rate': 0,
             'avg_profit_per_trade': 0, 'Max Drawdown': 0,
             'Profit Factor': 0, 'avg_duration': 0, '最长持仓时间': 0}
        return empty, s, empty, s

    # 1) 生成信号 (注意：我们不在乎 stop_f 和 kill_f，不再传入 Numba)
    signals, sig_fb, sig_fa, sig_es, sig_estd, _, _ = \
        _compute_signals_only(pc, cfg, z_lookback, z_entry, z_exit, delta, ve,
                              min_hold_bars, cooldown_bars, max_hold_bars)

    # 2) 1 分钟数据
    m1t_i64 = m1c['times'].view(np.int64)
    m1_cm = m1c['close_main']; m1_cs = m1c['close_sub']
    m1_lm = m1c['log_main'];   m1_ls = m1c['log_sub']

    bd_i64 = np.int64(int(tf_min) * 60_000_000_000)
    bar_l, bar_r = _build_bar_mapping(times_i64, m1t_i64, bd_i64, n)

    # 3) 运行极致精简的 numba 循环
    (eq_v_0, eq_t, eq_fc,
     t_pnls_0, t_durs, n_tr,
     final_dir, cum_r_0, _, _) = _fast_bt_loop(
        n,
        signals, sig_fb, sig_fa, sig_es, sig_estd,
        close_main, close_sub, log_main, log_sub,
        m1_cm, m1_cs, m1_lm, m1_ls, m1t_i64,
        bar_l, bar_r, times_i64, bd_i64,
        cfg.sl_mult,
        0,
    )

    f = fee_real

    # 4a) fee=0 结果
    eff_0 = t_pnls_0.copy()
    if final_dir != 0 and len(eq_v_0) > 0:
        closed_sum_0 = float(np.sum(t_pnls_0))
        eff_0 = np.append(eff_0, float(eq_v_0[-1]) - closed_sum_0)
    stats_0 = _fast_stats(eff_0, eq_v_0, t_durs, n_tr)

    # 4b) fee=real 结果
    t_pnls_r = t_pnls_0 - 200.0 * f
    eq_v_r = eq_v_0 - eq_fc.astype(np.float64) * f * 100.0
    eff_r = t_pnls_r.copy()
    if final_dir != 0 and len(eq_v_r) > 0:
        closed_sum_r = float(np.sum(t_pnls_r))
        eff_r = np.append(eff_r, float(eq_v_r[-1]) - closed_sum_r)
    stats_r = _fast_stats(eff_r, eq_v_r, t_durs, n_tr)

    return eff_0, stats_0, eff_r, stats_r


def _fast_stats(pnl_arr, eq_curve, durs, n_trades):
    """从数组直接计算 backtest_stats, 无需 DataFrame."""
    nt = len(pnl_arr)
    if nt > 0:
        tot_r = float(np.sum(pnl_arr))
        avg_r = float(np.mean(pnl_arr))
        wr = float(np.sum(pnl_arr > 0)) / nt
        gp = float(np.sum(pnl_arr[pnl_arr > 0]))
        gl = float(abs(np.sum(pnl_arr[pnl_arr <= 0])))
        pf = (gp / gl) if gl > 0 else (float('inf') if gp > 0 else 0.0)
    else:
        tot_r = avg_r = wr = pf = 0.0

    avg_dur = float(np.mean(durs)) if n_trades > 0 else 0.0
    max_dur = float(np.max(durs)) if n_trades > 0 else 0.0

    if eq_curve is not None and len(eq_curve) > 0:
        total_ret = float(eq_curve[-1])
        eq_ext = np.empty(len(eq_curve) + 1, dtype=np.float64)
        eq_ext[0] = 0.0
        eq_ext[1:] = eq_curve
    else:
        total_ret = tot_r
        if nt > 0:
            eq_ext = np.empty(nt + 1, dtype=np.float64)
            eq_ext[0] = 0.0
            np.cumsum(pnl_arr, out=eq_ext[1:])
        else:
            eq_ext = np.array([0.0])

    peak = np.maximum.accumulate(eq_ext)
    mdd = float(np.min(eq_ext - peak))

    return {
        'Total Return': total_ret, 'total_trades': nt, 'Win Rate': wr,
        'avg_profit_per_trade': avg_r, 'Max Drawdown': mdd,
        'Profit Factor': pf, 'avg_duration': avg_dur, '最长持仓时间': max_dur,
    }


# =============================================================================
# §9 参数灵敏度扫描 (并行) — 替换原有版本
# =============================================================================
_G_BASE = None; _G_M1C = None; _G_LM = None; _G_LS = None
_G_TIMES_I64 = None; _G_CM = None; _G_CS = None


def _init_worker(base_df, m1c, lm, ls, times_i64, cm, cs):
    global _G_BASE, _G_M1C, _G_LM, _G_LS, _G_TIMES_I64, _G_CM, _G_CS
    _G_BASE = base_df; _G_M1C = m1c; _G_LM = lm; _G_LS = ls
    _G_TIMES_I64 = times_i64; _G_CM = cm; _G_CS = cs


def _eval_group_worker(args):
    # 解包新增的 task_id 和 total_tasks
    task_id, total_tasks, dpd, ve, lb_h, delta, tf_min, cd, mhb, combos = args
    pid = os.getpid()
    total_combos = len(combos)
    start_time = time.time()

    print(f"[Worker {pid}] 📥 收到任务包 [{task_id}/{total_tasks}]: dpd={dpd}, ve={ve}, lb_h={lb_h} | 共 {total_combos} 个组合")
    try:
        global _G_BASE, _G_M1C, _G_LM, _G_LS, _G_TIMES_I64, _G_CM, _G_CS

        # 1. KF + Rolling (与原版一致)
        t0_kf = time.time()
        kf_st = (0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
        betas, alphas, sp, ni, trP, _, z_raw, rs_raw, z_c, rs_c, ni_c, trP_c = \
            _run_kf_and_rolling(_G_LS, _G_LM, delta, ve, kf_st, hours_to_bars(lb_h, tf_min))
        print(f"[Worker {pid}] ⚡ 包 [{task_id}/{total_tasks}] KF与Rolling初始化完成，耗时: {time.time() - t0_kf:.2f}s")

        pc = dict(betas=betas, alphas=alphas, spreads=sp, ni=ni, trP=trP,
                  z_raw=z_raw, rs_raw=rs_raw, z_clean=z_c, rs_clean=rs_c,
                  ni_clean=ni_c, trP_clean=trP_c, log_main=_G_LM, log_sub=_G_LS)
        z_lb = hours_to_bars(lb_h, tf_min)
        results = []

        # 2. 遍历组合
        for idx, (ze, zx, mh_h) in enumerate(combos):
            t_combo_start = time.time()
            mh = hours_to_bars(mh_h, tf_min)

            # 使用快速回测
            e0, s0, er, sr = _fast_scan_bt(
                pc, _G_TIMES_I64, _G_CM, _G_CS, _G_LM, _G_LS,
                _G_M1C, CFG, tf_min,
                z_lb, ze, zx, delta, ve,
                mh, cd, mhb,
                CFG.single_leg_fee)

            combo_total_time = time.time() - t_combo_start
            # 组合级别的日志（如果不嫌刷屏可以保留，方便观察卡在哪个参数）
            # print(f"[Worker {pid}]   包[{task_id}] 组合[{idx + 1}/{total_combos}] z_e={ze}, z_x={zx}, mh={mh_h} | "
            #       f"耗时:{combo_total_time:.2f}s 当前时间: {datetime.now().strftime('%H:%M:%S')}")

            if len(e0) < 15:
                continue

            fpt = s0['avg_profit_per_trade'] - sr['avg_profit_per_trade']

            results.append({
                'z_entry': ze, 'z_exit': zx,
                'lookback_hours': lb_h, 'z_lookback_bars': z_lb,
                'delta_per_day': dpd, 'delta_per_bar': delta, 've': ve,
                'min_hold_hours': mh_h, 'min_hold_bars': mh,
                'trades': s0['total_trades'],
                'zero_total_ret': round(s0['Total Return'], 2),
                'zero_avg_pnl': round(s0['avg_profit_per_trade'], 4),
                'zero_win_rate': round(s0['Win Rate'], 4),
                'zero_pf': round(s0['Profit Factor'], 3),
                'zero_max_dd': round(s0['Max Drawdown'], 2),
                'real_total_ret': round(sr['Total Return'], 2),
                'real_avg_pnl': round(sr['avg_profit_per_trade'], 4),
                'real_win_rate': round(sr['Win Rate'], 4),
                'real_pf': round(sr['Profit Factor'], 3),
                'real_max_dd': round(sr['Max Drawdown'], 2),
                'fee_per_trade': round(fpt, 4),
                'avg_hold_hrs': round(s0['avg_duration'] / 60, 1),
                'max_hold_hrs': round(s0['最长持仓时间'] / 60, 1),
            })

        print(f"[Worker {pid}] ✅ 任务包 [{task_id}/{total_tasks}] 处理完毕，产出 {len(results)} 条有效结果 当前时间: {datetime.now().strftime('%H:%M:%S')} 耗时: {time.time() - start_time:.2f}s\n")
        return results
    except Exception as e:
        print(f"[Worker {pid}] ❌ 任务包 [{task_id}/{total_tasks}] 报错: {e}")
        traceback.print_exc()
        return []


def parameter_sensitivity_scan(df_file, timeframe='15min', n_workers=20):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    key = os.path.basename(df_file).replace('.csv', '')

    odf = pd.read_csv(df_file)
    odf['open_time'] = pd.to_datetime(odf['open_time'])
    if timeframe != '1min':
        df = resample_data(odf, timeframe)
        print(f"[{key}] 重采样 {len(odf)} → {len(df)} ({timeframe})")
    else:
        df = odf.copy()

    tf_min = get_tf_minutes(timeframe)
    m1c = build_1min_cache(odf) if (CFG.use_next_bar_exec or CFG.use_1min_path) else None
    grids = Config.param_grids()
    cd = hours_to_bars(1.0, tf_min)
    mhb = days_to_bars(4.0, tf_min)
    cm = df['close_main'].values.astype(np.float64)
    cs = df['close_sub'].values.astype(np.float64)
    lm = np.log(cm)
    ls = np.log(cs)
    times_i64 = pd.to_datetime(df['open_time']).values.astype('datetime64[ns]').view(np.int64)
    base = df[['open_time', 'close_main', 'close_sub']].copy()

    tasks_raw = []
    for dpd in grids['delta_per_day']:
        d = delta_per_day_to_bar(dpd, tf_min)
        for ve in grids['ve']:
            for lbh in grids['lookback_hours']:
                for ze in grids['z_entry']:
                    fast = [(ze, zx, mh) for zx in grids['z_exit'] if zx < ze
                            for mh in grids['min_hold_hours']]
                    if fast:
                        tasks_raw.append((dpd, ve, lbh, d, tf_min, cd, mhb, fast))

    total_tasks = len(tasks_raw)
    # 为每一个任务包加上序号 task_id 和 total_tasks
    tasks = [(i + 1, total_tasks) + t for i, t in enumerate(tasks_raw)]

    print(f"🚀 已生成 {total_tasks} 个任务包，准备分发给 {n_workers} 个 Worker 执行...")

    # 预热 Numba
    _d = np.zeros(5, dtype=np.float64)
    _di = np.zeros(5, dtype=np.int64)
    try:
        kalman_filter(_d, _d, 0.001, 0.001)
        generate_signals(_d, _d, _d, _d, _d, _d, _d, _d, _d,
                         2, 2.0, 0.5, 1, 1, 1, 1.0, 0.001, 3.0, 5.0, 3, 3.0)
        scan_1min_equity(_d, _d, _d, _d, _di, 0, 2, 1., 1., 1., 1,
                         0., 0., 0., 1., 0., 0.001)
        # 预热 _fast_bt_loop
        _si = np.zeros(5, dtype=np.int64)
        _bl = np.zeros(5, dtype=np.intp)
        _br = np.ones(5, dtype=np.intp)
        try:
            _fast_bt_loop(
                5,
                _si, _d, _d, _d, _d,  # signals_in, sig_fb_in...
                _d, _d, _d, _d,  # close_main, close_sub, log_main, log_sub
                _d, _d, _d, _d, _di,  # m1_cm... m1_ls, m1t_i64
                _bl, _br, _di, np.int64(60_000_000_000),  # bar_l, bar_r, times_i64, bd_i64
                1.0,  # sl_mult
                0  # prev_sig_init
            )
        except Exception:
            pass
    except Exception:
        pass

    results = []
    t0 = time.time()
    completed_tasks = 0  # 记录已完成的任务包数

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             initargs=(base, m1c, lm, ls, times_i64, cm, cs)) as exe:
        futs = {exe.submit(_eval_group_worker, t): t for t in tasks}
        for f in as_completed(futs):
            completed_tasks += 1
            r = f.result()
            if r:
                results.extend(r)
            # 实时刷新总进度，原位打印
            print(
                f"  [总进度 {completed_tasks}/{total_tasks}] 有效策略累积: {len(results)} 条 | 已耗时: {time.time() - t0:.1f}s",
                end='\r')

    elapsed = time.time() - t0
    print(f"\n扫描完成: {len(results)} 条, {elapsed:.1f}s")
    if not results:
        return pd.DataFrame()

    rdf = pd.DataFrame(results)
    os.makedirs('backtest_pair', exist_ok=True)
    out = f'backtest_pair/param_scan_{key}_{timeframe}.csv'
    rdf.to_csv(out, index=False)

    print(f"\n{'=' * 100}\nTOP 20 | {key} | {timeframe}\n{'=' * 100}")
    top20 = rdf.sort_values('real_total_ret', ascending=False).head(20)
    cols = ['z_entry', 'z_exit', 'lookback_hours', 'delta_per_day', 'min_hold_hours',
            'trades', 'zero_avg_pnl', 'real_avg_pnl', 'fee_per_trade',
            'real_total_ret', 'real_win_rate', 'real_pf', 'real_max_dd', 'avg_hold_hrs']
    print(top20[cols].to_string(index=False))

    print(f"\n{'=' * 100}\n各维度边际效应 | {key} | {timeframe}\n{'=' * 100}")
    for dim in ['z_entry', 'z_exit', 'lookback_hours', 'delta_per_day', 'min_hold_hours']:
        g = rdf.groupby(dim).agg({'zero_avg_pnl': 'mean', 'real_avg_pnl': 'mean',
                                  'trades': 'mean', 'real_total_ret': 'mean'}).round(4)
        print(f"\n--- {dim} ---\n{g.to_string()}")

    prof = rdf[rdf['real_total_ret'] > 0]
    print(f"\n{'=' * 80}\n核心判断 | {key} | {timeframe}\n{'=' * 80}")
    print(f"  总: {len(rdf)} | 盈利: {len(prof)}")
    if len(prof):
        print(f"  最高: {prof['real_total_ret'].max():.2f}% | "
              f"每笔: {prof['real_avg_pnl'].min():.4f}%~{prof['real_avg_pnl'].max():.4f}% | "
              f"持仓: {prof['avg_hold_hrs'].min():.1f}h~{prof['avg_hold_hrs'].max():.1f}h")
        print(f"  ✅ 存在盈利区间, 值得WF验证")
    else:
        print(f"  ❌ 无盈利组合")
    print(f"结果: {out}")
    return rdf

# =============================================================================
# §10 Walk-Forward 优化
# =============================================================================
def _param_neighbors(key, grids):
    nbs = []
    for di in range(len(key)):
        g = grids[di]; v = key[di]
        if v in g:
            p = g.index(v)
            for off in [-1, 1]:
                np_ = p + off
                if 0 <= np_ < len(g):
                    nk = list(key); nk[di] = g[np_]; nbs.append(tuple(nk))
    return nbs


def _wf_force_close(oos_trades, oos_eq_raw, test_es, bnd_eq, test_portion):
    if test_es is None or float(test_es[2]) == 0.0 or len(oos_eq_raw) == 0:
        return oos_trades, test_es, False

    oos_eq_raw[-1] -= CFG.single_leg_fee * 100.0
    closed_sum = float(oos_trades['net_pnl'].sum()) if not oos_trades.empty else 0.0
    fc_pnl = round(float(oos_eq_raw[-1]) - bnd_eq - closed_sum, 4)

    # 从 exec_state 读取持仓信息
    sn = len(test_es)
    g = lambda i, d=0.0: test_es[i] if i < sn else d
    d, b = int(float(g(2))), float(g(3))
    em, esub = float(g(0)), float(g(1))
    mm, ms = float(g(6)), float(g(7))

    fc_open = pd.to_datetime(test_portion['open_time'].iloc[0]) if len(test_portion) else pd.NaT
    fc_close = pd.to_datetime(test_portion['open_time'].iloc[-1]) if len(test_portion) else pd.NaT
    fc_dur = 0.0
    e_time = g(5, _NAT)
    if not pd.isna(e_time) and not pd.isna(fc_close):
        try: fc_dur = (pd.Timestamp(fc_close) - pd.Timestamp(e_time)).total_seconds() / 60
        except Exception: pass

    dir_s = (f'做多价差 (Long BTC / Short {b:.3f} ETH)' if d == -1
             else f'做空价差 (Short BTC / Long {b:.3f} ETH)')
    fc = pd.DataFrame([{
        'open_time': fc_open, 'actual_open_time': e_time,
        'close_time': fc_close, 'duration_mins': fc_dur, 'direction': dir_s,
        'btc_roi': f"{((mm-em)/em*100):.2f}" if em > 0 else "0.00",
        'eth_roi': f"{((ms-esub)/esub*100):.2f}" if esub > 0 else "0.00",
        'net_pnl': fc_pnl, 'open_main': em, 'close_main': mm,
        'open_sub': esub, 'close_sub': ms,
        'entry_z': np.nan, 'exit_z': np.nan,
        'entry_spread': np.nan, 'exit_spread': np.nan,
        'entry_beta': b, 'exit_beta': b,
        'entry_alpha': np.nan, 'exit_alpha': np.nan,
        'signal': d, 'inherited_from_prev_window': True,
        'close_reason': 'wf_boundary_force_close',
    }])
    oos_trades = pd.concat([oos_trades, fc], ignore_index=True) if not oos_trades.empty else fc.copy()
    st = list(test_es); st[2] = 0.0
    print(f"  [FIX] OOS强制平仓: dir={d}, pnl={fc_pnl:.4f}%")
    return oos_trades, tuple(st), True


def walk_forward_optimization(df_file, timeframe='15min',
                              train_months=6, test_months=1,
                              cooldown_hours=1.0, max_hold_days=4.0):
    key = os.path.basename(df_file).replace('.csv', '') + f'_{timeframe}'
    odf = pd.read_csv(df_file);
    odf['open_time'] = pd.to_datetime(odf['open_time'])
    df = resample_data(odf, timeframe) if timeframe != '1min' else odf.copy()
    tf_min = get_tf_minutes(timeframe)
    cd = hours_to_bars(cooldown_hours, tf_min);
    mhb = days_to_bars(max_hold_days, tf_min)
    m1c = build_1min_cache(odf) if (CFG.use_next_bar_exec or CFG.use_1min_path) else None
    grids = Config.param_grids()
    pgrids = [grids['z_entry'], grids['z_exit'], grids['lookback_hours'],
              grids['delta_per_day'], grids['ve'], grids['min_hold_hours']]
    min_trades = max(18, train_months * 3)

    all_oos_trades, all_oos_stats, all_oos_eq, all_oos_eff = [], [], [], []
    param_hist, deg_log, diverse_hist = [], [], []
    start, end = df['open_time'].min(), df['open_time'].max()
    train_d, test_d = pd.DateOffset(months=train_months), pd.DateOffset(months=test_months)
    cur, wid = start, 0

    while cur + train_d + test_d <= end:
        tr_end, te_end = cur + train_d, cur + train_d + test_d
        tr_df = df[(df['open_time'] >= cur) & (df['open_time'] < tr_end)].copy()
        te_df = df[(df['open_time'] >= tr_end) & (df['open_time'] < te_end)].copy()
        if len(tr_df) < 500 or len(te_df) < 50:
            cur += test_d;
            wid += 1;
            continue

        print(f"\n{'=' * 60}\nW{wid}: Train [{cur.date()}~{tr_end.date()}] → Test [{tr_end.date()}~{te_end.date()}]")

        tr_start_ns, tr_end_ns = _segment_bounds_ns(tr_df, tf_min)
        lm = np.log(tr_df['close_main'].values.astype(float))
        ls = np.log(tr_df['close_sub'].values.astype(float))
        base_tr = tr_df[['open_time', 'close_main', 'close_sub']].copy()

        cands = {};
        all_scores = {};
        combo_n = valid_n = 0

        for dpd in grids['delta_per_day']:
            dlt = delta_per_day_to_bar(dpd, tf_min)
            for ve in grids['ve']:
                # 【修复】移除无意义的 _run_kf_and_rolling(..., z_lookback=1)
                # 直接调用底层的 kalman_filter，避免在未知 lookback 时浪费 CPU 算 Pandas Rolling
                kf_st = (0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
                betas, alphas, sp, ni, trP, _, _, _, _, _, _ = kalman_filter(
                    ls, lm, delta=dlt, ve=ve,
                    init_alpha=kf_st[0], init_beta=kf_st[1],
                    init_P00=kf_st[2], init_P01=kf_st[3],
                    init_P10=kf_st[4], init_P11=kf_st[5]
                )
                ni_c = clean_nan(ni)
                trP_c = clean_nan(trP)

                for lbh in grids['lookback_hours']:
                    z_lb = hours_to_bars(lbh, tf_min)
                    # 在此处执行唯一的有效 Rolling 计算
                    ss = pd.Series(sp)
                    rm = ss.rolling(z_lb, min_periods=z_lb).mean()
                    rs = ss.rolling(z_lb, min_periods=z_lb).std()
                    z_raw = ((ss - rm) / rs).values;
                    rs_raw = rs.values
                    z_c = clean_nan(z_raw);
                    rs_c = clean_nan(rs_raw)

                    pc = dict(betas=betas, alphas=alphas, spreads=sp, ni=ni, trP=trP,
                              z_raw=z_raw, rs_raw=rs_raw, z_clean=z_c, rs_clean=rs_c,
                              ni_clean=ni_c, trP_clean=trP_c, log_main=lm, log_sub=ls)

                    for ze in grids['z_entry']:
                        for zx in grids['z_exit']:
                            if zx >= ze: continue
                            for mh_h in grids['min_hold_hours']:
                                mh = hours_to_bars(mh_h, tf_min)
                                combo_n += 1
                                pk = (ze, zx, lbh, dpd, ve, mh_h)
                                try:
                                    fdf, _, _ = build_signal_df(
                                        base_tr, z_lookback=z_lb, z_entry=ze, z_exit=zx,
                                        delta=dlt, ve=ve, min_hold_bars=mh,
                                        cooldown_bars=cd, max_hold_bars=mhb, precomputed=pc)

                                    # 【修复】接住 tdf，解决 n_closed 永远是 0 的 Bug
                                    tdf, eq, et, _, eff = _quick_bt(fdf, m1c, tf_min)

                                    if len(eff) >= 2 and np.std(eff) > 0:
                                        rs_ = np.mean(eff) / np.std(eff) * np.sqrt(len(eff))
                                    elif len(eff) >= 2:
                                        rs_ = np.sign(np.mean(eff)) * 0.01
                                    else:
                                        rs_ = 0.0
                                    all_scores[pk] = rs_

                                    if len(eff) < min_trades or np.std(eff) == 0 or np.mean(eff) <= 0:
                                        continue
                                    _, _, _, _, e15 = _quick_bt(fdf, m1c, tf_min,
                                                                fee=CFG.single_leg_fee * 1.5)
                                    if len(e15) == 0 or np.mean(e15) <= 0: continue

                                    mr = _split_monthly_returns(tr_start_ns, tr_end_ns, et, eq, train_months)
                                    if len(mr) < 3: continue
                                    pm = sum(1 for r in mr if r > 0)
                                    if pm < max(2, int(len(mr) * 0.6)): continue

                                    ma = np.array(mr)
                                    stab = (np.mean(ma) / np.std(ma)) if np.std(ma) > 0 else (
                                        np.sign(np.mean(ma)) * 10 if np.mean(ma) > 0 else 0)
                                    comp = rs_ * (1 + max(0, stab)) / 2

                                    if len(eq) > 0:
                                        _e = np.concatenate(([0.0], eq))
                                        if float(np.min(_e - np.maximum.accumulate(_e))) < CFG.wf_max_train_mdd:
                                            continue

                                    valid_n += 1
                                    tr_ret = float(eq[-1]) if len(eq) else float(np.sum(eff))
                                    cands[pk] = {
                                        'score': comp, 'raw_t': rs_, 'stab': stab,
                                        'pm': pm, 'tm': len(mr),
                                        'params': {'z_entry': ze, 'z_exit': zx,
                                                   'lookback_hours': lbh, 'z_lookback': z_lb,
                                                   'delta_per_day': dpd, 'delta': dlt, 've': ve,
                                                   'min_hold_hours': mh_h, 'min_hold': mh},
                                        'n_trades': len(eff),
                                        'n_closed': len(tdf),  # 【修复】传入正确的闭仓次数
                                        'tr_ret': tr_ret, 'tr_avg': float(np.mean(eff)),
                                    }
                                except Exception:
                                    all_scores[pk] = 0.0

        if not cands:
            print(f"  W{wid}: 无通过筛选 ({combo_n} 组合)");
            cur += test_d;
            wid += 1;
            continue

        # 高原选择 (与之前保持一致)
        sk = sorted(cands, key=lambda k: cands[k]['score'], reverse=True)
        top_n = max(1, len(sk) // 5);
        top_set = set(sk[:top_n])
        best_ps, best_k = -np.inf, sk[0]
        for pk in top_set:
            nbs = _param_neighbors(pk, pgrids)
            ns_ = [all_scores[n] for n in nbs if n in all_scores]
            avg_nb = float(np.median(ns_)) if ns_ else 0.0
            ps = cands[pk]['score'] * 0.5 + avg_nb * 0.5
            if ps > best_ps: best_ps, best_k = ps, pk
        ch = cands[best_k];
        bp = ch['params']

        param_hist.append({'window_id': wid, **bp, 'score': ch['score'],
                           'raw_t': ch['raw_t'], 'stab': round(ch['stab'], 4),
                           'pm': ch['pm'], 'tm': ch['tm'], 'plateau': best_ps,
                           'trades': ch['n_trades'],
                           'tr_closed': ch['n_closed'],  # 将修复的次数记录进 CSV
                           'tr_ret': round(ch['tr_ret'], 4),
                           'tr_avg': round(ch['tr_avg'], 4),
                           'combos': combo_n, 'passed': len(cands)})
        print(f"  Best: {bp}\n  t={ch['raw_t']:.4f} stab={ch['stab']:.4f} "
              f"({ch['pm']}/{ch['tm']}mo+) trades={ch['n_trades']} closed={ch['n_closed']} "
              f"ret={ch['tr_ret']:.2f}% plateau={best_ps:.4f}")

        # 多样化集成 (与之前保持一致)
        lb_groups = {}
        for pk, info in cands.items():
            lb = info['params']['lookback_hours']
            lb_groups.setdefault(lb, []).append((pk, info))
        div = []
        for lb in sorted(lb_groups):
            g = sorted(lb_groups[lb], key=lambda x: x[1]['score'], reverse=True)
            if g: div.append(g[0])
        if len(div) >= 2:
            ei = [{'lookback_hours': i['params']['lookback_hours'],
                   'z_entry': i['params']['z_entry'], 'score': round(i['score'], 4)}
                  for _, i in div[:5]]
            diverse_hist.append({'window_id': wid, 'candidates': ei})

        # OOS 测试 (与之前保持一致)
        combined = pd.concat([tr_df, te_df], ignore_index=True)
        test_start = te_df['open_time'].iloc[0]
        test_full, _, _ = build_signal_df(
            combined, z_lookback=bp['z_lookback'], z_entry=bp['z_entry'],
            z_exit=bp['z_exit'], delta=bp['delta'], ve=bp['ve'],
            min_hold_bars=bp['min_hold'], cooldown_bars=cd, max_hold_bars=mhb)

        tmask = test_full['open_time'] >= test_start
        tr_part = test_full[~tmask].copy();
        te_part = test_full[tmask].copy()
        _, tr_eq, _, tr_es = compute_equity_and_trades(tr_part, min1_cache=m1c, tf_minutes=tf_min)
        bnd_eq = tr_eq[-1] if len(tr_eq) else 0.0

        oos_t, oos_eq, oos_et, te_es = compute_equity_and_trades(
            te_part, min1_cache=m1c, exec_state=tr_es, tf_minutes=tf_min)
        oos_t, te_es, _ = _wf_force_close(oos_t, oos_eq, te_es, bnd_eq, te_part)

        oos_eq_adj = oos_eq - bnd_eq if len(oos_eq) else oos_eq
        oos_eff = build_effective_pnls(oos_t, oos_eq_adj, te_es)

        if len(oos_eq_adj): all_oos_eq.append(oos_eq_adj + (all_oos_eq[-1][-1] if all_oos_eq else 0))
        if len(oos_eff): all_oos_eff.append(oos_eff)
        if not oos_t.empty:
            oos_t['window_id'] = wid;
            oos_t['best_params'] = str(bp)
            all_oos_trades.append(oos_t)

        st = backtest_stats(oos_t, oos_eq_adj, effective_pnls=oos_eff)
        if st:
            st['window_id'] = wid;
            all_oos_stats.append(st)
            print(f"  OOS: trades={st['total_trades']} avg={st['avg_profit_per_trade']:.4f}% "
                  f"win={st['Win Rate']:.2%} ret={st['Total Return']:.2f}% MDD={st['Max Drawdown']:.4f}%")
            oos_avg = st['avg_profit_per_trade']
            deg = (ch['tr_avg'] / oos_avg) if oos_avg > 0 else float('inf')
            deg_log.append({'window_id': wid, 'tr_avg': round(ch['tr_avg'], 4),
                            'oos_avg': round(oos_avg, 4), 'deg': round(deg, 2) if np.isfinite(deg) else float('inf')})
            ds = f"{deg:.2f}" if np.isfinite(deg) else "inf"
            print(f"  退化比: {ds}")

        cur += test_d;
        wid += 1

    # ========== 汇总 (与之前完全保持一致) ==========
    print(f"\n{'=' * 60}\nWF 汇总 | {key}\n{'=' * 60}")
    os.makedirs('backtest_pair', exist_ok=True)

    coos = pd.concat(all_oos_trades, ignore_index=True) if all_oos_trades else pd.DataFrame()
    ceq = np.concatenate(all_oos_eq) if all_oos_eq else np.array([])
    ceff = np.concatenate(all_oos_eff) if all_oos_eff else np.array([])

    if not coos.empty:
        coos.to_csv(f'backtest_pair/{key}_wf_oos_trades.csv', index=False)

    if len(ceq) or not coos.empty:
        fin = backtest_stats(coos, ceq, effective_pnls=ceff)
        if fin:
            print(f"  总交易: {fin['total_trades']} | 总收益: {fin['Total Return']:.4f}%")
            print(f"  每笔: {fin['avg_profit_per_trade']:.4f}% | 胜率: {fin['Win Rate']:.2%} | "
                  f"PF: {fin['Profit Factor']:.4f} | MDD: {fin['Max Drawdown']:.4f}%")

        if all_oos_stats:
            odf = pd.DataFrame(all_oos_stats)
            for _, r in odf.iterrows():
                print(f"  W{int(r['window_id'])}: trades={int(r['total_trades'])} "
                      f"ret={r['Total Return']:.2f}% win={r['Win Rate']:.1%} "
                      f"pf={r['Profit Factor']:.2f} MDD={r['Max Drawdown']:.4f}%")
            ww = len(odf[odf['Total Return'] > 0]);
            tw = len(odf)
            if tw: print(f"\n  窗口胜率: {ww}/{tw} = {ww / tw:.1%}")

        if deg_log:
            ddf = pd.DataFrame(deg_log)
            for _, r in ddf.iterrows():
                d = r['deg'];
                f = " ⚠️" if (not np.isfinite(d) or d > 3) else ""
                print(f"  W{int(r['window_id'])}: tr={r['tr_avg']:.4f}% oos={r['oos_avg']:.4f}% "
                      f"deg={f'{d:.2f}' if np.isfinite(d) else 'inf'}{f}")
            fr = [r['deg'] for r in deg_log if np.isfinite(r['deg'])]
            if fr:
                am = np.mean(fr);
                hi = sum(1 for r in fr if r > 3)
                print(f"\n  平均退化: {am:.2f} | >3: {hi}/{len(fr)}")
                if am > 3:
                    print("  🔴 退化严重")
                elif am > 2:
                    print("  🟡 有退化")
                else:
                    print("  🟢 合理")
            ddf.to_csv(f'backtest_pair/{key}_oos_degradation.csv', index=False)

    if param_hist:
        ph = pd.DataFrame(param_hist)
        ph.to_csv(f'backtest_pair/{key}_param_history.csv', index=False)
        print(f"\n  --- 参数收敛性 ---")
        for c in ['z_entry', 'z_exit', 'lookback_hours', 'delta_per_day', 've', 'min_hold_hours']:
            if c in ph.columns:
                print(f"  {c}: mode={ph[c].mode().values} std={ph[c].std():.4f}")


# =============================================================================
# §11 主入口
# =============================================================================
if __name__ == '__main__':

    # 跳过 rebuild 是一种近似，但它对排名的影响极小。具体来说，回测结果的差异来源于**“影子冷却期 (Shadow Cooldown)”**
    target_dir = 'kline_data'
    suffix = '1m.csv'
    df_files = [os.path.join(target_dir, f)
                for f in os.listdir(target_dir)
                if os.path.isfile(os.path.join(target_dir, f))
                and f.endswith(suffix) and f.count('_') == 2]
    timeframes = ['1min', '5min', '10min', '15min', '30min']

    for df_file in df_files:
        for tf in timeframes:
            print(f"\n{'='*70}\n参数扫描 | {os.path.basename(df_file)} | {tf}\n{'='*70}")
            parameter_sensitivity_scan(df_file, timeframe=tf)

    walk_forward_optimization('kline_data/DOGE_ETH_1m.csv', timeframe='30min',
                              train_months=6, test_months=1,
                              cooldown_hours=1.0, max_hold_days=4.0)