"""
突破策略的信号生成以及回测（优化版）
"""
import itertools
import json
import math
import re
from multiprocessing import Pool
import multiprocessing as mp

import multiprocessing
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numba import njit
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
import ast

from tqdm import tqdm

def custom_merge_intervals(intervals):
    """
    归并区间 (忽略 revenue 信息，只处理 [start, end])，
    保证归并后的区间为不重叠且连续的。
    输入的 intervals 应该是按 start 排序的列表
    """
    if not intervals:
        return []
    merged = [[intervals[0][0], intervals[0][1]]]
    for s, e, _ in intervals[1:]:
        # “相连”也算重叠，所以条件用 <=
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def custom_compute_overlapping_metrics(intervals, union_intervals):
    """
    使用两路指针扫描计算重叠时长、重叠收益和重叠区间数量。
    输入：
      intervals: 已排序且格式为 (start, end, revenue) 的列表
      union_intervals: 归并后的区间列表，格式为 [start, end]
    对于每个区间：
      - 找到与归并区间发生重叠部分，计算重叠时长；
      - 若有重叠，则将该区间的 revenue 全计入，同时计数一次重叠。
    返回：
      total_overlap_time: 重叠时长总和
      total_overlap_revenue: 重叠收益总和
      overlap_count: 重叠的区间个数（即有重叠的 intervals 数量）
    """
    total_overlap_time = 0
    total_overlap_revenue = 0
    overlap_count = 0

    j = 0
    n_union = len(union_intervals)
    for s, e, rev in intervals:
        # 移动指针：跳过所有在当前区间左侧结束的归并区间
        while j < n_union and union_intervals[j][1] <= s:
            j += 1
        temp_overlap = 0
        k = j
        while k < n_union and union_intervals[k][0] < e:
            u_start, u_end = union_intervals[k]
            overlap = max(0, min(e, u_end) - max(s, u_start))
            temp_overlap += overlap
            k += 1
        if temp_overlap > 0:
            total_overlap_time += temp_overlap
            total_overlap_revenue += rev
            overlap_count += 1
    return total_overlap_time, total_overlap_revenue, overlap_count


def preprocess_row_custom(kai_data):
    """
    对每一行的 kai_data_df_tuples 进行解析（如果为字符串则转换），
    然后预计算:
      - 排序后的区间列表 (start, end, revenue)
      - 归并区间
      - 总持有时长与总收益
    """
    if isinstance(kai_data, str):
        kai_data = ast.literal_eval(kai_data)
    intervals = sorted(
        [(start, start + duration, revenue) for start, duration, revenue in kai_data],
        key=lambda x: x[0]
    )
    union_int = custom_merge_intervals(intervals)
    total_holding = sum(e - s for s, e, _ in intervals)
    total_revenue = sum(rev for _, _, rev in intervals)
    return {
        'intervals': intervals,
        'union': union_int,
        'total_holding': total_holding,
        'total_revenue': total_revenue,
    }


# 全局变量，用于存放所有行的预处理结果
unique_precomputed_data = []


def init_custom_worker(pre_data):
    global unique_precomputed_data
    unique_precomputed_data = pre_data


def process_pair_custom(pair):
    """
    每个 pair 的任务：
      参数为 (i, j)，直接从全局 unique_precomputed_data 中取出两行数据，
      分别对 row i 和 row j 计算重叠指标。
      同时统计重叠区间的个数及其占比（相对于各自的总区间个数）。
    """
    i, j = pair
    data1 = unique_precomputed_data[i]
    data2 = unique_precomputed_data[j]

    # 计算 row1 与 row2 的重叠情况
    overlap_time1, overlap_rev1, overlap_count1 = custom_compute_overlapping_metrics(data1['intervals'], data2['union'])
    # 计算 row2 与 row1 的重叠情况
    overlap_time2, overlap_rev2, overlap_count2 = custom_compute_overlapping_metrics(data2['intervals'], data1['union'])

    overlap_ratio1 = overlap_time1 / data1['total_holding'] if data1['total_holding'] > 0 else 0
    overlap_ratio2 = overlap_time2 / data2['total_holding'] if data2['total_holding'] > 0 else 0
    overlap_rev_ratio1 = overlap_rev1 / data1['total_revenue'] if data1['total_revenue'] > 0 else 0
    overlap_rev_ratio2 = overlap_rev2 / data2['total_revenue'] if data2['total_revenue'] > 0 else 0

    total_intervals1 = len(data1['intervals'])
    total_intervals2 = len(data2['intervals'])
    overlap_count_ratio1 = overlap_count1 / total_intervals1 if total_intervals1 > 0 else 0
    overlap_count_ratio2 = overlap_count2 / total_intervals2 if total_intervals2 > 0 else 0

    return {
        "row1_index": i,
        "row2_index": j,
        "total_holding1": data1['total_holding'],
        "total_holding2": data2['total_holding'],
        "overlapping_time1": overlap_time1,
        "overlapping_time2": overlap_time2,
        "overlap_ratio1": overlap_ratio1,
        "overlap_ratio2": overlap_ratio2,
        "total_revenue1": data1['total_revenue'],
        "total_revenue2": data2['total_revenue'],
        "overlapping_revenue1": overlap_rev1,
        "overlapping_revenue2": overlap_rev2,
        "overlapping_revenue_ratio1": overlap_rev_ratio1,
        "overlapping_revenue_ratio2": overlap_rev_ratio2,
        "overlap_count1": overlap_count1,
        "overlap_count2": overlap_count2,
        "overlap_count_ratio1": overlap_count_ratio1,
        "overlap_count_ratio2": overlap_count_ratio2,
    }



def iterative_search(func, base, step, max_range, tol):
    """
    在 [base - max_range, base + max_range] 内以步长 step 遍历，寻找使 func(candidate) 接近 0（|f(candidate)| < tol）的 candidate，
    若找到则返回该 candidate，否则返回 None。
    """
    candidates = np.arange(base - max_range, base + max_range + step, step)
    best_candidate = None
    best_val = float('inf')
    for candidate in candidates:
        val = abs(func(candidate))
        if val < tol:
            return candidate
        if val < best_val:
            best_val = val
            best_candidate = candidate
    return best_candidate if best_val < tol * 10 else None


def compute_future_signal(df, col_name):
    """
    根据历史行情数据 (df) 和信号名称 (col_name)，计算未来一个周期触发该信号所需的目标价格及比较方向。
    若上周期未满足前置条件，则返回 (None, None) 表示无效信号。

    参数:
      df: pandas.DataFrame，至少包含 "close", "high", "low" 列，
         数据按时间顺序排列（旧数据在前，新数据在后）。
      col_name: 信号名称字符串，格式为 "signalType_param1_param2_..._direction"，
                举例: "abs_20_2_long"、"boll_20_2_long"、"macross_12_26_long"、"macd_12_26_9_short"、
                "rsi_5_30_long"（RSI 设定在超卖 30）或 "cci_5_long"（CCI 长仓默认目标 -100）

    返回:
      tuple: (target_price, op)
         - target_price: float，未来周期需要达到/触发信号时的目标价格（四舍五入到小数点后4位）；若前置条件不满足，则为 None
         - op: str，对于多头信号，op 为 ">"，表示未来价格大于 target_price 时触发信号；
               对于空头信号，op 为 "<"；若前置条件不满足则返回 (None, None)。
    """
    parts = col_name.split('_')
    signal_type = parts[0]
    direction = parts[-1]

    # ------------------------ abs 信号 ------------------------
    if signal_type == 'abs':
        period = int(parts[1])
        abs_rate = float(parts[2]) / 100.0

        if len(df) < period + 1:
            raise ValueError("数据不足，无足够历史数据计算 abs 信号")

        if direction == "long":
            min_low = df['low'].iloc[-(period + 1):-1].min()
            target_price = round(min_low * (1 + abs_rate), 4)
            return target_price, ">"
        else:
            max_high = df['high'].iloc[-(period + 1):-1].max()
            target_price = round(max_high * (1 - abs_rate), 4)
            return target_price, "<"

    # ------------------------ relate 信号 ------------------------
    elif signal_type == 'relate':
        period = int(parts[1])
        percent = float(parts[2]) / 100.0

        if len(df) < period + 1:
            raise ValueError("数据不足，无足够历史数据计算 relate 信号")

        min_low = df['low'].iloc[-(period + 1):-1].min()
        max_high = df['high'].iloc[-(period + 1):-1].max()

        if direction == "long":
            target_price = round(min_low + percent * (max_high - min_low), 4)
            return target_price, ">"
        else:
            target_price = round(max_high - percent * (max_high - min_low), 4)
            return target_price, "<"

    # ------------------------ donchian 信号 ------------------------
    elif signal_type == 'donchian':
        period = int(parts[1])
        if len(df) < period + 1:
            raise ValueError("数据不足，无足够历史数据计算 donchian 信号")

        if direction == "long":
            highest_high = df['high'].iloc[-(period + 1):-1].max()
            target_price = round(highest_high, 4)
            return target_price, ">"
        else:
            lowest_low = df['low'].iloc[-(period + 1):-1].min()
            target_price = round(lowest_low, 4)
            return target_price, "<"

    # ------------------------ boll 信号 ------------------------
    elif signal_type == 'boll':
        period = int(parts[1])
        std_multiplier = float(parts[2])
        if len(df) < period:
            raise ValueError("数据不足，无足够历史数据计算 boll 信号")

        # 计算上周期 Bollinger 带（窗口取倒数 period+1 至倒数第1 行）
        hist_window = df['close'].iloc[-(period + 1):-1]
        if len(hist_window) < period:
            raise ValueError("数据不足，无足够历史数据计算 Bollinger 上周期指标")
        pre_ma = hist_window.mean()
        pre_std = hist_window.std(ddof=1)
        pre_lower = pre_ma - std_multiplier * pre_std
        pre_upper = pre_ma + std_multiplier * pre_std

        # 前置条件检查：多头要求上周期收盘价低于下轨；空头要求上周期收盘价高于上轨
        if direction == "long":
            if df['close'].iloc[-2] >= pre_lower:
                return None, None
        else:
            if df['close'].iloc[-2] <= pre_upper:
                return None, None

        # 模拟未来周期：新的窗口由最近 (period-1) 个收盘价加上未来价格 candidate 组成
        recent = df['close'].iloc[-(period - 1):]
        base = df['close'].iloc[-1]
        step = base * 0.001  # 0.1% 步长
        max_range = base * 0.1  # 最大 ±10%

        if direction == "long":
            # 对多头：要求 candidate = new_ma - std_multiplier * new_std
            def f(x):
                arr = np.append(recent.values, x)
                new_ma = arr.mean()
                new_std = np.std(arr, ddof=1)
                return x - (new_ma - std_multiplier * new_std)

            candidate = iterative_search(f, base, step, max_range, tol=1e-4)
            if candidate is None:
                return None, None
            return round(candidate, 4), ">"
        else:
            # 对空头：要求 candidate = new_ma + std_multiplier * new_std
            def f(x):
                arr = np.append(recent.values, x)
                new_ma = arr.mean()
                new_std = np.std(arr, ddof=1)
                return x - (new_ma + std_multiplier * new_std)

            candidate = iterative_search(f, base, step, max_range, tol=1e-4)
            if candidate is None:
                return None, None
            return round(candidate, 4), "<"

    # ------------------------ macross 信号 ------------------------
    elif signal_type == 'macross':
        fast_period = int(parts[1])
        slow_period = int(parts[2])
        if fast_period >= slow_period:
            raise ValueError("macross 信号中 fast_period 必须小于 slow_period")
        if len(df) < slow_period:
            raise ValueError("数据不足，无足够历史数据计算 macross 信号")

        # 计算上周期的均值
        fast_ma_prev = df['close'].iloc[-fast_period:].mean()
        slow_ma_prev = df['close'].iloc[-slow_period:].mean()
        # 前置条件：多头要求上周期 fast_ma < slow_ma；空头相反
        if direction == "long":
            if fast_ma_prev >= slow_ma_prev:
                return None, None
        else:
            if fast_ma_prev <= slow_ma_prev:
                return None, None

        # 模拟未来周期：新快均线 = (sum(最近 fast_period-1 个数据) + x)/fast_period，
        # 新慢均线 = (sum(最近 slow_period-1 个数据) + x)/slow_period，
        # 令两者相等求 x 的闭合解
        if fast_period > 1:
            fast_window = df['close'].iloc[-(fast_period - 1):]
            sum_fast = fast_window.sum()
        else:
            sum_fast = 0.0
        if slow_period > 1:
            slow_window = df['close'].iloc[-(slow_period - 1):]
            sum_slow = slow_window.sum()
        else:
            sum_slow = 0.0
        denominator = slow_period - fast_period
        x = (fast_period * sum_slow - slow_period * sum_fast) / denominator
        return round(x, 4), ">" if direction == "long" else "<"

    # ------------------------ macd 信号 ------------------------
    elif signal_type == 'macd':
        fast_period = int(parts[1])
        slow_period = int(parts[2])
        signal_period = int(parts[3])
        if len(df) < 1:
            raise ValueError("数据不足，无足够历史数据计算 macd 信号")

        fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        prev_macd = macd_line.iloc[-1]
        prev_signal = signal_line.iloc[-1]

        # 前置条件：对于多头，要求上周期 MACD < signal；空头相反
        if direction == "long":
            if prev_macd >= prev_signal:
                return None, None
        else:
            if prev_macd <= prev_signal:
                return None, None

        # 根据 EMA 的递推公式模拟未来一周期
        alpha_fast = 2 / (fast_period + 1)
        alpha_slow = 2 / (slow_period + 1)
        A = alpha_fast - alpha_slow
        B = fast_ema.iloc[-1] * (1 - alpha_fast) - slow_ema.iloc[-1] * (1 - alpha_slow)
        if A == 0:
            raise ValueError("macd 参数错误，导致除零")
        x = (prev_signal - B) / A
        return round(x, 4), ">" if direction == "long" else "<"

    # ------------------------ rsi 信号 ------------------------
    elif signal_type == 'rsi':
        period = int(parts[1])
        overbought = int(parts[2])
        oversold = int(parts[3])
        if len(df) < period + 1:
            raise ValueError("数据不足，无足够历史数据计算 rsi 信号")

        # 利用最近 period+1 个数据计算上周期 RSI
        window_prev = df['close'].iloc[-(period + 1):].values
        diffs = np.diff(window_prev)
        gains = np.maximum(diffs, 0)
        losses = -np.minimum(diffs, 0)
        avg_gain = gains.mean()
        avg_loss = losses.mean() if losses.mean() != 0 else 1e-6
        prev_rsi = 100 - 100 / (1 + avg_gain / avg_loss)

        # 前置条件：多头要求上周期 RSI < oversold；空头要求上周期 RSI > overbought
        if direction == "long":
            if prev_rsi >= oversold:
                return None, None
            # 未来窗口：取最近 period 个收盘价，加上 candidate 作为未来周期数据
            hist = df['close'].iloc[-period:].values
            base = df['close'].iloc[-1]
            step = base * 0.001
            max_range = base * 0.1

            def f(x):
                window_new = np.append(hist, x)
                d = np.diff(window_new)
                gains_new = np.maximum(d, 0)
                losses_new = -np.minimum(d, 0)
                avg_gain_new = gains_new.mean()
                avg_loss_new = losses_new.mean() if losses_new.mean() != 0 else 1e-6
                rsi_new = 100 - 100 / (1 + avg_gain_new / avg_loss_new)
                return rsi_new - oversold

            candidate = iterative_search(f, base, step, max_range, tol=1e-2)
            if candidate is None:
                return None, None
            return round(candidate, 4), ">"
        else:
            if prev_rsi <= overbought:
                return None, None
            hist = df['close'].iloc[-period:].values
            base = df['close'].iloc[-1]
            step = base * 0.001
            max_range = base * 0.1

            def f(x):
                window_new = np.append(hist, x)
                d = np.diff(window_new)
                gains_new = np.maximum(d, 0)
                losses_new = -np.minimum(d, 0)
                avg_gain_new = gains_new.mean()
                avg_loss_new = losses_new.mean() if losses_new.mean() != 0 else 1e-6
                rsi_new = 100 - 100 / (1 + avg_gain_new / avg_loss_new)
                return rsi_new - overbought

            candidate = iterative_search(f, base, step, max_range, tol=1e-2)
            if candidate is None:
                return None, None
            return round(candidate, 4), "<"

    # ------------------------ cci 信号 ------------------------
    elif signal_type == 'cci':
        period = int(parts[1])
        constant = 0.015
        tp = (df['high'] + df['low'] + df['close']) / 3
        if len(tp) < period:
            raise ValueError("数据不足，无足够历史数据计算 cci 信号")
        hist_window = tp.iloc[-period:]
        pre_ma = hist_window.mean()
        pre_md = hist_window.apply(lambda x: abs(x - pre_ma)).mean()
        pre_cci = (tp.iloc[-1] - pre_ma) / (constant * pre_md) if pre_md != 0 else 0

        # 前置条件：多头要求上周期 CCI < -100；空头要求上周期 CCI > 100
        if direction == "long":
            if pre_cci >= -100:
                return None, None
            hist_tp = tp.iloc[-(period - 1):].values  # 最近 period-1 个历史典型价格
            base = df['close'].iloc[-1]  # 以最后收盘价作为基准
            step = base * 0.001
            max_range = base * 0.1

            def f(x):
                # 假设未来周期 high, low, close 均等于 x, 则典型价格即为 x
                window_new = np.append(hist_tp, x)
                new_ma = window_new.mean()
                new_md = np.mean(np.abs(window_new - new_ma))
                new_cci = (x - new_ma) / (constant * new_md) if new_md != 0 else 0
                # 希望 new_cci 刚好达到 -100
                return new_cci + 100

            candidate = iterative_search(f, base, step, max_range, tol=1e-2)
            if candidate is None:
                return None, None
            return round(candidate, 4), ">"
        else:
            if pre_cci <= 100:
                return None, None
            hist_tp = tp.iloc[-(period - 1):].values
            base = df['close'].iloc[-1]
            step = base * 0.001
            max_range = base * 0.1

            def f(x):
                window_new = np.append(hist_tp, x)
                new_ma = window_new.mean()
                new_md = np.mean(np.abs(window_new - new_ma))
                new_cci = (x - new_ma) / (constant * new_md) if new_md != 0 else 0
                return new_cci - 100

            candidate = iterative_search(f, base, step, max_range, tol=1e-2)
            if candidate is None:
                return None, None
            return round(candidate, 4), "<"

    # ------------------------ atr 信号 ------------------------
    elif signal_type == 'atr':
        period = int(parts[1])
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        atr_ma = atr.rolling(period).mean()
        if direction == "long":
            if atr.iloc[-2] >= atr_ma.iloc[-2]:
                return None, None
        else:
            if atr.iloc[-2] <= atr_ma.iloc[-2]:
                return None, None
        target_price = round(df['close'].iloc[-1], 4)
        return target_price, ">" if direction == "long" else "<"

    else:
        raise ValueError(f"未知信号类型: {signal_type}")


def compute_signal(df, col_name):
    """
    根据历史行情数据(df)和指定信号名称(col_name)，生成交易信号和对应目标价格。

    说明：
      - 信号的目标价格不再使用 clip() 调整，
        而是在判断目标价格是否落在当前bar的 low 和 high 区间内，
        若目标价格超出区间，则认为信号无效（不产生信号）。
      - 当前支持的信号类型包括：
          - abs: 绝对百分比突破信号
              示例："abs_20_2_long" (20周期内最低价向上2%多头突破)
          - relate: 相对区间百分比突破信号
              示例："relate_20_50_short" (20周期区间顶部向下50%空头突破)
          - donchian: 唐奇安通道突破信号（实时价格触发优化）
              示例："donchian_20_long" (20周期最高价向上突破多头信号)
          - boll: 布林带信号
              示例："boll_20_2_long" 或 "boll_20_2_short"
          - macross: MACROSS 信号 (双均线交叉信号)
              示例："macross_10_20_long"
          - rsi: RSI 超买超卖反转信号
              示例："rsi_14_70_30_long"
          - macd: MACD交叉信号
              示例："macd_12_26_9_long"
          - cci: 商品通道指数超买超卖反转信号
              示例："cci_20_short"
              （若传入参数不足，则采用默认常数0.015）
          - atr: ATR波动率突破信号
              示例："atr_14_long"

    参数:
      df: pandas.DataFrame，必须包含以下列：
          "close": 收盘价
          "high": 最高价
          "low": 最低价
      col_name: 信号名称，格式如 "signalType_param1_param2_..._direction"

    返回:
      tuple:
        - signal_series: pandas.Series(bool)，当满足信号条件时为 True，否则为 False。
        - trade_price_series: pandas.Series(float)，信号触发时建议的目标交易价格；
          若目标价格超出当前bar的 low 和 high，则不产生信号。
    """

    parts = col_name.split('_')
    signal_type = parts[0]
    direction = parts[-1]

    if signal_type == 'abs':
        period = int(parts[1])
        abs_value = float(parts[2]) / 100
        if direction == "long":
            min_low_series = df['low'].shift(1).rolling(period).min()
            target_price = (min_low_series * (1 + abs_value)).round(4)
            signal_series = df['high'] > target_price
        else:
            max_high_series = df['high'].shift(1).rolling(period).max()
            target_price = (max_high_series * (1 - abs_value)).round(4)
            signal_series = df['low'] < target_price

        # 检查目标价格是否落在当前bar的low与high之间
        valid_price = (target_price >= df['low']) & (target_price <= df['high'])
        signal_series = signal_series & valid_price
        trade_price_series = target_price  # 直接使用计算的目标价格

        # 可选调试记录
        df['target_price'] = target_price
        df['signal_series'] = signal_series
        df['trade_price_series'] = trade_price_series
        return signal_series, trade_price_series

    elif signal_type == 'relate':
        period = int(parts[1])
        percent = float(parts[2]) / 100
        min_low_series = df['low'].shift(1).rolling(period).min()
        max_high_series = df['high'].shift(1).rolling(period).max()
        if direction == "long":
            target_price = (min_low_series + percent * (max_high_series - min_low_series)).round(4)
            signal_series = df['high'] > target_price
        else:
            target_price = (max_high_series - percent * (max_high_series - min_low_series)).round(4)
            signal_series = df['low'] < target_price

        valid_price = (target_price >= df['low']) & (target_price <= df['high'])
        signal_series = signal_series & valid_price
        trade_price_series = target_price
        return signal_series, trade_price_series

    elif signal_type == 'donchian':
        period = int(parts[1])
        if direction == "long":
            highest_high = df['high'].shift(1).rolling(period).max()
            signal_series = df['high'] > highest_high
            target_price = highest_high
        else:
            lowest_low = df['low'].shift(1).rolling(period).min()
            signal_series = df['low'] < lowest_low
            target_price = lowest_low

        valid_price = (target_price >= df['low']) & (target_price <= df['high'])
        signal_series = signal_series & valid_price
        trade_price_series = target_price.round(4)
        return signal_series, trade_price_series

    elif signal_type == 'boll':
        period = int(parts[1])
        std_multiplier = float(parts[2])
        ma = df['close'].rolling(window=period, min_periods=period).mean()
        std_dev = df['close'].rolling(window=period, min_periods=period).std()
        upper_band = (ma + std_multiplier * std_dev).round(4)
        lower_band = (ma - std_multiplier * std_dev).round(4)
        if direction == "long":
            signal_series = (df['close'].shift(1) < lower_band.shift(1)) & (df['close'] >= lower_band)
        else:  # short
            signal_series = (df['close'].shift(1) > upper_band.shift(1)) & (df['close'] <= upper_band)
        # 此处直接返回收盘价作为交易价格
        return signal_series, df["close"]

    elif signal_type == 'macross':
        fast_period = int(parts[1])
        slow_period = int(parts[2])
        fast_ma = df["close"].rolling(window=fast_period, min_periods=fast_period).mean().round(4)
        slow_ma = df["close"].rolling(window=slow_period, min_periods=slow_period).mean().round(4)
        if direction == "long":
            signal_series = (fast_ma.shift(1) < slow_ma.shift(1)) & (fast_ma >= slow_ma)
        else:
            signal_series = (fast_ma.shift(1) > slow_ma.shift(1)) & (fast_ma <= slow_ma)
        trade_price = df["close"]
        return signal_series, trade_price

    elif signal_type == 'rsi':
        period = int(parts[1])
        overbought = int(parts[2])
        oversold = int(parts[3])
        delta = df['close'].diff(1).astype(np.float32)
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        # 防止除0错误
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        if direction == "long":
            signal_series = (rsi.shift(1) < oversold) & (rsi >= oversold)
        else:
            signal_series = (rsi.shift(1) > overbought) & (rsi <= overbought)
        return signal_series, df['close']

    elif signal_type == 'macd':
        fast_period, slow_period, signal_period = map(int, parts[1:4])
        fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        if direction == "long":
            signal_series = (macd_line.shift(1) < signal_line.shift(1)) & (macd_line >= signal_line)
        else:
            signal_series = (macd_line.shift(1) > signal_line.shift(1)) & (macd_line <= signal_line)
        return signal_series, df["close"]

    elif signal_type == 'cci':
        period = int(parts[1])
        # 若参数不足，采用默认常数0.015
        if len(parts) == 3:
            constant = 0.015
        else:
            constant = float(parts[2]) / 100
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma = tp.rolling(period).mean()
        md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        cci = (tp - ma) / (constant * md)
        if direction == "long":
            signal_series = (cci.shift(1) < -100) & (cci >= -100)
        else:
            signal_series = (cci.shift(1) > 100) & (cci <= 100)
        return signal_series, df['close']

    elif signal_type == 'atr':
        period = int(parts[1])
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        atr_ma = atr.rolling(period).mean()
        if direction == "long":
            signal_series = (atr.shift(1) < atr_ma.shift(1)) & (atr >= atr_ma)
        else:
            signal_series = (atr.shift(1) > atr_ma.shift(1)) & (atr <= atr_ma)
        return signal_series, df['close']

    else:
        raise ValueError(f"未知信号类型: {signal_type}")


def calculate_max_sequence(kai_data_df):
    series = kai_data_df['true_profit'].to_numpy()
    min_sum, cur_sum = 0, 0
    start_idx, min_start, min_end = None, None, None
    trade_count, max_trade_count = 0, 0

    for i, profit in enumerate(series):
        if cur_sum == 0:
            start_idx = i
            trade_count = 0

        cur_sum += profit
        trade_count += 1

        if cur_sum < min_sum:
            min_sum = cur_sum
            min_start, min_end = start_idx, i
            max_trade_count = trade_count

        if cur_sum > 0:
            cur_sum = 0
            trade_count = 0

    return min_sum, min_start, min_end, max_trade_count


def calculate_max_profit(kai_data_df):
    series = kai_data_df['true_profit'].to_numpy()
    max_sum, cur_sum = 0, 0
    start_idx, max_start, max_end = None, None, None
    trade_count, max_trade_count = 0, 0

    for i, profit in enumerate(series):
        if cur_sum == 0:
            start_idx = i
            trade_count = 0

        cur_sum += profit
        trade_count += 1

        if cur_sum > max_sum:
            max_sum = cur_sum
            max_start, max_end = start_idx, i
            max_trade_count = trade_count

        if cur_sum < 0:
            cur_sum = 0
            trade_count = 0

    return max_sum, max_start, max_end, max_trade_count


def get_detail_backtest_result(df, kai_column, pin_column, signal_cache, is_filter=True):
    """
    根据传入的信号列（开仓信号 kai_column 与平仓信号 pin_column），在原始 df 上动态生成信号，
    进行回测计算，并返回对应的明细 DataFrame 和统计信息（statistic_dict）。

    使用 signal_cache 缓存同一信号的计算结果，避免重复计算。
    """
    kai_side = 'long' if 'long' in kai_column else 'short'

    # 计算或从缓存获取 kai_column 的信号与价格序列
    if kai_column in signal_cache:
        kai_signal, kai_price_series = signal_cache[kai_column]
    else:
        kai_signal, kai_price_series = compute_signal(df, kai_column)
        signal_cache[kai_column] = (kai_signal, kai_price_series)

    # 计算或从缓存获取 pin_column 的信号与价格序列
    if pin_column in signal_cache:
        pin_signal, pin_price_series = signal_cache[pin_column]
    else:
        pin_signal, pin_price_series = compute_signal(df, pin_column)
        signal_cache[pin_column] = (pin_signal, pin_price_series)

    # 根据信号挑选出开仓（或者平仓）时的行
    kai_data_df = df[kai_signal].copy()
    pin_data_df = df[pin_signal].copy()

    # 添加价格列（开仓价格和平仓价格）
    kai_data_df['kai_price'] = kai_price_series[kai_signal].values
    pin_data_df = pin_data_df.copy()
    pin_data_df['pin_price'] = pin_price_series[pin_signal].values

    # 通过 searchsorted 匹配平仓信号（注意 df 的索引默认保持原 CSV 行号）
    kai_data_df['pin_index'] = pin_data_df.index.searchsorted(kai_data_df.index, side='right')
    valid_mask = kai_data_df['pin_index'] < len(pin_data_df)
    kai_data_df = kai_data_df[valid_mask]
    kai_data_df['kai_side'] = kai_side

    matched_pin = pin_data_df.iloc[kai_data_df['pin_index'].values]
    kai_data_df['pin_price'] = matched_pin['pin_price'].values
    kai_data_df['pin_time'] = matched_pin['timestamp'].values
    kai_data_df['hold_time'] = matched_pin.index.values - kai_data_df.index.values

    # 计算交易收益率（profit）以及扣除成本后的收益（true_profit）
    if kai_side == 'long':
        kai_data_df['profit'] = ((kai_data_df['pin_price'] - kai_data_df['kai_price']) /
                                 kai_data_df['kai_price'] * 100).round(4)
    else:
        kai_data_df['profit'] = ((kai_data_df['kai_price'] - kai_data_df['pin_price']) /
                                 kai_data_df['pin_price'] * 100).round(4)
    kai_data_df['true_profit'] = kai_data_df['profit'] - 0.07

    # 如果is_filter为True，则相同pin_time的交易只保留最早的一笔
    if is_filter:
        kai_data_df = kai_data_df.sort_values('timestamp').drop_duplicates('pin_time', keep='first')

    # 获取kai_data_df['true_profit']的最大值和最小值
    max_single_profit = kai_data_df['true_profit'].max()
    min_single_profit = kai_data_df['true_profit'].min()

    # 计算最大连续亏损
    max_loss, max_loss_start_idx, max_loss_end_idx, loss_trade_count = calculate_max_sequence(kai_data_df)
    max_loss_start_time = (kai_data_df.loc[max_loss_start_idx]['timestamp']
                           if max_loss_start_idx is not None else None)
    max_loss_end_time = (kai_data_df.loc[max_loss_end_idx]['timestamp']
                         if max_loss_end_idx is not None else None)
    max_loss_hold_time = (max_loss_end_idx - max_loss_start_idx
                          if max_loss_start_idx is not None and max_loss_end_idx is not None else None)

    # 计算最大连续盈利
    max_profit, max_profit_start_idx, max_profit_end_idx, profit_trade_count = calculate_max_profit(kai_data_df)
    max_profit_start_time = (kai_data_df.loc[max_profit_start_idx]['timestamp']
                             if max_profit_start_idx is not None else None)
    max_profit_end_time = (kai_data_df.loc[max_profit_end_idx]['timestamp']
                           if max_profit_end_idx is not None else None)
    max_profit_hold_time = (max_profit_end_idx - max_profit_start_idx
                            if max_profit_start_idx is not None and max_profit_end_idx is not None else None)

    # # 平仓时间出现次数统计
    # pin_time_counts = kai_data_df['pin_time'].value_counts()
    # count_list = pin_time_counts.head(10).values.tolist()
    # top_10_pin_time_str = ','.join([str(x) for x in count_list])
    # max_pin_count = pin_time_counts.max() if not pin_time_counts.empty else 0
    # avg_pin_time_counts = pin_time_counts.mean() if not pin_time_counts.empty else 0

    # 分别筛选出true_profit大于0和小于0的数据
    profit_df = kai_data_df[kai_data_df['true_profit'] > 0]
    loss_df = kai_data_df[kai_data_df['true_profit'] < 0]

    loss_rate = loss_df.shape[0] / kai_data_df.shape[0] if kai_data_df.shape[0] > 0 else 0

    loss_time = loss_df['hold_time'].sum() if not loss_df.empty else 0
    profit_time = profit_df['hold_time'].sum() if not profit_df.empty else 0

    loss_time_rate = loss_time / (loss_time + profit_time) if (loss_time + profit_time) > 0 else 0

    # 生成统计数据字典 statistic_dict
    statistic_dict = {
        'kai_side': kai_side,
        'kai_column': kai_column,
        'pin_column': pin_column,
        'total_count': df.shape[0],
        'kai_count': kai_data_df.shape[0],
        'trade_rate': round(kai_data_df.shape[0] / df.shape[0], 4) if df.shape[0] > 0 else 0,
        'hold_time_mean': kai_data_df['hold_time'].mean() if not kai_data_df.empty else 0,
        'loss_rate': loss_rate,
        'loss_time_rate': loss_time_rate,
        'profit_rate': kai_data_df['profit'].sum(),
        'max_profit': max_single_profit,
        'min_profit': min_single_profit,
        'cost_rate': kai_data_df.shape[0] * 0.07,
        'net_profit_rate': kai_data_df['profit'].sum() - kai_data_df.shape[0] * 0.07,
        'avg_profit_rate': (round((kai_data_df['profit'].sum() - kai_data_df.shape[0] * 0.07)
                                  / kai_data_df.shape[0] * 100, 4)
                            if kai_data_df.shape[0] > 0 else 0),
        'max_consecutive_loss': round(max_loss, 4),
        'max_loss_trade_count': loss_trade_count,
        'max_loss_hold_time': max_loss_hold_time,
        'max_loss_start_time': max_loss_start_time,
        'max_loss_end_time': max_loss_end_time,
        'max_consecutive_profit': round(max_profit, 4),
        'max_profit_trade_count': profit_trade_count,
        'max_profit_hold_time': max_profit_hold_time,
        'max_profit_start_time': max_profit_start_time,
        'max_profit_end_time': max_profit_end_time,
        # 'max_pin_count': max_pin_count,
        # 'top_10_pin_time_count': top_10_pin_time_str,
        # 'avg_pin_time_counts': avg_pin_time_counts,
    }

    return kai_data_df, statistic_dict


def calculate_failure_rates(df: pd.DataFrame, period_list: list) -> dict:
    """
    计算不同周期的失败率（收益和小于0的比例）。

    参数：
    df : pd.DataFrame
        包含 'true_profit' 列的数据框。
    period_list : list
        需要计算的周期列表，例如 [1, 2]。

    返回：
    dict
        以周期为键，失败率为值的字典。
    """
    failure_rates = {}
    true_profit = df['true_profit'].values  # 转换为 NumPy 数组，加速计算
    total_periods = len(true_profit)

    for period in period_list:
        if period > total_periods:
            # failure_rates[period] = None  # 如果 period 超过数据长度，返回 None
            break

        # 计算滑动窗口和
        rolling_sums = [sum(true_profit[i:i + period]) for i in range(total_periods - period + 1)]

        # 计算失败次数（即滑动窗口和小于 0 的情况）
        failure_count = sum(1 for x in rolling_sums if x < 0)

        # 计算失败率
        failure_rates[period] = failure_count / len(rolling_sums)

    return failure_rates


@njit
def compute_low_min_range(low_array, start_pos, end_pos):
    n = start_pos.shape[0]
    out = np.empty(n, dtype=low_array.dtype)
    for i in range(n):
        s = start_pos[i]
        e = end_pos[i] + 1  # 因为切片包含终点
        m = low_array[s]
        for j in range(s + 1, e):
            if low_array[j] < m:
                m = low_array[j]
        out[i] = m
    return out


@njit
def compute_high_max_range(high_array, start_pos, end_pos):
    n = start_pos.shape[0]
    out = np.empty(n, dtype=high_array.dtype)
    for i in range(n):
        s = start_pos[i]
        e = end_pos[i] + 1  # 因为切片包含终点
        m = high_array[s]
        for j in range(s + 1, e):
            if high_array[j] > m:
                m = high_array[j]
        out[i] = m
    return out


def optimize_parameters(df, tp_range=None, sl_range=None):
    """
    优化止盈和止损参数（向量化实现）。

    输入:
        df: DataFrame，必须包含三列：'true_profit', 'max_true_profit', 'min_true_profit'
        tp_range: 用于搜索止盈参数的候选值数组。如果未提供，则从 df['max_true_profit'] 提取所有大于 0 的值，
                  保留两位小数并去重。
        sl_range: 用于搜索止损参数的候选值数组。如果未提供，则从 df['min_true_profit'] 提取所有小于 0 的值，
                  保留两位小数并去重。

    输出:
        返回一个字典，包含下列字段：
            max_optimal_value, max_optimal_profit, max_optimal_loss_rate,
            min_optimal_value, min_optimal_profit, min_optimal_loss_rate
    """
    # 构造候选参数
    if tp_range is None:
        tp_range = df['max_true_profit'].values
        tp_range = tp_range[tp_range > 0]  # 只保留正值
        tp_range = np.round(tp_range, 2)
        tp_range = np.unique(tp_range)
    if sl_range is None:
        sl_range = df['min_true_profit'].values
        sl_range = sl_range[sl_range < 0]  # 只保留负值
        sl_range = np.round(sl_range, 2)
        sl_range = np.unique(sl_range)

    # 提前将 DataFrame 的列转换为 NumPy 数组（加速计算）
    true_profit = df['true_profit'].values  # 实际利润
    max_true_profit = df['max_true_profit'].values  # 持有期内最大利润
    min_true_profit = df['min_true_profit'].values  # 持有期内最小利润
    n_trades = true_profit.shape[0]

    # ---------------------------
    # 只设置止盈时的模拟
    # 如果持有期内最大利润 >= tp，则取tp；否则取实际的true_profit
    # 利用广播：tp_range.shape=(n_tp,), true_profit.shape=(n_trades,)
    simulated_tp = np.where(
        max_true_profit[np.newaxis, :] >= tp_range[:, np.newaxis],
        tp_range[:, np.newaxis],
        true_profit[np.newaxis, :]
    )
    # 对每个候选参数计算累计利润
    total_profits_tp = simulated_tp.sum(axis=1)
    # 计算每个候选参数下最终利润为负的比例（失败率）
    loss_rates_tp = (simulated_tp < 0).sum(axis=1) / n_trades

    best_tp_index = np.argmax(total_profits_tp)
    best_tp = tp_range[best_tp_index]
    best_tp_profit = total_profits_tp[best_tp_index]
    best_tp_loss_rate = loss_rates_tp[best_tp_index]

    # ---------------------------
    # 只设置止损时的模拟
    # 如果持有期内最小利润 <= sl，则取 sl；否则取实际的 true_profit
    simulated_sl = np.where(
        min_true_profit[np.newaxis, :] <= sl_range[:, np.newaxis],
        sl_range[:, np.newaxis],
        true_profit[np.newaxis, :]
    )
    total_profits_sl = simulated_sl.sum(axis=1)
    loss_rates_sl = (simulated_sl < 0).sum(axis=1) / n_trades

    best_sl_index = np.argmax(total_profits_sl)
    best_sl = sl_range[best_sl_index]
    best_sl_profit = total_profits_sl[best_sl_index]
    best_sl_loss_rate = loss_rates_sl[best_sl_index]

    # 返回最终结果
    return {
        'max_optimal_value': best_tp,
        'max_optimal_profit': best_tp_profit,
        'max_optimal_loss_rate': best_tp_loss_rate,
        'min_optimal_value': best_sl,
        'min_optimal_profit': best_sl_profit,
        'min_optimal_loss_rate': best_sl_loss_rate
    }


@njit
def calculate_max_sequence_numba(series):
    n = series.shape[0]
    min_sum = 0.0
    cur_sum = 0.0
    start_idx = 0
    min_start = 0
    min_end = 0
    trade_count = 0
    max_trade_count = 0

    for i in range(n):
        if cur_sum == 0:
            start_idx = i
            trade_count = 0
        cur_sum += series[i]
        trade_count += 1
        if cur_sum < min_sum:
            min_sum = cur_sum
            min_start = start_idx
            min_end = i
            max_trade_count = trade_count
        if cur_sum > 0:
            cur_sum = 0
            trade_count = 0
    return min_sum, min_start, min_end, max_trade_count


@njit
def calculate_max_profit_numba(series):
    n = series.shape[0]
    max_sum = 0.0
    cur_sum = 0.0
    start_idx = 0
    max_start = 0
    max_end = 0
    trade_count = 0
    max_trade_count = 0

    for i in range(n):
        if cur_sum == 0:
            start_idx = i
            trade_count = 0
        cur_sum += series[i]
        trade_count += 1
        if cur_sum > max_sum:
            max_sum = cur_sum
            max_start = start_idx
            max_end = i
            max_trade_count = trade_count
        if cur_sum < 0:
            cur_sum = 0
            trade_count = 0
    return max_sum, max_start, max_end, max_trade_count

def get_detail_backtest_result_op(signal_cache, df, kai_column, pin_column, is_filter=True, is_detail=False, is_reverse=False):
    """
    优化后的 get_detail_backtest_result_op 函数：
      1. 从全局预计算的稀疏信号数据中提取非零索引及对应价格；
      2. 利用向量化操作（缓存中间变量、减少重复转换）获取回测数据并计算各类指标；

    参数:
      df          : 原始行情 DataFrame，要求包含 timestamp, open, high, low, close 等列；
      kai_column  : 主信号列名，对应 GLOBAL_SIGNALS 中的 key；
      pin_column  : 对应匹配信号列名；
      is_filter   : 是否对结果过滤（按照 timestamp 排序并去重）；
      is_detail   : 是否执行详细计算（如 low_min、high_max 以及区间收益率计算等）。

    返回:
      kai_data_df : 最终包含匹配信号数据及计算结果的 DataFrame；
      statistic_dict : 包含回测统计指标的字典；
    """
    def get_signal_and_price(column):
        if column in signal_cache:
            return signal_cache[column]
        signal_data = compute_signal(df, column)
        signal_cache[column] = signal_data
        return signal_data

    # 取出信号和对应价格序列
    kai_signal, kai_price_series = get_signal_and_price(kai_column)
    pin_signal, pin_price_series = get_signal_and_price(pin_column)

    if kai_signal.sum() < 0 or pin_signal.sum() < 0:
        return None, None

    # 从 df 中取出符合条件的数据，并预先拷贝数据
    kai_data_df = df.loc[kai_signal].copy()
    pin_data_df = df.loc[pin_signal].copy()

    # 缓存价格数据，避免重复转换
    kai_prices = kai_price_series[kai_signal].to_numpy()
    pin_prices = pin_price_series[pin_signal].to_numpy()

    kai_data_df['kai_price'] = kai_prices
    pin_data_df['pin_price'] = pin_prices

    # 判断两个信号的公共索引数（用来过滤不匹配的组合）
    common_index = kai_data_df.index.intersection(pin_data_df.index)
    same_count = len(common_index)
    kai_count = len(kai_data_df)
    pin_count = len(pin_data_df)
    same_count_rate = (100 * same_count / min(kai_count, pin_count)) if min(kai_count, pin_count) > 0 else 0
    # if same_count_rate > 80:
    #     return None, None

    # 使用 numpy 向量化查找匹配：对 kai_data_df 中的时间索引在 pin_data_df 中进行右侧查找
    kai_idx_arr = np.asarray(kai_data_df.index)
    pin_idx_arr = np.asarray(pin_data_df.index)
    pin_match_indices = np.searchsorted(pin_idx_arr, kai_idx_arr, side="right")
    valid_mask = pin_match_indices < len(pin_idx_arr)
    if valid_mask.sum() == 0:
        return None, None

    # 筛选有效数据，及对应的匹配结果
    kai_data_df = kai_data_df.iloc[valid_mask].copy()
    kai_idx_valid = kai_idx_arr[valid_mask]
    pin_match_indices_valid = pin_match_indices[valid_mask]
    matched_pin = pin_data_df.iloc[pin_match_indices_valid].copy()

    # 更新匹配数据：引入匹配的 pin_price 与 timestamp，并计算持仓时长（假设 index 为可直接相减的数值）
    kai_data_df["pin_price"] = matched_pin["pin_price"].values
    kai_data_df["pin_time"] = matched_pin["timestamp"].values
    # 利用匹配后 DataFrame 的索引值进行差值计算（采用 .values 避免重复转换）
    kai_data_df["hold_time"] = matched_pin.index.values - kai_idx_valid

    # 判断方向，仅判断一次，避免多处调用字符串查找
    if is_reverse:
        is_long = "short" in kai_column.lower()
    else:
        is_long = "long" in kai_column.lower()

    # 若要求详细计算，用已缓存的 NumPy 数组及向量化操作计算区间最低和最高价格，进而计算收益率区间
    if is_detail:
        df_index_arr = np.asarray(df.index)
        low_array = df["low"].values
        high_array = df["high"].values

        start_times = np.asarray(kai_data_df.index)
        end_times = np.asarray(matched_pin.index)
        start_pos = np.searchsorted(df_index_arr, start_times, side="left")
        end_pos = np.searchsorted(df_index_arr, end_times, side="right") - 1

        low_min_arr = compute_low_min_range(low_array, start_pos, end_pos)
        high_max_arr = compute_high_max_range(high_array, start_pos, end_pos)
        kai_data_df["low_min"] = low_min_arr
        kai_data_df["high_max"] = high_max_arr

        if is_long:
            kai_data_df["max_true_profit"] = (
                    ((kai_data_df["high_max"] - kai_data_df["kai_price"]) / kai_data_df["kai_price"] * 100) - 0.07
            ).round(4)
            kai_data_df["min_true_profit"] = (
                    ((kai_data_df["low_min"] - kai_data_df["kai_price"]) / kai_data_df["kai_price"] * 100) - 0.07
            ).round(4)
        else:
            kai_data_df["max_true_profit"] = (
                    ((kai_data_df["kai_price"] - kai_data_df["low_min"]) / kai_data_df["kai_price"] * 100) - 0.07
            ).round(4)
            kai_data_df["min_true_profit"] = (
                    ((kai_data_df["kai_price"] - kai_data_df["high_max"]) / kai_data_df["kai_price"] * 100) - 0.07
            ).round(4)

    # 若需要过滤，则对结果按 timestamp 排序，并根据 pin_time 去重
    if is_filter:
        kai_data_df = kai_data_df.sort_values("timestamp").drop_duplicates("pin_time", keep="first")
    # 基本统计指标
    trade_count = len(kai_data_df)
    total_count = len(df)

    # 根据 pin_time 建立映射，更新 kai_price，使得价格更准确
    pin_price_map = kai_data_df.set_index("pin_time")["pin_price"]
    mapped_prices = kai_data_df["timestamp"].map(pin_price_map)
    if same_count > 0 and not mapped_prices.isna().all():
        kai_data_df["kai_price"] = mapped_prices.combine_first(kai_data_df["kai_price"])
    modification_rate = round(100 * mapped_prices.notna().sum() / trade_count, 4) if trade_count else 0

    # 利用向量化方式计算收益率
    if is_long:
        profit_series = ((kai_data_df["pin_price"] - kai_data_df["kai_price"]) / kai_data_df["kai_price"] * 100).round(
            4)
    else:
        profit_series = ((kai_data_df["kai_price"] - kai_data_df["pin_price"]) / kai_data_df["kai_price"] * 100).round(
            4)
    kai_data_df["profit"] = profit_series
    kai_data_df["true_profit"] = profit_series - 0.07
    profit_sum = profit_series.sum()

    fix_profit = round(kai_data_df[mapped_prices.notna()]["true_profit"].sum(),
                       4)  # 收到影响的交易的收益，实盘交易时可以设置不得连续开平来避免。也就是将fix_profit减去就是正常的利润

    max_single_profit = kai_data_df["true_profit"].max()
    min_single_profit = kai_data_df["true_profit"].min()
    temp_dict = {}

    true_profit_std = kai_data_df["true_profit"].std()
    true_profit_mean = kai_data_df["true_profit"].mean() * 100 if trade_count > 0 else 0

    profits_arr = kai_data_df["true_profit"].values
    max_loss, max_loss_start_idx, max_loss_end_idx, loss_trade_count = calculate_max_sequence_numba(profits_arr)
    max_profit, max_profit_start_idx, max_profit_end_idx, profit_trade_count = calculate_max_profit_numba(profits_arr)

    # 根据索引获取最大连续亏损的起止时间和持仓时长
    if trade_count > 0 and max_loss_start_idx < len(kai_data_df) and max_loss_end_idx < len(kai_data_df):
        max_loss_start_time = kai_data_df.iloc[max_loss_start_idx]["timestamp"]
        max_loss_end_time = kai_data_df.iloc[max_loss_end_idx]["timestamp"]
        max_loss_hold_time = kai_data_df.index[max_loss_end_idx] - kai_data_df.index[max_loss_start_idx]
    else:
        max_loss_start_time = max_loss_end_time = max_loss_hold_time = None

    # 同理，计算最大连续盈利的起止时间和持仓时长
    if trade_count > 0 and max_profit_start_idx < len(kai_data_df) and max_profit_end_idx < len(kai_data_df):
        max_profit_start_time = kai_data_df.iloc[max_profit_start_idx]["timestamp"]
        max_profit_end_time = kai_data_df.iloc[max_profit_end_idx]["timestamp"]
        max_profit_hold_time = kai_data_df.index[max_profit_end_idx] - kai_data_df.index[max_profit_start_idx]
    else:
        max_profit_start_time = max_profit_end_time = max_profit_hold_time = None

    # 计算盈利与亏损相关指标
    profit_df = kai_data_df[kai_data_df["true_profit"] > 0]
    loss_df = kai_data_df[kai_data_df["true_profit"] < 0]
    fu_profit_sum = loss_df["true_profit"].sum()
    fu_profit_mean = round(loss_df["true_profit"].mean() if not loss_df.empty else 0, 4)
    zhen_profit_sum = profit_df["true_profit"].sum()
    zhen_profit_mean = round(profit_df["true_profit"].mean() if not profit_df.empty else 0, 4)
    loss_rate = loss_df.shape[0] / trade_count if trade_count else 0
    loss_time = loss_df["hold_time"].sum() if not loss_df.empty else 0
    profit_time = profit_df["hold_time"].sum() if not profit_df.empty else 0
    loss_time_rate = loss_time / (loss_time + profit_time) if (loss_time + profit_time) else 0

    trade_rate = round(100 * trade_count / total_count, 4) if total_count else 0
    hold_time_mean = kai_data_df["hold_time"].mean() if trade_count else 0

    monthly_groups = kai_data_df["timestamp"].dt.to_period("M")
    monthly_agg = kai_data_df.groupby(monthly_groups)["true_profit"].agg(["sum", "mean", "count"])
    monthly_trade_std = float(monthly_agg["count"].std())
    active_months = monthly_agg.shape[0]
    total_months = 22
    active_month_ratio = active_months / total_months if total_months else 0
    monthly_net_profit_std = float(monthly_agg["sum"].std())
    monthly_avg_profit_std = float(monthly_agg["mean"].std())
    monthly_net_profit_min = monthly_agg["sum"].min()
    monthly_net_profit_max = monthly_agg["sum"].max()
    monthly_loss_rate = (monthly_agg["sum"] < 0).sum() / active_months if active_months else 0

    if is_detail:
        max_single_profit = kai_data_df["max_true_profit"].max()
        min_single_profit = kai_data_df["min_true_profit"].min()
        temp_dict = optimize_parameters(kai_data_df) if trade_count > 0 else {}

        # 新增指标：每个月净利润和交易个数
        monthly_net_profit_detail = {str(month): round(val, 4) for month, val in monthly_agg["sum"].to_dict().items()}
        monthly_trade_count_detail = {str(month): int(val) for month, val in monthly_agg["count"].to_dict().items()}
        # 获取kai_data_df每一行的index，hold_time，true_profit保存成为一个元组列表，相当于这个列表的元素是(index, hold_time, true_profit)
        kai_data_df_tuples = [(row.Index, row.hold_time, row.true_profit) for row in kai_data_df.itertuples()]

        temp_dict.update({
            "monthly_net_profit_detail": monthly_net_profit_detail,
            "monthly_trade_count_detail": monthly_trade_count_detail,
            "kai_data_df_tuples": kai_data_df_tuples
        })
    hold_time_std = kai_data_df["hold_time"].std()

    # 前10%盈利/亏损的比率计算
    if not profit_df.empty:
        top_profit_count = max(1, int(np.ceil(len(profit_df) * 0.1)))
        profit_sorted = profit_df.sort_values("true_profit", ascending=False)
        top_profit_sum = profit_sorted["true_profit"].iloc[:top_profit_count].sum()
        total_profit_sum = profit_df["true_profit"].sum()
        top_profit_ratio = top_profit_sum / total_profit_sum if total_profit_sum != 0 else 0
    else:
        top_profit_ratio = 0

    if not loss_df.empty:
        top_loss_count = max(1, int(np.ceil(len(loss_df) * 0.1)))
        loss_sorted = loss_df.sort_values("true_profit", ascending=True)
        top_loss_sum = loss_sorted["true_profit"].iloc[:top_loss_count].sum()
        total_loss_sum = loss_df["true_profit"].sum()
        top_loss_ratio = (abs(top_loss_sum) / abs(total_loss_sum)) if total_loss_sum != 0 else 0
    else:
        top_loss_ratio = 0

    statistic_dict = {
        "kai_side": "long" if is_long else "short",
        "kai_column": kai_column,
        "pin_column": pin_column,
        "kai_count": trade_count,
        "total_count": total_count,
        "trade_rate": trade_rate,
        "hold_time_mean": hold_time_mean,
        "hold_time_std": hold_time_std,
        "loss_rate": loss_rate,
        "loss_time_rate": loss_time_rate,
        'zhen_profit_sum': zhen_profit_sum,
        'zhen_profit_mean': zhen_profit_mean,
        'fu_profit_sum': fu_profit_sum,
        'fu_profit_mean': fu_profit_mean,
        "profit_rate": profit_sum,
        "max_profit": max_single_profit,
        "min_profit": min_single_profit,
        "cost_rate": trade_count * 0.07,
        "net_profit_rate": kai_data_df["true_profit"].sum(),
        "fix_profit": fix_profit,
        "avg_profit_rate": round(true_profit_mean, 4),
        "true_profit_std": true_profit_std,
        "max_consecutive_loss": round(max_loss, 4),
        "max_loss_trade_count": loss_trade_count,
        "max_loss_hold_time": max_loss_hold_time,
        "max_loss_start_time": max_loss_start_time,
        "max_loss_end_time": max_loss_end_time,
        "max_consecutive_profit": round(max_profit, 4),
        "max_profit_trade_count": profit_trade_count,
        "max_profit_hold_time": max_profit_hold_time,
        "max_profit_start_time": max_profit_start_time,
        "max_profit_end_time": max_profit_end_time,
        "same_count": same_count,
        "same_count_rate": round(same_count_rate, 4),
        "true_same_count_rate": modification_rate,
        "monthly_trade_std": round(monthly_trade_std, 4),
        "active_month_ratio": round(active_month_ratio, 4),
        "monthly_loss_rate": round(monthly_loss_rate, 4),
        "monthly_net_profit_min": round(monthly_net_profit_min, 4),
        "monthly_net_profit_max": round(monthly_net_profit_max, 4),
        "monthly_net_profit_std": round(monthly_net_profit_std, 4),
        "monthly_avg_profit_std": round(monthly_avg_profit_std, 4),
        "top_profit_ratio": round(top_profit_ratio, 4),
        "top_loss_ratio": round(top_loss_ratio, 4),
        'is_reverse': is_reverse
    }
    statistic_dict.update(temp_dict)
    kai_data_df = kai_data_df[['hold_time', 'true_profit']]
    return kai_data_df, statistic_dict


def generate_numbers(start, end, number, even=True):
    """
    生成start到end之间的number个数字。

    Args:
        start: 区间起始值 (包含).
        end: 区间结束值 (包含).
        number: 生成数字的个数.
        even: 是否均匀生成。True表示均匀生成，False表示非均匀（指数增长）生成。

    Returns:
        包含生成数字的列表，如果start > end或number <= 0，则返回空列表。
    """
    if start > end or number <= 0:
        return []
    if number == 1:
        return []

    result = []
    if even:
        if number > 1:
            step = (end - start) / (number - 1)
            for i in range(number):
                result.append(int(round(start + i * step)))
        else:
            result = [start]
    else:  # uneven, exponential-like
        power = 2  # 可以调整power值来控制指数增长的程度
        for i in range(number):
            normalized_index = i / (number - 1) if number > 1 else 0
            value = start + (end - start) * (normalized_index ** power)
            result.append(int(round(value)))

    # 确保生成的数字在[start, end]范围内，并去除重复值 (虽然按理说不会有重复，但以防万一)
    final_result = []
    last_val = None
    for val in result:
        if start <= val <= end and val != last_val:
            final_result.append(val)
            last_val = val
    return final_result[:number]



def gen_ma_signal_name(start_period, end_period, step):
    """
    生成 ma 信号的列名列表。
    :param start_period:
    :param end_period:
    :param step:
    :return:
    """
    period_list = generate_numbers(start_period, end_period, step, even=False)
    long_columns = [f"ma_{period}_high_long" for period in period_list]
    short_columns = [f"ma_{period}_low_short" for period in period_list]
    key_name = f'ma_{start_period}_{end_period}_{step}'
    print(f"ma一共生成 {len(long_columns)} 个信号列名。参数为：{start_period}, {end_period}, {step}")
    return long_columns, short_columns, key_name


def gen_rsi_signal_name(start_period, end_period, step):
    """
    生成 rsi 信号的列名列表。
    :param start_period:
    :param end_period:
    :param step:
    :return:
    """
    period_list = generate_numbers(start_period, end_period, step, even=False)
    temp_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    long_columns = [f"rsi_{period}_{overbought}_{100 - overbought}_high_long" for period in period_list for overbought
                    in temp_list]
    short_columns = [f"rsi_{period}_{overbought}_{100 - overbought}_low_short" for period in period_list for overbought
                     in temp_list]
    key_name = f'rsi_{start_period}_{end_period}_{step}'
    print(f"rsi一共生成 {len(long_columns)} 个信号列名。参数为：{start_period}, {end_period}, {step}")
    return long_columns, short_columns, key_name


def gen_peak_signal_name(start_period, end_period, step):
    """
    生成 peak 信号的列名列表。
    :param start_period:
    :param end_period:
    :param step:
    :return:
    """
    period_list = generate_numbers(start_period, end_period, step, even=False)
    long_columns = [f"peak_{period}_high_long" for period in period_list]
    short_columns = [f"peak_{period}_low_short" for period in period_list]
    key_name = f'peak_{start_period}_{end_period}_{step}'
    print(f"peak一共生成 {len(long_columns)} 个信号列名。参数为：{start_period}, {end_period}, {step}")
    return long_columns, short_columns, key_name


def gen_continue_signal_name(start_period, end_period, step):
    """"""
    period_list = range(start_period, end_period, step)
    long_columns = [f"continue_{period}_high_long" for period in period_list]
    short_columns = [f"continue_{period}_low_short" for period in period_list]
    key_name = f'continue_{start_period}_{end_period}_{step}'
    print(f"continue一共生成 {len(long_columns)} 个信号列名。参数为：{start_period}, {end_period}, {step}")
    return long_columns, short_columns, key_name


def gen_abs_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    """"""
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = range(start_period1, end_period1, step1)
    period_list1 = [x / 10 for x in period_list1]
    long_columns = [f"abs_{period}_{period1}_high_long" for period in period_list for period1 in period_list1 if
                    period >= period1]
    short_columns = [f"abs_{period}_{period1}_low_short" for period in period_list for period1 in period_list1 if
                     period >= period1]
    key_name = f'abs_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}'
    print(
        f"abs一共生成 {len(long_columns)} 个信号列名。参数为：{start_period}, {end_period}, {step}, {start_period1}, {end_period1}, {step1}")
    return long_columns, short_columns, key_name


def gen_relate_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    """"""
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = range(start_period1, end_period1, step1)
    long_columns = [f"relate_{period}_{period1}_high_long" for period in period_list for period1 in period_list1 if
                    period >= period1]
    short_columns = [f"relate_{period}_{period1}_low_short" for period in period_list for period1 in period_list1 if
                     period >= period1]
    key_name = f'relate_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}'
    print(
        f"relate一共生成 {len(long_columns)} 个信号列名。参数为：{start_period}, {end_period}, {step}, {start_period1}, {end_period1}, {step1}")
    return long_columns, short_columns, key_name


def gen_macross_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    """"""
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = generate_numbers(start_period1, end_period1, step1, even=False)
    long_columns = [f"macross_{period}_{period1}_high_long" for period in period_list for period1 in period_list1]
    short_columns = [f"macross_{period}_{period1}_low_short" for period in period_list for period1 in period_list1]
    key_name = f'macross_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}'
    print(
        f"macross一共生成 {len(long_columns)} 个信号列名。参数为：{start_period}, {end_period}, {step}, {start_period1}, {end_period1}, {step1}")
    return long_columns, short_columns, key_name


def optimal_leverage_opt(max_loss_rate, num_losses, max_profit_rate, num_profits,
                         max_single_loss, max_single_profit, other_rate, other_count,
                         L_min=1):
    """
    利用向量化计算不同杠杆下的最终收益，并返回使最终收益最大的杠杆值和对应收益。
    参数含义与原函数一致，不再赘述。
    """
    # 将百分比转换为小数
    max_loss_rate = max_loss_rate / 100.0
    max_profit_rate = max_profit_rate / 100.0
    max_single_loss = max_single_loss / 100.0

    # 计算每次交易亏损率
    r_loss = max_loss_rate / num_losses

    # 计算避免爆仓的最大杠杆
    L_max = abs(1 / max_single_loss)

    # 直接构造整数候选杠杆序列
    L_values = np.arange(L_min, int(L_max) + 1, dtype=float)

    # 向量化计算最终收益
    # 计算因亏损累计的收益（先计算亏损部分）
    after_loss = (1 + L_values * r_loss) ** num_losses
    after_loss *= (1 + L_values * max_single_loss)

    # 对于 after_loss<=0 的情况认为爆仓，收益记为 0
    valid = after_loss > 0
    final_balance = np.zeros_like(L_values)

    if np.any(valid):
        after_gain = after_loss[valid] * (1 + L_values[valid] * max_profit_rate)
        after_gain *= (1 + L_values[valid] * max_single_profit)
        final_balance[valid] = after_gain * (1 + L_values[valid] * other_rate)

    # 找到最佳杠杆对应的索引
    optimal_idx = np.argmax(final_balance)
    optimal_L = int(L_values[optimal_idx])
    max_balance = final_balance[optimal_idx]

    return optimal_L, max_balance


def count_L():
    file_list = os.listdir('temp')
    file_list = [file for file in file_list if
                 'True' in file and '1m' in file and '2000' in file and 'withL' not in file]
    for file in file_list:
        print(f'开始处理 {file}')
        out_file = file.replace('.csv', '_withL.csv')
        if os.path.exists(f'temp/{out_file}'):
            print(f'已存在 {out_file}')
            continue
        start_time = time.time()

        try:
            signal_data_df = pd.read_csv(f'temp/{file}')
            signal_data_df[['optimal_L', 'max_balance']] = signal_data_df.apply(
                lambda row: optimal_leverage_opt(
                    row['max_consecutive_loss'], row['max_loss_trade_count'], row['max_consecutive_profit'],
                    row['max_profit_trade_count'],
                    row['min_profit'], row['max_profit'], row['net_profit_rate'], row['kai_count']
                ), axis=1, result_type='expand'
            )

            signal_data_df['filename'] = file.split('_')[5]
            signal_data_df.to_csv(f'temp/{out_file}')
            print(f'{file} 耗时 {time.time() - start_time:.2f} 秒。 长度 {signal_data_df.shape[0]}')
        except Exception as e:
            pass


def choose_good_strategy():
    # df = pd.read_csv('temp/temp.csv')
    start_time = time.time()
    count_L()
    # 找到temp下面所有包含False的文件
    file_list = os.listdir('temp')
    file_list = [file for file in file_list if
                 'True' in file and 'ETH' in file and '0' in file and '1m' in file and 'with' in file]
    df_list = []
    df_map = {}
    for file in file_list:
        file_key = file.split('_')[4]
        df = pd.read_csv(f'temp/{file}')

        df['filename'] = file.split('_')[5]
        df = df[(df['avg_profit_rate'] > 0)]
        if file_key not in df_map:
            df_map[file_key] = []
        df['score'] = df['avg_profit_rate']
        df['score1'] = df['avg_profit_rate'] / (df['hold_time_mean'] + 20) * 1000
        df['score2'] = df['avg_profit_rate'] / (
                df['hold_time_mean'] + 20) * 1000 * (df['trade_rate'] + 0.001)
        df['score3'] = df['avg_profit_rate'] * (df['trade_rate'] + 0.0001)
        df_map[file_key].append(df)
    for key in df_map:
        df = pd.concat(df_map[key])
        df_list.append(df)
    print(f'耗时 {time.time() - start_time:.2f} 秒。')

    temp = pd.merge(df_list[0], df_list[1], on=['kai_side', 'kai_column', 'pin_column'], how='inner')
    # 需要计算的字段前缀
    fields = ['avg_profit_rate', 'net_profit_rate', 'max_balance']

    # 遍历字段前缀，统一计算
    for field in fields:
        x_col = f"{field}_x"
        y_col = f"{field}_y"

        temp[f"{field}_min"] = temp[[x_col, y_col]].min(axis=1)
        temp[f"{field}_mean"] = temp[[x_col, y_col]].mean(axis=1)
        temp[f"{field}_plus"] = temp[x_col] + temp[y_col]
        temp[f"{field}_mult"] = np.where(
            (temp[x_col] < 0) & (temp[y_col] < 0),
            0,  # 如果两个都小于 0，则赋值 0
            temp[x_col] * temp[y_col]  # 否则正常相乘
        )

    temp = temp[(temp['avg_profit_rate_min'] > 0)]
    return temp


def generate_reverse_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据回测统计数据中各字段的含义计算逆向版本，
    注意：所有逆向计算均基于原始 df 的数值，结果保留原字段名称。
    """
    # 复制一份用于生成逆向数据
    df_rev = df.copy()

    # ----- 收益相关指标 -----
    # profit_rate 直接取负
    if 'profit_rate' in df.columns:
        df_rev['profit_rate'] = -df['profit_rate']

    # cost_rate 保持不变
    if 'cost_rate' in df.columns:
        df_rev['cost_rate'] = df['cost_rate']

    # 实际净收益：原来的 net_profit_rate = profit_rate - cost_rate,
    # 逆向时，净收益为 -profit_rate - cost_rate
    if 'net_profit_rate' in df.columns and 'profit_rate' in df.columns and 'cost_rate' in df.columns:
        df_rev['net_profit_rate'] = -df['profit_rate'] - df['cost_rate']

    # 平均收益率，假设为 ( -profit_rate - cost_rate ) / kai_count
    if 'avg_profit_rate' in df.columns and 'profit_rate' in df.columns \
            and 'cost_rate' in df.columns and 'kai_count' in df.columns:
        df_rev['avg_profit_rate'] = (-df['profit_rate'] - df['cost_rate']) / df['kai_count']

    # true_profit_std 和其它标准差类指标不受方向影响，直接复制
    if 'true_profit_std' in df.columns:
        df_rev['true_profit_std'] = df['true_profit_std']

    # ----- 最大/最小盈利指标 -----
    # 逆向时，最大盈利 = -原最小盈利，最小盈利 = -原最大盈利
    if 'min_profit' in df.columns and 'max_profit' in df.columns:
        df_rev['max_profit'] = -df['min_profit']
        df_rev['min_profit'] = -df['max_profit']

    # ----- 连续性指标（例如连续盈亏）-----
    if 'max_consecutive_loss' in df.columns and 'max_consecutive_profit' in df.columns:
        df_rev['max_consecutive_loss'] = -df['max_consecutive_profit']
        df_rev['max_consecutive_profit'] = -df['max_consecutive_loss']

    if 'max_loss_trade_count' in df.columns and 'max_profit_trade_count' in df.columns:
        # 亏损交易数与盈利交易数交换
        df_rev['max_loss_trade_count'] = df['max_profit_trade_count']
        df_rev['max_profit_trade_count'] = df['max_loss_trade_count']

    if 'max_loss_hold_time' in df.columns and 'max_profit_hold_time' in df.columns:
        # 持仓时长交换
        df_rev['max_loss_hold_time'] = df['max_profit_hold_time']
        df_rev['max_profit_hold_time'] = df['max_loss_hold_time']

    if 'max_loss_start_time' in df.columns and 'max_profit_start_time' in df.columns:
        df_rev['max_loss_start_time'] = df['max_profit_start_time']
        df_rev['max_profit_start_time'] = df['max_loss_start_time']

    if 'max_loss_end_time' in df.columns and 'max_profit_end_time' in df.columns:
        df_rev['max_loss_end_time'] = df['max_profit_end_time']
        df_rev['max_profit_end_time'] = df['max_loss_end_time']

    # ----- 胜率相关指标 -----
    if 'loss_rate' in df.columns:
        df_rev['loss_rate'] = 1 - df['loss_rate']

    if 'loss_time_rate' in df.columns:
        df_rev['loss_time_rate'] = 1 - df['loss_time_rate']

    # ----- 其它不受方向影响或数值拷贝 -----
    if 'trade_rate' in df.columns:
        df_rev['trade_rate'] = df['trade_rate']
    if 'hold_time_mean' in df.columns:
        df_rev['hold_time_mean'] = df['hold_time_mean']
    if 'hold_time_std' in df.columns:
        df_rev['hold_time_std'] = df['hold_time_std']

    # ----- 月度统计指标 -----
    if 'monthly_trade_std' in df.columns:
        df_rev['monthly_trade_std'] = df['monthly_trade_std']
    if 'active_month_ratio' in df.columns:
        df_rev['active_month_ratio'] = df['active_month_ratio']
    if 'monthly_loss_rate' in df.columns:
        df_rev['monthly_loss_rate'] = 1 - df['monthly_loss_rate']

    if 'monthly_net_profit_min' in df.columns and 'monthly_net_profit_max' in df.columns:
        df_rev['monthly_net_profit_min'] = -df['monthly_net_profit_max']
        df_rev['monthly_net_profit_max'] = -df['monthly_net_profit_min']
    else:
        if 'monthly_net_profit_min' in df.columns:
            df_rev['monthly_net_profit_min'] = -df['monthly_net_profit_min']
        if 'monthly_net_profit_max' in df.columns:
            df_rev['monthly_net_profit_max'] = -df['monthly_net_profit_max']

    if 'monthly_net_profit_std' in df.columns:
        df_rev['monthly_net_profit_std'] = df['monthly_net_profit_std']
    if 'monthly_avg_profit_std' in df.columns:
        df_rev['monthly_avg_profit_std'] = df['monthly_avg_profit_std']

    # ----- 前10%盈利/亏损比率 -----
    if 'top_profit_ratio' in df.columns and 'top_loss_ratio' in df.columns:
        # 逆向时前10%盈利比率变为原前10%亏损比率，反之亦然
        df_rev['top_profit_ratio'] = df['top_loss_ratio']
        df_rev['top_loss_ratio'] = df['top_profit_ratio']

    # # ----- 信号方向与信号字段 -----
    # if 'kai_side' in df.columns:
    #     df_rev['kai_side'] = df['kai_side'].apply(
    #         lambda x: "short" if isinstance(x, str) and x.lower() == "long"
    #         else ("long" if isinstance(x, str) and x.lower() == "short" else x)
    #     )
    #
    # if 'kai_column' in df.columns and 'pin_column' in df.columns:
    #     df_rev['kai_column'] = df['pin_column']
    #     df_rev['pin_column'] = df['kai_column']

    # ----- 其它计数类字段，直接复制 -----
    for col in ['same_count', 'same_count_rate', 'kai_count', 'total_count']:
        if col in df.columns:
            df_rev[col] = df[col]

    # 对于未涉及具体逆向逻辑的其他字段，保持原值
    return df_rev


def add_reverse(df: pd.DataFrame) -> pd.DataFrame:
    """
    接受原始的回测统计数据 DataFrame，
    为每一行生成逆向数据（按字段计算规则），
    最后返回的 DataFrame 包含原始数据行和逆向数据行（顺序拼接），
    行数为原来的2倍，字段名称保持一致。
    """
    # 生成逆向数据 DataFrame（所有计算均基于原始 df 的数值）
    df_rev = generate_reverse_df(df)
    df_rev['is_reverse'] = True
    # 拼接原始数据和逆向数据（重置索引）
    df_result = pd.concat([df, df_rev], ignore_index=True)
    return df_result


def calculate_final_score(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    根据聚合后的 result_df 中各信号的统计指标，计算最终综合评分。

    核心指标：
      盈利指标：
        - net_profit_rate: 扣除交易成本后的累计收益率
        - avg_profit_rate: 平均每笔交易收益率
      风险/稳定性指标：
        - loss_rate: 亏损交易比例（越低越好）
        - monthly_loss_rate: 亏损月份比例（越低越好）
        - monthly_avg_profit_std: 月度收益标准差
        - monthly_net_profit_std: 月度净收益标准差

    分析思路：
      1. 对盈利指标使用 min-max 归一化，数字越大表示盈利能力越好；
      2. 对风险指标（loss_rate、monthly_loss_rate）归一化后取1-值，保证数值越大越稳定；
      3. 计算波动性：
           - risk_volatility = monthly_avg_profit_std / (abs(avg_profit_rate) + eps)
           - risk_volatility_net = monthly_net_profit_std / (abs(net_profit_rate) + eps)
         归一化后取 1 - normalized_value（值越大表示波动性较低，相对稳健)；
      4. 稳定性子评分取这四个风险因子的算数平均；
      5. 最终得分综合盈利能力和稳定性评分，举例盈利权重0.4，稳定性权重0.6。

    参数:
      result_df: 包含各信号统计指标的 DataFrame，
                 需要包含以下列（或部分列）：
                   - "net_profit_rate"
                   - "avg_profit_rate"
                   - "loss_rate"
                   - "monthly_loss_rate"
                   - "monthly_avg_profit_std"
                   - "monthly_net_profit_std"

    返回:
      带有新增列 "final_score"（以及中间归一化和子评分列）的 DataFrame
    """
    eps = 1e-8  # 防止除 0
    temp_value = 1
    df = result_df.copy()

    # -------------------------------
    # 1. 盈利能力指标归一化
    # -------------------------------
    for col in ['net_profit_rate', 'avg_profit_rate']:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if abs(max_val - min_val) < eps:
                df[col + '_norm'] = 1.0
            else:
                df[col + '_norm'] = df[col] / 200
        else:
            df[col + '_norm'] = 0.0

    # 盈利子评分：将归一化后的 net_profit_rate 和 avg_profit_rate 取平均
    df['profitability_score'] = (df['net_profit_rate_norm'] + df['avg_profit_rate_norm'])


    # -------------------------------
    # 2. 稳定性/风险指标归一化
    # 对于以下指标，原始数值越低越好，归一化后使用 1 - normalized_value
    # -------------------------------
    for col in ['loss_rate', 'monthly_loss_rate']:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if abs(max_val - min_val) < eps:
                df[col + '_score'] = 1.0
            else:
                df[col + '_score'] = 0.5 - df[col]
        else:
            df[col + '_score'] = 1.0

    # 基于月度平均收益标准差的波动性指标计算
    df['monthly_avg_profit_std_score'] = temp_value - df['monthly_avg_profit_std'] / (df['avg_profit_rate'].abs() + eps) * 100

    # 新增：基于月度净收益标准差的波动性指标计算
    df['monthly_net_profit_std_score'] = temp_value - df['monthly_net_profit_std'] / (df['net_profit_rate'].abs() + eps) * 22

    # 新增：整体平均收益的波动性指标计算
    df['avg_profit_std_score'] = temp_value - df['true_profit_std'] / df['avg_profit_rate'] * 100
    # -------------------------------
    # 3. 稳定性子评分构造
    # 四个风险指标平均：
    #   - loss_rate_score
    #   - monthly_loss_rate_score
    #   - risk_volatility_score (基于月均收益标准差)
    #   - risk_volatility_net_score (基于月净收益标准差)
    # -------------------------------
    df['stability_score'] = (
                                    df['loss_rate_score'] +
                                    df['monthly_loss_rate_score'] +
                                    df['monthly_net_profit_std_score']
                                    # df['monthly_avg_profit_std_score']
                                    # df['risk_volatility_avg_score'] / 2
                            )

    # -------------------------------
    # 4. 综合评分计算（加权组合）
    # 根据偏好：宁愿利润少一点，也不想经常亏损，故稳定性权重设为更高
    # -------------------------------
    profit_weight = 0.4  # 盈利性的权重
    stability_weight = 0.6  # 稳定性（风险控制）的权重
    df['final_score'] = profit_weight * df['profitability_score'] + stability_weight * df['stability_score']
    df['final_score'] = df['stability_score'] * df['profitability_score']
    # 删除final_score小于0的
    # df = df[(df['final_score'] > 0)]
    return df


def optimize_weekly_net_profit_detail_in_df(df, colname):
    """
    对 DataFrame 中指定列（colname）进行优化，
    将 weekly_net_profit_detail 或 monthly_net_profit_detail 数据转为 NumPy 数组存储，
    并将 None 或 NaN 值替换为 0。

    参数:
      df: pandas DataFrame
      colname: 需要处理的列名，列中每个元素的类型可以是字典、列表或其它。

    返回:
      处理后的 DataFrame，指定列中的数据转换为 NumPy 数组（dtype 为 float32）。
    """

    def optimize_detail(detail):
        # 缓存 pd.isna 函数以减少查找开销
        isna = pd.isna

        # 如果 detail 为 None 或 NaN，返回数组 [0]
        if detail is None or isna(detail):
            return np.array([0], dtype=np.float32)

        # 如果 detail 是字典：按 key 中日期排序（假定 key 格式为 "YYYY-MM-DD/..."），同时替换 None/NaN
        if isinstance(detail, dict):
            # sorted(detail.items()) 避免多余的 dict 查询操作
            sorted_items = sorted(detail.items(), key=lambda item: item[0].split('/')[0])
            return np.array(
                [val if (val is not None and not isna(val)) else 0 for _, val in sorted_items],
                dtype=np.float32
            )

        # 如果 detail 是列表，逐个检查并替换 None 或 NaN
        if isinstance(detail, list):
            return np.array(
                [val if (val is not None and not isna(val)) else 0 for val in detail],
                dtype=np.float32
            )

        # 对其他类型的数据，返回默认数组 [0]
        return np.array([0], dtype=np.float32)

    # 建议使用列表推导来替换 apply，提高一点性能
    df[colname] = [optimize_detail(detail) for detail in df[colname]]
    return df


def process_file(file):
    """处理单个文件的函数"""
    try:
        # 如果已知只需要部分列，可以利用 pd.read_parquet 的 columns 参数来加速读取
        df = pd.read_parquet(f'temp/{file}')

        # 丢弃不需要的列
        drop_cols = [
            # 'monthly_net_profit_detail',
            'monthly_trade_count_detail',
            # 'weekly_net_profit_detail',
            'weekly_trade_count_detail',
            'max_loss_start_time',
            'max_loss_end_time',
            'max_profit_start_time',
            'max_profit_end_time',
            'kai_side'
        ]
        df = df.drop(columns=drop_cols, errors='ignore')

        # 预计算 net_profit_rate 的平方，简化 profit_risk_score 的计算
        npr = df['net_profit_rate']
        df['profit_risk_score_con'] = -(npr * npr) / df['max_consecutive_loss']
        df['profit_risk_score'] = -(npr * npr) / df['fu_profit_sum']
        df['profit_risk_score_pure'] = -npr / df['fu_profit_sum']

        # 如有需要，可根据条件过滤数据（目前已注释）
        # df = df[(df['avg_profit_rate'] > 5)]
        # df = df[(df['hold_time_mean'] < 5000)]

        # 优化指定的 detail 列
        df = optimize_weekly_net_profit_detail_in_df(df, 'monthly_net_profit_detail')
        df = optimize_weekly_net_profit_detail_in_df(df, 'weekly_net_profit_detail')

        # 调用精度缩减函数（假设 auto_reduce_precision 是已定义函数）
        df = auto_reduce_precision(df)

        return df
    except Exception as e:
        print(f"处理文件 {file} 时报错: {e}")
        return None



def choose_good_strategy_debug(inst_id='BTC'):
    file_list = os.listdir('temp')
    file_list = [file for file in file_list if inst_id in file and
                 'donchian_1_20_1_relate_400_1000_100_1_40_6_cci_1_2000_1000_1_2_1_atr_1_3000_3000_boll_1_3000_100_1_50_2_rsi_1_1000_500_abs_1_100_100_40_100_1_macd_300_1000_50_macross_1_3000_100_1_3000_100_' in file and
                 'pkl' not in file and 'sta'  in file and '1m'  in file]
    print(f"找到 {len(file_list)} 个文件")
    # 使用多进程池并行处理文件
    with mp.Pool(processes=10) as pool:
        df_list = pool.map(process_file, file_list)

    # 过滤掉 None 值
    df_list = [df for df in df_list if df is not None]
    # 合并所有 DataFrame
    if df_list:
        result_df = pd.concat(df_list, ignore_index=True)
        return result_df
    else:
        print("没有符合条件的数据")
        return pd.DataFrame()


def generate_feature(df, feature):
    """
    根据传入的 df 和 feature 名称生成对应的特征列，
    支持两种格式：

    1. 单特征转换：格式为 "原始列名-转换类型"，
       转换类型包括: log, square, reciprocal, cube, sqrt, tanh, zscore

    2. 两两派生特征：格式为 "原始列1-原始列2-操作"，
       操作包括: ratio, diff, abs_diff, mean, prod, norm_diff

    如果传入的 feature 与上述格式均不匹配，但正好在 df 中存在，则不做转换直接返回 df。

    参数:
        df: pandas.DataFrame
            原始数据，要求包含执行转换所需的原始列。
        feature: str
            指定要生成的特征名称。

    返回:
        pandas.DataFrame: 包含新生成特征列的完整 DataFrame。
    """
    # 如果 feature 已经存在于 df 中，则认为无需生成，直接返回
    if feature in df.columns:
        print(f"特征 '{feature}' 已存在于 DataFrame 中，直接返回。")
        return df

    # 拆分 feature 名称，判断转换类型
    parts = feature.split("-")

    # 单特征转换：格式要求有两个部分：原始列名和转换类型
    if len(parts) == 2:
        orig_col, trans = parts
        if orig_col not in df.columns:
            raise KeyError(f"原始列 '{orig_col}' 不存在于 DataFrame 中")

        a = df[orig_col]

        if trans == "log":
            if (a <= 0).any():
                raise ValueError(f"对数变换要求所有值大于0，列：{orig_col}")
            df[feature] = np.log1p(a)
        elif trans == "square":
            df[feature] = a ** 2
        elif trans == "reciprocal":
            df[feature] = np.where(a == 0, np.nan, 1 / a)
        elif trans == "cube":
            df[feature] = a ** 3
        elif trans == "sqrt":
            if (a < 0).any():
                raise ValueError(f"平方根变换要求所有值不小于0，列：{orig_col}")
            df[feature] = np.sqrt(a)
        elif trans == "tanh":
            df[feature] = np.tanh(a)
        elif trans == "zscore":
            std_val = a.std()
            mean_val = a.mean()
            if std_val == 0:
                df[feature] = np.full_like(a, np.nan, dtype=np.float32)
            else:
                df[feature] = ((a - mean_val) / std_val).astype(np.float32)
        else:
            raise ValueError(f"未知的单特征转换类型：{trans}")
        print(f"成功生成单特征转换列：{feature}")

    # 两两派生特征：格式要求有三个部分：原始列1、原始列2及操作类型
    elif len(parts) == 3:
        col1, col2, op = parts
        if col1 not in df.columns or col2 not in df.columns:
            raise KeyError(f"原始列 '{col1}' 或 '{col2}' 不存在于 DataFrame 中")

        a = df[col1]
        b = df[col2]

        if op == "ratio":
            df[feature] = np.where(b == 0, np.nan, a / b)
        elif op == "diff":
            df[feature] = a - b
        elif op == "abs_diff":
            df[feature] = np.abs(a - b)
        elif op == "mean":
            df[feature] = (a + b) / 2
        elif op == "prod":
            # 若 a 和 b 均为负，则返回 -(|a|*|b|)
            df[feature] = np.where((a < 0) & (b < 0),
                                   - (np.abs(a) * np.abs(b)),
                                   a * b)
        elif op == "norm_diff":
            # 使用一个极小值，防止除 0 问题
            df[feature] = np.where((a + b) == 0, np.nan, (a - b) / (a + b + 1e-8))
        else:
            raise ValueError(f"未知的派生特征操作类型：{op}")
        print(f"成功生成派生特征列：{feature}")

    else:
        # 如果格式不满足上面两种情况，检查是否在DataFrame列中存在
        if feature in df.columns:
            print(f"特征 '{feature}' 已存在于 DataFrame 中，直接返回。")
        else:
            raise ValueError("feature 格式不正确，请使用 'col-transformation' 或 'col1-col2-operation' 格式")

    return df


def remove_duplicates(df, threshold, target_columns):
    """
    参数:
        df: pandas DataFrame 数据集
        threshold: 数值型阈值，当两行中 target_columns 对应的值的差均小于该阈值时，认为重复
        target_columns: 需要比较的目标列列表

    功能:
        删除重复的行，只保留重复组中“最大的”那一行。
        这里我们通过 target_columns 对应数值的和作为评判大小的标准，
        即和越大认为该行“越大”，故先对 DataFrame 按此得分降序排序，再进行去重。

        在去重的过程中，将记录所有的重复索引对，格式为 (保留行的原始索引, 重复行的原始索引)，
        方便后续排查和调试。

    返回:
        result_df: 处理后的 DataFrame（保持原始索引）
        duplicate_index_pairs: 记录的重复索引对列表，每个元素为 (保留行的原始索引, 重复行的原始索引)
    """
    # 重置df索引
    df = df.reset_index(drop=True)
    # 复制一份，避免修改原始数据
    df_copy = df.copy()

    # 计算一个“得分”用于排序（这里使用 target_columns 各列的和作为比较依据）
    df_copy["_score"] = df_copy[target_columns].sum(axis=1)

    # 按 _score 降序排序，得分最高的排在前面
    # 同时保留原始索引，存储在 "index" 列中
    df_copy = df_copy.sort_values(by="_score", ascending=False).reset_index(drop=False)

    n = df_copy.shape[0]
    kept_positions = []  # 存放保留行在排序后 DataFrame 中的位置
    dropped_positions = set()  # 存放已淘汰行的位置
    duplicate_index_pairs = []  # 用于记录重复的索引对 (kept_index, duplicate_index)

    # 遍历排序后的 DataFrame
    for i in range(n):
        if i in dropped_positions:
            continue  # 跳过已被其它行淘汰的行
        kept_positions.append(i)
        # 当前行在 target_columns 上的值，用于与之后的行比较
        row_i = df_copy.loc[i, target_columns]
        for j in range(i + 1, n):
            if j in dropped_positions:
                continue
            row_j = df_copy.loc[j, target_columns]
            # 如果 target_columns 所有列的差的绝对值均小于阈值，则认为这两行重复
            if all(abs(row_i[col] - row_j[col]) < threshold for col in target_columns):
                dropped_positions.add(j)
                # 记录重复索引对，格式: (当前保留行的原始索引, 被淘汰行的原始索引)
                duplicate_index_pairs.append((df_copy.loc[i, "index"], df_copy.loc[j, "index"]))

    # 取出保留的行，并删除辅助列 _score，将原始索引设置为 DataFrame 索引
    result_df = df_copy.loc[kept_positions].drop(columns=["_score"]).set_index("index")
    # 根据原始索引排序
    result_df = result_df.sort_index()

    return result_df, duplicate_index_pairs


def compute_similarity(index_set1, index_set2):
    """
    计算两个 index 集合的相似度，定义为：
      similarity = |A ∩ B| / min(|A|, |B|)
    如果任一集合为空，则返回 0.0
    """
    if not index_set1 or not index_set2:
        return 0.0
    return len(index_set1 & index_set2) / min(len(index_set1), len(index_set2))


def choose_good_strategy_detail(inst_id='BTC'):
    # 存储已保存数据的 index 集合（每个 accepted_set 对应一个已保存的 filter_data_df）
    accepted_index_sets = []
    # 用于存储所有已保存的 filter_data_df（用于最后合并保存）
    all_filtered_dfs = []

    # 计数：总共处理的 feature 数、保存的 feature 数、跳过的 feature 数
    total_features = 0
    saved_count = 0
    skipped_count = 0

    # 载入全部结果数据，并做初步筛选
    good_feature_df = pd.read_parquet('temp/all_result_df.parquet')
    sort_key = 'avg_net_profit_rate_new_positive_ratio'

    # 筛选 bin_seq 为 1 或 1000 的行，并进一步筛选 sort_key > 0 与 count_min 大于 10000
    good_feature_df = good_feature_df[
        # good_feature_df['bin_seq'].isin([1, 1000]) &
        (good_feature_df['avg_net_profit_rate_new_min'] > -10) &
        (good_feature_df[sort_key] > 0.8) &
        (good_feature_df['count_min'] > 10000)
        ]

    # # 保留 spearman_pos_ratio_min 与 spearman_pos_ratio_max 同号的数据
    # condition = (
    #         ((good_feature_df['spearman_pos_ratio_min'] > 0) & (good_feature_df['spearman_pos_ratio_max'] > 0)) |
    #         ((good_feature_df['spearman_pos_ratio_min'] < 0) & (good_feature_df['spearman_pos_ratio_max'] < 0))
    # )
    # good_feature_df = good_feature_df[condition]

    # 按 sort_key 降序排序，并对 feature 去重（保留第一次出现的记录）
    good_feature_df = good_feature_df.sort_values(by=sort_key, ascending=False)
    # good_feature_df = good_feature_df.drop_duplicates(subset=['feature'], keep='first')
    print(f"good_feature_df的行数为：{len(good_feature_df)}")

    # 载入详细区间数据及基础数据
    detail_data_df = pd.read_parquet(f'temp_back/combined_bin_analysis_{inst_id}_false.parquet')
    data_df = pd.read_parquet(f'temp/final_good_{inst_id}_false.parquet')
    data_df = data_df[data_df['hold_time_mean'] < 5000]
    data_df = data_df[data_df['kai_column'].str.contains('long', na=False)]
    print(f"【提示】：数据加载完成，数据行数：{data_df.shape[0]}")

    total_features = len(good_feature_df)

    # 遍历每个特征，生成对应 filter_data_df 后判断和已有数据的相似度
    for _, row in good_feature_df.iterrows():
        feature = row['feature']
        bin_seq = row['bin_seq']

        # 从 detail_data_df 中找到对应记录
        detail_row = detail_data_df[
            (detail_data_df['bin_seq'] == bin_seq) & (detail_data_df['feature'] == feature)
            ]
        if detail_row.empty:
            print(f"警告：未找到 feature '{feature}' 对应的 detail 数据")
            continue

        # 从 detail 中解析区间字符串，提取起始值和结束值
        bin_interval_str = detail_row['bin_interval'].values[0]
        numbers = re.findall(r"[-+]?\d*\.?\d+", bin_interval_str)
        if len(numbers) < 2:
            raise ValueError(f"无法从字符串中提取两个数值: {bin_interval_str}")
        start_value, end_value = map(float, numbers[:2])

        # 生成指定特征列 (假设 generate_feature 已定义)
        data_df = generate_feature(data_df, feature)

        # 筛选符合区间条件的数据
        filter_data_df = data_df[(data_df[feature] >= start_value) & (data_df[feature] < end_value)]
        if filter_data_df.shape[0] < 10000 or filter_data_df.shape[0] > 20000:
            print(f"警告：特征 '{feature}' 的筛选数据行数不足或包含 NaN 值，跳过该特征")
            continue
        print(
            f"针对特征 '{feature}' 初步筛选出的数据行数为：{len(filter_data_df)}，原始 count 为 {detail_row['count'].values[0]}"
        )

        # 根据特征名称判断是否删除data_df中的该列（避免重复计算）
        if len(feature.split("-")) > 1:
            data_df = data_df.drop(columns=[feature], errors='ignore')

        # 获取当前 filter_data_df 的 index 集合
        new_index_set = set(filter_data_df.index)

        # 检查与已保存数据的相似度，计算最大相似度
        similar_found = False
        max_similarity = 0
        for accepted_set in accepted_index_sets:
            similarity = compute_similarity(new_index_set, accepted_set)
            if similarity > max_similarity:
                max_similarity = similarity
            if similarity >= 0.5:
                similar_found = True
                break

        if similar_found:
            # 找到相似数据，则跳过保存该特征数据
            skipped_count += 1
            continue
        else:
            print(f"日志: 特征 '{feature}' 发现的最高相似度为 {max_similarity:.2f}，未发现相似数据，保存该特征数据")

        # 保存数据，并记录该 filter_data_df 的 index 集合
        accepted_index_sets.append(new_index_set)
        saved_count += 1
        output_filepath = f'temp/corr/{inst_id}_feature_{feature}_{bin_seq}.parquet'
        filter_data_df.to_parquet(output_filepath, index=False)
        print(f"特征 '{feature}' 的数据已保存到 {output_filepath}")

        # 同时加入列表以便最后合并
        all_filtered_dfs.append(filter_data_df)

    # 合并所有已保存的 filter_data_df，并去重后保存
    if all_filtered_dfs:
        combined_df = pd.concat(all_filtered_dfs).drop_duplicates(subset=['kai_column', 'pin_column'], keep='first')
        combined_filepath = f'temp/corr/{inst_id}_all_df.parquet'
        combined_df.to_parquet(combined_filepath, index=False)
        print(f"所有合并后的数据已保存到 {combined_filepath} 长度为：{len(combined_df)}")
    else:
        print("没有数据被保存，未生成合并文件。")

    # 打印整体处理情况
    print("============================================")
    print(f"总共处理的特征数量：{total_features}")
    print(f"最终保存的特征数量：{saved_count}")
    print(f"跳过保存的特征数量：{skipped_count}")
    print("============================================")



def delete_rows_based_on_sort_key(result_df, sort_key, range_key):
    """
    删除 DataFrame 中的行，使得每一行的 sort_key 都是当前及后续行中最大的。

    Args:
        result_df: Pandas DataFrame，必须包含 'sort_key' 列。

    Returns:
        Pandas DataFrame: 处理后的 DataFrame，删除了符合条件的行。
    """
    if result_df.empty:
        return result_df
    # 将result_df按照range_key升序排列
    result_df = result_df.sort_values(by=range_key, ascending=True)

    # 逆序遍历，保留 sort_key 最大的行
    max_sort_key = -float('inf')
    keep_mask = []  # 记录哪些行需要保留

    for sort_key_value in reversed(result_df[sort_key].values):  # .values 避免索引问题
        if sort_key_value >= max_sort_key:
            keep_mask.append(True)
            max_sort_key = sort_key_value
        else:
            keep_mask.append(False)

    # 由于是逆序遍历，最终的 keep_mask 需要反转
    keep_mask.reverse()

    return result_df[keep_mask].reset_index(drop=True)


def select_best_rows_in_ranges(df, range_size, sort_key, range_key='total_count'):
    """
    从 DataFrame 中按照指定范围选择最佳行，范围由 range_key 确定，排序由 sort_key 决定。

    Args:
        df (pd.DataFrame): 输入的 DataFrame。
        range_size (int): 每个范围的大小。
        sort_key (str): 用于排序的列名。
        range_key (str) : 用于确定范围的列名。

    Returns:
        pd.DataFrame: 包含每个范围内最佳行的 DataFrame。
    """

    # 确保输入的是 DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # 确保 range_size 是正整数
    if not isinstance(range_size, int) or range_size <= 0:
        raise ValueError("range_size must be a positive integer.")
    # 找到range_key大于0的行
    df = df[df[range_key] > 0]
    df = delete_rows_based_on_sort_key(df, sort_key, range_key)
    # 确保 sort_key 和 range_key 列存在于 DataFrame 中
    if sort_key not in df.columns:
        raise ValueError(f"Column '{sort_key}' not found in DataFrame.")
    if range_key not in df.columns:
        raise ValueError(f"Column '{range_key}' not found in DataFrame.")
    # 只保留sort_key大于0的行
    # df = df[df[sort_key] > 0]
    if df.empty:
        return df

    # 计算 DataFrame 的最大值，用于确定范围的上限
    max_value = df[range_key].max()
    min_value = df[range_key].min()

    # 初始化结果 DataFrame
    result_df = pd.DataFrame()

    # 循环遍历所有范围
    for start in range(min_value, int(max_value) + range_size, range_size):
        end = start + range_size

        # 筛选出当前范围的行, 注意这里用 range_key
        current_range_df = df[(df[range_key] >= start) & (df[range_key] < end)]

        # 如果当前范围有行，则按照 sort_key 排序选择最佳行并添加到结果 DataFrame
        if not current_range_df.empty:
            best_row = current_range_df.sort_values(by=sort_key, ascending=False).iloc[0]
            result_df = pd.concat([result_df, best_row.to_frame().T], ignore_index=True)
    result_df = delete_rows_based_on_sort_key(result_df, sort_key, range_key)

    return result_df


def merge_dataframes(df_list):
    if not df_list:
        return None  # 如果列表为空，返回None

    # 以第一个 DataFrame 为基准
    merged_df = df_list[0]
    # 生成一个空的DataFrame
    temp_df = pd.DataFrame()

    # 遍历后续 DataFrame 进行合并
    for i, df in enumerate(df_list[1:], start=1):
        merged_df = merged_df.merge(
            df,
            on=['kai_column', 'pin_column'],
            how='inner',
            suffixes=('', f'_{i}')  # 给重复列添加索引后缀
        )

    # **步骤 1：获取 df_list[0] 中所有数值列的前缀**
    numeric_cols = df_list[0].select_dtypes(include=[np.number]).columns  # 仅选择数值列
    temp_df['kai_side'] = merged_df['kai_side']
    temp_df['kai_column'] = merged_df['kai_column']
    temp_df['pin_column'] = merged_df['pin_column']

    # **步骤 2 & 3：在 merged_df 中找到这些前缀的列，并计算统计量**
    for prefix in numeric_cols:
        if 'net_profit_rate' in prefix:
            print()
        part_count = len(prefix.split('_'))
        try:
            relevant_cols = [col for col in merged_df.columns if col.startswith(prefix)]  # 找到所有相关列
            # relevant_cols中part_count只能小于等于part_count + 1
            relevant_cols = [col for col in relevant_cols if len(col.split('_')) <= part_count + 1]

            # relevant_cols必须以数字结尾
            relevant_cols = [col for col in relevant_cols if col.split('_')[-1].isdigit()]
            relevant_cols.append(prefix)

            if relevant_cols:  # 确保该前缀有对应的列
                merged_df[f'{prefix}_min'] = merged_df[relevant_cols].min(axis=1)
                merged_df[f'{prefix}_max'] = merged_df[relevant_cols].max(axis=1)
                merged_df[f'{prefix}_mean'] = merged_df[relevant_cols].mean(axis=1)
                merged_df[f'{prefix}_sum'] = merged_df[relevant_cols].sum(axis=1)
                merged_df[f'{prefix}_prod'] = merged_df[relevant_cols].prod(axis=1)
                temp_df[f'{prefix}_min'] = merged_df[f'{prefix}_min']
                temp_df[f'{prefix}_max'] = merged_df[f'{prefix}_max']
                temp_df[f'{prefix}_mean'] = merged_df[f'{prefix}_mean']
                temp_df[f'{prefix}_sum'] = merged_df[f'{prefix}_sum']
                temp_df[f'{prefix}_prod'] = merged_df[f'{prefix}_prod']
        except Exception as e:
            traceback.print_exc()
            print(f'出错：{e}')
    # 重新排序列
    columns = merged_df.columns.tolist()
    columns = columns[:3] + sorted(columns[3:])
    merged_df = merged_df[columns]
    return merged_df, temp_df


def gen_score(origin_good_df, key_name):
    origin_good_df[f'{key_name}_cha'] = origin_good_df[f'{key_name}_max'] - origin_good_df[f'{key_name}_min']
    origin_good_df[f'{key_name}_cha_ratio'] = origin_good_df[f'{key_name}_cha'] / origin_good_df[
        f'{key_name}_max'] * 100
    origin_good_df[f'{key_name}_score'] = 1 - origin_good_df[f'{key_name}_cha_ratio']
    return f'{key_name}_score'


def safe_parse_dict(val):
    """
    将字符串转换为字典，如果 val 已经是 dict 类型，则直接返回。
    如果转换失败，返回空字典。
    """
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            # 尝试将字符串转换为字典
            return ast.literal_eval(val)
        except Exception as e:
            # 转换失败时打印错误信息，并返回空字典
            print(f"转换错误: {e}，值为: {val}")
            return {}
    return {}

def compute_robust_correlation(detail_dict1, detail_dict2):
    """
    根据两个字典（key 为月份，value 为数据值）计算稳健相关性。

    计算方法：
      - 先得到两个字典共有的月份（排序后保证时间序列顺序）
      - 当共同月份少于 3 或任一数据序列标准差为 0 时，返回 0
      - 分别计算 Pearson 与 Spearman 相关系数，若 Spearman 相关系数为 nan 则置为 0
      - 返回两者均值作为稳健相关性
    """
    common_keys = sorted(set(detail_dict1.keys()) & set(detail_dict2.keys()))
    if len(common_keys) < 3:
        return 0

    x = np.array([detail_dict1[k] for k in common_keys])
    y = np.array([detail_dict2[k] for k in common_keys])

    std_x = np.std(x)
    std_y = np.std(y)
    if std_x == 0 or std_y == 0:
        return 0

    # 计算 Pearson 相关系数
    pearson_corr = np.corrcoef(x, y)[0, 1]

    # 计算 Spearman 相关系数
    spearman_corr, _ = spearmanr(x, y)
    if np.isnan(spearman_corr):
        spearman_corr = 0

    robust_corr = (pearson_corr + spearman_corr) / 2
    return robust_corr

def plot_comparison_chart(detail_dict1, detail_dict2, metric_name):
    """
    绘制两个字典数据的对比曲线图：
      - 仅绘制共同月份数据（排序后按时间顺序展示）
      - metric_name 为图标题及 y 轴标签
    """
    common_keys = sorted(set(detail_dict1.keys()) & set(detail_dict2.keys()))
    if not common_keys:
        print(f"没有共同月份数据，无法绘制 {metric_name} 的图表。")
        return

    x = common_keys
    y1 = [detail_dict1[k] for k in common_keys]
    y2 = [detail_dict2[k] for k in common_keys]

    plt.figure(figsize=(10, 5))
    plt.plot(x, y1, marker='o', label="Row1")
    plt.plot(x, y2, marker='o', label="Row2")
    plt.xlabel("month")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} curve")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def calculate_row_correlation(row1, row2, is_debug=False):
    """
    输入:
      row1, row2 : 两行回测结果（例如从 DataFrame 中提取的记录），
                   每行需包含以下字段：
                   - "monthly_net_profit_detail": 每个月的净利润数据（字典类型，预解析后）
                   - "monthly_trade_count_detail": 每个月的交易次数数据（字典类型，预解析后）
    计算方法:
      1. 对两个指标（净利润和交易次数），利用共同月份内的数据分别计算 Pearson 与 Spearman 相关性的均值；
      2. 取两项指标相关性的简单平均（范围 [-1, 1]）；
      3. 映射到 [-100, 100] 并返回整数。

    如 is_debug 为 True，同时绘制出对应的曲线图以直观观察数据对比。
    """
    profit_detail1 = row1.get("monthly_net_profit_detail", {})
    profit_detail2 = row2.get("monthly_net_profit_detail", {})
    trade_detail1 = row1.get("monthly_trade_count_detail", {})
    trade_detail2 = row2.get("monthly_trade_count_detail", {})

    if is_debug:
        # 绘制净利润对比图及交易次数对比图
        plot_comparison_chart(profit_detail1, profit_detail2, "net_profit")
        plot_comparison_chart(trade_detail1, trade_detail2, "kai_count")

    net_profit_corr = compute_robust_correlation(profit_detail1, profit_detail2)
    trade_count_corr = compute_robust_correlation(trade_detail1, trade_detail2)
    combined_corr = (net_profit_corr + trade_count_corr) / 2.0
    # 保证结果在 [-1, 1] 内
    combined_corr = max(min(combined_corr, 1), -1)
    # 映射到 [-100, 100] 并转换为整数
    final_value = int(round(combined_corr * 100))
    return final_value


def filter_similar_rows(inst_id, sort_key, threshold=10):
    """
    根据相关性过滤高度相似的数据行。
    逻辑说明：
      1. 按照sort_key从高到低排序，并筛选出sort_key大于0.1的行；
      2. 遍历排序后的每一行，与已经筛选出的行进行两两相关性比较；
      3. 如果该行与已经筛选出的每一行的相关性都小于或等于threshold，
         则将该行加入筛选结果 filtered_rows 中。

    参数:
      inst_id (str): 用于构成文件名的实例ID；
      sort_key (str): 用于排序的键；
      threshold (float): 相关性阈值，若相关性大于该值则认为两行过于相关，默认值为10。

    返回:
      pd.DataFrame: 筛选后的数据。
    """
    # 读取并预处理数据
    df = pd.read_csv(f'temp/final_good.csv')
    # df = df.sort_values(sort_key, ascending=False)
    # df = df[df[sort_key] > 0.1]


    # df = df[df['net_profit_rate'] > 50]
    # df = df[df['hold_time_mean'] < 1000]
    # 重置索引，并保留原始行标到 "index" 列中
    df = df.reset_index(drop=True)
    df = df.reset_index()  # 将原先的行号存到 "index" 列中

    # 对部分需要用字典进行解析的字段进行预处理
    df["monthly_net_profit_detail"] = df["monthly_net_profit_detail"].apply(safe_parse_dict)
    df["monthly_trade_count_detail"] = df["monthly_trade_count_detail"].apply(safe_parse_dict)

    # 转换为字典列表，保证遍历顺序与原 DataFrame 顺序一致
    parsed_rows = df.to_dict("records")
    filtered_rows = []
    print(f"初始数据量：{len(df)}")

    start_time = time.time()
    i = 0
    # 遍历每一条记录
    for candidate in parsed_rows:
        candidate_kai_count = candidate.get("kai_count")
        i += 1
        add_candidate = True
        # 与已筛选记录进行遍历对比
        for accepted in filtered_rows:
            accepted_kai_count = accepted.get("kai_count")
            corr_val = calculate_row_correlation(candidate, accepted)
            # 如果任一相关性大于阈值，则不加入该候选记录
            if abs(accepted_kai_count - candidate_kai_count) < 1 or corr_val > threshold:
                add_candidate = False
                break
        if add_candidate:
            filtered_rows.append(candidate)

    print(f"过滤后数据量：{len(filtered_rows)}")
    print(f"过滤耗时：{time.time() - start_time:.2f} 秒")

    # 构造返回数据 DataFrame
    filtered_df = pd.DataFrame(filtered_rows)
    filtered_df.to_csv(f'temp/{inst_id}_filtered_data.csv', index=False)
    return filtered_df


PARSED_ROWS = None

def init_worker(rows):
    """
    每个 worker 进程初始化时调用，将 parsed_rows 保存为全局变量 PARSED_ROWS
    """
    global PARSED_ROWS
    PARSED_ROWS = rows

def process_pair(pair):
    """
    处理单个行对的任务。
    参数:
      pair: 一个二元组 (i, j) 对应 parsed_rows 中的两个索引
    返回:
      如果计算出的相关性 < 1000，则返回包含相关信息的字典，否则返回 None。
    """
    i, j = pair
    row_a = PARSED_ROWS[i]
    row_b = PARSED_ROWS[j]
    corr_val = calculate_row_correlation(row_a, row_b)
    if corr_val < 1000:
        return {
            "Row1": row_a['index'],
            "Row2": row_b['index'],
            "Correlation": corr_val,
            "Row1_kai_side": row_a.get("kai_side"),
            "Row2_kai_side": row_b.get("kai_side"),
            "Row1_kai_column": row_a.get("kai_column"),
            "Row2_kai_column": row_b.get("kai_column"),
            "Row1_pin_column": row_a.get("pin_column"),
            "Row2_pin_column": row_b.get("pin_column"),
            "Row1_kai_count": row_a.get("kai_count"),
            "Row2_kai_count": row_b.get("kai_count"),
            "Row1_net_profit_rate": row_a.get("net_profit_rate"),
            "Row2_net_profit_rate": row_b.get("net_profit_rate"),
            "Row1_avg_profit_rate": row_a.get("avg_profit_rate"),
            "Row2_avg_profit_rate": row_b.get("avg_profit_rate")
        }
    return None


def process_group(args):
    """
    处理一个目标分组：
      - 根据 group_keys 筛选原始数据；
      - 根据 sort_key 降序排序；
      - 遍历排序后的记录，比较每行与已保留记录的相关性；
      - 相关性高于 threshold 则舍弃该行，否则保留。
    """
    group_keys, origin_good_df, target_column, sort_key, threshold = args
    group_df = origin_good_df[origin_good_df[target_column].isin(group_keys)]
    start_time = time.time()
    group_sorted = group_df.sort_values(by=sort_key, ascending=False)
    keep_rows = []

    for _, row in group_sorted.iterrows():
        drop_flag = False
        for kept_row in keep_rows:
            corr = calculate_row_correlation(row, kept_row)
            if corr > threshold:
                drop_flag = True
                break
        if not drop_flag:
            keep_rows.append(row)

    print(f"分组 {group_keys} 处理耗时：{time.time() - start_time:.2f} 秒 原始长度：{len(group_df)} 保留长度：{len(keep_rows)}")
    if keep_rows:
        return pd.DataFrame(keep_rows)
    else:
        # 返回空 DataFrame，列名与原始 DataFrame 保持一致
        return pd.DataFrame(columns=origin_good_df.columns)


def filtering(origin_good_df, target_column, sort_key, threshold):
    """
    对 DataFrame 进行预过滤，并使用多进程处理每个分组。处理逻辑如下：
      1. 对 target_column 升序排序，对唯一值进行分组。分组规则：
         - 相邻 target_column 值与当前组首个值的差值 <= 2；
         - 并且累计行数不超过 1000 行，累计行数基于各唯一值在原始数据中的出现次数计算。
         例如：若唯一值为 [55, 56, 57, 58, ...]，且 55 对应 400 行、56 对应 350 行，
         57 对应 300 行，则 [55, 56] 的累计为 750 行，加入 57 后累计达到 1050 行，超过限制，
         所以 57 单独起一个新组，即使 57-55 <= 2。
      2. 对每个分组内部根据 sort_key 降序排序后，依次比较各记录与已保留记录的相关性，
         若相关性大于 threshold，则该记录将被舍弃。
      3. 使用多进程（设置进程数为 5）同时处理各分组，提高过滤效率。

    参数:
      origin_good_df: pandas.DataFrame，原始数据
      target_column: str，用于分组的列名（要求数据为数值类型）
      sort_key: str，用于比较优先级的列名，值较大者优先保留
      threshold: float，相关性阈值，若两行的相关性大于该值，则认为两行高度相关

    返回:
      filtered_df: pandas.DataFrame，过滤后保留的记录
    """
    # 先按照 target_column 升序排序
    origin_good_df = origin_good_df.sort_values(by=target_column, ascending=True)

    # 提取 target_column 的所有唯一值，并保证升序
    unique_keys = sorted(origin_good_df[target_column].unique())
    # 获取每个唯一值对应的行数
    counts = origin_good_df[target_column].value_counts().to_dict()

    # 根据「目标值差值 <= 2」以及累计行数不超过 1000 行的规则进行分组
    grouped_keys = []  # 存储所有分组，每个分组为一组 target_column 值列表
    current_group = []
    current_group_count = 0

    for key in unique_keys:
        if not current_group:
            current_group = [key]
            current_group_count = counts.get(key, 0)
        else:
            # 如果当前 key 与组首的差值不超过 2 且累计行数+当前 key 的行数不超过 1000，则放入同一组
            if (key - current_group[0] <= 50) and (current_group_count + counts.get(key, 0) <= 1000):
                current_group.append(key)
                current_group_count += counts.get(key, 0)
            else:
                grouped_keys.append(current_group)
                current_group = [key]
                current_group_count = counts.get(key, 0)

    if current_group:
        grouped_keys.append(current_group)

    # 为每个分组准备处理时的参数列表
    pool_input_args = [
        (group_keys, origin_good_df, target_column, sort_key, threshold)
        for group_keys in grouped_keys
    ]
    print(f"分组数量：{len(pool_input_args)}")
    # 使用进程数为 5 的多进程池处理各个分组
    with Pool(processes=20) as pool:
        results = pool.map(process_group, pool_input_args)

    # 合并所有分组的过滤结果
    filtered_groups = [df for df in results if not df.empty]
    if filtered_groups:
        filtered_df = pd.concat(filtered_groups, ignore_index=True)
    else:
        filtered_df = pd.DataFrame(columns=origin_good_df.columns)

    return filtered_df

def gen_statistic_data(origin_good_df, threshold=99):
    """
    对原始 DataFrame 进行预处理：
      1. 重置索引并将原始索引保存到一列中；
      2. 对指标字段解析（调用 safe_parse_dict）；
      3. 计算所有行对的相关性（调用 calculate_row_correlation），采用并行化方法；
      4. 针对 negative_corr_df 中 Correlation 大于阈值的记录，
         比较 net_profit_rate，保留 net_profit_rate 较大的那一行，
         同时更新删除 origin_good_df 中对应的行；
      5. 返回更新后的 negative_corr_df 和 origin_good_df。

    返回的 negative_corr_df 包含以下列：
      'Row1', 'Row2', 'Correlation', 以及其他额外信息列。
    """
    start_time = time.time()

    # 重置索引，并保存原始索引到 "index" 列
    origin_good_df = origin_good_df.reset_index(drop=True)
    origin_good_df = origin_good_df.reset_index()  # "index" 列保存原始行标

    # 对指定字段进行解析
    origin_good_df["monthly_net_profit_detail"] = origin_good_df["monthly_net_profit_detail"].apply(safe_parse_dict)
    origin_good_df["monthly_trade_count_detail"] = origin_good_df["monthly_trade_count_detail"].apply(safe_parse_dict)
    print(f'待计算的数据量：{len(origin_good_df)}')
    origin_good_df = filtering(origin_good_df, 'kai_count', 'net_profit_rate', 95)
    print(f'过滤后的数据量：{len(origin_good_df)}')

    # 转换为字典列表，保持 DataFrame 内的顺序
    parsed_rows = origin_good_df.to_dict("records")
    n = len(parsed_rows)

    # 生成所有行对的索引组合
    pair_indices = list(itertools.combinations(range(n), 2))

    results = []
    # 使用 ProcessPoolExecutor 并行计算行对相关性
    with ProcessPoolExecutor(max_workers=30, initializer=init_worker, initargs=(parsed_rows,)) as executor:
        for res in executor.map(process_pair, pair_indices, chunksize=1000):
            if res is not None:
                results.append(res)

    columns = [
        "Row1", "Row2", "Correlation",
        "Row1_kai_side", "Row2_kai_side",
        "Row1_kai_column", "Row2_kai_column",
        "Row1_pin_column", "Row2_pin_column",
        "Row1_kai_count", "Row2_kai_count",
        "Row1_net_profit_rate", "Row2_net_profit_rate",
        "Row1_avg_profit_rate", "Row2_avg_profit_rate"
    ]
    negative_corr_df = pd.DataFrame(results, columns=columns)
    negative_corr_df.to_csv('temp/negative_corr.csv', index=False)
    origin_good_df.to_csv('temp/origin_good.csv', index=False)
    print(f'计算耗时：{time.time() - start_time:.2f} 秒')

    # -------------------------------
    # 根据阈值 threshold 处理负相关数据及更新原始数据
    # -------------------------------

    # 1. 标记需要删除的行 —— 对于 negative_corr_df 中 Correlation 大于阈值的记录，
    #    比较 Row1_net_profit_rate 和 Row2_net_profit_rate，删除净利润较低者
    indices_to_remove = set()
    high_corr = negative_corr_df[negative_corr_df['Correlation'] > threshold]
    for _, row in high_corr.iterrows():
        if row['Row1_net_profit_rate'] >= row['Row2_net_profit_rate']:
            remove_idx = row['Row2']
        else:
            remove_idx = row['Row1']
        indices_to_remove.add(remove_idx)
    print(f'需要删除的行数：{len(indices_to_remove)}')
    # 2. 更新 origin_good_df：将被标记删除的行移除
    origin_good_df = origin_good_df[~origin_good_df['index'].isin(indices_to_remove)]
    origin_good_df = origin_good_df.reset_index(drop=True)

    # 3. 更新 negative_corr_df：删除包含已删除行的记录
    negative_corr_df = negative_corr_df[~(negative_corr_df['Row1'].isin(indices_to_remove) | negative_corr_df['Row2'].isin(indices_to_remove))]

    return negative_corr_df, origin_good_df


def filter_param(inst_id):
    """
    生成更加仔细的搜索参数
    :return:
    """
    range_size = 1
    output_path = f'temp/{inst_id}_good.csv'
    if os.path.exists(output_path):
        return pd.read_csv(output_path)
    range_key = 'kai_count'

    target_key = ['net_profit_rate', 'avg_profit_rate', 'stability_score', 'final_score', 'score', 'monthly_net_profit_min', 'loss_rate_score', 'monthly_loss_rate_score', 'avg_profit_std_score',
                  'monthly_net_profit_std_score','monthly_avg_profit_std_score'
                  ]
    max_consecutive_loss_list = [-5, -10, -15, -100]
    good_df_list = []
    all_origin_good_df = pd.read_csv(f'temp/{inst_id}_origin_good_op_all.csv')
    false_all_origin_good_df = all_origin_good_df[(all_origin_good_df['is_reverse'] == False)]
    true_all_origin_good_df = all_origin_good_df[(all_origin_good_df['is_reverse'] == True)]

    for max_consecutive_loss in max_consecutive_loss_list:
        origin_good_df = false_all_origin_good_df.copy()
        origin_good_df['score'] = -origin_good_df['net_profit_rate'] / origin_good_df['max_consecutive_loss'] * \
                                  origin_good_df['net_profit_rate']
        origin_good_df = origin_good_df[(origin_good_df['max_consecutive_loss'] > max_consecutive_loss)]

        origin_good_df = origin_good_df.drop_duplicates(subset=['kai_column', 'pin_column'], keep='first')

        for sort_key in target_key:
            origin_good_df = origin_good_df.drop_duplicates(subset=['kai_column', 'pin_column'], keep='first')
            good_df = origin_good_df.sort_values(sort_key, ascending=False)
            long_good_strategy_df = good_df[good_df['kai_side'] == 'long']
            short_good_strategy_df = good_df[good_df['kai_side'] == 'short']

            # 将long_good_strategy_df按照net_profit_rate_mult降序排列
            long_good_select_df = select_best_rows_in_ranges(long_good_strategy_df, range_size=range_size,sort_key=sort_key, range_key=range_key)
            short_good_select_df = select_best_rows_in_ranges(short_good_strategy_df, range_size=range_size,sort_key=sort_key, range_key=range_key)
            good_df = pd.concat([long_good_select_df, short_good_select_df])
            good_df_list.append(good_df)
    for max_consecutive_loss in max_consecutive_loss_list:
        origin_good_df = true_all_origin_good_df.copy()
        origin_good_df['score'] = -origin_good_df['net_profit_rate'] / origin_good_df['max_consecutive_loss'] * \
                                  origin_good_df['net_profit_rate']
        origin_good_df = origin_good_df[(origin_good_df['max_consecutive_loss'] > max_consecutive_loss)]

        origin_good_df = origin_good_df.drop_duplicates(subset=['kai_column', 'pin_column'], keep='first')

        for sort_key in target_key:
            origin_good_df = origin_good_df.drop_duplicates(subset=['kai_column', 'pin_column'], keep='first')
            good_df = origin_good_df.sort_values(sort_key, ascending=False)
            long_good_strategy_df = good_df[good_df['kai_side'] == 'long']
            short_good_strategy_df = good_df[good_df['kai_side'] == 'short']

            # 将long_good_strategy_df按照net_profit_rate_mult降序排列
            long_good_select_df = select_best_rows_in_ranges(long_good_strategy_df, range_size=range_size,sort_key=sort_key, range_key=range_key)
            short_good_select_df = select_best_rows_in_ranges(short_good_strategy_df, range_size=range_size,sort_key=sort_key, range_key=range_key)
            good_df = pd.concat([long_good_select_df, short_good_select_df])
            good_df_list.append(good_df)
    result_df = pd.concat(good_df_list)
    result_df = result_df.drop_duplicates(subset=['kai_column', 'pin_column'], keep='first')
    result_df.to_csv(output_path, index=False)
    return result_df

def gen_extend_columns(columns):
    """
    生成扩展列。columns可能的格式有rsi_75_30_70_high_long，abs_6_3.6_high_long，relate_1067_4_high_long
    :param columns:
    :return:
    """
    parts = columns.split('_')
    period = int(parts[1])
    type = parts[0]
    # 生成period前后100个period
    period_list = [str(i) for i in range(period - 5, period + 5)]
    # 筛选出大于1
    period_list = [i for i in period_list if int(i) > 0]
    if type == 'rsi':
        value1 = int(parts[2])
        value2 = int(parts[3])
        value1_list = [str(i) for i in range(value1 - 5, value1 + 5)]
        value2_list = [str(i) for i in range(value2 - 5, value2 + 5)]
        value1_list = [i for i in value1_list if int(i) > 0 and int(i) < 100]
        value2_list = [i for i in value2_list if int(i) > 0 and int(i) < 100]
        return [f'{type}_{period}_{value1}_{value2}_{parts[4]}_{parts[5]}' for period in period_list for value1 in value1_list for value2 in value2_list]
    if type == 'macross':
        value1 = int(parts[2])
        value1_list = [i for i in range(value1 - 5, value1 + 5)]
        value1_list = [i for i in value1_list if i > 0]
        return [f'{type}_{period}_{value1}_{parts[3]}_{parts[4]}' for period in period_list for value1 in value1_list]
    elif type == 'abs':
        value1 = int(float(parts[2]) * 10)
        value1_list = [i / 10 for i in range(value1 - 5, value1 + 5)]
        value1_list = [i for i in value1_list if i > 0]
        return [f'{type}_{period}_{value1}_{parts[3]}_{parts[4]}' for period in period_list for value1 in value1_list]
    elif type == 'relate':
        value1 = int(parts[2])
        value1_list = [i for i in range(value1 - 5, value1 + 5)]
        value1_list = [i for i in value1_list if i > 0]
        return [f'{type}_{period}_{value1}_{parts[3]}_{parts[4]}' for period in period_list for value1 in value1_list]
    elif type == 'ma':
        return [f'{type}_{period}_{parts[2]}_{parts[3]}' for period in period_list]

    elif type == 'peak':
        return [f'{type}_{period}_{parts[2]}_{parts[3]}' for period in period_list]
    elif type == 'continue':
        return [f'{type}_{period}_{parts[2]}_{parts[3]}' for period in period_list]
    else:
        print(f'error type:{type}')
        return columns

def gen_search_param(inst_id, is_reverse=False):
    all_task_list = []
    good_df = filter_param(inst_id)
    good_df = good_df[(good_df['is_reverse'] == is_reverse)]
    all_columns = []
    # 遍历每一行
    for index, row in good_df.iterrows():
        kai_column = row['kai_column']
        pin_column = row['pin_column']
        kai_column_list = gen_extend_columns(kai_column)
        pin_column_list = gen_extend_columns(pin_column)
        task_list = list(product(kai_column_list, pin_column_list))
        all_task_list.extend(task_list)
        all_columns.extend(kai_column_list)
        all_columns.extend(pin_column_list)
        # all_task_list = list(set(all_task_list))
    # 删除all_task_list中重复的元素
    all_task_list = list(set(all_task_list))
    all_columns = list(set(all_columns))
    return all_task_list, all_columns


def find_all_valid_groups(origin_good_df, threshold, sort_key='net_profit_rate', min_group_size=3):
    """
    枚举 origin_good_df 处理后的统计数据中所有满足条件的 row 组合，
    使得组合中任意两个 row 的 Correlation 都低于给定阈值。

    参数:
      origin_good_df: pandas.DataFrame，原始数据（须包含如 monthly_net_profit_detail 等字段）
      threshold: float，判定相关系数是否“过高”的阈值

    返回:
      groups_with_avg: list，每个元素是一个 tuple (group, avg_corr)，
         group 为 list，表示一组满足条件的 row 集合（极大独立集，长度至少2）
         avg_corr 为 float，该集合中所有两两关系的平均相关性
      df: 生成统计数据的 DataFrame
    """
    df, origin_good_df = gen_statistic_data(origin_good_df)
    df.to_csv('temp/df.csv', index=False)
    return origin_good_df,origin_good_df,df
    total_start = time.time()

    ### 阶段1: 读取数据（优先读取 Parquet 文件）
    csv_path = 'temp/df.csv'
    print("[阶段1] 开始读取 CSV 文件...")
    t_stage = time.time()
    df = pd.read_csv(csv_path)
    print(f"[阶段1] CSV 文件读取完成, 总记录数: {df.shape[0]}, 耗时 {time.time() - t_stage:.6f} 秒.")

    ### 阶段2: 生成 key 列（利用向量化计算）
    print("[阶段2] 开始生成 min_vals、max_vals 及 key...")
    t_stage = time.time()
    min_vals = df[["Row1", "Row2"]].min(axis=1)
    max_vals = df[["Row1", "Row2"]].max(axis=1)
    df["key"] = list(zip(min_vals, max_vals))
    print(f"[阶段2] 完成 key 生成, 耗时 {time.time() - t_stage:.6f} 秒.")

    ### 阶段3: 构造相关性字典（保留所有记录，用于后续统计）
    print("[阶段3] 构造相关性字典...")
    t_stage = time.time()
    # 为了保证 (a, b) 与 (b, a) 一致，以排序后的 tuple 作为 key
    corr_dict = dict(zip(df["key"], df["Correlation"]))
    print(f"[阶段3] 完成构造相关性字典, 总记录数: {len(corr_dict)}, 耗时 {time.time() - t_stage:.6f} 秒.")

    ### 阶段4: 利用阈值过滤边数据，并构造 igraph 图
    print("[阶段4] 过滤相关性满足条件的边...")
    t_stage = time.time()
    edge_df = df[df["Correlation"] >= threshold]
    print(f"[阶段4] 过滤后边数量: {edge_df.shape[0]}, 耗时 {time.time() - t_stage:.6f} 秒.")

    print("[阶段4] 构建节点集合...")
    t_stage = time.time()
    nodes = list(set(df["Row1"]).union(set(df["Row2"])))
    print(f"[阶段4] 节点集合构建完成, 总节点数: {len(nodes)}, 耗时 {time.time() - t_stage:.6f} 秒.")

    print("[阶段4] 构造 igraph 图...")
    t_stage = time.time()
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    # 构造边列表，转换为节点索引对
    edges_list = [(node_to_index[u], node_to_index[v])
                  for u, v in edge_df[["Row1", "Row2"]].to_numpy()]
    g = ig.Graph()
    g.add_vertices(len(nodes))
    g.add_edges(edges_list)
    print(f"[阶段4] igraph 图构建完成, 边数量: {len(edges_list)}, 耗时 {time.time() - t_stage:.6f} 秒.")

    ### 阶段5: 构造补图，并利用 igraph 高效求解 maximal cliques（对应原图独立集）
    print("[阶段5] 构造补图...")
    t_stage = time.time()
    gc = g.complementer()
    print(f"[阶段5] 补图构造完成, 节点数: {len(gc.vs)}, 边数: {len(gc.es)}, 耗时 {time.time() - t_stage:.6f} 秒.")

    print("[阶段5] 求解满足条件的 maximal cliques (独立集)...")
    t_stage = time.time()
    all_cliques = gc.maximal_cliques()
    cliques_raw = [clique for clique in all_cliques if len(clique) >= min_group_size]
    print(f"[阶段5] 找到 clique 数量: {len(cliques_raw)}, 耗时 {time.time() - t_stage:.6f} 秒.")

    print("[阶段5] 转换 igraph 顶点索引为节点名称...")
    t_stage = time.time()
    cliques = [[nodes[i] for i in clique] for clique in cliques_raw]
    print(f"[阶段5] 转换完成, 耗时 {time.time() - t_stage:.6f} 秒.")

    ### 阶段6: 依次计算每个 clique 的统计指标（顺序计算，不使用多进程）
    print("[阶段6] 计算每个 clique 的统计指标 (平均、最小、最大相关性) - 顺序计算...")
    t_stage = time.time()

    def calc_stats(group):
        """计算 group 内所有 pair 的相关性统计指标：平均、最小、最大"""
        group_sorted = sorted(group)
        combs = list(itertools.combinations(group_sorted, 2))
        if not combs:
            return 0, 0, 0
        # 确保 (a, b) 与 (b, a) 统一，直接对 pair 排序后查找 corr_dict
        corr_values = [corr_dict.get(tuple(sorted(pair)), 0) for pair in combs]
        avg_corr = np.mean(corr_values)
        min_corr = np.min(corr_values)
        max_corr = np.max(corr_values)
        return avg_corr, min_corr, max_corr

    groups_stats = []
    for clique in cliques:
        avg_corr, min_corr, max_corr = calc_stats(clique)
        groups_stats.append((clique, avg_corr, min_corr, max_corr))
    print(f"[阶段6] 完成统计指标计算, 处理组合数量: {len(groups_stats)}, 耗时 {time.time() - t_stage:.6f} 秒.")

    ### 阶段7: 对组合统计指标进行排序（先按组合大小降序，再按平均相关性升序）
    print("[阶段7] 排序组合统计指标...")
    t_stage = time.time()
    groups_stats.sort(key=lambda x: (-len(x[0]), x[1]))
    print(f"[阶段7] 排序完成, 耗时 {time.time() - t_stage:.6f} 秒.")

    ### 阶段8: 构建 sort_key 映射
    print("[阶段8] 构建 sort_key 映射...")
    t_stage = time.time()
    sort_key_mapping = {str(k): v for k, v in origin_good_df.set_index("index")[sort_key].to_dict().items()}
    print(f"[阶段8] 映射构建完成, 总映射键数量: {len(sort_key_mapping)}, 耗时 {time.time() - t_stage:.6f} 秒.")

    ### 阶段9: 组装最终结果
    print("[阶段9] 组装最终结果...")
    t_stage = time.time()
    results = []
    for clique, avg_corr, min_corr, max_corr in groups_stats:
        sort_values = [sort_key_mapping.get(str(r), np.nan) for r in clique]
        avg_sort_key_value = np.nanmean(sort_values)
        results.append({
            "row_list": clique,
            "avg_corr": avg_corr,
            "row_len": len(clique),
            "avg_sort_key_value": avg_sort_key_value,
            "min_corr": min_corr,
            "max_corr": max_corr
        })
    print(f"[阶段9] 组装完成, 最终组合数量: {len(results)}, 耗时 {time.time() - t_stage:.6f} 秒.")

    final_df = pd.DataFrame(results)
    total_time = time.time() - total_start
    print(f"[完成] 所有阶段完成，总耗时 {total_time:.6f} 秒.")

    return final_df, origin_good_df, df

def get_metrics_df(df):
    # # debug
    # df = pd.read_csv("temp/metrics_df.csv")
    # # 获取df['overlap_ratio1']和df['overlap_ratio2']中小的值作为新的一列
    # df['overlap_ratio'] = df[['overlap_ratio1', 'overlap_ratio2']].max(axis=1)
    # df['overlapping_revenue_ratio'] = df[['overlapping_revenue_ratio1', 'overlapping_revenue_ratio2']].max(axis=1)
    # df['score'] = df['overlap_ratio'] + df['overlapping_revenue_ratio']
    # df['score1'] = df['overlap_ratio'] * df['overlapping_revenue_ratio']
    # # 找到index为[584,3207]的行
    # print(df.loc[[584, 3207]])
    start_time = time.time()
    print("开始处理数据...长度为", len(df))
    # 预处理每一行数据，使用 tqdm 显示进度，并计算预处理结果
    custom_pre_data = [
        preprocess_row_custom(row["kai_data_df_tuples"])
        for _, row in tqdm(df.iterrows(), total=len(df), desc="预处理行数据", unit="row")
    ]

    num_rows = len(custom_pre_data)
    # 构造所有的两两组合（保证使用连续的索引）
    pairs = itertools.combinations(range(num_rows), 2)
    total_tasks = num_rows * (num_rows - 1) // 2  # 两两组合的总数量

    max_workers = os.cpu_count() or 1
    factor = 20  # 可调参数，factor 越大，chunksize 越小
    chunksize = max(10000, total_tasks // (max_workers * factor))
    print(f"总任务数: {total_tasks}, 工作进程数: {max_workers}, 动态计算 chunksize: {chunksize}")

    # 使用 ProcessPoolExecutor，并在初始化时传入 custom_pre_data
    with ProcessPoolExecutor(max_workers=30, initializer=init_custom_worker, initargs=(custom_pre_data,)) as executor:
        # 使用 tqdm 包裹 executor.map 的返回结果显示进度（单位为 pair）
        results = list(tqdm(
            executor.map(process_pair_custom, pairs, chunksize=chunksize),
            total=total_tasks,
            desc="处理 pairs",
            unit="pair"
        ))

    # 将结果转换为 DataFrame，并写入 CSV
    metrics_df = pd.DataFrame(results)
    metrics_df['overlap_count_ratio'] = metrics_df[['overlap_count_ratio1', 'overlap_count_ratio2']].max(axis=1)
    metrics_df['overlap_ratio'] = metrics_df[['overlap_ratio1', 'overlap_ratio2']].max(axis=1)
    metrics_df['overlapping_revenue_ratio'] = metrics_df[
        ['overlapping_revenue_ratio1', 'overlapping_revenue_ratio2']].max(axis=1)
    metrics_df['score'] = metrics_df['overlap_ratio'] + metrics_df['overlapping_revenue_ratio']
    metrics_df['score1'] = metrics_df['overlap_ratio'] * metrics_df['overlapping_revenue_ratio']
    metrics_df.to_csv("temp/metrics_df.csv", index=False)
    print(f'行组合数量: {metrics_df.shape[0]} 完成, 总耗时 {time.time() - start_time:.2f} 秒.')

def get_statistic(good_df):
    """
    使用 groupby().transform('size') 优化统计函数。
    """
    # 1. 使用 transform('size') 直接计算并对齐计数值
    # 它会按 'kai_column' 分组，计算每组的大小，并将该大小赋给组内所有行
    good_df['kai_value'] = good_df.groupby('kai_column')['kai_column'].transform('size')
    good_df['pin_value'] = good_df.groupby('pin_column')['pin_column'].transform('size')

    # 2. 后续计算已经是向量化操作，本身是高效的
    good_df['value_score'] = good_df['kai_value'] + good_df['pin_value']
    good_df['value_score1'] = good_df['kai_value'] * good_df['pin_value']

    return good_df

def get_data(origin_good_df):

    # 2. 定义需要计算的分位数列表
    # 对于 score_score，我们希望取“高分”部分，因此利用 (1 - quantile) 来确定临界值，
    # 而对于 score_score1，我们希望取“低分”部分，因此直接利用 quantile。
    # quantiles = [x / 100 for x in range(50, 70)]
    # quantiles.append(1)

    quantiles = [x / 10 for x in range(1, 11)]


    results = []

    for q in quantiles:
        # 计算 score_score 的临界值
        # 注意：由于 score_score 越大越好，取 top q 的数据，
        # 因此临界值为该列在 (1-q) 分位数位置的值。
        threshold1 = origin_good_df['score_score'].quantile(1 - q)

        # 计算 score_score1 的临界值
        # 由于 score_score1 越小越好，取低分部分，即下 q 分位数
        threshold2 = origin_good_df['score_score1'].quantile(q)

        # 筛选数据：要求 score_score 大于 threshold1 且 score_score1 小于 threshold2
        filtered_df = origin_good_df[
            (origin_good_df['score_score'] >= threshold1) &
            (origin_good_df['score_score1'] <= threshold2)
            ]

        # 统计筛选后数据中 avg_net_profit_rate_new 为正的个数、比例和均值
        total_count = filtered_df.shape[0]
        pos_count = (filtered_df['net_profit_rate_new'] > 0).sum()
        pos_ratio = pos_count / total_count if total_count > 0 else np.nan
        avg_value = filtered_df['net_profit_rate_new'].mean() if total_count > 0 else np.nan

        results.append({
            'quantile': q,
            'threshold_score': threshold1,
            'threshold_score1': threshold2,
            'filtered_count': total_count,
            'positive_count': pos_count,
            'positive_ratio': pos_ratio,
            'avg_net_profit_rate_new': avg_value
        })

    # 3. 将结果汇总到一个 DataFrame 中
    result_df = pd.DataFrame(results)
    return result_df


def get_data_detail(origin_good_df, half_range_list=[x/100 for x in range(1, 21)]):
    """
    针对每个目标列和每个给定的半区间值，生成区间 [0.5 - half_range, 0.5 + half_range]，
    筛选出满足该区间条件的数据，并统计数据条数、正向个数、正向比例，以及 net_profit_rate_new 的平均值。

    参数:
        origin_good_df: 原始 DataFrame 数据，必须包含 'net_profit_rate_new' 列。
        target_columns: 需要进行筛选的目标列名列表。
        half_range_list: 半区间列表，例如 [0.01, 0.02] 将生成区间 [0.49, 0.51] 和 [0.48, 0.52]。

    返回:
        汇总统计结果的 DataFrame，每一行为一个目标列在某个区间的统计结果。
    """
    all_numeric_columns = origin_good_df.select_dtypes(include=np.number).columns.tolist()
    orig_feature_cols = [col for col in all_numeric_columns
                         if col not in ['timestamp', 'net_profit_rate_new'] and 'new' not in col]
    target_columns = orig_feature_cols
    results = []
    half_range_list.append(0.5)
    # 遍历目标列
    for target in target_columns:
        if target not in origin_good_df.columns:
            print(f"列 '{target}' 不在 DataFrame 中，跳过该列。")
            continue

        # 遍历每个半区间值，计算对应的百分位区间
        for half_range in half_range_list:
            left_percent = 0.5 - half_range
            right_percent = 0.5 + half_range

            # 获取目标列左右百分位对应的数值（临界值）
            lower_threshold = origin_good_df[target].quantile(left_percent)
            upper_threshold = origin_good_df[target].quantile(right_percent)

            # 根据临界值筛选数据：选出目标列数值位于 [lower_threshold, upper_threshold] 范围内的记录
            filtered_df = origin_good_df[(origin_good_df[target] >= lower_threshold) &
                                         (origin_good_df[target] <= upper_threshold)]

            total_count = filtered_df.shape[0]
            pos_count = (filtered_df['net_profit_rate_new'] > 0).sum()
            pos_ratio = pos_count / total_count if total_count > 0 else np.nan
            avg_value = filtered_df['net_profit_rate_new'].mean() if total_count > 0 else np.nan

            results.append({
                'target_column': target,
                'half_range': half_range,
                'left_percent': left_percent,
                'right_percent': right_percent,
                'lower_threshold': lower_threshold,  # 左百分位对应的数值
                'upper_threshold': upper_threshold,  # 右百分位对应的数值
                'filtered_count': total_count,
                'positive_count': pos_count,
                'positive_ratio': pos_ratio,
                'avg_net_profit_rate_new': avg_value
            })

    # 将所有统计结果汇总到一个 DataFrame 中
    result_df = pd.DataFrame(results)
    return result_df


def get_data_detail_op(origin_good_df, half_range_list=None):
    """
    针对所有目标列（排除 'timestamp', 'net_profit_rate_new' 以及包含 'new' 的列）和每个给定的半区间值，
    分别计算每个目标列在 [0.5 - half_range, 0.5 + half_range] 区间对应的临界值，
    然后筛选出所有目标列数据均位于各自对应区间内的记录，
    最后统计该筛选结果的记录条数、正向（net_profit_rate_new > 0）个数、正向比例以及 net_profit_rate_new 的均值.

    参数:
        origin_good_df: 原始 DataFrame 数据，必须包含 'net_profit_rate_new' 列.
        half_range_list: 半区间列表，例如 [0.01, 0.02] 会生成区间 [0.49, 0.51] 和 [0.48, 0.52]，默认范围为 0.01 至 0.49（同时确保 0.5 被包含）.

    返回:
        汇总统计结果的 DataFrame，每一行为一个半区间对应的总体统计结果.
    """

    # 默认半区间列表
    if half_range_list is None:
        half_range_list = [x / 100 for x in range(1, 50)]
    # 确保列表中包含 0.5
    if 0.5 not in half_range_list:
        half_range_list.append(0.5)
    half_range_list = sorted(list(set(half_range_list))) # 去重并排序，避免重复计算

    # 获取数值型列，排除指定的列
    all_numeric_columns = origin_good_df.select_dtypes(include=np.number).columns.tolist()
    target_columns = [col for col in all_numeric_columns
                      if col not in ['timestamp', 'net_profit_rate_new'] and 'new' not in col]

    # --- 优化点 0: 提前获取需要计算的 Series ---
    # 避免在循环中反复通过列名索引 Series
    target_data = origin_good_df[target_columns]
    profit_rate_series = origin_good_df['net_profit_rate_new']

    results = []

    # 预先确定所有需要计算的百分比
    required_percents = set()
    for half_range in half_range_list:
        # 确保百分比在有效范围内 [0, 1]
        left_percent = max(0, 0.5 - half_range)
        right_percent = min(1, 0.5 + half_range)
        required_percents.add(left_percent)
        required_percents.add(right_percent)
    required_percents = sorted(required_percents)

    # 预计算所有目标列在各个百分比位置上的阈值
    # 如果 target_columns 为空，提前返回，避免 quantile 出错
    if not target_columns:
         print("警告：没有找到符合条件的目标列 (target_columns)。")
         # 可以选择返回空 DataFrame 或根据业务逻辑处理
         return pd.DataFrame(results)

    # --- 性能关键点: Quantile 计算 ---
    # 这一步如果耗时，可以考虑：
    # 1. 抽样计算（如果精度要求不高）
    # 2. 使用更快的库（如 Polars）
    # 3. 检查数据类型是否最优 (e.g., float32 vs float64)
    quantiles_df = target_data.quantile(q=required_percents)

    # --- 优化点 1: 预计算 profit rate 的正负性 ---
    # 避免在循环中重复比较
    is_positive = profit_rate_series > 0
    is_negative = profit_rate_series < 0


    # 对每个半区间分别计算统计量
    for half_range in half_range_list:
        # 确保百分比在有效范围内 [0, 1]
        left_percent = max(0, 0.5 - half_range)
        right_percent = min(1, 0.5 + half_range)

        # 从预计算数据中提取每个目标列的左右阈值
        thresholds_lower = quantiles_df.loc[left_percent]
        thresholds_upper = quantiles_df.loc[right_percent]

        # --- 性能关键点: 计算布尔掩码 (Mask) ---
        # 利用 Pandas 广播机制，一次性对所有目标列进行条件判断
        # 使用预先获取的 target_data 提高效率
        condition_df = (target_data.ge(thresholds_lower)) & \
                       (target_data.le(thresholds_upper))
        # 得到所有目标列均满足条件的样本（每行所有值均为 True）
        # .values.all(axis=1) 可能比 .all(axis=1) 稍微快一点，尤其列多时
        mask = condition_df.values.all(axis=1) # 或者保持 .all(axis=1)

        # --- 优化点 2: 直接使用 mask 计算统计量，避免创建 filtered_df ---
        total_count = mask.sum() # 直接对布尔 Series 求和得到 True 的数量

        if total_count > 0:
            # 使用预计算的 is_positive / is_negative 和 mask 进行与运算
            pos_count = (mask & is_positive).sum()
            neg_count = (mask & is_negative).sum()

            pos_ratio = pos_count / total_count
            pos_count_neg_count_ratio = pos_count / neg_count if neg_count > 0 else np.nan

            # 直接对满足 mask 条件的 profit_rate_series 求均值
            # profit_rate_series[mask] 会创建一个视图或副本，但通常比创建整个 filtered_df 快
            avg_value = profit_rate_series[mask].mean() # .mean() 自动处理空 Series 情况 (返回 NaN)

        else:
            # 如果没有行满足条件
            pos_count = 0
            neg_count = 0
            pos_ratio = np.nan
            pos_count_neg_count_ratio = np.nan
            avg_value = np.nan

        results.append({
            'half_range': half_range,
            'left_percent': left_percent,
            'right_percent': right_percent,
            'filtered_count': total_count,
            'positive_count': pos_count,
            'positive_ratio': pos_ratio,
            'pos_count_neg_count_ratio': pos_count_neg_count_ratio,
            'avg_net_profit_rate_new': avg_value
        })

    result_df = pd.DataFrame(results)
    return result_df


def get_data_detail_op2(origin_good_df, half_range_list=[x / 100 for x in range(1, 51)]):
    """
    针对所有目标列（排除 'timestamp', 'net_profit_rate_new' 以及包含 'new' 的列），
    根据给定的半区间列表进行如下优化：

    1. 对于每个半区间（half_range），先预先获取列的对应百分位值阈值，
       即 left_percent = 0.5 - half_range,  right_percent = 0.5 + half_range；
    2. 对每个目标列进行向量化（基于 NumPy 数组）的比较，累计每行的命中个数，
       用 hit_counts 数组替代逐次累加；
    3. 不再复制和排序 DataFrame，而是直接根据 hit_counts 的最大值利用布尔索引计算过滤统计指标；

    参数:
      origin_good_df: 原始 DataFrame，必须包含 'net_profit_rate_new' 列。
      half_range_list: 半区间列表，默认为 0.01 ~ 0.50.

    返回:
      汇总统计结果的 DataFrame，每一行为一个半区间下对应不同 top_pct（通过 offset 控制）的统计结果，
      同时包含该 top 区间内的最小与最大命中个数。
    """
    # 筛选所有数值型列，排除不需要的列
    all_numeric_columns = origin_good_df.select_dtypes(include=np.number).columns.tolist()
    target_columns = [col for col in all_numeric_columns
                      if col not in ['timestamp', 'net_profit_rate_new'] and 'new' not in col]

    results = []

    # 预先计算每个目标列在所有需要百分比下的分位数
    # 针对每个半区间，我们需要计算两个百分位： 0.5 - half_range 和 0.5 + half_range
    # 因此先计算所有可能需要的百分位值
    needed_quantiles = sorted(set(
        [0.5 - half for half in half_range_list] + [0.5 + half for half in half_range_list]
    ))

    # 对每个目标列预先计算分位数（存入字典，方便后面查表）
    precomputed_q = {}
    for col in target_columns:
        # 这里一次性计算所有需要的分位数
        precomputed_q[col] = origin_good_df[col].quantile(needed_quantiles)

    # 遍历所有半区间
    for half_range in half_range_list:
        left_percent = 0.5 - half_range
        right_percent = 0.5 + half_range

        # 初始化命中个数（使用 NumPy 数组提高性能）
        hit_counts = np.zeros(len(origin_good_df), dtype=int)

        # 针对每个目标列，获取预先计算的下界和上界，然后判断是否命中
        for col in target_columns:
            lower_threshold = precomputed_q[col][left_percent]
            upper_threshold = precomputed_q[col][right_percent]
            col_values = origin_good_df[col].values  # 转为 numpy 数组
            # 利用向量化条件判断，并累加产生的 0-1 布尔数组
            hit_counts += ((col_values >= lower_threshold) & (col_values <= upper_threshold)).astype(int)

        # 获取 hit_count 的最大值
        max_hit_count = hit_counts.max()

        # 不再对 DataFrame 复制和排序，将 hit_counts 与原始数据行对应起来
        # 对于每个 offset（原代码中 top_pct 用 [0,1,2,3,4] 表示和 max_hit_count 的差值），
        # 利用布尔索引直接筛选满足条件的行
        for offset in [0, 1, 2, 3, 4]:
            # 筛选命中大于等于 (max - offset) 的行
            mask = hit_counts >= (max_hit_count - offset)
            filtered_count = mask.sum()
            if filtered_count > 0:
                net_profit_series = origin_good_df.loc[mask, 'net_profit_rate_new']
                pos_count = (net_profit_series > 0).sum()
                pos_ratio = pos_count / filtered_count
                avg_net_profit_rate_new = net_profit_series.mean()
                min_hit_count = hit_counts[mask].min()
                max_hit_count_group = hit_counts[mask].max()
            else:
                pos_count = 0
                pos_ratio = np.nan
                avg_net_profit_rate_new = np.nan
                min_hit_count = None
                max_hit_count_group = None

            results.append({
                'half_range': half_range,
                'top_pct': offset,  # 此处 top_pct 表示 与 max_hit_count 差距，若需要百分比，可进一步转换
                'filtered_count': filtered_count,
                'positive_count': pos_count,
                'positive_ratio': pos_ratio,
                'avg_net_profit_rate_new': avg_net_profit_rate_new,
                'min_hit_count': min_hit_count,
                'max_hit_count': max_hit_count_group
            })

    result_df = pd.DataFrame(results)
    return result_df


def auto_reduce_precision(df, verbose=True, column_list=None):
    """
    自动降低DataFrame中数值列的精度：
      - 对于浮点列，若数据范围在float16可表示范围内，则转换为float16，否则转换为float32。
      - 对于整数列，尝试使用downcast降低其位宽。
    另外，如果提供了column_list参数，则对指定列进行去重操作，使得相同的行只保留一行。

    参数:
      df: 需要降低精度的DataFrame。
      verbose: 是否打印内存占用及行数变化信息，默认为True。
      column_list: 用于判断重复行的列名列表。如果为None，则不进行去重操作。

    返回:
      转换精度和（如果指定）去重处理后的DataFrame副本。
    """
    # 创建DataFrame副本，避免直接修改原始数据
    df_new = df.copy()
    start_mem = df_new.memory_usage(deep=True).sum() / 1024 ** 2

    # 数字列精度降低处理
    for col in df_new.columns:
        col_dtype = df_new[col].dtype
        # 只检查数值型数据
        if pd.api.types.is_numeric_dtype(col_dtype):
            if pd.api.types.is_float_dtype(col_dtype):
                col_min = df_new[col].min()
                col_max = df_new[col].max()
                # 判断是否可以使用float16表示，float16的表示范围大约为[-65504, 65504]
                if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:
                    df_new[col] = df_new[col].astype(np.float16)
                else:
                    df_new[col] = df_new[col].astype(np.float32)
            elif pd.api.types.is_integer_dtype(col_dtype):
                df_new[col] = pd.to_numeric(df_new[col], downcast='integer')

    all_numeric_columns = df_new.select_dtypes(include=np.number).columns.tolist()
    target_columns = [col for col in all_numeric_columns
                      if col not in ['timestamp', 'net_profit_rate_new'] and 'new' not in col]
    column_list = target_columns
    # 如果指定了列，则进行行去重操作：相同列值的行只保留第一行
    if column_list is not None:
        # 检查传入的列是否都存在于DataFrame中
        for col in column_list:
            if col not in df_new.columns:
                raise ValueError(f"列 {col} 不存在于DataFrame中")
        df_new = df_new.drop_duplicates(subset=column_list, keep='first')

    end_mem = df_new.memory_usage(deep=True).sum() / 1024 ** 2

    if verbose:
        print(f"内存占用从 {start_mem:.2f} MB 降低到 {end_mem:.2f} MB")
        if column_list is not None:
            original_rows = df.shape[0]
            new_rows = df_new.shape[0]
            row_reduction = 100 * (original_rows - new_rows) / original_rows
            print(f"经由指定的列进行去重后，行数从 {original_rows} 行减少到 {new_rows} 行，减少 {row_reduction:.1f}%")

    return df_new


def add_column(inst_id):
    # 定义目录
    corr_dir = "temp/corr"
    back_dir = "temp_back"

    # 遍历 temp/corr 目录，找到文件名中包含 inst_id 且包含 "feature" 的文件
    files = os.listdir(corr_dir)
    filter_files = [filename for filename in files if inst_id in filename and "feature" in filename]
    print(f"找到 {inst_id} 的文件个数: {len(filter_files)}")

    # 定义需要使用的列
    merge_columns = ["kai_column", "pin_column"]
    target_columns = [
        "kai_count", "hold_time_mean", "net_profit_rate",
        "fix_profit", "avg_profit_rate", "same_count",
        "weekly_net_profit_detail"
    ]
    columns_to_read = merge_columns + target_columns

    # 构造读取的新数据文件路径
    source_filename = f"{inst_id}_all_df.parquet_1m_15000_{inst_id}-USDT-SWAP_2025-04-20.csvstatistic_results_final.parquet"
    source_filepath = os.path.join(back_dir, source_filename)

    # 读取指定列
    new_df = pd.read_parquet(source_filepath, columns=columns_to_read)

    # 预先生成新的 DataFrame：复制、重命名 target_columns 字段，并设置索引
    rename_dict = {col: f"{col}_new20" for col in target_columns}
    new_df_renamed = new_df.rename(columns=rename_dict).set_index(merge_columns)

    # 遍历每个文件，读取、合并后重新写回
    for file in filter_files:
        print(f"正在处理文件: {file}")
        corr_filepath = os.path.join(corr_dir, file)
        # 读取文件
        origin_good_df = pd.read_parquet(corr_filepath)
        # 删除包含_new20的列
        columns_to_drop = [col for col in origin_good_df.columns if "_new20" in col]
        origin_good_df = origin_good_df.drop(columns=columns_to_drop, errors='ignore')
        # 设置 merge_columns 为索引，加快 join 操作
        origin_good_df = origin_good_df.set_index(merge_columns)
        # 左连接合并数据，然后重置索引
        merged_df = origin_good_df.join(new_df_renamed, how="left").reset_index()
        # 将合并后的数据写回去，采用 snappy 压缩
        merged_df.to_parquet(corr_filepath, index=False, compression="snappy")

def debug():
    # good_df = pd.read_csv('temp/final_good.csv')

    # origin_good_df_list = []
    # inst_id_list = ['BTC', 'ETH', 'SOL']
    # for inst_id in inst_id_list:
    #     origin_good_df = pd.read_csv(f'temp_back/{inst_id}_origin_good_op_all_true.csv')
    #     # origin_good_df = origin_good_df[(origin_good_df['avg_profit_rate'] > 100)]
    #     # origin_good_df = origin_good_df[(origin_good_df['net_profit_rate'] > 80)]
    #     # origin_good_df = choose_good_strategy_debug(inst_id)
    #     # origin_good_df.to_csv(f'temp/{inst_id}_df.csv', index=False)
    #     origin_good_df_list.append(origin_good_df)
    # # all_df = pd.concat(origin_good_df_list)
    # # all_df = pd.read_csv('temp/all.csv')
    # merged_df, temp_df = merge_dataframes(origin_good_df_list)
    # merged_df.to_csv('temp/merged_df.csv', index=False)
    # temp_df.to_csv('temp/temp_df.csv', index=False)
    # origin_good_df = pd.read_csv('temp/temp.csv')
    # sort_key = gen_score(origin_good_df, 'kai_count')

    # debug
    statistic_df_list = []
    range_key = 'kai_count'
    sort_key = 'net_profit_rate'
    sort_key = 'hold_time_score2'
    # sort_key = 'hold_time_score1'
    # sort_key = 'true_profit_std_score'

    # sort_key = 'stability_score'
    # sort_key = 'profit_risk_score'
    # sort_key = 'monthly_net_profit_min'
    # sort_key = 'monthly_net_profit_std_score'
    # sort_key = 'profit_risk_score'
    # sort_key = 'profit_risk_score_con'
    range_size = 10
    # sort_key = 'max_consecutive_loss_score'
    # origin_good_df = choose_good_strategy_debug('')
    inst_id_list =  ['BTC','ETH', 'SOL', 'TON', 'DOGE', 'XRP', 'PEPE']
    for inst_id in inst_id_list:
        # df = pd.read_csv('temp/origin_good_reverse.csv')
        # print()
        # gen_search_param(inst_id)
        # origin_good_df = pd.read_csv(f'temp/{inst_id}_final_good.csv')
        # origin_good_df = pd.read_csv(f'temp/{inst_id}_df.csv')
        # origin_good_df = origin_good_df[(origin_good_df['monthly_net_profit_min'] > 0)]
        # origin_good_df['hold_time_score'] = origin_good_df['hold_time_std'] / origin_good_df['hold_time_mean']
        # origin_good_df['loss_score'] = 1 - origin_good_df['loss_rate'] - origin_good_df['loss_time_rate']
        # origin_good_df = origin_good_df[(origin_good_df['loss_score'] > 0)]
        # # good_df = pd.read_csv('temp/final_good.csv')

        # filtered_df = pd.read_parquet(f'temp/{inst_id}_target.parquet')
        # pairs = list(filtered_df[['kai_column', 'pin_column']].itertuples(index=False, name=None))

        add_column(inst_id)
        # origin_good_df = choose_good_strategy_detail(inst_id)
        # origin_good_df.to_parquet(f'temp/{inst_id}_target.parquet', index=False, compression='snappy')
        continue


        origin_good_df = choose_good_strategy_debug(inst_id)
        # origin_good_df = origin_good_df.drop_duplicates(subset=["kai_column", "pin_column"])
        # origin_good_df = calculate_final_score(origin_good_df)
        # origin_good_df.to_parquet(f'temp_back/{inst_id}_origin_good_op_all_false.parquet', index=False, compression='snappy')
        # continue
        origin_good_df = pd.read_parquet(f'temp_back/{inst_id}_origin_good_op_all_false.parquet')
        # origin_good_df = origin_good_df[(origin_good_df['kai_count'] > 50)]
        # origin_good_df = origin_good_df[(origin_good_df['max_consecutive_loss'] > -20)]


        # # 计算每一列的内存使用（单位：字节），并转换成 MB（1 MB = 1024 * 1024 字节）
        # memory_usage_bytes = origin_good_df.memory_usage(deep=True, index=False)
        # memory_usage_mb = memory_usage_bytes / (1024 * 1024)
        #
        # # 将结果格式化输出
        # memory_df = pd.DataFrame({
        #     'Column': memory_usage_mb.index,
        #     'Memory Usage (MB)': memory_usage_mb.values
        # })
        # print(memory_df)
        #
        #
        #
        #
        # origin_good_df = pd.read_parquet(f'temp/final_good_{inst_id}_false.parquet')

        # kai_column_list = origin_good_df['kai_column'].to_list()
        # pin_column_list = origin_good_df['pin_column'].to_list()
        # key_kai_column_list = []
        # for kai_column in kai_column_list:
        #     key = kai_column
        #     key_kai_column_list.append(key)
        # key_pin_column_list = []
        # for pin_column in pin_column_list:
        #     key = pin_column
        #     key_pin_column_list.append(key)
        # # 统计key的数量，从大到小排序
        # kai_column_value = pd.Series(key_kai_column_list).value_counts()
        # pin_column_value = pd.Series(key_pin_column_list).value_counts()
        #



        # # 删除score_score和score_score1列
        # origin_good_df['hold_time_score'] = origin_good_df['hold_time_mean'] / origin_good_df['hold_time_std'] / origin_good_df['hold_time_std']
        # origin_good_df['monthly_net_profit_std_score1'] = -origin_good_df['monthly_net_profit_std'] * origin_good_df['monthly_net_profit_std'] / origin_good_df['net_profit_rate']
        #
        # # origin_good_df = origin_good_df[origin_good_df['kai_side'] == 'short']
        # origin_good_df = origin_good_df[(origin_good_df['kai_count_new'] > 0)]
        # if 'score_score' in origin_good_df.columns and 'score_score1' in origin_good_df.columns:
        #     origin_good_df = origin_good_df.drop(columns=['score_score', 'score_score1','kai_count', 'trade_rate','cost_rate', 'score_score1'])
        # # origin_good_df = origin_good_df.drop(
        # #     columns=['kai_count', 'trade_rate', 'cost_rate'])
        #
        # result_df = get_data_detail_op(origin_good_df)
        # result_df2 = get_data_detail_op2(origin_good_df)
        # print()
        # continue

        # 定义需要的列
        merge_columns = ["kai_column", "pin_column"]
        target_columns = [
            "kai_count", "hold_time_mean", "net_profit_rate",
            "fix_profit", "avg_profit_rate", "same_count",
            "weekly_net_profit_detail"
        ]
        columns_to_read = merge_columns + target_columns
        # 读取 Parquet 文件时仅读取需要的列，指定 engine 可根据实际情况选择
        new_df = pd.read_parquet(
            f'temp_back/{inst_id}_origin_good_op_all_false.parquet_1m_140000_{inst_id}-USDT-SWAP_2025-04-10.csvstatistic_results_final.parquet',
            columns=columns_to_read,
            engine='pyarrow'
        )

        # 选择需要的列，并重命名以区分
        new_df_selected = new_df.copy()
        new_df_selected = new_df_selected.rename(
            columns={col: col + "_new20" for col in target_columns}
        )

        # 如果数据量较大，考虑使用索引加速合并操作
        # 为 origin_good_df 和 new_df_selected 设置 merge_columns 为索引
        origin_good_df = origin_good_df.set_index(merge_columns)
        new_df_selected = new_df_selected.set_index(merge_columns)

        # 使用 join 进行合并（左连接），然后重置索引
        origin_good_df = origin_good_df.join(new_df_selected, how="left").reset_index()
        origin_good_df = get_statistic(origin_good_df)
        origin_good_df.to_parquet(f'temp/corr/{inst_id}_false_double.parquet', index=False, compression='snappy')
        continue

        origin_good_df['score_score'] = origin_good_df['loss_rate'] / origin_good_df['top_profit_ratio']
        origin_good_df['score_score1'] = origin_good_df['loss_time_rate'] * origin_good_df['monthly_net_profit_max']

        temp_df = get_data(origin_good_df)

        temp_df = pd.read_csv('temp/temp_df.csv')
        # 只保留 origin_good_df 中那些 (kai_column, pin_column) 组合在 temp_df 中存在的行
        origin_good_df = origin_good_df.merge(temp_df[['kai_column', 'pin_column']], on=['kai_column', 'pin_column'],how='inner')

        origin_good_df = origin_good_df[(origin_good_df['net_profit_rate'] > 100)]
        origin_good_df['hold_time_score'] = origin_good_df['hold_time_mean'] / origin_good_df['hold_time_std'] / origin_good_df['hold_time_std']
        origin_good_df['hold_time_score1'] = origin_good_df['net_profit_rate'] / origin_good_df['hold_time_std'] / origin_good_df['kai_count']
        origin_good_df['hold_time_score2'] = origin_good_df['net_profit_rate'] * origin_good_df['net_profit_rate']  * origin_good_df['net_profit_rate'] / origin_good_df['hold_time_std'] / origin_good_df['kai_count']
        origin_good_df['hold_time_score3'] = origin_good_df['net_profit_rate'] / (np.log1p(origin_good_df['hold_time_std']) + 1)
        origin_good_df['true_profit_std_score'] =  origin_good_df['net_profit_rate'] * origin_good_df['net_profit_rate'] / origin_good_df['true_profit_std'] / origin_good_df['kai_count']

        origin_good_df['max_consecutive_loss_score'] = origin_good_df['max_consecutive_loss'] / origin_good_df['max_loss_trade_count']
        origin_good_df['max_profit_score'] = -origin_good_df['max_profit'] / origin_good_df['net_profit_rate']
        origin_good_df['monthly_trade_std_score'] = -origin_good_df['monthly_trade_std'] / origin_good_df['kai_count']
        origin_good_df['monthly_net_profit_std_score'] = -origin_good_df['monthly_net_profit_std'] * origin_good_df['monthly_net_profit_std'] / origin_good_df['net_profit_rate'] * 22
        origin_good_df['monthly_avg_profit_std_score'] = -origin_good_df['monthly_avg_profit_std'] * origin_good_df['monthly_avg_profit_std'] / origin_good_df['avg_profit_rate'] * 100 * 100

        # origin_good_df = origin_good_df[(origin_good_df['monthly_net_profit_std_score'] > -3)]
        # origin_good_df = origin_good_df[(origin_good_df['max_consecutive_loss_score'] > -1)]
        # origin_good_df = origin_good_df[(origin_good_df['monthly_avg_profit_std_score'] > -100)]
        origin_good_df = origin_good_df[(origin_good_df['net_profit_rate'] > 100)]
        origin_good_df = origin_good_df[(origin_good_df['avg_profit_rate'] > 100)]
        # origin_good_df = origin_good_df[(origin_good_df['true_same_count_rate'] < 1)]


        # origin_good_df.to_csv(f'temp/final_good.csv', index=False)




        # origin_good_df = origin_good_df[(origin_good_df['avg_profit_rate'] > 20)]
        # origin_good_df = origin_good_df[(origin_good_df['monthly_net_profit_std_score'] > 0)]

        # origin_good_df['net_profit_rate'] = origin_good_df['net_profit_rate'] - origin_good_df['fix_profit']
        #
        # origin_good_df['profit_risk_score_con'] = -origin_good_df['net_profit_rate'] / origin_good_df['max_consecutive_loss'] * origin_good_df['net_profit_rate']
        # origin_good_df['profit_risk_score'] = -origin_good_df['net_profit_rate'] / origin_good_df['fu_profit_sum'] * origin_good_df['net_profit_rate']
        # origin_good_df['profit_risk_score_pure'] = -origin_good_df['net_profit_rate'] / origin_good_df['fu_profit_sum']
        #
        # origin_good_df = origin_good_df[(origin_good_df['profit_risk_score_pure'] > 0.5)]
        #
        # origin_good_df = origin_good_df[(origin_good_df['net_profit_rate'] > 50)]
        # origin_good_df = origin_good_df[(origin_good_df['max_consecutive_loss'] > -50)]
        # origin_good_df.to_csv(f'temp/{inst_id}_origin_good_op_all_filter.csv', index=False)

        # origin_good_df = pd.read_csv(f'temp/{inst_id}_origin_good_op_true_close.csv')
        # # origin_good_df = pd.read_csv(f'temp/{inst_id}_origin_good_op_false.csv')
        # # origin_good_df = pd.concat([origin_good_df, origin_good_df1])
        # # origin_good_df[sort_key] = -origin_good_df[sort_key]
        # # 删除kai_column和pin_column中包含 macross的行
        # # origin_good_df = origin_good_df[~origin_good_df['kai_column'].str.contains('abs') & ~origin_good_df['pin_column'].str.contains('macross')]
        # # origin_good_df['zhen_fu_mean_score'] = -origin_good_df['zhen_profit_mean'] / origin_good_df['fu_profit_mean']
        # origin_good_df['monthly_trade_std_score'] = origin_good_df['monthly_trade_std'] / origin_good_df['kai_count'] * 22 * origin_good_df['active_month_ratio']
        # # origin_good_df = origin_good_df[(origin_good_df['monthly_trade_std_score'] < 0.3)]
        # # origin_good_df = origin_good_df[(origin_good_df['profit_risk_score'] > 700)]
        # origin_good_df = origin_good_df[(origin_good_df['hold_time_mean'] < 3000)]
        # # origin_good_df = origin_good_df[(origin_good_df['hold_time_std'] < origin_good_df['hold_time_mean'])]
        # # origin_good_df = origin_good_df[(origin_good_df['max_consecutive_loss'] > -10)]
        # # origin_good_df = origin_good_df[(origin_good_df['stability_score'] > 0)]
        # # origin_good_df = origin_good_df[(origin_good_df['avg_profit_rate'] > 100)]
        # origin_good_df = origin_good_df[(origin_good_df['net_profit_rate'] > 200)]
        # # good_df = pd.read_csv('temp/final_good.csv')
        #
        #
        # kai_column和pin_column相同的时候取第一行
        origin_good_df = origin_good_df.drop_duplicates(subset=['kai_column', 'pin_column'], keep='first')
        good_df = origin_good_df.sort_values(sort_key, ascending=False)
        long_good_strategy_df = good_df[good_df['kai_side'] == 'long']
        short_good_strategy_df = good_df[good_df['kai_side'] == 'short']

        # 将long_good_strategy_df按照net_profit_rate_mult降序排列
        long_good_select_df = select_best_rows_in_ranges(long_good_strategy_df, range_size=range_size,
                                                         sort_key=sort_key, range_key=range_key)
        short_good_select_df = select_best_rows_in_ranges(short_good_strategy_df, range_size=range_size,
                                                          sort_key=sort_key, range_key=range_key)
        good_df = pd.concat([long_good_select_df, short_good_select_df])
        #
        #
        # good_df = origin_good_df.sort_values(sort_key, ascending=False)
        # # 重置good_df的索引
        # good_df = good_df.reset_index(drop=True)
        # # good_df = good_df.sort_values(by=sort_key, ascending=True)
        # # good_df = good_df.drop_duplicates(subset=['kai_column', 'kai_side'], keep='first')
        #
        # good_df = pd.read_csv(f'temp/1m_10000000_SOL_is_detail_True_is_reverse_False_3126_0.csv')
        # result, good_df, df = find_all_valid_groups(good_df, 30)
        # good_df.to_csv('temp/final_good.csv', index=False)
        # get_metrics_df(good_df)
        # return

        # good_df = pd.read_csv('temp/final_good.csv')
        # good_df = origin_good_df
        good_df = good_df.sort_values(sort_key, ascending=False)
        # 重置good_df的索引
        good_df = good_df.reset_index(drop=True)
        good_df['index'] = good_df.index
        good_index_lists = [
            [0,1,2,3]

        ]

        good_index_list = []
        for good_index in good_index_lists:
            good_index_list.extend(good_index)
        for i in range(300000):
            good_index_list.append(i)
        good_df = good_df[good_df['index'].isin( good_index_list  )]

        is_filter = True
        is_detail = False
        file_list = []
        file_list.append(f'kline_data/origin_data_1m_110000_{inst_id}-USDT-SWAP.csv')
        file_list.append(f'kline_data/origin_data_1m_40000_{inst_id}-USDT-SWAP.csv')
        file_list.append(f'kline_data/origin_data_1m_30000_{inst_id}-USDT-SWAP.csv')
        # # file_list.append(f'kline_data/origin_data_1m_20000_{inst_id}-USDT-SWAP.csv')
        file_list.append(f'kline_data/origin_data_1m_4000_{inst_id}-USDT-SWAP.csv')
        file_list.append(f'kline_data/origin_data_1m_3000_{inst_id}-USDT-SWAP.csv')
        # file_list.append(f'kline_data/origin_data_1m_2000_{inst_id}-USDT-SWAP.csv')
        file_list.append(f'kline_data/origin_data_1m_1000_{inst_id}-USDT-SWAP.csv')
        good_df_list = []

        for file in file_list:
            df = pd.read_csv(file)
            # 获取第一行和最后一行的close，计算涨跌幅
            start_close = df.iloc[0]['close']
            end_close = df.iloc[-1]['close']
            total_chg = (end_close - start_close) / start_close * 100

            # 计算每一行的涨跌幅
            df['chg'] = df['close'].pct_change() * 100
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            signal_cache = {}
            statistic_dict_list = []
            # good_df = good_df.reset_index(drop=True)
            start_time = time.time()
            for index, row in good_df.iterrows():
                long_column = row['kai_column']
                short_column = row['pin_column']
                if 'is_reverse' not in row:
                    is_reverse = False
                else:
                    is_reverse = row['is_reverse']
                # long_column = 'ma_1_low_short'
                # short_column = 'abs_1920_0.5_low_short'
                kai_data_df, statistic_dict = get_detail_backtest_result_op(signal_cache, df, long_column, short_column,
                                                                            is_filter, is_detail, is_reverse)
                if statistic_dict is None:
                    continue
                # 为每一行添加统计数据，需要修改到原始数据中
                # 直接修改 `good_df` 中的相应列
                good_df.at[index, 'kai_count_new'] = statistic_dict['kai_count']
                good_df.at[index, 'trade_rate_new'] = statistic_dict['trade_rate']
                good_df.at[index, 'hold_time_mean_new'] = statistic_dict['hold_time_mean']
                good_df.at[index, 'net_profit_rate_new'] = statistic_dict['net_profit_rate'] - statistic_dict['fix_profit']
                good_df.at[index, 'fix_profit_new'] = statistic_dict['fix_profit']
                good_df.at[index, 'avg_profit_rate_new'] = statistic_dict['avg_profit_rate']
                good_df.at[index, 'same_count_new'] = statistic_dict['same_count']
                # good_df.at[index, 'max_profit_new'] = statistic_dict['max_profit']
                # good_df.at[index, 'min_profit_new'] = statistic_dict['min_profit']
                if is_detail:
                    good_df.at[index, 'max_optimal_value'] = statistic_dict['max_optimal_value']
                    good_df.at[index, 'max_optimal_profit'] = statistic_dict['max_optimal_profit']
                    good_df.at[index, 'max_optimal_loss_rate'] = statistic_dict['max_optimal_loss_rate']
                    good_df.at[index, 'min_optimal_value'] = statistic_dict['min_optimal_value']
                    good_df.at[index, 'min_optimal_profit'] = statistic_dict['min_optimal_profit']
                    good_df.at[index, 'min_optimal_loss_rate'] = statistic_dict['min_optimal_loss_rate']

                statistic_dict_list.append(statistic_dict)
            if is_detail:
                good_df['max_optimal_profit_cha'] = good_df['max_optimal_profit'] - good_df['net_profit_rate_new']
                good_df['max_optimal_profit_rate'] = good_df['max_optimal_profit_cha'] / good_df['net_profit_rate_new']
                good_df['min_optimal_profit_cha'] = good_df['min_optimal_profit'] - good_df['net_profit_rate_new']
                good_df['min_optimal_profit_rate'] = good_df['min_optimal_profit_cha'] / good_df['net_profit_rate_new']
            statistic_df = pd.DataFrame(statistic_dict_list)
            statistic_df_list.append(statistic_df)
            statistic_df.to_csv('temp/all_statistic_df.csv', index=False)
            print(f'耗时 {time.time() - start_time:.2f} 秒。')
            # 获取good_df的kai_column的分布情况
            kai_value = good_df['kai_column'].value_counts()
            pin_value = good_df['pin_column'].value_counts()
            # 为good_df新增两列，分别为kai_column的分布情况和pin_column的分布情况
            good_df['kai_value'] = good_df['kai_column'].apply(lambda x: kai_value[x])
            good_df['pin_value'] = good_df['pin_column'].apply(lambda x: pin_value[x])
            good_df['value_score'] = good_df['kai_value'] + good_df['pin_value']
            good_df['value_score1'] = good_df['kai_value'] * good_df['pin_value']
            good_df['total_chg'] = total_chg
            good_df_list.append(good_df.copy())
            # 获取索引为109，876，926的行
            # row_list = [303, 4144, 3949]
            # 找到good_df中score字段值在row_list中的行
            # good_df[good_df['index'].isin(     [24, 445, 528]        )]
            # good_df_list[0][good_df_list[0]['index'].isin([339, 990])]
            # result = find_all_valid_groups(good_df, 100)
            # good_df.loc[  [1073, 1874]   ]
            good_temp_df_list = []
            for good_index in good_index_lists:
                good_temp_df_list.append(good_df[good_df['index'].isin(  good_index  )])



            print(inst_id)
    merged_df, temp_df = merge_dataframes(statistic_df_list)
    print()


def example():
    debug()
    # get_data()


if __name__ == '__main__':
    example()
