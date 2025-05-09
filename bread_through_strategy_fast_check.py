#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
突破策略信号生成及回测代码 —— 直接获取所有组合的回测结果版

说明：
  1. 原有遗传算法部分已被替换。现改为直接对所有候选信号（长信号和短信号）的组合进行回测，
     并采用每一定数量（如 100,000 个组合为一批次）的方式处理（批次内并行计算）。
  2. 计算有效组合后，将回测结果保存至本地文件，文件名包含批次号以便后续汇总和分析。
"""

import os
import random
import sys
import time
import pickle
import traceback
import itertools
import multiprocessing

import numpy as np
import pandas as pd
from numba import njit
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import chain

import warnings
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message=".*SettingWithCopyWarning.*",
    category=pd.errors.SettingWithCopyWarning
)

# 全局参数及变量
IS_REVERSE = False  # 是否反向操作
checkpoint_dir = 'temp'
GLOBAL_SIGNALS = {}
df = None  # 回测数据，在子进程中通过初始化传入

pd.options.mode.chained_assignment = None
def series_to_numpy(series):
    """将 Pandas Series 转为 NumPy 数组。"""
    return series.to_numpy(copy=False) if hasattr(series, "to_numpy") else np.asarray(series)

def compute_signal(df, col_name):
    """
    根据历史行情数据(df)和指定信号名称(col_name)生成交易信号和目标价格。
    支持的信号类型：abs, relate, donchian, boll, macross, rsi, macd, cci, atr。
    """
    parts = col_name.split("_")
    signal_type = parts[0]
    direction = parts[-1]

    if signal_type == "abs":
        period = int(parts[1])
        abs_value = float(parts[2]) / 100
        if direction == "long":
            min_low = df["low"].shift(1).rolling(period).min()
            target_price = (min_low * (1 + abs_value)).round(4)
            signal_series = df["high"] > target_price
        else:
            max_high = df["high"].shift(1).rolling(period).max()
            target_price = (max_high * (1 - abs_value)).round(4)
            signal_series = df["low"] < target_price
        valid_trade = (target_price >= df["low"]) & (target_price <= df["high"])
        signal_series = signal_series & valid_trade
        return signal_series, target_price

    elif signal_type == "relate":
        period = int(parts[1])
        percent = float(parts[2]) / 100
        min_low = df["low"].shift(1).rolling(period).min()
        max_high = df["high"].shift(1).rolling(period).max()
        if direction == "long":
            target_price = (min_low + percent * (max_high - min_low)).round(4)
            signal_series = df["high"] > target_price
        else:
            target_price = (max_high - percent * (max_high - min_low)).round(4)
            signal_series = df["low"] < target_price
        valid_trade = (target_price >= df["low"]) & (target_price <= df["high"])
        return signal_series & valid_trade, target_price

    elif signal_type == "donchian":
        period = int(parts[1])
        if direction == "long":
            highest_high = df["high"].shift(1).rolling(period).max()
            signal_series = df["high"] > highest_high
            target_price = highest_high
        else:
            lowest_low = df["low"].shift(1).rolling(period).min()
            signal_series = df["low"] < lowest_low
            target_price = lowest_low
        valid_trade = (target_price >= df["low"]) & (target_price <= df["high"])
        signal_series = signal_series & valid_trade
        return signal_series, target_price.round(4)

    elif signal_type == "boll":
        period = int(parts[1])
        std_multiplier = float(parts[2])
        ma = df["close"].rolling(window=period, min_periods=period).mean()
        std_dev = df["close"].rolling(window=period, min_periods=period).std()
        upper_band = (ma + std_multiplier * std_dev).round(4)
        lower_band = (ma - std_multiplier * std_dev).round(4)
        if direction == "long":
            signal_series = (df["close"].shift(1) < lower_band.shift(1)) & (df["close"] >= lower_band)
        else:
            signal_series = (df["close"].shift(1) > upper_band.shift(1)) & (df["close"] <= upper_band)
        return signal_series, df["close"]

    elif signal_type == "macross":
        fast_period = int(parts[1])
        slow_period = int(parts[2])
        fast_ma = df["close"].rolling(window=fast_period, min_periods=fast_period).mean().round(4)
        slow_ma = df["close"].rolling(window=slow_period, min_periods=slow_period).mean().round(4)
        if direction == "long":
            signal_series = (fast_ma.shift(1) < slow_ma.shift(1)) & (fast_ma >= slow_ma)
        else:
            signal_series = (fast_ma.shift(1) > slow_ma.shift(1)) & (fast_ma <= slow_ma)
        return signal_series, df["close"]

    elif signal_type == "rsi":
        period = int(parts[1])
        overbought = int(parts[2])
        oversold = int(parts[3])
        delta = df["close"].diff(1).astype(np.float32)
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        if direction == "long":
            signal_series = (rsi.shift(1) < oversold) & (rsi >= oversold)
        else:
            signal_series = (rsi.shift(1) > overbought) & (rsi <= overbought)
        return signal_series, df["close"]

    elif signal_type == "macd":
        fast_period, slow_period, signal_period = map(int, parts[1:4])
        fast_ema = df["close"].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df["close"].ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        if direction == "long":
            signal_series = (macd_line.shift(1) < signal_line.shift(1)) & (macd_line >= signal_line)
        else:
            signal_series = (macd_line.shift(1) > signal_line.shift(1)) & (macd_line <= signal_line)
        return signal_series, df["close"]

    elif signal_type == "cci":
        period = int(parts[1])
        tp = (df["high"] + df["low"] + df["close"]) / 3
        ma = tp.rolling(period).mean()
        md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        cci = (tp - ma) / (0.015 * md)
        if direction == "long":
            signal_series = (cci.shift(1) < -100) & (cci >= -100)
        else:
            signal_series = (cci.shift(1) > 100) & (cci <= 100)
        return signal_series, df["close"]

    elif signal_type == "atr":
        period = int(parts[1])
        tr = pd.concat([
            df["high"] - df["low"],
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        atr_ma = atr.rolling(period).mean()
        if direction == "long":
            signal_series = (atr.shift(1) < atr_ma.shift(1)) & (atr >= atr_ma)
        else:
            signal_series = (atr.shift(1) > atr_ma.shift(1)) & (atr <= atr_ma)
        return signal_series, df["close"]

    else:
        raise ValueError(f"未知信号类型: {signal_type}")


def op_signal(df, sig):
    """
    计算单个信号，并筛选出交易次数大于 100 的稀疏信号。
    """
    s, p = compute_signal(df, sig)
    s_np = series_to_numpy(s)
    p_np = series_to_numpy(p)
    if p_np.dtype == np.float64:
        p_np = p_np.astype(np.float32)
    indices = np.nonzero(s_np)[0]
    # if indices.size < 100:
    #     return None
    # # 判断最大idx和最小idx的差值是否大于0.1 * len(df)
    # if len(df) > 0:
    #     max_idx = indices.max()
    #     min_idx = indices.min()
    #     if max_idx - min_idx < 0.1 * len(df):
    #         return None
    return (indices.astype(np.int32), p_np[indices])


@njit(cache=True, fastmath=True)
def fast_check(k_idx, k_price, p_idx, p_price, min_trades=10, loss_th=-30.0, is_reverse=False):
    """
    使用单持仓模式计算交易组合的盈亏情况：
      - 在持仓未平仓前，后续的开仓信号将被忽略；
      - k_idx, k_price 为开仓信号对应的索引和价格；
      - p_idx, p_price 为平仓信号对应的索引和价格。
    如果匹配过程中的最小累计收益（连续亏损）低于 loss_th，则提前返回 False，
    最后判断是否满足最少 min_trades 笔交易要求。
    """
    i = 0  # 开仓信号的指针
    j = 0  # 平仓信号的指针
    n_k = k_idx.shape[0]
    n_p = p_idx.shape[0]

    # # 如果任一信号触发次数少于最小交易次数，直接返回 False
    # if n_k < min_trades or n_p < min_trades:
    #     return False

    trades = 0
    cur_sum = 0.0
    min_sum = 0.0
    trade_count = 0

    # 记录上一次平仓的时刻，未平仓前不允许新开仓
    last_closed_time = -1  # 假设所有索引均 >= 0

    while i < n_k and j < n_p:
        # 如果当前开仓信号在上一次平仓之前，说明此信号处于上一笔仓位未平仓期间，跳过
        if k_idx[i] <= last_closed_time:
            i += 1
            continue

        # 找到第一个有效平仓信号，其时间必须晚于当前开仓信号
        while j < n_p and p_idx[j] <= k_idx[i]:
            j += 1
        if j >= n_p:
            break

        # 执行一笔交易：开仓在 k_idx[i]，平仓在 p_idx[j]
        chg = (p_price[j] - k_price[i]) / k_price[i] * 100.0
        if not is_reverse:
            chg = - chg
        pnl = chg - 0.07  # 计算收益率并扣除手续费
        trades += 1

        # 累计连续亏损计算逻辑
        if cur_sum == 0:
            trade_count = 0
        cur_sum += pnl
        trade_count += 1
        if cur_sum < min_sum:
            min_sum = cur_sum
        if cur_sum > 0:
            cur_sum = 0.0
            trade_count = 0

        # 若累计亏损超过允许阈值，则提前退出
        if min_sum < loss_th:
            return False

        # 更新上一次平仓时刻，之后开仓信号必须晚于此时刻才能开新仓
        last_closed_time = p_idx[j]

        # 移动指针，进行下一笔交易的匹配
        i += 1
        j += 1

    return (trades >= min_trades) and (min_sum >= loss_th)


def check_max_loss(df, kai_column, pin_column, is_reverse=False):
    """
    融合了 fast_check 的版本：
      1. 首先尝试从 GLOBAL_SIGNALS 中获取 kai 与 pin 信号数据，若不存在则计算。
      2. 控制条件：若 kai_idx 与 pin_idx 重叠率（交集除以较少的触发次数） >= 1%，直接返回 False。
      3. 调用 fast_check 进行加速计算，匹配方式假设两个信号生成的数据均为时间上递增的 NumPy 数组，
         并计算累计盈亏；若最大连续亏损未低于 -30% 且交易次数不少于 100 笔，则返回 True。
    """
    try:
        kai_idx, kai_prices = GLOBAL_SIGNALS[kai_column]
        pin_idx, pin_prices = GLOBAL_SIGNALS[pin_column]
    except KeyError:
        kai_idx, kai_prices = op_signal(df, kai_column)
        pin_idx, pin_prices = op_signal(df, pin_column)

    # --- 新增筛选条件：检查 kai_idx 与 pin_idx 之间的重叠率 ---
    # 以较小信号数量为分母，确保对少量信号敏感
    common = np.intersect1d(kai_idx, pin_idx)
    overlap_ratio = common.size / min(kai_idx.size, pin_idx.size)
    if overlap_ratio >= 0.01:
        return False
    side = True
    if 'short' in kai_column:
        side = False
    if is_reverse:
        side = not side
    return fast_check(kai_idx, kai_prices, pin_idx, pin_prices, is_reverse=side)

def generate_numbers(start, end, number, even=True):
    """
    生成区间内均匀或非均匀分布的一组整数。
    """
    if start > end or number <= 0 or number == 1:
        return []
    if even:
        step = (end - start) / (number - 1)
        result = [int(round(start + i * step)) for i in range(number)]
    else:
        result = [int(round(start + (end - start) * ((i / (number - 1)) ** 2))) for i in range(number)]
    final_result = []
    last_val = None
    for val in result:
        if start <= val <= end and val != last_val:
            final_result.append(val)
            last_val = val
    return final_result[:number]


def gen_abs_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = [x / 20 for x in range(start_period1, end_period1, step1)]
    long_columns = [f"abs_{p}_{p1}_long" for p in period_list for p1 in period_list1 if p >= p1]
    short_columns = [f"abs_{p}_{p1}_short" for p in period_list for p1 in period_list1 if p >= p1]
    key_name = f"abs_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}"
    print(f"abs 生成 {len(long_columns)} 长信号和 {len(short_columns)} 短信号。")
    return long_columns, short_columns, key_name


def gen_macd_signal_name(start_period, end_period, step):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    signal_list = [9, 12, 15, 40]
    long_columns = [f"macd_{fast}_{slow}_{signal}_long" for fast in period_list for slow in period_list if slow > fast
                    for signal in signal_list]
    short_columns = [f"macd_{fast}_{slow}_{signal}_short" for fast in period_list for slow in period_list if slow > fast
                     for signal in signal_list]
    key_name = f"macd_{start_period}_{end_period}_{step}"
    print(f"MACD 生成 {len(long_columns)} 信号。")
    return long_columns, short_columns, key_name


def gen_cci_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = [x / 10 for x in range(start_period1, end_period1, step1)]
    long_columns = [f"cci_{p}_{p1}_long" for p in period_list for p1 in period_list1 if p >= p1]
    short_columns = [f"cci_{p}_{p1}_short" for p in period_list for p1 in period_list1 if p >= p1]
    key_name = f"cci_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}"
    print(f"cci 生成 {len(long_columns)} 信号。")
    return long_columns, short_columns, key_name


def gen_relate_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = list(range(start_period1, end_period1, step1))
    long_columns = [f"relate_{p}_{p1}_long" for p in period_list for p1 in period_list1 if p >= p1]
    short_columns = [f"relate_{p}_{p1}_short" for p in period_list for p1 in period_list1 if p >= p1]
    key_name = f"relate_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}"
    print(f"relate 生成 {len(long_columns)} 信号。")
    return long_columns, short_columns, key_name


def gen_rsi_signal_name(start_period, end_period, step):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    temp_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    long_columns = [f"rsi_{p}_{ob}_{100 - ob}_long" for p in period_list for ob in temp_list]
    short_columns = [f"rsi_{p}_{ob}_{100 - ob}_short" for p in period_list for ob in temp_list]
    key_name = f"rsi_{start_period}_{end_period}_{step}"
    print(f"rsi 生成 {len(long_columns)} 信号。")
    return long_columns, short_columns, key_name


def gen_atr_signal_name(start_period, end_period, step):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    long_columns = [f"atr_{p}_long" for p in period_list]
    short_columns = [f"atr_{p}_short" for p in period_list]
    key_name = f"atr_{start_period}_{end_period}_{step}"
    print(f"atr 生成 {len(long_columns)} 信号。")
    return long_columns, short_columns, key_name


def gen_donchian_signal_name(start_period, end_period, step):
    period_list = list(range(start_period, end_period, step))
    long_columns = [f"donchian_{p}_long" for p in period_list]
    short_columns = [f"donchian_{p}_short" for p in period_list]
    key_name = f"donchian_{start_period}_{end_period}_{step}"
    print(f"donchian 生成 {len(long_columns)} 信号。")
    return long_columns, short_columns, key_name


def gen_boll_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = [x / 10 for x in range(start_period1, end_period1, step1)]
    long_columns = [f"boll_{p}_{p1}_long" for p in period_list for p1 in period_list1 if p >= p1]
    short_columns = [f"boll_{p}_{p1}_short" for p in period_list for p1 in period_list1 if p >= p1]
    key_name = f"boll_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}"
    print(f"boll 生成 {len(long_columns)} 信号。")
    return long_columns, short_columns, key_name


def gen_macross_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = generate_numbers(start_period1, end_period1, step1, even=False)
    long_columns = [f"macross_{p}_{p1}_long" for p in period_list for p1 in period_list1]
    short_columns = [f"macross_{p}_{p1}_short" for p in period_list for p1 in period_list1]
    key_name = f"macross_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}"
    print(f"macross 生成 {len(long_columns)} 信号。")
    return long_columns, short_columns, key_name


def generate_all_signals():
    """
    生成所有候选信号，目前基于 abs、relate、donchian、boll、macross、rsi、macd、cci、atr。
    """
    column_list = []
    abs_long, abs_short, abs_key = gen_abs_signal_name(1, 100, 100, 40, 100, 1)
    column_list.append((abs_long, abs_short, abs_key))
    relate_long, relate_short, relate_key = gen_relate_signal_name(400, 1000, 100, 1, 40, 6)
    column_list.append((relate_long, relate_short, relate_key))
    donchian_long, donchian_short, donchian_key = gen_donchian_signal_name(1, 20, 1)
    column_list.append((donchian_long, donchian_short, donchian_key))
    boll_long, boll_short, boll_key = gen_boll_signal_name(1, 3000, 100, 1, 50, 2)
    column_list.append((boll_long, boll_short, boll_key))
    macross_long, macross_short, macross_key = gen_macross_signal_name(1, 3000, 100, 1, 3000, 100)
    column_list.append((macross_long, macross_short, macross_key))
    rsi_long, rsi_short, rsi_key = gen_rsi_signal_name(1, 1000, 500)
    column_list.append((rsi_long, rsi_short, rsi_key))
    macd_long, macd_short, macd_key = gen_macd_signal_name(300, 1000, 50)
    column_list.append((macd_long, macd_short, macd_key))
    cci_long, cci_short, cci_key = gen_cci_signal_name(1, 2000, 1000, 1, 2, 1)
    column_list.append((cci_long, cci_short, cci_key))
    atr_long, atr_short, atr_key = gen_atr_signal_name(1, 3000, 3000)
    column_list.append((atr_long, atr_short, atr_key))
    column_list = sorted(column_list, key=lambda x: len(x[0]))
    all_signals = []
    key_name = ""
    for long_cols, short_cols, temp_key in column_list:
        temp = long_cols + short_cols
        key_name += temp_key + "_"
        all_signals.extend(temp)
    return all_signals, key_name


##############################################
# 信号预计算及多进程工具函数
##############################################
def process_signal(sig):
    """
    计算单个信号的预计算数据。若交易信号数不足 100，则返回 None。
    """
    try:
        s, p = compute_signal(df, sig)
        s_np = series_to_numpy(s)
        p_np = series_to_numpy(p)
        if p_np.dtype == np.float64:
            p_np = p_np.astype(np.float32)
        indices = np.nonzero(s_np)[0]
        if indices.size < 100:
            return None
        return (sig, (indices.astype(np.int32), p_np[indices]))
    except Exception as e:
        print(f"预计算 {sig} 时出错：{e}")
        return None


def precompute_signals(df, signals, chunk_size=100):
    """
    使用多进程预计算所有候选信号数据，每个进程一次处理 chunk_size 个任务。
    返回 dict 格式：{signal_name: (indices, prices)}。
    """
    num_workers = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_workers, initializer=init_worker1, initargs=(df,)) as pool:
        results = pool.imap(process_signal, signals, chunksize=chunk_size)

        precomputed = {}
        for res in results:
            if res is not None:
                sig, data = res
                precomputed[sig] = data

    return precomputed


def load_or_compute_precomputed_signals(df, signals, key_name):
    """
    尝试加载预计算结果，若无或加载出错则重新计算并保存。
    """
    file_path = os.path.join("temp_back", f"precomputed_signals_{key_name}_{len(signals)}.pkl")
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                precomputed = pickle.load(f)
            print(f"从 {file_path} 加载预计算结果，共 {len(precomputed)} 个信号。")
            return precomputed
        except Exception as e:
            print(f"加载失败：{e}，重新计算。")
    print("开始计算预计算信号 ...")
    precomputed = precompute_signals(df, signals)
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(precomputed, f)
        print(f"预计算结果已保存到：{file_path}")
    except Exception as e:
        print(f"保存预计算结果时出错：{e}")
    return precomputed


def init_worker1(dataframe):
    """子进程初始化函数，将 df 加载到全局变量。"""
    global df
    df = dataframe


##############################################
# 穷举回测相关函数（候选信号两两组合）
##############################################
def init_worker_brute(precomputed_signals, dataframe):
    """子进程初始化函数，将 GLOBAL_SIGNALS 和 df 加载到全局变量。"""
    global GLOBAL_SIGNALS, df
    GLOBAL_SIGNALS = precomputed_signals
    df = dataframe


def candidate_pairs_generator(long_signals, short_signals):
    """
    生成长信号与短信号所有可能的组合，采用生成器避免一次性加载所有组合到内存中。
    """
    for long_sig in long_signals:
        for short_sig in short_signals:
            yield (long_sig, short_sig)


def evaluate_candidate(candidate):
    """
    对单个候选组合进行回测，返回 statistic_dict；
    若回测条件不满足，则返回 None。
    """
    long_sig, short_sig = candidate
    check_result = check_max_loss(df, long_sig, short_sig, is_reverse=False)
    if check_result:
        return (long_sig, short_sig)
    return None


def brute_force_backtesting(df, long_signals, short_signals, batch_size=100000, key_name="brute_force_results",all_files_df=None):
    """
    穷举遍历所有 (长信号, 短信号) 组合，每 batch_size 个组合为一个批次，
    使用多进程并行计算每个候选组合的回测结果，保存每个批次的有效回测结果到文件。
    同时累积所有有效组合并在结束时返回结果列表。

    主要改进：
      1. 在整个遍历过程中只创建一次进程池，避免每个批次中频繁创建/销毁进程导致的长尾问题。
      2. 根据当前批次任务数量动态设置 chunksize，降低进程间切换的消耗。
    """
    if all_files_df is None:
        total_pairs = len(long_signals) * len(short_signals)
        predict_batch_number = total_pairs // batch_size + 1
        files = [f for f in os.listdir(checkpoint_dir) if key_name in f]
        print(f"候选组合总数: {total_pairs} 预计批次数: {total_pairs // batch_size + 1}")
        print(f"已存在 {len(files)} 个批次的回测结果")
        if len(files) >= predict_batch_number:
            print(f"已存在 {len(files)} 个批次的回测结果，跳过。")
            return

        candidate_gen = candidate_pairs_generator(long_signals, short_signals)
    else:
        # 只保留all_files_df信号在long_signals和short_signals中的组合
        all_files_df = all_files_df[all_files_df["kai_column"].isin(long_signals) & all_files_df["pin_column"].isin(short_signals)]
        # 直接从 all_files_df 中获取所有信号对
        candidate_gen = zip(all_files_df["kai_column"].to_numpy(), all_files_df["pin_column"].to_numpy())
        predict_batch_number = len(all_files_df) // batch_size + 1
        files = [f for f in os.listdir(checkpoint_dir) if key_name in f]
        print(f"候选组合总数: {len(all_files_df)} 预计批次数: {predict_batch_number}")
        print(f"已存在 {len(files)} 个批次的回测结果")
        if len(files) >= predict_batch_number:
            print(f"已存在 {len(files)} 个批次的回测结果，跳过。")
            return
    batch_index = 0
    start_time = time.time()
    pool_processes = 30
    chunk_size = max(100, batch_size // (pool_processes * 20))

    print(f"开始穷举回测，批次大小: {batch_size}，进程池大小: {pool_processes}，chunk_size: {chunk_size}。时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    # 创建持久进程池，避免多个批次中重复创建/销毁进程的开销
    pool = multiprocessing.Pool(processes=pool_processes,
                                initializer=init_worker_brute,
                                initargs=(GLOBAL_SIGNALS, df))
    print(f"进程池已创建，共 {pool._processes} 个进程。耗时 {time.time() - start_time:.2f} 秒")
    try:
        while True:
            start_time = time.time()
            batch = list(itertools.islice(candidate_gen, batch_size))
            if not batch:
                break
            print(f"开始处理批次 {batch_index}，当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            stats_file_name = os.path.join(checkpoint_dir, f"{key_name}_{batch_index}_{IS_REVERSE}_stats_debug.parquet")
            if os.path.exists(stats_file_name):
                print(f"批次 {batch_index} 的统计文件已存在，跳过。")
                batch_index += 1
                continue
            results = pool.imap_unordered(evaluate_candidate, batch, chunksize=chunk_size)
            valid_results = [res for res in results if res is not None]
            if valid_results:  # 确保 valid_results 不为空，否则创建空 DataFrame 时指定列名可能无意义或报错
                temp_df = pd.DataFrame(valid_results, columns=['kai_column', 'pin_column'])
            else:
                temp_df = pd.DataFrame(columns=['kai_column', 'pin_column'])  # 创建一个空的但有列名的DataFrame
            temp_df.to_parquet(stats_file_name, index=False)
            print(
                f"批次 {batch_index} 处理完毕，有效组合数量: {len(valid_results)} 耗时 {time.time() - start_time:.2f} 秒 当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            batch_index += 1
    finally:
        pool.close()
        pool.join()


def load_file(file_path):
    try:
        temp_df = pd.read_parquet(file_path)
        return temp_df
    except Exception as e:
        print(f"加载 {file_path} 时出错：{e}")
        return None


def load_files_in_parallel(checkpoint_dir, pre_key_name):
    all_files = []

    if pre_key_name is not None:
        files = [f for f in os.listdir(checkpoint_dir) if pre_key_name in f]
        print(f"加载 {len(files)} 个数据文件 ...")
        file_paths = [os.path.join(checkpoint_dir, file) for file in files]
        for file_path in file_paths:
            result = load_file(file_path)
            all_files.append(result)

        if all_files:
            # 合并所有 DataFrame（避免不必要的复制）
            all_files_df = pd.concat(all_files, ignore_index=True, copy=False)

            # 精准内存占用
            mem_usage_mb = all_files_df.memory_usage(deep=True).sum() / (1024 * 1024)
            print(f"all_files_df 内存占用: {mem_usage_mb:.2f} MB")

            # 更快的方式统计信号集合
            all_signals = set(chain(all_files_df["kai_column"], all_files_df["pin_column"]))
            print(f'加载 {len(all_files)} 个数据，当前信号数量: {len(all_signals)}。 信号对个数: {len(all_files_df)}')
        return all_files_df, all_signals

def brute_force_optimize_breakthrough_signal(data_path="temp/TON_1m_2000.csv"):
    """
    加载数据后直接对所有 (长信号, 短信号) 组合进行回测，
    每一定数量的组合为一个批次，保存回测结果。
    """
    os.makedirs("temp", exist_ok=True)
    pre_key_name = None
    base_name = os.path.basename(data_path).replace("-USDT-SWAP.csv", "").replace("origin_data_", "")
    base_name = base_name.split("-")[0]
    df_local = pd.read_csv(data_path)
    needed_columns = ["timestamp", "high", "low", "close"]
    df_local = df_local[needed_columns]

    # 将时间列转换为 datetime 类型
    df_local["timestamp"] = pd.to_datetime(df_local["timestamp"])
    # 过滤掉首尾月数据（避免数据不完整问题），可根据实际情况调整
    df_monthly = df_local["timestamp"].dt.to_period("Y")
    df_local = df_local[(df_monthly != df_monthly.min()) & (df_monthly != df_monthly.max())]
    # 添加年份列，按照年份分段回测
    df_local["year"] = df_local["timestamp"].dt.year
    year_list = df_local["year"].unique()
    # 生成所有候选信号
    all_signals, key_name = generate_all_signals()
    all_files_df = None
    # 对预计算使用所有信号（长短信号均在 all_signals 内）
    print(f"生成 {len(all_signals)} 候选信号。")
    for year in year_list:
        print(f"数据 {base_name} 的第一年: {year}")
        temp_df_local = df_local[df_local["year"] == year]
        temp_df_local.drop(columns=["year"], inplace=True)

        # 保证数值合理，若最低价小于1则扩大价格
        while temp_df_local["low"].min() < 1:
            temp_df_local[["high", "low", "close"]] *= 10
        jingdu = "float32"
        temp_df_local["chg"] = (temp_df_local["close"].pct_change() * 100).astype("float16")
        temp_df_local["high"] = temp_df_local["high"].astype(jingdu)
        temp_df_local["low"] = temp_df_local["low"].astype(jingdu)
        temp_df_local["close"] = temp_df_local["close"].astype(jingdu)
        temp_df_local["timestamp"] = pd.to_datetime(temp_df_local["timestamp"])
        df_monthly = temp_df_local["timestamp"].dt.to_period("M")
        temp_df_local = temp_df_local[(df_monthly != df_monthly.min()) & (df_monthly != df_monthly.max())]
        print(f"\n开始基于暴力穷举回测 {base_name} ... 数据长度 {temp_df_local.shape[0]} 时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        # temp_df_local.to_csv("kline_data/origin_data_1m_500000_BTC-USDT-SWAP_2025-05-06.csv")
        global df
        df = temp_df_local.copy()

        if pre_key_name is not None:
            all_files_df, all_signals = load_files_in_parallel(checkpoint_dir, pre_key_name)

        # 预计算所有候选信号数据
        precomputed = load_or_compute_precomputed_signals(df, all_signals, f'{year}_{base_name}_{key_name}')
        total_size = sys.getsizeof(precomputed) + sum(
            sys.getsizeof(sig) + s.nbytes + p.nbytes for sig, (s, p) in precomputed.items())
        print(f"预计算信号内存大小: {total_size / (1024 * 1024):.2f} MB 信号数量: {len(precomputed)} 总体信号个数: {len(all_signals)}")

        global GLOBAL_SIGNALS
        GLOBAL_SIGNALS = precomputed
        # 本例中长信号与短信号均取预计算结果中的所有信号
        candidate_signals = list(GLOBAL_SIGNALS.keys())
        print(f"候选信号数量: {len(candidate_signals)}。")

        # 穷举回测所有候选组合，每一批次计算并保存结果
        brute_force_backtesting(df, candidate_signals, candidate_signals, batch_size=10000000,
                                                key_name=f'{year}_{base_name}_{key_name}', all_files_df=all_files_df)
        pre_key_name = f'{year}_{base_name}_{key_name}'

def example():
    """
    示例入口：处理多个数据文件调用暴力穷举回测流程。
    """
    start_time = time.time()
    data_path_list = [
        "kline_data/origin_data_1m_5000000_BTC-USDT-SWAP_2025-05-06.csv",
        "kline_data/origin_data_1m_5000000_ETH-USDT-SWAP_2025-05-06.csv",
        "kline_data/origin_data_1m_5000000_SOL-USDT-SWAP_2025-05-06.csv",
        "kline_data/origin_data_1m_5000000_TON-USDT-SWAP_2025-05-06.csv",
        "kline_data/origin_data_1m_5000000_DOGE-USDT-SWAP_2025-05-06.csv",
        "kline_data/origin_data_1m_5000000_XRP-USDT-SWAP_2025-05-06.csv",
        "kline_data/origin_data_1m_5000000_OKB-USDT_2025-05-06.csv",
    ]
    for data_path in data_path_list:
        try:
            brute_force_optimize_breakthrough_signal(data_path)
            print(f"{data_path} 总耗时 {time.time() - start_time:.2f} 秒。")
        except Exception as e:
            traceback.print_exc()
            print(f"处理 {data_path} 出错：{e}")


if __name__ == "__main__":
    example()