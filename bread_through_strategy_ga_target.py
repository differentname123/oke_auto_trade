#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
说明：
  1. 输入一个 stat_df 文件，提取其中的 kai_column 与 pin_column 列，
     得到需要计算回测结果的信号对列表。
  2. 根据所有涉及到的信号，从行情数据中提前计算信号数据，并保存到 temp_back 目录下。
  3. 利用多进程分批次计算各信号对的回测结果。
  4. 最后将所有信号对的统计指标（statistic_dict）合并成一个 DataFrame，
     并保存到 temp_back 目录下。

使用方法：
  python script.py <stat_df_file.csv> <market_data_file.csv>
"""
import itertools
import math
import os
import sys
import time
import pickle
import multiprocessing
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import random
from numba import njit

# 全局变量，存储预计算信号数据及行情数据（df）
GLOBAL_SIGNALS = {}
df = None


##############################################
# 辅助函数
##############################################

def series_to_numpy(series):
    """
    将 Pandas Series 转为 NumPy 数组。
    """
    return series.to_numpy(copy=False) if hasattr(series, "to_numpy") else np.asarray(series)


def safe_round(value, ndigits=4):
    """
    数值四舍五入。
    """
    return round(value, ndigits)


##############################################
# 信号生成及回测函数
##############################################

def compute_signal(df, col_name):
    """
    根据历史行情数据(df)和指定信号名称(col_name)生成交易信号。
    支持的信号类型包括：abs, relate, donchian, boll, macross, rsi, macd, cci, atr。
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
        trade_price_series = target_price
        # 可选记录调试数据
        df["target_price"] = target_price
        df["signal_series"] = signal_series
        df["trade_price_series"] = trade_price_series
        return signal_series, trade_price_series

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
        signal_series = signal_series & valid_trade
        return signal_series, target_price

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
        trade_price_series = target_price.round(4)
        return signal_series, trade_price_series

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


@njit
def calculate_max_sequence_numba(series):
    """
    利用 numba 加速计算连续亏损（最小累计收益）及对应的交易数等信息。
    """
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
    """
    利用 numba 加速计算连续盈利（最大累计收益）及对应的交易数等信息。
    """
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


def op_signal(df, sig):
    """
    计算单个信号（调用 compute_signal 生成信号及目标价格），并筛选出交易次数大于 100 的信号。
    """
    s, p = compute_signal(df, sig)
    s_np = series_to_numpy(s)
    p_np = series_to_numpy(p)
    if p_np.dtype == np.float64:
        p_np = p_np.astype(np.float32)
    indices = np.nonzero(s_np)[0]
    if indices.size < 0:
        return None
    return (indices.astype(np.int32), p_np[indices])

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

def get_detail_backtest_result_op(df, kai_column, pin_column, is_filter=True, is_detail=False, is_reverse=False):
    """
    根据预计算的信号数据获取回测统计结果，只计算所需字段：
    kai_column, pin_column, hold_time_mean, max_hold_time, hold_time_std,
    loss_rate, net_profit_rate, fix_profit, avg_profit_rate, same_count
    返回值为 (kai_data_df, statistic_dict)。
    当交易信号数为 0 时，返回的统计信息各项值均为 0，而不是返回 None。
    """
    # 注意：需要保证 GLOBAL_SIGNALS, op_signal 和 safe_round 已经在其他地方定义好。
    global GLOBAL_SIGNALS

    try:
        kai_idx, kai_prices = GLOBAL_SIGNALS[kai_column]
        pin_idx, pin_prices = GLOBAL_SIGNALS[pin_column]
    except KeyError:
        result_kai = op_signal(df, kai_column)
        result_pin = op_signal(df, pin_column)
        if result_kai is None or result_pin is None:
            # 即使获取不到信号，也返回统计信息为 0
            statistic_dict = {
                "kai_column": kai_column,
                "pin_column": pin_column,
                "hold_time_mean": 0,
                "max_hold_time": 0,
                "hold_time_std": 0,
                "loss_rate": 0,
                "net_profit_rate": 0,
                "fix_profit": 0,
                "avg_profit_rate": 0,
                "same_count": 0
            }
            return df.iloc[[]].copy(), statistic_dict
        kai_idx, kai_prices = result_kai
        pin_idx, pin_prices = result_pin

    # 构造信号对应的 DataFrame（拷贝以防止修改原数据）
    kai_data_df = df.iloc[kai_idx].copy()
    pin_data_df = df.iloc[pin_idx].copy()
    kai_data_df["kai_price"] = kai_prices
    pin_data_df["pin_price"] = pin_prices

    # 根据时间索引匹配信号（可能存在重复匹配）
    # 同时计算 common_index 用于 same_count 字段
    common_index = kai_data_df.index.intersection(pin_data_df.index)
    kai_idx_arr = np.asarray(kai_data_df.index)
    pin_idx_arr = np.asarray(pin_data_df.index)
    pin_match_indices = np.searchsorted(pin_idx_arr, kai_idx_arr, side="right")
    valid_mask = pin_match_indices < len(pin_idx_arr)

    kai_data_df = kai_data_df.iloc[valid_mask].copy()
    kai_idx_valid = kai_idx_arr[valid_mask]
    pin_match_indices_valid = pin_match_indices[valid_mask]
    matched_pin = pin_data_df.iloc[pin_match_indices_valid].copy()

    kai_data_df["pin_price"] = matched_pin["pin_price"].values
    kai_data_df["pin_time"] = matched_pin["timestamp"].values
    kai_data_df["hold_time"] = matched_pin.index.values - kai_idx_valid

    # 过滤重复匹配
    if is_filter:
        kai_data_df = kai_data_df.sort_values("timestamp").drop_duplicates("pin_time", keep="first")

    trade_count = len(kai_data_df)
    if trade_count == 0:
        statistic_dict = {
            "kai_column": kai_column,
            "pin_column": pin_column,
            'kai_count':trade_count,
            "hold_time_mean": 0,
            "max_hold_time": 0,
            "hold_time_std": 0,
            "loss_rate": 0,
            "net_profit_rate": 0,
            "fix_profit": 0,
            "avg_profit_rate": 0,
            "same_count": len(common_index)
        }
        return kai_data_df, statistic_dict

    # 若存在价格修正，则对交易价格进行映射
    pin_price_map = kai_data_df.set_index("pin_time")["pin_price"]
    mapped_prices = kai_data_df["timestamp"].map(pin_price_map)
    if mapped_prices.notna().sum() > 0:
        kai_data_df["kai_price"] = mapped_prices.combine_first(kai_data_df["kai_price"])

    # 判断交易方向，根据 is_reverse 参数以及 kai_column 名称判断买/卖
    if is_reverse:
        is_long = "short" in kai_column.lower()
    else:
        is_long = "long" in kai_column.lower()

    # 计算收益率（含交易成本 0.07）
    if is_long:
        profit_series = ((kai_data_df["pin_price"] - kai_data_df["kai_price"]) /
                         kai_data_df["kai_price"] * 100).round(4)
    else:
        profit_series = ((kai_data_df["kai_price"] - kai_data_df["pin_price"]) /
                         kai_data_df["kai_price"] * 100).round(4)
    # true_profit 扣除固定交易成本
    kai_data_df["true_profit"] = profit_series - 0.07

    # fix_profit: 累积经过价格修正的交易收益
    fix_profit = safe_round(kai_data_df[mapped_prices.notna()]["true_profit"].sum(), ndigits=4)
    # net_profit_rate：所有 true_profit 累计减去 fix_profit
    net_profit_rate = kai_data_df["true_profit"].sum() - fix_profit

    # 计算 avg_profit_rate：所有 true_profit 均值乘以 100
    avg_profit_rate = safe_round(kai_data_df["true_profit"].mean() * 100, 4)

    # 计算持仓时间指标
    hold_time_mean = kai_data_df["hold_time"].mean()
    max_hold_time = kai_data_df["hold_time"].max()
    hold_time_std = kai_data_df["hold_time"].std()

    # 计算 loss_rate：亏损交易比例（true_profit < 0）
    loss_count = (kai_data_df["true_profit"] < 0).sum()
    loss_rate = loss_count / trade_count

    # same_count: 两个信号原始时间索引的交集数
    same_count = len(common_index)

    # Weekly statistics
    weekly_groups = kai_data_df["timestamp"].dt.to_period("W")
    weekly_agg = kai_data_df.groupby(weekly_groups)["true_profit"].agg(["sum", "mean", "count"])
    active_weeks = weekly_agg.shape[0]
    total_weeks = len(
        pd.period_range(start=kai_data_df["timestamp"].min(), end=kai_data_df["timestamp"].max(), freq='W'))
    active_week_ratio = active_weeks / total_weeks if total_weeks else 0
    weekly_net_profit_std = weekly_agg["sum"].std() if "sum" in weekly_agg else 0
    weekly_avg_profit_std = weekly_agg["mean"].std() if "mean" in weekly_agg else 0
    weekly_net_profit_min = weekly_agg["sum"].min() if "sum" in weekly_agg else 0
    weekly_net_profit_max = weekly_agg["sum"].max() if "sum" in weekly_agg else 0
    weekly_loss_rate = ((weekly_agg["sum"] < 0).sum() / active_weeks) if active_weeks else 0
    weekly_net_profit_detail = {str(week): round(val, 4) for week, val in weekly_agg["sum"].to_dict().items()}

    statistic_dict = {
        "kai_column": kai_column,
        "pin_column": pin_column,
        'kai_count': trade_count,
        "hold_time_mean": hold_time_mean,
        "max_hold_time": max_hold_time,
        "hold_time_std": hold_time_std,
        "loss_rate": loss_rate,
        "net_profit_rate": net_profit_rate,
        "fix_profit": fix_profit,
        "avg_profit_rate": avg_profit_rate,
        "same_count": same_count,
        "active_week_ratio": safe_round(active_week_ratio, 4),
        "weekly_loss_rate": safe_round(weekly_loss_rate, 4),
        "weekly_net_profit_min": safe_round(weekly_net_profit_min, 4),
        "weekly_net_profit_max": safe_round(weekly_net_profit_max, 4),
        "weekly_net_profit_std": safe_round(weekly_net_profit_std, 4),
        "weekly_avg_profit_std": safe_round(weekly_avg_profit_std, 4),
        "weekly_net_profit_detail": optimize_detail(weekly_net_profit_detail),
    }
    return kai_data_df, statistic_dict


##############################################
# 多进程预计算及回测工具函数
##############################################

def process_signal(signal):
    """
    计算单个信号的预计算数据，若交易信号数不足 100，则返回 None。
    """
    try:
        s, p = compute_signal(df, signal)
        s_np = series_to_numpy(s)
        p_np = series_to_numpy(p)
        if p_np.dtype == np.float64:
            p_np = p_np.astype(np.float32)
        indices = np.nonzero(s_np)[0]
        if indices.size < 0:
            return None
        return (signal, (indices.astype(np.int32), p_np[indices]))
    except Exception as e:
        print(f"Error processing signal {signal}: {e}")
        return None


def process_batch(signal_batch):
    """
    处理一批信号，返回该批次中成功预计算的信号数据列表。
    """
    batch_results = []
    for sig in signal_batch:
        res = process_signal(sig)
        if res is not None:
            batch_results.append(res)
    return batch_results


def compute_precomputed_signals(df, signals):
    """
    利用多进程计算所有指定信号的预计算数据，返回一个字典：
      {signal: (indices, prices)}
    """
    num_workers = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker1, initargs=(df,)) as pool:
        results = pool.map(process_signal, signals)
    precomputed = {}
    for res in results:
        if res is not None:
            sig, data = res
            precomputed[sig] = data
    return precomputed


def init_worker1(dataframe):
    """
    进程池初始化函数，将全局的 df 设置为传入的 dataframe。
    """
    global df
    df = dataframe


def init_worker_with_signals(signals, dataframe):
    """
    进程池初始化函数，将全局变量 GLOBAL_SIGNALS 和 df 同时设置。
    """
    global GLOBAL_SIGNALS, df
    GLOBAL_SIGNALS = signals
    df = dataframe


def process_signal_pair(pair):
    """
    处理一个信号对，调用 get_detail_backtest_result_op 计算回测结果，
    返回 (kai_column, pin_column, statistic_dict)。
    """
    kai_column, pin_column = pair
    try:
        _, stat = get_detail_backtest_result_op(df, kai_column, pin_column, is_filter=True, is_detail=False,
                                                is_reverse=False)
    except Exception as e:
        print(f"Error processing pair ({kai_column}, {pin_column}): {e}")
        stat = None
    return (kai_column, pin_column, stat)


##############################################
# 主流程：加载文件、预计算信号、多进程计算回测结果并保存
##############################################


def precompute_signals(df, signals):
    """
    对传入的 signals 列表（信号名称）采用多进程进行预计算，返回 dict 格式数据
    """
    num_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_workers, initializer=init_worker1, initargs=(df,))
    results = pool.map(process_signal, signals)
    pool.close()
    pool.join()
    precomputed = {}
    for res in results:
        if res is not None:
            sig, data = res
            precomputed[sig] = data
    return precomputed


def load_or_compute_precomputed_signals(df, signals, key_name):
    """
    先尝试从 temp_back 中加载预计算数据，否则计算后保存
    """
    if not os.path.exists("temp_back"):
        os.makedirs("temp_back")
    file_path = os.path.join("temp_back", f"precomputed_signals_{key_name}_{len(signals)}.pkl")
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                precomputed = pickle.load(f)
            print(f"Loaded precomputed signals from {file_path}, total {len(precomputed)} signals.")
            return precomputed
        except Exception as e:
            print(f"Failed to load precomputed signals: {e}. Recomputing.")
    print("Precomputing signals ...")
    precomputed = precompute_signals(df, signals)
    try:
        with open(file_path, "wb") as f:
            pickle.dump(precomputed, f)
        print(f"Saved precomputed signals to {file_path}.")
    except Exception as e:
        print(f"Error saving precomputed signals: {e}")
    return precomputed


def validation(market_data_file):
    base_name = os.path.basename(market_data_file)
    base_name = base_name.replace("-USDT-SWAP.csv", "").replace("origin_data_", "")
    inst_id_list = ['BTC', 'ETH', 'SOL', 'TON', 'DOGE', 'XRP', 'PEPE']
    for inst_id in inst_id_list:
        if inst_id in base_name:
            break

    # 2. 加载行情数据（market data），确保字段正确并转换 timestamp 字段
    df_local = pd.read_csv(market_data_file)
    needed_columns = ["timestamp", "high", "low", "close"]
    for col in needed_columns:
        if col not in df_local.columns:
            print(f"行情数据缺少必要字段：{col}")
            sys.exit(1)
    df_local = df_local[needed_columns]
    min_price = df_local["low"].min()
    # 如果min_price值小于1，则对价格进行放大
    while min_price < 1:
        df_local["high"] *= 10
        df_local["low"] *= 10
        df_local["close"] *= 10
        min_price *= 10

    jingdu = "float32"
    df_local["chg"] = (df_local["close"].pct_change() * 100).astype("float16")
    df_local["high"] = df_local["high"].astype(jingdu)
    df_local["low"] = df_local["low"].astype(jingdu)
    df_local["close"] = df_local["close"].astype(jingdu)
    df_local["timestamp"] = pd.to_datetime(df_local["timestamp"])

    print(f"Loaded market data: 共 {df_local.shape[0]} 行")

    # 3. 设置主进程全局变量，用于预计算
    global df
    df = df_local

    # 注意：这里只读取需要的列： kai_column 和 pin_column
    stat_df_file_list = [os.path.join('temp_back', f'{inst_id}_origin_good_op_all_false.parquet')]
    for stat_df_file in stat_df_file_list:
        try:
            # 1. 加载 stat_df 文件（只读取必要的两列）
            stat_df = pd.read_parquet(stat_df_file, columns=["kai_column", "pin_column"])
            stat_df_base_name = os.path.basename(stat_df_file)
            if "kai_column" not in stat_df.columns or "pin_column" not in stat_df.columns:
                print("输入的 stat_df 文件中必须包含 'kai_column' 和 'pin_column' 两列")
                sys.exit(1)
            print(f"Loaded stat_df: 共 {stat_df.shape[0]} 行")

            # 2. 获取所有信号对
            pairs = list(stat_df[['kai_column', 'pin_column']].itertuples(index=False, name=None))

            # 3. 提取候选信号（取 kai_column 和 pin_column 并集），以便预计算
            unique_signals = set(stat_df["kai_column"].unique()).union(set(stat_df["pin_column"].unique()))
            unique_signals = list(unique_signals)
            print(f"Total unique signals to precompute: {len(unique_signals)}")

            # 4. 预计算信号并保存（key_name 可根据需要自定义）
            key_name = base_name
            precomputed = load_or_compute_precomputed_signals(df_local, unique_signals, key_name)
            # 更新全局预计算信号数据
            global GLOBAL_SIGNALS
            GLOBAL_SIGNALS = precomputed
            total_size = sys.getsizeof(precomputed)
            for sig, (s_np, p_np) in precomputed.items():
                total_size += sys.getsizeof(sig) + s_np.nbytes + p_np.nbytes
            print(f"precomputed_signals 占用内存总大小: {total_size / (1024 * 1024):.2f} MB")
            print(f"Total candidate signal pairs: {len(pairs)}")
            # 删除 stat_df，释放内存
            del stat_df

            # 根据内存限制调整进程数
            max_memory = 50  # 单位：GB
            pool_processes = min(30, int(max_memory * 1024 * 1024 * 1024 / total_size) if total_size > 0 else 1)
            print(f"进程数限制为 {pool_processes}，根据内存限制调整。")

            # 定义每个批次处理的 pair 数量
            BATCH_SIZE = 10000000
            total_pairs = len(pairs)
            total_batches = (total_pairs - 1) // BATCH_SIZE + 1

            # 用于存储每个批次产出的结果文件路径，用于后续合并
            batch_output_files = []

            for batch_index, start_idx in enumerate(range(0, total_pairs, BATCH_SIZE)):
                # 获取当前时间
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                start_time = time.time()
                output_file = os.path.join("temp_back",
                                           f"{stat_df_base_name}_{base_name}statistic_results_{batch_index}.parquet")
                if os.path.exists(output_file):
                    print(f"{output_file} 已存在，跳过处理。")
                    batch_output_files.append(output_file)
                    continue
                end_idx = min(start_idx + BATCH_SIZE, total_pairs)
                batch_pairs = pairs[start_idx:end_idx]
                print(f"Processing batch {batch_index + 1}/{total_batches} with {len(batch_pairs)} pairs... [{current_time}]")

                with multiprocessing.Pool(processes=pool_processes,
                                          initializer=init_worker_with_signals,
                                          initargs=(GLOBAL_SIGNALS, df)) as pool:
                    results = pool.map(process_signal_pair, batch_pairs, chunksize=1000)

                # 过滤掉返回 None 的结果
                results_filtered = [r for r in results if r[2] is not None]
                print(f"Batch {batch_index + 1}: 成功处理 {len(results_filtered)} 个信号对。")

                # 合并所有统计字典为 DataFrame 并保存到文件
                stats_list = [r[2] for r in results_filtered]
                stats_df = pd.DataFrame(stats_list)
                stats_df.to_parquet(output_file, index=False, compression='snappy')
                batch_output_files.append(output_file)
                print(f"Batch {batch_index + 1} 统计结果已保存到 {output_file} (共 {stats_df.shape[0]} 行) 耗时 {time.time() - start_time:.2f} 秒。")

            # 合并所有批次的结果文件并保存最终结果
            merged_dfs = []
            for file in batch_output_files:
                if os.path.exists(file):
                    print(f"正在加载批次文件: {file}")
                    batch_df = pd.read_parquet(file)
                    merged_dfs.append(batch_df)
                else:
                    print(f"文件 {file} 不存在，跳过加载。")
            if merged_dfs:
                merged_df = pd.concat(merged_dfs, ignore_index=True)
                final_output_file = os.path.join("temp_back", f"{stat_df_base_name}_{base_name}statistic_results_final.parquet")
                merged_df.to_parquet(final_output_file, index=False, compression='snappy')
                print(f"所有批次结果已合并，并保存到 {final_output_file} (共 {merged_df.shape[0]} 行)。")
            else:
                print("没有找到任何批次结果文件，无法合并。")

        except Exception as e:
            print(f"处理 {stat_df_file} 时出错：{e}")


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
        result = [int(round(start + (end - start) * ((i/(number-1))**2))) for i in range(number)]
    final_result = []
    last_val = None
    for val in result:
        if start <= val <= end and val != last_val:
            final_result.append(val)
            last_val = val
    return final_result[:number]

def gen_abs_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = [x/20 for x in range(start_period1, end_period1, step1)]
    long_columns = [f"abs_{p}_{p1}_long" for p in period_list for p1 in period_list1 if p >= p1]
    short_columns = [f"abs_{p}_{p1}_short" for p in period_list for p1 in period_list1 if p >= p1]
    key_name = f"abs_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}"
    print(f"abs 生成 {len(long_columns)} 长信号和 {len(short_columns)} 短信号。")
    return long_columns, short_columns, key_name

def gen_macd_signal_name(start_period, end_period, step):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    signal_list = [9, 12, 15, 40]
    long_columns = [f"macd_{fast}_{slow}_{signal}_long" for fast in period_list for slow in period_list if slow > fast for signal in signal_list]
    short_columns = [f"macd_{fast}_{slow}_{signal}_short" for fast in period_list for slow in period_list if slow > fast for signal in signal_list]
    key_name = f"macd_{start_period}_{end_period}_{step}"
    print(f"MACD 生成 {len(long_columns)} 信号。")
    return long_columns, short_columns, key_name

def gen_cci_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = [x/10 for x in range(start_period1, end_period1, step1)]
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
    long_columns = [f"rsi_{p}_{ob}_{100-ob}_long" for p in period_list for ob in temp_list]
    short_columns = [f"rsi_{p}_{ob}_{100-ob}_short" for p in period_list for ob in temp_list]
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
    period_list1 = [x/10 for x in range(start_period1, end_period1, step1)]
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

    # 如有需要，可启用其他类型信号（目前部分信号生成被注释）
    # relate_long, relate_short, relate_key = gen_relate_signal_name(400, 1000, 100, 1, 40, 6)
    # column_list.append((relate_long, relate_short, relate_key))
    #
    # donchian_long, donchian_short, donchian_key = gen_donchian_signal_name(1, 20, 1)
    # column_list.append((donchian_long, donchian_short, donchian_key))

    # boll_long, boll_short, boll_key = gen_boll_signal_name(1, 3000, 100, 1, 50, 2)
    # column_list.append((boll_long, boll_short, boll_key))

    macross_long, macross_short, macross_key = gen_macross_signal_name(1, 3000, 100, 1, 3000, 100)
    column_list.append((macross_long, macross_short, macross_key))

    # rsi_long, rsi_short, rsi_key = gen_rsi_signal_name(1, 1000, 500)
    # column_list.append((rsi_long, rsi_short, rsi_key))

    # macd_long, macd_short, macd_key = gen_macd_signal_name(300, 1000, 50)
    # column_list.append((macd_long, macd_short, macd_key))

    # cci_long, cci_short, cci_key = gen_cci_signal_name(1, 2000, 1000, 1, 2, 1)
    # column_list.append((cci_long, cci_short, cci_key))

    # atr_long, atr_short, atr_key = gen_atr_signal_name(1, 3000, 3000)
    # column_list.append((atr_long, atr_short, atr_key))

    column_list = sorted(column_list, key=lambda x: len(x[0]))
    all_signals = []
    key_name = ""
    for long_cols, short_cols, temp_key in column_list:
        temp = long_cols + short_cols
        key_name += temp_key + "_"
        all_signals.extend(temp)
    return all_signals, key_name


def target_all(market_data_file):
    base_name = os.path.basename(market_data_file)
    base_name = base_name.replace("-USDT-SWAP.csv", "").replace("origin_data_", "")
    inst_id = base_name.split("_")[-1]

    # 1. 加载行情数据，确保必要字段存在，并转换 timestamp 字段
    df_local = pd.read_csv(market_data_file)
    needed_columns = ["timestamp", "high", "low", "close"]
    for col in needed_columns:
        if col not in df_local.columns:
            print(f"行情数据缺少必要字段：{col}")
            sys.exit(1)
    df_local = df_local[needed_columns]
    min_price = df_local["low"].min()
    # 如果 min_price 在小数点后面较多，则乘以 10，直到大于 1
    while min_price < 1:
        df_local["high"] *= 10
        df_local["low"] *= 10
        df_local["close"] *= 10
        min_price *= 10

    jingdu = "float32"
    df_local["chg"] = (df_local["close"].pct_change() * 100).astype("float16")
    df_local["high"] = df_local["high"].astype(jingdu)
    df_local["low"] = df_local["low"].astype(jingdu)
    df_local["close"] = df_local["close"].astype(jingdu)
    df_local["timestamp"] = pd.to_datetime(df_local["timestamp"])

    print(f"Loaded market data: 共 {df_local.shape[0]} 行")

    # 2. 将行情数据赋值给全局变量（用于后面传递给多进程进程）
    global df
    df = df_local

    # 3. 生成所有信号及候选信号对
    all_signals, key_name = generate_all_signals()
    stat_df_base_name = f'{base_name}_{key_name}'

    candidate_long_signals = [sig for sig in all_signals if "abs" in sig]
    candidate_short_signals = [sig for sig in all_signals if "macross" in sig]
    all_precompute_signals = list(set(candidate_long_signals + candidate_short_signals))


    # 4. 加载或者计算预计算信号（并更新全局预计算变量）
    precomputed = load_or_compute_precomputed_signals(df, all_precompute_signals, key_name)

    candidate_long_signals = [sig for sig in all_signals if "abs" in sig and 'long' in sig]


    total_combinations = len(candidate_long_signals) * len(candidate_short_signals)

    print(f"总共需要回测的组合数量: {total_combinations}")
    pairs = list(itertools.product(candidate_long_signals, candidate_short_signals))
    print(f"Total candidate signal pairs: {len(pairs)}")

    global GLOBAL_SIGNALS
    GLOBAL_SIGNALS = precomputed
    total_size = sys.getsizeof(precomputed)
    for sig, (s_np, p_np) in precomputed.items():
        total_size += sys.getsizeof(sig) + s_np.nbytes + p_np.nbytes
    print(f"precomputed_signals 占用内存总大小: {total_size / (1024 * 1024):.2f} MB")

    # 5. 根据内存限制确定多进程进程数
    max_memory = 45  # 单位：GB
    pool_processes = min(32, int(max_memory * 1024 * 1024 * 1024 / total_size) if total_size > 0 else 1)
    print(f"进程数限制为 {pool_processes}，根据内存限制调整。")

    # 6. 以 10_000_000 个组合为一组进行计算
    chunk_size = 10_000_00
    total_pairs = len(pairs)
    num_parts = math.ceil(total_pairs / chunk_size)
    print(f"将分 {num_parts} 个部分进行计算，每部分大小为 {chunk_size} 个组合")

    # 确保结果输出文件夹存在
    output_folder = "temp"
    os.makedirs(output_folder, exist_ok=True)

    for part in range(num_parts):
        # 获取当前时间，以可读性高的格式
        start_time = time.time()
        part_start = part * chunk_size
        part_end = min(total_pairs, part_start + chunk_size)
        part_file = os.path.join(output_folder, f"{stat_df_base_name}_part{part}_statistic_results.csv")
        if os.path.exists(part_file):
            print(f"文件 {part_file} 已存在，跳过此部分。")
            continue

        current_pairs = pairs[part_start:part_end]
        print(f"Processing part {part}: 处理组合索引 {part_start} ~ {part_end}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        with multiprocessing.Pool(processes=pool_processes, initializer=init_worker_with_signals,
                                  initargs=(GLOBAL_SIGNALS, df)) as pool:
            results = pool.map(process_signal_pair, current_pairs, chunksize=1000)

        # 过滤掉返回 None 的结果（假设 process_signal_pair 返回的结果结构为 (sig1, sig2, stat_dict)）
        results_filtered = [r for r in results if r and r[2] is not None]
        print(f"Part {part}: 成功处理 {len(results_filtered)} 个信号对。")
        stats_list = [r[2] for r in results_filtered]
        stats_df = pd.DataFrame(stats_list)
        stats_df.to_csv(part_file, index=False)
        print(f"部分统计结果已保存到 {part_file} (耗时 {time.time() - start_time:.2f} 秒) 当前时间 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("所有部分处理完成。")


if __name__ == "__main__":
    start_time = time.time()
    data_path_list = [
        # "kline_data/origin_data_1m_110000_BTC-USDT-SWAP.csv",
        # "kline_data/origin_data_1m_10000_BTC-USDT-SWAP.csv",
        # "kline_data/origin_data_1m_110000_SOL-USDT-SWAP.csv",
        # "kline_data/origin_data_1m_10000_SOL-USDT-SWAP.csv",

        # "kline_data/origin_data_1m_110000_ETH-USDT-SWAP.csv",
        # "kline_data/origin_data_1m_10000_ETH-USDT-SWAP.csv",
        #
        # "kline_data/origin_data_1m_110000_TON-USDT-SWAP.csv",
        # "kline_data/origin_data_1m_10000_TON-USDT-SWAP.csv",
        # "kline_data/origin_data_1m_110000_DOGE-USDT-SWAP.csv",
        # "kline_data/origin_data_1m_110000_XRP-USDT-SWAP.csv",
        # "kline_data/origin_data_1m_110000_PEPE-USDT-SWAP.csv",
        # "kline_data/origin_data_1m_10000_DOGE-USDT-SWAP.csv",
        # "kline_data/origin_data_1m_10000_XRP-USDT-SWAP.csv",
        # "kline_data/origin_data_1m_10000_PEPE-USDT-SWAP.csv",

        # "kline_data/origin_data_1m_140000_BTC-USDT-SWAP_2025-04-10.csv",
        # "kline_data/origin_data_1m_140000_SOL-USDT-SWAP_2025-04-10.csv",
        # "kline_data/origin_data_1m_140000_ETH-USDT-SWAP_2025-04-10.csv",
        # "kline_data/origin_data_1m_140000_TON-USDT-SWAP_2025-04-10.csv",
        # "kline_data/origin_data_1m_140000_DOGE-USDT-SWAP_2025-04-10.csv",
        # "kline_data/origin_data_1m_140000_XRP-USDT-SWAP_2025-04-10.csv",
        "kline_data/origin_data_1m_140000_PEPE-USDT-SWAP_2025-04-10.csv",
        # "kline_data/origin_data_1m_10000_ETH-USDT-SWAP_2025-04-07.csv",
        # "kline_data/origin_data_1m_10000_SOL-USDT-SWAP_2025-04-07.csv",
        # "kline_data/origin_data_1m_10000_TON-USDT-SWAP_2025-04-07.csv",
        # "kline_data/origin_data_1m_10000_DOGE-USDT-SWAP_2025-04-07.csv",
        # "kline_data/origin_data_1m_10000_XRP-USDT-SWAP_2025-04-07.csv",
        # "kline_data/origin_data_1m_10000_PEPE-USDT-SWAP_2025-04-07.csv",


        # "kline_data/origin_data_1m_10000000_SOL-USDT-SWAP.csv",
        # "kline_data/origin_data_1m_10000000_BTC-USDT-SWAP.csv",
        # "kline_data/origin_data_1m_10000000_ETH-USDT-SWAP.csv",
        # "kline_data/origin_data_1m_10000000_TON-USDT-SWAP.csv",
        # "kline_data/origin_data_1m_10000000_DOGE-USDT-SWAP.csv",
        # "kline_data/origin_data_1m_10000000_XRP-USDT-SWAP.csv",
        # "kline_data/origin_data_1m_10000000_PEPE-USDT-SWAP.csv"

        # "kline_data/origin_data_5m_10000000_SOL-USDT-SWAP.csv",
        # "kline_data/origin_data_5m_10000000_BTC-USDT-SWAP.csv",
        # "kline_data/origin_data_5m_10000000_ETH-USDT-SWAP.csv",
        # "kline_data/origin_data_5m_10000000_TON-USDT-SWAP.csv",
    ]
    for data_path in data_path_list:
        try:
            validation(data_path)
            print(f"{data_path} 总耗时 {time.time() - start_time:.2f} 秒。")
        except Exception as e:
            traceback.print_exc()
            print(f"处理 {data_path} 时出错：{e}")