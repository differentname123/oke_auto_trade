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

import os
import sys
import time
import pickle
import multiprocessing
import traceback

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


def get_detail_backtest_result_op(df, kai_column, pin_column, is_filter=True, is_detail=False, is_reverse=False):
    """
    根据预计算的信号数据获取回测统计结果。返回值为 (kai_data_df, statistic_dict)。
    若交易信号不足或数据不满足过滤条件，则返回 (None, None)。
    """
    global GLOBAL_SIGNALS

    try:
        kai_idx, kai_prices = GLOBAL_SIGNALS[kai_column]
        pin_idx, pin_prices = GLOBAL_SIGNALS[pin_column]
    except KeyError:
        result_kai = op_signal(df, kai_column)
        result_pin = op_signal(df, pin_column)
        if result_kai is None or result_pin is None:
            return None, None
        kai_idx, kai_prices = result_kai
        pin_idx, pin_prices = result_pin

    # 构造信号对应的 DataFrame（拷贝以防止修改原数据）
    kai_data_df = df.iloc[kai_idx].copy()
    pin_data_df = df.iloc[pin_idx].copy()
    kai_data_df["kai_price"] = kai_prices
    pin_data_df["pin_price"] = pin_prices

    # 根据时间索引匹配信号（可能存在重复匹配）
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

    # 判断方向
    if is_reverse:
        is_long = "short" in kai_column.lower()
    else:
        is_long = "long" in kai_column.lower()

    if is_filter:
        kai_data_df = kai_data_df.sort_values("timestamp").drop_duplicates("pin_time", keep="first")

    trade_count = len(kai_data_df)
    total_count = len(df)

    # 若存在价格修正，则对交易价格进行映射
    pin_price_map = kai_data_df.set_index("pin_time")["pin_price"]
    mapped_prices = kai_data_df["timestamp"].map(pin_price_map)
    if mapped_prices.notna().sum() > 0:
        kai_data_df["kai_price"] = mapped_prices.combine_first(kai_data_df["kai_price"])
    modification_rate = (100 * mapped_prices.notna().sum() / trade_count) if trade_count else 0

    # 计算收益率（含交易成本0.07）
    if is_long:
        profit_series = ((kai_data_df["pin_price"] - kai_data_df["kai_price"]) /
                         kai_data_df["kai_price"] * 100).round(4)
    else:
        profit_series = ((kai_data_df["kai_price"] - kai_data_df["pin_price"]) /
                         kai_data_df["kai_price"] * 100).round(4)
    kai_data_df["profit"] = profit_series
    kai_data_df["true_profit"] = profit_series - 0.07
    profit_sum = profit_series.sum()
    max_single_profit = kai_data_df["true_profit"].max()
    min_single_profit = kai_data_df["true_profit"].min()

    true_profit_std = kai_data_df["true_profit"].std()
    true_profit_mean = kai_data_df["true_profit"].mean() * 100 if trade_count > 0 else 0
    fix_profit = safe_round(kai_data_df[mapped_prices.notna()]["true_profit"].sum(), ndigits=4)
    net_profit_rate = kai_data_df["true_profit"].sum() - fix_profit

    profits_arr = kai_data_df["true_profit"].values
    max_loss, max_loss_start_idx, max_loss_end_idx, loss_trade_count = calculate_max_sequence_numba(profits_arr)

    max_profit, max_profit_start_idx, max_profit_end_idx, profit_trade_count = calculate_max_profit_numba(profits_arr)

    if (trade_count > 0 and max_loss_start_idx < len(kai_data_df) and max_loss_end_idx < len(kai_data_df)):
        max_loss_start_time = kai_data_df.iloc[max_loss_start_idx]["timestamp"]
        max_loss_end_time = kai_data_df.iloc[max_loss_end_idx]["timestamp"]
        max_loss_hold_time = kai_data_df.index[max_loss_end_idx] - kai_data_df.index[max_loss_start_idx]
    else:
        max_loss_start_time = max_loss_end_time = max_loss_hold_time = None

    if (trade_count > 0 and max_profit_start_idx < len(kai_data_df) and max_profit_end_idx < len(kai_data_df)):
        max_profit_start_time = kai_data_df.iloc[max_profit_start_idx]["timestamp"]
        max_profit_end_time = kai_data_df.iloc[max_profit_end_idx]["timestamp"]
        max_profit_hold_time = kai_data_df.index[max_profit_end_idx] - kai_data_df.index[max_profit_start_idx]
    else:
        max_profit_start_time = max_profit_end_time = max_profit_hold_time = None

    profit_df = kai_data_df[kai_data_df["true_profit"] > 0]
    loss_df = kai_data_df[kai_data_df["true_profit"] < 0]
    fu_profit_sum = loss_df["true_profit"].sum()
    fu_profit_mean = safe_round(loss_df["true_profit"].mean() if not loss_df.empty else 0, ndigits=4)
    zhen_profit_sum = profit_df["true_profit"].sum()
    zhen_profit_mean = safe_round(profit_df["true_profit"].mean() if not profit_df.empty else 0, ndigits=4)
    loss_rate = (loss_df.shape[0] / trade_count) if trade_count else 0
    loss_time = loss_df["hold_time"].sum() if not loss_df.empty else 0
    profit_time = profit_df["hold_time"].sum() if not profit_df.empty else 0
    loss_time_rate = (loss_time / (loss_time + profit_time)) if (loss_time + profit_time) else 0

    trade_rate = (100 * trade_count / total_count) if total_count else 0
    hold_time_mean = kai_data_df["hold_time"].mean() if trade_count else 0
    max_hold_time = kai_data_df["hold_time"].max() if trade_count else 0


    monthly_groups = kai_data_df["timestamp"].dt.to_period("M")
    monthly_agg = kai_data_df.groupby(monthly_groups)["true_profit"].agg(["sum", "mean", "count"])
    monthly_trade_std = monthly_agg["count"].std() if "count" in monthly_agg else 0
    active_months = monthly_agg.shape[0]
    total_months = 22
    active_month_ratio = active_months / total_months if total_months else 0
    monthly_net_profit_std = monthly_agg["sum"].std() if "sum" in monthly_agg else 0
    monthly_avg_profit_std = monthly_agg["mean"].std() if "mean" in monthly_agg else 0
    monthly_net_profit_min = monthly_agg["sum"].min() if "sum" in monthly_agg else 0
    monthly_net_profit_max = monthly_agg["sum"].max() if "sum" in monthly_agg else 0
    monthly_loss_rate = ((monthly_agg["sum"] < 0).sum() / active_months) if active_months else 0

    monthly_net_profit_detail = {str(month): round(val, 4) for month, val in monthly_agg["sum"].to_dict().items()}
    monthly_trade_count_detail = {str(month): int(val) for month, val in monthly_agg["count"].to_dict().items()}

    hold_time_std = kai_data_df["hold_time"].std()

    if not profit_df.empty:
        top_profit_count = max(1, int(np.ceil(len(profit_df) * 0.1)))
        profit_sorted = profit_df.sort_values("true_profit", ascending=False)
        top_profit_sum = profit_sorted["true_profit"].iloc[:top_profit_count].sum()
        total_profit_sum = profit_df["true_profit"].sum()
        top_profit_ratio = (top_profit_sum / total_profit_sum) if total_profit_sum != 0 else 0
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
    same_count_rate = safe_round(100 * len(common_index) / min(len(kai_data_df), len(pin_data_df)) if trade_count else 0, 4)
    statistic_dict = {
        "kai_side": "long" if is_long else "short",
        "kai_column": kai_column,
        "pin_column": pin_column,
        "kai_count": trade_count,
        "total_count": total_count,
        "trade_rate": safe_round(trade_rate, 4),
        "hold_time_mean": hold_time_mean,
        "max_hold_time": max_hold_time,
        "hold_time_std": hold_time_std,
        "loss_rate": loss_rate,
        "loss_time_rate": loss_time_rate,
        "zhen_profit_sum": zhen_profit_sum,
        "zhen_profit_mean": zhen_profit_mean,
        "fu_profit_sum": fu_profit_sum,
        "fu_profit_mean": fu_profit_mean,
        "profit_rate": profit_sum,
        "max_profit": max_single_profit,
        "min_profit": min_single_profit,
        "cost_rate": trade_count * 0.07,
        "net_profit_rate": net_profit_rate,
        "fix_profit": fix_profit,
        "avg_profit_rate": safe_round(true_profit_mean, 4),
        "true_profit_std": true_profit_std,
        "max_consecutive_loss": safe_round(max_loss, 4),
        "max_loss_trade_count": loss_trade_count,
        "max_loss_hold_time": max_loss_hold_time,
        "max_loss_start_time": max_loss_start_time,
        "max_loss_end_time": max_loss_end_time,
        "max_consecutive_profit": safe_round(max_profit, 4),
        "max_profit_trade_count": profit_trade_count,
        "max_profit_hold_time": max_profit_hold_time,
        "max_profit_start_time": max_profit_start_time,
        "max_profit_end_time": max_profit_end_time,
        "same_count": len(common_index),
        "same_count_rate": same_count_rate,
        "true_same_count_rate": modification_rate,
        "monthly_trade_std": safe_round(monthly_trade_std, 4),
        "active_month_ratio": safe_round(active_month_ratio, 4),
        "monthly_loss_rate": safe_round(monthly_loss_rate, 4),
        "monthly_net_profit_min": safe_round(monthly_net_profit_min, 4),
        "monthly_net_profit_max": safe_round(monthly_net_profit_max, 4),
        "monthly_net_profit_std": safe_round(monthly_net_profit_std, 4),
        "monthly_avg_profit_std": safe_round(monthly_avg_profit_std, 4),
        "top_profit_ratio": safe_round(top_profit_ratio, 4),
        "top_loss_ratio": safe_round(top_loss_ratio, 4),
        "is_reverse": is_reverse,
        "monthly_net_profit_detail": monthly_net_profit_detail,
        "monthly_trade_count_detail": monthly_trade_count_detail
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
    inst_id = base_name.split("_")[-1]

    # 2. 加载行情数据（market data），确保字段正确并转换 timestamp 字段
    df_local = pd.read_csv(market_data_file)
    needed_columns = ["timestamp", "high", "low", "close"]
    for col in needed_columns:
        if col not in df_local.columns:
            print(f"行情数据缺少必要字段：{col}")
            sys.exit(1)
    df_local = df_local[needed_columns]
    min_price = df_local["low"].min()
    # 如果min_price在小数点很后面，那需要乘以10，直到大于1
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

    # 3. 设置主进程 global df，用于预计算
    global df
    df = df_local

    stat_df_file_list = [f'temp_back/{inst_id}_origin_good_op_all_false.csv']
    for stat_df_file in stat_df_file_list:
        try:
            # 1. 加载 stat_df 文件
            stat_df = pd.read_csv(stat_df_file)
            stat_df_base_name = os.path.basename(stat_df_file)
            if "kai_column" not in stat_df.columns or "pin_column" not in stat_df.columns:
                print("输入的 stat_df 文件中必须包含 'kai_column' 和 'pin_column' 两列")
                sys.exit(1)
            print(f"Loaded stat_df: 共 {stat_df.shape[0]} 行")

            pairs = list(stat_df[['kai_column', 'pin_column']].itertuples(index=False, name=None))

            # 4. 从 stat_df 中提取所有候选信号（取 kai_column 和 pin_column 的并集）
            unique_signals = set(stat_df["kai_column"].unique()).union(set(stat_df["pin_column"].unique()))
            unique_signals = list(unique_signals)
            print(f"Total unique signals to precompute: {len(unique_signals)}")

            # 5. 预计算信号并保存（key_name 可根据需要自定义）
            key_name = base_name
            precomputed = load_or_compute_precomputed_signals(df_local, unique_signals, key_name)
            # 更新全局预计算信号数据
            global GLOBAL_SIGNALS
            GLOBAL_SIGNALS = precomputed
            total_size = sys.getsizeof(precomputed)
            for sig, (s_np, p_np) in precomputed.items():
                total_size += sys.getsizeof(sig) + s_np.nbytes + p_np.nbytes
            print(f"precomputed_signals 占用内存总大小: {total_size / (1024 * 1024):.2f} MB")

            # 6. 构造 candidate 列表，每个 candidate 为：(行号, kai_column, pin_column)
            candidates = []
            for idx, row in stat_df.iterrows():
                candidates.append((idx, row["kai_column"], row["pin_column"]))
            print(f"Total candidate signal pairs: {len(candidates)}")

            max_memory = 45
            pool_processes = min(32, int(max_memory * 1024 * 1024 * 1024 / total_size) if total_size > 0 else 1)
            print(f"进程数限制为 {pool_processes}，根据内存限制调整。")
            with multiprocessing.Pool(processes=pool_processes, initializer=init_worker_with_signals,
                                      initargs=(GLOBAL_SIGNALS, df)) as pool:
                results = pool.map(process_signal_pair, pairs, chunksize=100)

            # 过滤掉返回 None 的结果
            results_filtered = [r for r in results if r[2] is not None]
            print(f"成功处理 {len(results_filtered)} 个信号对。")

            # 合并所有统计字典为 DataFrame 并保存到文件
            stats_list = [r[2] for r in results_filtered]
            stats_df = pd.DataFrame(stats_list)
            output_file = os.path.join("temp_back", f"{stat_df_base_name}_statistic_results.csv")
            stats_df.to_csv(output_file, index=False)
            print(f"所有统计结果已保存到 {output_file}")
        except Exception as e:
            print(f"处理 {stat_df_file} 时出错：{e}")


if __name__ == "__main__":
    start_time = time.time()
    data_path_list = [
        "kline_data/origin_data_1m_110000_SOL-USDT-SWAP.csv",
        "kline_data/origin_data_1m_110000_BTC-USDT-SWAP.csv",
        "kline_data/origin_data_1m_110000_ETH-USDT-SWAP.csv",
        "kline_data/origin_data_1m_110000_TON-USDT-SWAP.csv",
        "kline_data/origin_data_1m_110000_DOGE-USDT-SWAP.csv",
        "kline_data/origin_data_1m_110000_XRP-USDT-SWAP.csv",
        "kline_data/origin_data_1m_110000_PEPE-USDT-SWAP.csv",

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