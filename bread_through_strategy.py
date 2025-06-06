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
##############################################
# 辅助函数
##############################################
def series_to_numpy(series):
    """将 Pandas Series 转为 NumPy 数组。"""
    return series.to_numpy(copy=False) if hasattr(series, "to_numpy") else np.asarray(series)


def safe_round(value, ndigits=4):
    """对数值执行四舍五入转换。"""
    return round(value, ndigits)


##############################################
# 信号生成及回测函数
##############################################
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


@njit
def calculate_max_sequence_numba(series):
    """
    利用 numba 加速计算连续亏损（最小累计收益）及对应的交易数量与区间。
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
    if indices.size < 100:
        return None
    return (indices.astype(np.int32), p_np[indices])


def get_detail_backtest_result_op_simple(df, kai_column, pin_column, is_filter=True, is_reverse=False):
    """
    优化后的函数：提前进行部分判断以避免后续不必要的计算。
    """
    # --- 1. 获取信号数据 ---
    try:
        kai_idx, kai_prices = GLOBAL_SIGNALS[kai_column]
        pin_idx, pin_prices = GLOBAL_SIGNALS[pin_column]
    except KeyError:
        kai_idx, kai_prices = op_signal(df, kai_column)
        pin_idx, pin_prices = op_signal(df, pin_column)

    if (kai_idx is None or pin_idx is None or kai_idx.size < 100 or pin_idx.size < 100):
        return None, None

    # --- 2. 提取子数据集并匹配信号 ---
    kai_data_df = df.iloc[kai_idx].copy()
    pin_data_df = df.iloc[pin_idx].copy()
    kai_data_df["kai_price"] = kai_prices
    pin_data_df["pin_price"] = pin_prices

    kai_idx_arr = np.asarray(kai_data_df.index)
    pin_idx_arr = np.asarray(pin_data_df.index)
    pin_match_indices = np.searchsorted(pin_idx_arr, kai_idx_arr, side="right")
    valid_mask = pin_match_indices < len(pin_idx_arr)
    if valid_mask.sum() == 0:
        return None, None

    kai_data_df = kai_data_df.iloc[valid_mask].copy()
    kai_idx_valid = kai_idx_arr[valid_mask]
    pin_match_indices_valid = pin_match_indices[valid_mask]
    matched_pin = pin_data_df.iloc[pin_match_indices_valid].copy()

    # 添加匹配的 pin 数据
    kai_data_df["pin_price"] = matched_pin["pin_price"].values
    kai_data_df["pin_time"] = matched_pin["timestamp"].values
    kai_data_df["hold_time"] = matched_pin.index.values - kai_idx_valid

    if is_filter:
        kai_data_df = kai_data_df.sort_values("timestamp").drop_duplicates("pin_time", keep="first")

    trade_count = len(kai_data_df)
    if trade_count < 25:
        return None, None

    # --- 3. 策略方向判断和初步盈亏计算 ---
    is_long = (("long" in kai_column.lower()) if not is_reverse else ("short" in kai_column.lower()))
    # 使用价格映射更新 kai_price（耗时较低）
    pin_price_map = kai_data_df.set_index("pin_time")["pin_price"]
    mapped_prices = kai_data_df["timestamp"].map(pin_price_map)
    if mapped_prices.notna().sum() > 0:
        kai_data_df["kai_price"] = mapped_prices.combine_first(kai_data_df["kai_price"])

    if is_long:
        profit_series = ((kai_data_df["pin_price"] - kai_data_df["kai_price"]) /
                         kai_data_df["kai_price"] * 100).round(4)
    else:
        profit_series = ((kai_data_df["kai_price"] - kai_data_df["pin_price"]) /
                         kai_data_df["kai_price"] * 100).round(4)
    kai_data_df["true_profit"] = profit_series - 0.07

    # --- 4. 初步检测净盈亏率 ---
    fix_profit = safe_round(kai_data_df[mapped_prices.notna()]["true_profit"].sum(), ndigits=4)
    net_profit_rate = kai_data_df["true_profit"].sum() - fix_profit
    if net_profit_rate < 25:
        return None, None

    # --- 5. 快速判断：检查平均收益和持有时间 ---
    true_profit_mean = kai_data_df["true_profit"].mean() * 100 if trade_count > 0 else 0
    hold_time_mean = kai_data_df["hold_time"].mean() if trade_count else 0
    max_hold_time = kai_data_df["hold_time"].max() if trade_count else 0

    if true_profit_mean < 10 or max_hold_time > 10000 or hold_time_mean > 3000:
        return None, None

    # --- 6. 耗时操作：计算最大连续亏损 ---
    profits_arr = kai_data_df["true_profit"].values
    max_loss, max_loss_start_idx, max_loss_end_idx, _ = calculate_max_sequence_numba(profits_arr)
    if max_loss < -30:
        return None, None

    # --- 7. 月度和周度统计，判断活跃情况和亏损比例 ---
    full_start_time = df["timestamp"].min()
    full_end_time = df["timestamp"].max()

    # 月度统计
    monthly_groups = kai_data_df["timestamp"].dt.to_period("M")
    monthly_agg = kai_data_df.groupby(monthly_groups)["true_profit"].sum()
    active_months = monthly_agg.shape[0]
    total_months = len(pd.period_range(start=full_start_time.to_period("M"),
                                       end=full_end_time.to_period("M"),
                                       freq="M"))
    active_month_ratio = active_months / total_months if total_months else 0
    monthly_loss_rate = (np.sum(monthly_agg < 0) / active_months) if active_months else 0
    if active_month_ratio < 0.5 or monthly_loss_rate > 0.3:
        return None, None

    # 周度统计
    weekly_groups = kai_data_df["timestamp"].dt.to_period("W")
    weekly_agg = kai_data_df.groupby(weekly_groups)["true_profit"].sum()
    active_weeks = weekly_agg.shape[0]
    total_weeks = len(pd.period_range(start=full_start_time.to_period("W"),
                                      end=full_end_time.to_period("W"),
                                      freq="W"))
    active_week_ratio = active_weeks / total_weeks if total_weeks else 0
    weekly_loss_rate = (np.sum(weekly_agg < 0) / active_weeks) if active_weeks else 0
    if active_week_ratio < 0.5 or weekly_loss_rate > 0.3:
        return None, None

    # --- 8. 判断前10%盈利贡献比例 ---
    profit_df = kai_data_df[kai_data_df["true_profit"] > 0]
    if not profit_df.empty:
        top_profit_count = max(1, int(np.ceil(len(profit_df) * 0.1)))
        profit_sorted = profit_df.sort_values("true_profit", ascending=False)
        top_profit_sum = profit_sorted["true_profit"].iloc[:top_profit_count].sum()
        total_profit_sum = profit_df["true_profit"].sum()
        top_profit_ratio = top_profit_sum / total_profit_sum if total_profit_sum != 0 else 0
    else:
        top_profit_ratio = 0

    if top_profit_ratio > 0.5:
        return None, None

    # --- 9. 构造结果字典 ---
    statistic_dict = {
        "kai_column": kai_column,
        "pin_column": pin_column,
        "kai_count": trade_count,
        "net_profit_rate": net_profit_rate,
        "max_consecutive_loss": max_loss,
        "active_week_ratio": active_week_ratio,
        "active_month_ratio": active_month_ratio,
        "avg_profit_rate": true_profit_mean,
        "hold_time_mean": hold_time_mean,
        "max_hold_time": max_hold_time,
        "top_profit_ratio": top_profit_ratio,
        "monthly_loss_rate": monthly_loss_rate,
        "weekly_loss_rate": weekly_loss_rate,
        "is_reverse": is_reverse,
    }

    return None, statistic_dict


##############################################
# 信号生成名称函数
##############################################
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
    _, stat = get_detail_backtest_result_op_simple(df, long_sig, short_sig, is_filter=True, is_reverse=False)
    if stat is None:
        return None
    return stat


def brute_force_backtesting(df, long_signals, short_signals, batch_size=100000, key_name="brute_force_results"):
    """
    穷举遍历所有 (长信号, 短信号) 组合，每 batch_size 个组合为一个批次，
    使用多进程并行计算每个候选组合的回测结果，保存每个批次的有效回测结果到文件。
    同时累积所有有效组合并在结束时返回结果列表。

    主要改进：
      1. 在整个遍历过程中只创建一次进程池，避免每个批次中频繁创建/销毁进程导致的长尾问题。
      2. 根据当前批次任务数量动态设置 chunksize，降低进程间切换的消耗。
    """
    total_pairs = len(long_signals) * len(short_signals)
    print(f"候选组合总数: {total_pairs} 预计批次数: {total_pairs // batch_size + 1}")
    candidate_gen = candidate_pairs_generator(long_signals, short_signals)
    all_valid_results = []
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
            stats_file_name = os.path.join(checkpoint_dir, f"{key_name}_{batch_index}_{IS_REVERSE}_stats.parquet")
            if os.path.exists(stats_file_name):
                print(f"批次 {batch_index} 的统计文件已存在，跳过。")
                batch_index += 1
                continue
            # 动态设置较大的 chunksize 以减少调度次数
            # results = pool.map(evaluate_candidate, batch, chunksize=chunk_size)
            results = pool.imap_unordered(evaluate_candidate, batch, chunksize=chunk_size)
            valid_results = [res for res in results if res is not None]
            all_valid_results.extend(valid_results)
            temp_df = pd.DataFrame(valid_results)
            temp_df.to_parquet(stats_file_name, index=False)
            print(
                f"批次 {batch_index} 处理完毕，有效组合数量: {len(valid_results)} 耗时 {time.time() - start_time:.2f} 秒 当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            batch_index += 1
    finally:
        pool.close()
        pool.join()

    return all_valid_results


##############################################
# 主流程：直接对所有候选组合进行回测
##############################################
def brute_force_optimize_breakthrough_signal(data_path="temp/TON_1m_2000.csv"):
    """
    加载数据后直接对所有 (长信号, 短信号) 组合进行回测，
    每一定数量的组合为一个批次，保存回测结果。
    """
    os.makedirs("temp", exist_ok=True)
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

        global df
        df = temp_df_local.copy()
        # 预计算所有候选信号数据
        precomputed = load_or_compute_precomputed_signals(df, all_signals, f'{year}_{base_name}_{key_name}')
        total_size = sys.getsizeof(precomputed) + sum(
            sys.getsizeof(sig) + s.nbytes + p.nbytes for sig, (s, p) in precomputed.items())
        print(f"预计算信号内存大小: {total_size / (1024 * 1024):.2f} MB")

        global GLOBAL_SIGNALS
        GLOBAL_SIGNALS = precomputed
        # 本例中长信号与短信号均取预计算结果中的所有信号
        candidate_signals = list(GLOBAL_SIGNALS.keys())
        print(f"候选信号数量: {len(candidate_signals)}。")

        # 穷举回测所有候选组合，每一批次计算并保存结果
        valid_results = brute_force_backtesting(df, candidate_signals, candidate_signals, batch_size=10000000,
                                                key_name=f'{year}_{base_name}_{key_name}')
        print(f"\n暴力回测结束，共找到 {len(valid_results)} 个有效信号组合。")



##############################################
# 示例入口函数
##############################################
def example():
    """
    示例入口：处理多个数据文件调用暴力穷举回测流程。
    """
    start_time = time.time()
    data_path_list = [
        "kline_data/origin_data_1m_5000000_BTC-USDT-SWAP_2025-05-06.csv",
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