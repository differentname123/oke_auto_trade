#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
该脚本的主要任务是对候选信号对进行策略回测，并生成统计指标。流程主要包括：

候选信号对提取
从输入的 stat_df 文件中读取 “kai_column” 和 “pin_column” 两列，形成需要计算回测结果的信号对列表。

预计算交易信号
根据所有涉及到的信号名称，从行情数据（CSV 格式）中使用多种技术指标（如 abs、relate、donchian、boll、macross、rsi、macd、cci、atr 等）计算相应的交易信号和目标价格，并将预计算结果保存到本地目录（temp_back）。

回测模拟与统计计算
利用预计算的信号数据，采用多进程分批次模拟交易（根据信号匹配方式确定开仓和平仓），计算每一对信号对应的交易统计指标（如交易次数、持有时间、盈亏、连续亏损/盈利、月度/周度统计、最佳杠杆等）。

结果合并与存储
将所有批次中有效的信号对的回测统计结果（statistic_dict）合并成一个 DataFrame，并以 parquet 格式保存在 temp_back 目录下，便于后续的汇总和分析。

总体来说，该脚本实现了候选信号对的回测流程，通过预计算、并行批次处理和详细的统计指标计算，为策略评价和筛选提供了数据支撑。
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



def find_optimal_leverage(df):
    """
    根据df中每笔交易的百分比收益（字段 true_profit），返回三个值：
    1. 最佳整数杠杆
    2. 在该杠杆下的累计收益率（假设初始本金为1）
    3. 不加杠杆时的累计收益率（即每笔交易收益率直接相乘）

    其中：
      - 当按杠杆计算时，每笔交易的本金变化为：
          capital *= 1 + (true_profit/100) * leverage
      - 为确保安全，要求任一交易后本金大于0。
      - 自适应确定最大可用杠杆：根据所有交易中最亏的一笔计算允许的最大杠杆，
        即满足：1 + (true_profit/100)*杠杆 > 0。
    """

    # 1. 计算不加杠杆的累计收益率
    capital_no_leverage = 1.0
    for profit in df['true_profit']:
        capital_no_leverage *= (1 + profit / 100)

    # 2. 自适应确定最大可用杠杆
    min_profit = df['true_profit'].min()  # 取最小值（亏损最大的那一笔，注意亏损值为负）
    if min_profit >= 0:
        # 如果没有亏损，则理论上可用杠杆无限大，此处设定一个默认上限
        max_possible_leverage = 10
    else:
        # 对于亏损交易，需要保证：1 + (min_profit/100)*L > 0
        # 解得 L < 1 / (abs(min_profit)/100)
        max_possible_leverage = int(1 / (abs(min_profit) / 100))

    # 3. 遍历寻找最佳整数杠杆
    optimal_leverage = None
    optimal_capital = -float('inf')

    for L in range(1, max_possible_leverage + 1):
        current_capital = 1.0
        safe = True
        for profit in df['true_profit']:
            factor = 1 + (profit / 100) * L
            # 如果这笔交易导致资本为0或负数，则杠杆L不安全
            if factor <= 0:
                safe = False
                break
            current_capital *= factor
        # 在安全的杠杆下，选择累计收益率最高的杠杆
        if safe and current_capital > optimal_capital:
            optimal_capital = current_capital
            optimal_leverage = L

    return optimal_leverage, optimal_capital, capital_no_leverage

def get_detail_backtest_result_op(df, kai_column, pin_column, is_filter=True, is_detail=False, is_reverse=False):
    """
    根据预计算的稀疏信号数据获取回测数据和统计指标。
    返回：
      - kai_data_df：含持有时间、真实盈亏的 DataFrame。
      - statistic_dict：统计指标字典。
    """
    global GLOBAL_SIGNALS

    try:
        kai_idx, kai_prices = GLOBAL_SIGNALS[kai_column]
        pin_idx, pin_prices = GLOBAL_SIGNALS[pin_column]
    except KeyError:
        kai_idx, kai_prices = op_signal(df, kai_column)
        pin_idx, pin_prices = op_signal(df, pin_column)

    # 根据信号索引提取子数据集
    kai_data_df = df.iloc[kai_idx].copy()
    pin_data_df = df.iloc[pin_idx].copy()
    kai_data_df["kai_price"] = kai_prices
    pin_data_df["pin_price"] = pin_prices

    # 取出开仓信号和平仓信号的索引（假定 DataFrame 的索引已经排序）
    kai_idx_sorted = list(kai_data_df.index)
    pin_idx_sorted = list(pin_data_df.index)

    matched_pairs = []  # 存放匹配的交易对，形式为 (entry_idx, exit_idx)
    last_exit = -float('inf')  # 上一笔交易的平仓信号索引
    pin_ptr = 0  # 平仓信号的指针

    for kai_val in kai_idx_sorted:
        # 如果当前开仓信号小于等于上一笔交易的平仓信号，则跳过
        if kai_val <= last_exit:
            continue

        # 移动平仓信号指针，直到找到严格大于 kai_val 的平仓信号
        while pin_ptr < len(pin_idx_sorted) and pin_idx_sorted[pin_ptr] <= kai_val:
            pin_ptr += 1

        # 如果没有找到合适的平仓信号则退出
        if pin_ptr >= len(pin_idx_sorted):
            break

        exit_val = pin_idx_sorted[pin_ptr]
        matched_pairs.append((kai_val, exit_val))
        last_exit = exit_val  # 更新最后一笔交易的平仓信号
        pin_ptr += 1  # 使用过的平仓信号后移，确保不再复用

    # 如果匹配不到交易，则返回 None
    if not matched_pairs:
        return None, None

    # 构造最终的 kai_data_df，只保留匹配到的开仓信号数据
    matched_kai_idx = [pair[0] for pair in matched_pairs]
    kai_data_df = kai_data_df.loc[matched_kai_idx].copy()

    # 提取匹配的平仓信号数据
    matched_pin_idx = [pair[1] for pair in matched_pairs]
    matched_pin = pin_data_df.loc[matched_pin_idx].copy()

    # 将匹配的平仓数据加入到开仓数据中
    kai_data_df["pin_price"] = matched_pin["pin_price"].values
    kai_data_df["pin_time"] = matched_pin["timestamp"].values
    # 这里假设索引即代表时间，持有时间为平仓索引减去开仓索引
    kai_data_df["hold_time"] = np.array(matched_pin_idx) - np.array(matched_kai_idx)
    ############################################################################

    # 如果需过滤重复的平仓时间记录（is_filter==True），则进行排序去重
    if is_filter:
        kai_data_df = kai_data_df.sort_values("timestamp").drop_duplicates("pin_time", keep="first")
    trade_count = len(kai_data_df)

    # 使用 pin_time 的价格映射 kai_price，如果存在更新价格
    pin_price_map = kai_data_df.set_index("pin_time")["pin_price"]
    mapped_prices = kai_data_df["timestamp"].map(pin_price_map)
    if mapped_prices.notna().sum() > 0:
        kai_data_df["kai_price"] = mapped_prices.combine_first(kai_data_df["kai_price"])
    modification_rate = (100 * mapped_prices.notna().sum() / trade_count) if trade_count else 0

    # 根据传入的参数判断做多或做空策略，计算盈亏比例（百分比）并扣除交易成本
    is_long = (("long" in kai_column.lower()) if not is_reverse else ("short" in kai_column.lower()))
    if is_long:
        profit_series = ((kai_data_df["pin_price"] - kai_data_df["kai_price"]) /
                         kai_data_df["kai_price"] * 100).round(4)
    else:
        profit_series = ((kai_data_df["kai_price"] - kai_data_df["pin_price"]) /
                         kai_data_df["kai_price"] * 100).round(4)
    kai_data_df["profit"] = profit_series
    kai_data_df["true_profit"] = profit_series - 0.07  # 扣除交易成本
    profit_sum = profit_series.sum()
    max_single_profit = kai_data_df["true_profit"].max()
    min_single_profit = kai_data_df["true_profit"].min()

    true_profit_std = kai_data_df["true_profit"].std()
    true_profit_mean = kai_data_df["true_profit"].mean() * 100 if trade_count > 0 else 0
    fix_profit = safe_round(kai_data_df[mapped_prices.notna()]["true_profit"].sum(), ndigits=4)
    net_profit_rate = kai_data_df["true_profit"].sum() - fix_profit

    # 计算连续盈利或亏损序列
    # calculate_max_sequence_numba 返回： (max_loss, max_loss_start_idx, max_loss_end_idx, loss_trade_count)
    profits_arr = kai_data_df["true_profit"].values
    max_loss, max_loss_start_idx, max_loss_end_idx, loss_trade_count = calculate_max_sequence_numba(profits_arr)
    if max_loss_start_idx < len(kai_data_df) and max_loss_end_idx < len(kai_data_df):
        max_loss_hold_time = kai_data_df.index[max_loss_end_idx] - kai_data_df.index[max_loss_start_idx]
        max_loss_start_time = kai_data_df["timestamp"].iloc[max_loss_start_idx]
        max_loss_end_time = kai_data_df["timestamp"].iloc[max_loss_end_idx]
    else:
        max_loss_hold_time = None
        max_loss_start_time = None
        max_loss_end_time = None

    if max_loss_start_idx < len(kai_data_df) and max_loss_end_idx < len(kai_data_df):
        max_profit, max_profit_start_idx, max_profit_end_idx, profit_trade_count = calculate_max_profit_numba(profits_arr)
        max_profit_hold_time = kai_data_df.index[max_profit_end_idx] - kai_data_df.index[max_profit_start_idx]
    else:
        max_profit, max_profit_start_idx, max_profit_end_idx, profit_trade_count = None, None, None, None
        max_profit_hold_time = None

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

    hold_time_mean = kai_data_df["hold_time"].mean() if trade_count else 0
    max_hold_time = kai_data_df["hold_time"].max() if trade_count else 0

    # 使用整个原始 df 的最早和最晚时间作为统计范围
    full_start_time = df["timestamp"].min()
    full_end_time = df["timestamp"].max()

    # --------------------- 月度统计 ---------------------
    monthly_groups = kai_data_df["timestamp"].dt.to_period("M")
    monthly_agg = kai_data_df.groupby(monthly_groups)["true_profit"].agg(["sum", "mean", "count"])
    monthly_trade_std = monthly_agg["count"].std() if "count" in monthly_agg else 0
    active_months = monthly_agg.shape[0]

    start_month_all = full_start_time.to_period("M")
    end_month_all = full_end_time.to_period("M")
    all_months = pd.period_range(start=start_month_all, end=end_month_all, freq="M")
    total_months = len(all_months)
    active_month_ratio = active_months / total_months if total_months else 0

    monthly_net_profit_std = monthly_agg["sum"].std() if "sum" in monthly_agg else 0
    monthly_avg_profit_std = monthly_agg["mean"].std() if "mean" in monthly_agg else 0
    monthly_net_profit_min = monthly_agg["sum"].min() if "sum" in monthly_agg else 0
    monthly_net_profit_max = monthly_agg["sum"].max() if "sum" in monthly_agg else 0
    monthly_loss_rate = ((monthly_agg["sum"] < 0).sum() / active_months) if active_months else 0

    monthly_count_series = monthly_agg["count"].reindex(all_months, fill_value=0)
    monthly_kai_count_detail = monthly_count_series.values
    monthly_kai_count_std = monthly_count_series.std()

    monthly_net_profit_series = monthly_agg["sum"].reindex(all_months, fill_value=0)
    monthly_net_profit_detail = monthly_net_profit_series.round(4).values

    # --------------------- 周度统计 ---------------------
    weekly_groups = kai_data_df["timestamp"].dt.to_period("W")
    weekly_agg = kai_data_df.groupby(weekly_groups)["true_profit"].agg(["sum", "mean", "count"])
    weekly_trade_std = weekly_agg["count"].std() if "count" in weekly_agg else 0
    active_weeks = weekly_agg.shape[0]

    start_week_all = full_start_time.to_period("W")
    end_week_all = full_end_time.to_period("W")
    all_weeks = pd.period_range(start=start_week_all, end=end_week_all, freq="W")
    total_weeks = len(all_weeks)
    active_week_ratio = active_weeks / total_weeks if total_weeks else 0

    weekly_net_profit_std = weekly_agg["sum"].std() if "sum" in weekly_agg else 0
    weekly_avg_profit_std = weekly_agg["mean"].std() if "mean" in weekly_agg else 0
    weekly_net_profit_min = weekly_agg["sum"].min() if "sum" in weekly_agg else 0
    weekly_net_profit_max = weekly_agg["sum"].max() if "sum" in weekly_agg else 0
    weekly_loss_rate = ((weekly_agg["sum"] < 0).sum() / active_weeks) if active_weeks else 0

    weekly_count_series = weekly_agg["count"].reindex(all_weeks, fill_value=0)
    weekly_kai_count_detail = weekly_count_series.values
    weekly_kai_count_std = weekly_count_series.std()

    weekly_net_profit_series = weekly_agg["sum"].reindex(all_weeks, fill_value=0)
    weekly_net_profit_detail = weekly_net_profit_series.round(4).values

    hold_time_std = kai_data_df["hold_time"].std()

    # 统计 top 10% 盈利和亏损的比率
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

    common_index = kai_data_df.index.intersection(pin_data_df.index)
    same_count_rate = safe_round(
        100 * len(common_index) / min(len(kai_data_df), len(pin_data_df)) if trade_count else 0, 4)

    optimal_leverage, optimal_capital, capital_no_leverage = find_optimal_leverage(kai_data_df)

    statistic_dict = {
        "kai_column": kai_column,
        "pin_column": pin_column,
        "kai_count": trade_count,
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
        # 新增最大亏损的开始和结束时间
        "max_loss_start_time": max_loss_start_time,
        "max_loss_end_time": max_loss_end_time,
        "max_consecutive_profit": safe_round(max_profit, 4) if max_profit is not None else None,
        "max_profit_trade_count": profit_trade_count if max_profit is not None else None,
        "max_profit_hold_time": max_profit_hold_time,
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
        "monthly_kai_count_detail": monthly_kai_count_detail,
        "monthly_kai_count_std": safe_round(monthly_kai_count_std, 4),
        "monthly_net_profit_detail": monthly_net_profit_detail,
        "weekly_trade_std": safe_round(weekly_trade_std, 4),
        "active_week_ratio": safe_round(active_week_ratio, 4),
        "weekly_loss_rate": safe_round(weekly_loss_rate, 4),
        "weekly_net_profit_min": safe_round(weekly_net_profit_min, 4),
        "weekly_net_profit_max": safe_round(weekly_net_profit_max, 4),
        "weekly_net_profit_std": safe_round(weekly_net_profit_std, 4),
        "weekly_avg_profit_std": safe_round(weekly_avg_profit_std, 4),
        "weekly_net_profit_detail": weekly_net_profit_detail,
        "weekly_kai_count_detail": weekly_kai_count_detail,
        "weekly_kai_count_std": weekly_kai_count_std,
        "top_profit_ratio": safe_round(top_profit_ratio, 4),
        "top_loss_ratio": safe_round(top_loss_ratio, 4),
        "optimal_leverage": optimal_leverage,
        "optimal_capital": safe_round(optimal_capital, 4),
        "capital_no_leverage": safe_round(capital_no_leverage, 4),
        "is_reverse": is_reverse,
    }
    return None, statistic_dict


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
                                                is_reverse=True)
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

def get_property_name(inst_id, is_reverse):
    year_list = [20231,20232,20241,20242,20251]
    for year in year_list:
        file_path = f'temp_back/all_files_{year}_1m_5000000_{inst_id}_short_donchian_1_20_1_relate_400_1000_100_1_40_6_cci_1_2000_1000_1_2_1_atr_1_3000_3000_boll_1_3000_100_1_50_2_rsi_1_1000_500_abs_1_100_100_40_100_1_macd_300_1000_50_macross_1_3000_100_1_3000_100__{is_reverse}.parquet'
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            if len(df) < 200000:
                return file_path
    return file_path

def validation(market_data_file):
    base_name = os.path.basename(market_data_file)
    base_name = base_name.replace("-USDT-SWAP.csv", "").replace("origin_data_", "")
    inst_id_list = ['BTC', 'ETH', 'SOL', 'TON', 'DOGE', 'XRP', 'OKB']
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
    is_reverse = True

    # 3. 设置主进程全局变量，用于预计算
    global df
    df = df_local

    # 注意：这里只读取需要的列： kai_column 和 pin_column
    stat_df_file_list = [get_property_name(inst_id, is_reverse)]
    for stat_df_file in stat_df_file_list:
        try:
            # 1. 加载 stat_df 文件（只读取必要的两列）
            stat_df = pd.read_parquet(stat_df_file, columns=["kai_column", "pin_column"])
            # 去重
            stat_df = stat_df.drop_duplicates(subset=["kai_column", "pin_column"])
            print(f"Loaded stat_df: 共 {stat_df.shape[0]} 行 {stat_df_file}")
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
            max_memory = 40  # 单位：GB
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
                output_file = os.path.join("temp_back", f"{inst_id}_{is_reverse}_{batch_index}.parquet")
                # if os.path.exists(output_file):
                #     print(f"{output_file} 已存在，跳过处理。")
                #     batch_output_files.append(output_file)
                #     continue
                end_idx = min(start_idx + BATCH_SIZE, total_pairs)
                batch_pairs = pairs[start_idx:end_idx]
                print(f"Processing batch {batch_index + 1}/{total_batches} with {len(batch_pairs)} pairs... [{current_time}]")

                with multiprocessing.Pool(processes=pool_processes,
                                          initializer=init_worker_with_signals,
                                          initargs=(GLOBAL_SIGNALS, df)) as pool:
                    results = pool.map(process_signal_pair, batch_pairs, chunksize=10)

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
                final_output_file = os.path.join("temp_back", f"statistic_results_final_{inst_id}_{is_reverse}.parquet")
                merged_df.to_parquet(final_output_file, index=False, compression='snappy')
                print(f"所有批次结果已合并，并保存到 {final_output_file} (共 {merged_df.shape[0]} 行)。")
            else:
                print("没有找到任何批次结果文件，无法合并。")

        except Exception as e:
            traceback.print_exc()
            print(f"处理 {stat_df_file} 时出错：{e}")




if __name__ == "__main__":
    start_time = time.time()
    data_path_list = [
        "kline_data/origin_data_1m_5000000_ETH-USDT-SWAP_2025-05-06.csv",
        "kline_data/origin_data_1m_5000000_BTC-USDT-SWAP_2025-05-06.csv",
        "kline_data/origin_data_1m_5000000_SOL-USDT-SWAP_2025-05-06.csv",
        "kline_data/origin_data_1m_5000000_TON-USDT-SWAP_2025-05-06.csv",
        "kline_data/origin_data_1m_5000000_DOGE-USDT-SWAP_2025-05-06.csv",
        "kline_data/origin_data_1m_5000000_XRP-USDT-SWAP_2025-05-06.csv",
        "kline_data/origin_data_1m_5000000_OKB-USDT_2025-05-06.csv",
    ]
    for data_path in data_path_list:
        try:
            validation(data_path)
            print(f"{data_path} 总耗时 {time.time() - start_time:.2f} 秒。")
        except Exception as e:
            traceback.print_exc()
            print(f"处理 {data_path} 时出错：{e}")