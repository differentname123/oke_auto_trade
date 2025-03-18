"""
突破策略信号生成及回测代码 —— 基于遗传算法的启发式搜索版 (优化后)

优化说明：
  1. 利用批次方式进行多进程计算种群候选个体（减少进程切换的开销）。
  2. 增加断点续跑功能，生成 checkpoint 文件，可在中断后继续运行。
  3. 在遗传算法中注入多样性（动态增加随机新个体比例）以避免陷入局部最优，
     并在连续多代无改进时执行局部重启机制。
  4. 对锦标赛选择函数引入一定的随机性，降低选择压力，保留更多个体多样性。
  5. 引入自适应变异率机制，当连续多代无改进时提高变异率，增加探索可能。
  6. 引入岛屿模型（Island Model）：将种群分割成多个子种群（岛），每个岛独立进化，
     并在固定代数后进行个体迁移，从而增强种群多样性，避免局部最优。

说明：
  1. 预计算所有候选信号（GLOBAL_SIGNALS）以提高后续回测速度，并保存至 temp 目录，方便下次加载。
  2. 记录遗传算法历史统计（stats），即使后续步骤只关注最优组合，此处信息能用于进一步分析。
"""

import os
import sys
import time
import pickle
import random
import traceback
from itertools import product

import numpy as np
import pandas as pd
from numba import njit
import multiprocessing

# 全局变量，用于存储预计算信号数据和行情数据
GLOBAL_SIGNALS = {}
df = None  # 回测数据，子进程中通过初始化传入

##############################################
# 辅助函数
##############################################

def series_to_numpy(series):
    """
    将 Pandas Series 转为 NumPy 数组。如果 Series 有 to_numpy 方法则直接调用。
    """
    return series.to_numpy(copy=False) if hasattr(series, "to_numpy") else np.asarray(series)


def safe_round(value, ndigits=4):
    """
    对数值执行四舍五入转换，便于后续比较与打印。
    """
    return round(value, ndigits)


##############################################
# 信号生成及回测函数
##############################################

def compute_signal(df, col_name):
    """
    根据历史行情数据(df)和指定信号名称(col_name)，生成交易信号和对应目标价格。
    支持的信号类型包括：abs, relate, donchian, boll, macross, rsi, macd, cci, atr。
    """
    parts = col_name.split("_")
    signal_type = parts[0]
    direction = parts[-1]

    # abs 信号
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

        # 可选调试数据记录
        df["target_price"] = target_price
        df["signal_series"] = signal_series
        df["trade_price_series"] = trade_price_series
        return signal_series, trade_price_series

    # relate 信号
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

    # donchian 信号
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

    # boll 信号
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

    # macross 信号
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

    # rsi 信号
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

    # macd 信号
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

    # cci 信号
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

    # atr 信号
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
    采用 numba 加速，计算连续亏损（最小累计收益）及其对应的交易数量与区间。
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
    采用 numba 加速，计算连续盈利（最大累计收益）及其对应的交易数量与区间。
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


def get_detail_backtest_result_op(df, kai_column, pin_column, is_filter=True, is_detail=False, is_reverse=False):
    """
    根据预计算的稀疏信号数据，获取回测数据及统计指标。

    返回：
      - kai_data_df：包含持有时间、真实盈亏的 DataFrame（用于进一步分析）。
      - statistic_dict：统计指标字典。
    """
    global GLOBAL_SIGNALS

    # 尝试从 GLOBAL_SIGNALS 中读取预计算数据；否则调用 op_signal 计算
    try:
        kai_idx, kai_prices = GLOBAL_SIGNALS[kai_column]
        pin_idx, pin_prices = GLOBAL_SIGNALS[pin_column]
    except KeyError:
        kai_idx, kai_prices = op_signal(df, kai_column)
        pin_idx, pin_prices = op_signal(df, pin_column)

    # 若某一信号交易次数太少，则忽略
    if (kai_idx is None or pin_idx is None or
            kai_idx.size < 100 or pin_idx.size < 100):
        return None, None

    # 构造信号对应的 DataFrame（均拷贝一份以避免原数据被修改）
    kai_data_df = df.iloc[kai_idx].copy()
    pin_data_df = df.iloc[pin_idx].copy()
    kai_data_df["kai_price"] = kai_prices
    pin_data_df["pin_price"] = pin_prices

    # 根据时间索引找出匹配的信号时刻，注意有可能存在重复匹配
    common_index = kai_data_df.index.intersection(pin_data_df.index)
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

    # 统计交易次数与总体数据量
    trade_count = len(kai_data_df)
    total_count = len(df)

    # 若存在价格修正的情况，则对原始交易价格做映射
    pin_price_map = kai_data_df.set_index("pin_time")["pin_price"]
    mapped_prices = kai_data_df["timestamp"].map(pin_price_map)
    if mapped_prices.notna().sum() > 0:
        kai_data_df["kai_price"] = mapped_prices.combine_first(kai_data_df["kai_price"])
    modification_rate = (100 * mapped_prices.notna().sum() / trade_count) if trade_count else 0

    # 计算收益率
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
    # 利用 numba 计算连续亏损与连续盈利数据
    profits_arr = kai_data_df["true_profit"].values
    max_loss, max_loss_start_idx, max_loss_end_idx, loss_trade_count = calculate_max_sequence_numba(profits_arr)
    if max_loss < -20 or net_profit_rate < 50:
        # 若连续亏损太大，则舍弃该组合
        return None, None

    max_profit, max_profit_start_idx, max_profit_end_idx, profit_trade_count = calculate_max_profit_numba(profits_arr)

    # 获取亏损与盈利段的相关时间及持有周期
    if (trade_count > 0 and
            max_loss_start_idx < len(kai_data_df) and max_loss_end_idx < len(kai_data_df)):
        max_loss_start_time = kai_data_df.iloc[max_loss_start_idx]["timestamp"]
        max_loss_end_time = kai_data_df.iloc[max_loss_end_idx]["timestamp"]
        max_loss_hold_time = kai_data_df.index[max_loss_end_idx] - kai_data_df.index[max_loss_start_idx]
    else:
        max_loss_start_time = max_loss_end_time = max_loss_hold_time = None

    if (trade_count > 0 and
            max_profit_start_idx < len(kai_data_df) and max_profit_end_idx < len(kai_data_df)):
        max_profit_start_time = kai_data_df.iloc[max_profit_start_idx]["timestamp"]
        max_profit_end_time = kai_data_df.iloc[max_profit_end_idx]["timestamp"]
        max_profit_hold_time = kai_data_df.index[max_profit_end_idx] - kai_data_df.index[max_profit_start_idx]
    else:
        max_profit_start_time = max_profit_end_time = max_profit_hold_time = None

    # 分离盈利与亏损交易，计算各项汇总指标
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

    if hold_time_mean > 1000 or true_profit_mean < 10:
        return None, None

    # 计算月度统计指标
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

    hold_time_std = kai_data_df["hold_time"].std()

    # 计算盈利排行比例
    if not profit_df.empty:
        top_profit_count = max(1, int(np.ceil(len(profit_df) * 0.1)))
        profit_sorted = profit_df.sort_values("true_profit", ascending=False)
        top_profit_sum = profit_sorted["true_profit"].iloc[:top_profit_count].sum()
        total_profit_sum = profit_df["true_profit"].sum()
        top_profit_ratio = (top_profit_sum / total_profit_sum) if total_profit_sum != 0 else 0
    else:
        top_profit_ratio = 0

    # 计算亏损排行比例
    if not loss_df.empty:
        top_loss_count = max(1, int(np.ceil(len(loss_df) * 0.1)))
        loss_sorted = loss_df.sort_values("true_profit", ascending=True)
        top_loss_sum = loss_sorted["true_profit"].iloc[:top_loss_count].sum()
        total_loss_sum = loss_df["true_profit"].sum()
        top_loss_ratio = (abs(top_loss_sum) / abs(total_loss_sum)) if total_loss_sum != 0 else 0
    else:
        top_loss_ratio = 0

    # 整理统计结果字典
    statistic_dict = {
        "kai_side": "long" if is_long else "short",
        "kai_column": kai_column,
        "pin_column": pin_column,
        "kai_count": trade_count,
        "total_count": total_count,
        "trade_rate": safe_round(trade_rate, 4),
        "hold_time_mean": hold_time_mean,
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
        "same_count_rate": safe_round(100 * len(common_index) / min(len(kai_data_df), len(pin_data_df)), 4),
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
        "is_reverse": is_reverse
    }
    kai_data_df = kai_data_df[["hold_time", "true_profit"]]
    return kai_data_df, statistic_dict


##############################################
# 信号名称生成相关函数
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
        result = []
        for i in range(number):
            normalized_index = i / (number - 1) if number > 1 else 0
            value = start + (end - start) * (normalized_index ** 2)
            result.append(int(round(value)))
    # 去重
    final_result = []
    last_val = None
    for val in result:
        if start <= val <= end and val != last_val:
            final_result.append(val)
            last_val = val
    return final_result[:number]


def gen_abs_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    """
    生成 abs 信号的候选名称列表，分别产生长信号和短信号。
    """
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = range(start_period1, end_period1, step1)
    period_list1 = [x / 20 for x in period_list1]
    long_columns = [f"abs_{p}_{p1}_long" for p in period_list for p1 in period_list1 if p >= p1]
    short_columns = [f"abs_{p}_{p1}_short" for p in period_list for p1 in period_list1 if p >= p1]
    key_name = f"abs_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}"
    print(
        f"abs 一共生成 {len(long_columns)} 个长信号和 {len(short_columns)} 个短信号。参数: {start_period}, {end_period}, {step}, {start_period1}, {end_period1}, {step1}")
    return long_columns, short_columns, key_name


def gen_macd_signal_name(start_period, end_period, step):
    period_list = generate_numbers(start_period, end_period, step, even=False)

    # 选择合理的 signal_period 值（通常较小）
    signal_list = [9, 12, 15, 40]  # 预定义 MACD 计算常见的信号周期

    long_columns = [
        f"macd_{fast}_{slow}_{signal}_long"
        for fast in period_list
        for slow in period_list if slow > fast
        for signal in signal_list
    ]

    short_columns = [
        f"macd_{fast}_{slow}_{signal}_short"
        for fast in period_list
        for slow in period_list if slow > fast
        for signal in signal_list
    ]

    key_name = f'macd_{start_period}_{end_period}_{step}'
    print(f"MACD 生成了 {len(long_columns)} 个信号列名，优化后减少了无效组合。")

    return long_columns, short_columns, key_name


def gen_cci_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = range(start_period1, end_period1, step1)
    period_list1 = [x / 10 for x in period_list1]
    long_columns = [f"cci_{period}_{period1}_long"
                    for period in period_list for period1 in period_list1 if period >= period1]
    short_columns = [f"cci_{period}_{period1}_short"
                     for period in period_list for period1 in period_list1 if period >= period1]
    key_name = f'cci_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}'
    print(
        f"cci 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}, {start_period1}, {end_period1}, {step1}")
    return long_columns, short_columns, key_name


def gen_relate_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = range(start_period1, end_period1, step1)
    long_columns = [f"relate_{period}_{period1}_long"
                    for period in period_list for period1 in period_list1 if period >= period1]
    short_columns = [f"relate_{period}_{period1}_short"
                     for period in period_list for period1 in period_list1 if period >= period1]
    key_name = f'relate_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}'
    print(
        f"relate 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}, {start_period1}, {end_period1}, {step1}")
    return long_columns, short_columns, key_name


def gen_rsi_signal_name(start_period, end_period, step):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    temp_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    long_columns = [f"rsi_{period}_{overbought}_{100 - overbought}_long"
                    for period in period_list for overbought in temp_list]
    short_columns = [f"rsi_{period}_{overbought}_{100 - overbought}_short"
                     for period in period_list for overbought in temp_list]
    key_name = f'rsi_{start_period}_{end_period}_{step}'
    print(f"rsi 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}")
    return long_columns, short_columns, key_name


def gen_atr_signal_name(start_period, end_period, step):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    long_columns = [f"atr_{period}_long" for period in period_list]
    short_columns = [f"atr_{period}_short" for period in period_list]
    key_name = f'atr_{start_period}_{end_period}_{step}'
    print(f"atr 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}")
    return long_columns, short_columns, key_name


def gen_donchian_signal_name(start_period, end_period, step):
    period_list = range(start_period, end_period, step)
    long_columns = [f"donchian_{period}_long" for period in period_list]
    short_columns = [f"donchian_{period}_short" for period in period_list]
    key_name = f'donchian_{start_period}_{end_period}_{step}'
    print(f"donchian 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}")
    return long_columns, short_columns, key_name

def gen_boll_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = range(start_period1, end_period1, step1)
    period_list1 = [x / 10 for x in period_list1]
    long_columns = [f"boll_{period}_{period1}_long"
                    for period in period_list for period1 in period_list1 if period >= period1]
    short_columns = [f"boll_{period}_{period1}_short"
                     for period in period_list for period1 in period_list1 if period >= period1]
    key_name = f'boll_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}'
    print(f"boll 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}, {start_period1}, {end_period1}, {step1}")
    return long_columns, short_columns, key_name


def gen_macross_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = generate_numbers(start_period1, end_period1, step1, even=False)
    long_columns = [f"macross_{period}_{period1}_long"
                    for period in period_list for period1 in period_list1]
    short_columns = [f"macross_{period}_{period1}_short"
                     for period in period_list for period1 in period_list1]
    key_name = f'macross_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}'
    print(f"macross 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}, {start_period1}, {end_period1}, {step1}")
    return long_columns, short_columns, key_name


def generate_all_signals():
    """
    生成所有候选信号，目前仅基于部分信号（如 relate），后续可扩展其他类型。
    """
    column_list = []
    abs_long_cols, abs_short_cols, abs_key = gen_abs_signal_name(1, 100, 100, 40, 100, 1)
    column_list.append((abs_long_cols, abs_short_cols, abs_key))
    # 265M 373

    relate_long_columns, relate_short_columns, relate_key_name = gen_relate_signal_name(400, 1000, 100, 1, 40, 6)
    column_list.append((relate_long_columns, relate_short_columns, relate_key_name))
    # 397M 182

    donchian_long_columns, donchian_short_columns, donchian_key_name = gen_donchian_signal_name(1, 20, 1)
    column_list.append((donchian_long_columns, donchian_short_columns, donchian_key_name))
    # 52M -10000

    boll_long_columns, boll_short_columns, boll_key_name = gen_boll_signal_name(1, 3000, 100, 1, 50, 2)
    column_list.append((boll_long_columns, boll_short_columns, boll_key_name))
    # 564m 405

    macross_long_columns, macross_short_columns, macross_key_name = gen_macross_signal_name(1, 3000, 100, 1, 3000, 100)
    column_list.append((macross_long_columns, macross_short_columns, macross_key_name))
    # 308M  365

    rsi_long_columns, rsi_short_columns, rsi_key_name = gen_rsi_signal_name(1, 1000, 500)
    column_list.append((rsi_long_columns, rsi_short_columns, rsi_key_name))
    # # 369M


    macd_long_columns, macd_short_columns, macd_key_name = gen_macd_signal_name(300, 1000, 50)
    column_list.append((macd_long_columns, macd_short_columns, macd_key_name))
    # 367M

    cci_long_columns, cci_short_columns, cci_key_name = gen_cci_signal_name(1, 2000, 1000, 1, 2, 1)
    column_list.append((cci_long_columns, cci_short_columns, cci_key_name))
    # 154M

    atr_long_columns, atr_short_columns, atr_key_name = gen_atr_signal_name(1, 3000, 3000)
    column_list.append((atr_long_columns, atr_short_columns, atr_key_name))
    # 77.87M

    # 按长信号数量排序（目前只有一组，此处结构便于扩展）
    column_list = sorted(column_list, key=lambda x: len(x[0]))
    all_signals = []
    key_name = ""
    for long_columns, short_columns, temp_key in column_list:
        temp = long_columns + short_columns
        key_name += temp_key + "_"
        all_signals.extend(temp)
    return all_signals, key_name


def split_into_batches(signal_list, batch_size):
    """
    把信号列表分批，每批 batch_size 个信号
    """
    return [signal_list[i: i + batch_size] for i in range(0, len(signal_list), batch_size)]


##############################################
# 信号预计算及多进程工具函数
##############################################

def process_signal(sig):
    """
    计算单个信号的预计算数据。如果交易信号数不足 100，则返回 None。
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


def process_batch(sig_batch):
    """
    处理一批信号，返回该批次中成功预计算的信号数据。
    """
    batch_results = []
    for sig in sig_batch:
        res = process_signal(sig)
        if res is not None:
            batch_results.append(res)
    return batch_results


def compute_precomputed_for_batch(signals, df):
    """
    针对一批信号，使用多进程预计算各信号的交易数据。
    """
    num_workers = multiprocessing.cpu_count()
    sub_batches = [signals[k: k + 10] for k in range(0, len(signals), 10)]
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker1, initargs=(df,)) as pool:
        sub_results = pool.map(process_batch, sub_batches)
    precomputed = {}
    for sub_res in sub_results:
        for sig, data in sub_res:
            precomputed[sig] = data
    return precomputed


def load_or_compute_batch(batch_index, signals, df, base_name, key_name):
    """
    尝试加载批次预计算结果，若失败则重新计算并保存。
    """
    batch_file = os.path.join("temp", f"{base_name}_{key_name}_batch_{batch_index}.pkl")
    if os.path.exists(batch_file):
        try:
            with open(batch_file, "rb") as f:
                precomputed = pickle.load(f)
            print(f"加载批次 {batch_index} 的预计算结果，信号数：{len(precomputed)}。")
            return precomputed
        except Exception as e:
            print(f"加载批次 {batch_index} 失败：{e}")
    print(f"开始计算批次 {batch_index} 的预计算结果：共 {len(signals)} 个信号。")
    start = time.time()
    precomputed = compute_precomputed_for_batch(signals, df)
    elapsed = time.time() - start
    print(f"批次 {batch_index} 完成预计算，共 {len(precomputed)} 个信号，耗时 {elapsed:.2f} 秒。")
    try:
        with open(batch_file, "wb") as f:
            pickle.dump(precomputed, f)
        print(f"批次 {batch_index} 的预计算结果已保存至 {batch_file}。")
    except Exception as e:
        print(f"保存批次 {batch_index} 时出错：{e}")
    return precomputed


def get_precomputed(batch_index, batches, df, base_name, key_name, cache):
    """
    从 cache 获取当前批次预计算结果，不存在则加载或重新计算。
    """
    if batch_index not in cache:
        signals = batches[batch_index]
        cache[batch_index] = load_or_compute_batch(batch_index, signals, df, base_name, key_name)
    return cache[batch_index]


def init_worker(precomputed_signals):
    """
    进程池初始化函数，将预计算的信号数据加载至全局变量。
    """
    global GLOBAL_SIGNALS
    GLOBAL_SIGNALS = precomputed_signals


def init_worker1(dataframe):
    """
    子进程初始化函数，将 df 加载至全局变量，便于 compute_signal 调用。
    """
    global df
    df = dataframe


##############################################
# 遗传算法优化相关函数（引入岛屿模型）
##############################################

def precompute_signals(df, signals):
    """
    使用多进程预计算所有候选信号数据，返回 dict 格式：
      {signal_name: (indices, prices)}
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


def load_or_compute_precomputed_signals(df, signals, key_name):
    """
    尝试从 temp 目录加载预计算结果，若无或加载出错则重新计算并保存。
    """
    file_path = os.path.join("temp_back", f"precomputed_signals_{key_name}_{len(signals)}.pkl")
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                precomputed = pickle.load(f)
            print(f"加载预计算结果：从 {file_path} 加载，共 {len(precomputed)} 个信号。")
            return precomputed
        except Exception as e:
            print(f"加载预计算结果失败：{e}，将重新计算。")
    print("开始计算预计算信号 ...")
    precomputed = precompute_signals(df, signals)
    try:
        with open(file_path, "wb") as f:
            pickle.dump(precomputed, f)
        print(f"预计算结果已保存至：{file_path}")
    except Exception as e:
        print(f"保存预计算结果时出错：{e}")
    return precomputed


def init_worker_ga(signals, dataframe):
    """
    遗传算法中多进程池的初始化函数，将预计算数据和 df 同时加载到全局变量中。
    """
    global GLOBAL_SIGNALS, df
    GLOBAL_SIGNALS = signals
    df = dataframe



def get_fitness(stat):
    """
    从统计结果中提取适应度值。
    """
    if stat is None:
        return -10000
    else:
        net_profit_rate = stat.get("net_profit_rate", -10000)
        # monthly_net_profit_std = stat.get("monthly_net_profit_std", 10000)
        max_consecutive_loss = stat.get("max_consecutive_loss", -100000)
        # fitness = net_profit_rate
        # fitness = 1 - monthly_net_profit_std / (net_profit_rate) * 22
        fitness = -net_profit_rate * net_profit_rate / max_consecutive_loss
        return fitness

def evaluate_candidate_batch(candidates):
    """
    对一批候选个体进行评价，返回列表 [(fitness, candidate, stat), ...]。
    """
    batch_results = []
    for candidate in candidates:
        long_sig, short_sig = candidate
        _, stat = get_detail_backtest_result_op(df, long_sig, short_sig, is_filter=True, is_detail=False,
                                                is_reverse=False)
        fitness = get_fitness(stat)
        batch_results.append((fitness, candidate, stat))
    return batch_results


def tournament_selection(population, fitnesses, tournament_size=3, selection_pressure=0.75):
    """
    锦标赛选择：从种群中随机挑选 tournament_size 个个体，根据一定概率（selection_pressure）
    优先选择适应度最高的个体，否则随机选取其他个体，以保留多样性。
    """
    new_population = []
    pop_with_fit = list(zip(population, fitnesses))
    for _ in range(len(population)):
        competitors = random.sample(pop_with_fit, tournament_size)
        competitors.sort(key=lambda x: x[1], reverse=True)
        if random.random() < selection_pressure:
            winner = competitors[0]
        else:
            winner = random.choice(competitors[1:])
        new_population.append(winner[0])
    return new_population


def crossover(parent1, parent2, crossover_rate=0.8):
    """
    对两个父代个体进行交叉操作（交换部分基因）。
    若随机数大于交叉率则直接返回父代个体。
    """
    if random.random() < crossover_rate:
        if random.random() < 0.5:
            child1 = (parent2[0], parent1[1])
            child2 = (parent1[0], parent2[1])
        else:
            child1 = (parent1[0], parent2[1])
            child2 = (parent2[0], parent1[1])
        return child1, child2
    else:
        return parent1, parent2


def mutate(individual, mutation_rate, candidate_long_signals, candidate_short_signals):
    """
    对单个个体进行变异：以 mutation_rate 概率替换长信号或短信号。
    """
    long_gene, short_gene = individual
    if random.random() < mutation_rate:
        long_gene = random.choice(candidate_long_signals)
    if random.random() < mutation_rate:
        short_gene = random.choice(candidate_short_signals)
    return (long_gene, short_gene)


def filter_existing_individuals(candidate_list, global_generated_individuals):
    """
    过滤掉已经评价过的个体。
    """
    return [ind for ind in candidate_list if ind not in global_generated_individuals]

def get_unique_candidate(candidate_long_signals, candidate_short_signals,
                         global_generated_individuals, candidate_list, target_size):
    """
    辅助函数：补充 candidate_list 直至其长度达到 target_size。

    参数说明：
      - candidate_long_signals: 长信号候选集合（列表）。
      - candidate_short_signals: 短信号候选集合（列表）。
      - global_generated_individuals: 已评价的候选个体集合（全局集合）。
      - candidate_list: 现有候选个体列表（可能已部分填充）。
      - target_size: 期望候选个体列表的长度。

    在补充过程中，只有当生成的新候选个体既不在 global_generated_individuals
    中，也不在 candidate_list 中时，才将其添加到 candidate_list 中。

    返回：
      - 补充后的 candidate_list，当其长度达到 target_size 时返回。
    """
    while len(candidate_list) < target_size:
        candidate = (random.choice(candidate_long_signals), random.choice(candidate_short_signals))
        if candidate in global_generated_individuals or candidate in candidate_list:
            continue
        candidate_list.append(candidate)
    return candidate_list

def genetic_algorithm_optimization(df, candidate_long_signals, candidate_short_signals,
                                   population_size=50, generations=20,
                                   crossover_rate=0.8, mutation_rate=0.1, key_name="default",
                                   islands_count=4, migration_interval=10, migration_rate=0.1,
                                   restart_similarity_threshold=10):
    """
    利用遗传算法结合岛屿模型搜索净利率较高的 (长信号, 短信号) 组合。
    全局已评价个体集合 global_generated_individuals 仅在每个岛种群评价后更新，
    其生成的新候选个体通过 get_unique_candidate 辅助函数确保不重复，
    并且只在评价后统一更新全局集合记录。

    参数说明：
      - candidate_long_signals: 长信号候选集合。
      - candidate_short_signals: 短信号候选集合。
      - islands_count: 岛屿数量。
      - migration_interval: 每隔多少代进行一次迁移。
      - migration_rate: 迁移时迁出个体占岛内个体比例。
      - restart_similarity_threshold: 岛屿间相似性阈值，相似性低于此阈值则重启岛屿。
    """
    # 确保断点存储目录存在
    checkpoint_dir = "temp"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 使用候选集并集作为预计算信号的基础
    all_signals = list(set(candidate_long_signals + candidate_short_signals))
    print(f"开始预计算 GLOBAL_SIGNALS ... {key_name}")
    precomputed = load_or_compute_precomputed_signals(df, all_signals, key_name)
    total_size = sys.getsizeof(precomputed)
    for sig, (s_np, p_np) in precomputed.items():
        total_size += sys.getsizeof(sig) + s_np.nbytes + p_np.nbytes
    print(f"precomputed_signals 占用内存总大小: {total_size / (1024 * 1024):.2f} MB")

    global GLOBAL_SIGNALS
    GLOBAL_SIGNALS = precomputed
    print(f"预计算完成，总共有 {len(GLOBAL_SIGNALS)} 个信号数据。")

    # 更新候选集合：使用所有预计算信号作为候选
    candidate_long_signals = list(GLOBAL_SIGNALS.keys())
    candidate_short_signals = list(GLOBAL_SIGNALS.keys())

    # 定义全局已评价候选个体集合，用于判重
    global_generated_individuals = set()

    # 尝试加载断点续跑数据
    checkpoint_file = os.path.join(checkpoint_dir, f"{key_name}_ga_checkpoint.pkl")
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "rb") as f:
                checkpoint_data = pickle.load(f)
                if len(checkpoint_data) == 6:
                    start_gen, islands, overall_best, overall_best_fitness, all_history, global_generated_individuals = checkpoint_data
                else:
                    start_gen, islands, overall_best, overall_best_fitness, all_history = checkpoint_data
                    global_generated_individuals = set()
            print(
                f"加载断点续跑信息，恢复到第 {start_gen} 代。 全局最好组合: {overall_best}，净利率: {overall_best_fitness}")
        except Exception as e:
            print(f"加载断点文件失败：{e}，将从头开始。")
            start_gen = 0
            islands = []
            overall_best = None
            overall_best_fitness = -1e9
            all_history = []
            global_generated_individuals = set()
    else:
        start_gen = 0
        islands = []
        overall_best = None
        overall_best_fitness = -1e9
        all_history = []

    # 每个岛的种群规模
    island_pop_size = population_size // islands_count

    # 若未恢复断点则初始化各岛种群及状态
    if not islands:
        for i in range(islands_count):
            # 使用新的 get_unique_candidate 批量生成种群：candidate_list 初始为空，目标 size 为 island_pop_size
            pop = get_unique_candidate(candidate_long_signals, candidate_short_signals,
                                       global_generated_individuals, candidate_list=[], target_size=island_pop_size)
            island_state = {
                "population": pop,
                "best_candidate": None,
                "best_fitness": -1e9,
                "no_improve_count": 0,
                "adaptive_mutation_rate": mutation_rate,
            }
            islands.append(island_state)

    # 自适应及精英保留参数
    elite_fraction = 0.05  # 保留前 5%
    no_improvement_threshold = 3  # 连续 3 代无改进则提升变异率
    restart_threshold = 5  # 连续 5 代无改进则进行局部重启
    max_memory = 45
    pool_processes = min(32, int(max_memory * 1024 * 1024 * 1024 / total_size) if total_size > 0 else 1)
    print(f"进程数限制为 {pool_processes}，根据内存限制调整。")

    batch_size = 10  # 每批候选个体评价数

    # 初始化全局最优追踪变量，用于全局重启判断
    prev_overall_best = overall_best
    global_no_improve_count = 0

    with multiprocessing.Pool(processes=10, initializer=init_worker_ga,
                              initargs=(GLOBAL_SIGNALS, df)) as pool:
        for gen in range(start_gen, generations):
            start_time = time.time()
            print(f"\n========== 开始第 {gen} 代遗传算法搜索 （岛屿模型） ==========")
            island_stats_list = []

            # ----------------- 各岛进化操作 -----------------
            for idx, island in enumerate(islands):
                population = island["population"]
                result2 = overall_best in population
                print(f"开始岛 {idx} 种群的进化，{overall_best} 是否存在 {result2} 当前种群大小: {len(population)} ")
                # 评价当前岛种群，评价后统一更新全局已评价集合
                pop_batches = [population[i:i + batch_size] for i in range(0, len(population), batch_size)]
                results_batches = pool.map(evaluate_candidate_batch, pop_batches)
                # 在评价后将当前岛内所有候选个体更新到全局集合中
                global_generated_individuals.update(population)

                fitness_results = []
                for batch_res in results_batches:
                    fitness_results.extend(batch_res)
                if not fitness_results:
                    continue

                # 更新岛内属性（用于后续相似性检测及统计）
                island["sorted_fitness"] = sorted([fr[0] for fr in fitness_results], reverse=True)
                island_stats = [stat for (_, _, stat) in fitness_results if stat is not None]
                island_stats_list.extend(island_stats)

                island_best = max(fitness_results, key=lambda x: x[0])
                current_best_fitness = island_best[0]
                if current_best_fitness > island["best_fitness"]:
                    island["best_fitness"] = current_best_fitness
                    island["best_candidate"] = island_best[1]
                    island["no_improve_count"] = 0
                else:
                    island["no_improve_count"] += 1

                # 自适应调整变异率
                if island["no_improve_count"] >= no_improvement_threshold:
                    island["adaptive_mutation_rate"] = min(1, island["adaptive_mutation_rate"] + 0.05)
                    print(
                        f"岛 {idx} 连续 {island['no_improve_count']} 代无改进，提升变异率至 {island['adaptive_mutation_rate']:.2f}")
                else:
                    island["adaptive_mutation_rate"] = max(mutation_rate, island["adaptive_mutation_rate"] - 0.01)

                # 精英保留（取适应度最高的 elite_count 个体）
                elite_count = max(1, int(elite_fraction * island_pop_size))
                sorted_pop = [ind for _, ind, _ in sorted(fitness_results, key=lambda x: x[0], reverse=True)]
                elites = sorted_pop[:elite_count]

                # 锦标赛选择构造剩余候选个体
                pop_fitness = [fr[0] for fr in fitness_results]
                selected_population = tournament_selection(population, pop_fitness, tournament_size=3,
                                                           selection_pressure=0.75)

                # 交叉操作（两两交叉）
                next_population = []
                for i in range(0, len(selected_population) - 1, 2):
                    parent1 = selected_population[i]
                    parent2 = selected_population[i + 1]
                    child1, child2 = crossover(parent1, parent2, crossover_rate)
                    next_population.extend([child1, child2])
                if len(selected_population) % 2 == 1:
                    next_population.append(selected_population[-1])
                # 变异操作：使用自适应变异率
                mutated_population = [mutate(ind, island["adaptive_mutation_rate"],
                                             candidate_long_signals, candidate_short_signals)
                                      for ind in next_population]
                # 注意：这里不更新全局集合，只在评价后统一更新
                # 多样性注入：注入一定比例的随机个体
                diversity_percent = 0.1 + (0.05 * island["no_improve_count"])
                diversity_count = max(1, int(diversity_percent * island_pop_size))
                for _ in range(diversity_count):
                    # 单个新候选个体：调用后返回列表，所以取第一个元素
                    new_candidate = get_unique_candidate(candidate_long_signals, candidate_short_signals,
                                                         global_generated_individuals, candidate_list=[],
                                                         target_size=1)[0]
                    replace_index = random.randint(0, len(mutated_population) - 1)
                    mutated_population[replace_index] = new_candidate
                # 合并精英与变异个体，并去重（确保岛内个体不重复）
                mutated_population = filter_existing_individuals(mutated_population, global_generated_individuals)
                unique_population = elites + mutated_population
                # 删除mutated_population中的重复元素
                unique_population = list({ind: None for ind in unique_population}.keys())
                # 利用新的 get_unique_candidate 补充个体直到达到 island_pop_size
                unique_population = get_unique_candidate(candidate_long_signals, candidate_short_signals,
                                                         global_generated_individuals, candidate_list=unique_population,
                                                         target_size=island_pop_size)

                # 局部重启：若连续 restart_threshold 代无改进，则重启当前岛
                if island["no_improve_count"] >= restart_threshold:
                    print(f"\n岛 {idx} 连续 {restart_threshold} 代无改进, 执行局部重启增强种群多样性。")
                    new_population_count = int(0.5 * island_pop_size)
                    random_candidates = get_unique_candidate(candidate_long_signals, candidate_short_signals,
                                                             global_generated_individuals, candidate_list=[],
                                                             target_size=new_population_count)
                    unique_population = elites + mutated_population + random_candidates
                    unique_population = list({ind: None for ind in unique_population}.keys())[:island_pop_size]
                    island["no_improve_count"] = 0
                    island["adaptive_mutation_rate"] = mutation_rate
                    print(f"岛 {idx} 局部重启完成，种群多样性增强。")
                else:
                    unique_population = unique_population[:island_pop_size]

                island["population"] = unique_population
                print(f"岛 {idx} 第 {gen} 代最优个体: {island['best_candidate']}，适应度: {island['best_fitness']}")

            # ----------------- 岛屿间相似性检测 -----------------
            for i in range(len(islands)):
                for j in range(i + 1, len(islands)):
                    if "sorted_fitness" in islands[i] and "sorted_fitness" in islands[j]:
                        sorted_fit1 = islands[i]["sorted_fitness"]
                        sorted_fit2 = islands[j]["sorted_fitness"]
                        n = len(sorted_fit1) // 2  # 取前50%个体
                        sim = sum(abs(a - b) for a, b in zip(sorted_fit1[:n], sorted_fit2[:n])) / n if n > 0 else float(
                            'inf')
                        print(f"岛 {i} 与岛 {j} 的前50%个体相似性评价: {sim:.4f}")
                        if sim < restart_similarity_threshold:
                            if islands[i]["best_fitness"] < islands[j]["best_fitness"]:
                                restart_idx = i
                            else:
                                restart_idx = j
                            print(f"岛 {restart_idx} 判定为过于相似且适应度较低，执行重启。")
                            new_population = get_unique_candidate(candidate_long_signals, candidate_short_signals,
                                                                  global_generated_individuals, candidate_list=[],
                                                                  target_size=island_pop_size)
                            islands[restart_idx]["population"] = new_population
                            islands[restart_idx]["best_candidate"] = None
                            islands[restart_idx]["best_fitness"] = -1e9
                            islands[restart_idx]["no_improve_count"] = 0
                            islands[restart_idx]["adaptive_mutation_rate"] = mutation_rate
                            islands[restart_idx].pop("sorted_fitness", None)



            # ----------------- 更新全局最优及全局重启 -----------------
            for island in islands:
                if island["best_fitness"] > overall_best_fitness:
                    overall_best_fitness = island["best_fitness"]
                    overall_best = island["best_candidate"]

            elapsed_gen = time.time() - start_time

            if prev_overall_best is not None and overall_best == prev_overall_best:
                global_no_improve_count += 1
            else:
                global_no_improve_count = 0
            prev_overall_best = overall_best

            print(f"第 {gen} 代全局最优个体: {overall_best}，适应度: {overall_best_fitness}，耗时 {elapsed_gen:.2f} 秒。连续 {global_no_improve_count} 代无改进。 当前全局评价组合数: {len(global_generated_individuals)}")

            if global_no_improve_count >= 20:
                overall_best_fitness = -1e9
                overall_best = None
                print(f"连续 {global_no_improve_count} 代全局最优未改进，进行全局重启。")
                for idx, island in enumerate(islands):
                    new_population = get_unique_candidate(candidate_long_signals, candidate_short_signals,
                                                          global_generated_individuals, candidate_list=[],
                                                          target_size=island_pop_size)
                    island["population"] = new_population
                    island["best_candidate"] = None
                    island["best_fitness"] = -1e9
                    island["no_improve_count"] = 0
                    island["adaptive_mutation_rate"] = mutation_rate
                global_no_improve_count = 0

            # ----------------- 岛屿间迁移 -----------------
            if (gen + 1) % migration_interval == 0:
                print("进行岛屿间迁移...")
                for i in range(islands_count):
                    source_idx = i
                    target_idx = (i + 1) % islands_count
                    source_island = islands[source_idx]
                    target_island = islands[target_idx]
                    src_batches = [source_island["population"][j:j + batch_size]
                                   for j in range(0, len(source_island["population"]), batch_size)]
                    src_results = pool.map(evaluate_candidate_batch, src_batches)
                    src_fitness_results = [item for batch in src_results for item in batch]
                    tgt_batches = [target_island["population"][j:j + batch_size]
                                   for j in range(0, len(target_island["population"]), batch_size)]
                    tgt_results = pool.map(evaluate_candidate_batch, tgt_batches)
                    tgt_fitness_results = [item for batch in tgt_results for item in batch]
                    if not src_fitness_results or not tgt_fitness_results:
                        continue
                    sorted_src = sorted(src_fitness_results, key=lambda x: x[0], reverse=True)
                    sorted_tgt = sorted(tgt_fitness_results, key=lambda x: x[0])
                    migration_num = max(1, int(migration_rate * island_pop_size))
                    emigrants = [ind for (fit, ind, stat) in sorted_src[:migration_num]]
                    worst_tgt = {ind for (fit, ind, stat) in sorted_tgt[:migration_num]}
                    new_target_population = [ind for ind in target_island["population"] if ind not in worst_tgt]
                    new_target_population.extend(emigrants)
                    # 利用新函数补充迁移后不足的个体
                    new_target_population = get_unique_candidate(candidate_long_signals, candidate_short_signals,
                                                                 global_generated_individuals,
                                                                 candidate_list=new_target_population,
                                                                 target_size=island_pop_size)
                    target_island["population"] = new_target_population[:island_pop_size]
                    print(f"岛 {source_idx} 向岛 {target_idx} 迁移了 {migration_num} 个个体。")

            # ----------------- 保存当前代统计信息 -----------------
            if island_stats_list:
                df_stats = pd.DataFrame(island_stats_list)
                print(f"保存第 {gen} 代统计信息 ...长度 {df_stats.shape[0]}")
                df_stats = df_stats.drop_duplicates(subset=["kai_column", "pin_column"])
                print(f"去重后长度 {df_stats.shape[0]}")
                file_name = os.path.join(checkpoint_dir, f"{key_name}_{gen}_stats.csv")
                df_stats.to_csv(file_name, index=False)

            all_history.append({
                "generation": gen,
                "islands": islands,
                "overall_best_candidate": overall_best,
                "overall_best_fitness": overall_best_fitness,
            })
            try:
                with open(checkpoint_file, "wb") as f:
                    pickle.dump((gen + 1, islands, overall_best, overall_best_fitness, all_history,
                                 global_generated_individuals), f)
                print(f"第 {gen} 代信息已保存到 checkpoint。")
            except Exception as e:
                print(f"保存 checkpoint 时出错：{e}")

    print(f"\n遗传算法结束，全局最优组合: {overall_best}，净利率: {overall_best_fitness}")
    return overall_best, overall_best_fitness, all_history


##############################################
# 主流程及数据加载
##############################################

def ga_optimize_breakthrough_signal(data_path="temp/TON_1m_2000.csv"):
    """
    读取原始数据，预处理后调用遗传算法搜索最佳 (长信号, 短信号) 组合。
    """
    # 确保 temp 目录存在
    if not os.path.exists("temp"):
        os.makedirs("temp")
    base_name = os.path.basename(data_path)
    base_name = base_name.replace("-USDT-SWAP.csv", "").replace("origin_data_", "")
    df_local = pd.read_csv(data_path)
    needed_columns = ["timestamp", "high", "low", "close"]
    df_local = df_local[needed_columns]
    jingdu = "float32"
    df_local["chg"] = (df_local["close"].pct_change() * 100).astype("float16")
    df_local["high"] = df_local["high"].astype(jingdu)
    df_local["low"] = df_local["low"].astype(jingdu)
    df_local["close"] = df_local["close"].astype(jingdu)
    df_local["timestamp"] = pd.to_datetime(df_local["timestamp"])
    df_monthly = df_local["timestamp"].dt.to_period("M")
    min_df_month = df_monthly.min()
    max_df_month = df_monthly.max()
    df_local = df_local[(df_monthly != min_df_month) & (df_monthly != max_df_month)]
    print(
        f"\n开始基于遗传算法回测 {base_name} ... 数据长度 {df_local.shape[0]} 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    # 生成所有信号，并筛选长、短信号候选集合
    all_signals, key_name = generate_all_signals()

    long_signals = [sig for sig in all_signals if "long" in sig]
    short_signals = [sig for sig in all_signals if "short" in sig]
    print(f"共生成 {len(long_signals)} 个长信号和 {len(short_signals)} 个短信号。")

    # 将全局 df 设置为本地数据（主进程用，子进程通过 init_worker1/ga 初始化）
    global df
    df = df_local.copy()

    population_size = min(100000, int(len(long_signals) * len(short_signals) * 0.1))
    print(f"种群规模: {population_size}，信号总数: {len(all_signals)}")

    # 调用遗传算法搜索最佳信号组合（使用岛屿模型，参数可根据需求调整）
    best_candidate, best_fitness, history = genetic_algorithm_optimization(
        df_local, all_signals, all_signals,
        population_size=population_size, generations=1000,
        crossover_rate=0.9, mutation_rate=0.2,
        key_name=f'{base_name}_{key_name}',
        islands_count=8, migration_interval=10, migration_rate=0.05
    )
    print(f"数据 {base_name} 最优信号组合: {best_candidate}，净利率: {best_fitness}")


def example():
    """
    示例入口：依次处理多个数据文件，并调用信号优化流程。
    """
    start_time = time.time()
    data_path_list = [
        "kline_data/origin_data_1m_10000000_SOL-USDT-SWAP.csv",
        "kline_data/origin_data_1m_10000000_BTC-USDT-SWAP.csv",
        "kline_data/origin_data_1m_10000000_ETH-USDT-SWAP.csv",
        "kline_data/origin_data_1m_10000000_TON-USDT-SWAP.csv",

        "kline_data/origin_data_5m_10000000_SOL-USDT-SWAP.csv",
        "kline_data/origin_data_5m_10000000_BTC-USDT-SWAP.csv",
        "kline_data/origin_data_5m_10000000_ETH-USDT-SWAP.csv",
        "kline_data/origin_data_5m_10000000_TON-USDT-SWAP.csv",
    ]
    for data_path in data_path_list:
        try:
            ga_optimize_breakthrough_signal(data_path)
            print(f"{data_path} 总耗时 {time.time() - start_time:.2f} 秒。")
        except Exception as e:
            traceback.print_exc()
            print(f"处理 {data_path} 时出错：{e}")


if __name__ == "__main__":
    example()