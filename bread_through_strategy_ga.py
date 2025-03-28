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
from functools import partial

import numpy as np
import pandas as pd
from numba import njit
import multiprocessing

# 全局变量，用于存储预计算信号数据和行情数据
GLOBAL_SIGNALS = {}
df = None  # 回测数据，在子进程中通过初始化传入

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
        trade_price_series = target_price
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

@njit
def calculate_max_profit_numba(series):
    """
    利用 numba 加速计算连续盈利（最大累计收益）及对应的交易数量与区间。
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

def get_detail_backtest_result_op(df, kai_column, pin_column, is_filter=True, is_reverse=False):
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

    if (kai_idx is None or pin_idx is None or kai_idx.size < 100 or pin_idx.size < 100):
        return None, None

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

    kai_data_df["pin_price"] = matched_pin["pin_price"].values
    kai_data_df["pin_time"] = matched_pin["timestamp"].values
    kai_data_df["hold_time"] = matched_pin.index.values - kai_idx_valid

    is_long = (("long" in kai_column.lower()) if not is_reverse else ("short" in kai_column.lower()))

    if is_filter:
        kai_data_df = kai_data_df.sort_values("timestamp").drop_duplicates("pin_time", keep="first")

    trade_count = len(kai_data_df)
    total_count = len(df)

    pin_price_map = kai_data_df.set_index("pin_time")["pin_price"]
    mapped_prices = kai_data_df["timestamp"].map(pin_price_map)
    if mapped_prices.notna().sum() > 0:
        kai_data_df["kai_price"] = mapped_prices.combine_first(kai_data_df["kai_price"])
    modification_rate = (100 * mapped_prices.notna().sum() / trade_count) if trade_count else 0

    if is_long:
        profit_series = ((kai_data_df["pin_price"] - kai_data_df["kai_price"]) / kai_data_df["kai_price"] * 100).round(4)
    else:
        profit_series = ((kai_data_df["kai_price"] - kai_data_df["pin_price"]) / kai_data_df["kai_price"] * 100).round(4)
    kai_data_df["profit"] = profit_series
    kai_data_df["true_profit"] = profit_series - 0.07  # 扣除交易成本
    profit_sum = profit_series.sum()
    max_single_profit = kai_data_df["true_profit"].max()
    min_single_profit = kai_data_df["true_profit"].min()

    true_profit_std = kai_data_df["true_profit"].std()
    true_profit_mean = kai_data_df["true_profit"].mean() * 100 if trade_count > 0 else 0
    fix_profit = safe_round(kai_data_df[mapped_prices.notna()]["true_profit"].sum(), ndigits=4)
    net_profit_rate = kai_data_df["true_profit"].sum() - fix_profit

    profits_arr = kai_data_df["true_profit"].values
    max_loss, max_loss_start_idx, max_loss_end_idx, loss_trade_count = calculate_max_sequence_numba(profits_arr)
    if net_profit_rate < 25:
        return None, None

    if max_loss_start_idx < len(kai_data_df) and max_loss_end_idx < len(kai_data_df):
        max_loss_start_time = kai_data_df.iloc[max_loss_start_idx]["timestamp"]
        max_loss_end_time = kai_data_df.iloc[max_loss_end_idx]["timestamp"]
        max_loss_hold_time = kai_data_df.index[max_loss_end_idx] - kai_data_df.index[max_loss_start_idx]
    else:
        max_loss_start_time = max_loss_end_time = max_loss_hold_time = None

    if max_loss_start_idx < len(kai_data_df) and max_loss_end_idx < len(kai_data_df):
        max_profit, max_profit_start_idx, max_profit_end_idx, profit_trade_count = calculate_max_profit_numba(profits_arr)
        max_profit_start_time = kai_data_df.iloc[max_profit_start_idx]["timestamp"]
        max_profit_end_time = kai_data_df.iloc[max_profit_end_idx]["timestamp"]
        max_profit_hold_time = kai_data_df.index[max_profit_end_idx] - kai_data_df.index[max_profit_start_idx]
    else:
        max_profit, max_profit_start_time, max_profit_end_time, max_profit_hold_time = None, None, None, None

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

    # if hold_time_mean > 2000 or true_profit_mean < 10:
    #     return None, None

    # Monthly statistics
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

    # Weekly statistics
    weekly_groups = kai_data_df["timestamp"].dt.to_period("W")
    weekly_agg = kai_data_df.groupby(weekly_groups)["true_profit"].agg(["sum", "mean", "count"])
    weekly_trade_std = weekly_agg["count"].std() if "count" in weekly_agg else 0
    active_weeks = weekly_agg.shape[0]
    total_weeks = len(pd.period_range(start=kai_data_df["timestamp"].min(), end=kai_data_df["timestamp"].max(), freq='W'))
    active_week_ratio = active_weeks / total_weeks if total_weeks else 0
    weekly_net_profit_std = weekly_agg["sum"].std() if "sum" in weekly_agg else 0
    weekly_avg_profit_std = weekly_agg["mean"].std() if "mean" in weekly_agg else 0
    weekly_net_profit_min = weekly_agg["sum"].min() if "sum" in weekly_agg else 0
    weekly_net_profit_max = weekly_agg["sum"].max() if "sum" in weekly_agg else 0
    weekly_loss_rate = ((weekly_agg["sum"] < 0).sum() / active_weeks) if active_weeks else 0
    weekly_net_profit_detail = {str(week): round(val, 4) for week, val in weekly_agg["sum"].to_dict().items()}
    weekly_trade_count_detail = {str(week): int(val) for week, val in weekly_agg["count"].to_dict().items()}

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

    common_index = kai_data_df.index.intersection(pin_data_df.index)
    same_count_rate = safe_round(100 * len(common_index) / min(len(kai_data_df), len(pin_data_df)) if trade_count else 0, 4)

    statistic_dict = {
        # "kai_side": "long" if is_long else "short",
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
        # "max_loss_start_time": max_loss_start_time,
        # "max_loss_end_time": max_loss_end_time,
        "max_consecutive_profit": safe_round(max_profit, 4) if max_profit is not None else None,
        "max_profit_trade_count": profit_trade_count if max_profit is not None else None,
        "max_profit_hold_time": max_profit_hold_time,
        # "max_profit_start_time": max_profit_start_time,
        # "max_profit_end_time": max_profit_end_time,
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
        "monthly_trade_count_detail": monthly_trade_count_detail,
        "weekly_trade_std": safe_round(weekly_trade_std, 4),
        "active_week_ratio": safe_round(active_week_ratio, 4),
        "weekly_loss_rate": safe_round(weekly_loss_rate, 4),
        "weekly_net_profit_min": safe_round(weekly_net_profit_min, 4),
        "weekly_net_profit_max": safe_round(weekly_net_profit_max, 4),
        "weekly_net_profit_std": safe_round(weekly_net_profit_std, 4),
        "weekly_avg_profit_std": safe_round(weekly_avg_profit_std, 4),
        "weekly_net_profit_detail": weekly_net_profit_detail,
        "weekly_trade_count_detail": weekly_trade_count_detail
    }
    kai_data_df = kai_data_df[["hold_time", "true_profit"]]
    return kai_data_df, statistic_dict



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

def precompute_signals(df, signals):
    """
    使用多进程预计算所有候选信号数据，返回 dict 格式：{signal_name: (indices, prices)}。
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
# 遗传算法优化相关函数（岛屿模型）
##############################################

def init_worker_ga(signals, dataframe):
    """遗传算法多进程初始化，将预计算数据和 df 加载到全局变量。"""
    global GLOBAL_SIGNALS, df
    GLOBAL_SIGNALS = signals
    df = dataframe

def get_fitness_net(stat):
    """从统计结果中提取适应度值（净利率），若失败则返回 -10000。"""
    if stat is None:
        return -10000
    return stat.get("net_profit_rate", -10000)

def get_fitness(stat, key, invert=False):
    """
    从统计结果 stat 中提取适应度值。如果 stat 为 None 或者 key 不存在，则返回 -10000，
    否则返回 stat 对应 key 的值。若 invert 为 True，则返回该值的相反数。
    """
    if stat is None:
        return -10000
    max_loss = stat.get("max_consecutive_loss", -10000)
    net_profit_rate = stat.get("net_profit_rate", -10000)
    trade_count = stat.get("kai_count", 0)
    if max_loss < -10 or net_profit_rate < 100 or trade_count < 10:
        return -10000
    hold_time_mean = stat.get("hold_time_mean", 0)
    true_profit_mean = stat.get("avg_profit_rate", 0)
    if hold_time_mean > 2000 or true_profit_mean < 10:
        return -10000

    value = stat.get(key, -10000)
    return -value if invert else value

# 声明两组 key:
normal_keys = ["net_profit_rate", "kai_count",'min_profit','avg_profit_rate','fu_profit_mean', 'fu_profit_sum','monthly_net_profit_min','weekly_net_profit_min']
inverted_keys = ["hold_time_mean", "hold_time_std", "loss_rate", "loss_time_rate",'max_profit', 'true_profit_std', 'monthly_trade_std', 'monthly_net_profit_std', 'monthly_avg_profit_std',
                 'top_loss_ratio', 'top_profit_ratio','weekly_trade_std', 'weekly_net_profit_std', 'weekly_avg_profit_std', 'weekly_loss_rate', 'weekly_net_profit_max']

# 利用 functools.partial 生成各个适应度提取函数，并存储在字典中
fitness_getters = {}

for key in normal_keys:
    fitness_getters[key] = partial(get_fitness, key=key, invert=False)

for key in inverted_keys:
    fitness_getters[key] = partial(get_fitness, key=key, invert=True)
order_key = []
# 如果需要以特定顺序生成一个列表，包含所有适应度提取函数
get_fitness_list = [fitness_getters[key] for key in normal_keys + inverted_keys]

def evaluate_candidate_batch(candidates, fitness_func=get_fitness_net):
    """
    对一批候选个体进行评价，返回列表 [(fitness, candidate, stat), ...]。
    """
    batch_results = []
    for candidate in candidates:
        long_sig, short_sig = candidate
        _, stat = get_detail_backtest_result_op(df, long_sig, short_sig, is_filter=True, is_reverse=False)
        fitness = fitness_func(stat)
        batch_results.append((fitness, candidate, stat))
    return batch_results

def tournament_selection(population, fitnesses, tournament_size=3, selection_pressure=0.75):
    """
    锦标赛选择：随机挑选 tournament_size 个个体，根据一定概率选择适应度最高个体，否则随机选取其他个体。
    """
    new_population = []
    pop_with_fit = list(zip(population, fitnesses))
    for _ in range(len(population)):
        competitors = random.sample(pop_with_fit, tournament_size)
        competitors.sort(key=lambda x: x[1], reverse=True)
        winner = competitors[0] if random.random() < selection_pressure else random.choice(competitors[1:])
        new_population.append(winner[0])
    return new_population

def crossover(parent1, parent2, crossover_rate=0.8):
    """对两个父代个体进行交叉操作，若未交叉则返回原个体。"""
    if random.random() < crossover_rate:
        if random.random() < 0.5:
            return (parent2[0], parent1[1]), (parent1[0], parent2[1])
        else:
            return (parent1[0], parent2[1]), (parent2[0], parent1[1])
    return parent1, parent2

def mutate(individual, mutation_rate, candidate_long_signals, candidate_short_signals):
    """以 mutation_rate 概率对个体进行变异，替换长信号或短信号。"""
    long_gene, short_gene = individual
    if random.random() < mutation_rate:
        long_gene = random.choice(candidate_long_signals)
    if random.random() < mutation_rate:
        short_gene = random.choice(candidate_short_signals)
    return (long_gene, short_gene)

def filter_existing_individuals(candidate_list, global_generated_individuals):
    """过滤掉已经评价过的个体。"""
    return [ind for ind in candidate_list if ind not in global_generated_individuals]

def get_unique_candidate(candidate_long_signals, candidate_short_signals, global_generated_individuals, candidate_list, target_size):
    """
    补充 candidate_list 至 target_size，生成的新候选个体不能重复。
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
    利用遗传算法和岛屿模型搜索净利率高的 (长信号, 短信号) 组合，支持断点续跑。
    """
    checkpoint_dir = "temp"
    os.makedirs(checkpoint_dir, exist_ok=True)

    all_signals = list(set(candidate_long_signals + candidate_short_signals))
    print(f"开始预计算 GLOBAL_SIGNALS ... {key_name}")
    precomputed = load_or_compute_precomputed_signals(df, all_signals, key_name)
    total_size = sys.getsizeof(precomputed) + sum(sys.getsizeof(sig) + s.nbytes + p.nbytes for sig, (s, p) in precomputed.items())
    print(f"预计算信号内存大小: {total_size/(1024*1024):.2f} MB")

    global GLOBAL_SIGNALS
    GLOBAL_SIGNALS = precomputed
    print(f"预计算完成，共 {len(GLOBAL_SIGNALS)} 个信号数据。")

    # 重置候选信号为预计算结果的 key
    candidate_long_signals = list(GLOBAL_SIGNALS.keys())
    candidate_short_signals = list(GLOBAL_SIGNALS.keys())
    global_generated_individuals = set()

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
            print(f"加载断点，恢复至第 {start_gen} 代。全局最优: {overall_best}，净利率: {overall_best_fitness}")
        except Exception as e:
            print(f"加载断点失败：{e}，从头开始。")
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

    island_pop_size = population_size // islands_count
    if not islands:
        for _ in range(islands_count):
            pop = get_unique_candidate(candidate_long_signals, candidate_short_signals, global_generated_individuals, [], island_pop_size)
            island_state = {
                "population": pop,
                "best_candidate": None,
                "best_fitness": -1e9,
                "no_improve_count": 0,
                "adaptive_mutation_rate": mutation_rate,
            }
            islands.append(island_state)

    elite_fraction = 0.05
    no_improvement_threshold = 3
    restart_threshold = 5
    max_memory = 40
    pool_processes = min(32, int(max_memory * 1024 * 1024 * 1024 / total_size) if total_size > 0 else 1)
    print(f"使用 {pool_processes} 个进程。")
    batch_size = 10
    prev_overall_best = overall_best
    global_no_improve_count = 0
    single_generations_count = int(generations / len(get_fitness_list))  # 实际为 generations
    fitness_index = 0
    pre_fitness_index = 0
    partial_eval = partial(evaluate_candidate_batch, fitness_func=get_fitness_list[fitness_index])
    print(f"开始搜索，总代数: {generations}，每代种群大小: {population_size}，岛屿数量: {islands_count}，适应度函数个数: {len(get_fitness_list)} single_generations_count: {single_generations_count}")

    with multiprocessing.Pool(processes=pool_processes, initializer=init_worker_ga, initargs=(GLOBAL_SIGNALS, df)) as pool:
        for gen in range(start_gen, generations):
            start_time = time.time()
            island_stats_list = []
            print(f"\n========== 第 {gen} 代搜索，适应度函数: {get_fitness_list[fitness_index].func.__name__} ==========")
            for idx, island in enumerate(islands):
                pop = island["population"]
                print(f"岛 {idx} 进化开始，overall_best 在种群中: {overall_best in pop}，种群大小: {len(pop)}")
                pop_batches = [pop[i:i+batch_size] for i in range(0, len(pop), batch_size)]
                results_batches = pool.map(partial_eval, pop_batches)
                global_generated_individuals.update(pop)
                fitness_results = [item for batch in results_batches for item in batch]
                if not fitness_results:
                    continue
                island["sorted_fitness"] = sorted([fr[0] for fr in fitness_results], reverse=True)
                island_stats_list.extend([stat for (_, _, stat) in fitness_results if stat is not None])
                island_best = max(fitness_results, key=lambda x: x[0])
                current_best_fitness = island_best[0]
                if current_best_fitness > island["best_fitness"]:
                    island["best_fitness"] = current_best_fitness
                    island["best_candidate"] = island_best[1]
                    island["no_improve_count"] = 0
                else:
                    island["no_improve_count"] += 1
                if island["no_improve_count"] >= no_improvement_threshold:
                    island["adaptive_mutation_rate"] = min(1, island["adaptive_mutation_rate"] + 0.05)
                    print(f"岛 {idx} 连续 {island['no_improve_count']} 代无改进，变异率升至 {island['adaptive_mutation_rate']:.2f}")
                else:
                    island["adaptive_mutation_rate"] = max(mutation_rate, island["adaptive_mutation_rate"] - 0.01)
                elite_count = max(1, int(elite_fraction * island_pop_size))
                sorted_pop = [ind for _, ind, _ in sorted(fitness_results, key=lambda x: x[0], reverse=True)]
                elites = sorted_pop[:elite_count]
                pop_fitness = [fr[0] for fr in fitness_results]
                selected_population = tournament_selection(pop, pop_fitness, tournament_size=3, selection_pressure=0.75)
                next_population = []
                for i in range(0, len(selected_population)-1, 2):
                    parent1 = selected_population[i]
                    parent2 = selected_population[i+1]
                    child1, child2 = crossover(parent1, parent2, crossover_rate)
                    next_population.extend([child1, child2])
                if len(selected_population) % 2 == 1:
                    next_population.append(selected_population[-1])
                mutated_population = [mutate(ind, island["adaptive_mutation_rate"], candidate_long_signals, candidate_short_signals)
                                      for ind in next_population]
                diversity_count = max(1, int((0.1 + 0.05 * island["no_improve_count"]) * island_pop_size))
                for _ in range(diversity_count):
                    new_candidate = get_unique_candidate(candidate_long_signals, candidate_short_signals, global_generated_individuals, [], 1)[0]
                    replace_index = random.randint(0, len(mutated_population)-1)
                    mutated_population[replace_index] = new_candidate
                mutated_population = filter_existing_individuals(mutated_population, global_generated_individuals)
                unique_population = list({ind: None for ind in elites + mutated_population}.keys())
                unique_population = get_unique_candidate(candidate_long_signals, candidate_short_signals, global_generated_individuals, unique_population, island_pop_size)
                if island["no_improve_count"] >= restart_threshold:
                    print(f"岛 {idx} 连续 {restart_threshold} 代无改进，执行局部重启。")
                    new_population_count = int(0.5 * island_pop_size)
                    random_candidates = get_unique_candidate(candidate_long_signals, candidate_short_signals, global_generated_individuals, [], new_population_count)
                    unique_population = list({ind: None for ind in elites + mutated_population + random_candidates}.keys())[:island_pop_size]
                    island["no_improve_count"] = 0
                    island["adaptive_mutation_rate"] = mutation_rate
                else:
                    unique_population = unique_population[:island_pop_size]
                island["population"] = unique_population
                print(f"岛 {idx} 第 {gen} 代最优: {island['best_candidate']}，适应度: {island['best_fitness']}")
            for i in range(len(islands)):
                for j in range(i+1, len(islands)):
                    if "sorted_fitness" in islands[i] and "sorted_fitness" in islands[j]:
                        sorted_fit1 = islands[i]["sorted_fitness"]
                        sorted_fit2 = islands[j]["sorted_fitness"]
                        n = len(sorted_fit1) // 2
                        sim = sum(abs(a-b) for a, b in zip(sorted_fit1[:n], sorted_fit2[:n])) / n if n > 0 else float('inf')
                        print(f"岛 {i} 与岛 {j} 前50%个体相似度: {sim:.4f}")
                        if sim < restart_similarity_threshold:
                            restart_idx = i if islands[i]["best_fitness"] < islands[j]["best_fitness"] else j
                            print(f"岛 {restart_idx} 适应度较低且过于相似，执行重启。")
                            new_population = get_unique_candidate(candidate_long_signals, candidate_short_signals, global_generated_individuals, [], island_pop_size)
                            islands[restart_idx]["population"] = new_population
                            islands[restart_idx]["best_candidate"] = None
                            islands[restart_idx]["best_fitness"] = -1e9
                            islands[restart_idx]["no_improve_count"] = 0
                            islands[restart_idx]["adaptive_mutation_rate"] = mutation_rate
                            islands[restart_idx].pop("sorted_fitness", None)
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
            print(f"第 {gen} 代全局最优: {overall_best}，适应度: {overall_best_fitness}，耗时: {elapsed_gen:.2f} 秒。连续无改进: {global_no_improve_count}，评估组合数: {len(global_generated_individuals)}")
            need_restart = False
            fitness_index = gen // single_generations_count
            if fitness_index != pre_fitness_index:
                partial_eval = partial(evaluate_candidate_batch, fitness_func=get_fitness_list[fitness_index])
                pre_fitness_index = fitness_index
                need_restart = True
            if global_no_improve_count >= 10 or need_restart:
                overall_best_fitness = -1e9
                overall_best = None
                print(f"连续 {global_no_improve_count} 代无改进，进行全局重启。")
                for island in islands:
                    new_population = get_unique_candidate(candidate_long_signals, candidate_short_signals, global_generated_individuals, [], island_pop_size)
                    island["population"] = new_population
                    island["best_candidate"] = None
                    island["best_fitness"] = -1e9
                    island["no_improve_count"] = 0
                    island["adaptive_mutation_rate"] = mutation_rate
                global_no_improve_count = 0
            if (gen+1) % migration_interval == 0:
                print("岛屿间进行迁移...")
                for i in range(islands_count):
                    target_idx = (i+1) % islands_count
                    source_island = islands[i]
                    target_island = islands[target_idx]
                    src_batches = [source_island["population"][j:j+batch_size] for j in range(0, len(source_island["population"]), batch_size)]
                    src_results = pool.map(partial_eval, src_batches)
                    src_fitness_results = [item for batch in src_results for item in batch]
                    tgt_batches = [target_island["population"][j:j+batch_size] for j in range(0, len(target_island["population"]), batch_size)]
                    tgt_results = pool.map(partial_eval, tgt_batches)
                    tgt_fitness_results = [item for batch in tgt_results for item in batch]
                    if not src_fitness_results or not tgt_fitness_results:
                        continue
                    sorted_src = sorted(src_fitness_results, key=lambda x: x[0], reverse=True)
                    sorted_tgt = sorted(tgt_fitness_results, key=lambda x: x[0])
                    migration_num = max(1, int(migration_rate * island_pop_size))
                    emigrants = [ind for (fit, ind, _) in sorted_src[:migration_num]]
                    worst_tgt = {ind for (fit, ind, _) in sorted_tgt[:migration_num]}
                    new_target_population = [ind for ind in target_island["population"] if ind not in worst_tgt]
                    new_target_population.extend(emigrants)
                    new_target_population = get_unique_candidate(candidate_long_signals, candidate_short_signals, global_generated_individuals, new_target_population, island_pop_size)
                    target_island["population"] = new_target_population[:island_pop_size]
                    print(f"岛 {i} 向岛 {target_idx} 迁移 {migration_num} 个个体。")
            if island_stats_list:
                df_stats = pd.DataFrame(island_stats_list).drop_duplicates(subset=["kai_column", "pin_column"])
                file_name = os.path.join(checkpoint_dir, f"{key_name}_{gen}_stats.csv")
                df_stats.to_csv(file_name, index=False)
                print(f"保存第 {gen} 代统计信息，去重后长度 {df_stats.shape[0]}")
            all_history.append({
                "generation": gen,
                "islands": islands,
                "overall_best_candidate": overall_best,
                "overall_best_fitness": overall_best_fitness,
            })
            if (gen+1) % 100 == 0:
                try:
                    with open(checkpoint_file, "wb") as f:
                        pickle.dump((gen+1, islands, overall_best, overall_best_fitness, all_history, global_generated_individuals), f)
                    print(f"第 {gen} 代 checkpoint 已保存。")
                except Exception as e:
                    print(f"保存 checkpoint 时出错：{e}")
    print(f"\n遗传算法结束，全局最优: {overall_best}，净利率: {overall_best_fitness}")
    return overall_best, overall_best_fitness, all_history

##############################################
# 主流程及数据加载
##############################################

def ga_optimize_breakthrough_signal(data_path="temp/TON_1m_2000.csv"):
    """
    加载数据后调用遗传算法，搜索最佳 (长信号, 短信号) 组合。
    """
    os.makedirs("temp", exist_ok=True)
    base_name = os.path.basename(data_path).replace("-USDT-SWAP.csv", "").replace("origin_data_", "")
    df_local = pd.read_csv(data_path)
    needed_columns = ["timestamp", "high", "low", "close"]
    df_local = df_local[needed_columns]
    while df_local["low"].min() < 1:
        df_local[["high", "low", "close"]] *= 10
    jingdu = "float32"
    df_local["chg"] = (df_local["close"].pct_change()*100).astype("float16")
    df_local["high"] = df_local["high"].astype(jingdu)
    df_local["low"] = df_local["low"].astype(jingdu)
    df_local["close"] = df_local["close"].astype(jingdu)
    df_local["timestamp"] = pd.to_datetime(df_local["timestamp"])
    df_monthly = df_local["timestamp"].dt.to_period("M")
    df_local = df_local[(df_monthly != df_monthly.min()) & (df_monthly != df_monthly.max())]
    print(f"\n开始基于遗传算法回测 {base_name} ... 数据长度 {df_local.shape[0]} 时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    all_signals, key_name = generate_all_signals()
    long_signals = [sig for sig in all_signals if "long" in sig]
    short_signals = [sig for sig in all_signals if "short" in sig]
    print(f"生成 {len(long_signals)} 长信号和 {len(short_signals)} 短信号。")
    global df
    df = df_local.copy()
    population_size = min(100000, int(len(long_signals) * len(short_signals) * 0.1))
    print(f"种群规模: {population_size}，信号总数: {len(all_signals)}")
    best_candidate, best_fitness, history = genetic_algorithm_optimization(
        df_local, all_signals, all_signals,
        population_size=population_size, generations=2400,
        crossover_rate=0.9, mutation_rate=0.2,
        key_name=f'{base_name}_{key_name}',
        islands_count=4, migration_interval=10, migration_rate=0.05
    )
    print(f"数据 {base_name} 最优信号组合: {best_candidate}，净利率: {best_fitness}")

def example():
    """
    示例入口：处理多个数据文件调用信号优化流程。
    """
    start_time = time.time()
    data_path_list = [
        "kline_data/origin_data_1m_10000000_SOL-USDT-SWAP.csv",
        "kline_data/origin_data_1m_10000000_BTC-USDT-SWAP.csv",
        "kline_data/origin_data_1m_10000000_ETH-USDT-SWAP.csv",
        "kline_data/origin_data_1m_10000000_TON-USDT-SWAP.csv",
        "kline_data/origin_data_1m_10000000_DOGE-USDT-SWAP.csv",
        "kline_data/origin_data_1m_10000000_XRP-USDT-SWAP.csv",
        "kline_data/origin_data_1m_10000000_PEPE-USDT-SWAP.csv"
    ]
    for data_path in data_path_list:
        try:
            ga_optimize_breakthrough_signal(data_path)
            print(f"{data_path} 总耗时 {time.time()-start_time:.2f} 秒。")
        except Exception as e:
            traceback.print_exc()
            print(f"处理 {data_path} 出错：{e}")

if __name__ == "__main__":
    example()