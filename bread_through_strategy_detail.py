"""
突破策略的信号生成以及回测（稀疏存储版，不进行整数转换）
"""

import multiprocessing
import os
import pickle
import sys
import time
import traceback
from itertools import product

import numpy as np
import pandas as pd
from numba import njit


# 全局信号字典（只存储非零索引和对应的价格数据，均保持 float 类型）
GLOBAL_SIGNALS = {}


def compute_signal(df, col_name):
    """
    计算给定信号名称对应的信号及其价格序列（均保留原始 float 精度）
    """
    parts = col_name.split('_')
    period = int(parts[1])
    signal_type = parts[0]
    direction = parts[-1]  # "long" 或 "short"

    if signal_type == 'peak':
        if direction == "long":
            price_series = df['high'].shift(1).rolling(window=period).max()
            signal_series = df['high'] > price_series
        else:
            price_series = df['low'].shift(1).rolling(window=period).min()
            signal_series = df['low'] < price_series
        # 可选：对价格保留4位小数
        price_series = price_series.round(4)
        return signal_series, price_series

    elif signal_type == 'continue':
        if direction == "long":
            condition = df['chg'] > 0
        else:
            condition = df['chg'] < 0
        rolling_sum = condition.rolling(window=period).sum()
        valid_count = df['chg'].rolling(window=period).count()
        signal_series = (rolling_sum == period) & (valid_count == period)
        price_series = df['close']
        return signal_series, price_series

    elif signal_type == 'abs':
        abs_value = float(parts[2])
        if direction == "long":
            min_low_series = df['low'].shift(1).rolling(window=period).min()
            target_price = min_low_series * (1 + abs_value / 100)
            target_price = target_price.round(4)
            signal_series = (df['high'].shift(1) <= target_price) & (df['high'] > target_price)
            return signal_series, target_price
        else:
            max_high_series = df['high'].shift(1).rolling(window=period).max()
            target_price = max_high_series * (1 - abs_value / 100)
            target_price = target_price.round(4)
            signal_series = (df['low'].shift(1) >= target_price) & (df['low'] < target_price)
            return signal_series, target_price

    elif signal_type == 'ma':
        moving_avg = df['close'].shift(1).rolling(window=period).mean()
        moving_avg = moving_avg.round(4)
        if direction == "long":
            signal_series = (df['high'].shift(1) <= moving_avg) & (df['high'] > moving_avg)
        else:
            signal_series = (df['low'].shift(1) >= moving_avg) & (df['low'] < moving_avg)
        return signal_series, moving_avg

    elif signal_type == 'macross':
        fast_period = int(parts[1])
        slow_period = int(parts[2])
        fast_ma = df['close'].rolling(window=fast_period).mean().shift(1)
        slow_ma = df['close'].rolling(window=slow_period).mean().shift(1)
        fast_ma = fast_ma.round(4)
        slow_ma = slow_ma.round(4)
        if direction == "long":
            signal_series = (fast_ma.shift(1) <= slow_ma.shift(1)) & (fast_ma > slow_ma)
        else:
            signal_series = (fast_ma.shift(1) >= slow_ma.shift(1)) & (fast_ma < slow_ma)
        # 直接返回 close 价格作为交易价格
        return signal_series, df['close']

    elif signal_type == 'rsi':
        period = int(parts[1])
        overbought = int(parts[2])
        oversold = int(parts[3])
        delta = df['close'].diff(1).astype(np.float32)
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        if direction == "long":
            signal_series = (rsi.shift(1) > overbought) & (rsi <= overbought)
        else:
            signal_series = (rsi.shift(1) < oversold) & (rsi >= oversold)
        return signal_series, df['close']

    elif signal_type == 'relate':
        abs_value = float(parts[2])
        if direction == "long":
            min_low_series = df['low'].shift(1).rolling(window=period).min()
            max_high_series = df['high'].shift(1).rolling(window=period).max()
            target_price = min_low_series + abs_value / 100 * (max_high_series - min_low_series)
            target_price = target_price.round(4)
            signal_series = (df['high'].shift(1) <= target_price) & (df['high'] > target_price)
            return signal_series, target_price
        else:
            max_high_series = df['high'].shift(1).rolling(window=period).max()
            min_low_series = df['low'].shift(1).rolling(window=period).min()
            target_price = max_high_series - abs_value / 100 * (max_high_series - min_low_series)
            target_price = target_price.round(4)
            signal_series = (df['low'].shift(1) >= target_price) & (df['low'] < target_price)
            return signal_series, target_price

    else:
        raise ValueError(f"未知的信号类型: {signal_type}")


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


@njit
def compute_low_min_range(low_array, start_pos, end_pos):
    n = start_pos.shape[0]
    out = np.empty(n, dtype=low_array.dtype)
    for i in range(n):
        s = start_pos[i]
        e = end_pos[i] + 1  # 包含终点
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
        e = end_pos[i] + 1  # 包含终点
        m = high_array[s]
        for j in range(s + 1, e):
            if high_array[j] > m:
                m = high_array[j]
        out[i] = m
    return out


def optimize_parameters(df, tp_range=None, sl_range=None):
    if tp_range is None:
        tp_range = df['max_true_profit'].values
        tp_range = tp_range[tp_range > 0]
        tp_range = np.round(tp_range, 2)
        tp_range = np.unique(tp_range)
    if sl_range is None:
        sl_range = df['min_true_profit'].values
        sl_range = sl_range[sl_range < 0]
        sl_range = np.round(sl_range, 2)
        sl_range = np.unique(sl_range)

    true_profit = df['true_profit'].values
    max_true_profit = df['max_true_profit'].values
    min_true_profit = df['min_true_profit'].values
    n_trades = true_profit.shape[0]

    simulated_tp = np.where(
        max_true_profit[np.newaxis, :] >= tp_range[:, np.newaxis],
        tp_range[:, np.newaxis],
        true_profit[np.newaxis, :]
    )
    total_profits_tp = simulated_tp.sum(axis=1)
    loss_rates_tp = (simulated_tp < 0).sum(axis=1) / n_trades
    best_tp_index = np.argmax(total_profits_tp)
    best_tp = tp_range[best_tp_index]
    best_tp_profit = total_profits_tp[best_tp_index]
    best_tp_loss_rate = loss_rates_tp[best_tp_index]

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

    return {
        'max_optimal_value': best_tp,
        'max_optimal_profit': best_tp_profit,
        'max_optimal_loss_rate': best_tp_loss_rate,
        'min_optimal_value': best_sl,
        'min_optimal_profit': best_sl_profit,
        'min_optimal_loss_rate': best_sl_loss_rate
    }



def get_detail_backtest_result_op(df, kai_column, pin_column, is_filter=True, is_detail=False, is_reverse=False):
    """
    优化后的 get_detail_backtest_result_op 函数（不依赖预计算信号，每次调用时重新生成信号）：
      1. 根据传入的 df 和信号名称，直接调用 compute_signal 生成信号及其价格序列；
      2. 从生成的信号中提取非零索引及对应价格；
      3. 利用向量化操作获取回测数据并计算各类指标。
    """
    # 生成信号及价格序列（考虑可能产生异常）
    try:
        kai_signal, kai_price_series = compute_signal(df, kai_column)
        pin_signal, pin_price_series = compute_signal(df, pin_column)
    except Exception as e:
        print(f"信号生成错误：{e}")
        return None, None

    # 转换为 NumPy 数组（避免重复转换）
    kai_signal_np = kai_signal.to_numpy(copy=False) if hasattr(kai_signal, "to_numpy") else np.asarray(kai_signal)
    pin_signal_np = pin_signal.to_numpy(copy=False) if hasattr(pin_signal, "to_numpy") else np.asarray(pin_signal)
    kai_price_np = kai_price_series.to_numpy(copy=False) if hasattr(kai_price_series, "to_numpy") else np.asarray(kai_price_series)
    pin_price_np = pin_price_series.to_numpy(copy=False) if hasattr(pin_price_series, "to_numpy") else np.asarray(pin_price_series)

    # 保证价格数据类型为 float32
    if kai_price_np.dtype == np.float64:
        kai_price_np = kai_price_np.astype(np.float32)
    if pin_price_np.dtype == np.float64:
        pin_price_np = pin_price_np.astype(np.float32)

    # 获取信号中非零的位置（True 的索引）
    kai_idx = np.nonzero(kai_signal_np)[0]
    pin_idx = np.nonzero(pin_signal_np)[0]

    # 如果信号数量不足，则直接返回
    if kai_idx.size < 100 or pin_idx.size < 100:
        return None, None

    # 提取对应行，并赋值信号触发时的价格
    kai_data_df = df.iloc[kai_idx].copy()
    pin_data_df = df.iloc[pin_idx].copy()
    kai_data_df["kai_price"] = kai_price_np[kai_idx]
    pin_data_df["pin_price"] = pin_price_np[pin_idx]

    # 判断两个信号的公共索引数（用来过滤不匹配的组合）
    common_index = kai_data_df.index.intersection(pin_data_df.index)
    same_count = len(common_index)
    kai_count = len(kai_data_df)
    pin_count = len(pin_data_df)
    same_count_rate = (100 * same_count / min(kai_count, pin_count)) if min(kai_count, pin_count) > 0 else 0
    if same_count_rate > 1:
        return None, None

    # 利用 numpy 向量化查找匹配：对 kai_data_df 中的时间索引在 pin_data_df 中进行右侧查找
    kai_idx_arr = np.asarray(kai_data_df.index)
    pin_idx_arr = np.asarray(pin_data_df.index)
    pin_match_indices = np.searchsorted(pin_idx_arr, kai_idx_arr, side="right")
    valid_mask = pin_match_indices < len(pin_idx_arr)
    if valid_mask.sum() == 0:
        return None, None

    # 筛选有效数据以及对应匹配结果
    kai_data_df = kai_data_df.iloc[valid_mask].copy()
    kai_idx_valid = kai_idx_arr[valid_mask]
    pin_match_indices_valid = pin_match_indices[valid_mask]
    matched_pin = pin_data_df.iloc[pin_match_indices_valid].copy()

    # 更新匹配数据：引入匹配的 pin_price 与 timestamp，并计算持仓时长（假设索引间可直接相减）
    kai_data_df["pin_price"] = matched_pin["pin_price"].values
    kai_data_df["pin_time"] = matched_pin["timestamp"].values
    kai_data_df["hold_time"] = matched_pin.index.values - kai_idx_valid

    # 判断当前方向（仅一次判断，避免多次字符串查找）
    if is_reverse:
        is_long = "short" in kai_column.lower()
    else:
        is_long = "long" in kai_column.lower()

    # 若要求详细计算，则利用 NumPy 数组及向量化操作计算区间最低和最高价格
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

    # 根据 pin_time 建立映射，更新 kai_price，使得价格更准确
    pin_price_map = kai_data_df.set_index("pin_time")["pin_price"]
    mapped_prices = kai_data_df["timestamp"].map(pin_price_map)
    if same_count > 0 and not mapped_prices.isna().all():
        kai_data_df["kai_price"] = mapped_prices.combine_first(kai_data_df["kai_price"])

    # 利用向量化方式计算收益率
    if is_long:
        profit_series = ((kai_data_df["pin_price"] - kai_data_df["kai_price"]) / kai_data_df["kai_price"] * 100).round(4)
    else:
        profit_series = ((kai_data_df["kai_price"] - kai_data_df["pin_price"]) / kai_data_df["kai_price"] * 100).round(4)
    kai_data_df["profit"] = profit_series
    kai_data_df["true_profit"] = profit_series - 0.07

    # 基本统计指标计算
    trade_count = len(kai_data_df)
    total_count = len(df)
    profit_sum = profit_series.sum()

    if is_detail and trade_count > 0:
        max_single_profit = kai_data_df["max_true_profit"].max()
        min_single_profit = kai_data_df["min_true_profit"].min()
        temp_dict = optimize_parameters(kai_data_df) if trade_count > 0 else {}
    else:
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
    zhen_profit_sum = profit_df["true_profit"].sum()
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

    # 新增指标：每个月净利润和交易个数
    monthly_net_profit_detail = {str(month): round(val, 4) for month, val in monthly_agg["sum"].to_dict().items()}
    monthly_trade_count_detail = {str(month): int(val) for month, val in monthly_agg["count"].to_dict().items()}

    hold_time_std = kai_data_df["hold_time"].std()

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
        "zhen_profit_sum": zhen_profit_sum,
        "fu_profit_sum": fu_profit_sum,
        "profit_rate": profit_sum,
        "max_profit": max_single_profit,
        "min_profit": min_single_profit,
        "cost_rate": trade_count * 0.07,
        "net_profit_rate": kai_data_df["true_profit"].sum(),
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
        "monthly_trade_std": round(monthly_trade_std, 4),
        "active_month_ratio": round(active_month_ratio, 4),
        "monthly_loss_rate": round(monthly_loss_rate, 4),
        "monthly_net_profit_min": round(monthly_net_profit_min, 4),
        "monthly_net_profit_max": round(monthly_net_profit_max, 4),
        "monthly_net_profit_std": round(monthly_net_profit_std, 4),
        "monthly_avg_profit_std": round(monthly_avg_profit_std, 4),
        "top_profit_ratio": round(top_profit_ratio, 4),
        "top_loss_ratio": round(top_loss_ratio, 4),
        "is_reverse": is_reverse,
        # 新增的每月净利润和交易个数的详细数据
        "monthly_net_profit_detail": monthly_net_profit_detail,
        "monthly_trade_count_detail": monthly_trade_count_detail
    }
    statistic_dict.update(temp_dict)
    return kai_data_df, statistic_dict


def generate_numbers(start, end, number, even=True):
    if start > end or number <= 0:
        return []
    if number == 1:
        return []
    result = []
    if even:
        step = (end - start) / (number - 1)
        for i in range(number):
            result.append(int(round(start + i * step)))
    else:
        power = 2
        for i in range(number):
            normalized_index = i / (number - 1) if number > 1 else 0
            value = start + (end - start) * (normalized_index ** power)
            result.append(int(round(value)))
    final_result = []
    last_val = None
    for val in result:
        if start <= val <= end and val != last_val:
            final_result.append(val)
            last_val = val
    return final_result[:number]


def process_tasks(task_chunk, df, is_filter, is_reverse):
    start_time = time.time()
    results = []
    for long_column, short_column in task_chunk:
        _, stat_long = get_detail_backtest_result_op(df, long_column, short_column, is_filter, is_reverse=is_reverse)
        # _, stat_long_reverse = get_detail_backtest_result_op(df, long_column, short_column, is_filter, is_reverse=True)
        results.append(stat_long)
        # results.append(stat_long_reverse)
    print(f"处理 {len(task_chunk)} 个任务，耗时 {time.time() - start_time:.2f} 秒。")
    return results


def gen_ma_signal_name(start_period, end_period, step):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    long_columns = [f"ma_{period}_high_long" for period in period_list]
    short_columns = [f"ma_{period}_low_short" for period in period_list]
    key_name = f'ma_{start_period}_{end_period}_{step}'
    print(f"ma 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}")
    return long_columns, short_columns, key_name


def gen_rsi_signal_name(start_period, end_period, step):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    temp_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    long_columns = [f"rsi_{period}_{overbought}_{100 - overbought}_high_long"
                    for period in period_list for overbought in temp_list]
    short_columns = [f"rsi_{period}_{overbought}_{100 - overbought}_low_short"
                     for period in period_list for overbought in temp_list]
    key_name = f'rsi_{start_period}_{end_period}_{step}'
    print(f"rsi 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}")
    return long_columns, short_columns, key_name


def gen_peak_signal_name(start_period, end_period, step):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    long_columns = [f"peak_{period}_high_long" for period in period_list]
    short_columns = [f"peak_{period}_low_short" for period in period_list]
    key_name = f'peak_{start_period}_{end_period}_{step}'
    print(f"peak 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}")
    return long_columns, short_columns, key_name


def gen_continue_signal_name(start_period, end_period, step):
    period_list = range(start_period, end_period, step)
    long_columns = [f"continue_{period}_high_long" for period in period_list]
    short_columns = [f"continue_{period}_low_short" for period in period_list]
    key_name = f'continue_{start_period}_{end_period}_{step}'
    print(f"continue 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}")
    return long_columns, short_columns, key_name


def gen_abs_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = range(start_period1, end_period1, step1)
    period_list1 = [x / 10 for x in period_list1]
    long_columns = [f"abs_{period}_{period1}_high_long"
                    for period in period_list for period1 in period_list1 if period >= period1]
    short_columns = [f"abs_{period}_{period1}_low_short"
                     for period in period_list for period1 in period_list1 if period >= period1]
    key_name = f'abs_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}'
    print(f"abs 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}, {start_period1}, {end_period1}, {step1}")
    return long_columns, short_columns, key_name


def gen_relate_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = range(start_period1, end_period1, step1)
    long_columns = [f"relate_{period}_{period1}_high_long"
                    for period in period_list for period1 in period_list1 if period >= period1]
    short_columns = [f"relate_{period}_{period1}_low_short"
                     for period in period_list for period1 in period_list1 if period >= period1]
    key_name = f'relate_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}'
    print(f"relate 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}, {start_period1}, {end_period1}, {step1}")
    return long_columns, short_columns, key_name


def gen_macross_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = generate_numbers(start_period1, end_period1, step1, even=False)
    long_columns = [f"macross_{period}_{period1}_high_long"
                    for period in period_list for period1 in period_list1]
    short_columns = [f"macross_{period}_{period1}_low_short"
                     for period in period_list for period1 in period_list1]
    key_name = f'macross_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}'
    print(f"macross 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}, {start_period1}, {end_period1}, {step1}")
    return long_columns, short_columns, key_name


def worker_func(args):
    chunk, df, is_filter, is_reverse = args
    return process_tasks(chunk, df, is_filter, is_reverse)


def init_worker(precomputed_signals):
    """
    进程池初始化函数，将预计算的稀疏信号数据设置为各进程的全局变量。
    """
    global GLOBAL_SIGNALS
    GLOBAL_SIGNALS = precomputed_signals


def init_worker1(dataframe):
    """
    在每个子进程中初始化全局变量 df，使得 compute_signal 能够访问到它。
    """
    global df
    df = dataframe

def process_signal(sig):
    """
    针对单个信号进行计算：
      - 调用 compute_signal 函数获得 s, p
      - 将 s, p 转换为 numpy 数组
      - 获取非零索引，如果数量不足 100 个则跳过
      - 返回 (sig, (indices, 对应的 p 值)) 的结果
    如果计算出错，则打印错误信息并返回 None。
    """
    try:
        s, p = compute_signal(df, sig)
        s_np = s.to_numpy(copy=False) if hasattr(s, "to_numpy") else np.asarray(s)
        p_np = p.to_numpy(copy=False) if hasattr(p, "to_numpy") else np.asarray(p)
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
    针对一批信号进行处理：
      遍历列表并调用 process_signal, 如果返回非 None 则加入结果列表
    """
    batch_results = []
    for sig in sig_batch:
        res = process_signal(sig)
        if res is not None:
            batch_results.append(res)
    return batch_results

def filter_param(inst_id):
    """
    生成更加仔细的搜索参数
    :return:
    """
    range_size = 1
    output_path = f'temp/{inst_id}_good.csv'
    if os.path.exists(output_path):
        return pd.read_csv(output_path)
def gen_period_list(base_value, ratio, count):
    result_list = []
    period_extend = max(1, int(round(base_value * ratio)))
    for i in range(1, count + 1):
        result_list.append(base_value - period_extend * i)
        result_list.append(base_value + period_extend * i)
    result_list.append(base_value)
    result_list = list(set(result_list))
    return result_list



def gen_extend_columns(columns):
    """
    生成扩展列。columns可能的格式有rsi_75_30_70_high_long，abs_6_3.6_high_long，relate_1067_4_high_long
    :param columns:
    :return:
    """
    parts = columns.split('_')
    period = int(parts[1])
    type = parts[0]
    ratio = 0.05
    extend_value = 2
    # 生成period前后100个period
    period_list = gen_period_list(period, ratio, extend_value)
    # 筛选出大于1
    period_list = [i for i in period_list if int(i) > 0]
    if type == 'rsi':
        value1 = int(parts[2])
        value2 = int(parts[3])
        value1_list = gen_period_list(value1, ratio, extend_value)
        value2_list = gen_period_list(value2, ratio, extend_value)
        value1_list = [i for i in value1_list if int(i) > 0 and int(i) < 100]
        value2_list = [i for i in value2_list if int(i) > 0 and int(i) < 100]
        return [f'{type}_{period}_{value1}_{value2}_{parts[4]}_{parts[5]}' for period in period_list for value1 in value1_list for value2 in value2_list]
    if type == 'macross':
        value1 = int(parts[2])
        value1_list = gen_period_list(value1, ratio, extend_value)
        value1_list = [i for i in value1_list if i > 0]
        return [f'{type}_{period}_{value1}_{parts[3]}_{parts[4]}' for period in period_list for value1 in value1_list]
    elif type == 'abs':
        value1 = int(float(parts[2]) * 10)
        value1_list = gen_period_list(value1, ratio, extend_value)
        value1_list = [i / 10 for i in value1_list if i > 0]
        return [f'{type}_{period}_{value1}_{parts[3]}_{parts[4]}' for period in period_list for value1 in value1_list]
    elif type == 'relate':
        value1 = int(parts[2])
        value1_list = gen_period_list(value1, ratio, extend_value)
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

def backtest_breakthrough_strategy(df, base_name, is_filter, is_reverse):
    """
    回测函数（不依赖预计算信号，每次调用时在 get_detail_backtest_result_op 内重新生成信号）：
      1. 生成各信号列名；
      2. 基于所有信号构造任务，利用多进程逐块回测，并将结果保存为 CSV 文件。
    """
    # column_list = []
    # continue_long_columns, continue_short_columns, continue_key_name = gen_continue_signal_name(1, 20, 1)
    # column_list.append((continue_long_columns, continue_short_columns, continue_key_name))
    #
    # macross_long_columns, macross_short_columns, macross_key_name = gen_macross_signal_name(1, 100, 10, 1, 100, 10)
    # column_list.append((macross_long_columns, macross_short_columns, macross_key_name))
    #
    # ma_long_columns, ma_short_columns, ma_key_name = gen_ma_signal_name(1, 100, 20)
    # column_list.append((ma_long_columns, ma_short_columns, ma_key_name))
    #
    # relate_long_columns, relate_short_columns, relate_key_name = gen_relate_signal_name(1, 2000, 90, 1, 100, 3)
    # column_list.append((relate_long_columns, relate_short_columns, relate_key_name))
    #
    # peak_long_columns, peak_short_columns, peak_key_name = gen_peak_signal_name(1, 100, 20)
    # column_list.append((peak_long_columns, peak_short_columns, peak_key_name))
    #
    # rsi_long_columns, rsi_short_columns, rsi_key_name = gen_rsi_signal_name(1, 1000, 130)
    # column_list.append((rsi_long_columns, rsi_short_columns, rsi_key_name))
    #
    # abs_long_columns, abs_short_columns, abs_key_name = gen_abs_signal_name(1, 2000, 100, 1, 40, 1)
    # column_list.append((abs_long_columns, abs_short_columns, abs_key_name))
    #
    # # 按信号数量升序排列
    # column_list = sorted(column_list, key=lambda x: len(x[0]))
    # all_columns = []
    # key_name = ''
    # for column_pair in column_list:
    #     long_columns, short_columns, temp_key_name = column_pair
    #     temp = long_columns + short_columns
    #     key_name += temp_key_name + '_'
    #     all_columns.extend(temp)
    key_name = 'SOL'
    task_list, all_columns = gen_search_param(key_name, is_reverse)
    start_time = time.time()
    print("当前信号总数：{} 个。".format(len(all_columns)))

    # 直接根据所有信号构造任务（不做预计算）
    # task_list = list(product(all_columns, all_columns))
    print(f"共有 {len(task_list)} 个任务。")

    # 将任务分块（每块 1,000,000 个任务）
    big_chunk_size = 100000
    big_task_chunks = [task_list[i:i + big_chunk_size] for i in range(0, len(task_list), big_chunk_size)]
    print(f"任务分为 {len(big_task_chunks)} 大块。")

    pool_processes = max(1, multiprocessing.cpu_count())
    with multiprocessing.Pool(processes=pool_processes - 2) as pool:
        for i, task_chunk in enumerate(big_task_chunks):
            output_path = os.path.join('temp', f"statistic_{base_name}_{key_name}_is_filter-{is_filter}_is_reverse-{is_reverse}part{i}_op.csv")
            if os.path.exists(output_path):
                print(f'已存在 {output_path}')
                continue

            task_chunk = list(task_chunk)
            np.random.shuffle(task_chunk)
            # 将任务进一步分块，确保每块任务数量合理
            chunk_size = int(np.ceil(len(task_chunk) / (pool_processes * 12)))
            chunk_size = max(50, chunk_size)
            task_chunks = [task_chunk[j:j + chunk_size] for j in range(0, len(task_chunk), chunk_size)]
            print(
                f'当前处理文件: {output_path}\n'
                f'共有 {len(task_chunk)} 个任务，分为 {len(task_chunks)} 块，'
                f'每块任务约 {len(task_chunks[0])} 个。'
            )
            start_time_block = time.time()
            tasks_args = [(chunk, df, is_filter, is_reverse) for chunk in task_chunks]
            statistic_dict_list = []
            for result in pool.imap_unordered(worker_func, tasks_args, chunksize=1):
                statistic_dict_list.extend(result)

            statistic_dict_list = [x for x in statistic_dict_list if x is not None]
            statistic_df = pd.DataFrame(statistic_dict_list)
            statistic_df.to_csv(output_path, index=False)
            print(f'耗时 {time.time() - start_time_block:.2f} 秒。结果已保存到 {output_path} 当前时间 {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')


def gen_breakthrough_signal(data_path='temp/TON_1m_2000.csv'):
    """
    主函数：
      1. 加载 CSV 数据（保留 timestamp, open, high, low, close）；
      2. 转换所有价格为 float 类型；
      3. 计算涨跌幅 chg，过滤月初与月末数据，然后启动回测。
    """
    base_name = os.path.basename(data_path)
    is_filter = True
    df = pd.read_csv(data_path)
    needed_columns = ['timestamp', 'high', 'low', 'close']
    df = df[needed_columns]

    df['chg'] = (df['close'].pct_change() * 100).astype('float32')
    df['high'] = df['high'].astype('float32')
    df['low'] = df['low'].astype('float32')
    df['close'] = df['close'].astype('float32')

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_monthly = df['timestamp'].dt.to_period('M')
    min_df_month = df_monthly.min()
    max_df_month = df_monthly.max()
    df = df[(df_monthly != min_df_month) & (df_monthly != max_df_month)]
    print(f'开始回测 {base_name} ... 数据长度 {df.shape[0]} 当前时间 {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    backtest_breakthrough_strategy(df, base_name, is_filter, is_reverse=False)
    backtest_breakthrough_strategy(df, base_name, is_filter, is_reverse=True)



def example():
    start_time = time.time()
    data_path_list = [
        'kline_data/origin_data_1m_10000000_SOL-USDT-SWAP.csv',
        # 'kline_data/origin_data_1m_10000000_BTC-USDT-SWAP.csv',
        # 'kline_data/origin_data_1m_10000000_ETH-USDT-SWAP.csv',
        # 'kline_data/origin_data_1m_10000000_TON-USDT-SWAP.csv',
        # 'kline_data/origin_data_1m_10000000_DOGE-USDT-SWAP.csv',
        # 'kline_data/origin_data_1m_10000000_XRP-USDT-SWAP.csv',
        # 'kline_data/origin_data_1m_10000000_PEPE-USDT-SWAP.csv',
    ]
    for data_path in data_path_list:
        try:
            gen_breakthrough_signal(data_path)
            print(f'{data_path} 总耗时 {time.time() - start_time:.2f} 秒。')
        except Exception as e:
            traceback.print_exc()
            print(f'处理 {data_path} 时出错：{e}')


if __name__ == '__main__':
    example()