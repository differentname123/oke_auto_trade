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

  7. 优化 global_generated_individuals，使用 Bloom Filter（布隆过滤器），预估最大独立个体数量为 2400 * 100000，
     可接受的错误率为 0.01，相关的 get_unique_candidate 等函数做了相应适配。
"""

import os
import sys
import time
import pickle
import random
import traceback
import math
from functools import partial
import threading

import numpy as np
import pandas as pd
from numba import njit
import multiprocessing
import mmh3  # 请确保安装 mmh3: pip install mmh3

IS_REVERSE = False  # 是否反向操作


##############################################
# 布隆过滤器实现，用于替代保存全局已生成个体的 set
##############################################
class BloomFilter:
    def __init__(self, capacity, error_rate):
        """
        capacity: 预估最大元素个数
        error_rate: 可接受的误报率
        """
        self.capacity = capacity
        self.error_rate = error_rate
        # m = - (n * ln(p)) / (ln2^2)
        self.size = math.ceil(-capacity * math.log(error_rate) / (math.log(2) ** 2))
        # k = (m/n) * ln2
        self.hash_count = math.ceil((self.size / capacity) * math.log(2))
        self.bit_array = bytearray((self.size + 7) // 8)
        self.count = 0  # 用于记录实际添加元素个数

    def _hashes(self, item):
        """采用双哈希（利用 mmh3）得到 k 个哈希值"""
        item_str = str(item)
        hash1 = mmh3.hash(item_str, seed=0)
        hash2 = mmh3.hash(item_str, seed=1)
        for i in range(self.hash_count):
            yield (hash1 + i * hash2) % self.size

    def add(self, item):
        if item in self:
            return
        self.count += 1
        for pos in self._hashes(item):
            byte_index = pos // 8
            bit_index = pos % 8
            self.bit_array[byte_index] |= (1 << bit_index)

    def __contains__(self, item):
        for pos in self._hashes(item):
            byte_index = pos // 8
            bit_index = pos % 8
            if not (self.bit_array[byte_index] & (1 << bit_index)):
                return False
        return True

    def __len__(self):
        return self.count


##############################################
# 全局变量，用于存储预计算信号数据和行情数据
##############################################

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


def get_detail_backtest_result_op_simple(df, kai_column, pin_column, is_filter=True, is_reverse=False):
    """
    优化后的函数：提前进行部分判断以避免后续不必要的计算。

    执行步骤和提前退出判断：
      1. 尝试获取信号数据，若信号数据不足（例如数量不足 100），则立即返回。
      2. 匹配信号后，如果有效交易数量不足（trade_count < 25），直接返回。
      3. 计算盈亏及净盈亏率，如果净盈亏率小于 25，则直接退出。
      4. 在进行耗时的最大连续亏损计算前，先检查平均收益、持有时长等快速指标。
      5. 月度和周度统计操作放在最后，只有在前面条件满足时才进行，如果活跃月份或周不足或亏损率过高，也直接退出。
      6. 最后判断前10%盈利贡献比例，再做一次判断。
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

    # 根据信号索引提取子数据集
    kai_data_df = df.iloc[kai_idx].copy()
    pin_data_df = df.iloc[pin_idx].copy()
    kai_data_df["kai_price"] = kai_prices
    pin_data_df["pin_price"] = pin_prices

    # 信号匹配：使用 np.searchsorted 找到 pin 的匹配位置
    kai_idx_arr = np.asarray(kai_data_df.index)
    pin_idx_arr = np.asarray(pin_data_df.index)
    pin_match_indices = np.searchsorted(pin_idx_arr, kai_idx_arr, side="right")
    valid_mask = pin_match_indices < len(pin_idx_arr)
    if valid_mask.sum() == 0:
        return None, None

    # 只保留有匹配的交易
    kai_data_df = kai_data_df.iloc[valid_mask].copy()
    kai_idx_valid = kai_idx_arr[valid_mask]
    pin_match_indices_valid = pin_match_indices[valid_mask]
    matched_pin = pin_data_df.iloc[pin_match_indices_valid].copy()

    # 增加匹配的 pin 数据
    kai_data_df["pin_price"] = matched_pin["pin_price"].values
    kai_data_df["pin_time"] = matched_pin["timestamp"].values
    kai_data_df["hold_time"] = matched_pin.index.values - kai_idx_valid

    # 根据传入的参数判断做多或做空策略
    is_long = (("long" in kai_column.lower()) if not is_reverse else ("short" in kai_column.lower()))

    if is_filter:
        # 排序并去除重复的pin_time记录
        kai_data_df = kai_data_df.sort_values("timestamp").drop_duplicates("pin_time", keep="first")

    trade_count = len(kai_data_df)

    # 使用 pin_time 的价格映射 kai_price，如果存在更新价格
    pin_price_map = kai_data_df.set_index("pin_time")["pin_price"]
    mapped_prices = kai_data_df["timestamp"].map(pin_price_map)
    if mapped_prices.notna().sum() > 0:
        kai_data_df["kai_price"] = mapped_prices.combine_first(kai_data_df["kai_price"])
    modification_rate = (100 * mapped_prices.notna().sum() / trade_count) if trade_count else 0

    # 计算盈亏比例（百分比）并扣除交易成本
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

    # 计算连续盈利金额或者亏损的序列，函数 calculate_max_sequence_numba 和 calculate_max_profit_numba 假设已定义
    profits_arr = kai_data_df["true_profit"].values
    max_loss, max_loss_start_idx, max_loss_end_idx, loss_trade_count = calculate_max_sequence_numba(profits_arr)
    if net_profit_rate < 25 or trade_count < 25 or max_loss < -30:
        return None, None

    if max_loss_start_idx < len(kai_data_df) and max_loss_end_idx < len(kai_data_df):
        max_loss_hold_time = kai_data_df.index[max_loss_end_idx] - kai_data_df.index[max_loss_start_idx]
    else:
        max_loss_hold_time = None

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

    # 使用整个原始 df 的最早和最晚时间作为统计范围，确保不同信号使用统一范围
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

    # 当统计时间范围中活跃的月份或周不足时，则返回 None
    if active_week_ratio < 0.5 or active_month_ratio < 0.5:
        return None, None

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


    if true_profit_mean < 10 or max_hold_time > 10000 or hold_time_mean > 3000 or top_profit_ratio > 0.5 or monthly_loss_rate > 0.2 or weekly_loss_rate > 0.2 or monthly_loss_rate > 0.2:
        return None, None

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
        "is_reverse": is_reverse,
    }
    return None, statistic_dict


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
    trade_count = stat.get("kai_count", -10000)
    active_month_ratio = stat.get("active_month_ratio", -10000)
    if max_loss < -20 or net_profit_rate < 100 or trade_count < 100 or active_month_ratio < 0.8:
        return -10000

    weekly_loss_rate = stat.get("weekly_loss_rate", 10000)
    monthly_loss_rate = stat.get("monthly_loss_rate", 10000)
    top_profit_ratio = stat.get("top_profit_ratio", 10000)
    if weekly_loss_rate > 0.2 or monthly_loss_rate > 0.2 or top_profit_ratio > 0.5:
        return -10000

    hold_time_mean = stat.get("hold_time_mean", 100000)
    max_hold_time = stat.get("max_hold_time", 100000)
    true_profit_mean = stat.get("avg_profit_rate", -10000)
    if hold_time_mean > 3000 or true_profit_mean < 10 or max_hold_time > 10000:
        return -10000

    value = stat.get(key, -10000)
    return -value if invert else value


def get_fitness_op(stat, key, invert=False):
    """
    根据统计结果 stat 和目标指标 key 计算适应度。

    对于每个指标，如果不满足限定条件，则按照原逻辑采用二次惩罚；
    如果满足限定条件，则适当计算奖励值：
      - 对于“下限型”指标，如 max_consecutive_loss、net_profit_rate、kai_count、
        active_month_ratio 和 avg_profit_rate，当指标值 ≥ 阈值时，
        奖励值 = (实际值 - 阈值) / (阈值的绝对值) ；
      - 对于“上限型”指标，如 weekly_loss_rate、monthly_loss_rate、top_profit_ratio、
        hold_time_mean 和 max_hold_time，当指标值 ≤ 阈值时，
        奖励值 = (阈值 - 实际值) / (阈值) ；

    如果 stat 为 None 或目标指标 key 不存在，则基础值 base 设为 0。

    最终适应度 = 基础值 + 累加的奖励值 - 累加的惩罚项
    对于需要反转的指标（invert=True），采用：
         fitness = base - total_reward + total_penalty, 并返回 -fitness
    """
    if stat is None:
        return -1000000000000

    # 取目标指标的基础值，若不存在，则设为 0
    base = stat.get(key, 0)
    penalty = 0.0
    reward = 0.0

    # 1. 最大连续亏损（max_consecutive_loss）：要求 >= -20
    #   下限型指标，采用阈值 abs(-20)=20
    max_loss = stat.get("max_consecutive_loss", -10000)
    if max_loss < -20:
        diff = -20 - max_loss  # diff 为正
        penalty += (diff ** 2) * 10
    else:
        diff = max_loss - (-20)  # 实际值超出阈值的幅度
        reward += diff / 20.0

    # 2. 净利润率（net_profit_rate）：要求 >= 50（原代码中要求50）
    #   下限型指标
    net_profit_rate = stat.get("net_profit_rate", -10000)
    if net_profit_rate < 50:
        diff = 50 - net_profit_rate
        penalty += (diff ** 2) * 10
    else:
        diff = net_profit_rate - 50
        reward += diff / 50.0

    # 3. 交易次数（kai_count）：要求 >= 50（原代码中要求50）
    #   下限型指标
    trade_count = stat.get("kai_count", -10000)
    if trade_count < 50:
        diff = 50 - trade_count
        penalty += (diff ** 2) * 10
    else:
        diff = trade_count - 50
        reward += diff / 50.0

    # 4. 活跃月份比例（active_month_ratio）：要求 >= 0.7（原代码中要求0.7）
    #   下限型指标
    active_month_ratio = stat.get("active_month_ratio", -10000)
    if active_month_ratio < 0.7:
        diff = 0.7 - active_month_ratio
        penalty += (diff ** 2) * 10000
    else:
        diff = active_month_ratio - 0.7
        reward += diff / 0.7

    # 5. 每周亏损率（weekly_loss_rate）：要求 <= 0.3
    #   上限型指标
    weekly_loss_rate = stat.get("weekly_loss_rate", 10000)
    if weekly_loss_rate > 0.3:
        diff = weekly_loss_rate - 0.3
        penalty += (diff ** 2) * 1000
    else:
        diff = 0.3 - weekly_loss_rate
        reward += diff / 0.3

    # 6. 每月亏损率（monthly_loss_rate）：要求 <= 0.3
    #   上限型指标
    monthly_loss_rate = stat.get("monthly_loss_rate", 10000)
    if monthly_loss_rate > 0.3:
        diff = monthly_loss_rate - 0.3
        penalty += (diff ** 2) * 1000
    else:
        diff = 0.3 - monthly_loss_rate
        reward += diff / 0.3

    # 7. 盈利集中度（top_profit_ratio）：要求 <= 0.5
    #   上限型指标
    top_profit_ratio = stat.get("top_profit_ratio", 10000)
    if top_profit_ratio > 0.5:
        diff = top_profit_ratio - 0.5
        penalty += (diff ** 2) * 1000
    else:
        diff = 0.5 - top_profit_ratio
        reward += diff / 0.5

    # 8. 持仓时间均值（hold_time_mean）：要求 <= 3000
    #   上限型指标
    hold_time_mean = stat.get("hold_time_mean", 100000)
    if hold_time_mean > 3000:
        diff = hold_time_mean - 3000
        penalty += (diff ** 2) / 10000.0
    else:
        diff = 3000 - hold_time_mean
        reward += diff / 3000.0

    # 9. 最大持仓时间（max_hold_time）：要求 <= 10000
    #   上限型指标
    max_hold_time = stat.get("max_hold_time", 100000)
    if max_hold_time > 10000:
        diff = max_hold_time - 10000
        penalty += (diff ** 2) / 10000.0
    else:
        diff = 10000 - max_hold_time
        reward += diff / 10000.0

    # 10. 平均盈利率（avg_profit_rate）：要求 >= 10
    #    下限型指标
    true_profit_mean = stat.get("avg_profit_rate", -10000)
    if true_profit_mean < 10:
        diff = 10 - true_profit_mean
        penalty += (diff ** 2) * 1
    else:
        diff = true_profit_mean - 10
        reward += diff / 10.0

    # 按原代码将总惩罚项再乘以 100（以匹配量级）
    penalty = penalty * 100

    # 根据是否需要反转计算最终适应度：
    # 正常指标：适应度 = base + reward - penalty
    # 反转指标：适应度 = base - reward + penalty, 并返回 -fitness，使得更优表现对应更高的返回值
    if invert:
        fitness = base - reward + penalty
    else:
        fitness = base + reward - penalty

    return -fitness if invert else fitness

# 声明两组 key:
normal_keys = ['max_consecutive_loss', 'monthly_net_profit_min', 'weekly_net_profit_min', 'net_profit_rate',
               'min_profit', 'fu_profit_mean', 'avg_profit_rate']

inverted_keys = ['monthly_net_profit_std', 'monthly_loss_rate', 'weekly_loss_rate', 'true_profit_std',
                 'weekly_net_profit_std', 'loss_rate', 'top_loss_ratio']

combined_keys = [
    'max_consecutive_loss', 'monthly_net_profit_min', 'weekly_net_profit_min', 'net_profit_rate', 'min_profit',
    'fu_profit_mean', 'avg_profit_rate', 'monthly_net_profit_std', 'monthly_loss_rate', 'weekly_loss_rate',
    'true_profit_std', 'weekly_net_profit_std', 'loss_rate', 'top_loss_ratio'

]

# 利用 functools.partial 生成各个适应度提取函数，并存储在字典中
fitness_getters = {}

for key in normal_keys:
    fitness_getters[key] = partial(get_fitness_op, key=key, invert=False)

for key in inverted_keys:
    fitness_getters[key] = partial(get_fitness_op, key=key, invert=True)
order_key = []
# 如果需要以特定顺序生成一个列表，包含所有适应度提取函数
get_fitness_list = [fitness_getters[key] for key in combined_keys]


def evaluate_candidate_batch(candidates, fitness_func=get_fitness_net, is_reverse=False):
    """
    对一批候选个体进行评价，返回列表 [(fitness, candidate, stat), ...]。
    """
    batch_results = []
    for candidate in candidates:
        long_sig, short_sig = candidate
        _, stat = get_detail_backtest_result_op_simple(df, long_sig, short_sig, is_filter=True, is_reverse=is_reverse)
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


def get_unique_candidate(candidate_long_signals, candidate_short_signals,
                         global_generated_individuals, existing_individuals, count):
    """
    优化后的 get_unique_candidate
    --------------------------------------------------
    基于批量采样、向量化和集合加速查重的方式生成指定数量的唯一候选个体。

    参数：
        candidate_long_signals (list): 长信号列表
        candidate_short_signals (list): 短信号列表
        global_generated_individuals: 用于检测候选个体是否已生成的全局记录（例如 BloomFilter，对 in 操作具有支持）
        existing_individuals (list): 已经存在的候选个体列表（新生成的候选个体应避免重复插入）
        count (int): 所需生成的候选个体数量

    返回：
        list: count 个唯一候选个体的列表，每个候选个体是由 (long_signal, short_signal) 组成的元组
    """
    # 将已有候选个体转为集合，便于快速查重
    existing_set = set(existing_individuals)
    output = []

    total_long = len(candidate_long_signals)
    total_short = len(candidate_short_signals)
    # 批量尺寸可以适当调大，以减少循环次数
    batch_size = max(100, count * 2)

    while len(output) < count:
        # 向量化生成候选个体的下标
        idx_long = np.random.randint(0, total_long, size=batch_size)
        idx_short = np.random.randint(0, total_short, size=batch_size)
        # 批量构造候选个体列表
        batch_candidates = [
            (candidate_long_signals[i], candidate_short_signals[j])
            for i, j in zip(idx_long, idx_short)
        ]
        for cand in batch_candidates:
            # 若候选个体在已有集合或全局记录中，则跳过
            if cand in existing_set or cand in global_generated_individuals:
                continue
            # 添加到局部集合和最终输出列表中
            existing_set.add(cand)
            output.append(cand)
            if len(output) == count:
                break

    return output


def genetic_algorithm_optimization(df, candidate_long_signals, candidate_short_signals,
                                   population_size=50, generations=20,
                                   crossover_rate=0.8, mutation_rate=0.1, key_name="default",
                                   islands_count=4, migration_interval=10, migration_rate=0.1,
                                   restart_similarity_threshold=10):
    """
    利用遗传算法和岛屿模型搜索净利率高的 (长信号, 短信号) 组合，支持断点续跑。
    优化内容：
      1. 在每个岛屿评估后保存适应度信息到字段 "population_with_fitness"（以及可选的 "sorted_fitness"）。
      2. 在迁移阶段直接利用这些已保存评估数据进行排序，无需重复回测。
      3. 详细记录各阶段耗时日志，但只有当耗时大于 1 秒时才打印，以便找到异常耗时的阶段。
      4. 引入异步保存检查点和统计信息的方式，降低 I/O 阻塞影响。
    """

    # 异步保存 helper 函数
    def save_checkpoint_async(data, checkpoint_file):
        def _save():
            with open(checkpoint_file, "wb") as f:
                pickle.dump(data, f)

        t = threading.Thread(target=_save)
        t.daemon = True
        t.start()

    def save_stats_async(df_stats, file_name):
        def _save():
            df_stats.to_parquet(file_name, index=False, compression='snappy')

        t = threading.Thread(target=_save)
        t.daemon = True
        t.start()

    # 辅助函数：当耗时超过 threshold 秒时打印日志
    def log_if_slow(label, delta, gen, idx=None, threshold=1.0):
        if delta > threshold:
            if idx is not None:
                print(f"[GEN {gen}][岛 {idx}] {label}耗时: {delta:.2f} 秒")
            else:
                print(f"[GEN {gen}] {label}耗时: {delta:.2f} 秒")

    checkpoint_dir = "temp"
    os.makedirs(checkpoint_dir, exist_ok=True)

    all_signals = list(set(candidate_long_signals + candidate_short_signals))
    print(f"开始预计算 GLOBAL_SIGNALS ... {key_name}")
    precomputed = load_or_compute_precomputed_signals(df, all_signals, key_name)
    total_size = sys.getsizeof(precomputed) + sum(
        sys.getsizeof(sig) + s.nbytes + p.nbytes for sig, (s, p) in precomputed.items())
    print(f"预计算信号内存大小: {total_size / (1024 * 1024):.2f} MB")

    global GLOBAL_SIGNALS
    GLOBAL_SIGNALS = precomputed
    print(f"预计算完成，共 {len(GLOBAL_SIGNALS)} 个信号数据。")

    # 重置候选信号为预计算结果的 key
    candidate_long_signals = list(GLOBAL_SIGNALS.keys())
    candidate_short_signals = list(GLOBAL_SIGNALS.keys())

    global IS_REVERSE
    IS_REVERSE = False
    checkpoint_file = os.path.join(checkpoint_dir, f"{key_name}_{IS_REVERSE}_ga_checkpoint.pkl")

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "rb") as f:
                checkpoint_data = pickle.load(f)
                if len(checkpoint_data) == 6:
                    start_gen, islands, overall_best, overall_best_fitness, all_history, loaded_gi = checkpoint_data
                    # 如果之前保存的是 set，则转换成 BloomFilter
                    if isinstance(loaded_gi, set):
                        global_generated_individuals = BloomFilter(generations * population_size, 0.01)
                        for cand in loaded_gi:
                            global_generated_individuals.add(cand)
                    else:
                        global_generated_individuals = loaded_gi
                else:
                    start_gen, islands, overall_best, overall_best_fitness, all_history = checkpoint_data
                    global_generated_individuals = BloomFilter(generations * population_size, 0.01)
            print(f"加载断点，恢复至第 {start_gen} 代。全局最优: {overall_best}，净利率: {overall_best_fitness}")
        except Exception as e:
            print(f"加载断点失败：{e}，从头开始。")
            start_gen = 0
            islands = []
            overall_best = None
            overall_best_fitness = -1e9
            all_history = []
            global_generated_individuals = BloomFilter(generations * population_size, 0.01)
    else:
        start_gen = 0
        islands = []
        overall_best = None
        overall_best_fitness = -1e9
        all_history = []
        global_generated_individuals = BloomFilter(generations * population_size, 0.01)
    print('finish load')
    island_pop_size = population_size // islands_count
    if not islands:
        for _ in range(islands_count):
            pop = get_unique_candidate(candidate_long_signals, candidate_short_signals,
                                       global_generated_individuals, [], island_pop_size)
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
    max_memory = 45
    pool_processes = min(30, int(max_memory * 1024 * 1024 * 1024 / total_size) if total_size > 0 else 1)
    print(f"使用 {pool_processes} 个进程。")
    batch_size = 1000
    prev_overall_best = overall_best
    global_no_improve_count = 0
    single_generations_count = int(generations / len(get_fitness_list))  # 实际为 generations
    fitness_index = 0
    pre_fitness_index = 0
    partial_eval = partial(evaluate_candidate_batch,
                           fitness_func=get_fitness_list[fitness_index],
                           is_reverse=IS_REVERSE)  # 显式传递 is_reverse
    print(
        f"开始搜索，总代数: {generations}，每代种群大小: {population_size}，岛屿数量: {islands_count}，适应度函数个数: {len(get_fitness_list)}。 是否反向评估: {IS_REVERSE}。适应度函数为{combined_keys[fitness_index]}")

    with multiprocessing.Pool(processes=pool_processes, initializer=init_worker_ga,
                              initargs=(GLOBAL_SIGNALS, df)) as pool:
        for gen in range(start_gen, generations):
            gen_start_time = time.time()
            print(f"\n========== 开始第 {gen} 代搜索 ==========")
            island_stats_list = []
            for idx, island in enumerate(islands):
                island_start_time = time.time()
                print(f"\n[GEN {gen}][岛 {idx}] 开始处理，当前种群大小: {len(island['population'])}")

                # -- 评估阶段 --
                eval_start_time = time.time()
                pop = island["population"]
                pop_batches = [pop[i:i + batch_size] for i in range(0, len(pop), batch_size)]
                results_batches = pool.map(partial_eval, pop_batches)
                # 将当前个体逐个添加至全局布隆过滤器
                for candidate in pop:
                    global_generated_individuals.add(candidate)
                fitness_results = [item for batch in results_batches for item in batch]
                eval_end_time = time.time()
                log_if_slow("评估", eval_end_time - eval_start_time, gen, idx)

                if not fitness_results:
                    continue

                # 保存本次评估的适应度-个体配对信息
                island["population_with_fitness"] = fitness_results
                island["sorted_fitness"] = sorted([fr[0] for fr in fitness_results], reverse=True)
                island_stats_list.extend([stat for (_, _, stat) in fitness_results if stat is not None])
                island_best = max(fitness_results, key=lambda x: x[0])
                current_best_fitness = island_best[0]

                # -- 更新岛屿最优与无改进计数 --
                update_start = time.time()
                if current_best_fitness > island["best_fitness"]:
                    island["best_fitness"] = current_best_fitness
                    island["best_candidate"] = island_best[1]
                    island["no_improve_count"] = 0
                else:
                    island["no_improve_count"] += 1
                update_end = time.time()
                log_if_slow("更新最优与无改进计数", update_end - update_start, gen, idx)

                # -- 自适应调整变异率 --
                adapt_start = time.time()
                if island["no_improve_count"] >= no_improvement_threshold:
                    island["adaptive_mutation_rate"] = min(1, island["adaptive_mutation_rate"] + 0.05)
                else:
                    island["adaptive_mutation_rate"] = max(mutation_rate, island["adaptive_mutation_rate"] - 0.01)
                adapt_end = time.time()
                log_if_slow("适应变异率调整", adapt_end - adapt_start, gen, idx)

                # -- 选择、交叉、变异及后续处理阶段 --
                process_start = time.time()
                elite_count = max(1, int(elite_fraction * island_pop_size))
                sorted_pop = [ind for _, ind, _ in sorted(fitness_results, key=lambda x: x[0], reverse=True)]
                elites = sorted_pop[:elite_count]
                pop_fitness = [fr[0] for fr in fitness_results]

                selection_start = time.time()
                selected_population = tournament_selection(pop, pop_fitness, tournament_size=3, selection_pressure=0.75)
                selection_end = time.time()
                log_if_slow("选择", selection_end - selection_start, gen, idx)

                next_population = []
                crossover_start = time.time()
                for i in range(0, len(selected_population) - 1, 2):
                    parent1 = selected_population[i]
                    parent2 = selected_population[i + 1]
                    child1, child2 = crossover(parent1, parent2, crossover_rate)
                    next_population.extend([child1, child2])
                if len(selected_population) % 2 == 1:
                    next_population.append(selected_population[-1])
                crossover_end = time.time()
                log_if_slow("交叉", crossover_end - crossover_start, gen, idx)

                mutation_start = time.time()
                mutated_population = [
                    mutate(ind, island["adaptive_mutation_rate"], candidate_long_signals, candidate_short_signals)
                    for ind in next_population]
                mutation_end = time.time()
                log_if_slow("变异", mutation_end - mutation_start, gen, idx)

                diversity_start = time.time()
                diversity_count = max(1, int((0.1 + 0.05 * island["no_improve_count"]) * island_pop_size))
                for _ in range(diversity_count):
                    new_candidate = get_unique_candidate(candidate_long_signals, candidate_short_signals,
                                                         global_generated_individuals, [], 1)[0]
                    replace_index = random.randint(0, len(mutated_population) - 1)
                    mutated_population[replace_index] = new_candidate
                diversity_end = time.time()
                log_if_slow("多样性补充", diversity_end - diversity_start, gen, idx)

                filter_start = time.time()
                mutated_population = filter_existing_individuals(mutated_population, global_generated_individuals)
                filter_end = time.time()
                log_if_slow("过滤重复个体", filter_end - filter_start, gen, idx)

                unique_population = list({ind: None for ind in elites + mutated_population}.keys())
                unique_population = get_unique_candidate(candidate_long_signals, candidate_short_signals,
                                                         global_generated_individuals, unique_population,
                                                         island_pop_size)
                if island["no_improve_count"] >= restart_threshold:
                    print(f"[GEN {gen}][岛 {idx}] 连续 {restart_threshold} 代无改进，执行局部重启。")
                    restart_start = time.time()
                    new_population_count = int(0.5 * island_pop_size)
                    random_candidates = get_unique_candidate(candidate_long_signals, candidate_short_signals,
                                                             global_generated_individuals, [], new_population_count)
                    unique_population = list(
                        {ind: None for ind in elites + mutated_population + random_candidates}.keys()
                    )[:island_pop_size]
                    island["no_improve_count"] = 0
                    island["adaptive_mutation_rate"] = mutation_rate
                    restart_end = time.time()
                    log_if_slow("局部重启", restart_end - restart_start, gen, idx)
                else:
                    unique_population = unique_population[:island_pop_size]
                process_end = time.time()
                log_if_slow("选择、交叉、变异及局部重启", process_end - process_start, gen, idx)

                island["population"] = unique_population
                island_elapsed_time = time.time() - island_start_time
                log_if_slow("整个岛屿处理", island_elapsed_time, gen, idx)
                print(f"[GEN {gen}][岛 {idx}] 当前代最优: {island['best_candidate']}，适应度: {island['best_fitness']}")

            # -- 岛屿间对比及局部重启策略 --
            pairwise_start = time.time()
            for i in range(len(islands)):
                for j in range(i + 1, len(islands)):
                    if "sorted_fitness" in islands[i] and "sorted_fitness" in islands[j]:
                        sorted_fit1 = islands[i]["sorted_fitness"]
                        sorted_fit2 = islands[j]["sorted_fitness"]
                        n = len(sorted_fit1) // 2
                        sim = sum(abs(a - b) for a, b in zip(sorted_fit1[:n], sorted_fit2[:n])) / n if n > 0 else float(
                            'inf')
                        print(f"[GEN {gen}] 岛 {i} 与岛 {j} 前50%个体相似度: {sim:.4f}")
                        if sim < restart_similarity_threshold:
                            restart_idx = i if islands[i]["best_fitness"] < islands[j]["best_fitness"] else j
                            print(f"[GEN {gen}] 岛 {restart_idx} 适应度较低且过于相似，执行重启。")
                            new_population = get_unique_candidate(candidate_long_signals, candidate_short_signals,
                                                                  global_generated_individuals, [], island_pop_size)
                            islands[restart_idx]["population"] = new_population
                            islands[restart_idx]["best_candidate"] = None
                            islands[restart_idx]["best_fitness"] = -1e9
                            islands[restart_idx]["no_improve_count"] = 0
                            islands[restart_idx]["adaptive_mutation_rate"] = mutation_rate
                            islands[restart_idx].pop("sorted_fitness", None)
            pairwise_end = time.time()
            log_if_slow("岛屿对比及局部重启", pairwise_end - pairwise_start, gen)

            # -- 更新全局最优 --
            update_global_start = time.time()
            for island in islands:
                if island["best_fitness"] > overall_best_fitness:
                    overall_best_fitness = island["best_fitness"]
                    overall_best = island["best_candidate"]
            update_global_end = time.time()
            log_if_slow("更新全局最优", update_global_end - update_global_start, gen)

            gen_elapsed_time = time.time() - gen_start_time
            if prev_overall_best is not None and overall_best == prev_overall_best:
                global_no_improve_count += 1
            else:
                global_no_improve_count = 0
            prev_overall_best = overall_best
            log_if_slow("全代总耗时", gen_elapsed_time, gen)
            print(
                f"[GEN {gen}] 全局最优: {overall_best}，适应度: {overall_best_fitness}，连续无改进: {global_no_improve_count}")

            # -- 判断是否需要切换适应度函数或执行全局重启 --
            need_restart = False
            fitness_index = gen // single_generations_count
            if fitness_index != pre_fitness_index:
                print(f"[GEN {gen}] 切换适应度函数: {combined_keys[fitness_index]}，当前代数: {gen}。")
                partial_eval = partial(evaluate_candidate_batch,
                                       fitness_func=get_fitness_list[fitness_index],
                                       is_reverse=IS_REVERSE)  # 显式传递 is_reverse
                pre_fitness_index = fitness_index
                need_restart = True
            if global_no_improve_count >= 10 or need_restart:
                restart_global_start = time.time()
                overall_best_fitness = -1e9
                overall_best = None
                print(f"[GEN {gen}] 连续 {global_no_improve_count} 代无改进，进行全局重启。")
                for island in islands:
                    new_population = get_unique_candidate(candidate_long_signals, candidate_short_signals,
                                                          global_generated_individuals, [], island_pop_size)
                    island["population"] = new_population
                    island["best_candidate"] = None
                    island["best_fitness"] = -1e9
                    island["no_improve_count"] = 0
                    island["adaptive_mutation_rate"] = mutation_rate
                global_no_improve_count = 0
                restart_global_end = time.time()
                log_if_slow("全局重启", restart_global_end - restart_global_start, gen)

            # -- 迁移阶段 --
            if (gen + 1) % migration_interval == 0:
                migration_start_time = time.time()
                print(f"[GEN {gen}] 开始岛屿迁移阶段...")
                migration_num = max(1, int(migration_rate * island_pop_size))
                for source_idx in range(islands_count):
                    target_idx = (source_idx + 1) % islands_count
                    source_island = islands[source_idx]
                    target_island = islands[target_idx]
                    if "population_with_fitness" not in source_island or "population_with_fitness" not in target_island:
                        print(f"[GEN {gen}] 岛 {source_idx} 或岛 {target_idx} 缺少评估数据，本次迁移跳过。")
                        continue
                    src_results = source_island["population_with_fitness"]
                    tgt_results = target_island["population_with_fitness"]
                    sorted_src = sorted(src_results, key=lambda x: x[0], reverse=True)
                    sorted_tgt = sorted(tgt_results, key=lambda x: x[0])
                    emigrants = [ind for (fit, ind, _) in sorted_src[:migration_num]]
                    worst_tgt_set = {ind for (fit, ind, _) in sorted_tgt[:migration_num]}
                    new_target_population = [ind for ind in target_island["population"] if ind not in worst_tgt_set]
                    new_target_population.extend(emigrants)
                    new_target_population = get_unique_candidate(
                        candidate_long_signals, candidate_short_signals,
                        global_generated_individuals, new_target_population,
                        island_pop_size
                    )
                    target_island["population"] = new_target_population[:island_pop_size]
                    print(f"[GEN {gen}] 岛 {source_idx} 向岛 {target_idx} 迁移 {migration_num} 个个体。")
                migration_end_time = time.time()
                log_if_slow("迁移阶段", migration_end_time - migration_start_time, gen)

            # -- 保存统计信息和检查点（异步保存） --
            if island_stats_list:
                stats_save_start = time.time()
                df_stats = pd.DataFrame(island_stats_list).drop_duplicates(subset=["kai_column", "pin_column"])
                stats_file_name = os.path.join(checkpoint_dir, f"{key_name}_{gen}_{IS_REVERSE}_stats.parquet")
                save_stats_async(df_stats, stats_file_name)
                stats_save_end = time.time()
                log_if_slow("保存统计信息异步发起", stats_save_end - stats_save_start, gen)
                print(f"[GEN {gen}] 异步保存统计信息已发起，记录数: {df_stats.shape[0]}")
            all_history.append({
                "generation": gen,
                "islands": islands,
                "overall_best_candidate": overall_best,
                "overall_best_fitness": overall_best_fitness,
            })
            if (gen + 1) % 2 == 0:
                try:
                    data_to_save = (gen + 1, islands, overall_best, overall_best_fitness, all_history,
                                    global_generated_individuals)
                    save_checkpoint_async(data_to_save, checkpoint_file)
                    print(f"[GEN {gen}] 第 {gen} 代 checkpoint 异步保存发起。")
                except Exception as e:
                    print(f"[GEN {gen}] 异步保存 checkpoint 时出错：{e}")
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

    # 只获取第一年的数据
    first_year = df_local["year"].min()
    print(f"数据 {base_name} 的第一年: {first_year}")
    df_local = df_local[df_local["year"] == first_year]
    # 删除年份列
    df_local.drop(columns=["year"], inplace=True)


    while df_local["low"].min() < 1:
        df_local[["high", "low", "close"]] *= 10
    jingdu = "float32"
    df_local["chg"] = (df_local["close"].pct_change() * 100).astype("float16")
    df_local["high"] = df_local["high"].astype(jingdu)
    df_local["low"] = df_local["low"].astype(jingdu)
    df_local["close"] = df_local["close"].astype(jingdu)
    df_local["timestamp"] = pd.to_datetime(df_local["timestamp"])
    df_monthly = df_local["timestamp"].dt.to_period("M")
    df_local = df_local[(df_monthly != df_monthly.min()) & (df_monthly != df_monthly.max())]
    print(
        f"\n开始基于遗传算法回测 {base_name} ... 数据长度 {df_local.shape[0]} 时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    all_signals, key_name = generate_all_signals()
    long_signals = [sig for sig in all_signals if "long" in sig]
    short_signals = [sig for sig in all_signals if "short" in sig]
    print(f"生成 {len(long_signals)} 长信号和 {len(short_signals)} 短信号。")
    global df
    df = df_local.copy()
    population_size = min(1000000, int(len(long_signals) * len(short_signals) * 0.1))
    print(f"种群规模: {population_size}，信号总数: {len(all_signals)}")
    best_candidate, best_fitness, history = genetic_algorithm_optimization(
        df_local, all_signals, all_signals,
        population_size=population_size, generations=700,
        crossover_rate=0.9, mutation_rate=0.2,
        key_name=f'{base_name}_{key_name}',
        islands_count=1, migration_interval=10, migration_rate=0.05
    )
    print(f"数据 {base_name} 最优信号组合: {best_candidate}，净利率: {best_fitness}")


def example():
    """
    示例入口：处理多个数据文件调用信号优化流程。
    """
    start_time = time.time()
    data_path_list = [
        "kline_data/origin_data_1m_5000000_BTC-USDT-SWAP_2025-05-06.csv",
    ]
    for data_path in data_path_list:
        try:
            ga_optimize_breakthrough_signal(data_path)
            print(f"{data_path} 总耗时 {time.time() - start_time:.2f} 秒。")
        except Exception as e:
            traceback.print_exc()
            print(f"处理 {data_path} 出错：{e}")


if __name__ == "__main__":
    example()
