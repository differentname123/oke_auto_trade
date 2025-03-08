"""
突破策略的信号生成以及回测（优化版）
"""
import itertools
import math
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

def get_next_threshold_abs(df, col_name):
    parts = col_name.split('_')
    direction = parts[-1]
    period = int(parts[1])
    abs_value = float(parts[2])

    if len(df) < period + 1:
        return None  # 数据不足，无法计算

    last_high = df['high'].iloc[-1]  # 当前 K 线的最高价
    last_low = df['low'].iloc[-1]    # 当前 K 线的最低价

    if direction == "long":
        # 计算过去 period 根 K 线的最低价（不包括当前 K 线）
        min_low_prev = df['low'].iloc[-(period+1):-1].min()
        threshold_price = round(min_low_prev * (1 + abs_value / 100), 4)

        # 确保当前 K 线有可能触发信号
        if last_high < threshold_price:
            return threshold_price, ">="
        else:
            return None  # 价格未突破，不会触发信号

    elif direction == "short":
        # 计算过去 period 根 K 线的最高价（不包括当前 K 线）
        max_high_prev = df['high'].iloc[-(period+1):-1].max()
        threshold_price = round(max_high_prev * (1 - abs_value / 100), 4)

        # 确保当前 K 线有可能触发信号
        if last_low > threshold_price:
            return threshold_price, "<="
        else:
            return None  # 价格未跌破，不会触发信号

    return None  # 方向无效


def get_next_threshold_relate(df, col_name):
    parts = col_name.split('_')
    direction = parts[-1]
    period = int(parts[1])
    abs_value = float(parts[2])

    last_high = df['high'].iloc[-1]  # 当前 K 线的最高价
    last_low = df['low'].iloc[-1]    # 当前 K 线的最低价

    # 检查数据是否足够（由于 shift(1) 后会丢失最新数据，需至少 period+1 行）
    if df.shape[0] < period + 1:
        return None

    if direction == "long":
        # 取前一周期数据（所有计算基于 shift(1)）
        min_low = df['low'].shift(1).rolling(window=period).min().iloc[-1]
        max_high = df['high'].shift(1).rolling(window=period).max().iloc[-1]
        target_price = round(min_low + abs_value / 100 * (max_high - min_low), 4)
        comp = ">"  # 下一周期若 high > target_price 则突破成功
        if last_high < target_price:
            return target_price, comp
    else:
        max_high = df['high'].shift(1).rolling(window=period).max().iloc[-1]
        min_low = df['low'].shift(1).rolling(window=period).min().iloc[-1]
        target_price = round(max_high - abs_value / 100 * (max_high - min_low), 4)
        comp = "<"  # 下一周期若 low < target_price 则突破成功
        if last_low > target_price:
            return target_price, comp
    return None

def get_next_threshold_rsi(df, col_name):
    parts = col_name.split('_')
    direction = parts[-1]
    period = int(parts[1])
    overbought = int(parts[2])
    oversold = int(parts[3])

    if len(df) < period + 1:
        return None

    # 计算价格变化
    delta = df['close'].diff(1).astype(np.float64)

    # 获取最近 `period` 个数据
    diffs = delta.iloc[-period:]

    if diffs.isnull().any():
        return None

    # 计算涨跌幅
    gains = diffs.clip(lower=0)
    losses = -diffs.clip(upper=0)

    S_gain = gains.sum()
    S_loss = losses.sum()

    # 如果 S_loss 为 0，避免除零错误
    if S_loss == 0:
        rs = float('inf')
    else:
        rs = S_gain / S_loss

    rsi = 100 - (100 / (1 + rs))

    # 获取最后的 RSI 值
    df.loc[df.index[-1], 'rsi'] = rsi
    last_rsi = df['rsi'].iloc[-1]

    # 获取最新收盘价
    C_last = df['close'].iloc[-1]

    # 计算门槛价格
    d0 = diffs.iloc[0]
    g0 = max(d0, 0)
    l0 = -min(d0, 0)

    if direction == "short":
        OS = oversold
        threshold_price = C_last + (OS / (100 - OS)) * (S_loss - l0) - (S_gain - g0)
        if last_rsi < OS:
            return threshold_price, ">"
    elif direction == "long":
        OB = overbought
        threshold_price = C_last - ((S_gain - g0) * ((100 - OB) / OB) - (S_loss - l0))
        if last_rsi > OB:
            return threshold_price, "<"

    return None

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

def compute_signal_old(df, col_name):
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
            signal_series = (df['close'].shift(1) <= target_price) & (df['high'] > target_price)
            return signal_series, target_price
        else:
            max_high_series = df['high'].shift(1).rolling(window=period).max()
            target_price = max_high_series * (1 - abs_value / 100)
            target_price = target_price.round(4)
            signal_series = (df['close'].shift(1) >= target_price) & (df['low'] < target_price)
            return signal_series, target_price

    elif signal_type == 'ma':
        moving_avg = df['close'].shift(1).rolling(window=period).mean()
        moving_avg = moving_avg.round(4)
        if direction == "long":
            signal_series = (df['close'].shift(1) <= moving_avg) & (df['high'] > moving_avg)
        else:
            signal_series = (df['close'].shift(1) >= moving_avg) & (df['low'] < moving_avg)
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
            signal_series = (df['close'].shift(1) <= target_price) & (df['high'] > target_price)
            return signal_series, target_price
        else:
            max_high_series = df['high'].shift(1).rolling(window=period).max()
            min_low_series = df['low'].shift(1).rolling(window=period).min()
            target_price = max_high_series - abs_value / 100 * (max_high_series - min_low_series)
            target_price = target_price.round(4)
            signal_series = (df['close'].shift(1) >= target_price) & (df['low'] < target_price)
            return signal_series, target_price

    else:
        raise ValueError(f"未知的信号类型: {signal_type}")


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


def get_detail_backtest_result_op(total_months, df, kai_column, pin_column, signal_cache, is_filter=True, is_detail=False, is_reverse=False):
    # 判断交易方向
    kai_side = 'long' if 'long' in kai_column.lower() else 'short'
    temp_dict = {}

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

    # 使用 index 交集计算重复匹配数
    common_index = kai_data_df.index.intersection(pin_data_df.index)
    same_count = len(common_index)
    pin_count = len(pin_data_df)
    kai_count = len(kai_data_df)
    if min(pin_count, kai_count) == 0:
        same_count_rate = 0
    else:
        same_count_rate = round(100 * same_count / min(pin_count, kai_count), 4)
    if same_count_rate > 500:
        return None, None

    # 对 kai_data_df 中每个时间点，找到 pin_data_df 中最接近右侧的匹配项
    kai_idx_all = kai_data_df.index.to_numpy()
    pin_idx = pin_data_df.index.to_numpy()
    # 使用 np.searchsorted 找到 pin_df 中的位置索引
    pin_indices = np.searchsorted(pin_idx, kai_idx_all, side='right')
    # 筛选出位置在有效范围内的记录
    valid_mask = pin_indices < len(pin_idx)
    kai_data_df = kai_data_df.iloc[valid_mask].copy()
    kai_idx_valid = kai_idx_all[valid_mask]
    pin_indices_valid = pin_indices[valid_mask]
    matched_pin = pin_data_df.iloc[pin_indices_valid].copy()

    # 将匹配的价格、时间及持仓时长加入 kai_data_df
    kai_data_df['pin_price'] = matched_pin['pin_price'].to_numpy()
    kai_data_df['pin_time']  = matched_pin['timestamp'].to_numpy()
    # 这里用 matched_pin.index（即 pin 数据的 index）减去 kai_data_df 的 index，二者必须均为时间类型
    kai_data_df['hold_time'] = matched_pin.index.to_numpy() - kai_idx_valid

    # 判断方向，仅判断一次，避免多处调用字符串查找
    if is_reverse:
        is_long = "short" in kai_column.lower()
    else:
        is_long = "long" in kai_column.lower()

    if is_detail:
        # 缓存 df 各列数据的 NumPy 数组，避免重复转换
        df_index_arr = df.index.to_numpy()
        low_array = df['low'].to_numpy()
        high_array = df['high'].to_numpy()

        start_times = kai_data_df.index.to_numpy()
        end_times = matched_pin.index.to_numpy()

        start_pos = np.searchsorted(df_index_arr, start_times, side='left')
        end_pos = np.searchsorted(df_index_arr, end_times, side='right') - 1

        low_min_arr = compute_low_min_range(low_array, start_pos, end_pos)
        high_max_arr = compute_high_max_range(high_array, start_pos, end_pos)

        kai_data_df['low_min'] = low_min_arr
        kai_data_df['high_max'] = high_max_arr

        if is_long:
            kai_data_df['max_true_profit'] = (((kai_data_df['high_max'] - kai_data_df['kai_price']) /
                                                kai_data_df['kai_price'] * 100 - 0.07)
                                               .round(4))
            kai_data_df['min_true_profit'] = (((kai_data_df['low_min'] - kai_data_df['kai_price']) /
                                                kai_data_df['kai_price'] * 100 - 0.07)
                                               .round(4))
        else:
            kai_data_df['max_true_profit'] = (((kai_data_df['kai_price'] - kai_data_df['low_min']) /
                                                kai_data_df['kai_price'] * 100 - 0.07)
                                               .round(4))
            kai_data_df['min_true_profit'] = (((kai_data_df['kai_price'] - kai_data_df['high_max']) /
                                                kai_data_df['kai_price'] * 100 - 0.07)
                                               .round(4))

    if is_filter:
        kai_data_df = kai_data_df.sort_values('timestamp').drop_duplicates('pin_time', keep='first')

    # 根据映射关系更新 kai_price，采用先构造以 pin_time 为索引的 Series，再 map timestamp 得到对应价格
    pin_price_map = kai_data_df.set_index('pin_time')['pin_price']
    mapped_prices = kai_data_df['timestamp'].map(pin_price_map)
    kai_data_df['kai_price'] = mapped_prices.combine_first(kai_data_df['kai_price'])

    # 计算收益率，均采用向量化计算
    if is_long:
        kai_data_df['profit'] = ((kai_data_df['pin_price'] - kai_data_df['kai_price']) /
                                 kai_data_df['kai_price'] * 100).round(4)
    else:
        kai_data_df['profit'] = ((kai_data_df['kai_price'] - kai_data_df['pin_price']) /
                                 kai_data_df['pin_price'] * 100).round(4)
    kai_data_df['true_profit'] = kai_data_df['profit'] - 0.07

    trade_count = len(kai_data_df)
    total_count = len(df)
    profit_sum = kai_data_df['profit'].sum()

    # 初步统计最大和最小收益率
    max_single_profit = kai_data_df['true_profit'].max()
    min_single_profit = kai_data_df['true_profit'].min()

    if is_detail:
        # 若开启 is_detail，则覆盖前面值，且优化参数（假定 optimize_parameters 已定义）
        max_single_profit = kai_data_df['max_true_profit'].max()
        min_single_profit = kai_data_df['min_true_profit'].min()
        if trade_count > 0:
            temp_dict = optimize_parameters(kai_data_df)

    true_profit_std = kai_data_df['true_profit'].std()
    true_profit_mean = kai_data_df['true_profit'].mean() * 100 if trade_count > 0 else 0

    max_loss, max_loss_start_idx, max_loss_end_idx, loss_trade_count = calculate_max_sequence(kai_data_df)
    max_profit, max_profit_start_idx, max_profit_end_idx, profit_trade_count = calculate_max_profit(kai_data_df)

    if max_loss_start_idx is not None and max_loss_end_idx is not None:
        max_loss_start_time = kai_data_df.iloc[max_loss_start_idx]['timestamp']
        max_loss_end_time = kai_data_df.iloc[max_loss_end_idx]['timestamp']
        max_loss_hold_time = kai_data_df.index[max_loss_end_idx] - kai_data_df.index[max_loss_start_idx]
    else:
        max_loss_start_time = max_loss_end_time = max_loss_hold_time = None

    if max_profit_start_idx is not None and max_profit_end_idx is not None:
        max_profit_start_time = kai_data_df.iloc[max_profit_start_idx]['timestamp']
        max_profit_end_time = kai_data_df.iloc[max_profit_end_idx]['timestamp']
        max_profit_hold_time = kai_data_df.index[max_profit_end_idx] - kai_data_df.index[max_profit_start_idx]
    else:
        max_profit_start_time = max_profit_end_time = max_profit_hold_time = None

    profit_df = kai_data_df[kai_data_df['true_profit'] > 0]
    loss_df = kai_data_df[kai_data_df['true_profit'] < 0]

    loss_rate = loss_df.shape[0] / trade_count if trade_count else 0
    loss_time = loss_df['hold_time'].sum() if not loss_df.empty else 0
    profit_time = profit_df['hold_time'].sum() if not profit_df.empty else 0
    loss_time_rate = loss_time / (loss_time + profit_time) if (loss_time + profit_time) else 0

    trade_rate = round(100 * trade_count / total_count, 4) if total_count else 0
    hold_time_mean = kai_data_df['hold_time'].mean() if trade_count else 0

    # 使用一次 groupby 运算计算月度统计指标
    monthly_groups = kai_data_df['timestamp'].dt.to_period('M')
    monthly_agg = kai_data_df.groupby(monthly_groups)['true_profit'].agg(['sum', 'mean', 'count'])
    monthly_trade_std = float(monthly_agg['count'].std())
    active_months = monthly_agg.shape[0]
    active_month_ratio = active_months / total_months if total_months else 0
    monthly_net_profit_std = float(monthly_agg['sum'].std())
    monthly_avg_profit_std = float(monthly_agg['mean'].std())
    monthly_net_profit_min = monthly_agg['sum'].min()
    # 统计小于0的比例
    monthly_loss_rate = (monthly_agg['sum'] < 0).sum() / active_months if active_months else 0

    hold_time_std = kai_data_df['hold_time'].std()

    # 盈亏贡献均衡性指标：前 10% 盈利/亏损交易贡献比例
    if not profit_df.empty:
        top_profit_count = max(1, int(np.ceil(len(profit_df) * 0.1)))
        profit_sorted = profit_df.sort_values('true_profit', ascending=False)
        top_profit_sum = profit_sorted['true_profit'].iloc[:top_profit_count].sum()
        total_profit_sum = profit_df['true_profit'].sum()
        top_profit_ratio = top_profit_sum / total_profit_sum if total_profit_sum != 0 else 0
    else:
        top_profit_ratio = 0

    if not loss_df.empty:
        top_loss_count = max(1, int(np.ceil(len(loss_df) * 0.1)))
        loss_sorted = loss_df.sort_values('true_profit', ascending=True)
        top_loss_sum = loss_sorted['true_profit'].iloc[:top_loss_count].sum()
        total_loss_sum = loss_df['true_profit'].sum()
        top_loss_ratio = (abs(top_loss_sum) / abs(total_loss_sum)) if total_loss_sum != 0 else 0
    else:
        top_loss_ratio = 0

    statistic_dict = {
        'kai_side': kai_side,
        'kai_column': kai_column,
        'pin_column': pin_column,
        'kai_count': trade_count,
        'total_count': total_count,
        'trade_rate': trade_rate,
        'hold_time_mean': hold_time_mean,
        'hold_time_std': hold_time_std,
        'loss_rate': loss_rate,
        'loss_time_rate': loss_time_rate,
        'profit_rate': profit_sum,
        'max_profit': max_single_profit,
        'min_profit': min_single_profit,
        'cost_rate': trade_count * 0.07,
        'net_profit_rate': kai_data_df['true_profit'].sum(),
        'avg_profit_rate': round(true_profit_mean, 4),
        'true_profit_std': true_profit_std,
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
        'same_count': same_count,
        'same_count_rate': same_count_rate,
        'monthly_trade_std': monthly_trade_std,
        'active_month_ratio': active_month_ratio,
        'monthly_loss_rate':monthly_loss_rate,
        'monthly_net_profit_min' : monthly_net_profit_min,
        'monthly_net_profit_std': monthly_net_profit_std,
        'monthly_avg_profit_std': monthly_avg_profit_std,
        'top_profit_ratio': top_profit_ratio,
        'top_loss_ratio': top_loss_ratio
    }
    statistic_dict.update(temp_dict)
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


def process_tasks(task_chunk, df, is_filter):
    """
    处理一块任务（每块任务包含 chunk_size 个任务）。
    在处理过程中，根据任务内的信号名称动态生成信号并缓存，降低内存和重复计算。
    """
    start_time = time.time()
    results = []
    signal_cache = {}  # 同一进程内对信号结果进行缓存
    for long_column, short_column in task_chunk:
        # long_column = 'abs_2_1_high_long'
        # short_column = 'abs_2_1_low_short'
        # 对应一次做「开仓」回测
        _, stat_long = get_detail_backtest_result_op(df, long_column, short_column, signal_cache, is_filter)
        results.append(stat_long)
        # # 再做「开空」方向回测（此时互换信号）
        # _, stat_short = get_detail_backtest_result_op(df, short_column, long_column, signal_cache, is_filter)
        # results.append(stat_short)
    print(f"处理 {len(task_chunk) * 1} 个任务，耗时 {time.time() - start_time:.2f} 秒。")
    return results


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


def backtest_breakthrough_strategy(df, base_name, is_filter):
    """
    回测函数：基于原始数据 df 和指定周期范围，
    生成所有 (kai, pin) 信号对（kai 信号命名为 "{period}_high_long"，pin 信号命名为 "{period}_low_short"），
    使用多进程并行调用 process_tasks() 完成回测，并将统计结果保存到 CSV 文件。
    """
    key_name = ''
    column_list = []
    continue_long_columns, continue_short_columns, continue_key_name = gen_continue_signal_name(1, 20, 1)
    column_list.append((continue_long_columns, continue_short_columns, continue_key_name))

    macross_long_columns, macross_short_columns, macross_key_name = gen_macross_signal_name(1, 1000, 20, 1, 1000, 20)
    column_list.append((macross_long_columns, macross_short_columns, macross_key_name))


    ma_long_columns, ma_short_columns, ma_key_name = gen_ma_signal_name(1, 3000, 300)
    column_list.append((ma_long_columns, ma_short_columns, ma_key_name))


    relate_long_columns, relate_short_columns, relate_key_name = gen_relate_signal_name(1, 1000, 30, 1, 100, 6)
    column_list.append((relate_long_columns, relate_short_columns, relate_key_name))






    peak_long_columns, peak_short_columns, peak_key_name = gen_peak_signal_name(1, 3000, 300)
    column_list.append((peak_long_columns, peak_short_columns, peak_key_name))


    rsi_long_columns, rsi_short_columns, rsi_key_name = gen_rsi_signal_name(1, 1000, 40)
    column_list.append((rsi_long_columns, rsi_short_columns, rsi_key_name))


    abs_long_columns, abs_short_columns, abs_key_name = gen_abs_signal_name(1, 1000, 30, 1, 30, 1)
    column_list.append((abs_long_columns, abs_short_columns, abs_key_name))

    for column_pair in column_list:
        long_columns, short_columns, key_name = column_pair
        all_columns = long_columns + short_columns
        task_list = list(product(all_columns, all_columns))
        # task_list = list(product(long_columns, short_columns))
        # task_list.extend(list(product(short_columns, long_columns)))
        # task_list = list(product(long_columns, long_columns))
        # task_list.extend(list(product(short_columns, short_columns)))

        big_chunk_size = 100000
        big_task_chunks = [task_list[i:i + big_chunk_size] for i in range(0, len(task_list), big_chunk_size)]
        print(f'共有 {len(task_list)} 个任务，分为 {len(big_task_chunks)} 大块。')
        for i, task_chunk in enumerate(big_task_chunks):
            # 将task_list打乱顺序
            output_path = f"temp/statistic_{base_name}_{key_name}_is_filter-{is_filter}_part{i}.csv"
            if os.path.exists(output_path):
                print(f'已存在 {output_path}')
                continue
            task_chunk = task_chunk.copy()
            np.random.shuffle(task_chunk)

            # 将任务分块，每块包含一定数量的任务
            chunk_size = 100
            task_chunks = [task_chunk[i:i + chunk_size] for i in range(0, len(task_chunk), chunk_size)]
            print(f'共有 {len(task_chunk)} 个任务，分为 {len(task_chunks)} 块。当前 {output_path} 。')

            # # debug
            # start_time = time.time()
            # statistic_dict_list = process_tasks(task_chunks[0], df, is_filter)
            # result = [x for x in statistic_dict_list if x is not None]
            # result_df = pd.DataFrame(result)
            # print(f'单块任务耗时 {time.time() - start_time:.2f} 秒。')

            statistic_dict_list = []
            pool_processes = max(1, multiprocessing.cpu_count())
            with multiprocessing.Pool(processes=pool_processes) as pool:
                results = pool.starmap(process_tasks, [(chunk, df, is_filter) for chunk in task_chunks])
            for res in results:
                statistic_dict_list.extend(res)
            statistic_dict_list = [x for x in statistic_dict_list if x is not None]

            statistic_df = pd.DataFrame(statistic_dict_list)
            statistic_df.to_csv(output_path, index=False)
            print(f'结果已保存到 {output_path} 当前时间 {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')


def gen_breakthrough_signal(data_path='temp/TON_1m_2000.csv'):
    """
    主函数：
      1. 加载 CSV 中原始数据（只保留 timestamp, open, high, low, close 五列）
      2. 指定周期范围（start_period, end_period, step）
      3. 调用 backtest_breakthrough_strategy 进行回测
    """
    base_name = os.path.basename(data_path)
    is_filter = True

    # # debug
    # df = pd.read_csv(data_path)
    # long_column = '2584_high_long'
    # short_column = '2849_low_short'
    # signal_cache = {}
    # # df = df[-50000:]
    # get_detail_backtest_result(df, long_column, short_column, signal_cache, is_filter)

    df = pd.read_csv(data_path)
    needed_columns = ['timestamp', 'open', 'high', 'low', 'close']
    df = df[needed_columns]
    # 计算每一行的涨跌幅
    df['chg'] = df['close'].pct_change() * 100
    print(
        f'开始回测 {base_name} ... 长度 {df.shape[0]} 当前时间 {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    backtest_breakthrough_strategy(df, base_name, is_filter)


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

def choose_good_strategy_debug(inst_id='BTC'):
    # df = pd.read_csv('temp/temp.csv')
    # count_L()
    # 找到temp下面所有包含False的文件
    file_list = os.listdir('temp')
    file_list = [file for file in file_list if 'True' in file and inst_id in file and '_is_filter-True_is_reverse-Falsepart' in file and 'close' not in file]
    # file_list = file_list[0:1]
    df_list = []
    df_map = {}
    for file in file_list:
        file_key = file.split('_')[4]
        df = pd.read_csv(f'temp/{file}')
        # 删除monthly_net_profit_detail和monthly_trade_count_detail两列
        # df = df.drop(columns=['monthly_net_profit_detail', 'monthly_trade_count_detail','total_count','trade_rate','profit_rate','max_loss_start_time','max_loss_end_time','max_profit_start_time','max_profit_end_time'])
        # df = df.drop(columns=['total_count','trade_rate','profit_rate','max_loss_start_time','max_loss_end_time','max_profit_start_time','max_profit_end_time'])


        # 去除最大的偶然利润
        # df['net_profit_rate'] = df['net_profit_rate'] - 1 * df['max_profit']
        # df['avg_profit_rate'] = df['net_profit_rate'] / df['kai_count'] * 100
        # df['max_beilv'] = df['net_profit_rate'] / df['max_profit']
        # df['loss_beilv'] = -df['net_profit_rate'] / df['max_consecutive_loss']
        # df['score'] = (df['true_profit_std']) / df['avg_profit_rate'] * 100

        # df = df[(df['is_reverse'] == False)]
        # df = add_reverse(df)
        # df['kai_period'] = df['kai_column'].apply(lambda x: int(x.split('_')[0]))
        # df['pin_period'] = df['pin_column'].apply(lambda x: int(x.split('_')[0]))

        df['filename'] = file.split('_')[5]
        df['profit_risk_score_con'] = -df['net_profit_rate'] / df['max_consecutive_loss'] * df['net_profit_rate']
        df['profit_risk_score'] = -df['net_profit_rate'] / df['fu_profit_sum'] * df['net_profit_rate']
        df['profit_risk_score_pure'] = -df['net_profit_rate'] / df['fu_profit_sum']
        # df['pin_side'] = df['pin_column'].apply(lambda x: x.split('_')[-1])
        # 删除kai_column和pin_column中不包含 ma的行
        # df = df[(df['kai_column'].str.contains('ma')) & (df['pin_column'].str.contains('ma'))]
        # 删除kai_column和pin_column中包含 abs的行
        # df = df[~(df['kai_column'].str.contains('abs')) & ~(df['pin_column'].str.contains('abs'))]

        # df = df[(df['true_profit_std'] < 10)]
        df = df[(df['max_consecutive_loss'] > -20)]
        # df = df[(df['pin_side'] != df['kai_side'])]
        # df = df[(df['profit_risk_score_pure'] > 1)]
        df = df[(df['net_profit_rate'] > 50)]
        # df = df[(df['avg_profit_rate'] > 10)]

        # df = df[(df['monthly_net_profit_std'] < 10)]
        # df = df[(df['same_count_rate'] < 1)]
        # df = df[(df['same_count_rate'] < 1)]
        # df['monthly_trade_std_score'] = df['monthly_trade_std'] / (df['kai_count']) * 22
        #
        # df['monthly_net_profit_std_score'] = df['monthly_net_profit_std'] / (df['net_profit_rate']) * 22
        # df['monthly_avg_profit_std_score'] = df['monthly_avg_profit_std'] / (df['avg_profit_rate']) * 100
        # df = df[(df['monthly_net_profit_std_score'] < 50)]
        # df = df[(df['score'] > 2)]
        # df = df[(df['avg_profit_rate'] > 5)]
        # df = df[(df['kai_side'] == 'short')]

        df = df[(df['hold_time_mean'] < 5000)]
        # df = df[(df['max_beilv'] > 5)]
        # df = df[(df['loss_beilv'] > 1)]
        # df = df[(df['kai_count'] > 50)]
        # df = df[(df['same_count_rate'] < 1)]
        # df = df[(df['pin_period'] < 50)]
        if file_key not in df_map:
            df_map[file_key] = []
        # df['score'] = df['max_consecutive_loss']
        # df['score1'] = df['avg_profit_rate'] / (df['hold_time_mean'] + 20) * 1000
        # df['score2'] = df['avg_profit_rate'] / (
        #         df['hold_time_mean'] + 20) * 1000 * (df['trade_rate'] + 0.001)
        # df['score3'] = df['avg_profit_rate'] * (df['trade_rate'] + 0.0001)
        # df['score4'] = (df['trade_rate'] + 0.0001) / df['loss_rate']
        # loss_rate_max = df['loss_rate'].max()
        # loss_time_rate_max = df['loss_time_rate'].max()
        # avg_profit_rate_max = df['avg_profit_rate'].max()
        # max_beilv_max = df['max_beilv'].max()
        # df['loss_score'] = 5 * (loss_rate_max - df['loss_rate']) / loss_rate_max + 1 * (loss_time_rate_max - df['loss_time_rate']) / loss_time_rate_max - 1 * (avg_profit_rate_max - df['avg_profit_rate']) / avg_profit_rate_max

        # # 找到所有包含failure_rate_的列，然后计算平均值
        # failure_rate_columns = [column for column in df.columns if 'failure_rate_' in column]
        # df['failure_rate_mean'] = df[failure_rate_columns].mean(axis=1)
        #
        # df['loss_score'] = 1 - df['loss_rate']
        #
        # df['beilv_score'] = 0 - (max_beilv_max - df['max_beilv']) / max_beilv_max - (
        #             avg_profit_rate_max - df['avg_profit_rate']) / avg_profit_rate_max
        df_map[file_key].append(df)
    for key in df_map:
        df = pd.concat(df_map[key])
        df_list.append(df)
        return df

    temp = pd.merge(df_list[0], df_list[1], on=['kai_side', 'kai_column', 'pin_column'], how='inner')
    # 需要计算的字段前缀
    fields = ['avg_profit_rate', 'net_profit_rate', 'max_beilv']

    # 遍历字段前缀，统一计算
    for field in fields:
        x_col = f"{field}_x"
        y_col = f"{field}_y"

        temp[f"{field}_min"] = temp[[x_col, y_col]].min(axis=1)
        temp[f"{field}_mean"] = temp[[x_col, y_col]].mean(axis=1)
        temp[f"{field}_plus"] = temp[x_col] + temp[y_col]
        temp[f"{field}_cha"] = temp[x_col] - temp[y_col]
        temp[f"{field}_mult"] = np.where(
            (temp[x_col] < 0) & (temp[y_col] < 0),
            0,  # 如果两个都小于 0，则赋值 0
            temp[x_col] * temp[y_col]  # 否则正常相乘
        )

    # temp = temp[(temp['avg_profit_rate_min'] > 0)]
    # temp.to_csv('temp/temp.csv', index=False)
    return temp


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
        try:
            relevant_cols = [col for col in merged_df.columns if col.startswith(prefix)]  # 找到所有相关列

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


def filtering(origin_good_df, target_column, sort_key, threshold):
    """
    对 DataFrame 进行预过滤，思路如下：
      1. 按照 target_column 分组
      2. 每个分组内部，根据 sort_key 降序排序（更优的记录先保留）
      3. 每两行计算相关性，如果相关性大于 threshold，则删除 sort_key 较小（即后出现）的行

    参数:
      origin_good_df: pandas.DataFrame，原始数据
      target_column: str，用于分组的列名
      sort_key: str，用于比较优先级的列名，值较大者优先保留
      threshold: float，相关性阈值，若两行的相关性大于该值，则认为两行高度相关

    返回:
      filtered_df: pandas.DataFrame，过滤后保留的记录
    """
    filtered_groups = []  # 存储每个分组过滤后的 DataFrame

    # 按 target_column 分组
    for group_value, group_df in origin_good_df.groupby(target_column):
        # 按 sort_key 降序排序（大值优先）
        group_sorted = group_df.sort_values(by=sort_key, ascending=False)
        keep_rows = []  # 用于保存本组中保留的行（记录 Series）

        # 遍历排序后的每一行
        for idx, row in group_sorted.iterrows():
            drop_flag = False
            # 与已保留的每一行两两比较相关性
            for kept_row in keep_rows:
                # 计算两行相关性，排除掉 target_column 和 sort_key 列
                corr = calculate_row_correlation(row, kept_row)
                if corr > threshold:
                    # 若相关性大于阈值，则当前行与已有行高度相关，且当前行的 sort_key 较小（因为排序中的后续行），直接舍弃当前行
                    drop_flag = True
                    break
            if not drop_flag:
                keep_rows.append(row)

        # 将本组的保留行合并为 DataFrame
        if keep_rows:
            filtered_groups.append(pd.DataFrame(keep_rows))

    # 合并所有组的过滤结果
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
    origin_good_df = filtering(origin_good_df, 'kai_count', 'net_profit_rate', 90)
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

def debug():
    # good_df = pd.read_csv('temp/final_good.csv')

    origin_good_df_list = []
    # inst_id_list = ['BTC', 'ETH', 'SOL']
    # for inst_id in inst_id_list:
    #     origin_good_df = choose_good_strategy_debug(inst_id)
    #     origin_good_df.to_csv(f'temp/{inst_id}_df.csv', index=False)
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
    # sort_key = 'final_score'
    # sort_key = 'stability_score'
    # sort_key = 'profit_risk_score'
    # sort_key = 'monthly_net_profit_min'
    # sort_key = 'monthly_net_profit_std_score'
    sort_key = 'profit_risk_score'
    # sort_key = 'top_profit_ratio'
    range_size = 1
    # sort_key = 'max_consecutive_loss'
    # origin_good_df = choose_good_strategy_debug('')
    inst_id_list = ['SOL', 'ETH', 'SOL', 'TON', 'DOGE', 'XRP', 'PEPE']
    for inst_id in inst_id_list:
        # gen_search_param(inst_id)
        # origin_good_df = pd.read_csv(f'temp/{inst_id}_final_good.csv')
        # origin_good_df = pd.read_csv(f'temp/{inst_id}_df.csv')
        # origin_good_df = origin_good_df[(origin_good_df['hold_time_mean'] < 10000)]
        # origin_good_df['hold_time_score'] = origin_good_df['hold_time_std'] / origin_good_df['hold_time_mean']
        # origin_good_df['loss_score'] = 1 - origin_good_df['loss_rate'] - origin_good_df['loss_time_rate']
        # origin_good_df = origin_good_df[(origin_good_df['loss_score'] > 0)]
        # # good_df = pd.read_csv('temp/final_good.csv')




        # origin_good_df = choose_good_strategy_debug(inst_id)
        # origin_good_df = calculate_final_score(origin_good_df)
        # origin_good_df.to_csv(f'temp/{inst_id}_origin_good_op_false.csv', index=False)
        origin_good_df = pd.read_csv(f'temp/{inst_id}_origin_good_op_false.csv')
        # origin_good_df = pd.read_csv(f'temp/{inst_id}_origin_good_op_false.csv')
        # origin_good_df = pd.concat([origin_good_df, origin_good_df1])
        # origin_good_df[sort_key] = -origin_good_df[sort_key]
        # 删除kai_column和pin_column中包含 macross的行
        # origin_good_df = origin_good_df[~origin_good_df['kai_column'].str.contains('abs') & ~origin_good_df['pin_column'].str.contains('macross')]
        # origin_good_df['zhen_fu_mean_score'] = -origin_good_df['zhen_profit_mean'] / origin_good_df['fu_profit_mean']
        origin_good_df['monthly_trade_std_score'] = origin_good_df['monthly_trade_std'] / origin_good_df['kai_count'] * 22 * origin_good_df['active_month_ratio']
        # origin_good_df = origin_good_df[(origin_good_df['monthly_trade_std_score'] < 0.3)]
        # origin_good_df = origin_good_df[(origin_good_df['profit_risk_score'] > 700)]
        origin_good_df = origin_good_df[(origin_good_df['hold_time_mean'] < 3000)]
        # origin_good_df = origin_good_df[(origin_good_df['hold_time_std'] < origin_good_df['hold_time_mean'])]
        # origin_good_df = origin_good_df[(origin_good_df['max_consecutive_loss'] > -10)]
        # origin_good_df = origin_good_df[(origin_good_df['stability_score'] > 0)]
        origin_good_df = origin_good_df[(origin_good_df['avg_profit_rate'] > 50)]
        origin_good_df = origin_good_df[(origin_good_df['net_profit_rate'] > 300)]
        # good_df = pd.read_csv('temp/final_good.csv')


        # # kai_column和pin_column相同的时候取第一行
        # origin_good_df = origin_good_df.drop_duplicates(subset=['kai_column', 'pin_column'], keep='first')
        # good_df = origin_good_df.sort_values(sort_key, ascending=False)
        # long_good_strategy_df = good_df[good_df['kai_side'] == 'long']
        # short_good_strategy_df = good_df[good_df['kai_side'] == 'short']
        #
        # # 将long_good_strategy_df按照net_profit_rate_mult降序排列
        # long_good_select_df = select_best_rows_in_ranges(long_good_strategy_df, range_size=range_size,
        #                                                  sort_key=sort_key, range_key=range_key)
        # short_good_select_df = select_best_rows_in_ranges(short_good_strategy_df, range_size=range_size,
        #                                                   sort_key=sort_key, range_key=range_key)
        # good_df = pd.concat([long_good_select_df, short_good_select_df])


        good_df = origin_good_df.sort_values(sort_key, ascending=False)
        # 重置good_df的索引
        good_df = good_df.reset_index(drop=True)
        # good_df = good_df.sort_values(by=sort_key, ascending=True)
        # good_df = good_df.drop_duplicates(subset=['kai_column', 'kai_side'], keep='first')

        # good_df.to_csv('temp/final_good.csv', index=False)
        result, good_df, df = find_all_valid_groups(good_df, 30)
        good_df.to_csv('temp/final_good.csv', index=False)
        return


        good_df = pd.read_csv('temp/final_good.csv')

        is_filter = True
        is_detail = False
        file_list = []
        file_list.append(f'kline_data/origin_data_1m_50000_{inst_id}-USDT-SWAP.csv')
        file_list.append(f'kline_data/origin_data_1m_40000_{inst_id}-USDT-SWAP.csv')
        # file_list.append(f'kline_data/origin_data_1m_30000_{inst_id}-USDT-SWAP.csv')
        # file_list.append(f'kline_data/origin_data_1m_20000_{inst_id}-USDT-SWAP.csv')
        file_list.append(f'kline_data/origin_data_1m_10000_{inst_id}-USDT-SWAP.csv')
        file_list.append(f'kline_data/origin_data_1m_3000_{inst_id}-USDT-SWAP.csv')
        file_list.append(f'kline_data/origin_data_1m_2000_{inst_id}-USDT-SWAP.csv')
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
            good_df = good_df.reset_index(drop=True)
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
                kai_data_df, statistic_dict = get_detail_backtest_result_op(22, df, long_column, short_column, signal_cache,
                                                                            is_filter, is_detail, is_reverse)
                # 为每一行添加统计数据，需要修改到原始数据中
                # 直接修改 `good_df` 中的相应列
                good_df.at[index, 'kai_count_new'] = statistic_dict['kai_count']
                good_df.at[index, 'trade_rate_new'] = statistic_dict['trade_rate']
                good_df.at[index, 'hold_time_mean_new'] = statistic_dict['hold_time_mean']
                good_df.at[index, 'net_profit_rate_new'] = statistic_dict['net_profit_rate']
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
                good_df['max_optimal_profit_rate'] = good_df['max_optimal_profit_cha'] / good_df['kai_count_new']
                good_df['min_optimal_profit_cha'] = good_df['min_optimal_profit'] - good_df['net_profit_rate_new']
                good_df['min_optimal_profit_rate'] = good_df['min_optimal_profit_cha'] / good_df['kai_count_new']
            statistic_df = pd.DataFrame(statistic_dict_list)
            statistic_df_list.append(statistic_df)
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
            # good_df[good_df['index'].isin([339, 990])]
            # good_df_list[0][good_df_list[0]['index'].isin([339, 990])]
            result = find_all_valid_groups(good_df, 100)



            print(inst_id)
    merged_df, temp_df = merge_dataframes(statistic_df_list)
    print()


def example():
    debug()


if __name__ == '__main__':
    example()
