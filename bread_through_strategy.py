"""
突破策略的信号生成以及回测（优化版）

优化说明：
1. 只加载原始的必要数据（timestamp, open, high, low, close），不提前生成所有周期的信号。
2. 在 process_tasks() 内，根据每个任务（信号的 pair）按需计算对应的突破信号，
   并采用 signal_cache 以避免同一信号的重复计算，从而降低内存占用。
3. 保持功能不变，回测结果与原始代码一致。
"""

import multiprocessing
import os
import time
import traceback
from itertools import product

import numpy as np
import pandas as pd
from numba import njit


def compute_signal(df, col_name):
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
        return signal_series, price_series

    elif signal_type == 'continue':

        if direction == "long":
            rolling_check = df['chg'].rolling(window=period).apply(lambda x: (x > 0).all(), raw=True)
        else:
            rolling_check = df['chg'].rolling(window=period).apply(lambda x: (x < 0).all(), raw=True)
        valid_count = df['chg'].rolling(window=period).count()  # 统计窗口内的有效值数量
        signal_series = (rolling_check == 1) & (valid_count == period)  # 仅当窗口填满时才允许 True
        price_series = df['close']  # 价格取 close
        return signal_series, price_series
    elif signal_type == 'abs':
        abs_value = int(parts[2])
        if direction == "long":
            # 找到不包含当前行的前period行中 low 的最小值
            min_low_series = df['low'].shift(1).rolling(window=period).min()
            # 计算该值上涨 abs_value% 后的价格
            target_price = min_low_series * (1 + abs_value / 100)
            # 如果当前行的 high 超过该价格，则发出信号
            signal_series = df['high'] > target_price
            price_series = target_price  # 价格设置为该价格
        else:  # direction == "short"
            # 找到不包含当前行的前period行中 high 的最大值
            max_high_series = df['high'].shift(1).rolling(window=period).max()
            # 计算该值下跌 abs_value% 后的价格
            target_price = max_high_series * (1 - abs_value / 100)
            # 如果当前行的 low 低于该价格，则发出信号
            signal_series = df['low'] < target_price
            price_series = target_price  # 价格设置为该价格

        return signal_series, price_series
    else:
        raise ValueError(f"未知的信号类型: {signal_type}")



def calculate_max_sequence(kai_data_df):
    """
    计算最大连续亏损（子序列的最小和）及对应的起始和结束索引。
    """
    true_profit_series = kai_data_df['true_profit']
    max_loss = 0
    current_loss = 0
    start_index = None
    temp_start_index = None
    end_index = None
    trade_count = 0  # 初始化交易计数器
    max_sequence_length = 0
    for i, profit in enumerate(true_profit_series):
        if current_loss == 0:
            temp_start_index = kai_data_df.index[i]
            trade_count = 0  # 在新的潜在亏损序列开始时，重置 trade_count (也可以不重置，在current_loss > 0 时重置更清晰)

        current_loss += profit
        trade_count += 1  # 每次迭代都增加交易计数

        if current_loss < max_loss:
            max_loss = current_loss
            start_index = temp_start_index
            end_index = kai_data_df.index[i]
            max_sequence_length = trade_count  # 直接使用 trade_count 更新最大子序列长度

        if current_loss > 0:
            current_loss = 0
            trade_count = 0  # 当盈利出现时，重置亏损和交易计数

    return max_loss, start_index, end_index, max_sequence_length


def calculate_max_profit(kai_data_df):
    """
    计算最大连续盈利（子序列的最大和）及对应的起始和结束索引。
    """
    true_profit_series = kai_data_df['true_profit']
    max_profit = 0
    current_profit = 0
    start_index = None
    temp_start_index = None
    end_index = None
    trade_count = 0  # 初始化交易计数器
    max_sequence_length = 0

    for i, profit in enumerate(true_profit_series):
        if current_profit == 0:
            temp_start_index = kai_data_df.index[i]
            trade_count = 0  # 在新的潜在盈利序列开始时，重置 trade_count (也可以不重置，在current_profit < 0 时重置更清晰)

        current_profit += profit
        trade_count += 1  # 每次迭代都增加交易计数

        if current_profit > max_profit:
            max_profit = current_profit
            start_index = temp_start_index
            end_index = kai_data_df.index[i]
            max_sequence_length = trade_count  # 直接使用 trade_count 更新最大子序列长度

        if current_profit < 0:
            current_profit = 0
            trade_count = 0  # 当亏损出现时，重置盈利和交易计数

    return max_profit, start_index, end_index, max_sequence_length


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


def get_detail_backtest_result_op(df, kai_column, pin_column, signal_cache, is_filter=True):
    """
    根据传入的信号列（开仓信号 kai_column 与平仓信号 pin_column），在原始 df 上动态生成信号，
    进行回测计算，并返回对应的明细 DataFrame 和统计信息（statistic_dict）。

    参数:
        df: 原始数据的 DataFrame，要求按时间索引排序，且包含 'timestamp' 列。
        kai_column: 开仓信号所在的列名称。
        pin_column: 平仓信号所在的列名称。
        signal_cache: 字典，用于缓存信号及相关数据，避免重复计算。
        is_filter: 是否过滤重复平仓时间的交易（默认 True）。

    返回:
        kai_data_df: 包含开仓和平仓、价格、收益率等信息的明细 DataFrame。
        statistic_dict: 回测统计数据的字典，其中包含各种指标信息。
    """

    # 判断开仓方向
    kai_side = 'long' if 'long' in kai_column.lower() else 'short'

    # 内部函数：从缓存中获取信号数据，否则计算并缓存
    def get_signal_and_price(column):
        if column in signal_cache:
            return signal_cache[column]
        # 假设 compute_signal 函数返回 (signal_mask, price_series)
        signal_data = compute_signal(df, column)
        signal_cache[column] = signal_data
        return signal_data

    # 获取开仓和平仓信号及对应价格序列
    kai_signal, kai_price_series = get_signal_and_price(kai_column)
    pin_signal, pin_price_series = get_signal_and_price(pin_column)
    # 如果信号为空，则直接返回
    if kai_signal.sum() == 0 or pin_signal.sum() == 0:
        return None, None

    # 筛选出对应的开仓和平仓数据行（建议开仓复制，平仓暂不复制以节省内存）
    kai_data_df = df.loc[kai_signal].copy()
    pin_data_df = df.loc[pin_signal]

    # 添加价格字段，注意这里利用 numpy 数组确保索引对齐处理
    kai_data_df['kai_price'] = np.asarray(kai_price_series[kai_signal])
    pin_data_df = pin_data_df.assign(pin_price=np.asarray(pin_price_series[pin_signal]))

    # 利用 searchsorted 找到每个开仓信号对应的平仓信号（平仓时间严格大于开仓时间）
    # 注意：假设两个 DataFrame 均按时间索引排序
    kai_data_df['pin_index'] = pin_data_df.index.searchsorted(kai_data_df.index, side='right')
    valid_mask = kai_data_df['pin_index'] < len(pin_data_df)
    kai_data_df = kai_data_df.loc[valid_mask].copy()  # 重新 copy 确保数据一致性
    kai_data_df['kai_side'] = kai_side

    # 根据计算出的索引，找到匹配的平仓数据
    pin_indices = kai_data_df['pin_index'].to_numpy()
    matched_pin = pin_data_df.iloc[pin_indices]
    kai_data_df['pin_price'] = np.asarray(matched_pin['pin_price'])
    kai_data_df['pin_time'] = np.asarray(matched_pin['timestamp'])
    kai_data_df['hold_time'] = matched_pin.index.to_numpy() - kai_data_df.index.to_numpy()

    # 计算收益率（向量化计算）
    if kai_side == 'long':
        profit = (kai_data_df['pin_price'] - kai_data_df['kai_price']) / kai_data_df['kai_price'] * 100
    else:
        profit = (kai_data_df['kai_price'] - kai_data_df['pin_price']) / kai_data_df['pin_price'] * 100
    kai_data_df['profit'] = profit.round(4)

    # 考虑交易成本（此处固定每笔 0.07 的成本）
    kai_data_df['true_profit'] = kai_data_df['profit'] - 0.07

    # 若需要过滤重复平仓时间的交易，则按 'timestamp' 排序后去重
    if is_filter:
        kai_data_df = kai_data_df.sort_values('timestamp').drop_duplicates('pin_time', keep='first')

    # 计算失败率（假设 calculate_failure_rates 接收 DataFrame 与列表参数）
    failure_rate_result = calculate_failure_rates(kai_data_df, list(range(1, 11)))

    # 收集交易数及全局数据量
    trade_count = kai_data_df.shape[0]
    total_count = df.shape[0]
    profit_sum = kai_data_df['profit'].sum()

    max_single_profit = kai_data_df['true_profit'].max()
    min_single_profit = kai_data_df['true_profit'].min()

    true_profit_std = kai_data_df['true_profit'].std()
    true_profit_mean = kai_data_df['true_profit'].mean() if trade_count > 0 else 0
    true_profit_mean = round(true_profit_mean * 100, 4)

    # 计算最大连续亏损相关指标（假设 calculate_max_sequence 返回：loss, start_idx, end_idx, count ）
    max_loss, max_loss_start_idx, max_loss_end_idx, loss_trade_count = calculate_max_sequence(kai_data_df)
    max_loss_start_time = kai_data_df.loc[max_loss_start_idx, 'timestamp'] if max_loss_start_idx is not None else None
    max_loss_end_time = kai_data_df.loc[max_loss_end_idx, 'timestamp'] if max_loss_end_idx is not None else None
    max_loss_hold_time = (max_loss_end_idx - max_loss_start_idx) if (
                max_loss_start_idx is not None and max_loss_end_idx is not None) else None

    # 计算最大连续盈利相关指标（假设 calculate_max_profit 返回：profit, start_idx, end_idx, count ）
    max_profit, max_profit_start_idx, max_profit_end_idx, profit_trade_count = calculate_max_profit(kai_data_df)
    max_profit_start_time = kai_data_df.loc[
        max_profit_start_idx, 'timestamp'] if max_profit_start_idx is not None else None
    max_profit_end_time = kai_data_df.loc[max_profit_end_idx, 'timestamp'] if max_profit_end_idx is not None else None
    max_profit_hold_time = (max_profit_end_idx - max_profit_start_idx) if (
                max_profit_start_idx is not None and max_profit_end_idx is not None) else None

    # 根据 true_profit 划分盈利和亏损交易
    profit_df = kai_data_df[kai_data_df['true_profit'] > 0]
    loss_df = kai_data_df[kai_data_df['true_profit'] < 0]
    net_profit_rate = kai_data_df['true_profit'].sum()

    loss_rate = (loss_df.shape[0] / trade_count) if trade_count > 0 else 0
    loss_time = loss_df['hold_time'].sum() if not loss_df.empty else 0
    profit_time = profit_df['hold_time'].sum() if not profit_df.empty else 0
    loss_time_rate = (loss_time / (loss_time + profit_time)) if (loss_time + profit_time) > 0 else 0

    trade_rate = round(100 * trade_count / total_count, 4) if total_count > 0 else 0
    hold_time_mean = kai_data_df['hold_time'].mean() if trade_count > 0 else 0

    # 汇总所有统计指标到字典
    statistic_dict = {
        'kai_side': kai_side,
        'kai_column': kai_column,
        'pin_column': pin_column,
        'kai_count': trade_count,
        'total_count': total_count,
        'trade_rate': trade_rate,
        'hold_time_mean': hold_time_mean,
        'loss_rate': loss_rate,
        'loss_time_rate': loss_time_rate,
        'profit_rate': profit_sum,
        'max_profit': max_single_profit,
        'min_profit': min_single_profit,
        'cost_rate': trade_count * 0.07,
        'net_profit_rate': net_profit_rate,
        'avg_profit_rate': true_profit_mean,
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
    }

    # 将失败率指标添加到统计字典中
    for key, value in failure_rate_result.items():
        statistic_dict[f'failure_rate_{key}'] = value

    return kai_data_df, statistic_dict


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
        # 再做「开空」方向回测（此时互换信号）
        _, stat_short = get_detail_backtest_result_op(df, short_column, long_column, signal_cache, is_filter)
        results.append(stat_short)
    print(f"处理 {len(task_chunk) * 2} 个任务，耗时 {time.time() - start_time:.2f} 秒。")
    return results


def gen_peak_signal_name(start_period, end_period, step):
    """
    生成 peak 信号的列名列表。
    :param start_period:
    :param end_period:
    :param step:
    :return:
    """
    period_list = range(start_period, end_period, step)
    long_columns = [f"peak_{period}_high_long" for period in period_list]
    short_columns = [f"peak_{period}_low_short" for period in period_list]
    key_name = f'peak_{start_period}_{end_period}_{step}'
    return long_columns, short_columns, key_name

def gen_continue_signal_name(start_period, end_period, step):
    """"""
    period_list = range(start_period, end_period, step)
    long_columns = [f"continue_{period}_high_long" for period in period_list]
    short_columns = [f"continue_{period}_low_short" for period in period_list]
    key_name = f'continue_{start_period}_{end_period}_{step}'
    return long_columns, short_columns, key_name

def gen_abs_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    """"""
    period_list = range(start_period, end_period, step)
    period_list1 = range(start_period1, end_period1, step1)
    long_columns = [f"abs_{period}_{period1}_high_long" for period in period_list for period1 in period_list1 if period >= period1]
    short_columns = [f"abs_{period}_{period1}_low_short" for period in period_list for period1 in period_list1 if period >= period1]
    key_name = f'abs_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}'
    return long_columns, short_columns, key_name

def backtest_breakthrough_strategy(df, base_name, is_filter):
    """
    回测函数：基于原始数据 df 和指定周期范围，
    生成所有 (kai, pin) 信号对（kai 信号命名为 "{period}_high_long"，pin 信号命名为 "{period}_low_short"），
    使用多进程并行调用 process_tasks() 完成回测，并将统计结果保存到 CSV 文件。
    """
    peak_long_columns, peak_short_columns, peak_key_name = gen_peak_signal_name(1, 1000, 1)
    continue_long_columns, continue_short_columns, continue_key_name = gen_continue_signal_name(1, 20, 1)
    abs_long_columns, abs_short_columns, abs_key_name = gen_abs_signal_name(1, 100, 2, 1, 10, 1)
    long_columns = peak_long_columns + continue_long_columns + abs_long_columns
    short_columns = peak_short_columns + continue_short_columns + abs_short_columns
    task_list = list(product(long_columns, short_columns))
    big_chunk_size = 100000
    big_task_chunks = [task_list[i:i + big_chunk_size] for i in range(0, len(task_list), big_chunk_size)]
    print(f'共有 {len(task_list)} 个任务，分为 {len(big_task_chunks)} 大块。')
    for i, task_chunk in enumerate(big_task_chunks):
        # 将task_list打乱顺序
        output_path = f"temp/statistic_{base_name}_{peak_key_name}_{continue_key_name}_{abs_key_name}is_filter-{is_filter}_part{i}.csv"
        if os.path.exists(output_path):
            print(f'已存在 {output_path}')
            continue
        task_chunk = task_chunk.copy()
        np.random.shuffle(task_chunk)

        # 将任务分块，每块包含一定数量的任务
        chunk_size = 100
        task_chunks = [task_chunk[i:i + chunk_size] for i in range(0, len(task_chunk), chunk_size)]
        print(f'共有 {len(task_chunk)} 个任务，分为 {len(task_chunks)} 块。')

        # # debug
        # result = process_tasks(task_chunks[0], df, is_filter)

        statistic_dict_list = []
        pool_processes = max(1, multiprocessing.cpu_count())
        with multiprocessing.Pool(processes=pool_processes) as pool:
            results = pool.starmap(process_tasks, [(chunk, df, is_filter) for chunk in task_chunks])
        for res in results:
            statistic_dict_list.extend(res)
        # 去除空值
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


def choose_good_strategy_debug(inst_id='BTC'):
    # df = pd.read_csv('temp/temp.csv')
    # count_L()
    # 找到temp下面所有包含False的文件
    file_list = os.listdir('temp')
    file_list = [file for file in file_list if 'True' in file and inst_id in file and '_1000_1_continue_1_20_1_' in file and '1m' in file and 'continue_' in file]
    df_list = []
    df_map = {}
    for file in file_list:
        file_key = file.split('_')[4]
        df = pd.read_csv(f'temp/{file}')


        # 去除最大的偶然利润
        # df['net_profit_rate'] = df['net_profit_rate'] - 1 * df['max_profit']
        # df['avg_profit_rate'] = df['net_profit_rate'] / df['kai_count'] * 100
        df['max_beilv'] = df['net_profit_rate'] / df['max_profit']
        # df['kai_period'] = df['kai_column'].apply(lambda x: int(x.split('_')[0]))
        # df['pin_period'] = df['pin_column'].apply(lambda x: int(x.split('_')[0]))

        df['filename'] = file.split('_')[5]
        # df = df[(df['max_consecutive_loss'] > -50)]
        df = df[(df['avg_profit_rate'] > 0)]
        # df = df[(df['hold_time_mean'] < 10000)]
        # df = df[(df['max_beilv'] > 0)]
        # df = df[(df['kai_count'] > 10)]
        # df = df[(df['pin_period'] < 50)]
        if file_key not in df_map:
            df_map[file_key] = []
        # df['score'] = df['avg_profit_rate']
        # df['score1'] = df['avg_profit_rate'] / (df['hold_time_mean'] + 20) * 1000
        # df['score2'] = df['avg_profit_rate'] / (
        #             df['hold_time_mean'] + 20) * 1000 * (df['trade_rate'] + 0.001)
        # df['score3'] = df['avg_profit_rate'] * (df['trade_rate'] + 0.0001)
        # df['score4'] = (df['trade_rate'] + 0.0001) / df['loss_rate']
        loss_rate_max = df['loss_rate'].max()
        loss_time_rate_max = df['loss_time_rate'].max()
        avg_profit_rate_max = df['avg_profit_rate'].max()
        max_beilv_max = df['max_beilv'].max()
        # df['loss_score'] = 5 * (loss_rate_max - df['loss_rate']) / loss_rate_max + 1 * (loss_time_rate_max - df['loss_time_rate']) / loss_time_rate_max - 1 * (avg_profit_rate_max - df['avg_profit_rate']) / avg_profit_rate_max


        # 找到所有包含failure_rate_的列，然后计算平均值
        failure_rate_columns = [column for column in df.columns if 'failure_rate_' in column]
        df['failure_rate_mean'] = df[failure_rate_columns].mean(axis=1)
        # 删除failure_rate_columns这些列
        df = df.drop(columns=failure_rate_columns)

        df['loss_score'] = 1 - df['loss_rate']

        # df['beilv_score'] = 0 - (max_beilv_max - df['max_beilv']) / max_beilv_max - (avg_profit_rate_max - df['avg_profit_rate']) / avg_profit_rate_max
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
    #生成一个空的DataFrame
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
    origin_good_df[f'{key_name}_cha_ratio'] = origin_good_df[f'{key_name}_cha'] / origin_good_df[f'{key_name}_max'] * 100
    origin_good_df[f'{key_name}_score'] = 1 - origin_good_df[f'{key_name}_cha_ratio']
    return f'{key_name}_score'

def debug():

    # good_df = pd.read_csv('temp/final_good.csv')

    # debug
    sort_key = 'kai_count_cha_score'
    range_size = 100
    range_key = 'kai_count_max'
    # sort_key = 'max_consecutive_loss'
    inst_id_list = ['BTC', 'ETH', 'SOL', 'TON', 'DOGE', 'XRP', 'PEPE']

    # origin_df_list = []
    # for inst_id in inst_id_list:
    #     origin_good_df = choose_good_strategy_debug(inst_id)
    #     origin_df_list.append(origin_good_df)
    # merged_df = merge_dataframes(origin_df_list)
    origin_good_df = pd.read_csv('temp/all_df_pure.csv')
    # origin_good_df = origin_good_df[(origin_good_df['net_profit_rate_min'] > 20)]
    sort_key = gen_score(origin_good_df, 'kai_count')
    # origin_good_df = choose_good_strategy_debug(inst_id)
    # 按照loss_score降序排列，选取前20行
    # origin_good_df = origin_good_df[(origin_good_df['kai_side'] == 'short')]
    good_df = origin_good_df.sort_values(sort_key, ascending=False)
    #
    # # 去除kai_column重复的行，只取第一行
    # good_df = good_df.drop_duplicates('kai_column', keep='first')
    # good_df = good_df.head(20)

    long_good_strategy_df = good_df[good_df['kai_side'] == 'long']
    short_good_strategy_df = good_df[good_df['kai_side'] == 'short']

    # 将long_good_strategy_df按照net_profit_rate_mult降序排列
    long_good_select_df = select_best_rows_in_ranges(long_good_strategy_df, range_size=range_size,
                                                     sort_key=sort_key, range_key=range_key)
    short_good_select_df = select_best_rows_in_ranges(short_good_strategy_df, range_size=range_size,
                                                      sort_key=sort_key, range_key=range_key)
    good_df = pd.concat([long_good_select_df, short_good_select_df])
    statistic_df_list = []
    for inst_id in inst_id_list:
        is_filter = True
        df = pd.read_csv(f'kline_data/origin_data_1m_50000_{inst_id}-USDT-SWAP.csv')
        # 计算每一行的涨跌幅
        df['chg'] = df['close'].pct_change() * 100
        signal_cache = {}
        statistic_dict_list = []
        for index, row in good_df.iterrows():
            long_column = row['kai_column']
            short_column = row['pin_column']
            # # long_column = '6_low_short'
            # # short_column = '3_high_long'
            # long_column = 'abs_2_1_high_long'
            # short_column = 'abs_2_1_low_short'
            kai_data_df, statistic_dict = get_detail_backtest_result_op(df, long_column, short_column, signal_cache, is_filter)
            statistic_dict_list.append(statistic_dict)
        statistic_df = pd.DataFrame(statistic_dict_list)
        statistic_df_list.append(statistic_df)
        print(inst_id)
    _, result_df = merge_dataframes(statistic_df_list)
    print()


def example():
    # debug()
    # choose_good_strategy()
    start_time = time.time()

    data_path_list = [
        'kline_data/origin_data_1m_10000000_BTC-USDT-SWAP.csv',
        # 'kline_data/origin_data_1m_86000_BTC-USDT-SWAP.csv',

        'kline_data/origin_data_1m_10000000_ETH-USDT-SWAP.csv',
        # 'kline_data/origin_data_1m_86000_ETH-USDT-SWAP.csv',

        'kline_data/origin_data_1m_10000000_SOL-USDT-SWAP.csv',
        # 'kline_data/origin_data_1m_86000_SOL-USDT-SWAP.csv',

        'kline_data/origin_data_1m_10000000_TON-USDT-SWAP.csv',
        # 'kline_data/origin_data_1m_86000_TON-USDT-SWAP.csv',

        'kline_data/origin_data_1m_10000000_DOGE-USDT-SWAP.csv',
        # 'kline_data/origin_data_1m_86000_DOGE-USDT-SWAP.csv',

        'kline_data/origin_data_1m_10000000_XRP-USDT-SWAP.csv',
        # 'kline_data/origin_data_1m_86000_XRP-USDT-SWAP.csv',

        'kline_data/origin_data_1m_10000000_PEPE-USDT-SWAP.csv',
        # 'kline_data/origin_data_1m_86000_PEPE-USDT-SWAP.csv'
    ]
    for data_path in data_path_list:
        try:
            gen_breakthrough_signal(data_path)
            print(f'{data_path} 总耗时 {time.time() - start_time:.2f} 秒。')
        except Exception as e:
            print(f'处理 {data_path} 时出错：{e}')


if __name__ == '__main__':
    example()