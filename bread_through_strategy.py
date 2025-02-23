"""
突破策略的信号生成以及回测（深度优化版）
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
    """
    计算给定列名对应的信号及价格序列，主要优化点：
      - 对于 continue 信号，采用向量化计算替换 lambda 滚动验证。
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
        return signal_series, price_series

    elif signal_type == 'continue':
        # 利用 rolling 求和来判断是否全正/全负
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
            signal_series = (df['high'].shift(1) <= target_price) & (df['high'] > target_price)
            price_series = target_price
        else:
            max_high_series = df['high'].shift(1).rolling(window=period).max()
            target_price = max_high_series * (1 - abs_value / 100)
            signal_series = (df['low'].shift(1) >= target_price) & (df['low'] < target_price)
            price_series = target_price
        return signal_series, price_series

    elif signal_type == 'ma':
        moving_avg = df['close'].shift(1).rolling(window=period).mean()  # 排除当前行
        if direction == "long":
            signal_series = (df['high'].shift(1) <= moving_avg) & (df['high'] > moving_avg)
        else:
            signal_series = (df['low'].shift(1) >= moving_avg) & (df['low'] < moving_avg)
        price_series = moving_avg
        return signal_series, price_series

    elif signal_type == 'macross':
        fast_period = int(parts[1])
        slow_period = int(parts[2])
        fast_ma = df['close'].rolling(window=fast_period).mean().shift(1)
        slow_ma = df['close'].rolling(window=slow_period).mean().shift(1)
        if direction == "long":
            signal_series = (fast_ma.shift(1) <= slow_ma.shift(1)) & (fast_ma > slow_ma)
        else:
            signal_series = (fast_ma.shift(1) >= slow_ma.shift(1)) & (fast_ma < slow_ma)
        price_series = df['close']
        return signal_series, price_series

    elif signal_type == 'rsi':
        period = int(parts[1])
        overbought = int(parts[2])
        oversold = int(parts[3])
        delta = df['close'].diff(1)
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
        price_series = df['close']
        return signal_series, price_series

    elif signal_type == 'relate':
        abs_value = float(parts[2])
        if direction == "long":
            min_low_series = df['low'].shift(1).rolling(window=period).min()
            max_high_series = df['high'].shift(1).rolling(window=period).max()
            target_price = min_low_series + abs_value / 100 * (max_high_series - min_low_series)
            signal_series = (df['high'].shift(1) <= target_price) & (df['high'] > target_price)
            price_series = target_price
        else:
            max_high_series = df['high'].shift(1).rolling(window=period).max()
            min_low_series = df['low'].shift(1).rolling(window=period).min()
            target_price = max_high_series - abs_value / 100 * (max_high_series - min_low_series)
            signal_series = (df['low'].shift(1) >= target_price) & (df['low'] < target_price)
            price_series = target_price
        return signal_series, price_series

    else:
        raise ValueError(f"未知的信号类型: {signal_type}")


@njit
def calculate_max_sequence_numba(series):
    """
    利用 numba 加速查找最大连续亏损序列。
    参数 series 为交易真实收益的 NumPy 数组。
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
    利用 numba 加速查找最大连续盈利序列。
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
    """
    向量化实现止盈和止损参数的优化。
    """
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

    # 只设置止盈时的模拟
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

    # 只设置止损时的模拟
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


def get_detail_backtest_result_op(df, kai_column, pin_column, signal_cache, is_filter=True, is_detail=False):
    """
    根据传入信号名获取详细回测结果：
      - 使用 signal_cache 避免重复计算
      - 利用 numba 加速关键计算
    """
    kai_side = 'long' if 'long' in kai_column.lower() else 'short'
    temp_dict = {}
    total_months = 22
    def get_signal_and_price(column):
        if column in signal_cache:
            return signal_cache[column]
        signal_data = compute_signal(df, column)
        signal_cache[column] = signal_data
        return signal_data

    kai_signal, kai_price_series = get_signal_and_price(kai_column)
    pin_signal, pin_price_series = get_signal_and_price(pin_column)

    if kai_signal.sum() < 100 or pin_signal.sum() < 100:
        return None, None

    kai_data_df = df.loc[kai_signal].copy()
    pin_data_df = df.loc[pin_signal].copy()
    # 缓存价格数据
    kai_prices = kai_price_series[kai_signal].to_numpy()
    pin_prices = pin_price_series[pin_signal].to_numpy()
    kai_data_df['kai_price'] = kai_prices
    pin_data_df['pin_price'] = pin_prices

    common_index = kai_data_df.index.intersection(pin_data_df.index)
    same_count = len(common_index)
    pin_count = len(pin_data_df)
    kai_count = len(kai_data_df)
    same_count_rate = round(100 * same_count / min(pin_count, kai_count), 4) if min(pin_count, kai_count) > 0 else 0
    if same_count_rate > 1:
        return None, None

    # 对 kai_data_df 中每个时间点，找到 pin_data_df 中最接近右侧的匹配项
    kai_idx_all = kai_data_df.index.to_numpy()
    pin_idx = pin_data_df.index.to_numpy()
    pin_indices = np.searchsorted(pin_idx, kai_idx_all, side='right')
    valid_mask = pin_indices < len(pin_idx)
    kai_data_df = kai_data_df.iloc[valid_mask].copy()
    kai_idx_valid = kai_idx_all[valid_mask]
    pin_indices_valid = pin_indices[valid_mask]
    matched_pin = pin_data_df.iloc[pin_indices_valid].copy()

    # 匹配价格、时间及持仓时长
    kai_data_df['pin_price'] = matched_pin['pin_price'].to_numpy()
    kai_data_df['pin_time'] = matched_pin['timestamp'].to_numpy()
    kai_data_df['hold_time'] = matched_pin.index.to_numpy() - kai_idx_valid

    if is_detail:
        # 缓存 df 中的 numpy 数组，避免重复转换
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

        if kai_side == 'long':
            kai_data_df['max_true_profit'] = (((kai_data_df['high_max'] - kai_data_df['kai_price']) / kai_data_df['kai_price'] * 100 - 0.07).round(4))
            kai_data_df['min_true_profit'] = (((kai_data_df['low_min'] - kai_data_df['kai_price']) / kai_data_df['kai_price'] * 100 - 0.07).round(4))
        else:
            kai_data_df['max_true_profit'] = (((kai_data_df['kai_price'] - kai_data_df['low_min']) / kai_data_df['kai_price'] * 100 - 0.07).round(4))
            kai_data_df['min_true_profit'] = (((kai_data_df['kai_price'] - kai_data_df['high_max']) / kai_data_df['kai_price'] * 100 - 0.07).round(4))

    if is_filter:
        kai_data_df = kai_data_df.sort_values('timestamp').drop_duplicates('pin_time', keep='first')

    # 根据 pin_time 映射更新 kai_price
    pin_price_map = kai_data_df.set_index('pin_time')['pin_price']
    mapped_prices = kai_data_df['timestamp'].map(pin_price_map)
    if same_count > 0 and not mapped_prices.isna().all():
        kai_data_df['kai_price'] = mapped_prices.combine_first(kai_data_df['kai_price'])

    # 向量化计算收益率
    if kai_side == 'long':
        kai_data_df['profit'] = ((kai_data_df['pin_price'] - kai_data_df['kai_price']) /
                                 kai_data_df['kai_price'] * 100).round(4)
    else:
        kai_data_df['profit'] = ((kai_data_df['kai_price'] - kai_data_df['pin_price']) /
                                 kai_data_df['pin_price'] * 100).round(4)
    kai_data_df['true_profit'] = kai_data_df['profit'] - 0.07

    trade_count = len(kai_data_df)
    total_count = len(df)
    profit_sum = kai_data_df['profit'].sum()

    max_single_profit = kai_data_df['true_profit'].max()
    min_single_profit = kai_data_df['true_profit'].min()

    if is_detail:
        max_single_profit = kai_data_df['max_true_profit'].max()
        min_single_profit = kai_data_df['min_true_profit'].min()
        if trade_count > 0:
            temp_dict = optimize_parameters(kai_data_df)

    true_profit_std = kai_data_df['true_profit'].std()
    true_profit_mean = kai_data_df['true_profit'].mean() * 100 if trade_count > 0 else 0

    profits_arr = kai_data_df['true_profit'].to_numpy()
    max_loss, max_loss_start_idx, max_loss_end_idx, loss_trade_count = calculate_max_sequence_numba(profits_arr)
    max_profit, max_profit_start_idx, max_profit_end_idx, profit_trade_count = calculate_max_profit_numba(profits_arr)

    if trade_count > 0 and max_loss_start_idx < len(kai_data_df) and max_loss_end_idx < len(kai_data_df):
        max_loss_start_time = kai_data_df.iloc[max_loss_start_idx]['timestamp']
        max_loss_end_time = kai_data_df.iloc[max_loss_end_idx]['timestamp']
        max_loss_hold_time = kai_data_df.index[max_loss_end_idx] - kai_data_df.index[max_loss_start_idx]
    else:
        max_loss_start_time = max_loss_end_time = max_loss_hold_time = None

    if trade_count > 0 and max_profit_start_idx < len(kai_data_df) and max_profit_end_idx < len(kai_data_df):
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

    monthly_groups = kai_data_df['timestamp'].dt.to_period('M')
    monthly_agg = kai_data_df.groupby(monthly_groups)['true_profit'].agg(['sum', 'mean', 'count'])
    monthly_trade_std = float(monthly_agg['count'].std())
    active_months = monthly_agg.shape[0]
    active_month_ratio = active_months / total_months if total_months else 0
    monthly_net_profit_std = float(monthly_agg['sum'].std())
    monthly_avg_profit_std = float(monthly_agg['mean'].std())
    monthly_net_profit_min = monthly_agg['sum'].min()
    monthly_net_profit_max = monthly_agg['sum'].max()
    monthly_loss_rate = (monthly_agg['sum'] < 0).sum() / active_months if active_months else 0

    hold_time_std = kai_data_df['hold_time'].std()

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
        'monthly_trade_std': round(monthly_trade_std, 4),
        'active_month_ratio': round(active_month_ratio, 4),
        'monthly_loss_rate': round(monthly_loss_rate, 4),
        'monthly_net_profit_min': round(monthly_net_profit_min, 4),
        'monthly_net_profit_max': round(monthly_net_profit_max, 4),
        'monthly_net_profit_std': round(monthly_net_profit_std, 4),
        'monthly_avg_profit_std': round(monthly_avg_profit_std, 4),
        'top_profit_ratio': round(top_profit_ratio, 4),
        'top_loss_ratio': round(top_loss_ratio, 4)
    }
    statistic_dict.update(temp_dict)
    return kai_data_df, statistic_dict


def generate_numbers(start, end, number, even=True):
    """
    生成从 start 到 end 范围内的 number 个数字。
    参数 even=True 时均匀生成，否则采用指数增长分布。
    """
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


def process_tasks(task_chunk, df, is_filter):
    """
    处理一块任务，每块任务包含多个 (kai, pin) 信号对。
    使用信号缓存避免重复计算，并统计任务耗时。
    """
    start_time = time.time()
    results = []
    signal_cache = {}  # 每个进程内部缓存
    for long_column, short_column in task_chunk:
        _, stat_long = get_detail_backtest_result_op(df, long_column, short_column, signal_cache, is_filter)
        results.append(stat_long)
    print(f"处理 {len(task_chunk)} 个任务，耗时 {time.time() - start_time:.2f} 秒。")
    return results


def gen_ma_signal_name(start_period, end_period, step):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    long_columns = [f"ma_{period}_high_long" for period in period_list]
    short_columns = [f"ma_{period}_low_short" for period in period_list]
    key_name = f'ma_{start_period}_{end_period}_{step}'
    print(f"ma一共生成 {len(long_columns)} 个信号列名。参数为：{start_period}, {end_period}, {step}")
    return long_columns, short_columns, key_name


def gen_rsi_signal_name(start_period, end_period, step):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    temp_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    long_columns = [f"rsi_{period}_{overbought}_{100 - overbought}_high_long" for period in period_list for overbought in temp_list]
    short_columns = [f"rsi_{period}_{overbought}_{100 - overbought}_low_short" for period in period_list for overbought in temp_list]
    key_name = f'rsi_{start_period}_{end_period}_{step}'
    print(f"rsi一共生成 {len(long_columns)} 个信号列名。参数为：{start_period}, {end_period}, {step}")
    return long_columns, short_columns, key_name


def gen_peak_signal_name(start_period, end_period, step):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    long_columns = [f"peak_{period}_high_long" for period in period_list]
    short_columns = [f"peak_{period}_low_short" for period in period_list]
    key_name = f'peak_{start_period}_{end_period}_{step}'
    print(f"peak一共生成 {len(long_columns)} 个信号列名。参数为：{start_period}, {end_period}, {step}")
    return long_columns, short_columns, key_name


def gen_continue_signal_name(start_period, end_period, step):
    period_list = range(start_period, end_period, step)
    long_columns = [f"continue_{period}_high_long" for period in period_list]
    short_columns = [f"continue_{period}_low_short" for period in period_list]
    key_name = f'continue_{start_period}_{end_period}_{step}'
    print(f"continue一共生成 {len(long_columns)} 个信号列名。参数为：{start_period}, {end_period}, {step}")
    return long_columns, short_columns, key_name


def gen_abs_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = range(start_period1, end_period1, step1)
    period_list1 = [x / 10 for x in period_list1]
    long_columns = [f"abs_{period}_{period1}_high_long" for period in period_list for period1 in period_list1 if period >= period1]
    short_columns = [f"abs_{period}_{period1}_low_short" for period in period_list for period1 in period_list1 if period >= period1]
    key_name = f'abs_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}'
    print(f"abs一共生成 {len(long_columns)} 个信号列名。参数为：{start_period}, {end_period}, {step}, {start_period1}, {end_period1}, {step1}")
    return long_columns, short_columns, key_name


def gen_relate_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = range(start_period1, end_period1, step1)
    long_columns = [f"relate_{period}_{period1}_high_long" for period in period_list for period1 in period_list1 if period >= period1]
    short_columns = [f"relate_{period}_{period1}_low_short" for period in period_list for period1 in period_list1 if period >= period1]
    key_name = f'relate_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}'
    print(f"relate一共生成 {len(long_columns)} 个信号列名。参数为：{start_period}, {end_period}, {step}, {start_period1}, {end_period1}, {step1}")
    return long_columns, short_columns, key_name


def gen_macross_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = generate_numbers(start_period1, end_period1, step1, even=False)
    long_columns = [f"macross_{period}_{period1}_high_long" for period in period_list for period1 in period_list1]
    short_columns = [f"macross_{period}_{period1}_low_short" for period in period_list for period1 in period_list1]
    key_name = f'macross_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}'
    print(f"macross一共生成 {len(long_columns)} 个信号列名。参数为：{start_period}, {end_period}, {step}, {start_period1}, {end_period1}, {step1}")
    return long_columns, short_columns, key_name

def worker_func(args):
    """
    用于在进程池中的包装函数，使得参数可以打包传递。
    """
    chunk, df, is_filter = args
    return process_tasks(chunk, df, is_filter)

def backtest_breakthrough_strategy(df, base_name, is_filter):
    """
    回测函数：
      根据指定信号策略组生成所有 (kai, pin) 信号对，并利用多进程并行进行回测，
      结果保存至 CSV 文件。
    """
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

    abs_long_columns, abs_short_columns, abs_key_name = gen_abs_signal_name(1, 1000, 20, 1, 25, 1)
    column_list.append((abs_long_columns, abs_short_columns, abs_key_name))
    # 将column_list按照第一个元素的长度升序排列
    column_list = sorted(column_list, key=lambda x: len(x[0]))
    all_columns = []
    key_name = ''
    for column_pair in column_list:
        long_columns, short_columns, temp_key_name = column_pair
        temp = long_columns + short_columns
        key_name += temp_key_name + '_'
        all_columns.extend(temp)

    task_list = list(product(all_columns, all_columns))

    # # 删除x[0].split('_')[0] == x[1].split('_')[0]的信号对
    # task_list = [x for x in task_list if x[0].split('_')[0] != x[1].split('_')[0]]

    # === 大块划分（每大块包含 100,000 个任务） ===
    big_chunk_size = 100_000
    big_task_chunks = [task_list[i:i + big_chunk_size] for i in range(0, len(task_list), big_chunk_size)]
    print(f'共有 {len(task_list)} 个任务，分为 {len(big_task_chunks)} 大块。')

    # 我们获取 CPU 核数来设置进程数
    pool_processes = max(1, multiprocessing.cpu_count())

    # 创建进程池一次并在后续大块的处理过程中复用
    with multiprocessing.Pool(processes=pool_processes) as pool:
        # 对于每个大块依次处理
        for i, task_chunk in enumerate(big_task_chunks):
            output_path = os.path.join('temp', f"statistic_{base_name}_{key_name}_is_filter-{is_filter}part{i}.csv")
            if os.path.exists(output_path):
                print(f'已存在 {output_path}')
                continue

            # 复制并打乱任务顺序
            task_chunk = list(task_chunk)  # 复制一份
            np.random.shuffle(task_chunk)

            # === 小块划分 ===
            # 根据当前大块任务数、CPU核数与经验因子 15 计算每个小块的容量
            chunk_size = int(np.ceil(len(task_chunk) / (pool_processes * 15)))
            chunk_size = max(50, chunk_size)  # 保证每块至少 50 个任务
            task_chunks = [task_chunk[j:j + chunk_size] for j in range(0, len(task_chunk), chunk_size)]

            print(
                f'当前处理文件: {output_path}\n'
                f'共有 {len(task_chunk)} 个任务，分为 {len(task_chunks)} 块，'
                f'单个块任务大小约为 {len(task_chunks[0])}。'
            )

            # debug
            start_time = time.time()
            statistic_dict_list = process_tasks(task_chunks[0], df, is_filter)
            result = [x for x in statistic_dict_list if x is not None]
            result_df = pd.DataFrame(result)
            print(f'单块任务耗时 {time.time() - start_time:.2f} 秒。')

            # 为进程函数准备参数列表，每个元素为 (子任务块, df, is_filter)
            tasks_args = [(chunk, df, is_filter) for chunk in task_chunks]

            # 使用 imap_unordered 动态调度任务：
            statistic_dict_list = []
            for result in pool.imap_unordered(worker_func, tasks_args, chunksize=1):
                statistic_dict_list.extend(result)

            # 如有必要，过滤掉返回结果中的 None 值
            statistic_dict_list = [x for x in statistic_dict_list if x is not None]

            # 将结果转换为 DataFrame 并保存成 CSV 文件
            statistic_df = pd.DataFrame(statistic_dict_list)
            statistic_df.to_csv(output_path, index=False)
            print(f'结果已保存到 {output_path} 当前时间 {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')


def gen_breakthrough_signal(data_path='temp/TON_1m_2000.csv'):
    """
    主函数：
      1. 加载 CSV 中原始数据（只保留 timestamp, open, high, low, close）
      2. 指定周期范围，调用回测函数
    """
    base_name = os.path.basename(data_path)
    is_filter = True
    df = pd.read_csv(data_path)
    needed_columns = ['timestamp', 'open', 'high', 'low', 'close']
    df = df[needed_columns]
    df['chg'] = df['close'].pct_change() * 100
    df['close'] = df['close'].astype('float32')
    df['high'] = df['high'].astype('float32')
    df['low'] = df['low'].astype('float32')
    df['open'] = df['open'].astype('float32')
    df['chg'] = df['chg'].astype('float32')


    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_monthly = df['timestamp'].dt.to_period('M')
    min_df_month = df_monthly.min()
    max_df_month = df_monthly.max()
    df = df[(df_monthly != min_df_month) & (df_monthly != max_df_month)]
    print(f'开始回测 {base_name} ... 长度 {df.shape[0]} 当前时间 {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    backtest_breakthrough_strategy(df, base_name, is_filter)


def example():
    start_time = time.time()
    data_path_list = [
        'kline_data/origin_data_1m_10000000_BTC-USDT-SWAP.csv',
        'kline_data/origin_data_1m_10000000_ETH-USDT-SWAP.csv',
        'kline_data/origin_data_1m_10000000_SOL-USDT-SWAP.csv',
        'kline_data/origin_data_1m_10000000_TON-USDT-SWAP.csv',
        'kline_data/origin_data_1m_10000000_DOGE-USDT-SWAP.csv',
        'kline_data/origin_data_1m_10000000_XRP-USDT-SWAP.csv',
        'kline_data/origin_data_1m_10000000_PEPE-USDT-SWAP.csv',
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