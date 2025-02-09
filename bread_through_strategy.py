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
from itertools import product

import numpy as np
import pandas as pd
from numba import njit


def compute_signal(df, col_name):
    """
    根据列名动态计算信号
    列名格式："{period}_{base}_{direction}"
      对于多头信号：例如 "5_high_long"，
         计算 price 用 high.shift(1).rolling(window=period).max()
         信号为 df['high'] > price
      对于空头信号：例如 "5_low_short"，
         计算 price 用 low.shift(1).rolling(window=period).min()
         信号为 df['low'] < price
    返回 (signal_series, price_series)
    """
    parts = col_name.split('_')
    period = int(parts[0])
    direction = parts[2]  # "long" 或 "short"
    if direction == "long":
        price_series = df['high'].shift(1).rolling(window=period).max()
        signal_series = df['high'] > price_series
    else:
        price_series = df['low'].shift(1).rolling(window=period).min()
        signal_series = df['low'] < price_series
    return signal_series, price_series


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

    for i, profit in enumerate(true_profit_series):
        if current_loss == 0:
            temp_start_index = kai_data_df.index[i]
            trade_count = 0 # 在新的潜在亏损序列开始时，重置 trade_count (也可以不重置，在current_loss > 0 时重置更清晰)

        current_loss += profit
        trade_count += 1 # 每次迭代都增加交易计数

        if current_loss < max_loss:
            max_loss = current_loss
            start_index = temp_start_index
            end_index = kai_data_df.index[i]
            max_sequence_length = trade_count  # 直接使用 trade_count 更新最大子序列长度

        if current_loss > 0:
            current_loss = 0
            trade_count = 0 # 当盈利出现时，重置亏损和交易计数

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

    for i, profit in enumerate(true_profit_series):
        if current_profit == 0:
            temp_start_index = kai_data_df.index[i]
            trade_count = 0 # 在新的潜在盈利序列开始时，重置 trade_count (也可以不重置，在current_profit < 0 时重置更清晰)

        current_profit += profit
        trade_count += 1 # 每次迭代都增加交易计数

        if current_profit > max_profit:
            max_profit = current_profit
            start_index = temp_start_index
            end_index = kai_data_df.index[i]
            max_sequence_length = trade_count  # 直接使用 trade_count 更新最大子序列长度

        if current_profit < 0:
            current_profit = 0
            trade_count = 0 # 当亏损出现时，重置盈利和交易计数

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

    # 生成统计数据字典 statistic_dict
    statistic_dict = {
        'kai_side': kai_side,
        'kai_column': kai_column,
        'pin_column': pin_column,
        'total_count': df.shape[0],
        'kai_count': kai_data_df.shape[0],
        'trade_rate': round(kai_data_df.shape[0] / df.shape[0], 4) if df.shape[0] > 0 else 0,
        'hold_time_mean': kai_data_df['hold_time'].mean() if not kai_data_df.empty else 0,
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


def process_tasks(task_chunk, df, is_filter):
    """
    处理一块任务（每块任务包含 chunk_size 个任务）。
    在处理过程中，根据任务内的信号名称动态生成信号并缓存，降低内存和重复计算。
    """
    start_time = time.time()
    results = []
    signal_cache = {}  # 同一进程内对信号结果进行缓存
    for long_column, short_column in task_chunk:
        # 对应一次做「开仓」回测
        _, stat_long = get_detail_backtest_result(df, long_column, short_column, signal_cache, is_filter)
        results.append(stat_long)
        # 再做「开空」方向回测（此时互换信号）
        _, stat_short = get_detail_backtest_result(df, short_column, long_column, signal_cache, is_filter)
        results.append(stat_short)
    print(f"处理 {len(task_chunk)*2} 个任务，耗时 {time.time()-start_time:.2f} 秒。")
    return results

def backtest_breakthrough_strategy(df, base_name, start_period, end_period, step, is_filter):
    """
    回测函数：基于原始数据 df 和指定周期范围，
    生成所有 (kai, pin) 信号对（kai 信号命名为 "{period}_high_long"，pin 信号命名为 "{period}_low_short"），
    使用多进程并行调用 process_tasks() 完成回测，并将统计结果保存到 CSV 文件。
    """


    period_list = range(start_period, end_period, step)
    long_columns = [f"{period}_high_long" for period in period_list]
    short_columns = [f"{period}_low_short" for period in period_list]
    task_list = list(product(long_columns, short_columns))
    big_chunk_size = 200000
    big_task_chunks = [task_list[i:i + big_chunk_size] for i in range(0, len(task_list), big_chunk_size)]
    print(f'共有 {len(task_list)} 个任务，分为 {len(big_task_chunks)} 大块。')
    for i, task_chunk in enumerate(big_task_chunks):
        # 将task_list打乱顺序
        output_path = f"temp/statistic_{base_name}_start_period-{start_period}_end_period-{end_period}_step-{step}_is_filter-{is_filter}_part{i}.csv"
        if os.path.exists(output_path):
            print(f'已存在 {output_path}')
            continue
        task_chunk = task_chunk.copy()
        np.random.shuffle(task_chunk)

        # 将任务分块，每块包含一定数量的任务
        chunk_size = 200
        task_chunks = [task_chunk[i:i + chunk_size] for i in range(0, len(task_chunk), chunk_size)]
        print(f'共有 {len(task_chunk)} 个任务，分为 {len(task_chunks)} 块。')

        statistic_dict_list = []
        pool_processes = max(1, multiprocessing.cpu_count())
        with multiprocessing.Pool(processes=pool_processes) as pool:
            results = pool.starmap(process_tasks, [(chunk, df, is_filter) for chunk in task_chunks])
        for res in results:
            statistic_dict_list.extend(res)

        statistic_df = pd.DataFrame(statistic_dict_list)
        statistic_df.to_csv(output_path, index=False)
        print(f'结果已保存到 {output_path}')

def gen_breakthrough_signal(data_path='temp/TON_1m_2000.csv'):
    """
    主函数：
      1. 加载 CSV 中原始数据（只保留 timestamp, open, high, low, close 五列）
      2. 指定周期范围（start_period, end_period, step）
      3. 调用 backtest_breakthrough_strategy 进行回测
    """
    base_name = os.path.basename(data_path)

    start_period = 1
    end_period = 3000
    step = 2
    is_filter = True


    # # debug
    # df = pd.read_csv(data_path)
    # long_column = '2584_high_long'
    # short_column = '2849_low_short'
    # signal_cache = {}
    # # df = df[-50000:]
    # get_detail_backtest_result(df, long_column, short_column, signal_cache, is_filter)



    # if not os.path.exists(output_path):
    df = pd.read_csv(data_path)
    needed_columns = ['timestamp', 'open', 'high', 'low', 'close']
    df = df[needed_columns]
    print(f'开始回测 {base_name} ... 长度 {df.shape[0]} 当前时间 {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    backtest_breakthrough_strategy(df, base_name, start_period, end_period, step, is_filter)
    # else:
    #     print(f'已存在 {output_path}')


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

def choose_good_strategy():
    # df = pd.read_csv('temp/temp.csv')
    start_time = time.time()
    # 找到temp下面所有包含False的文件
    file_list = os.listdir('temp')
    file_list = [file for file in file_list if 'True' in file and 'BTC' in file and '-USDT-SWAP.csv_start_period-1_end_period-3000_step-2_is_filter-True' in file and '1m' in file and '0000' in file]
    df_list = []
    for file in file_list:
        df = pd.read_csv(f'temp/{file}')
        # if df.shape[0] < 1000000:
        #     continue

        df['filename'] = file.split('_')[5]
        # 过滤掉avg_profit_rate小于0的数据
        df['reverse_profit_rate'] = df['profit_rate'] * -1
        df['reverse_net_profit_rate'] = df['reverse_profit_rate'] - df['cost_rate']
        df['reverse_avg_profit_rate'] = round(df['reverse_net_profit_rate'] / df['kai_count'] * 100, 4)
        df['reverse_avg_profit_rate'] = 0
        # df['avg_profit_rate'] = 0
        # 筛选出reverse_avg_profit_rate大于0的数据或者avg_profit_rate大于0的数据
        df = df[(df['reverse_avg_profit_rate'] > 0) | (df['avg_profit_rate'] > 0)]
        df_list.append(df)
    signal_data_df = pd.concat(df_list)
    signal_data_df['score'] = signal_data_df['avg_profit_rate']
    signal_data_df['score1'] = signal_data_df['avg_profit_rate'] / (signal_data_df['hold_time_mean'] + 20) * 1000
    signal_data_df['score2'] = signal_data_df['avg_profit_rate'] / (signal_data_df['hold_time_mean'] + 20) * 1000 * (signal_data_df['trade_rate'] + 0.001)
    signal_data_df['score3'] = signal_data_df['avg_profit_rate'] * (signal_data_df['trade_rate'] + 0.0001)
    print(f'耗时 {time.time() - start_time:.2f} 秒。 长度 {signal_data_df.shape[0]}')
    # 对 DataFrame 每一行计算 optimal_leverage
    signal_data_df[['optimal_L', 'max_balance']] = signal_data_df.apply(
        lambda row: optimal_leverage_opt(
            row['max_consecutive_loss'], row['max_loss_trade_count'], row['max_consecutive_profit'], row['max_profit_trade_count'],
            row['min_profit'], row['max_profit'], row['net_profit_rate'], row['kai_count']
        ), axis=1, result_type='expand'
    )

    print(f'耗时 {time.time() - start_time:.2f} 秒。')
    signal_data_df.to_csv('temp/temp.csv', index=False)

    # temp = pd.merge(df_list[0], df_list[1], on=['kai_side', 'kai_column', 'pin_column'], how='inner')
    # temp['avg_profit_rate_min'] = temp[['avg_profit_rate_x', 'avg_profit_rate_y']].min(axis=1)
    # temp['avg_profit_rate_mean'] = temp[['avg_profit_rate_x', 'avg_profit_rate_y']].mean(axis=1)
    # temp['avg_profit_rate_plus'] = temp['avg_profit_rate_x'] + temp['avg_profit_rate_y']
    # temp['avg_profit_rate_mult'] = np.where(
    #     (temp['avg_profit_rate_x'] < 0) & (temp['avg_profit_rate_y'] < 0),
    #     # 条件：avg_profit_rate_x 和 avg_profit_rate_y 都小于 0
    #     0,  # 如果条件为真，则赋值为 0
    #     temp['avg_profit_rate_x'] * temp['avg_profit_rate_y']  # 如果条件为假，则进行正常的乘法运算
    # )
    #
    # temp['reverse_avg_profit_rate_min'] = temp[['reverse_avg_profit_rate_x', 'reverse_avg_profit_rate_y']].min(axis=1)
    # temp['reverse_avg_profit_rate_mean'] = temp[['reverse_avg_profit_rate_x', 'reverse_avg_profit_rate_y']].mean(axis=1)
    # temp['reverse_avg_profit_rate_plus'] = temp['reverse_avg_profit_rate_x'] + temp['reverse_avg_profit_rate_y']
    # temp['reverse_avg_profit_rate_mult'] = np.where(
    #     (temp['reverse_avg_profit_rate_x'] < 0) & (temp['reverse_avg_profit_rate_y'] < 0),
    #     # 条件：reverse_avg_profit_rate_x 和 reverse_avg_profit_rate_y 都小于 0
    #     0,  # 如果条件为真，则赋值为 0
    #     temp['reverse_avg_profit_rate_x'] * temp['reverse_avg_profit_rate_y']  # 如果条件为假，则进行正常的乘法运算
    # )
    # temp['net_profit_rate_min'] = temp[['net_profit_rate_x', 'net_profit_rate_y']].min(axis=1)
    # temp['net_profit_rate_mean'] = temp[['net_profit_rate_x', 'net_profit_rate_y']].mean(axis=1)
    # temp['net_profit_rate_plus'] = temp['net_profit_rate_x'] + temp['net_profit_rate_y']
    # temp['net_profit_rate_mult'] = np.where(
    #     (temp['net_profit_rate_x'] < 0) & (temp['net_profit_rate_y'] < 0),
    #     # 条件：net_profit_rate_x 和 net_profit_rate_y 都小于 0
    #     0,  # 如果条件为真，则赋值为 0
    #     temp['net_profit_rate_x'] * temp['net_profit_rate_y']  # 如果条件为假，则进行正常的乘法运算
    # )
    # temp['reverse_net_profit_rate_min'] = temp[['reverse_net_profit_rate_x', 'reverse_net_profit_rate_y']].min(axis=1)
    # temp['reverse_net_profit_rate_mean'] = temp[['reverse_net_profit_rate_x', 'reverse_net_profit_rate_y']].mean(axis=1)
    # temp['reverse_net_profit_rate_plus'] = temp['reverse_net_profit_rate_x'] + temp['reverse_net_profit_rate_y']
    # temp['reverse_net_profit_rate_mult'] = np.where(
    #     (temp['reverse_net_profit_rate_x'] < 0) & (temp['reverse_net_profit_rate_y'] < 0),
    #     # 条件：reverse_net_profit_rate_x 和 reverse_net_profit_rate_y 都小于 0
    #     0,  # 如果条件为真，则赋值为 0
    #     temp['reverse_net_profit_rate_x'] * temp['reverse_net_profit_rate_y']  # 如果条件为假，则进行正常的乘法运算
    # )
    #
    # temp = temp[(temp['avg_profit_rate_min'] > 0) | (temp['reverse_avg_profit_rate_min'] > 0)]
    # return temp


def example():
    # choose_good_strategy()
    start_time = time.time()

    data_path_list = [
        # 'kline_data/origin_data_1m_10000000_BTC-USDT-SWAP.csv',
        # 'kline_data/origin_data_1m_86000_BTC-USDT-SWAP.csv',

        # 'kline_data/origin_data_1m_10000000_ETH-USDT-SWAP.csv',
        'kline_data/origin_data_1m_86000_ETH-USDT-SWAP.csv',

        'kline_data/origin_data_1m_10000000_SOL-USDT-SWAP.csv',
        'kline_data/origin_data_1m_86000_SOL-USDT-SWAP.csv',

        'kline_data/origin_data_1m_10000000_TON-USDT-SWAP.csv',
        'kline_data/origin_data_1m_86000_TON-USDT-SWAP.csv',

        'kline_data/origin_data_1m_10000000_DOGE-USDT-SWAP.csv',
        'kline_data/origin_data_1m_86000_DOGE-USDT-SWAP.csv',

        'kline_data/origin_data_1m_10000000_XRP-USDT-SWAP.csv',
        'kline_data/origin_data_1m_86000_XRP-USDT-SWAP.csv',

        'kline_data/origin_data_1m_10000000_PEPE-USDT-SWAP.csv',
        'kline_data/origin_data_1m_86000_PEPE-USDT-SWAP.csv'
    ]
    for data_path in data_path_list:
        try:
            gen_breakthrough_signal(data_path)
            print(f'{data_path} 总耗时 {time.time() - start_time:.2f} 秒。')
        except Exception as e:
            print(f'处理 {data_path} 时出错：{e}')


if __name__ == '__main__':
    example()