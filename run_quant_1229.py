import json
import math
import os
import multiprocessing as mp
import time
from datetime import datetime
import pyarrow.parquet as pq

import numpy as np
from numba import njit, prange
from tqdm import tqdm  # 用于显示进度条

import pandas as pd

from get_feature_op import generate_price_extremes_signals, generate_price_unextremes_signals, \
    generate_price_extremes_reverse_signals


def gen_buy_sell_signal(data_df, profit=1 / 100, period=10):
    """
    为data生成相应的买卖信号，并生成相应的buy_price, sell_price
    :param data_df:
    :param profit:
    :param period:
    :return:
    """
    # start_time = datetime.now()
    signal_df = generate_price_extremes_reverse_signals(data_df, periods=[period])
    # 找到包含Buy的列和包含Sell的列名
    buy_col = [col for col in signal_df.columns if 'Buy' in col]
    sell_col = [col for col in signal_df.columns if 'Sell' in col]
    # 将buy_col[0]重命名为Buy
    signal_df.rename(columns={buy_col[0]: 'Buy'}, inplace=True)
    signal_df.rename(columns={sell_col[0]: 'Sell'}, inplace=True)
    # 初始化 buy_price 和 sell_price 列，可以设置为 NaN 或者其他默认值
    signal_df['buy_price'] = None
    signal_df['sell_price'] = None

    # 找到 Buy 为 1 的行，设置 buy_price 和 sell_price
    buy_rows = signal_df['Buy'] == 1
    signal_df.loc[buy_rows, 'buy_price'] = signal_df.loc[buy_rows, 'close']
    signal_df.loc[buy_rows, 'sell_price'] = signal_df.loc[buy_rows, 'close'] * (1 + profit)

    # 找到 Sell 为 1 的行，设置 sell_price 和 buy_price
    sell_rows = signal_df['Sell'] == 1
    signal_df.loc[sell_rows, 'buy_price'] = signal_df.loc[sell_rows, 'close']
    signal_df.loc[sell_rows, 'sell_price'] = signal_df.loc[sell_rows, 'close'] * (1 - profit)
    # 初始化 count 列
    signal_df['count'] = 0.01
    # signal_df['Sell'] = 0
    signal_df['Buy'] = 0

    return signal_df


def analysis_position(pending_order_list, row, total_money, leverage=100):
    """
    分析持仓情况，得到当前的持仓数量，持仓均价，可使用资金
    :param pending_order_list: 持仓订单列表
    :param row: 包含当前市场价格的字典，包含high、low、close
    :param total_money: 总资金
    :param leverage: 杠杆倍数，默认100倍
    :return: 持仓数量、持仓均价、不同价格下的可用资金
    """
    long_sz = 0
    short_sz = 0
    long_cost = 0
    short_cost = 0
    long_avg_price = 0
    short_avg_price = 0

    # 提取市场价格
    high = row.high
    low = row.low
    close = row.close

    # 计算多空仓位的总大小和成本
    for order in pending_order_list:
        if order['side'] == 'ping':
            if order['type'] == 'long':
                long_sz += order['count']
                long_cost += order['count'] * order['buy_price']
            elif order['type'] == 'short':
                short_sz += order['count']
                short_cost += order['count'] * order['buy_price']

    # 计算多空仓位的平均价格
    if long_sz > 0:
        long_avg_price = long_cost / long_sz
    if short_sz > 0:
        short_avg_price = short_cost / short_sz

    # 计算浮动盈亏
    def calculate_floating_profit(price):
        long_profit = long_sz * (price - long_avg_price) if long_sz > 0 else 0
        short_profit = short_sz * (short_avg_price - price) if short_sz > 0 else 0
        return long_profit + short_profit

    close_profit = calculate_floating_profit(close)
    high_profit = calculate_floating_profit(high)
    low_profit = calculate_floating_profit(low)

    # 计算保证金占用
    def calculate_margin():
        long_margin = (long_sz * long_avg_price) / leverage if long_sz > 0 else 0
        short_margin = (short_sz * short_avg_price) / leverage if short_sz > 0 else 0
        net_margin = long_margin + short_margin
        return net_margin

    margin = calculate_margin()

    # 计算可用资金
    close_available_funds = total_money + close_profit - margin
    high_available_funds = total_money + high_profit - margin
    low_available_funds = total_money + low_profit - margin
    final_total_money_if_close = total_money + close_profit

    # 判断是否有小于0的可用资金
    if close_available_funds < 0 or high_available_funds < 0 or low_available_funds < 0:
        print("可用资金不足，无法进行交易！")

    return {
        'timestamp': row.timestamp,
        'long_sz': long_sz,
        'short_sz': short_sz,
        'long_avg_price': long_avg_price,
        'short_avg_price': short_avg_price,
        'close_available_funds': close_available_funds,
        'high_available_funds': high_available_funds,
        'low_available_funds': low_available_funds,
        'final_total_money_if_close': final_total_money_if_close

    }


def calculate_time_diff_minutes(time1, time2):
    """
    计算两个 datetime 对象之间相差的分钟数。
    """
    time_diff = time1 - time2
    return time_diff.total_seconds() / 60


def deal_pending_order(pending_order_list, row, position_info, lever, total_money, max_time_diff=2 * 1):
    """
    处理委托单
    """
    max_sell_time_diff = 1000000  # 最大卖出时间差
    high = row.high
    low = row.low
    close = row.close
    close_available_funds = position_info['close_available_funds']
    timestamp = row.timestamp
    history_order_list = []
    fee = 0.0007  # 手续费

    for order in pending_order_list:
        if order['side'] == 'kai':  # 开仓
            # 计算时间差
            time_diff = calculate_time_diff_minutes(timestamp, order['timestamp'])
            if time_diff < max_time_diff:
                if order['type'] == 'long':  # 开多仓
                    if order['buy_price'] > low:  # 买入价格高于最低价
                        # order['count'] += long_sz
                        # 判断可用资金是否足够开仓
                        required_margin = order['count'] * order['buy_price'] / lever
                        if close_available_funds >= required_margin:
                            order['side'] = 'ping'
                            order['kai_time'] = timestamp
                            close_available_funds -= required_margin  # 更新可用资金
                        else:
                            order['side'] = 'done'
                            order['message'] = 'insufficient funds'
                if order['type'] == 'short':  # 开空仓
                    if order['buy_price'] < high:
                        # order['count'] += short_sz
                        # 判断可用资金是否足够开仓
                        required_margin = order['count'] * order['buy_price'] / lever
                        if close_available_funds >= required_margin:
                            order['side'] = 'ping'
                            order['kai_time'] = timestamp
                            close_available_funds -= required_margin  # 更新可用资金
                        else:
                            order['side'] = 'done'
                            order['message'] = 'insufficient funds'
            else:
                order['side'] = 'done'
                order['message'] = 'time out'

        elif order['side'] == 'ping':  # 平仓
            pin_time_diff = calculate_time_diff_minutes(timestamp, order['kai_time'])
            if order['type'] == 'long':  # 平多仓
                if order['sell_price'] < high:
                    order['side'] = 'done'
                    order['ping_time'] = timestamp
                    # 计算收益并更新总资金
                    profit = order['count'] * (order['sell_price'] - order['buy_price'] - fee * order['sell_price'])
                    order['profit'] = profit
                    order['time_cost'] = pin_time_diff
                    total_money += profit
                else:
                    # 对超时的调整售出价格
                    if pin_time_diff > max_sell_time_diff:
                        order['sell_price'] = close - order['buy_price'] + order['sell_price']
                        order['kai_time'] = timestamp
                        order['message'] = 'sell time out'
            if order['type'] == 'short':  # 平空仓
                if order['sell_price'] > low:
                    order['side'] = 'done'
                    order['ping_time'] = timestamp
                    # 计算收益并更新总资金
                    profit = order['count'] * (order['buy_price'] - order['sell_price'] - fee * order['sell_price'])
                    order['profit'] = profit
                    order['time_cost'] = pin_time_diff
                    total_money += profit
                else:
                    # 对超时的调整售出价格
                    if pin_time_diff > max_sell_time_diff:
                        order['sell_price'] = close - order['buy_price'] + order['sell_price']
                        order['ping_time'] = timestamp
                        order['message'] = 'sell time out'

    # 删除已经完成的订单，移动到history_order_list
    history_order_list.extend([order for order in pending_order_list if order['side'] == 'done'])
    pending_order_list = [order for order in pending_order_list if order['side'] != 'done']
    return pending_order_list, history_order_list, total_money


def create_order(order_type, row, lever):
    """创建订单信息"""
    return {
        'buy_price': row.buy_price,
        'count': row.count,
        'timestamp': row.timestamp,
        'sell_price': row.sell_price,
        'type': order_type,
        'lever': lever,
        'side': 'kai'
    }


def process_signals(signal_df, lever, total_money, init_money):
    """处理信号生成的订单并计算收益"""
    pending_order_list = []
    all_history_order_list = []
    position_info_list = []
    # start_time = time.time()

    # 确保 timestamp 为 datetime 对象
    signal_df['timestamp'] = pd.to_datetime(signal_df['timestamp'])

    for row in signal_df.itertuples():
        # 分析持仓信息
        position_info = analysis_position(pending_order_list, row, total_money, lever)
        position_info_list.append(position_info)

        # 处理委托单
        pending_order_list, history_order_list, total_money = deal_pending_order(
            pending_order_list, row, position_info, lever, total_money
        )
        all_history_order_list.extend(history_order_list)

        # 根据信号生成新订单
        if row.Buy == 1:
            pending_order_list.append(create_order('long', row, lever))
        elif row.Sell == 1:
            pending_order_list.append(create_order('short', row, lever))
    # print(f"process_signals cost time: {time.time() - start_time}")
    # 计算最终结果
    position_info_df = pd.DataFrame(position_info_list)
    all_history_order_df = pd.DataFrame(all_history_order_list)
    final_total_money_if_close = position_info_df['final_total_money_if_close'].iloc[-1]
    min_available_funds = min(
        position_info_df['close_available_funds'].min(),
        position_info_df['high_available_funds'].min(),
        position_info_df['low_available_funds'].min()
    )
    max_cost_money = init_money - min_available_funds
    final_profit = final_total_money_if_close - init_money
    profit_ratio = final_profit / max_cost_money if max_cost_money > 0 else 0

    # 统计信号数量和占比
    total_signals = len(signal_df)
    buy_signals = signal_df['Buy'].sum()
    sell_signals = signal_df['Sell'].sum()
    buy_ratio = buy_signals / total_signals if total_signals > 0 else 0
    sell_ratio = sell_signals / total_signals if total_signals > 0 else 0

    # 统计 'time out' 订单数量
    if not all_history_order_df.empty:
        if 'message' not in all_history_order_df.columns:
            all_history_order_df['message'] = None
        timeout_orders = all_history_order_df[all_history_order_df['message'] == 'time out']
        timeout_long = len(timeout_orders[timeout_orders['type'] == 'long'])
        timeout_short = len(timeout_orders[timeout_orders['type'] == 'short'])
    else:
        timeout_long = 0
        timeout_short = 0

    if 'time_cost' not in all_history_order_df.columns:
        all_history_order_df['time_cost'] = None
    # 找到time_cost不为nan的行
    all_history_order_df = all_history_order_df[~all_history_order_df['time_cost'].isna()]
    # 计算time_cost的平均值
    time_cost = all_history_order_df['time_cost'].mean()

    last_data = position_info_df.iloc[-1].copy()
    last_data = last_data.to_dict()
    last_data.update({
        'final_total_money_if_close': final_total_money_if_close,
        'final_profit': final_profit,
        'max_cost_money': max_cost_money,
        'profit_ratio': profit_ratio,
        'total_signals': total_signals,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'buy_ratio': buy_ratio,
        'sell_ratio': sell_ratio,
        'timeout_long': timeout_long,
        'timeout_short': timeout_short,
        'hold_time': time_cost
    })
    # print(f'cost time: {time.time() - start_time}')

    return last_data


def calculate_combination(args):
    """多进程计算单个组合的回测结果"""
    profit, period, data_df, lever, init_money = args
    signal_df = gen_buy_sell_signal(data_df, profit=profit, period=period)
    last_data = process_signals(signal_df, lever, init_money, init_money)
    last_data.update({'profit': profit, 'period': period})
    return last_data

def generate_list(start, end, count, decimals):
  """
  生成一个从起始值到最终值的数字列表，包含指定数量的元素，并保留指定位数的小数。

  Args:
    start: 起始值。
    end: 最终值。
    count: 列表元素的数量。
    decimals: 保留的小数位数。

  Returns:
    一个包含指定数量元素的数字列表，从起始值线性递增到最终值，并保留指定位数的小数。
  """

  if count <= 0:
    return []
  elif count == 1:
    return [round(start, decimals)]

  step = (end - start) / (count - 1)
  result = []
  for i in range(count):
    value = start + i * step
    result.append(round(value, decimals))
  return result


def merge_dataframes(df_list):
    """
    将一个包含多个 DataFrame 的列表按照 'profit' 和 'period' 字段进行合并，并添加源 DataFrame 标识。

    Args:
      df_list: 一个列表，每个元素都是一个 pandas DataFrame。

    Returns:
      一个合并后的 pandas DataFrame，如果列表为空，则返回一个空的 DataFrame。
    """

    if not df_list:
        return pd.DataFrame()

    # 为每个 DataFrame 添加一个唯一的标识符列
    for i, df in enumerate(df_list):
        df['hold_time_score'] = 10000 * df['profit_ratio'] / df['hold_time']

        df_list[i] = df.copy()  # Create a copy to avoid modifying the original DataFrame
        df_list[i]['source_df'] = f'df_{i+1}'

    merged_df = df_list[0]
    for i in range(1, len(df_list)):
        merged_df = pd.merge(merged_df, df_list[i], on=['profit', 'period'], how='outer', suffixes=('', f'_{i+1}'))

    # 重命名和排序
    def categorize_and_sort_cols(df):
        # 识别不同类别的列
        source_df_cols = [col for col in df.columns if 'source_df' in col]
        profit_ratio_cols = [col for col in df.columns if 'profit_ratio' in col]
        other_cols = [col for col in df.columns if col not in source_df_cols and col not in profit_ratio_cols and col != 'score' and col != 'score_plus' and col != 'score_mul']

        # 对每种类别的列进行排序
        source_df_cols.sort()
        profit_ratio_cols.sort()
        other_cols.sort()

        # 重组列的顺序
        new_cols_order = other_cols + source_df_cols + profit_ratio_cols
        return new_cols_order

    new_cols_order = categorize_and_sort_cols(merged_df)
    merged_df = merged_df.reindex(columns=new_cols_order)

    #计算分数
    profit_ratio_cols = [col for col in merged_df.columns if 'profit_ratio' in col and 'source_df' not in col]
    profit_ratio_cols.sort()
    if len(profit_ratio_cols) >=3:
      merged_df['score'] = 10000 * merged_df[profit_ratio_cols[0]] * merged_df[profit_ratio_cols[1]] * merged_df[profit_ratio_cols[2]]
      merged_df['score_plus'] = merged_df[profit_ratio_cols[0]] + merged_df[profit_ratio_cols[1]] + merged_df[profit_ratio_cols[2]]
      merged_df['score_mul'] = merged_df['score_plus'] * merged_df['score']
      merged_df['hold_time_score_plus'] = merged_df['hold_time_score'] + merged_df['hold_time_score_2'] + merged_df['hold_time_score_3']
    elif len(profit_ratio_cols) >=2:
      merged_df['score'] = 10000 * merged_df[profit_ratio_cols[0]] * merged_df[profit_ratio_cols[1]]
      merged_df['score_plus'] = merged_df[profit_ratio_cols[0]] + merged_df[profit_ratio_cols[1]]
      merged_df['score_mul'] = merged_df['score_plus'] * merged_df['score']
      merged_df['hold_time_score_plus'] = merged_df['hold_time_score'] + merged_df['hold_time_score_2']
    return merged_df


def read_json(file_path):
    """
    读取 JSON 文件并返回 Python 对象。

    Args:
      file_path: JSON 文件的路径。

    Returns:
      一个 Python 对象，表示 JSON 文件的内容。
    """

    with open(file_path, 'r') as file:
        return json.load(file)


@njit(parallel=True)
def _calculate_time_to_targets_numba_parallel(
    timestamps,
    closes,
    highs,
    lows,
    increase_targets,
    decrease_targets,
    max_search=1000000
):
    n = len(timestamps)
    total_targets = len(increase_targets) + len(decrease_targets)
    result = np.full((n, total_targets), -1, dtype=np.int64)

    for i in prange(n - 1):  # 使用 prange 并行外层循环
        current_close = closes[i]

        # 先处理涨幅目标
        for inc_idx, inc_target in enumerate(increase_targets):
            target_price = current_close * (1.0 + inc_target)
            upper_bound = min(n, i + 1 + max_search)
            for j in range(i + 1, upper_bound):
                if highs[j] >= target_price:
                    result[i, inc_idx] = timestamps[j]
                    break

        # 再处理跌幅目标
        for dec_idx, dec_target in enumerate(decrease_targets):
            target_price = current_close * (1.0 - dec_target)
            out_col = len(increase_targets) + dec_idx
            upper_bound = min(n, i + 1 + max_search)
            for j in range(i + 1, upper_bound):
                if lows[j] <= target_price:
                    result[i, out_col] = timestamps[j]
                    break

    return result


def calculate_time_to_targets(file_path):
    """
    Calculates the time it takes for the 'close' price to increase/decrease
    by specified percentages, and returns the corresponding future timestamp.

    如果找不到满足条件的行，则返回 NaT。
    """
    result_file_path = file_path.replace('.csv', '_time_to_targets.csv')
    start_time = datetime.now()
    df = pd.read_csv(file_path)[-1000000:]
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 目标可根据需要调整
    increase_targets = [x / 10000 for x in range(1, 1000)]
    decrease_targets = [x / 10000 for x in range(1, 1000)]

    # 提取 Numpy 数组
    timestamps = df['timestamp'].astype('int64').values  # datetime64[ns] -> int64
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    # 运行 Numba 函数
    result_array = _calculate_time_to_targets_numba_parallel(
        timestamps, closes, highs, lows,
        np.array(increase_targets, dtype=np.float64),
        np.array(decrease_targets, dtype=np.float64),
    )

    all_targets = increase_targets + decrease_targets
    for col_idx, target in enumerate(all_targets):
        if col_idx < len(increase_targets):
            col_name = f'time_to_high_target_{target}'
        else:
            col_name = f'time_to_low_target_{target}'

        # 将 int64 的时间戳转换回 datetime64[ns]；为 -1 的维持 NaT
        col_data = result_array[:, col_idx]
        ts_series = pd.to_datetime(col_data, unit='ns', errors='coerce')
        df[col_name] = ts_series.where(col_data != -1, pd.NaT)

    print(f"calculate_time_to_targets cost time: {datetime.now() - start_time}")
    df.to_csv(result_file_path, index=False)
    return df


def load_and_optimize_time_to_targets(
        input_csv_path: str,
        output_parquet_path: str = None,
        downcast_prices: bool = True,
        floor_timestamp_to_minute: bool = True
):
    """
    加载已生成的 CSV 文件，只保留 'timestamp', 'close', 'high', 'low'
    以及所有包含 'time_to' 的列；并针对具体需求进行优化。

    参数：
    ----------
    input_csv_path : str
        已经生成的大 CSV 文件路径。
    output_parquet_path : str, optional
        优化后输出的 Parquet 文件路径。若不提供，则自动以 '_optimized.parquet' 结尾。
    downcast_prices : bool, default True
        是否将 close, high, low 列转为 float32 以减小内存。适合 0~200000 范围。
    floor_timestamp_to_minute : bool, default True
        是否将所有时间列 (包括 'timestamp' 和包含 'time_to' 的列) 保留到分钟精度。

    返回：
    ----------
    优化后的 DataFrame
    """

    # 0) 如果没有指定输出文件，则默认拼一个 '_optimized.parquet'
    if output_parquet_path is None:
        output_parquet_path = input_csv_path.rsplit('.', 1)[0] + '_optimized.parquet'

    # 1) 先读入一行，获取所有列名，以便筛选需要的列
    print("Reading CSV header to identify columns...")
    sample_df = pd.read_csv(input_csv_path, nrows=1)
    all_cols = sample_df.columns.tolist()

    # 2) 构造列筛选：需要 'timestamp', 'close', 'high', 'low' 以及包含 "time_to" 的列
    needed_cols = {'timestamp', 'close', 'high', 'low'}
    time_to_cols = {col for col in all_cols if 'time_to' in col}
    usecols = list(needed_cols.union(time_to_cols))

    print(f"Columns to be read: {usecols}")

    # 3) 再次读取 CSV 时，仅加载需要的列
    print(f"Loading CSV data with only required columns...")
    df = pd.read_csv(input_csv_path, usecols=usecols)

    # 4) 处理时间列：将 'timestamp' 和所有 'time_to' 列统一解析为 datetime
    #    若 floor_timestamp_to_minute=True，则只保留到分钟精度
    time_cols = ['timestamp'] + list(time_to_cols)
    for tcol in time_cols:
        if tcol in df.columns:
            df[tcol] = pd.to_datetime(df[tcol], errors='coerce')  # 先转为 datetime
            if floor_timestamp_to_minute:
                df[tcol] = df[tcol].dt.floor('T')  # 向下取整到分钟

    # 5) 将价格列 downcast 到 float32
    if downcast_prices:
        price_cols = ['close', 'high', 'low']
        for pcol in price_cols:
            if pcol in df.columns:
                df[pcol] = pd.to_numeric(df[pcol], errors='coerce', downcast='float')
                # downcast='float' 会自动转为 float32（范围足够 0~200000）

    print("Data types after conversion:")
    print(df.dtypes)

    # 6) 写入 Parquet 格式
    print(f"Saving optimized Parquet to: {output_parquet_path}")
    df.to_parquet(output_parquet_path, index=False)

    return df


def split_csv_by_columns(csv_path, num_splits):
    """
    将 CSV 文件按照列进行分割。

    Args:
        csv_path (str): CSV 文件的路径。
        num_splits (int): 分割的份数。
    """
    base_filename = os.path.splitext(os.path.basename(csv_path))[0]

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 {csv_path}")
        return
    except Exception as e:
        print(f"读取 CSV 文件时发生错误：{e}")
        return

    # 获取所有列名
    all_columns = df.columns.tolist()

    # 分离包含 "diff" 和不包含 "diff" 的列名
    diff_columns = [col for col in all_columns if 'diff' in col]
    non_diff_columns = [col for col in all_columns if 'diff' not in col]

    # 如果没有包含 "diff" 的列，则直接将不包含 "diff" 的列保存为一个文件并返回
    if not diff_columns:
        output_filename = os.path.splitext(os.path.basename(csv_path))[0] + "_non_diff.csv"
        df[non_diff_columns].to_csv(output_filename, index=False)
        print(f"已保存文件：{output_filename} (只包含非 diff 列)")
        return

    # 计算每个分割应该包含的 "diff" 列的数量
    n = len(diff_columns)
    split_size = math.ceil(n / num_splits)

    # 分割 "diff" 列
    diff_column_chunks = [diff_columns[i:i + split_size] for i in range(0, n, split_size)]

    # 组合列并保存文件
    for i, diff_chunk in enumerate(diff_column_chunks):
        # 组合列名
        combined_columns = non_diff_columns + diff_chunk

        # 提取相应的列
        split_df = df[combined_columns]

        # 生成输出文件名

        output_filename = csv_path.replace('.csv', f"_{i+1}.csv")

        # 保存为新的 CSV 文件
        try:
            split_df.to_csv(output_filename, index=False)
            print(f"已保存文件：{output_filename}")
        except Exception as e:
            print(f"保存文件 {output_filename} 时发生错误：{e}")


def compute_diff_statistics_signals_time_ranges(
        signal_df: pd.DataFrame,
        time_range_list: list,
        optimized_csv_path: str
) -> pd.DataFrame:
    """
    针对一个带有多个信号列的 signal_df（列名包含 'buy'/'sell' 或其它你自定义的关键字），
    以不同时间范围内（time_range_list）信号值为 1 的行，分别对每个 diff 列去除 NaN 并计算
    mean、median，以及 NaN 数量与占比。

    参数
    ----------
    signal_df : pd.DataFrame
        包含 'timestamp' 以及多个信号列（列名包含 'buy'/'sell' 等）。
        其中 signal_df[col] == 1 表示该时间戳下，该信号生效。
    time_range_list : list
        每个元素是 (start_time, end_time)，表示一个时间区间。例如:
        [
          ("2023-01-01", "2023-01-05"),
          ("2023-01-05", "2023-01-10"),
          ...
        ]
        start_time, end_time 可以是字符串或可被 pd.to_datetime 解析的格式。
    optimized_csv_path : str
        优化后的 CSV 文件路径（包含 'timestamp' 和若干 '..._diff' 列）。

    返回
    ----------
    pd.DataFrame
        统计结果的 DataFrame，包含以下结构：
        [
            time_range,
            signal_name,
            每个 diff_col 的 (mean, median, nan_count, nan_ratio), ...
        ]
    """
    start_time_all = time.time()
    print(">>> 开始执行 compute_diff_statistics_signals_time_ranges 函数")

    # ----------------------------------------------------------------------------------------
    print(">>> [Step 1] 读取并加载CSV文件")
    step_start_time = time.time()
    df_optimized = pd.read_csv(optimized_csv_path) # 根据需要自行调整
    df_optimized['timestamp'] = pd.to_datetime(df_optimized['timestamp'], errors='coerce')
    print(f"    完成, 耗时: {time.time() - step_start_time:.4f}秒")

    # ----------------------------------------------------------------------------------------
    print(">>> [Step 2] 转换 signal_df['timestamp'] 为 datetime")
    step_start_time = time.time()
    signal_df['timestamp'] = pd.to_datetime(signal_df['timestamp'], errors='coerce')
    print(f"    完成, 耗时: {time.time() - step_start_time:.4f}秒")

    # ----------------------------------------------------------------------------------------
    print(">>> [Step 3] 识别所有信号列（包含 'buy' 或 'sell'）")
    step_start_time = time.time()
    all_cols = signal_df.columns.tolist()
    signal_cols = [col for col in all_cols if ('buy' in col.lower() or 'sell' in col.lower())]
    print(f"    找到 {len(signal_cols)} 个信号列: {signal_cols}")
    print(f"    完成, 耗时: {time.time() - step_start_time:.4f}秒")

    # ----------------------------------------------------------------------------------------
    print(">>> [Step 4] 将 signal_df 中的信号列熔化(melt)")
    step_start_time = time.time()
    melted_signals = signal_df.melt(
        id_vars='timestamp',
        value_vars=signal_cols,
        var_name='signal_name',
        value_name='signal_value'
    )
    print(f"    完成, 耗时: {time.time() - step_start_time:.4f}秒")

    # ----------------------------------------------------------------------------------------
    print(">>> [Step 5] 与 df_optimized 合并(merge)")
    step_start_time = time.time()
    merged_df = pd.merge(melted_signals, df_optimized, on='timestamp', how='inner')
    print(f"    merged_df 行数: {len(merged_df)}")
    print(f"    完成, 耗时: {time.time() - step_start_time:.4f}秒")

    # ----------------------------------------------------------------------------------------
    print(">>> [Step 6] 标注 time_range_id 并过滤不在区间内的行")
    step_start_time = time.time()
    merged_df['time_range_id'] = -1
    time_ranges = [(pd.to_datetime(s), pd.to_datetime(e)) for (s, e) in time_range_list]
    for i, (start_time_val, end_time_val) in enumerate(time_ranges):
        mask = (merged_df['timestamp'] >= start_time_val) & (merged_df['timestamp'] < end_time_val)
        merged_df.loc[mask, 'time_range_id'] = i
    # 过滤不在这些区间内的行
    merged_df = merged_df[merged_df['time_range_id'] != -1].copy()
    print(f"    merged_df 行数(过滤后): {len(merged_df)}")
    print(f"    完成, 耗时: {time.time() - step_start_time:.4f}秒")

    # ----------------------------------------------------------------------------------------
    print(">>> [Step 7] 只保留 signal_value == 1 的行")
    step_start_time = time.time()
    merged_df = merged_df[merged_df['signal_value'] == 1].copy()
    print(f"    merged_df 行数(只保留 signal_value=1): {len(merged_df)}")
    print(f"    完成, 耗时: {time.time() - step_start_time:.4f}秒")

    # ----------------------------------------------------------------------------------------
    print(">>> [Step 8] 找到所有 _diff 结尾的列")
    step_start_time = time.time()
    diff_cols = [col for col in merged_df.columns if col.endswith('_diff')]
    print(f"    找到 {len(diff_cols)} 个 diff 列: {diff_cols}")
    print(f"    完成, 耗时: {time.time() - step_start_time:.4f}秒")

    # ----------------------------------------------------------------------------------------
    print(">>> [Step 9] 为后续统计准备一个基础表 base_df (含 time_range_id, signal_name)")
    step_start_time = time.time()
    # 先拿到 (time_range_id, signal_name) 的全部组合
    base_df = merged_df[['time_range_id', 'signal_name']].drop_duplicates().copy()
    # 同时计算 group_size (每个分组的总行数)
    group_size_series = merged_df.groupby(['time_range_id', 'signal_name']).size().rename('group_size')
    group_size_df = group_size_series.reset_index()
    base_df = base_df.merge(group_size_df, on=['time_range_id', 'signal_name'], how='left')
    # base_df 中现在有: [time_range_id, signal_name, group_size]
    print(f"    base_df 行数: {len(base_df)}")
    print(f"    完成, 耗时: {time.time() - step_start_time:.4f}秒")

    # ----------------------------------------------------------------------------------------
    print(">>> [Step 10] 分别对每个 diff 列单独去除 NaN 并统计 mean, median, nan_count, nan_ratio")
    step_start_time = time.time()

    # 我们将所有 diff 列的统计结果 merge 回 base_df
    result_df = base_df.copy()

    for diff_col in diff_cols:
        # print(f"    - 正在处理 {diff_col}")
        # 0) 计算该 diff_col 的 nan_count
        #    注意，这里只统计 NaN 行数，而没有过滤掉它们(先计算完再说)
        nan_count_series = (
            merged_df[merged_df[diff_col].isna()]
            .groupby(['time_range_id', 'signal_name'])
            .size()
            .rename(f"{diff_col}_nan_count")
        )
        # 合并到临时 DataFrame
        nan_count_df = nan_count_series.reset_index()

        # 1) 对该列非 NaN 的行，进行 mean 和 median 统计
        df_nonan = merged_df[~merged_df[diff_col].isna()].copy()
        grouped = df_nonan.groupby(['time_range_id', 'signal_name'])[diff_col]
        means = grouped.mean().rename(f"{diff_col}_mean").reset_index()
        medians = grouped.median().rename(f"{diff_col}_median").reset_index()

        # 2) 合并 mean, median, nan_count 三者
        tmp_stat_df = pd.merge(means, medians, on=['time_range_id', 'signal_name'], how='outer')
        tmp_stat_df = pd.merge(tmp_stat_df, nan_count_df, on=['time_range_id', 'signal_name'], how='outer')

        # 3) 计算 nan_ratio = diff_col_nan_count / group_size
        #    group_size 已经在 base_df 里了，先 merge tmp_stat_df 和 group_size
        tmp_stat_df = pd.merge(tmp_stat_df, base_df[['time_range_id', 'signal_name', 'group_size']],
                               on=['time_range_id', 'signal_name'], how='left')

        # 4) 有可能某些分组对这个 diff_col 完全没有任何数据，则 nan_count会是NaN -> 填0
        tmp_stat_df[f"{diff_col}_nan_count"] = tmp_stat_df[f"{diff_col}_nan_count"].fillna(0)
        tmp_stat_df[f"{diff_col}_nan_ratio"] = (
                tmp_stat_df[f"{diff_col}_nan_count"] / tmp_stat_df['group_size']
        )

        # 5) 将该列统计结果与 result_df 合并
        #    保留 [f"{diff_col}_mean", f"{diff_col}_median", f"{diff_col}_nan_count", f"{diff_col}_nan_ratio"]
        #    以及 time_range_id, signal_name（用于 on= ）
        keep_cols = [
            'time_range_id',
            'signal_name',
            f"{diff_col}_mean",
            f"{diff_col}_median",
            f"{diff_col}_nan_count",
            f"{diff_col}_nan_ratio"
        ]
        tmp_stat_df = tmp_stat_df[keep_cols]
        result_df = pd.merge(result_df, tmp_stat_df, on=['time_range_id', 'signal_name'], how='left')

    print(f"    完成所有 diff 列统计, 耗时: {time.time() - step_start_time:.4f}秒")

    # ----------------------------------------------------------------------------------------
    print(">>> [Step 11] 将 time_range_id 映射可读字符串, 并整理列顺序")
    step_start_time = time.time()
    time_range_mapping = {}
    for i, (start_time_val, end_time_val) in enumerate(time_ranges):
        time_range_mapping[i] = f"{start_time_val.strftime('%Y-%m-%d')} ~ {end_time_val.strftime('%Y-%m-%d')}"

    result_df['time_range'] = result_df['time_range_id'].map(time_range_mapping)

    # 将 time_range, signal_name 放到前面，其余列的顺序完全取决于上面合并进来的顺序
    front_cols = ['time_range', 'signal_name', 'group_size']
    other_cols = [c for c in result_df.columns if c not in ('time_range', 'signal_name', 'time_range_id', 'group_size')]
    final_cols = front_cols + other_cols
    result_df = result_df[final_cols]

    print(f"    完成, 耗时: {time.time() - step_start_time:.4f}秒")

    # ----------------------------------------------------------------------------------------
    total_time = time.time() - start_time_all
    print(f">>> 函数 compute_diff_statistics_signals_time_ranges 执行完毕, 总耗时: {total_time:.4f}秒")

    return result_df


def example():
    backtest_path = 'backtest_result'
    file_path_list = [ 'kline_data/origin_data_1m_10000000_ETH-USDT-SWAP.csv', 'kline_data/origin_data_1m_10000000_SOL-USDT-SWAP.csv', 'kline_data/origin_data_1m_10000000_TON-USDT-SWAP.csv']
    gen_signal_method = 'price_reverse_extremes_onlysell'
    profit_list = generate_list(0.001, 0.1, 100, 4)
    period_list = generate_list(10, 10000, 100, 0)
    # 将period_list变成int
    period_list = [int(period) for period in period_list]
    lever = 100
    init_money = 10000000
    longest_periods_info_path = 'kline_data/longest_periods_info.json'
    all_longest_periods_info = read_json(longest_periods_info_path)


    for file_path in file_path_list:
        df = pd.read_csv('kline_data/origin_data_1m_10000000_BTC-USDT-SWAP_statistic_1.csv')
        file_path_output = file_path.replace('.csv', '_time_to_targets_optimized.csv')

        split_csv_by_columns(file_path_output, 10)
        # file_path_output = file_path.replace('.csv', '_statistic.csv')
        # origin_df = pd.read_csv(file_path)
        # signal_df = generate_price_extremes_reverse_signals(origin_df, periods=[x for x in range(10, 10000, 10)])
        # result_file_path = file_path.replace('.csv', '_time_to_targets_optimized.csv')
        # result_df = compute_diff_statistics_signals_time_ranges(signal_df, [("2022-10-16", "2025-12-16")], result_file_path)
        # result_df.to_csv(file_path_output, index=False)



        # df = calculate_time_to_targets(file_path)
        # print()
        # base_name = file_path.split('/')[-1].split('.')[0]
        # origin_data_df = pd.read_csv(file_path)  # 只取最近1000条数据
        # origin_data_df['timestamp'] = pd.to_datetime(origin_data_df['timestamp'])
        #
        #
        # longest_periods_info = all_longest_periods_info[base_name]
        # for key, value in longest_periods_info.items():
        #     start_time_str, end_time_str = value.split('_')
        #     start_time = pd.to_datetime(start_time_str)
        #     end_time = pd.to_datetime(end_time_str)
        #     data_df = origin_data_df[(origin_data_df['timestamp'] >= start_time) & (origin_data_df['timestamp'] <= end_time)]
        #
        #     data_len = len(data_df)
        #
        #     # 获取data_df的初始时间与结束时间
        #     start_time = data_df.iloc[0].timestamp
        #     end_time = data_df.iloc[-1].timestamp
        #     print(f"开始时间：{start_time}，结束时间：{end_time} 长度：{data_len} key = {key}")
        #     # 生成time_key
        #     time_key_str = f"{start_time.strftime('%Y%m%d%H%M%S')}_{end_time.strftime('%Y%m%d%H%M%S')}"
        #
        #
        #     # 准备参数组合
        #     combinations = [(profit, period, data_df, lever, init_money) for profit in profit_list for period in period_list]
        #     print(f"共有 {len(combinations)} 个组合，开始计算...")
        #
        #     file_out = f'{backtest_path}/result_{data_len}_{len(combinations)}_{base_name}_{time_key_str}_{gen_signal_method}_{key}.csv'
        #     if os.path.exists(file_out):
        #         print(f"结果文件 {file_out} 已存在，跳过计算")
        #         continue
        #
        #     # 使用多进程计算
        #     with mp.Pool(processes=os.cpu_count() - 2) as pool:
        #         results = list(tqdm(pool.imap(calculate_combination, combinations), total=len(combinations)))
        #
        #     # 保存结果
        #     result_df = pd.DataFrame(results)
        #     result_df.to_csv(file_out, index=False)
        #     print(f"结果已保存到 {file_out}")


if __name__ == "__main__":
    example()
