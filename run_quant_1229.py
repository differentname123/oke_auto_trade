import json
import os
import multiprocessing as mp
import time
from datetime import datetime

import numpy as np
from numba import prange, njit
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
    signal_df = generate_price_extremes_signals(data_df, periods=[period])
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
    # signal_df['Buy'] = 0

    return signal_df


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


def compute_diff_statistics_signals_time_ranges_readable(
    signal_df: pd.DataFrame,
    time_range_list: list,
    optimized_parquet_path: str
) -> pd.DataFrame:
    """
    一个更直观（但不一定是最高效）的实现示例：
    1) 一次性读取并预处理 df_optimized (Parquet 格式)
    2) 识别所有diff列 & 所有信号列
    3) 三重循环：对每个 time_range -> 对每个 signal_col -> 对每个 diff_col 单独做统计
    """

    start_time_all = time.time()
    print(">>> 开始执行 compute_diff_statistics_signals_time_ranges_readable 函数")

    # ----------------------------------------------------------------------------------------
    # [Step 1] 读取并加载 Parquet 文件
    # ----------------------------------------------------------------------------------------
    print(">>> [Step 1] 读取并加载 Parquet 文件")
    step_start_time = time.time()
    df_optimized = pd.read_parquet(optimized_parquet_path)
    df_optimized['timestamp'] = pd.to_datetime(df_optimized['timestamp'], errors='coerce')

    # 计算 global_min、global_max，用于预过滤
    all_starts = [pd.to_datetime(s) for (s, e) in time_range_list]
    all_ends = [pd.to_datetime(e) for (s, e) in time_range_list]
    global_min = min(all_starts)
    global_max = max(all_ends)

    df_optimized = df_optimized[
        (df_optimized['timestamp'] >= global_min) &
        (df_optimized['timestamp'] < global_max)
        ].copy()

    # 识别所有 _diff 列（如有其他命名规则，请自行调整）
    diff_cols = [c for c in df_optimized.columns if c.endswith('_diff')]

    print(f"    加载后 df_optimized 行数: {len(df_optimized)}")
    print(f"    diff_cols: {diff_cols}")
    print(f"    完成, 耗时: {time.time() - step_start_time:.4f}秒")

    # ----------------------------------------------------------------------------------------
    # [Step 2] 预处理 signal_df：识别所有信号列
    # ----------------------------------------------------------------------------------------
    print(">>> [Step 2] 预处理 signal_df")
    step_start_time = time.time()
    signal_df['timestamp'] = pd.to_datetime(signal_df['timestamp'], errors='coerce')

    all_cols = signal_df.columns.tolist()
    signal_cols = [col for col in all_cols if ('buy' in col.lower() or 'sell' in col.lower())]

    print(f"    找到 {len(signal_cols)} 个信号列: {signal_cols}")
    print(f"    完成, 耗时: {time.time() - step_start_time:.4f}秒")

    # ----------------------------------------------------------------------------------------
    # [Step 3] 把 time_range_list 先转换成 (pd.Timestamp, pd.Timestamp) 类型，并做一个映射
    # ----------------------------------------------------------------------------------------
    print(">>> [Step 3] 整理 time_range_list")
    step_start_time = time.time()

    time_ranges = [(pd.to_datetime(s), pd.to_datetime(e)) for (s, e) in time_range_list]

    # 便于最终输出可读
    def format_range(s, e):
        return f"{s.strftime('%Y-%m-%d')} ~ {e.strftime('%Y-%m-%d')}"

    print(f"    完成, 耗时: {time.time() - step_start_time:.4f}秒")

    # ----------------------------------------------------------------------------------------
    # [Step 4] 多重循环：迭代 time_range -> (baseline + 每个 signal_col) -> diff_col
    # ----------------------------------------------------------------------------------------
    print(">>> [Step 4] 多重循环统计（含 baseline ）")
    step_start_time = time.time()

    results = []

    for (start_t, end_t) in time_ranges:
        time_range_str = format_range(start_t, end_t)

        # 在 df_optimized 中做时间过滤
        df_opt_in_range = df_optimized[
            (df_optimized['timestamp'] >= start_t) &
            (df_optimized['timestamp'] < end_t)
            ].copy()
        if df_opt_in_range.empty:
            continue

        # baseline 统计
        baseline_record = {
            'time_range': time_range_str,
            'signal_name': 'baseline'
        }
        baseline_group_size = len(df_opt_in_range)
        baseline_record['group_size'] = baseline_group_size

        for dc in diff_cols:
            series_dc = df_opt_in_range[dc]
            nan_count = series_dc.isna().sum()
            nonan_series = series_dc.dropna()

            mean_val = nonan_series.mean() if len(nonan_series) > 0 else None
            median_val = nonan_series.median() if len(nonan_series) > 0 else None
            std_val = nonan_series.std() if len(nonan_series) > 1 else None

            baseline_record[f"{dc}_mean"] = mean_val
            baseline_record[f"{dc}_median"] = median_val
            baseline_record[f"{dc}_std"] = std_val
            baseline_record[f"{dc}_nan_count"] = nan_count
            baseline_record[f"{dc}_nan_ratio"] = nan_count / baseline_group_size if baseline_group_size else None

        results.append(baseline_record)

        # 缓存 baseline 的均值和中位数，以便后续计算差值
        baseline_means = {dc: baseline_record[f"{dc}_mean"] for dc in diff_cols}
        baseline_medians = {dc: baseline_record[f"{dc}_median"] for dc in diff_cols}
        baseline_nan_ratios = {dc: baseline_record[f"{dc}_nan_ratio"] for dc in diff_cols}

        # signal 统计
        df_signal_in_range = signal_df[
            (signal_df['timestamp'] >= start_t) &
            (signal_df['timestamp'] < end_t)
            ].copy()
        if df_signal_in_range.empty:
            continue

        for sc in signal_cols:
            signal_start_time = time.time()

            df_signal_ones = df_signal_in_range[df_signal_in_range[sc] == 1].copy()
            if df_signal_ones.empty:
                continue

            # 拿到这些 timestamp 去 df_opt_in_range 匹配
            timestamps_needed = df_signal_ones['timestamp'].unique()
            df_opt_for_these_ts = df_opt_in_range[df_opt_in_range['timestamp'].isin(timestamps_needed)].copy()
            if df_opt_for_these_ts.empty:
                continue

            merged_df = pd.merge(
                df_signal_ones[['timestamp']],
                df_opt_for_these_ts,
                on='timestamp',
                how='inner'
            )
            group_size = len(merged_df)
            if group_size == 0:
                continue

            record = {
                'time_range': time_range_str,
                'signal_name': sc,
                'group_size': group_size
            }

            for dc in diff_cols:
                series_dc = merged_df[dc]
                nan_count = series_dc.isna().sum()
                nonan_series = series_dc.dropna()

                mean_val = nonan_series.mean() if len(nonan_series) > 0 else None
                median_val = nonan_series.median() if len(nonan_series) > 0 else None
                std_val = nonan_series.std() if len(nonan_series) > 1 else None

                record[f"{dc}_mean"] = mean_val
                record[f"{dc}_median"] = median_val
                record[f"{dc}_std"] = std_val
                record[f"{dc}_nan_count"] = nan_count
                record[f"{dc}_nan_ratio"] = nan_count / group_size if group_size else None

                # 与 baseline 的 mean 差值
                base_m = baseline_means.get(dc, None)
                if (mean_val is not None) and (base_m is not None):
                    record[f"{dc}_mean_vs_baseline"] = mean_val - base_m
                else:
                    record[f"{dc}_mean_vs_baseline"] = None

                # [新增] 与 baseline 的 median 差值
                base_med = baseline_medians.get(dc, None)
                if (median_val is not None) and (base_med is not None):
                    record[f"{dc}_median_vs_baseline"] = median_val - base_med
                else:
                    record[f"{dc}_median_vs_baseline"] = None

                base_nan_ratio = baseline_nan_ratios.get(dc, None)
                if (nan_count is not None) and (base_nan_ratio is not None):
                    record[f"{dc}_nan_ratio_vs_baseline"] = record[f"{dc}_nan_ratio"] - base_nan_ratio
                else:
                    record[f"{dc}_nan_ratio_vs_baseline"] = None

            results.append(record)

            # print(f"    {time_range_str} {sc} 统计完成, 耗时: {time.time() - signal_start_time:.4f}秒")

    print(f"    多重循环结束，合计生成 {len(results)} 条统计记录")
    print(f"    耗时: {time.time() - step_start_time:.4f}秒")

    # ----------------------------------------------------------------------------------------
    # [Step 5] 将 results 转为 DataFrame，并整理列顺序
    # ----------------------------------------------------------------------------------------
    print(">>> [Step 5] 构造最终结果 DataFrame")
    step_start_time = time.time()

    if not results:
        # 如果为空，返回一个空表
        final_cols = (
                ['time_range', 'signal_name', 'group_size'] +
                [f"{dc}_mean" for dc in diff_cols] +
                [f"{dc}_median" for dc in diff_cols] +
                [f"{dc}_std" for dc in diff_cols] +
                [f"{dc}_nan_count" for dc in diff_cols] +
                [f"{dc}_nan_ratio" for dc in diff_cols] +
                [f"{dc}_mean_vs_baseline" for dc in diff_cols] +
                [f"{dc}_median_vs_baseline" for dc in diff_cols] +
                [f"{dc}_nan_ratio_vs_baseline" for dc in diff_cols]
        )
        final_df = pd.DataFrame(columns=final_cols)
        print("    找不到任何统计数据，返回空 DataFrame")
    else:
        final_df = pd.DataFrame(results)

        # 组装列顺序
        front_cols = ['time_range', 'signal_name', 'group_size']
        diff_stat_cols = []
        for dc in diff_cols:
            diff_stat_cols.extend([
                f"{dc}_mean",
                f"{dc}_median",
                f"{dc}_std",
                f"{dc}_nan_count",
                f"{dc}_nan_ratio",
                f"{dc}_mean_vs_baseline",
                f"{dc}_median_vs_baseline",
                f"{dc}_nan_ratio_vs_baseline"
            ])

        final_cols = front_cols + diff_stat_cols

        # 补齐缺失列
        for c in final_cols:
            if c not in final_df.columns:
                final_df[c] = None

        # 排列出最终列顺序
        final_df = final_df[final_cols]

    # [新增] 最后一步：移除列名中的 "time_to" 前缀（如果有）
    # -----------------------------------------------------------------
    def remove_time_to_prefix(col_name: str) -> str:
        prefix = "time_to_"
        if col_name.startswith(prefix):
            return col_name[len(prefix):]
        else:
            return col_name

    final_df.rename(columns=lambda c: remove_time_to_prefix(c), inplace=True)

    print(f"    最终结果行数: {len(final_df)}")
    print(f"    耗时: {time.time() - step_start_time:.4f}秒")

    # ----------------------------------------------------------------------------------------
    total_time = time.time() - start_time_all
    print(f">>> 函数 compute_diff_statistics_signals_time_ranges_readable 执行完毕, 总耗时: {total_time:.4f}秒")

    return final_df


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

def generate_time_segments(origin_df, time_len=20):
    """
    Generates a list of 10 evenly spaced time segments based on the timestamp
    range found in a CSV file.

    Args:
        file_path (str): The path to the CSV file containing a 'timestamp' column.

    Returns:
        list: A list of tuples, where each tuple represents a time segment
              in the format ("YYYY-MM-DD", "YYYY-MM-DD").
    """
    start_timestamp = pd.to_datetime(origin_df.iloc[0].timestamp)
    end_timestamp = pd.to_datetime(origin_df.iloc[-1].timestamp)

    time_difference = end_timestamp - start_timestamp
    segment_duration = time_difference / time_len

    time_segments = []
    current_start = start_timestamp
    for _ in range(time_len):
        current_end = current_start + segment_duration
        time_segments.append((current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d")))
        current_start = current_end

    return time_segments


def statistic_data(file_path_list):
    """
    加速后的处理数据的代码
    :return:
    """
    for file_path in file_path_list:
        origin_df = pd.read_csv(file_path)
        time_range_list = generate_time_segments(origin_df)
        # 取time_range_list的后10个
        time_range_list = time_range_list[-10:]
        # 获取file_path的目录
        file_dir = os.path.dirname(file_path)
        base_name = file_path.split('/')[-1].split('.')[0]
        key_word_list = ["price_extremes", "price_reverse_extremes"]
        start_period = 10
        end_period = 10000
        step_period = 10
        for key_word in key_word_list:
            file_path_output_dir = file_dir + f"/{key_word}/{base_name}"
            # 如果file_path_output_dir不存在，则创建
            if not os.path.exists(file_path_output_dir):
                os.makedirs(file_path_output_dir)
            result_file_path = file_path.replace('.csv', f'_time_to_targets_optimized.csv')
            optimized_parquet_path = result_file_path.replace('.csv', f'.parquet')
            all_starts = [pd.to_datetime(s) for (s, e) in time_range_list]
            all_ends = [pd.to_datetime(e) for (s, e) in time_range_list]
            global_min = min(all_starts)
            global_max = max(all_ends)
            # time_range_list = [(global_min.strftime('%Y-%m-%d'), global_max.strftime('%Y-%m-%d'))]
            time_len = len(time_range_list)
            file_path_output = file_path_output_dir + f"/start_period_{start_period}_end_period_{end_period}_step_period_{step_period}_min_{global_min.strftime('%Y%m%d')}_max_{global_max.strftime('%Y%m%d')}_time_len_{time_len}.csv"


            if os.path.exists(file_path_output):
                print(f"结果文件 {file_path_output} 已存在，跳过计算")
                continue
            if key_word == "price_extremes":
                signal_df = generate_price_extremes_signals(origin_df, periods=[x for x in range(start_period, end_period, step_period)])
            else:
                signal_df = generate_price_extremes_reverse_signals(origin_df, periods=[x for x in range(start_period, end_period, step_period)])


            print(f"开始计算 {file_path_output}")
            result_df = compute_diff_statistics_signals_time_ranges_readable(signal_df, time_range_list,
                                                                    optimized_parquet_path)
            result_df.to_csv(file_path_output, index=False)

        # for i in range(1, 11):
        #     file_path_output = file_path.replace('.csv', f'_statistic_{i}.csv')
        #     result_file_path = file_path.replace('.csv', f'_time_to_targets_optimized_{i}.csv')
        #     optimized_parquet_path = result_file_path.replace('.csv', f'.parquet')
        #     # convert_csv_to_parquet(result_file_path, result_file_path.replace('.csv', f'.parquet'))
        #     # if os.path.exists(file_path_output):
        #     #     print(f"结果文件 {file_path_output} 已存在，跳过计算")
        #     #     continue
        #     result_df = compute_diff_statistics_signals_time_ranges_readable(signal_df, [("2022-10-16", "2025-12-16"),("2023-10-16", "2025-12-16")], optimized_parquet_path)
        #
        #
        #     result_df.to_csv(file_path_output, index=False)


def truly_backtest():
    """
    之前一一组合生成回测数据的代码
    :return:
    """
    backtest_path = 'backtest_result'
    file_path_list = ['kline_data/origin_data_1m_10000000_BTC-USDT-SWAP.csv', 'kline_data/origin_data_1m_10000000_ETH-USDT-SWAP.csv', 'kline_data/origin_data_1m_10000000_SOL-USDT-SWAP.csv', 'kline_data/origin_data_1m_10000000_TON-USDT-SWAP.csv']
    gen_signal_method = 'price_extremes'
    profit_list = generate_list(0.001, 0.1, 100, 4)
    period_list = generate_list(10, 10000, 100, 0)
    # 将period_list变成int
    period_list = [int(period) for period in period_list]
    lever = 100
    init_money = 10000000
    longest_periods_info_path = 'kline_data/longest_periods_info.json'
    all_longest_periods_info = read_json(longest_periods_info_path)


    for file_path in file_path_list:
        base_name = file_path.split('/')[-1].split('.')[0]
        origin_data_df = pd.read_csv(file_path)  # 只取最近1000条数据
        origin_data_df['timestamp'] = pd.to_datetime(origin_data_df['timestamp'])


        longest_periods_info = all_longest_periods_info[base_name]
        for key, value in longest_periods_info.items():
            start_time_str, end_time_str = value.split('_')
            start_time = pd.to_datetime(start_time_str)
            end_time = pd.to_datetime(end_time_str)
            data_df = origin_data_df[(origin_data_df['timestamp'] >= start_time) & (origin_data_df['timestamp'] <= end_time)]

            data_len = len(data_df)

            # 获取data_df的初始时间与结束时间
            start_time = data_df.iloc[0].timestamp
            end_time = data_df.iloc[-1].timestamp
            print(f"开始时间：{start_time}，结束时间：{end_time} 长度：{data_len} key = {key}")
            # 生成time_key
            time_key_str = f"{start_time.strftime('%Y%m%d%H%M%S')}_{end_time.strftime('%Y%m%d%H%M%S')}"


            # 准备参数组合
            combinations = [(profit, period, data_df, lever, init_money) for profit in profit_list for period in period_list]
            print(f"共有 {len(combinations)} 个组合，开始计算...")

            file_out = f'{backtest_path}/result_{data_len}_{len(combinations)}_{base_name}_{time_key_str}_{gen_signal_method}_{key}.csv'
            if os.path.exists(file_out):
                print(f"结果文件 {file_out} 已存在，跳过计算")
                continue

            # 使用多进程计算
            with mp.Pool(processes=os.cpu_count() - 2) as pool:
                results = list(tqdm(pool.imap(calculate_combination, combinations), total=len(combinations)))

            # 保存结果
            result_df = pd.DataFrame(results)
            result_df.to_csv(file_out, index=False)
            print(f"结果已保存到 {file_out}")

def covert_df(df):
    target = 'target'
    target_columns = [col for col in df.columns if target in col]
    non_target_columns = [col for col in df.columns if target not in col]

    if not target_columns:
        return df.copy()  # 如果没有目标列，则直接返回副本

    melted_df = pd.melt(
        df,
        id_vars=non_target_columns,  # 保留的标识列
        value_vars=target_columns,  # 需要融合的列
        var_name='target_name',  # 存储原始列名的列名
        value_name='target_value'  # 存储值的列名
    )
    return melted_df

def select_top_rows_by_group_size(grouped_df, score_columns, group_step=1000):
    """
    根据 group_size 区间选择 sc 和 sc_median 列最大值的行。

    Args:
        grouped_df: 包含数据的 Pandas DataFrame。
        score_columns: 包含需要比较的评分列名的列表。
        group_step: group_size 区间的大小。

    Returns:
        包含每个区间选择出的最大值行的 DataFrame。
    """
    all_top_rows = []
    max_group_size = grouped_df['group_size'].max()
    num_intervals = (max_group_size // group_step) + 1

    for i in range(num_intervals):
        lower_bound = i * group_step
        upper_bound = (i + 1) * group_step
        interval_df = grouped_df[(grouped_df['group_size'] >= lower_bound) & (grouped_df['group_size'] < upper_bound)]

        if not interval_df.empty:
            for sc in score_columns:
                # 检查 sc 列是否全部为负数
                if sc in interval_df.columns and not (interval_df[sc] >= 0).any():
                    continue  # 如果全部为负数，则跳过
                elif sc in interval_df.columns:
                    top_sc = interval_df.nlargest(1, sc)
                    all_top_rows.append(top_sc)

                median_col = f"{sc}_median"
                # 检查 median_col 列是否全部为负数
                if median_col in interval_df.columns and not (interval_df[median_col] >= 0).any():
                    continue  # 如果全部为负数，则跳过
                elif median_col in interval_df.columns:
                    top_median = interval_df.nlargest(1, median_col)
                    all_top_rows.append(top_median)

    if all_top_rows:
        return pd.concat(all_top_rows)
    else:
        print(f"没有找到任何行 {score_columns}")
        return pd.DataFrame()


def analyze_data(file_path, score_key='median'):
    """
    对回测数据进行分析，并将结果输出为 CSV 文件。
    """
    start_time = time.time()

    output_path = file_path.replace(".csv", f"_analyze_{score_key}.csv")
    # if os.path.exists(output_path):
    #     print(f"结果文件 {output_path} 已存在，跳过计算")
    #     return pd.read_csv(output_path)

    # 读取数据
    df = pd.read_csv(file_path)

    # 只保留不包含 'std' 或 'median' 的列
    need_columns = [col for col in df.columns if "std" not in col]
    df = df[need_columns]

    # 找到所有包含 'diff' 的列，以及所有不包含 'diff' 的列
    diff_columns = [col for col in df.columns if "diff" in col]
    no_diff_columns = [col for col in df.columns if "diff" not in col]

    # 按“以下划线分割列名的前 3 个元素”进行分组
    diff_columns_group = {}
    for col in diff_columns:
        key = "_".join(col.split("_")[:3])
        diff_columns_group.setdefault(key, []).append(col)

    result_df_list = []

    # 对每个分组进行计算及处理
    for key, columns in diff_columns_group.items():
        # 这里假设分组后会有两个列：median 和 nan_ratio
        # 第 3 项需求：改为 median
        mean_col = f"{key}_diff_{score_key}"
        nan_ratio_col = f"{key}_diff_nan_ratio"

        # 从 key 中提取 profit（示例：key = 'long_XXX_0.03'，则 profit = '0.03'）
        # 如果不需要 side，可省略解析
        profit = key.split("_")[2]

        # 选取计算所需的列（避免不必要的大范围复制）
        required_cols = no_diff_columns + [mean_col, nan_ratio_col]
        temp_df = df[required_cols].copy()

        # 定义新的列名
        profit_key = f"{key}_profit"
        score1_col = f"{key}_score1"
        score2_col = f"{key}_score2"
        score3_col = f"{key}_score3"
        score4_col = f"{key}_score4"
        score_columns = [score1_col, score2_col, score3_col, score4_col]

        # 计算 profit（示例中写死扣除了 0.0007）
        temp_df[profit_key] = float(profit) - 0.0007

        # 计算 4 种评分 (score1～4)
        # 注意要先排除 mean_col 和 nan_ratio_col 可能的 NaN 或 0，这里假设数据完整
        temp_df[score1_col] = (
            (temp_df[profit_key] - temp_df[nan_ratio_col]) / (0.11 - temp_df[profit_key])
            / temp_df[mean_col] * 10000
            / temp_df[mean_col] * 10000
        )
        temp_df[score2_col] = (
            (temp_df[profit_key] - temp_df[nan_ratio_col])
            / temp_df[mean_col] * 10000
            / temp_df[mean_col] * 10000
        )
        temp_df[score3_col] = (
            (temp_df[profit_key] - temp_df[nan_ratio_col]) / (0.11 - temp_df[profit_key])
            / temp_df[mean_col] * 10000
        )
        temp_df[score4_col] = (
            (temp_df[profit_key] - temp_df[nan_ratio_col])
            / temp_df[mean_col] * 10000
        )

        # 在原数据中必须有 time_range 这一列，否则需根据业务需求自行处理
        # 这里为确保 groupby 时能聚合到 time_range
        if "time_range" not in temp_df.columns:
            temp_df["time_range"] = "unknown"

        # -------------------------------------------------------------------
        # Step 1: groupby 求 sum，保留部分字段
        # -------------------------------------------------------------------
        agg_dict = {c: "first" for c in no_diff_columns if c not in ["time_range"]}
        # 让 time_range 聚合成列表，满足第 2 项需求
        # 如果您想要它是个 set，可以改为 x.unique().tolist() 或 set(...)
        agg_dict["time_range"] = lambda x: x.unique().tolist()

        # 保留 diff 列的原信息
        agg_dict[mean_col] = lambda x: x.tolist()
        agg_dict[nan_ratio_col] = lambda x: x.tolist()

        # score列使用 sum
        for sc in score_columns:
            agg_dict[sc] = "sum"

        agg_dict["group_size"] = "sum"

        grouped_df_sum = temp_df.groupby("signal_name", as_index=False).agg(agg_dict)

        # 新增字段：median_list、nan_ratio_list
        # 第 1 项需求：仅保留小数点后 4 位
        grouped_df_sum[f"{score_key}_list"] = grouped_df_sum[mean_col].apply(
            lambda lst: [round(v, 4) for v in lst]
        )
        grouped_df_sum["nan_ratio_list"] = grouped_df_sum[nan_ratio_col].apply(
            lambda lst: [round(v, 4) for v in lst]
        )

        # 删除不再需要的原列
        grouped_df_sum.drop(columns=[mean_col, nan_ratio_col], inplace=True)

        grouped_df_median = temp_df.groupby("signal_name", as_index=False)[score_columns].median()

        # 为避免和 sum 冲突，这里将中位数列重命名为 “scoreX_col_median”
        rename_dict = {sc: f"{sc}_median" for sc in score_columns}
        grouped_df_median.rename(columns=rename_dict, inplace=True)

        grouped_df = pd.merge(grouped_df_sum, grouped_df_median, on="signal_name", how="left")


        # -------------------------------------------------------------------
        # Step 2: 根据 score 列取 top (原逻辑: nlargest(1))
        # -------------------------------------------------------------------
        all_top_rows_grouped = select_top_rows_by_group_size(grouped_df, score_columns, group_step=2000)

        # 合并并去重，从而避免出现同一个分组的同一行在多个评分列里都重复入选
        merged_top_df = all_top_rows_grouped.drop_duplicates(subset="signal_name")

        # 最后一次性调用 convert_df()，把这个分组的 top 数据处理完
        melt_df = covert_df(merged_top_df)
        result_df_list.append(melt_df)

    # 合并所有结果并输出到 CSV
    result_df = pd.concat(result_df_list, ignore_index=True)
    result_df.to_csv(output_path, index=False)
    print(f"结果已保存到 {output_path} (耗时: {time.time() - start_time:.4f}秒)")
    return result_df


def example():
    file_path_list = ['kline_data/origin_data_1m_10000000_BTC-USDT-SWAP.csv', 'kline_data/origin_data_1m_10000000_ETH-USDT-SWAP.csv',
     'kline_data/origin_data_1m_10000000_SOL-USDT-SWAP.csv', 'kline_data/origin_data_1m_10000000_TON-USDT-SWAP.csv']

    # 示例一:传统的一一获取不同信号在三种指定时间段上面的表现结果
    # truly_backtest(file_path_list)

    # 示例二:使用预处理后的数据初略获取回测效果数据
    # statistic_data(file_path_list)


    # #示例三: 对原始数据进行预处理
    # for file_path in file_path_list:
    #     calculate_time_to_targets(file_path)

    # 示例四：分析初略的回测结果数据
    file_path_list = [
        # "temp/temp.csv",
        "kline_data/price_extremes/origin_data_1m_10000000_BTC-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_10.csv",
        "kline_data/price_extremes/origin_data_1m_10000000_ETH-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_10.csv",
        "kline_data/price_extremes/origin_data_1m_10000000_SOL-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_10.csv",
        "kline_data/price_extremes/origin_data_1m_10000000_TON-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_10.csv",
        "kline_data/price_reverse_extremes/origin_data_1m_10000000_BTC-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_10.csv",
        "kline_data/price_reverse_extremes/origin_data_1m_10000000_ETH-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_10.csv",
        "kline_data/price_reverse_extremes/origin_data_1m_10000000_SOL-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_10.csv",
        "kline_data/price_reverse_extremes/origin_data_1m_10000000_TON-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_10.csv",
        "kline_data/price_extremes/origin_data_1m_10000000_BTC-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_1.csv",
        "kline_data/price_extremes/origin_data_1m_10000000_ETH-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_1.csv",
        "kline_data/price_extremes/origin_data_1m_10000000_SOL-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_1.csv",
        "kline_data/price_extremes/origin_data_1m_10000000_TON-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_1.csv",
        "kline_data/price_reverse_extremes/origin_data_1m_10000000_BTC-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_1.csv",
        "kline_data/price_reverse_extremes/origin_data_1m_10000000_ETH-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_1.csv",
        "kline_data/price_reverse_extremes/origin_data_1m_10000000_SOL-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_1.csv",
        "kline_data/price_reverse_extremes/origin_data_1m_10000000_TON-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_1.csv"
    ]
    # result_df1 = pd.read_csv('backtest_result/analyze_data_mean.csv')
    # result_df2 = pd.read_csv('backtest_result/analyze_data_median.csv')
    analyze_df_list = []
    score_key = 'median'
    for file_path in file_path_list:
        analyze_df = analyze_data(file_path, score_key)
        file_path_split = file_path.split('_')
        key_name = f'{file_path_split[2]}_{file_path_split[6]}_{file_path_split[7]}'
        analyze_df['key_name'] = key_name
        analyze_df_list.append(analyze_df)
    analyze_df = pd.concat(analyze_df_list, ignore_index=True)
    analyze_df.to_csv(f'backtest_result/analyze_data_{score_key}.csv', index=False)

if __name__ == "__main__":
    example()
