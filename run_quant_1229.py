import json
import os
import multiprocessing as mp
from datetime import datetime
from numba import njit

import numpy as np
from tqdm import tqdm  # 用于显示进度条

import pandas as pd

from get_feature_op import generate_price_extremes_signals, generate_price_unextremes_signals, \
    generate_price_extremes_reverse_signals


@njit
def _calculate_time_to_targets_numba(
        timestamps,
        closes,
        highs,
        lows,
        increase_targets,
        decrease_targets
):
    n = len(timestamps)

    # 用来存储结果的二维数组：行数量 x (len(increase_targets) + len(decrease_targets))
    # 由于时间戳是 datetime64[ns] 类型，我们可以先把时间戳列转换成 int64 存储
    # 等返回时再转换回去或存成字符串。
    # 如果找不到就存 -1
    result = np.full((n, len(increase_targets) + len(decrease_targets)), -1, dtype=np.int64)

    for i in range(n - 1):
        current_close = closes[i]

        # 先处理涨幅目标
        for inc_idx, inc_target in enumerate(increase_targets):
            # 注意这里我们把 0.1 也纳入循环了。如果你想跳过它，改成 if inc_target == 0.1: pass
            target_price = current_close * (1.0 + inc_target)
            for j in range(i + 1, n):
                if highs[j] >= target_price:
                    result[i, inc_idx] = timestamps[j]  # earliest timestamp
                    break  # 找到第一个就可以 break

        # 再处理跌幅目标
        for dec_idx, dec_target in enumerate(decrease_targets):
            target_price = current_close * (1.0 - dec_target)
            # result 的列索引要加上 len(increase_targets)，以免覆盖
            out_col = len(increase_targets) + dec_idx
            for j in range(i + 1, n):
                if lows[j] <= target_price:
                    result[i, out_col] = timestamps[j]
                    break

    return result


def calculate_time_to_targets_op(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the time it takes for the 'close' price to increase/decrease
    by specified percentages, and returns the corresponding future timestamp.

    如果找不到满足条件的行，则返回 NaT。
    """
    start_time = datetime.now()

    # 目标可根据需要调整
    increase_targets = [0.01, 0.02, 0.1]
    decrease_targets = [0.01, 0.02, 0.1]

    # 提取 Numpy 数组
    timestamps = df['timestamp'].astype('int64').values  # datetime64[ns] -> int64
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    # 运行 Numba 函数
    result_array = _calculate_time_to_targets_numba(
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
    return df

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


def example():
    backtest_path = 'backtest_result'
    file_path_list = ['kline_data/origin_data_1m_10000000_BTC-USDT-SWAP.csv', 'kline_data/origin_data_1m_10000000_ETH-USDT-SWAP.csv', 'kline_data/origin_data_1m_10000000_SOL-USDT-SWAP.csv', 'kline_data/origin_data_1m_10000000_TON-USDT-SWAP.csv']
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
        base_name = file_path.split('/')[-1].split('.')[0]
        origin_data_df = pd.read_csv(file_path)  # 只取最近1000条数据
        origin_data_df['timestamp'] = pd.to_datetime(origin_data_df['timestamp'])
        df1 = calculate_time_to_targets_op(origin_data_df)
        print()




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
