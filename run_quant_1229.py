import bisect
import json
import multiprocessing
import os
import multiprocessing as mp
import time
import traceback
from datetime import datetime

import numpy as np
from numba import prange, njit
from tqdm import tqdm  # 用于显示进度条

import pandas as pd

from get_feature_op import generate_price_extremes_signals, generate_price_unextremes_signals, \
    generate_price_extremes_reverse_signals, generate_trend_signals

function_map = {
    "price_extremes": generate_price_extremes_signals,
    "price_reverse_extremes": generate_price_extremes_reverse_signals,
    "price_change": generate_trend_signals
}


def gen_buy_sell_signal(data_df, profit, signal_name, side, signal_func, signal_param):
    """
    为data生成相应的买卖信号，并生成相应的buy_price, sell_price
    :param data_df:
    :param profit:
    :param period:
    :return:
    """
    signal_df = signal_func(data_df, signal_param)
    # 初始化 buy_price 和 sell_price 列，可以设置为 NaN 或者其他默认值
    signal_df['buy_price'] = None
    signal_df['sell_price'] = None
    # 找到包含Buy的列和包含Sell的列名
    target_col = [col for col in signal_df.columns if signal_name in col]
    if side == 'long':
        signal_df.rename(columns={target_col[0]: 'Buy'}, inplace=True)
        # 找到 Buy 为 1 的行，设置 buy_price 和 sell_price
        buy_rows = signal_df['Buy'] == 1
        signal_df.loc[buy_rows, 'buy_price'] = signal_df.loc[buy_rows, 'close']
        signal_df.loc[buy_rows, 'sell_price'] = signal_df.loc[buy_rows, 'close'] * (1 + profit)
        signal_df['Sell'] = 0
    else:
        signal_df.rename(columns={target_col[0]: 'Sell'}, inplace=True)
        # 找到 Sell 为 1 的行，设置 sell_price 和 buy_price
        sell_rows = signal_df['Sell'] == 1
        signal_df.loc[sell_rows, 'buy_price'] = signal_df.loc[sell_rows, 'close']
        signal_df.loc[sell_rows, 'sell_price'] = signal_df.loc[sell_rows, 'close'] * (1 - profit)
        signal_df['Buy'] = 0
    # 初始化 count 列
    signal_df['count'] = 0.01

    return signal_df

def optimized_gen_buy_sell_signal(data_df, signal_func, param_info, is_need_profit=False):
    """
    优化后的买卖信号生成函数。
    """
    start_time = time.time()
    # 1. 先生成基础策略列(change_Buy, change_Sell)，并得到 Buy / Sell
    signal_df = signal_func(data_df, param_info)

    signal_df['Buy'] = ((signal_df['change_Buy'] == 1) &
                        (signal_df['change_Buy'].shift(1) == 0)).astype(int)
    signal_df['Sell'] = ((signal_df['change_Sell'] == 1) &
                         (signal_df['change_Sell'].shift(1) == 0)).astype(int)

    # 2. 初始化需要的列
    signal_df['buy_price'] = 0.0
    signal_df['sell_price'] = 0.0
    signal_df['hold_time'] = np.nan
    signal_df['sell_time'] = pd.NaT  # 若为 DatetimeIndex，可用 NaT 占位

    # 3. 提取所有 Buy / Sell 信号的行索引
    buy_indices = signal_df.index[signal_df['Buy'] == 1].tolist()
    sell_indices = signal_df.index[signal_df['Sell'] == 1].tolist()

    # ===============================
    #    A. 处理 Buy -> Sell
    # ===============================
    for idx in buy_indices:
        # 找到 sell_indices 中第一个“大于 idx”的位置
        pos = bisect.bisect_right(sell_indices, idx)
        buy_close = signal_df.at[idx, 'close']

        found_match = False
        while pos < len(sell_indices):
            j = sell_indices[pos]  # 这是实际的行索引，可能是整数或时间戳
            sell_close = signal_df.at[j, 'close']
            sell_time = signal_df.at[j, 'timestamp']

            if not is_need_profit:
                # 不需要利润，遇到的第一个 Sell 即可
                signal_df.at[idx, 'buy_price'] = buy_close
                signal_df.at[idx, 'sell_price'] = sell_close
                # 记录持仓时长 (j - idx)，若 index = int 则为行数差
                signal_df.at[idx, 'hold_time'] = j - idx
                # 记录卖出的时间戳 (若 DatetimeIndex 则 j 即为 Timestamp)
                signal_df.at[idx, 'sell_time'] = sell_time
                found_match = True
                break
            else:
                # 需要利润：卖出价 > 买入价
                if sell_close > buy_close:
                    signal_df.at[idx, 'buy_price'] = buy_close
                    signal_df.at[idx, 'sell_price'] = sell_close
                    signal_df.at[idx, 'hold_time'] = j - idx
                    signal_df.at[idx, 'sell_time'] = sell_time
                    found_match = True
                    break

            pos += 1

        # 如果没有匹配到任何符合条件的卖出信号，则用最后一行兜底
        if not found_match:
            # 用最后一行的收盘价
            last_idx = signal_df.index[-1]
            signal_df.at[idx, 'buy_price'] = signal_df.at[idx, 'close']
            signal_df.at[idx, 'sell_price'] = signal_df.at[last_idx, 'close']
            signal_df.at[idx, 'hold_time'] = last_idx - idx
            signal_df.at[idx, 'sell_time'] = signal_df.at[last_idx, 'timestamp']

    # ===============================
    #    B. 处理 Sell -> Buy
    # ===============================
    for idx in sell_indices:
        # 在 buy_indices 中找第一个“大于 idx”的位置
        pos = bisect.bisect_right(buy_indices, idx)
        buy_close = signal_df.at[idx, 'close']
        found_match = False
        while pos < len(buy_indices):
            j = buy_indices[pos]
            sell_close = signal_df.at[j, 'close']
            sell_time = signal_df.at[idx, 'timestamp']

            if not is_need_profit:
                # 不需要利润，遇到的第一个 Buy 直接反手
                signal_df.at[idx, 'buy_price'] = buy_close
                signal_df.at[idx, 'sell_price'] = sell_close
                # 以“卖出行”作为基准记录，也可以自行调整到买入行
                signal_df.at[idx, 'hold_time'] = j - idx
                signal_df.at[idx, 'sell_time'] = sell_time
                found_match = True
                break
            else:
                # 需要利润：买入价 < 卖出价(原先的价)
                if buy_close > sell_close:
                    signal_df.at[idx, 'buy_price'] = buy_close
                    signal_df.at[idx, 'sell_price'] = sell_close
                    signal_df.at[idx, 'hold_time'] = j - idx
                    signal_df.at[idx, 'sell_time'] = sell_time
                    found_match = True
                    break

            pos += 1

        # 若没匹配到任何符合条件的买入信号，则用最后一行兜底
        if not found_match:
            last_idx = signal_df.index[-1]
            signal_df.at[idx, 'buy_price'] = buy_close
            signal_df.at[idx, 'sell_price'] = signal_df.at[last_idx, 'close']
            signal_df.at[idx, 'hold_time'] = last_idx - idx
            signal_df.at[idx, 'sell_time'] = signal_df.at[last_idx, 'timestamp']

    # 4. 把指定列移动到前面；其余列保持原顺序
    front_cols = ['Buy', 'Sell', 'buy_price', 'sell_price', 'hold_time', 'sell_time']
    all_cols = list(signal_df.columns)
    remaining_cols = [col for col in all_cols if col not in front_cols]
    signal_df = signal_df[front_cols + remaining_cols]

    # 增加一个 count 列（与原逻辑保持一致）
    signal_df['count'] = 0.01
    print(f"optimized_gen_buy_sell_signal cost time: {time.time() - start_time}")
    return signal_df

def debug_gen_buy_sell_signal(data_df, signal_func, param_info):
    """
    为data生成相应的买卖信号，并生成相应的buy_price, sell_price
    :param data_df:
    :param profit:
    :param period:
    :return:
    """
    start_time = time.time()
    signal_df = signal_func(data_df, param_info)
    # 初始化 Buy 和 Sell 列
    signal_df['Buy'] = 0
    signal_df['Sell'] = 0
    is_need_profit = True

    # 生成 Buy 信号
    signal_df['Buy'] = ((signal_df['change_Buy'] == 1) & (signal_df['change_Buy'].shift(1) == 0)).astype(int)

    # 生成 Sell 信号
    signal_df['Sell'] = ((signal_df['change_Sell'] == 1) & (signal_df['change_Sell'].shift(1) == 0)).astype(int)

    # 初始化 buy_price 和 sell_price 列
    signal_df['buy_price'] = 0.0
    signal_df['sell_price'] = 0.0

    for i in range(len(signal_df)):
        if signal_df.loc[i, 'Buy'] == 1:
            signal_df.loc[i, 'buy_price'] = signal_df.loc[i, 'close']
            # 查找下一个 Sell 信号
            for j in range(i + 1, len(signal_df)):
                if signal_df.loc[j, 'Sell'] == 1:
                    if is_need_profit:
                         if signal_df.loc[j, 'close'] > signal_df.loc[i, 'close']:
                             signal_df.loc[i, 'sell_price'] = signal_df.loc[j, 'close']
                             break
                    else:
                        signal_df.loc[i, 'sell_price'] = signal_df.loc[j, 'close']
                        break
            signal_df.loc[i, 'sell_price'] = signal_df.loc[j, 'close']
        elif signal_df.loc[i, 'Sell'] == 1:
            signal_df.loc[i, 'buy_price'] = signal_df.loc[i, 'close']
            # 查找下一个 Buy 信号
            for j in range(i + 1, len(signal_df)):
                if signal_df.loc[j, 'Buy'] == 1:
                    if is_need_profit:
                        if signal_df.loc[j, 'close'] < signal_df.loc[i, 'close']:
                            signal_df.loc[i, 'sell_price'] = signal_df.loc[j, 'close']
                            break
                    else:
                        signal_df.loc[i, 'sell_price'] = signal_df.loc[j, 'close']
                        break
            signal_df.loc[i, 'sell_price'] = signal_df.loc[j, 'close']

    # 将最后四列移动到最前面
    cols = ['Buy', 'Sell', 'buy_price', 'sell_price'] + [col for col in signal_df.columns if
                                                         col not in ['Buy', 'Sell', 'buy_price', 'sell_price']]
    signal_df = signal_df[cols]
    signal_df['count'] = 0.01
    print(f"debug_gen_buy_sell_signal cost time: {time.time() - start_time}")
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


def deal_pending_order(pending_order_list, row, position_info, lever, total_money, max_time_diff=2 * 1,
                       max_sell_time_diff=1000000, power=1):
    """
    处理委托单
    """
    high = row.high
    low = row.low
    close = row.close
    close_available_funds = position_info['close_available_funds']
    timestamp = row.timestamp
    history_order_list = []
    fee = 0.0007  # 手续费
    check_flag = True  # 是否真实的按照能否买入来回测

    for order in pending_order_list:
        if order['side'] == 'kai':  # 开仓
            # 计算时间差
            time_diff = calculate_time_diff_minutes(timestamp, order['timestamp'])
            if time_diff < max_time_diff:
                if order['type'] == 'long':  # 开多仓
                    if order['buy_price'] > low or check_flag:  # 买入价格高于最低价
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
                    if order['buy_price'] < high or check_flag:  # 买入价格低于最高价
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
            profit_value = power * (order['sell_price'] - order['buy_price'])
            pin_time_diff = calculate_time_diff_minutes(timestamp, order['kai_time'])
            if order['type'] == 'long':  # 平多仓
                if order['sell_price'] < high:
                    order['side'] = 'done'
                    order['ping_time'] = timestamp
                    # 计算收益并更新总资金
                    profit = order['count'] * (order['sell_price'] - order['buy_price'] - fee * order['sell_price'])
                    order['profit'] = profit
                    order['time_cost'] = calculate_time_diff_minutes(timestamp, order['timestamp'])
                    total_money += profit
                else:
                    # 对超时的调整售出价格
                    if pin_time_diff > max_sell_time_diff:
                        order['sell_price'] = close + profit_value
                        order['kai_time'] = timestamp
                        order['message'] = 'sell time out'
            if order['type'] == 'short':  # 平空仓
                if order['sell_price'] > low:
                    order['side'] = 'done'
                    order['ping_time'] = timestamp
                    # 计算收益并更新总资金
                    profit = order['count'] * (order['buy_price'] - order['sell_price'] - fee * order['sell_price'])
                    order['profit'] = profit
                    order['time_cost'] = calculate_time_diff_minutes(timestamp, order['timestamp'])
                    total_money += profit
                else:
                    # 对超时的调整售出价格
                    if pin_time_diff > max_sell_time_diff:
                        order['sell_price'] = close + profit_value
                        order['kai_time'] = timestamp
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


def process_signals(signal_df, lever, total_money, init_money, max_sell_time_diff=1000000, power=1):
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
            pending_order_list, row, position_info, lever, total_money, max_sell_time_diff=max_sell_time_diff,
            power=power
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

    start_time = time.time()
    profit, period, data_df, lever, init_money, max_sell_time_diff, power, signal_name, side, signal_func, signal_param = args
    # print(f"time{datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')} profit: {profit}, period: {period},'max_sell_time_diff': {max_sell_time_diff}, 'power': {power} start")
    signal_df = gen_buy_sell_signal(data_df, profit, signal_name, side, signal_func, signal_param)
    last_data = process_signals(signal_df, lever, init_money, init_money, max_sell_time_diff=max_sell_time_diff,
                                power=power)
    last_data.update({'profit': profit, 'period': period, 'max_sell_time_diff': max_sell_time_diff, 'power': power})
    # print(f"time{datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')} profit: {profit}, period: {period},'max_sell_time_diff': {max_sell_time_diff}, 'power': {power}  cost time: {time.time() - start_time} end")
    return last_data


def add_trading_signals(data_df):
    """
    根据给定的交易逻辑，在股票k线数据DataFrame中添加交易信号、买卖价格和卖出时间。

    Args:
        data_df: pandas DataFrame，包含股票k线数据，字段包括 'high', 'low', 'close', 'open', 'timestamp'。

    Returns:
        pandas DataFrame，添加了 'signal', 'buy_price', 'sell_price', 'sell_time' 字段。
    """
    data_df['signal'] = 0
    data_df['buy_price'] = 0.0
    data_df['sell_price'] = 0.0
    data_df['sell_time'] = None

    # 初始化状态：开多
    current_signal = 1
    data_df.loc[0, 'signal'] = 1
    data_df.loc[0, 'buy_price'] = data_df.loc[0, 'close']
    last_buy_index = 0

    for i in range(1, len(data_df)):
        previous_close = data_df.loc[i - 1, 'close']
        previous_high = data_df.loc[i - 1, 'high']
        previous_low = data_df.loc[i - 1, 'low']
        current_high = data_df.loc[i, 'high']
        current_low = data_df.loc[i, 'low']

        if current_signal == 1:  # 持有多单
            if current_low < previous_low:
                # 平多开空
                data_df.loc[i, 'signal'] = -1
                data_df.loc[i, 'buy_price'] = previous_low

                # 回填之前的卖出信息
                data_df.loc[last_buy_index, 'sell_price'] = previous_low
                data_df.loc[last_buy_index, 'sell_time'] = data_df.loc[i, 'timestamp']

                current_signal = -1
                last_buy_index = i
        elif current_signal == -1:  # 持有空单
            if current_high > previous_high:
                # 平空开多
                data_df.loc[i, 'signal'] = 1
                data_df.loc[i, 'buy_price'] = previous_high

                # 回填之前的卖出信息
                data_df.loc[last_buy_index, 'sell_price'] = previous_high
                data_df.loc[last_buy_index, 'sell_time'] = data_df.loc[i, 'timestamp']

                current_signal = 1
                last_buy_index = i

    # 处理最后一笔交易
    if data_df.loc[last_buy_index, 'signal'] != 0:
        # 查找最后一次信号对应的卖出信息
        found_sell = False
        for i in range(last_buy_index + 1, len(data_df)):
            previous_close = data_df.loc[last_buy_index, 'close'] if data_df.loc[last_buy_index, 'signal'] == 1 else \
            data_df.loc[last_buy_index, 'buy_price']
            previous_high = data_df.loc[last_buy_index, 'high']
            previous_low = data_df.loc[last_buy_index, 'low']

            if data_df.loc[last_buy_index, 'signal'] == 1 and data_df.loc[i, 'low'] < previous_low:
                data_df.loc[last_buy_index, 'sell_price'] = previous_low
                data_df.loc[last_buy_index, 'sell_time'] = data_df.loc[i, 'timestamp']
                found_sell = True
                break
            elif data_df.loc[last_buy_index, 'signal'] == -1 and data_df.loc[i, 'high'] > previous_high:
                data_df.loc[last_buy_index, 'sell_price'] = previous_high
                data_df.loc[last_buy_index, 'sell_time'] = data_df.loc[i, 'timestamp']
                found_sell = True
                break

        # 如果循环结束仍然没有找到卖出信号，则以最后一行的收盘价作为卖出价
        if not found_sell:
            data_df.loc[last_buy_index, 'sell_price'] = data_df.loc[len(data_df) - 1, 'close']
            data_df.loc[last_buy_index, 'sell_time'] = data_df.loc[len(data_df) - 1, 'timestamp']
    data_df['profit_ratio'] = 0.0  # 初始化 profit_ratio 列
    for i in range(len(data_df)):
        if data_df.loc[i, 'signal'] != 0 and data_df.loc[i, 'sell_price'] != 0:
            # 当有卖出价格时才计算 profit_ratio
            if data_df.loc[i, 'buy_price']!=0:
                # 当之前的买入价格不为零的时候才计算
                if data_df.loc[i, 'signal'] == 1:
                    data_df.loc[i, 'profit_ratio'] = (data_df.loc[i, 'sell_price'] - data_df.loc[i, 'buy_price']) / data_df.loc[i, 'buy_price'] * 100
                elif data_df.loc[i, 'signal'] == -1:  # 上一笔是开空
                    data_df.loc[i, 'profit_ratio'] = (data_df.loc[i, 'buy_price'] - data_df.loc[i, 'sell_price']) / data_df.loc[i, 'sell_price'] * 100

    # 调整列的顺序
    new_column_order = ['signal', 'buy_price', 'sell_price', 'sell_time', 'profit_ratio'] + [col for col in data_df.columns if col not in ['signal', 'buy_price', 'sell_price', 'sell_time', 'profit_ratio']]
    data_df = data_df.reindex(columns=new_column_order)

    # 找到profit_ratio不为0的行的数量
    signal_count = data_df[data_df['signal'] != 0].shape[0]
    cost = 0.07 * signal_count
    # 输出profit_ratio的和
    print(f"profit_ratio: {data_df['profit_ratio'].sum()} 手续费: {cost} 数量: {signal_count}")
    return data_df

def add_trading_signals_op(data_df,is_need_profit=False, max_hold_time=999999):
    data_df = generate_signals(data_df)
    return match_sell_info(data_df, is_need_profit, max_hold_time)

def generate_signals(data_df):
    """
    根据交易逻辑生成买入和卖出信号。

    Args:
        data_df: pandas DataFrame，包含股票k线数据。

    Returns:
        pandas DataFrame，添加了 'signal', 'buy_price' 字段。
    """
    data_df['signal'] = 0
    data_df['buy_price'] = 0.0

    current_signal = 1
    data_df.loc[0, 'signal'] = 1
    data_df.loc[0, 'buy_price'] = data_df.loc[0, 'close']

    for i in range(1, len(data_df)):
        previous_high = data_df.loc[i - 1, 'high']
        previous_low = data_df.loc[i - 1, 'low']
        current_high = data_df.loc[i, 'high']
        current_low = data_df.loc[i, 'low']

        if current_signal == 1 and current_low < previous_low:  # 平多开空
            data_df.loc[i, 'signal'] = -1
            data_df.loc[i, 'buy_price'] = previous_low
            current_signal = -1
        elif current_signal == -1 and current_high > previous_high:  # 平空开多
            data_df.loc[i, 'signal'] = 1
            data_df.loc[i, 'buy_price'] = previous_high
            current_signal = 1

    return data_df

def match_sell_info(data_df, is_need_profit=False, max_hold_time=999999):
    """
    根据信号匹配卖出价格和时间。

    当 is_need_profit=True 时，会尝试在后续信号中寻找能带来正收益的平仓点；
    若一直找不到正收益的平仓点，或持仓天数超过 max_hold_time 时，则也可以使用后续信号（无论是否盈利）平仓；
    若到最后也没有信号，则使用最后一行收盘价平仓。

    Args:
        data_df (pd.DataFrame): 数据表，至少需要包含以下列：
            'signal' (int): 交易信号。1 表示开多，-1 表示开空，0 表示无交易。
            'buy_price' (float): 对应开仓时的价格。
            'close' (float): 数据的收盘价，用于无法找到平仓信号时使用。
            'timestamp': 时间戳/日期，用于记录交易时间，可为字符串或 datetime。
        is_need_profit (bool): 是否强制要求找到盈利的平仓点。
        max_hold_time (int): 允许持仓的最大天数(或K线数量)。超过则不再强制需求盈利。

    Returns:
        pd.DataFrame: 在原有数据表基础上，新增以下列：
            'sell_price' (float): 平仓价格
            'sell_time' (any): 平仓所在行的 timestamp
            'profit_ratio' (float): 收益率（单位：%）
            'hold_time' (int): 卖出行索引 - 买入行索引
    """

    # 新增需要的列
    data_df['sell_price'] = 0.0
    data_df['sell_time'] = None
    data_df['profit_ratio'] = 0.0
    data_df['hold_time'] = 0

    # 所有有交易信号的行索引
    signal_indexes = data_df.index[data_df['signal'] != 0].tolist()
    n = len(signal_indexes)

    # 如果没有任何信号，直接返回
    if n == 0:
        print("无任何交易信号")
        return data_df

    # 逐条处理每一个交易信号
    for i in range(n):
        buy_idx = signal_indexes[i]
        buy_signal = data_df.loc[buy_idx, 'signal']
        buy_price = data_df.loc[buy_idx, 'buy_price']

        # 如果已经是最后一个信号，则平仓价格只能用整张表最后一行的 close
        if i == n - 1:
            data_df.loc[buy_idx, 'sell_price'] = data_df.loc[len(data_df) - 1, 'close']
            data_df.loc[buy_idx, 'sell_time'] = data_df.loc[len(data_df) - 1, 'timestamp']
            data_df.loc[buy_idx, 'hold_time'] = (len(data_df) - 1) - buy_idx
        else:
            # 从下一条信号开始，依次向后寻找“可平仓”的信号
            matched = False
            for j in range(i + 1, n):
                sell_idx = signal_indexes[j]
                sell_signal = data_df.loc[sell_idx, 'signal']
                sell_price_candidate = data_df.loc[sell_idx, 'buy_price']

                # 计算潜在利润
                if buy_signal == 1:
                    # 多单，潜在利润 = 卖出价 - 买入价
                    potential_profit = sell_price_candidate - buy_price
                else:
                    # 空单，潜在利润 = 买入价 - 卖出价
                    potential_profit = buy_price - sell_price_candidate

                # 计算“持仓时长” = 卖出信号索引 - 买入信号索引
                potential_hold_time = sell_idx - buy_idx

                # 如果没有强制要求盈利，或已经盈利，或已经超时，则可平仓
                # 注意：一旦 hold_time 超过 max_hold_time，就不再要求是正收益，也可以平仓
                if (
                    (not is_need_profit)
                    or (potential_profit > 0)
                    or (potential_hold_time >= max_hold_time)
                ):
                    data_df.loc[buy_idx, 'sell_price'] = sell_price_candidate
                    data_df.loc[buy_idx, 'sell_time'] = data_df.loc[sell_idx, 'timestamp']
                    data_df.loc[buy_idx, 'hold_time'] = potential_hold_time
                    matched = True
                    break

            # 如果在剩余信号里都没匹配到，则用最后一根 K 线 收盘价平仓
            if not matched:
                data_df.loc[buy_idx, 'sell_price'] = data_df.loc[len(data_df) - 1, 'close']
                data_df.loc[buy_idx, 'sell_time'] = data_df.loc[len(data_df) - 1, 'timestamp']
                data_df.loc[buy_idx, 'hold_time'] = (len(data_df) - 1) - buy_idx

    # 修改 signal 为 1 的行
    mask_signal_1 = (data_df['signal'] == 1) & (data_df['buy_price'] > data_df['close'])
    data_df.loc[mask_signal_1, 'sell_price'] = data_df.loc[mask_signal_1, 'buy_price']

    # 修改 signal 为 -1 的行
    mask_signal_0 = (data_df['signal'] == -1) & (data_df['buy_price'] < data_df['close'])
    data_df.loc[mask_signal_0, 'sell_price'] = data_df.loc[mask_signal_0, 'buy_price']

    # 计算修改的行数
    fix_count = mask_signal_1.sum() + mask_signal_0.sum()
    print(f"fix_count: {fix_count} cost {0.07 * fix_count}")


    # ------ 统一计算收益率 ------
    for idx in signal_indexes:
        buy_signal = data_df.loc[idx, 'signal']
        buy_price = data_df.loc[idx, 'buy_price']
        sell_price = data_df.loc[idx, 'sell_price']

        if sell_price != 0:
            if buy_signal == 1:
                # 多单收益率
                data_df.loc[idx, 'profit_ratio'] = (sell_price - buy_price) / buy_price * 100
            else:
                # 空单收益率
                data_df.loc[idx, 'profit_ratio'] = (buy_price - sell_price) / sell_price * 100

    # ------ 输出统计信息 ------
    signal_count = len(signal_indexes)
    cost = 0.07 * signal_count
    total_profit = data_df['profit_ratio'].sum()

    print(
        f"所有信号条数 = {signal_count} | "
        f"总收益 = {total_profit:.2f}% | "
        f"手续费 = {cost:.2f}"
        f"平均持仓时间 = {data_df['hold_time'].sum() / signal_count:.2f}"
    )

    # ------ 调整列顺序 ------
    new_column_order = [
        'signal', 'buy_price', 'sell_price', 'sell_time',
        'profit_ratio', 'hold_time'
    ] + [
        col for col in data_df.columns
        if col not in ['signal', 'buy_price', 'sell_price', 'sell_time', 'profit_ratio', 'hold_time']
    ]
    data_df = data_df.reindex(columns=new_column_order)

    return data_df

def debug_calculate_combination():
    """多进程计算单个组合的回测结果"""
    lever = 100
    init_money = 1000000
    max_sell_time_diff = 1000000
    power = 1
    start_time = time.time()
    # data_df = pd.read_csv("temp/TON_1m_10000.csv")
    # data_df = pd.read_csv("temp/SOL_1m_10000.csv")
    data_df = pd.read_csv("kline_data/origin_data_30m_10000000_SOL-USDT-SWAP.csv")

    data_df = data_df[-50000:]

    # 重置索引
    data_df.reset_index(drop=True, inplace=True)
    print(f"长度: {data_df.shape[0]}")
    temp_df1 = add_trading_signals_op(data_df)
    temp_df2 = add_trading_signals_op(data_df, True)
    # for i in range(10, 100, 10):
    #     temp_df3 = add_trading_signals_op(data_df, True, i)
    #     print(f"max_hold_time: {i} profit: {temp_df3['profit_ratio'].sum()} cost time: {time.time() - start_time}")
    # last_data_list = []
    # for i in range(1, 10):
    #     param_info = {"period": i}
    #     signal_func = generate_trend_signals
    #     # print(f"time{datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')} profit: {profit}, period: {period},'max_sell_time_diff': {max_sell_time_diff}, 'power': {power} start")
    #     signal_df = optimized_gen_buy_sell_signal(data_df, signal_func, param_info)
    #     # 统计signal_df中hold_time不为nan的行中的hold_time的平均值
    #     hold_time = signal_df[~signal_df['hold_time'].isna()]['hold_time'].mean()
    #     print(f"hold_time: {hold_time}")
    #
    #     # 将signal_df的最后四列放到前面
    #
    #
    #     last_data = process_signals(signal_df, lever, init_money, init_money, max_sell_time_diff=max_sell_time_diff,power=power)
    #     last_data_list.append(last_data)
    # # print(f"time{datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')} profit: {profit}, period: {period},'max_sell_time_diff': {max_sell_time_diff}, 'power': {power}  cost time: {time.time() - start_time} end")
    # # 将last_data_list转换为DataFrame
    # last_data_df = pd.DataFrame(last_data_list)
    # return last_data_df


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
        df_list[i]['source_df'] = f'df_{i + 1}'

    merged_df = df_list[0]
    for i in range(1, len(df_list)):
        merged_df = pd.merge(merged_df, df_list[i], on=['profit', 'period'], how='outer', suffixes=('', f'_{i + 1}'))

    # 重命名和排序
    def categorize_and_sort_cols(df):
        # 识别不同类别的列
        source_df_cols = [col for col in df.columns if 'source_df' in col]
        profit_ratio_cols = [col for col in df.columns if 'profit_ratio' in col]
        other_cols = [col for col in df.columns if
                      col not in source_df_cols and col not in profit_ratio_cols and col != 'score' and col != 'score_plus' and col != 'score_mul']

        # 对每种类别的列进行排序
        source_df_cols.sort()
        profit_ratio_cols.sort()
        other_cols.sort()

        # 重组列的顺序
        new_cols_order = other_cols + source_df_cols + profit_ratio_cols
        return new_cols_order

    new_cols_order = categorize_and_sort_cols(merged_df)
    merged_df = merged_df.reindex(columns=new_cols_order)

    # 计算分数
    profit_ratio_cols = [col for col in merged_df.columns if 'profit_ratio' in col and 'source_df' not in col]
    profit_ratio_cols.sort()
    if len(profit_ratio_cols) >= 3:
        merged_df['score'] = 10000 * merged_df[profit_ratio_cols[0]] * merged_df[profit_ratio_cols[1]] * merged_df[
            profit_ratio_cols[2]]
        merged_df['score_plus'] = merged_df[profit_ratio_cols[0]] + merged_df[profit_ratio_cols[1]] + merged_df[
            profit_ratio_cols[2]]
        merged_df['score_mul'] = merged_df['score_plus'] * merged_df['score']
        merged_df['hold_time_score_plus'] = merged_df['hold_time_score'] + merged_df['hold_time_score_2'] + merged_df[
            'hold_time_score_3']
    elif len(profit_ratio_cols) >= 2:
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
                signal_df = generate_price_extremes_signals(origin_df, periods=[x for x in
                                                                                range(start_period, end_period,
                                                                                      step_period)])
            else:
                signal_df = generate_price_extremes_reverse_signals(origin_df, periods=[x for x in
                                                                                        range(start_period, end_period,
                                                                                              step_period)])

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


def safe_calculate_combination(params):
    """
    包装原先的 calculate_combination 函数，捕获异常，以免某一次计算失败导致整个进程中断。
    返回值中增加 'error' 字段，用于记录异常信息。
    """
    try:
        result = calculate_combination(params)
        # 为了更好地区分成功与失败，这里也可加入一个标志
        result["error"] = None
        return result
    except Exception as e:
        return {
            "error": str(e),
            # 如果需要保留部分上下文信息，可一并返回
            "params": params
        }

def process_with_timeout(func, args, timeout):
    num_processes = max(1, (os.cpu_count()))  # 避免出现负数
    with multiprocessing.Pool(processes=num_processes - 3) as pool:
        results = []
        async_results = []

        for param in args:
            async_result = pool.apply_async(func, (param,))
            async_results.append(async_result)

        for i, async_result in enumerate(async_results):
            try:
                res = async_result.get(timeout=timeout)
                results.append(res)
            except multiprocessing.TimeoutError:
                # 去除args[i]中的第三个参数
                param = args[i][:2] + args[i][3:]
                print(f"Task for parameter: {param} timed out and was skipped.")

        return results


def detail_backtest():
    """
    更加详细的回测，是为已经比较好的策略指定的参数组合生成详细回测数据。
    """

    good_strategy_path = 'backtest_result/good_strategy_df.csv'
    timeout_limit = 1000
    sell_time_diff_step = 100
    sell_time_diff_start = 100
    sell_time_diff_end = 10000
    max_sell_time_diff_list = [x for x in range(sell_time_diff_start, sell_time_diff_end, sell_time_diff_step)]
    # 为了覆盖更大范围，追加一个极大的阈值
    max_sell_time_diff_list.append(1000000)

    power_step = 1
    power_start = 0
    power_end = 5
    power_list = [x for x in range(power_start, power_end, power_step)]

    file_path_list = [
        'kline_data/origin_data_1m_10000000_BTC-USDT-SWAP.csv',
        'kline_data/origin_data_1m_10000000_ETH-USDT-SWAP.csv',
        'kline_data/origin_data_1m_10000000_SOL-USDT-SWAP.csv',
        'kline_data/origin_data_1m_10000000_TON-USDT-SWAP.csv'
    ]

    # 读取“好策略”列表
    df_good_strategies = pd.read_csv(good_strategy_path)
    # 删除signal_name为base_line的行
    df_good_strategies = df_good_strategies[~df_good_strategies['signal_name'].str.contains('baseline')]
    # 删除group_size大于10000的行
    df_good_strategies = df_good_strategies[df_good_strategies['group_size'] <= 10000]
    # 找到df_good_strategies中result_path为空的行
    df_good_strategies = df_good_strategies[df_good_strategies['result_path'].isna()]

    # 按照 key_name 分组
    df_group = df_good_strategies.groupby('key_name')

    # 遍历每个分组
    for key_name_group, value_df in df_group:
        # 根据 key_name 判断信号类型
        if 'reverse' in key_name_group:
            signal_type = 'price_reverse_extremes'
        else:
            signal_type = 'price_extremes'

        # 提取真正的币种标识，下面在 file_path_list 中匹配
        coin_name = key_name_group.split('-')[0].split('_')[-1]

        # 找到对应的文件路径
        matched_file_path = None
        for fpath in file_path_list:
            if coin_name in fpath:
                matched_file_path = fpath
                break

        if not matched_file_path:
            print(f"未找到 {coin_name} 对应的 K线文件路径，跳过该分组。")
            continue

        # 读取 K 线文件
        df_kline = pd.read_csv(matched_file_path)

        start_time = time.time()

        # 遍历该分组下每一行
        for index, row in value_df.iterrows():
            try:
                signal_name = row['signal_name']
                period = int(signal_name.split('_')[1])

                target_name = row['target_name']
                profit = float(target_name.split('_')[2])
                if profit > 0.02:
                    print(f"profit: {profit} 大于0.04，跳过")
                    continue

                # 根据 target_name 判断多空
                if "high" in target_name:
                    side = 'long'
                else:
                    side = 'short'

                # 组装 signal_param
                signal_param = {
                    "periods": [period]
                }

                # 构建参数列表
                parameter_list = []
                # 这里的 100 和 10000000 是原先固定传入的参数
                fixed_params = (profit, period, df_kline, 100, 10000000)

                for max_sell_time_diff in max_sell_time_diff_list:
                    for power in power_list:
                        params = (
                                fixed_params
                                + (max_sell_time_diff, power)
                                + (signal_name, side, function_map[signal_type], signal_param)
                        )
                        parameter_list.append(params)

                # 生成输出文件路径
                output_path = (
                    f"backtest_result/detail_{coin_name}"
                    f"_signal_name_{signal_name}_period_{period}_profit_{profit}_side_{side}"
                    f"_len_{len(parameter_list)}_{sell_time_diff_step}_{sell_time_diff_start}"
                    f"_{sell_time_diff_end}_{power_step}_{power_start}_{power_end}.csv"
                )

                # 如果结果文件已存在，跳过计算
                if os.path.exists(output_path):
                    print(f"结果文件 {output_path} 已存在，跳过计算")
                    # 如果想在原 DataFrame 中记录 result_path，需要用 loc
                    df_good_strategies.loc[index, 'result_path'] = output_path
                    # 将df_good_strategies保存到文件
                    df_good_strategies.to_csv(good_strategy_path, index=False)
                    continue
                print(f"开始计算: {output_path}")

                # for params in parameter_list:
                #     calculate_combination(params)

                # 使用进程池并行计算
                results = process_with_timeout(safe_calculate_combination, parameter_list, timeout_limit)

                # 将计算结果转换为 DataFrame
                result_df = pd.DataFrame(results)
                result_df.to_csv(output_path, index=False)

                print(f"结果已保存到 {output_path} 耗时：{time.time() - start_time:.2f} 秒")

                # 如果想在原 DataFrame 中记录 result_path，需要用 loc
                df_good_strategies.loc[index, 'result_path'] = output_path
                # 将df_good_strategies保存到文件
                df_good_strategies.to_csv(good_strategy_path, index=False)
            except Exception as e:
                traceback.print_exc()
                print(f"计算 时出现异常：{e} {row}")


def truly_backtest():
    """
    之前一一组合生成回测数据的代码
    :return:
    """
    backtest_path = 'backtest_result'
    file_path_list = ['kline_data/origin_data_1m_10000000_BTC-USDT-SWAP.csv',
                      'kline_data/origin_data_1m_10000000_ETH-USDT-SWAP.csv',
                      'kline_data/origin_data_1m_10000000_SOL-USDT-SWAP.csv',
                      'kline_data/origin_data_1m_10000000_TON-USDT-SWAP.csv']
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
            data_df = origin_data_df[
                (origin_data_df['timestamp'] >= start_time) & (origin_data_df['timestamp'] <= end_time)]

            data_len = len(data_df)

            # 获取data_df的初始时间与结束时间
            start_time = data_df.iloc[0].timestamp
            end_time = data_df.iloc[-1].timestamp
            print(f"开始时间：{start_time}，结束时间：{end_time} 长度：{data_len} key = {key}")
            # 生成time_key
            time_key_str = f"{start_time.strftime('%Y%m%d%H%M%S')}_{end_time.strftime('%Y%m%d%H%M%S')}"

            # 准备参数组合
            combinations = [(profit, period, data_df, lever, init_money) for profit in profit_list for period in
                            period_list]
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
        if float(profit) - 0.0007 < 0:
            continue

        # 计算 profit（示例中写死扣除了 0.0007）
        temp_df[profit_key] = float(profit) - 0.0007

        # 计算 4 种评分 (score1～4)
        # 注意要先排除 mean_col 和 nan_ratio_col 可能的 NaN 或 0，这里假设数据完整
        temp_df[score1_col] = (
                (temp_df[profit_key] - temp_df[nan_ratio_col]) * temp_df[profit_key] * 100
                / temp_df[mean_col] * 10000
                / temp_df[mean_col] * 10000
        )
        temp_df[score2_col] = (
                (temp_df[profit_key] - temp_df[nan_ratio_col]) * 100
                / temp_df[mean_col] * 10000
                / temp_df[mean_col] * 10000
        )
        temp_df[score3_col] = (
                (temp_df[profit_key] - temp_df[nan_ratio_col]) * temp_df[profit_key] * 100
                / temp_df[mean_col] * 10000
        )
        temp_df[score4_col] = (
                (temp_df[profit_key] - temp_df[nan_ratio_col]) * 100
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
        if merged_top_df.empty:
            continue
        for sc in score_columns:
            merged_top_df[f"{sc}_median"] = merged_top_df[f"{sc}_median"].round(4)
            # 计算f"{sc}_total"，值为f"{sc}_median"取log10后的和
            merged_top_df[f"{sc}_total"] = np.log10(merged_top_df[f"{sc}_median"]) + np.log10(merged_top_df[f"{sc}"])

        # 最后一次性调用 convert_df()，把这个分组的 top 数据处理完
        melt_df = covert_df(merged_top_df)
        result_df_list.append(melt_df)

    # 合并所有结果并输出到 CSV
    result_df = pd.concat(result_df_list, ignore_index=True)
    result_df.to_csv(output_path, index=False)
    print(f"结果已保存到 {output_path} (耗时: {time.time() - start_time:.4f}秒)")
    return result_df


def get_good_strategy():
    good_strategy_path = 'backtest_result/good_strategy_df.csv'
    # df = pd.read_csv(good_strategy_path)

    step = 100
    top_n = 2
    file_path_list = ['backtest_result/analyze_data_mean.csv', 'backtest_result/analyze_data_median.csv']
    result_list = []
    for file_path in file_path_list:
        low_output_path = file_path.replace('.csv', f'_low_{step}_{top_n}.csv')
        high_output_path = file_path.replace('.csv', f'_high_{step}_{top_n}.csv')
        if os.path.exists(low_output_path) and os.path.exists(high_output_path):
            result_list.append(pd.read_csv(low_output_path))
            result_list.append(pd.read_csv(high_output_path))

            print(f"结果文件 {low_output_path} 和 {high_output_path} 已存在，跳过计算")
            continue
        result_df = pd.read_csv(file_path)
        base_key_word_list = [f"score{i}" for i in range(1, 5)]
        key_word_list = base_key_word_list.copy()
        for key_word in base_key_word_list:
            key_word_list.append(f"{key_word}_median")
            key_word_list.append(f"{key_word}_total")
        # 过滤掉target_value小于0的行
        result_df = result_df[result_df['target_value'] > 0]
        # 筛选出target_name的值包含low的行
        low_result_df = result_df[result_df['target_name'].str.contains('low')]
        high_result_df = result_df[result_df['target_name'].str.contains('high')]

        low_result_df_list = []
        high_result_df_list = []
        for key_word in key_word_list:
            key_word_low_result_df = low_result_df[low_result_df['target_name'].str.contains(key_word)]
            key_word_high_result_df = high_result_df[high_result_df['target_name'].str.contains(key_word)]

            # 按照target_value降序排列
            key_word_low_result_df = key_word_low_result_df.sort_values(by='target_value', ascending=False)
            key_word_high_result_df = key_word_high_result_df.sort_values(by='target_value', ascending=False)

            current_df = key_word_low_result_df
            while not current_df.empty:
                top_2 = current_df.head(top_n)
                max_group_size = top_2['group_size'].max()
                current_df = current_df[current_df['group_size'] > max_group_size + step]
                low_result_df_list.append(top_2)

            current_df = key_word_high_result_df
            while not current_df.empty:
                top_2 = current_df.head(top_n)
                max_group_size = top_2['group_size'].max()
                current_df = current_df[current_df['group_size'] > max_group_size + step]
                high_result_df_list.append(top_2)
        final_low_result_df = pd.concat(low_result_df_list)
        final_high_result_df = pd.concat(high_result_df_list)
        result_list.append(final_low_result_df)
        result_list.append(final_high_result_df)
        final_low_result_df.to_csv(low_output_path, index=False)
        final_high_result_df.to_csv(high_output_path, index=False)
    result_df = pd.concat(result_list)
    result_df['result_path'] = ''
    result_df.to_csv(good_strategy_path, index=False)


def example():
    file_path_list = ['kline_data/origin_data_1m_10000000_BTC-USDT-SWAP.csv',
                      'kline_data/origin_data_1m_10000000_ETH-USDT-SWAP.csv',
                      'kline_data/origin_data_1m_10000000_SOL-USDT-SWAP.csv',
                      'kline_data/origin_data_1m_10000000_TON-USDT-SWAP.csv']

    # 示例-1：根据统一的数据获取好的策略，然后再获取详细的结果
    # get_good_strategy()

    debug_calculate_combination()

    # 示例0:更加详细的回测考虑超时的处理
    # detail_backtest()

    # 示例一:传统的一一获取不同信号在三种指定时间段上面的表现结果
    # truly_backtest(file_path_list)

    # 示例二:使用预处理后的数据初略获取回测效果数据
    # statistic_data(file_path_list)

    # #示例三: 对原始数据进行预处理
    # for file_path in file_path_list:
    #     calculate_time_to_targets(file_path)

    # # 示例四：分析初略的回测结果数据
    # file_path_list = [
    #     # "temp/temp.csv",
    #     "kline_data/price_extremes/origin_data_1m_10000000_BTC-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_10.csv",
    #     "kline_data/price_extremes/origin_data_1m_10000000_ETH-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_10.csv",
    #     "kline_data/price_extremes/origin_data_1m_10000000_SOL-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_10.csv",
    #     "kline_data/price_extremes/origin_data_1m_10000000_TON-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_10.csv",
    #     "kline_data/price_reverse_extremes/origin_data_1m_10000000_BTC-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_10.csv",
    #     "kline_data/price_reverse_extremes/origin_data_1m_10000000_ETH-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_10.csv",
    #     "kline_data/price_reverse_extremes/origin_data_1m_10000000_SOL-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_10.csv",
    #     "kline_data/price_reverse_extremes/origin_data_1m_10000000_TON-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_10.csv",
    #     "kline_data/price_extremes/origin_data_1m_10000000_BTC-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_1.csv",
    #     "kline_data/price_extremes/origin_data_1m_10000000_ETH-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_1.csv",
    #     "kline_data/price_extremes/origin_data_1m_10000000_SOL-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_1.csv",
    #     "kline_data/price_extremes/origin_data_1m_10000000_TON-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_1.csv",
    #     "kline_data/price_reverse_extremes/origin_data_1m_10000000_BTC-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_1.csv",
    #     "kline_data/price_reverse_extremes/origin_data_1m_10000000_ETH-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_1.csv",
    #     "kline_data/price_reverse_extremes/origin_data_1m_10000000_SOL-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_1.csv",
    #     "kline_data/price_reverse_extremes/origin_data_1m_10000000_TON-USDT-SWAP/start_period_10_end_period_10000_step_period_10_min_20230208_max_20250103_time_len_1.csv"
    # ]
    # analyze_df_list = []
    # for file_path in file_path_list:
    #     analyze_df = analyze_data(file_path)
    #     file_path_split = file_path.split('_')
    #     key_name = f'{file_path_split[2]}_{file_path_split[6]}_{file_path_split[7]}'
    #     analyze_df['key_name'] = key_name
    #     analyze_df_list.append(analyze_df)
    # analyze_df = pd.concat(analyze_df_list, ignore_index=True)
    # analyze_df.to_csv('backtest_result/analyze_data.csv', index=False)


if __name__ == "__main__":
    example()
