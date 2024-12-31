from datetime import datetime

import pandas as pd

from get_feature_op import generate_price_extremes_signals


def gen_buy_sell_signal(data_df, profit=1/100, period=10):
    """
    为data生成相应的买卖信号，并生成相应的buy_price, sell_price
    :param data:
    :return:
    """
    count = 0.001
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
    signal_df['count'] = count
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
    high = row['high']
    low = row['low']
    close = row['close']

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
        'timestamp': row['timestamp'],
        'long_sz': long_sz,
        'short_sz': short_sz,
        'long_avg_price': long_avg_price,
        'short_avg_price': short_avg_price,
        'close_available_funds': close_available_funds,
        'high_available_funds': high_available_funds,
        'low_available_funds': low_available_funds,
        'final_total_money_if_close': final_total_money_if_close

    }


def calculate_time_diff_minutes(time_str1, time_str2):
    """
    计算两个字符串格式时间之间相差的分钟数。

    Args:
        time_str1: 第一个时间字符串，格式为 'YYYY-MM-DD HH:MM:SS'。
        time_str2: 第二个时间字符串，格式为 'YYYY-MM-DD HH:MM:SS'。

    Returns:
        两个时间之间相差的分钟数（浮点数）。
        如果时间字符串格式错误，则返回 None。
    """
    try:
        time1 = datetime.strptime(time_str1, '%Y-%m-%d %H:%M:%S')
        time2 = datetime.strptime(time_str2, '%Y-%m-%d %H:%M:%S')
        time_diff = time1 - time2
        return abs(time_diff.total_seconds() / 60)
    except ValueError:
        print("错误：时间字符串格式不正确，请使用 'YYYY-MM-DD HH:MM:SS' 格式。")
        return None

def deal_pending_order(pending_order_list, row, position_info, lever, total_money, max_time_diff=2):
    """
    处理委托单
    :param pending_order_list: 委托单列表
    :param row: 当前市场数据，包含high、low、timestamp
    :param position_info: 持仓信息，包含可用资金和持仓均价等
    :param lever: 杠杆倍数
    :param total_money: 总资金
    :param max_time_diff: 最大时间差，单位秒，默认为5
    :return: 更新的历史订单列表和总资金
    """
    high = row['high']
    low = row['low']
    close_available_funds = position_info['close_available_funds']
    timestamp = row['timestamp']
    history_order_list = []
    fee = 0.0007  # 手续费

    for order in pending_order_list:
        if order['side'] == 'kai':  # 开仓
            # 计算时间差
            time_diff = calculate_time_diff_minutes(timestamp, order['timestamp'])
            if time_diff < max_time_diff:
                if order['type'] == 'long':  # 开多仓
                    if order['buy_price'] > low:  # 买入价格高于最低价
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
            if order['type'] == 'long':  # 平多仓
                if order['sell_price'] < high:
                    order['side'] = 'done'
                    order['ping_time'] = timestamp
                    # 计算收益并更新总资金
                    profit = order['count'] * (order['sell_price'] - order['buy_price'] - fee * order['sell_price'])
                    total_money += profit
            if order['type'] == 'short':  # 平空仓
                if order['sell_price'] > low:
                    order['side'] = 'done'
                    order['ping_time'] = timestamp
                    # 计算收益并更新总资金
                    profit = order['count'] * (order['buy_price'] - order['sell_price'] - fee * order['sell_price'])
                    total_money += profit

    # 删除已经完成的订单，移动到history_order_list
    history_order_list.extend([order for order in pending_order_list if order['side'] == 'done'])
    pending_order_list = [order for order in pending_order_list if order['side'] != 'done']
    return pending_order_list, history_order_list, total_money


import pandas as pd
import multiprocessing as mp
from tqdm import tqdm  # 用于显示进度条

def create_order(order_type, row, lever):
    """创建订单信息"""
    return {
        'buy_price': row['buy_price'],
        'count': row['count'],
        'timestamp': row['timestamp'],
        'sell_price': row['sell_price'],
        'type': order_type,
        'lever': lever,
        'side': 'kai'
    }

def process_signals(signal_df, lever, total_money, init_money):
    """处理信号生成的订单并计算收益"""
    pending_order_list = []
    all_history_order_list = []
    position_info_list = []

    for _, row in signal_df.iterrows():
        # 分析持仓信息
        position_info = analysis_position(pending_order_list, row, total_money, lever)
        position_info_list.append(position_info)

        # 处理委托单
        pending_order_list, history_order_list, total_money = deal_pending_order(
            pending_order_list, row, position_info, lever, total_money
        )
        all_history_order_list.extend(history_order_list)

        # 根据信号生成新订单
        if row['Buy'] == 1:
            pending_order_list.append(create_order('long', row, lever))
        elif row['Sell'] == 1:
            pending_order_list.append(create_order('short', row, lever))

    # 计算最终结果
    position_info_df = pd.DataFrame(position_info_list)
    final_total_money_if_close = position_info_df['final_total_money_if_close'].iloc[-1]
    min_available_funds = min(
        position_info_df['close_available_funds'].min(),
        position_info_df['high_available_funds'].min(),
        position_info_df['low_available_funds'].min()
    )
    max_cost_money = init_money - min_available_funds
    final_profit = final_total_money_if_close - init_money
    profit_ratio = final_profit / max_cost_money

    last_data = position_info_df.iloc[-1].copy()  # 避免视图警告
    last_data = last_data.to_dict()  # 转换为字典
    last_data.update({
        'final_total_money_if_close': final_total_money_if_close,
        'final_profit': final_profit,
        'max_cost_money': max_cost_money,
        'profit_ratio': profit_ratio
    })

    return last_data

def calculate_combination(args):
    """多进程计算单个组合的回测结果"""
    profit, period, data_df, lever, init_money = args
    signal_df = gen_buy_sell_signal(data_df, profit=profit, period=period)
    last_data = process_signals(signal_df, lever, init_money, init_money)
    last_data.update({'profit': profit, 'period': period})
    return last_data

def example():
    file_path = 'kline_data/max_1m_data.csv'
    data_df = pd.read_csv(file_path)[-100000:-1000]  # 只取最近1000条数据
    data_len = len(data_df)
    profit_list = [x / 1000 for x in range(1, 20)]
    period_list = list(range(10, 2000, 10))
    lever = 100
    init_money = 10000000

    # 准备参数组合
    combinations = [(profit, period, data_df, lever, init_money) for profit in profit_list for period in period_list]
    print(f"共有 {len(combinations)} 个组合，开始计算...")

    # 使用多进程计算
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(calculate_combination, combinations), total=len(combinations)))

    # 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_csv(f'result_{data_len}.csv', index=False)
    print("结果已保存到 f'result_{data_len}.csv'")

if __name__ == "__main__":
    data_df = pd.read_csv('result_9000.csv')
    example()
