import asyncio
import os

import numpy as np
import pandas as pd
import websockets
import json
import datetime

from trade_common import LatestDataManager, place_order

# WebSocket 服务器地址
# OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"

OKX_WS_URL = "wss://wspap.okx.com:8443/ws/v5/public"

# 订阅的交易对
INSTRUMENT = "BTC-USDT-SWAP"

# 初始化价格映射
kai_high_price_map = {}
kai_low_price_map = {}

pin_high_price_map = {}
pin_low_price_map = {}

order_detail_map = {}

kai_pin_map = {}

# 记录当前分钟
current_minute = None

def update_price_map(strategy_df, df, target_column='kai_column'):
    kai_column_list = strategy_df[target_column].unique().tolist()
    high_price_map = {}
    low_price_map = {}
    for kai_column in kai_column_list:
        period = int(kai_column.split('_')[0])
        price_side = kai_column.split('_')[1]
        if price_side == 'high':
            # 获取df最近period个数据的最高价
            max_price = df['high'].tail(period).max()
            high_price_map[kai_column] = max_price
        else:
            # 获取df最近period个数据的最低价
            min_price = df['low'].tail(period).min()
            low_price_map[kai_column] = min_price
    return high_price_map, low_price_map

async def fetch_new_data(strategy_df):
    """ 每分钟获取最新数据并更新 high_price_map 和 low_price_map """
    global kai_high_price_map, kai_low_price_map,pin_high_price_map, pin_low_price_map, current_minute, order_detail_map
    newest_data = LatestDataManager(100, INSTRUMENT)

    while True:
        now = datetime.datetime.now()
        if current_minute is None or now.minute != current_minute:
            print(f"🕐 {now.strftime('%H:%M')} 触发数据更新...")
            current_minute = now.minute  # 更新当前分钟
            df = newest_data.get_newest_data()  # 获取最新数据
            kai_high_price_map, kai_low_price_map = update_price_map(strategy_df, df)  # 更新映射
            pin_high_price_map, pin_low_price_map = update_price_map(strategy_df, df, target_column='pin_column')
            print(f"📈 更新开仓价格映射：{kai_high_price_map} 📉 更新开仓价格映射：{kai_low_price_map}")

        await asyncio.sleep(1)  # 每秒检查一次当前分钟

async def websocket_listener(kai_pin_map):
    default_size = 10
    """ 监听 WebSocket 实时数据，并对比 high_price_map 和 low_price_map """
    global kai_high_price_map, kai_low_price_map, pin_high_price_map, pin_low_price_map, order_detail_map
    async with websockets.connect(OKX_WS_URL) as ws:
        print("✅ 已连接到 OKX WebSocket")

        # 订阅 BTC-USDT-SWAP 的最新成交数据
        subscribe_msg = {
            "op": "subscribe",
            "args": [{"channel": "trades", "instId": INSTRUMENT}]
        }
        await ws.send(json.dumps(subscribe_msg))
        print(f"📡 已订阅 {INSTRUMENT} 实时成交数据")
        pre_price = 0
        # 持续监听 WebSocket 消息
        while True:
            try:
                response = await ws.recv()
                data = json.loads(response)

                if "data" in data:
                    for trade in data["data"]:
                        price = float(trade["px"])  # 最新成交价格
                        if price != pre_price:
                            for key, high_price in kai_high_price_map.items():
                                if price >= high_price:
                                    # 要求key 不在order_detail_map中，避免重复下单
                                    if key not in order_detail_map:
                                        result = place_order(INSTRUMENT, "buy", default_size)  # 以最优价格开多 0.01 BTC
                                        if result:
                                            order_detail_map[key] = {'price': price, 'side': 'buy', 'pin_side':'sell', 'time': datetime.datetime.now(), 'size': default_size}
                                            print(f"📈 开多仓 {key} 成交，价格：{price}，时间：{datetime.datetime.now()}")


                            for key, low_price in kai_low_price_map.items():
                                if price <= low_price:
                                    if key not in order_detail_map:
                                        result = place_order(INSTRUMENT, "sell", default_size)
                                        if result:
                                            order_detail_map[key] = {'price': price, 'side': 'sell', 'pin_side':'buy', 'time': datetime.datetime.now(), 'size': default_size}
                                            print(f"📉 开空仓 {key} 成交，价格：{price}，时间：{datetime.datetime.now()}")


                            # 如果order_detail_map中有数据，说明有订单成交
                            if order_detail_map:
                                keys_to_remove = []  # 存储需要删除的键，避免循环中修改字典

                                for kai_key, order_detail in list(order_detail_map.items()):  # 用 list() 避免字典修改问题
                                    pin_key = kai_pin_map.get(kai_key)  # 避免 KeyError
                                    if not pin_key:
                                        continue  # 如果 key 不存在，则跳过

                                    # 检查是否需要平仓
                                    if pin_key in pin_high_price_map:
                                        pin_price = pin_high_price_map[pin_key]
                                        if price >= pin_price:
                                            result = place_order(INSTRUMENT, order_detail['pin_side'],
                                                                 order_detail['size'], trade_action="close")
                                            if result:
                                                keys_to_remove.append(kai_key)  # 先记录 key，稍后删除
                                                print(
                                                    f"📈 【平空仓】 {order_detail['pin_side']} 成交，价格：{price}，时间：{datetime.datetime.now()}")

                                    elif pin_key in pin_low_price_map:
                                        pin_price = pin_low_price_map[pin_key]
                                        if price <= pin_price:
                                            result = place_order(INSTRUMENT, order_detail['pin_side'],
                                                                 order_detail['size'], trade_action="close")
                                            if result:
                                                keys_to_remove.append(kai_key)  # 先记录 key，稍后删除
                                                print(
                                                    f"📉 【平多仓】 {order_detail['pin_side']} 成交，价格：{price}，时间：{datetime.datetime.now()}")

                                # 在循环结束后删除已平仓的订单
                                for key in keys_to_remove:
                                    order_detail_map.pop(key, None)  # 使用 pop(key, None) 避免 KeyError
                        pre_price = price

            except websockets.exceptions.ConnectionClosed:
                print("🔴 WebSocket 连接断开，正在重连...")
                break


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
    df = df[df[sort_key] > 0]
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

def choose_good_strategy():
    # df = pd.read_csv('temp/temp.csv')
    # count_L()
    # 找到temp下面所有包含False的文件
    file_list = os.listdir('temp')
    file_list = [file for file in file_list if 'True' in file and INSTRUMENT in file and '0' in file and '1m' in file and 'with' not in file]
    df_list = []
    df_map = {}
    for file in file_list:
        file_key = file.split('_')[4]
        df = pd.read_csv(f'temp/{file}')

        df['filename'] = file.split('_')[5]
        # df = df[(df['avg_profit_rate'] > 0)]
        if file_key not in df_map:
            df_map[file_key] = []
        df['score'] = df['avg_profit_rate']
        df['score1'] = df['avg_profit_rate'] / (df['hold_time_mean'] + 20) * 1000
        df['score2'] = df['avg_profit_rate'] / (
                    df['hold_time_mean'] + 20) * 1000 * (df['trade_rate'] + 0.001)
        df['score3'] = df['avg_profit_rate'] * (df['trade_rate'] + 0.0001)
        df['score4'] = (df['trade_rate'] + 0.0001) / df['loss_rate']
        df_map[file_key].append(df)
    for key in df_map:
        df = pd.concat(df_map[key])
        df_list.append(df)

    temp = pd.merge(df_list[0], df_list[1], on=['kai_side', 'kai_column', 'pin_column'], how='inner')
    # 需要计算的字段前缀
    fields = ['avg_profit_rate', 'net_profit_rate']

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

    # temp = temp[(temp['avg_profit_rate_min'] > 0)]
    # temp.to_csv('temp/temp.csv', index=False)
    return temp

async def main():
    # good_strategy_df1 = pd.read_csv('temp/temp.csv')
    good_strategy_df = choose_good_strategy()
    # 筛选出kai_side为long的数据
    long_good_strategy_df = good_strategy_df[good_strategy_df['kai_side'] == 'long']
    short_good_strategy_df = good_strategy_df[good_strategy_df['kai_side'] == 'short']

    # 将long_good_strategy_df按照net_profit_rate_mult降序排列
    long_good_select_df = select_best_rows_in_ranges(long_good_strategy_df, range_size=100,
                                                     sort_key='avg_profit_rate_mult', range_key='kai_count_x')
    # long_good_select_df['kai_column'] = '1_high_long'
    # long_good_select_df['pin_column'] = '1_low_short'
    short_good_select_df = select_best_rows_in_ranges(short_good_strategy_df, range_size=100,
                                                      sort_key='avg_profit_rate_mult', range_key='kai_count_x')
    # short_good_select_df['kai_column'] = '1_low_short'
    # short_good_select_df['pin_column'] = '1_high_long'
    final_good_df = pd.concat([long_good_select_df, short_good_select_df])
    final_good_df.to_csv('temp/final_good.csv', index=False)
    print(f'final_good_df shape: {final_good_df.shape[0]}')
    # 遍历final_good_df，将kai_column和pin_column一一对应
    for index, row in final_good_df.iterrows():
        kai_column = row['kai_column']
        pin_column = row['pin_column']
        kai_pin_map[kai_column] = pin_column

    """ 启动 WebSocket 监听和定时任务 """
    await asyncio.gather(
        fetch_new_data(final_good_df),  # 定时更新数据
        websocket_listener(kai_pin_map)  # 监听实时数据
    )

# 运行 asyncio 事件循环
asyncio.run(main())