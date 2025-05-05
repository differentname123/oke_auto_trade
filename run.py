import asyncio
import os
import traceback
import json
import datetime
import multiprocessing

import numpy as np
import pandas as pd
import websockets

from trade_common import LatestDataManager, place_order

# WebSocket 服务器地址
OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
# 定义需要操作的多个交易对
INSTRUMENT_LIST = ["SOL-USDT-SWAP", "BTC-USDT-SWAP", "ETH-USDT-SWAP", "TON-USDT-SWAP", "DOGE-USDT-SWAP", "XRP-USDT-SWAP"]

# 各交易对最小下单量映射
min_count_map = {"BTC-USDT-SWAP": 0.01, "ETH-USDT-SWAP": 0.01, "SOL-USDT-SWAP": 0.01, "TON-USDT-SWAP": 1, "DOGE-USDT-SWAP": 0.01, "XRP-USDT-SWAP": 0.01, "PEPE-USDT-SWAP": 0.1}

##############################################
# 单进程全局变量（每个进程只处理单一 INSTRUMENT）
##############################################
INSTRUMENT = None           # 当前处理的交易对，由 run_instrument() 传入设置
MIN_COUNT = None            # 当前交易对的下单最小数量
order_detail_map = {}       # 记录当前交易对的持仓订单
price = 0                   # 当前最新成交价格
price_list = []             # 已处理价格列表, 用于去重
current_minute = None       # 用于记录数据更新的分钟
kai_target_price_info_map = {}  # 开仓目标价格映射
pin_target_price_info_map = {}  # 平仓目标价格映射
kai_pin_map = {}            # 开仓信号与平仓信号映射
kai_reverse_map = {}        # 记录每个开仓信号是否反向
strategy_df = None          # 当前交易对的策略数据 DataFrame

##############################################
# 信号计算函数（与之前一致）
##############################################
def compute_threshold_direction(df, col_name):
    parts = col_name.split("_")
    signal_type = parts[0]
    direction = parts[-1]  # long 或 short

    if signal_type == "abs":
        period = int(parts[1])
        abs_value = float(parts[2]) / 100.0
        if direction == "long":
            threshold_series = (df["low"].shift(1).rolling(period).min() * (1 + abs_value)).round(4)
            op = ">"
        else:
            threshold_series = (df["high"].shift(1).rolling(period).max() * (1 - abs_value)).round(4)
            op = "<"
        direction_series = pd.Series([op] * len(df), index=df.index)
        return threshold_series, direction_series

    elif signal_type == "relate":
        period = int(parts[1])
        percent = float(parts[2]) / 100.0
        min_low = df["low"].shift(1).rolling(period).min()
        max_high = df["high"].shift(1).rolling(period).max()
        if direction == "long":
            threshold_series = (min_low + percent * (max_high - min_low)).round(4)
            op = ">"
        else:
            threshold_series = (max_high - percent * (max_high - min_low)).round(4)
            op = "<"
        direction_series = pd.Series([op] * len(df), index=df.index)
        return threshold_series, direction_series

    elif signal_type == "donchian":
        period = int(parts[1])
        if direction == "long":
            threshold_series = df["high"].shift(1).rolling(period).max().round(4)
            op = ">"
        else:
            threshold_series = df["low"].shift(1).rolling(period).min().round(4)
            op = "<"
        direction_series = pd.Series([op] * len(df), index=df.index)
        return threshold_series, direction_series

    elif signal_type == "boll":
        period = int(parts[1])
        std_multiplier = float(parts[2])
        ma = df["close"].rolling(window=period, min_periods=period).mean()
        std_dev = df["close"].rolling(window=period, min_periods=period).std()
        upper_band = (ma + std_multiplier * std_dev).round(4)
        lower_band = (ma - std_multiplier * std_dev).round(4)
        if direction == "long":
            threshold_series = lower_band
            op = ">"
            if df["close"].iloc[-1] > lower_band.iloc[-1]:
                op = None
        else:
            threshold_series = upper_band
            op = "<"
            if df["close"].iloc[-1] < upper_band.iloc[-1]:
                op = None
        direction_series = pd.Series([op] * len(df), index=df.index)
        return threshold_series, direction_series

    elif signal_type == "macross":
        fast_period = int(parts[1])
        slow_period = int(parts[2])
        fast_ma = df["close"].rolling(window=fast_period, min_periods=fast_period).mean().round(4)
        slow_ma = df["close"].rolling(window=slow_period, min_periods=slow_period).mean().round(4)
        threshold_series = slow_ma
        op = ">" if direction == "long" else "<"
        direction_series = pd.Series([op] * len(df), index=df.index)
        return threshold_series, direction_series

    elif signal_type == "rsi":
        period = int(parts[1])
        overbought = int(parts[2])
        oversold = int(parts[3])
        delta = df["close"].diff(1).astype(np.float32)
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        if direction == "long":
            threshold_series = pd.Series([oversold] * len(df), index=df.index)
            op = ">"
        else:
            threshold_series = pd.Series([overbought] * len(df), index=df.index)
            op = "<"
        direction_series = pd.Series([op] * len(df), index=df.index)
        return threshold_series, direction_series

    elif signal_type == "macd":
        fast_period, slow_period, signal_period = map(int, parts[1:4])
        fast_ema = df["close"].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df["close"].ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        threshold_series = signal_line.round(4)
        op = ">" if direction == "long" else "<"
        direction_series = pd.Series([op] * len(df), index=df.index)
        return threshold_series, direction_series

    elif signal_type == "cci":
        period = int(parts[1])
        tp = (df["high"] + df["low"] + df["close"]) / 3
        ma = tp.rolling(period).mean()
        md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        cci = (tp - ma) / (0.015 * md)
        if direction == "long":
            threshold_series = pd.Series([-100] * len(df), index=df.index)
            op = ">"
        else:
            threshold_series = pd.Series([100] * len(df), index=df.index)
            op = "<"
        direction_series = pd.Series([op] * len(df), index=df.index)
        return threshold_series, direction_series

    elif signal_type == "atr":
        period = int(parts[1])
        tr = pd.concat([
            df["high"] - df["low"],
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        atr_ma = atr.rolling(period).mean()
        threshold_series = atr_ma.round(4)
        op = ">" if direction == "long" else "<"
        direction_series = pd.Series([op] * len(df), index=df.index)
        return threshold_series, direction_series

    else:
        raise ValueError(f"未知或不支持的信号类型: {signal_type}")

def update_price_map(strategy_df, df, target_column='kai_column'):
    """
    根据策略 DataFrame 中的信号，对最新行情数据生成目标价格映射
    """
    kai_column_list = strategy_df[target_column].unique().tolist()
    target_price_info_map = {}
    for kai_column in kai_column_list:
        try:
            threshold_price_series, direction = compute_threshold_direction(df, kai_column)
        except Exception as e:
            print(f"❌ 计算 {kai_column} 时出现错误：", e)
            continue
        threshold_price = threshold_price_series.iloc[-1]
        # 判断阈值是否有效
        if pd.isna(threshold_price) or threshold_price == 0:
            print(f"❌ {kai_column} 的阈值计算失败，跳过该信号")
            continue
        if direction.iloc[-1] != None:
            target_price_info_map[kai_column] = (threshold_price, direction.iloc[-1])
    return target_price_info_map

##############################################
# 异步任务：数据更新
##############################################
async def fetch_new_data(max_period):
    global kai_target_price_info_map, pin_target_price_info_map, current_minute, price, price_list, strategy_df, INSTRUMENT
    newest_data = LatestDataManager(max_period, INSTRUMENT)
    max_attempts = 200
    previous_timestamp = None
    kai_column_list = strategy_df['kai_column'].unique().tolist()
    result = {
        "instrument": INSTRUMENT,
        "total_strategy_count": len(strategy_df),  # 所有策略个数
        "signals": {}  # 以信号为 key 存储信息
    }
    while True:
        try:
            now = datetime.datetime.now()
            if current_minute is None or now.minute != current_minute:
                print(f"🕐 {now.strftime('%H:%M')} {INSTRUMENT} 触发数据更新...")
                attempt = 0
                while attempt < max_attempts:
                    df = newest_data.get_newest_data()
                    latest_timestamp = df.iloc[-1]['timestamp'] if not df.empty else None
                    if previous_timestamp is None or latest_timestamp != previous_timestamp:
                        print(f"✅ {INSTRUMENT} 数据已更新, 最新 timestamp: {latest_timestamp} 实时最新价格: {price}")
                        price_list.clear()
                        kai_target_price_info_map = update_price_map(strategy_df, df, target_column='kai_column')
                        pin_target_price_info_map = update_price_map(strategy_df, df, target_column='pin_column')

                        for kai in kai_column_list:
                            kai_value = kai_target_price_info_map.get(kai)
                            pin = kai_pin_map.get(kai)
                            pin_value = pin_target_price_info_map.get(pin)

                            # 使用 kai 作为 key 存储对应信号的数据
                            result["signals"][kai] = {
                                "open_target_price": kai_value,  # 开仓目标价格
                                "close_signal": pin,  # 平仓信号
                                "close_target_price": pin_value  # 平仓目标价格
                            }

                        print(f"{INSTRUMENT} 开仓信号个数 {len(kai_target_price_info_map)} 平仓信号个数{len(pin_target_price_info_map)}  详细结果：{result}")
                        previous_timestamp = latest_timestamp
                        current_minute = now.minute
                        break
                    else:
                        print(f"⚠️ {INSTRUMENT} 数据未变化, 尝试重新获取 ({attempt + 1}/{max_attempts})")
                        attempt += 1
                if attempt == max_attempts:
                    print(f"❌ {INSTRUMENT} 多次尝试数据仍未更新，跳过本轮更新")
            await asyncio.sleep(1)
        except Exception as e:
            pin_target_price_info_map = {}
            kai_target_price_info_map = {}
            traceback.print_exc()

##############################################
# 异步任务：WebSocket 连接与监听
##############################################
async def subscribe_channel(ws):
    subscribe_msg = {
        "op": "subscribe",
        "args": [{"channel": "trades", "instId": INSTRUMENT}]
    }
    await ws.send(json.dumps(subscribe_msg))
    print(f"📡 {INSTRUMENT} 已订阅实时数据")

async def websocket_listener():
    global price, price_list
    while True:
        try:
            async with websockets.connect(OKX_WS_URL) as ws:
                print(f"✅ {INSTRUMENT} 连接到 OKX WebSocket")
                await subscribe_channel(ws)
                while True:
                    try:
                        response = await ws.recv()
                        data = json.loads(response)
                        if "data" not in data:
                            continue
                        for trade in data["data"]:
                            price_val = float(trade["px"])
                            # 去重处理
                            if price_val in price_list:
                                continue
                            price_list.append(price_val)
                            price = price_val
                            process_open_orders(price_val)
                            process_close_orders(price_val)
                    except websockets.exceptions.ConnectionClosed:
                        print(f"🔴 {INSTRUMENT} WebSocket 连接断开，重连中...")
                        break
        except Exception as e:
            traceback.print_exc()

##############################################
# 订单处理逻辑：开仓和平仓（单交易对版本）
##############################################
def process_open_orders(price_val):
    global kai_target_price_info_map, order_detail_map, current_minute, kai_pin_map, kai_reverse_map, INSTRUMENT, MIN_COUNT
    for key, target_info in kai_target_price_info_map.items():
        if target_info is not None:
            threshold_price, comp = target_info
            is_reverse = kai_reverse_map.get(key, False)
            side = 'buy' if 'long' in key else 'sell'
            if is_reverse:
                side = 'buy' if side == 'sell' else 'sell'
            pin_side = 'sell' if side == 'buy' else 'buy'
            if comp == '>' and price_val >= threshold_price and key not in order_detail_map:
                result = place_order(INSTRUMENT, side, MIN_COUNT)
                if result:
                    order_detail_map[key] = {
                        'price': price_val,
                        'side': side,
                        'pin_side': pin_side,
                        'time': current_minute,
                        'size': MIN_COUNT
                    }
                    print(f"开仓成功 {key} for {INSTRUMENT} 成交, 价格: {price_val}, 时间: {datetime.datetime.now()}")
                    save_order_detail_map()
            elif comp == '<' and price_val <= threshold_price and key not in order_detail_map:
                result = place_order(INSTRUMENT, side, MIN_COUNT)
                if result:
                    order_detail_map[key] = {
                        'price': price_val,
                        'side': side,
                        'pin_side': pin_side,
                        'time': current_minute,
                        'size': MIN_COUNT
                    }
                    print(f"开仓成功 {key} for {INSTRUMENT} 成交, 价格: {price_val}, 时间: {datetime.datetime.now()}")
                    save_order_detail_map()

def process_close_orders(price_val):
    global order_detail_map, current_minute, pin_target_price_info_map, kai_pin_map, INSTRUMENT
    keys_to_remove = []
    for kai_key, order in list(order_detail_map.items()):
        if current_minute == order['time']:
            continue
        pin_key = kai_pin_map.get(kai_key)
        if not pin_key:
            continue
        kai_price = order['price']
        if pin_key in pin_target_price_info_map:
            target_info = pin_target_price_info_map[pin_key]
            if target_info is not None:
                threshold_price, comp = target_info
                if comp == '>' and price_val > threshold_price:
                    result = place_order(INSTRUMENT, order['pin_side'], order['size'], trade_action="close")
                    if result:
                        keys_to_remove.append(kai_key)
                        print(f"【平仓】 {pin_key} for {INSTRUMENT} {order['pin_side']} 成交, 价格: {price_val}, 开仓价格: {kai_price}, 时间: {datetime.datetime.now()}")
                    else:
                        print(f"❌ {pin_key} for {INSTRUMENT} 平仓失败, 价格: {price_val}, 开仓价格: {kai_price}, 时间: {datetime.datetime.now()}")
                elif comp == '<' and price_val < threshold_price:
                    result = place_order(INSTRUMENT, order['pin_side'], order['size'], trade_action="close")
                    if result:
                        keys_to_remove.append(kai_key)
                        print(f"【平仓】 {pin_key} for {INSTRUMENT} {order['pin_side']} 成交, 价格: {price_val}, 开仓价格: {kai_price}, 时间: {datetime.datetime.now()}")
                    else:
                        print(f"❌ {pin_key} for {INSTRUMENT} 平仓失败, 价格: {price_val}, 开仓价格: {kai_price}, 时间: {datetime.datetime.now()}")
    if keys_to_remove:
        for k in keys_to_remove:
            order_detail_map.pop(k, None)
        save_order_detail_map()

##############################################
# 订单持久化相关函数（单交易对）
##############################################
def save_order_detail_map():
    global order_detail_map, INSTRUMENT
    try:
        if not os.path.exists("temp"):
            os.makedirs("temp")
        file_path = f"temp/order_detail_map_{INSTRUMENT}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(order_detail_map, f)
    except Exception as e:
        traceback.print_exc()

def load_order_detail_map():
    global order_detail_map, INSTRUMENT
    file_path = f"temp/order_detail_map_{INSTRUMENT}.json"
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                order_detail_map.update(json.load(f))
            print(f"✅ {INSTRUMENT} 已加载之前的订单信息")
        except Exception as e:
            print(f"❌ {INSTRUMENT} 加载订单信息失败:", e)
    else:
        order_detail_map.clear()

##############################################
# 主异步函数：加载策略、初始化、启动任务（单交易对）
##############################################
async def main_instrument():
    global INSTRUMENT, MIN_COUNT, strategy_df, kai_pin_map, kai_reverse_map

    # 加载历史订单记录
    load_order_detail_map()

    # 加载策略数据（例如 parquet 文件）
    inst_id = INSTRUMENT.split('-')[0]
    all_df = []
    exclude_str = ['macross', 'rsi', 'macd', 'cci', 'atr']
    for is_reverse in [True, False]:
        # file_path = f'temp/final_good_{inst_id}_{is_reverse}_filter_all.parquet'
        file_path = f'temp/corr/final_good_{inst_id}_{is_reverse}_filter_all.parquet_origin_good_weekly_net_profit_detail.parquet'

        if os.path.exists(file_path):
            final_good_df = pd.read_parquet(file_path)
            for exclude in exclude_str:
                final_good_df = final_good_df[~final_good_df['kai_column'].str.contains(exclude)]
                final_good_df = final_good_df[~final_good_df['pin_column'].str.contains(exclude)]
            final_good_df = final_good_df.sort_values(by='score_final', ascending=False).head(10)
            all_df.append(final_good_df)
            print(f'{INSTRUMENT} final_good_df shape: {final_good_df.shape[0]} 来自 {file_path}')
    if all_df:
        strategy_df_local = pd.concat(all_df)
        # 将全局策略 DataFrame 指向它
        global strategy_df
        strategy_df = strategy_df_local
    else:
        print(f"❌ {INSTRUMENT} 策略数据不存在!")
        return

    # 构造 kai_pin_map 与 kai_reverse_map
    period_list = []
    for idx, row in strategy_df.iterrows():
        is_reverse = row.get('is_reverse', False)
        kai = row['kai_column']
        pin = row['pin_column']
        kai_pin_map[kai] = pin
        kai_reverse_map[kai] = is_reverse
        period_list.append(int(kai.split('_')[1]))
        period_list.append(int(pin.split('_')[1]))
    max_period = int(np.ceil(max(period_list) / 100) * 100) if period_list else 100

    # 设置当前交易对的最小下单量
    global MIN_COUNT
    MIN_COUNT = min_count_map.get(INSTRUMENT, 0)

    # 同时启动数据更新任务和 WebSocket 监听任务
    await asyncio.gather(
        fetch_new_data(max_period),
        websocket_listener()
    )

##############################################
# 进程入口：每个进程处理一个交易对
##############################################
def run_instrument(instrument):
    global INSTRUMENT
    INSTRUMENT = instrument
    print(f"【进程启动】开始处理 {INSTRUMENT}")
    asyncio.run(main_instrument())

##############################################
# 主入口：多进程启动每个交易对
##############################################
if __name__ == '__main__':
    processes = []
    for instr in INSTRUMENT_LIST:
        p = multiprocessing.Process(target=run_instrument, args=(instr,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()