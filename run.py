import asyncio
import os
import traceback
import json
import datetime

import numpy as np
import pandas as pd
import websockets

from trade_common import LatestDataManager, place_order

# WebSocket 服务器地址
OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
# OKX_WS_URL = "wss://wspap.okx.com:8443/ws/v5/public"

# 订阅的交易对
INSTRUMENT = "SOL-USDT-SWAP"
min_count_map = {"BTC-USDT-SWAP": 0.01, "ETH-USDT-SWAP": 0.01, "SOL-USDT-SWAP": 0.1, "TON-USDT-SWAP": 1}

# 初始化价格映射
kai_target_price_info_map = {}
pin_target_price_info_map = {}

# 订单详情映射，全局变量，保存持仓订单信息
order_detail_map = {}
order_file_path = f"temp/order_detail_map_{INSTRUMENT}.json"
price = 0
price_list = []

kai_pin_map = {}
kai_reverse_map = {}

# 记录当前分钟
current_minute = None

# -----------------------------
# 新增：订单持久化处理函数
# -----------------------------
def save_order_detail_map():
    """每次order_detail_map变化时保存到temp目录下，文件名为order_detail_map.json"""
    try:
        # 确保目录存在
        if not os.path.exists("temp"):
            os.makedirs("temp")
        with open(order_file_path, "w", encoding="utf-8") as f:
            json.dump(order_detail_map, f)
        # print("✅ 订单状态已保存到temp/order_detail_map.json")
    except Exception as e:
        traceback.print_exc()


def load_order_detail_map():
    """启动时从temp目录下加载之前保存的订单信息"""
    global order_detail_map
    file_path = order_file_path
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                order_detail_map = json.load(f)
            print("✅ 已加载之前的订单信息")
        except Exception as e:
            print("❌ 加载订单信息失败:", e)

# -----------------------------
# 以下为原有函数
# -----------------------------
def get_next_threshold_abs(df, col_name):
    parts = col_name.split('_')
    direction = parts[-1]
    period = int(parts[1])
    abs_value = float(parts[2])

    if len(df) < period + 1:
        return None  # 数据不足，无法计算

    last_high = df['high'].iloc[-1]  # 当前 K 线的最高价
    last_low = df['low'].iloc[-1]    # 当前 K 线的最低价

    if direction == "long":
        # 计算过去 period 根 K 线的最低价（不包括当前 K 线）
        min_low_prev = df['low'].iloc[-(period):].min()
        threshold_price = round(min_low_prev * (1 + abs_value / 100), 4)

        # 确保当前 K 线有可能触发信号
        if last_high < threshold_price:
            return threshold_price, ">"
        else:
            return None  # 价格未突破，不会触发信号

    elif direction == "short":
        # 计算过去 period 根 K 线的最高价（不包括当前 K 线）
        max_high_prev = df['high'].iloc[-(period):].max()
        threshold_price = round(max_high_prev * (1 - abs_value / 100), 4)

        # 确保当前 K 线有可能触发信号
        if last_low > threshold_price:
            return threshold_price, "<"
        else:
            return None  # 价格未跌破，不会触发信号

    return None  # 方向无效


def get_next_threshold_relate(df, col_name):
    parts = col_name.split('_')
    direction = parts[-1]
    period = int(parts[1])
    abs_value = float(parts[2])

    last_high = df['high'].iloc[-1]  # 当前 K 线的最高价
    last_low = df['low'].iloc[-1]    # 当前 K 线的最低价

    # 检查数据是否足够（由于 shift(1) 后会丢失最新数据，需至少 period+1 行）
    if df.shape[0] < period + 1:
        return None

    if direction == "long":
        # 取前一周期数据（所有计算基于 shift(1)）
        min_low = df['low'].rolling(window=period).min().iloc[-1]
        max_high = df['high'].rolling(window=period).max().iloc[-1]
        target_price = round(min_low + abs_value / 100 * (max_high - min_low), 4)
        comp = ">"  # 下一周期若 high > target_price 则突破成功
        if last_high < target_price:
            return target_price, comp
    else:
        max_high = df['high'].shift(1).rolling(window=period).max().iloc[-1]
        min_low = df['low'].shift(1).rolling(window=period).min().iloc[-1]
        target_price = round(max_high - abs_value / 100 * (max_high - min_low), 4)
        comp = "<"  # 下一周期若 low < target_price 则突破成功
        if last_low > target_price:
            return target_price, comp
    return None


def get_next_threshold_rsi(df, col_name):
    parts = col_name.split('_')
    direction = parts[-1]
    period = int(parts[1])
    overbought = int(parts[2])
    oversold = int(parts[3])

    if len(df) < period + 1:
        return None

    # 计算价格变化
    delta = df['close'].diff(1).astype(np.float64)

    # 获取最近 `period` 个数据
    diffs = delta.iloc[-period:]

    if diffs.isnull().any():
        return None

    # 计算涨跌幅
    gains = diffs.clip(lower=0)
    losses = -diffs.clip(upper=0)

    S_gain = gains.sum()
    S_loss = losses.sum()

    # 如果 S_loss 为 0，避免除零错误
    if S_loss == 0:
        rs = float('inf')
    else:
        rs = S_gain / S_loss

    rsi = 100 - (100 / (1 + rs))

    # 获取最后的 RSI 值
    df.loc[df.index[-1], 'rsi'] = rsi
    last_rsi = df['rsi'].iloc[-1]

    # 获取最新收盘价
    C_last = df['close'].iloc[-1]

    # 计算门槛价格
    d0 = diffs.iloc[0]
    g0 = max(d0, 0)
    l0 = -min(d0, 0)

    if direction == "short":
        OS = oversold
        threshold_price = C_last + (OS / (100 - OS)) * (S_loss - l0) - (S_gain - g0)
        if last_rsi < OS:
            return threshold_price, ">"
    elif direction == "long":
        OB = overbought
        threshold_price = C_last - ((S_gain - g0) * ((100 - OB) / OB) - (S_loss - l0))
        if last_rsi > OB:
            return threshold_price, "<"

    return None


def gen_signal_price(df, col_name):
    """
    生成信号价格
    :param df:
    :param column:
    :return:
    """
    parts = col_name.split('_')
    signal_type = parts[0]
    if signal_type == "rsi":
        target_info = get_next_threshold_rsi(df, col_name)
    elif signal_type == "abs":
        target_info = get_next_threshold_abs(df, col_name)
    elif signal_type == "relate":
        target_info = get_next_threshold_relate(df, col_name)
    else:
        target_info = None
        print(f"❌ 未知信号类型：{col_name}")
    return target_info


def forecast_signal_price_range(df, col_name):
    parts = col_name.split('_')
    direction = parts[-1]
    period = int(parts[1])
    overbought = int(parts[2])
    oversold = int(parts[3])

    if len(df) < period + 1:
        raise None

    # 计算价格变化
    delta = df['close'].diff(1).astype(np.float64)
    # 提取最新 period 个差值（正好构成当前滚动窗口）
    diffs = delta.iloc[-period:]

    if diffs.isnull().any():
        raise ValueError("数据不足，无法计算完整的滚动窗口 RSI")

    # 分别计算每个差值的正值（涨幅）与负值（跌幅，正数）贡献
    gains = diffs.clip(lower=0)
    losses = -diffs.clip(upper=0)

    S_gain = gains.sum()
    S_loss = losses.sum()

    # 当前窗口中最早的那笔差值，在更新时会被剔除
    d0 = diffs.iloc[0]
    g0 = max(d0, 0)  # 若 d0 为正，其贡献；否则为 0
    l0 = -min(d0, 0)  # 若 d0 为负，其转化为正数的贡献；否则为 0

    C_last = df['close'].iloc[-1]

    if direction == "short":
        # 对 short 信号，新差值应为正 => X - C_last > 0
        OS = oversold
        # 根据公式：
        #   ( (S_gain - g0) + (X - C_last) ) / (S_loss - l0) = OS/(100-OS)
        # 解得：
        threshold_price = C_last + (OS / (100 - OS)) * (S_loss - l0) - (S_gain - g0)
        comp = ">="
        return threshold_price, comp
    elif direction == "long":
        # 对 long 信号，新差值应为负 => X - C_last < 0
        OB = overbought
        # 根据公式：
        #   (S_gain - g0) / ((S_loss - l0) + (C_last - X)) = OB/(100-OB)
        # 解得：
        threshold_price = C_last - ((S_gain - g0) * ((100 - OB) / OB) - (S_loss - l0))
        comp = "<="
        return threshold_price, comp
    else:
        return None


def compute_threshold_direction(df, col_name):
    """
    根据历史行情数据(df)和指定信号名称(col_name)，生成每一行的触发阈值及对应的比较方向。

    支持的信号类型：
        abs, relate, donchian, boll, macross, rsi, macd, cci, atr。

    参数：
        df: 包含行情数据的 DataFrame，要求包含相应的列（例如 "high", "low", "close"）。
        col_name: 信号名称，格式约定如 "abs_20_10_long"、"relate_20_10_short" 等，
                  其中最后一项必须为 "long" 或 "short"。

    返回：
        threshold_series: 一个 Series，记录每一行计算出的触发阈值。
        direction_series: 一个 Series，记录每一行的比较方向，值为 ">" 或 "<"。

    如果遇到无法转换的信号类型，则抛出 ValueError。
    """
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
        else:
            threshold_series = upper_band
            op = "<"
        direction_series = pd.Series([op] * len(df), index=df.index)
        return threshold_series, direction_series

    elif signal_type == "macross":
        fast_period = int(parts[1])
        slow_period = int(parts[2])
        fast_ma = df["close"].rolling(window=fast_period, min_periods=fast_period).mean().round(4)
        slow_ma = df["close"].rolling(window=slow_period, min_periods=slow_period).mean().round(4)
        # 用长期均线作为参考阈值
        if direction == "long":
            threshold_series = slow_ma
            op = ">"
        else:
            threshold_series = slow_ma
            op = "<"
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
            # 阈值取超卖水平
            threshold_series = pd.Series([oversold] * len(df), index=df.index)
            op = ">"
        else:
            # 阈值取超买水平
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
        if direction == "long":
            threshold_series = signal_line.round(4)
            op = ">"
        else:
            threshold_series = signal_line.round(4)
            op = "<"
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
        if direction == "long":
            threshold_series = atr_ma.round(4)
            op = ">"
        else:
            threshold_series = atr_ma.round(4)
            op = "<"
        direction_series = pd.Series([op] * len(df), index=df.index)
        return threshold_series, direction_series

    else:
        # 对于无法转换的信号类型，直接跳过转换
        raise ValueError(f"未知或不支持的信号类型: {signal_type}")

def update_price_map(strategy_df, df, target_column='kai_column'):
    kai_column_list = strategy_df[target_column].unique().tolist()
    target_price_info_map = {}
    for kai_column in kai_column_list:
        threshold_price_series, direction = compute_threshold_direction(df, kai_column)
        # 获取threshold_price_series最后一行的值
        threshold_price = threshold_price_series.iloc[-1]
        last_direction = direction.iloc[-1]
        target_price_info_map[kai_column] = (threshold_price, last_direction)
    return target_price_info_map


async def fetch_new_data(strategy_df, max_period):
    """ 每分钟获取最新数据并更新 high_price_map 和 low_price_map """
    global kai_target_price_info_map, pin_target_price_info_map, current_minute, order_detail_map, price, price_list
    newest_data = LatestDataManager(max_period, INSTRUMENT)
    max_attempts = 200
    previous_timestamp = None
    while True:
        try:
            now = datetime.datetime.now()
            if current_minute is None or now.minute != current_minute:
                print(f"🕐 {now.strftime('%H:%M')} 触发数据更新...")
                # await asyncio.sleep(7)
                attempt = 0
                while attempt < max_attempts:
                    df = newest_data.get_newest_data()  # 获取最新数据

                    # 获取当前 df 最后一行的 timestamp
                    latest_timestamp = df.iloc[-1]['timestamp'] if not df.empty else None

                    if previous_timestamp is None or latest_timestamp != previous_timestamp:
                        print(f"✅ 数据已更新，最新 timestamp: {latest_timestamp} 实时最新价格 {price}")
                        price_list = []

                        # 更新映射
                        kai_target_price_info_map = update_price_map(strategy_df, df, target_column='kai_column')
                        pin_target_price_info_map = update_price_map(strategy_df, df, target_column='pin_column')

                        print(f"📈 更新开仓价格映射：{kai_target_price_info_map}  📈 更新平仓价格映射：{pin_target_price_info_map}")
                        previous_timestamp = latest_timestamp
                        current_minute = now.minute  # 更新当前分钟
                        break  # 数据已更新，跳出循环
                    else:
                        print(f"⚠️ 数据未变化，尝试重新获取 ({attempt + 1}/{max_attempts})...")
                        attempt += 1

                if attempt == max_attempts:
                    print("❌ 3 次尝试后数据仍未更新，跳过此轮更新。")

            await asyncio.sleep(1)  # 每秒检查一次当前分钟
        except Exception as e:
            pin_target_price_info_map = {}
            kai_target_price_info_map = {}
            traceback.print_exc()


async def subscribe_channel(ws, inst_id):
    """
    订阅指定交易对的最新成交数据
    """
    subscribe_msg = {
        "op": "subscribe",
        "args": [{"channel": "trades", "instId": inst_id}]
    }
    await ws.send(json.dumps(subscribe_msg))
    print(f"📡 已订阅 {inst_id} 实时数据")


def process_open_orders(price, default_size, kai_reverse_map):
    """
    根据最新成交价判断是否需要开仓（买多或卖空）
    """
    # 检查高价策略（买多）
    for key, target_info in kai_target_price_info_map.items():
        if target_info is not None:
            threshold_price, comp = target_info
            is_reverse = kai_reverse_map[key]
            side = 'buy' if 'long' in key else 'sell'
            if is_reverse:
                side = 'buy' if side == 'sell' else 'sell'
            pin_side = 'sell' if side == 'buy' else 'buy'
            # print(f"open: {key:<25} [开仓检测] 检查信号 {key:<25}: 当前价格 {price:>10.2f}, 阈值 {threshold_price:>10.2f}, 比较符 {comp:^5}")
            if comp == '>':
                if price >= threshold_price and key not in order_detail_map:
                    result = place_order(INSTRUMENT, side, default_size)
                    if result:
                        order_detail_map[key] = {
                            'price': price,
                            'side': side,
                            'pin_side': pin_side,
                            'time': current_minute,
                            'size': default_size
                        }
                        print(f"开仓成功 {key} 成交，价格：{price}，时间：{datetime.datetime.now()}")
                        # 持久化订单详情
                        save_order_detail_map()
            if comp == '<':
                if price <= threshold_price and key not in order_detail_map:
                    result = place_order(INSTRUMENT, side, default_size)
                    if result:
                        order_detail_map[key] = {
                            'price': price,
                            'side': side,
                            'pin_side': pin_side,
                            'time': current_minute,
                            'size': default_size
                        }
                        print(f"开仓成功 {key} 成交，价格：{price}，时间：{datetime.datetime.now()}")
                        # 持久化订单详情
                        save_order_detail_map()


def process_close_orders(price, kai_pin_map):
    """
    根据最新成交价判断是否需要平仓
    """
    keys_to_remove = []  # 暂存需要移除的订单 key
    for kai_key, order_detail in list(order_detail_map.items()):
        # 如果该订单是在当前分钟下的单则跳过平仓检测
        if current_minute == order_detail['time']:
            continue

        pin_key = kai_pin_map.get(kai_key)
        if not pin_key:
            continue

        kai_price = order_detail['price']
        if pin_key in pin_target_price_info_map:
            target_info = pin_target_price_info_map[pin_key]
            if target_info is not None:
                threshold_price, comp = target_info
                # print(f"close: {kai_key:<20} [平仓检测] 检查信号 {pin_key:<20} 对应开仓 {kai_key:<10}: 当前价格 {price:>10.2f}, 阈值 {threshold_price:>10.2f}, 比较符 {comp:^5}, 开仓价格 {kai_price:>10.2f}")
                if comp == '>':
                    if price > threshold_price:
                        result = place_order(INSTRUMENT, order_detail['pin_side'], order_detail['size'],
                                             trade_action="close")
                        if result:
                            keys_to_remove.append(kai_key)
                            print(
                                f"📈 【平仓】{pin_key} {order_detail['pin_side']} 成交，价格：{price}，开仓价格 {kai_price} "
                                f"kai_key {kai_key} pin_key {pin_key} order_time {order_detail['time']} "
                                f"current_minute {current_minute} 时间：{datetime.datetime.now()}")
                if comp == '<':
                    if price < threshold_price:
                        result = place_order(INSTRUMENT, order_detail['pin_side'], order_detail['size'],
                                             trade_action="close")
                        if result:
                            keys_to_remove.append(kai_key)
                            print(
                                f"📉 【平仓】{pin_key} {order_detail['pin_side']} 成交，价格：{price}，开仓价格 {kai_price} "
                                f"kai_key {kai_key} pin_key {pin_key} order_time {order_detail['time']} "
                                f"current_minute {current_minute} 时间：{datetime.datetime.now()}")

    # 移除已经平仓完成的订单
    if keys_to_remove:
        for key in keys_to_remove:
            order_detail_map.pop(key, None)
        # 平仓操作后保存订单状态
        save_order_detail_map()


def delete_rows_based_on_sort_key(result_df, sort_key, range_key):
    """
    删除 DataFrame 中的行，使得每一行的 sort_key 都是当前及后续行中最大的。
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


def choose_good_strategy_debug(inst_id='BTC'):
    # df = pd.read_csv('temp/temp.csv')
    # count_L()
    # 找到temp下面所有包含False的文件
    file_list = os.listdir('temp')
    file_list = [file for file in file_list if 'True' in file and inst_id in file and 'USDT-SWAP.csv_continue_1_20_1_ma_1_3000_300_peak_1_3000_300_rsi_1_1000_40_macross_1_1000_20_1_1000_20_relate_1_1000_30_1_100_6_abs_1_1000_20_1_25_1_'  in file and 'op' in file]
    # file_list = file_list[0:1]
    df_list = []
    df_map = {}
    for file in file_list:
        file_key = file.split('_')[4]
        df = pd.read_csv(f'temp/{file}')
        df['max_beilv'] = df['net_profit_rate'] / df['max_profit']
        df['loss_beilv'] = -df['net_profit_rate'] / df['max_consecutive_loss']
        temp_value = 1
        df['filename'] = file.split('_')[5]
        df = df[(df['kai_column'].str.contains('abs')) & (df['pin_column'].str.contains('abs'))]
        df = df[(df['net_profit_rate'] > 1)]
        df['monthly_trade_std_score'] = df['monthly_trade_std'] / (df['kai_count']) * 22
        df['monthly_net_profit_std_score'] = df['monthly_net_profit_std'] / (df['net_profit_rate']) * 22
        df['monthly_avg_profit_std_score'] = df['monthly_avg_profit_std'] / (df['avg_profit_rate']) * 100
        df = df[(df['avg_profit_rate'] > 1)]
        if file_key not in df_map:
            df_map[file_key] = []
        df_map[file_key].append(df)
    for key in df_map:
        df = pd.concat(df_map[key])
        df_list.append(df)
        return df

    temp = pd.merge(df_list[0], df_list[1], on=['kai_side', 'kai_column', 'pin_column'], how='inner')
    fields = ['avg_profit_rate', 'net_profit_rate', 'max_beilv']
    for field in fields:
        x_col = f"{field}_x"
        y_col = f"{field}_y"
        temp[f"{field}_min"] = temp[[x_col, y_col]].min(axis=1)
        temp[f"{field}_mean"] = temp[[x_col, y_col]].mean(axis=1)
        temp[f"{field}_plus"] = temp[x_col] + temp[y_col]
        temp[f"{field}_cha"] = temp[x_col] - temp[y_col]
        temp[f"{field}_mult"] = np.where(
            (temp[x_col] < 0) & (temp[y_col] < 0),
            0,
            temp[x_col] * temp[y_col]
        )
    return temp


def calculate_final_score(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    根据聚合后的 result_df 中各信号的统计指标，计算最终综合评分。
    """
    eps = 1e-8  # 防止除 0
    df = result_df.copy()

    # 1. 盈利能力指标归一化
    for col in ['net_profit_rate', 'avg_profit_rate']:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if abs(max_val - min_val) < eps:
                df[col + '_norm'] = 1.0
            else:
                df[col + '_norm'] = df[col] / 100
        else:
            df[col + '_norm'] = 0.0

    df['profitability_score'] = (df['net_profit_rate_norm'] + df['avg_profit_rate_norm']) / 2.0

    # 2. 稳定性/风险指标归一化
    for col in ['loss_rate', 'monthly_loss_rate']:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if abs(max_val - min_val) < eps:
                df[col + '_score'] = 1.0
            else:
                df[col + '_score'] = 1 - df[col]
        else:
            df[col + '_score'] = 1.0

    temp_value = 2
    if 'monthly_avg_profit_std' in df.columns and 'avg_profit_rate' in df.columns:
        df['risk_volatility'] = df['monthly_avg_profit_std'] / (df['avg_profit_rate'].abs() + eps) * 100
        min_val = df['risk_volatility'].min()
        max_val = df['risk_volatility'].max()
        if abs(max_val - min_val) < eps:
            df['risk_volatility_score'] = 1.0
        else:
            df['risk_volatility_score'] = temp_value - df['risk_volatility']
    else:
        df['risk_volatility_score'] = 1.0

    if 'monthly_net_profit_std' in df.columns and 'net_profit_rate' in df.columns:
        df['risk_volatility_net'] = df['monthly_net_profit_std'] / (df['net_profit_rate'].abs() + eps) * 22
        min_val = df['risk_volatility_net'].min()
        max_val = df['risk_volatility_net'].max()
        if abs(max_val - min_val) < eps:
            df['risk_volatility_net_score'] = 1.0
        else:
            df['risk_volatility_net_score'] = temp_value - df['risk_volatility_net']
    else:
        df['risk_volatility_net_score'] = 1.0

    df['risk_volatility_avg_score'] = temp_value - df['true_profit_std'] / df['avg_profit_rate'] * 100

    df['stability_score'] = (
        df['loss_rate_score'] +
        df['monthly_loss_rate_score'] +
        df['risk_volatility_score'] +
        df['risk_volatility_net_score'] +
        df['risk_volatility_avg_score']
    ) / 5

    profit_weight = 0.4
    stability_weight = 0.6
    df['final_score'] = profit_weight * df['profitability_score'] + stability_weight * df['stability_score']
    df['final_score'] = df['stability_score'] * df['profitability_score']
    df = df[(df['final_score'] > 0)]
    return df





async def websocket_listener(kai_pin_map, kai_reverse_map):
    """
    监听 OKX WebSocket 实时数据，处理开仓和平仓逻辑
    """
    default_size = min_count_map[INSTRUMENT]
    global kai_target_price_info_map, pin_target_price_info_map, order_detail_map, current_minute, price, price_list

    while True:
        try:
            async with websockets.connect(OKX_WS_URL) as ws:
                print("✅ 已连接到 OKX WebSocket")
                await subscribe_channel(ws, INSTRUMENT)
                # 持续监听 WebSocket 消息
                while True:
                    try:
                        response = await ws.recv()
                        data = json.loads(response)

                        if "data" not in data:
                            continue

                        for trade in data["data"]:
                            price = float(trade["px"])
                            if price in price_list:
                                continue
                            price_list.append(price)
                            process_open_orders(price, default_size, kai_reverse_map)
                            process_close_orders(price, kai_pin_map)

                    except websockets.exceptions.ConnectionClosed:
                        print("🔴 WebSocket 连接断开，正在重连...")
                        break

        except Exception as e:
            traceback.print_exc()


async def main():
    # 确保temp目录存在并加载之前保存的order_detail_map
    if not os.path.exists("temp"):
        os.makedirs("temp")
    load_order_detail_map()
    exclude_str = ['macross', 'rsi', 'macd', 'cci', 'atr']

    inst_id = INSTRUMENT.split('-')[0]
    final_good_df = pd.read_parquet(f'temp/final_good_{inst_id}_True_filter_all.parquet')
    # 过滤掉kai_column和pin_column中包含exclude_str的行
    for exclude in exclude_str:
        final_good_df = final_good_df[~final_good_df['kai_column'].str.contains(exclude)]
        final_good_df = final_good_df[~final_good_df['pin_column'].str.contains(exclude)]
    # 只保留score_final最大的前10行
    final_good_df = final_good_df.sort_values(by='score_final', ascending=False)
    final_good_df = final_good_df.head(10)

    print(f'final_good_df shape: {final_good_df.shape[0]}')
    period_list = []
    for index, row in final_good_df.iterrows():
        if 'is_reverse' not in row:
            is_reverse = False
        else:
            is_reverse = row['is_reverse']
        kai_column = row['kai_column']
        kai_period = int(kai_column.split('_')[1])
        period_list.append(kai_period)
        pin_column = row['pin_column']
        pin_period = int(pin_column.split('_')[1])
        kai_pin_map[kai_column] = pin_column
        kai_reverse_map[kai_column] = is_reverse
        period_list.append(pin_period)

    max_period = max(period_list)
    max_period = int(np.ceil(max_period / 100) * 100)
    await asyncio.gather(
        fetch_new_data(final_good_df, max_period),  # 定时更新数据
        websocket_listener(kai_pin_map, kai_reverse_map)  # 监听实时数据
    )

# 运行 asyncio 事件循环
asyncio.run(main())