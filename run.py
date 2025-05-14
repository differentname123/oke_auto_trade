import asyncio
import os
import traceback
import json
import datetime
import multiprocessing

import numpy as np
import pandas as pd
import websockets
from functools import lru_cache

from common_utils import compute_signal,select_strategies_optimized
from trade_common import LatestDataManager, place_order

# WebSocket 服务器地址
OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
# 定义需要操作的多个交易对
# INSTRUMENT_LIST = ["SOL-USDT-SWAP", "BTC-USDT-SWAP", "ETH-USDT-SWAP", "TON-USDT-SWAP", "DOGE-USDT-SWAP", "XRP-USDT-SWAP"]
INSTRUMENT_LIST = [ "BTC-USDT-SWAP"]

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
is_new_minute = True        # 表示是否是新的一分钟

def get_newest_threshold_price(
    df: pd.DataFrame,
    signal_name: str,
    search_percent: float = 0.1,
    step: float = 0.01,
):
    """
    两阶段搜索：
      1. 先用 step 走等距粗网格，确定所有连续 True 片段；
      2. 再对首片段下边界、末片段上边界做二分细化。
    若区间内无 True，返回 (None, None)。
    """

    # ---------- 预处理 ----------
    idx = df.index[-1]                      # 最后一根 bar 的行号
    orig_high: float = df.at[idx, "high"]
    orig_low: float = df.at[idx, "low"]
    last_close: float = df.at[idx, "close"]

    lower_bound = last_close * (1 - search_percent)
    upper_bound = last_close * (1 + search_percent)

    # 用 linspace 生成包含端点的等距网格
    n_points = int(round((upper_bound - lower_bound) / step)) + 1
    coarse_prices = np.linspace(lower_bound, upper_bound, n_points, dtype=float)

    # ---------- 核心计算 ----------
    @lru_cache(maxsize=4096)
    def is_signal_true(price: float) -> bool:
        """
        修改最后一根 K 线的高低收 -> 计算信号 -> 返回最新一条信号值
        采用就地修改 + 事后还原，避免整表 copy。
        """
        # 备份原值
        bak_high, bak_low, bak_close = df.loc[idx, ["high", "low", "close"]]

        # 写入新值
        df.at[idx, "high"] = max(price, orig_high)
        df.at[idx, "low"] = min(price, orig_low)
        df.at[idx, "close"] = price

        sig_series, _ = compute_signal(df, signal_name)
        result = bool(sig_series.iat[-1])

        # 还原
        df.at[idx, "high"] = bak_high
        df.at[idx, "low"] = bak_low
        df.at[idx, "close"] = bak_close
        return result

    # 1) 粗网格扫描
    coarse_flags = np.fromiter(
        (is_signal_true(p) for p in coarse_prices),
        dtype=bool,
        count=n_points,
    )

    # 2) NumPy 一行找连续 True 片段
    diff = np.diff(np.concatenate(([0], coarse_flags.view("i1"), [0])))
    seg_starts = np.where(diff == 1)[0]
    seg_ends = np.where(diff == -1)[0] - 1
    segments = list(zip(seg_starts, seg_ends))

    if not segments:  # 全 False
        return (None, None)

    # ---------- 二分细化 ----------
    tol = step / 10.0
    max_iter = 50

    def bisect_first_true(lo: float, hi: float) -> float:
        """闭区间内找第一个 True（返回值向左逼近）"""
        for _ in range(max_iter):
            mid = (lo + hi) * 0.5
            if is_signal_true(mid):
                hi = mid
            else:
                lo = mid
            if hi - lo < tol:
                break
        return hi

    def bisect_last_true(lo: float, hi: float) -> float:
        """闭区间内找最后一个 True（返回值向右逼近）"""
        for _ in range(max_iter):
            mid = (lo + hi) * 0.5
            if is_signal_true(mid):
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return lo

    # ---- 细化第一段下边界 ----
    first_seg_start, _ = segments[0]
    coarse_lower = coarse_prices[first_seg_start]
    if first_seg_start == 0:
        refined_lower = coarse_lower
    else:
        false_left = coarse_prices[first_seg_start - 1]
        refined_lower = (
            coarse_lower
            if is_signal_true(false_left)
            else bisect_first_true(false_left, coarse_lower)
        )

    # ---- 细化最后一段上边界 ----
    _, last_seg_end = segments[-1]
    coarse_upper = coarse_prices[last_seg_end]
    if last_seg_end == len(coarse_prices) - 1:
        refined_upper = coarse_upper
    else:
        false_right = coarse_prices[last_seg_end + 1]
        refined_upper = (
            coarse_upper
            if is_signal_true(false_right)
            else bisect_last_true(coarse_upper, false_right)
        )

    return refined_lower, refined_upper

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
            min_price, max_price = get_newest_threshold_price(df, kai_column)
        except Exception as e:
            print(f"❌ 计算 {kai_column} 时出现错误：", e)
            continue
        if min_price is None or max_price is None:
            print(f"❌ {kai_column} 的目标价格计算失败")
            continue
        target_price_info_map[kai_column] = (min_price, max_price)
    return target_price_info_map

##############################################
# 异步任务：数据更新
##############################################
async def fetch_new_data(max_period):
    global kai_target_price_info_map, pin_target_price_info_map, current_minute, price, price_list, strategy_df, INSTRUMENT, is_new_minute
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
                        is_new_minute = True
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
            is_new_minute = True
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
    global price, price_list, is_new_minute
    current_high = 0
    current_low = 0
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
                            if is_new_minute:
                                print(f"🕐 {INSTRUMENT} 新的一分钟，当前价格: {price_val}上一分钟最高价: {current_high}上一分钟最低价: {current_low}")
                                current_high = 0
                                current_low = 0
                                is_new_minute = False
                            if price_val in price_list:
                                continue
                            if current_high == 0 or current_high < price_val:
                                current_high = price_val
                            if current_low == 0 or current_low > price_val:
                                current_low = price_val

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
            min_price, max_price = target_info
            is_reverse = kai_reverse_map.get(key, False)
            side = 'buy' if 'long' in key else 'sell'
            if is_reverse:
                side = 'buy' if side == 'sell' else 'sell'
            pin_side = 'sell' if side == 'buy' else 'buy'
            if min_price < price_val and max_price > price_val:
                result = place_order(INSTRUMENT, side, MIN_COUNT)
                if result:
                    order_detail_map[key] = {
                        'price': price_val,
                        'side': side,
                        'pin_side': pin_side,
                        'time': current_minute,
                        'size': MIN_COUNT
                    }
                    print(f"开仓成功 {key} for {INSTRUMENT} 成交, 价格: {price_val}, 时间: {datetime.datetime.now()} 最小价格: {min_price}, 最大价格: {max_price}")
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
                min_price, max_price = target_info
                if min_price < price_val and max_price > price_val:
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
    for is_reverse in [True, False]:

        corr_path = f'temp/corr/{inst_id}_{is_reverse}_filter_similar_strategy.parquet_corr_weekly_net_profit_detail.parquet'
        origin_good_path = f'temp/corr/{inst_id}_{is_reverse}_filter_similar_strategy.parquet_origin_good_weekly_net_profit_detail.parquet'



        if os.path.exists(origin_good_path):
            temp_strategy_df = pd.read_parquet(origin_good_path)
            correlation_df = pd.read_parquet(corr_path)
            selected_strategies, selected_correlation_df = select_strategies_optimized(temp_strategy_df, correlation_df,k=10, penalty_scaler=0.1, use_absolute_correlation=True)
            all_df.append(selected_strategies)
            print(f'{INSTRUMENT} final_good_df shape: {selected_strategies.shape[0]} 来自 {origin_good_path}')
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
    max_period = max_period * 2
    print(f"【{INSTRUMENT}】最大周期: {max_period}")

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