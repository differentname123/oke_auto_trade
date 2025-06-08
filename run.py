#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
该代码实现了基于 OKX 交易所实时数据的多品种自动交易系统，其主要功能和流程包括：

多品种并行运行

针对多个交易品种（例如 SOL-USDT-SWAP、BTC-USDT-SWAP、ETH-USDT-SWAP 等），通过多进程分别启动各自的交易任务，保证各品种独立运行。
实时数据订阅与更新

通过 WebSocket 连接 OKX 实时数据接口，订阅交易品种的交易数据。
同步拉取最新行情数据（借助外部数据管理器），确保及时获得更新的价格和交易信息。
交易信号生成与运算

根据历史行情和多种技术指标（支持 abs、relate、donchian、boll、macross、rsi、macd、cci、atr 等），在最新数据基础上计算开仓（kai）和平仓（pin）信号及目标价格。
实时判断信号条件是否满足，辅助制定交易决策。
自动下单与订单管理

当满足开仓或平仓信号时，调用外部下单接口自动执行买卖操作，并记录下单详情。
对订单信息进行管理和持久化（保存至 JSON 文件），同时记录平仓订单，便于后续跟踪与统计。
策略数据加载与优化

在交易启动前加载预先筛选好的策略数据，并使用优化工具对信号进行映射（构建 kai 平仓信号的对应关系及反向标记），为实时交易提供依据。
动态更新目标价格范围，确保信号判断与交易执行的准确性。
总体而言，该系统结合了实时 WebSocket 数据、历史行情分析、多种技术指标计算以及自动下单执行，通过异步与多进程协同，实现了多个交易品种的实时自动化交易和订单管理。
"""

import asyncio
import os
import time
import traceback
import json
import datetime
import multiprocessing

import numpy as np
import pandas as pd
import websockets

from common_utils import select_strategies_optimized
from trade_common import LatestDataManager, place_order

# --------------------
# 自定义日志函数，仅记录当前代码输出的日志
# 日志文件存储在 log 文件夹下，文件名按照日期划分
# --------------------
def get_log_file_path():
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    log_dir = "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return os.path.join(log_dir, f"{today_str}.log")

def log_info(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{timestamp} [INFO] {message}"
    print(log_message)
    with open(get_log_file_path(), "a", encoding="utf-8") as f:
        f.write(log_message + "\n")

def log_warning(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{timestamp} [WARNING] {message}"
    print(log_message)
    with open(get_log_file_path(), "a", encoding="utf-8") as f:
        f.write(log_message + "\n")

def log_error(message, exc_info=False):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{timestamp} [ERROR] {message}"
    print(log_message)
    with open(get_log_file_path(), "a", encoding="utf-8") as f:
        f.write(log_message + "\n")
        if exc_info:
            error_trace = traceback.format_exc()
            f.write(error_trace + "\n")

# WebSocket 服务器地址
OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
# 定义需要操作的多个交易对
INSTRUMENT_LIST = ["SOL-USDT-SWAP", "BTC-USDT-SWAP", "ETH-USDT-SWAP", "TON-USDT-SWAP", "DOGE-USDT-SWAP", "XRP-USDT-SWAP"]
# INSTRUMENT_LIST = ["BTC-USDT-SWAP"]

# 各交易对最小下单量映射
min_count_map = {
    "BTC-USDT-SWAP": 0.01,
    "ETH-USDT-SWAP": 0.01,
    "SOL-USDT-SWAP": 0.01,
    "TON-USDT-SWAP": 1,
    "DOGE-USDT-SWAP": 0.01,
    "XRP-USDT-SWAP": 0.01
}
max_leverage_map = {
    "BTC-USDT-SWAP": 100,
    "ETH-USDT-SWAP": 100,
    "SOL-USDT-SWAP": 50,
    "TON-USDT-SWAP": 20,
    "DOGE-USDT-SWAP": 50,
    "XRP-USDT-SWAP": 50
}
true_leverage_map = {
    "BTC-USDT-SWAP": 10000,
    "ETH-USDT-SWAP": 1000,
    "SOL-USDT-SWAP": 5000,
    "TON-USDT-SWAP": 20,
    "DOGE-USDT-SWAP": 0.05,
    "XRP-USDT-SWAP": 0.5
}

class InstrumentTrader:
    def __init__(self, instrument):
        self.instrument = instrument
        self.min_count = min_count_map.get(instrument, 0) * 1
        self.order_detail_map = {}
        self.price = 0.0
        self.price_list = []
        self.current_minute = None
        self.kai_target_price_info_map = {}
        self.pin_target_price_info_map = {}
        self.kai_pin_map = {}
        self.kai_reverse_map = {}
        self.strategy_df = None
        self.is_new_minute = True

    @staticmethod
    def compute_last_signal(df, col_name):
        """
        根据历史行情数据(df)和指定信号名称(col_name)生成最后一行的交易信号与价格。
        支持的信号类型包括：abs, relate, donchian, boll, macross, rsi, macd, cci, atr。
        当数据不足时返回 (False, np.nan)
        """
        parts = col_name.split("_")
        signal_type = parts[0]
        direction = parts[-1]
        N = len(df)

        if N == 0:
            raise ValueError("DataFrame 为空！")

        if signal_type == "abs":
            period = int(parts[1])
            abs_value = float(parts[2]) / 100
            if N < period + 1:
                return False, np.nan
            if direction == "long":
                window = df["low"].iloc[N - period - 1: N - 1]
                min_low = window.min()
                target_price = round(min_low * (1 + abs_value), 4)
                cond = df["high"].iloc[-1] > target_price
            else:
                window = df["high"].iloc[N - period - 1: N - 1]
                max_high = window.max()
                target_price = round(max_high * (1 - abs_value), 4)
                cond = df["low"].iloc[-1] < target_price

            valid_trade = (target_price >= df["low"].iloc[-1]) and (target_price <= df["high"].iloc[-1])
            return (cond and valid_trade), target_price

        elif signal_type == "relate":
            period = int(parts[1])
            percent = float(parts[2]) / 100
            if N < period + 1:
                return False, np.nan
            low_window = df["low"].iloc[N - period - 1: N - 1]
            high_window = df["high"].iloc[N - period - 1: N - 1]
            min_low = low_window.min()
            max_high = high_window.max()
            if direction == "long":
                target_price = round(min_low + percent * (max_high - min_low), 4)
                cond = df["high"].iloc[-1] > target_price
            else:
                target_price = round(max_high - percent * (max_high - min_low), 4)
                cond = df["low"].iloc[-1] < target_price
            valid_trade = (target_price >= df["low"].iloc[-1]) and (target_price <= df["high"].iloc[-1])
            return (cond and valid_trade), target_price

        elif signal_type == "donchian":
            period = int(parts[1])
            if N < period + 1:
                return False, np.nan
            if direction == "long":
                highest_high = df["high"].iloc[N - period - 1: N - 1].max()
                cond = df["high"].iloc[-1] > highest_high
                target_price = highest_high
            else:
                lowest_low = df["low"].iloc[N - period - 1: N - 1].min()
                cond = df["low"].iloc[-1] < lowest_low
                target_price = lowest_low
            valid_trade = (target_price >= df["low"].iloc[-1]) and (target_price <= df["high"].iloc[-1])
            return (cond and valid_trade), round(target_price, 4)

        elif signal_type == "boll":
            period = int(parts[1])
            std_multiplier = float(parts[2])
            if N < period + 1:
                return False, np.nan
            current_window = df["close"].iloc[N - period: N]
            prev_window = df["close"].iloc[N - period - 1: N - 1]
            current_ma = current_window.mean()
            current_std = current_window.std(ddof=1)
            current_upper = round(current_ma + std_multiplier * current_std, 4)
            current_lower = round(current_ma - std_multiplier * current_std, 4)
            prev_ma = prev_window.mean()
            prev_std = prev_window.std(ddof=1)
            prev_upper = round(prev_ma + std_multiplier * prev_std, 4)
            prev_lower = round(prev_ma - std_multiplier * prev_std, 4)
            if direction == "long":
                cond = (df["close"].iloc[-2] < prev_lower) and (df["close"].iloc[-1] >= current_lower)
            else:
                cond = (df["close"].iloc[-2] > prev_upper) and (df["close"].iloc[-1] <= current_upper)
            return cond, df["close"].iloc[-1]

        elif signal_type == "macross":
            fast_period = int(parts[1])
            slow_period = int(parts[2])
            if N < max(fast_period, slow_period) + 1:
                return False, np.nan
            curr_fast = df["close"].iloc[N - fast_period: N].mean()
            curr_slow = df["close"].iloc[N - slow_period: N].mean()
            prev_fast = df["close"].iloc[N - fast_period - 1: N - 1].mean()
            prev_slow = df["close"].iloc[N - slow_period - 1: N - 1].mean()
            if direction == "long":
                cond = (prev_fast < prev_slow) and (curr_fast >= curr_slow)
            else:
                cond = (prev_fast > prev_slow) and (curr_fast <= curr_slow)
            return cond, df["close"].iloc[-1]

        elif signal_type == "rsi":
            period = int(parts[1])
            overbought = int(parts[2])
            oversold = int(parts[3])
            if N < period + 2:
                return False, np.nan
            current_window = df["close"].iloc[N - period - 1: N].to_numpy(dtype=np.float64)
            prev_window = df["close"].iloc[N - period - 2: N - 1].to_numpy(dtype=np.float64)
            current_diff = np.diff(current_window)
            prev_diff = np.diff(prev_window)
            current_gain = np.maximum(current_diff, 0)
            current_loss = np.maximum(-current_diff, 0)
            avg_gain_current = current_gain.mean()
            avg_loss_current = current_loss.mean()
            rs_current = avg_gain_current / avg_loss_current if avg_loss_current != 0 else np.inf
            rsi_current = 100 - 100 / (1 + rs_current)
            prev_gain = np.maximum(prev_diff, 0)
            prev_loss = np.maximum(-prev_diff, 0)
            avg_gain_prev = prev_gain.mean()
            avg_loss_prev = prev_loss.mean()
            rs_prev = avg_gain_prev / avg_loss_prev if avg_loss_prev != 0 else np.inf
            rsi_prev = 100 - 100 / (1 + rs_prev)
            if direction == "long":
                cond = (rsi_prev < oversold) and (rsi_current >= oversold)
            else:
                cond = (rsi_prev > overbought) and (rsi_current <= overbought)
            return cond, df["close"].iloc[-1]

        elif signal_type == "macd":
            fast_period, slow_period, signal_period = map(int, parts[1:4])
            if N < 2:
                return False, np.nan
            fast_ema_series = df["close"].ewm(span=fast_period, adjust=False).mean()
            slow_ema_series = df["close"].ewm(span=slow_period, adjust=False).mean()
            macd_series = fast_ema_series - slow_ema_series
            signal_series = macd_series.ewm(span=signal_period, adjust=False).mean()
            macd_prev = macd_series.iloc[-2]
            macd_current = macd_series.iloc[-1]
            signal_prev = signal_series.iloc[-2]
            signal_current = signal_series.iloc[-1]
            if direction == "long":
                cond = (macd_prev < signal_prev) and (macd_current >= signal_current)
            else:
                cond = (macd_prev > signal_prev) and (macd_current <= signal_current)
            return cond, df["close"].iloc[-1]

        elif signal_type == "cci":
            period = int(parts[1])
            if N < period:
                return False, np.nan
            tp = (df["high"] + df["low"] + df["close"]) / 3
            tp_window = tp.iloc[N - period: N]
            current_ma = tp_window.mean()
            current_md = np.mean(np.abs(tp_window - current_ma))
            cci_current = (tp.iloc[-1] - current_ma) / (0.015 * current_md) if current_md != 0 else 0
            tp_window_prev = tp.iloc[N - period - 1: N - 1]
            prev_ma = tp_window_prev.mean()
            prev_md = np.mean(np.abs(tp_window_prev - prev_ma))
            cci_prev = (tp.iloc[-2] - prev_ma) / (0.015 * prev_md) if prev_md != 0 else 0
            if direction == "long":
                cond = (cci_prev < -100) and (cci_current >= -100)
            else:
                cond = (cci_prev > 100) and (cci_current <= 100)
            return cond, df["close"].iloc[-1]

        elif signal_type == "atr":
            period = int(parts[1])
            # 至少需要 2*period 个点才能计算 atr 和其均线
            if N < 2 * period:
                return False, np.nan
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"] - df["close"].shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(window=period, min_periods=period).mean()
            atr_ma = atr.rolling(window=period, min_periods=period).mean()
            atr_prev = atr.iloc[-2]
            atr_current = atr.iloc[-1]
            atr_ma_prev = atr_ma.iloc[-2]
            atr_ma_current = atr_ma.iloc[-1]
            if direction == "long":
                cond = (atr_prev < atr_ma_prev) and (atr_current >= atr_ma_current)
            else:
                cond = (atr_prev > atr_ma_prev) and (atr_current <= atr_ma_current)
            return cond, df["close"].iloc[-1]

        else:
            raise ValueError(f"未知信号类型: {signal_type}")

    def open_order(self, signal_name, price_val):
        is_reverse = self.kai_reverse_map.get(signal_name, False)
        side = "buy" if "long" in signal_name else "sell"
        if is_reverse:
            side = "buy" if side == "sell" else "sell"
        pin_side = "sell" if side == "buy" else "buy"
        result = place_order(self.instrument, side, self.min_count)
        # 获取易读的当前时间
        current_time_human = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if result:
            self.order_detail_map[signal_name] = {
                "open_time": current_time_human,
                "price": price_val,
                "side": side,
                "pin_side": pin_side,
                "time": self.current_minute,
                "size": self.min_count,
            }
            log_info(f"开仓成功 {side} {signal_name} for {self.instrument} 成交, 价格: {price_val}, 时间: {datetime.datetime.now()}")
            self.save_order_detail_map()

    def close_order(self, signal_name, price_val):
        keys_to_remove = []
        for kai_key, order in list(self.order_detail_map.items()):
            current_time_human = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if current_time_human == order["time"]:
                log_info(f"当前时间与订单时间相同，跳过平仓: {current_time_human} == {order['time']}")
                continue
            pin_key = self.kai_pin_map.get(kai_key)
            if pin_key == signal_name:
                kai_price = order["price"]
                side = order["side"]
                result = place_order(
                    self.instrument, order["pin_side"], order["size"], trade_action="close"
                )
                if result:
                    keys_to_remove.append(kai_key)
                    log_info(f"【平仓成功】 {pin_key} for {self.instrument} 开仓方向 {side}成交, 开仓价格: {kai_price} 平仓价格: {price_val}, 开仓时间: {order['open_time']} 平仓时间: {datetime.datetime.now()}")
                    # 记录平仓订单详情
                    close_record = {
                        "instrument": self.instrument,
                        'kai_signal': kai_key,
                        "pin_signal": pin_key,
                        "open_time": order["open_time"],
                        "close_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "open_price": kai_price,
                        "close_price": price_val,
                        "side": side,
                        "pin_side": order["pin_side"],
                        "size": order["size"],
                    }
                    self.record_closed_order(close_record)
                else:
                    log_error(f"❌ {pin_key} for {self.instrument} 平仓失败, 价格: {price_val}, 开仓价格: {kai_price}, 时间: {datetime.datetime.now()}")
        if keys_to_remove:
            for k in keys_to_remove:
                self.order_detail_map.pop(k, None)
            self.save_order_detail_map()

    def process_open_orders(self, price_val):
        for key, target_info in self.kai_target_price_info_map.items():
            if target_info is not None:
                min_price, max_price = target_info
                is_reverse = self.kai_reverse_map.get(key, False)
                side = "buy" if "long" in key else "sell"
                if is_reverse:
                    side = "buy" if side == "sell" else "sell"
                pin_side = "sell" if side == "buy" else "buy"
                if min_price < price_val < max_price:
                    result = place_order(self.instrument, side, self.min_count)
                    current_time_human = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if result:
                        self.order_detail_map[key] = {
                            "open_time": current_time_human,
                            "price": price_val,
                            "side": side,
                            "pin_side": pin_side,
                            "time": self.current_minute,
                            "size": self.min_count,
                        }
                        log_info(f"开仓成功 {key} for {self.instrument} 成交, 价格: {price_val}, 时间: {datetime.datetime.now()} 最小价格: {min_price}, 最大价格: {max_price}")
                        self.save_order_detail_map()

    def process_close_orders(self, price_val):
        keys_to_remove = []
        for kai_key, order in list(self.order_detail_map.items()):
            current_time_human = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if current_time_human == order["time"]:
                log_info(f"当前时间与订单时间相同，跳过平仓: {current_time_human} == {order['time']}")
                continue
            pin_key = self.kai_pin_map.get(kai_key)
            if not pin_key:
                continue
            kai_price = order["price"]
            side = order["side"]
            if pin_key in self.pin_target_price_info_map:
                target_info = self.pin_target_price_info_map[pin_key]
                if target_info is not None:
                    min_price, max_price = target_info
                    if min_price < price_val < max_price:
                        result = place_order(
                            self.instrument, order["pin_side"], order["size"], trade_action="close"
                        )
                        if result:
                            keys_to_remove.append(kai_key)
                            log_info(f"【平仓成功】 {pin_key} for {self.instrument} 开仓方向 {side}成交, 开仓价格: {kai_price} 平仓价格: {price_val}, 开仓时间: {order['open_time']} 平仓时间: {datetime.datetime.now()}")
                            # 记录平仓订单详情
                            close_record = {
                                "instrument": self.instrument,
                                'kai_signal': kai_key,
                                "pin_signal": pin_key,
                                "open_time": order["open_time"],
                                "close_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "open_price": kai_price,
                                "close_price": price_val,
                                "side": side,
                                "pin_side": order["pin_side"],
                                "size": order["size"],
                            }
                            self.record_closed_order(close_record)
                        else:
                            log_error(f"❌ {pin_key} for {self.instrument} 平仓失败, 价格: {price_val}, 开仓价格: {kai_price}, 时间: {datetime.datetime.now()}")
        if keys_to_remove:
            for k in keys_to_remove:
                self.order_detail_map.pop(k, None)
            self.save_order_detail_map()

    async def fetch_new_data(self, max_period):
        kai_column_list = self.strategy_df["kai_column"].unique().tolist()
        pin_column_list = self.strategy_df["pin_column"].unique().tolist()
        log_info(f"【{self.instrument}】当前策略数据的开仓信号数量: {len(kai_column_list)} 平仓信号数量: {len(pin_column_list)}")
        not_close_signal_key = ["abs", "relate", "donchian"]
        newest_data = LatestDataManager(max_period, self.instrument)
        max_attempts = 200
        previous_timestamp = None

        while True:
            try:
                now = datetime.datetime.now()
                if self.current_minute is None or now.minute != self.current_minute:
                    attempt = 0
                    while attempt < max_attempts:
                        origin_df = newest_data.get_newnewest_data()
                        df = origin_df[origin_df["confirm"] == "1"]
                        latest_timestamp = df.iloc[-1]["timestamp"] if not df.empty else None
                        if previous_timestamp is None or latest_timestamp != previous_timestamp:
                            log_info(f"✅ {self.instrument} 数据已更新, 最新 timestamp: {latest_timestamp} {origin_df.iloc[-1]['close']} 实时最新价格: {self.price} 最新数据的时间: {origin_df.iloc[-1]['timestamp']}")
                            exist_kai_keys = list(self.order_detail_map.keys())
                            exist_pin_keys = [self.kai_pin_map[k] for k in exist_kai_keys]
                            log_info(f"【{self.instrument}】当前持仓的开仓信号数量: {len(exist_kai_keys)} 平仓信号数量: {len(exist_pin_keys)}")
                            need_close_kai = []
                            not_need_close_kai = []
                            for kai in kai_column_list:
                                if kai in exist_kai_keys:
                                    continue
                                if any(k in kai for k in not_close_signal_key):
                                    not_need_close_kai.append(kai)
                                else:
                                    need_close_kai.append(kai)
                            start_time = datetime.datetime.now()
                            detail_map = {}
                            for kai in need_close_kai:
                                signal_flag, target_price = self.compute_last_signal(df, kai)
                                if signal_flag:
                                    detail_map[kai] = target_price
                                    self.open_order(kai, target_price)
                            log_info(f"【{self.instrument}】 耗时: {int((datetime.datetime.now() - start_time).total_seconds() * 1000)}ms 需要close价格开仓的开仓信号:{len(need_close_kai)} {detail_map} 不需要close价格开仓的开仓信号: {len(not_need_close_kai)} {not_need_close_kai}")

                            need_close_pin = []
                            not_need_close_pin = []
                            for pin in exist_pin_keys:
                                if any(k in pin for k in not_close_signal_key):
                                    not_need_close_pin.append(pin)
                                else:
                                    need_close_pin.append(pin)
                            start_time = datetime.datetime.now()
                            detail_map = {}
                            for pin in need_close_pin:
                                signal_flag, target_price = self.compute_last_signal(df, pin)
                                if signal_flag:
                                    detail_map[pin] = target_price
                                    self.close_order(pin, target_price)
                            log_info(f"【{self.instrument}】 耗时: {int((datetime.datetime.now() - start_time).total_seconds() * 1000)} ms 需要close价格开仓的平仓信号:{len(need_close_pin)} {detail_map} 不需要close价格开仓的平仓信号: {len(not_need_close_pin)} {not_need_close_pin}")

                            for kai in not_need_close_kai:
                                signal_flag, target_price = self.compute_last_signal(origin_df, kai)
                                if "long" in kai:
                                    self.kai_target_price_info_map[kai] = (target_price, 10000000)
                                else:
                                    self.kai_target_price_info_map[kai] = (0, target_price)

                            for pin in not_need_close_pin:
                                signal_flag, target_price = self.compute_last_signal(origin_df, pin)
                                if "long" in pin:
                                    self.pin_target_price_info_map[pin] = (target_price, 10000000)
                                else:
                                    self.pin_target_price_info_map[pin] = (0, target_price)

                            self.price_list.clear()
                            log_info(f"{self.instrument} 开仓信号个数 {len(self.kai_target_price_info_map)}  详细结果：{self.kai_target_price_info_map} 平仓信号个数{len(self.pin_target_price_info_map)}  详细结果：{self.pin_target_price_info_map}")
                            self.is_new_minute = True
                            previous_timestamp = latest_timestamp
                            self.current_minute = now.minute
                            break
                        else:
                            attempt += 1
                    if attempt == max_attempts:
                        log_error(f"❌ {self.instrument} 多次尝试数据仍未更新，跳过本轮更新")
                await asyncio.sleep(1)
            except Exception as e:
                self.pin_target_price_info_map = {}
                self.kai_target_price_info_map = {}
                self.is_new_minute = True
                log_error("Error in fetch_new_data", exc_info=True)

    async def subscribe_channel(self, ws):
        subscribe_msg = {
            "op": "subscribe",
            "args": [{"channel": "trades", "instId": self.instrument}],
        }
        await ws.send(json.dumps(subscribe_msg))
        log_info(f"📡 {self.instrument} 已订阅实时数据")

    async def websocket_listener(self):
        while True:
            try:
                async with websockets.connect(OKX_WS_URL) as ws:
                    log_info(f"✅ {self.instrument} 连接到 OKX WebSocket")
                    await self.subscribe_channel(ws)
                    while True:
                        try:
                            response = await ws.recv()
                            data = json.loads(response)
                            if "data" not in data:
                                continue
                            for trade in data["data"]:
                                price_val = float(trade["px"])
                                if self.is_new_minute:
                                    self.is_new_minute = False
                                if price_val in self.price_list:
                                    continue
                                self.price_list.append(price_val)
                                self.price = price_val
                                self.process_open_orders(price_val)
                                self.process_close_orders(price_val)
                        except websockets.exceptions.ConnectionClosed:
                            log_warning(f"🔴 {self.instrument} WebSocket 连接断开，重连中...")
                            await asyncio.sleep(2)  # 休息2秒再尝试连接
                            break
                        except Exception as e:
                            log_error("Error in websocket_listener inner loop", exc_info=True)
            except Exception as e:
                log_error("Error in websocket_listener", exc_info=True)

    def save_order_detail_map(self):
        try:
            if not os.path.exists("temp"):
                os.makedirs("temp")
            file_path = f"temp/order_detail_map_{self.instrument}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.order_detail_map, f, indent=4, ensure_ascii=False)
        except Exception as e:
            log_error("Error in save_order_detail_map", exc_info=True)

    def load_order_detail_map(self):
        file_path = f"temp/order_detail_map_{self.instrument}.json"
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    self.order_detail_map.update(json.load(f))
                log_info(f"✅ {self.instrument} 已加载之前的订单信息")
            except Exception as e:
                log_error(f"❌ {self.instrument} 加载订单信息失败", exc_info=True)
        else:
            self.order_detail_map.clear()

    def record_closed_order(self, record):
        """
        将平仓订单记录保存到文件中，文件路径 temp/closed_order_record_<instrument>.json
        """
        try:
            if not os.path.exists("temp"):
                os.makedirs("temp")
            file_path = f"temp/closed_order_record_{self.instrument}.json"
            records = []
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        records = json.load(f)
                    except Exception:
                        records = []
            records.append(record)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=4, ensure_ascii=False)
        except Exception as e:
            log_error("Error in record_closed_order", exc_info=True)

    async def main_trading_loop(self, inst_info):
        # 加载历史订单记录
        self.load_order_detail_map()
        # 加载策略数据（例如 parquet 文件）
        all_dfs = []
        for type in ['all_short', 'all']:
            selected_strategies = inst_info[type]["strategies"]
            all_dfs.append(selected_strategies)
            log_info(f"{self.instrument} final_good_df shape: {selected_strategies.shape[0]}")
        if all_dfs:
            self.strategy_df = pd.concat(all_dfs)
            buy_count = (self.strategy_df["side"] == "buy").sum()
            sell_count = (self.strategy_df["side"] == "sell").sum()
            self.strategy_df.to_parquet(f"temp/strategy_df_{self.instrument}.parquet", index=False)
            log_info(f"【{self.instrument}】策略数据加载成功, 策略数量: {self.strategy_df.shape[0]} 做多信号数量: {buy_count} 做空信号数量: {sell_count}")
        else:
            log_error(f"❌ {self.instrument} 策略数据不存在!")
            return

        # 构造 kai_pin_map 与 kai_reverse_map
        period_list = []
        for idx, row in self.strategy_df.iterrows():
            is_reverse_flag = row.get("is_reverse", False)
            kai = row["kai_column"]
            pin = row["pin_column"]
            self.kai_pin_map[kai] = pin
            self.kai_reverse_map[kai] = is_reverse_flag
            try:
                period_list.append(int(kai.split("_")[1]))
                period_list.append(int(pin.split("_")[1]))
            except Exception as ex:
                log_error("Error parsing period from signal name", exc_info=True)
        max_period = int(np.ceil(max(period_list) / 100) * 100) if period_list else 100
        max_period = max_period * 2
        log_info(f"【{self.instrument}】最大周期: {max_period}")

        # 同时启动数据更新任务和 WebSocket 监听任务
        await asyncio.gather(
            self.fetch_new_data(max_period),
            # self.websocket_listener(),
        )

def run_instrument(inst_info):
    instrument = inst_info['instrument']
    log_info(f"【进程启动】开始处理 {instrument}")
    trader = InstrumentTrader(instrument)
    asyncio.run(trader.main_trading_loop(inst_info))

def calc_leverage_metrics(agg_profit: np.ndarray,
                          agg_kai: np.ndarray,
                          cand_length: int):
    """
    根据累计收益数据（仅针对活跃周：kai > 0）计算杠杆指标：
      - optimal_leverage: 最优整数杠杆
      - optimal_capital: 在最优杠杆下的累计收益率（初始本金为 1）
      - capital_no_leverage: 不加杠杆情况下的累计收益率
    """
    avg_profit = agg_profit / cand_length
    active_mask = agg_kai > 0
    active_count = np.count_nonzero(active_mask)

    if active_count == 0:
        return np.nan, np.nan, np.nan

    active_avg = avg_profit[active_mask]
    capital_no_leverage = np.prod(1 + active_avg / 100)

    min_profit = active_avg.min()
    if min_profit >= 0:
        max_possible_leverage = 30
    else:
        max_possible_leverage = int(1 / (abs(min_profit) / 100))

    L_values = np.arange(1, max_possible_leverage + 1)
    factors = 1 + np.outer(L_values, active_avg) / 100  # shape: (num_leverages, active_weeks)
    safe = np.all(factors > 0, axis=1)
    capitals = np.where(safe, np.prod(factors, axis=1), -np.inf)
    optimal_index = np.argmax(capitals)
    optimal_leverage = int(L_values[optimal_index])
    optimal_capital = capitals[optimal_index]

    return optimal_leverage, optimal_capital, capital_no_leverage

def init():
    """
    进行初始化，主要是进行资金的分配，以及每个策略的单次买入数量
    :return:
    """
    total_capital = 1000000  # 总资金
    final_score_total = 0
    beam_width = 100000
    out_dir = 'temp_back'
    inst_map_info = {}
    for type in ['all_short', 'all']:
        temp_info = {}
        for inst in INSTRUMENT_LIST:
            inst_id = inst.split("-")[0]
            elements_path = f"{out_dir}/result_elements_{inst_id}_{beam_width}_{type}_op.parquet"
            origin_df_path = f"{out_dir}/{inst_id}_True_{type}_filter_similar_strategy.parquet"
            if not os.path.exists(elements_path) or not os.path.exists(origin_df_path):
                log_error(f"❌ {inst} 的元素文件或原始数据文件不存在，跳过初始化")
                continue
            elements_df = pd.read_parquet(elements_path)
            origin_df = pd.read_parquet(origin_df_path)
            elements_df = elements_df.sort_values(by='score_merged', ascending=False)
            row = elements_df.iloc[0]
            indices = row['strategies']
            score_merged = row['score_merged']
            weekly_net_profit_sum_merged = row['weekly_net_profit_sum_merged']
            strategies = origin_df.iloc[list(indices)].copy()

            # 将所有 weekly_net_profit_detail 堆叠成一个二维数组
            weekly_arrays = np.stack(strategies["weekly_net_profit_detail"].values)
            weekly_count_arrays = np.stack(strategies["weekly_kai_count_detail"].values)

            # 计算按列（每周）平均
            average_weekly_net_profit_detail = weekly_arrays.mean(axis=0)
            average_weekly_kai_count_detail = weekly_count_arrays.mean(axis=0)
            optimal_leverage, optimal_capital, no_leverage_capital = calc_leverage_metrics(average_weekly_net_profit_detail, average_weekly_kai_count_detail, 1)


            final_score = weekly_net_profit_sum_merged / score_merged
            temp_info[inst] = {
                'strategies': strategies,
                'score_merged': score_merged,
                'weekly_net_profit_sum_merged': weekly_net_profit_sum_merged,
                'final_score': final_score,
                'optimal_leverage':optimal_leverage,
                'optimal_capital':optimal_capital,
                'no_leverage_capital': no_leverage_capital
            }
            final_score_total += final_score
        inst_map_info[type] = temp_info
    # 计算每一个final_score占总共的百分比
    for type, inst_info in inst_map_info.items():
        for inst, info in inst_info.items():
            final_score = info['final_score']
            percent = final_score / final_score_total
            capital_no_leverage = total_capital * percent
            info['capital_no_leverage'] = capital_no_leverage
            log_info(f"【{inst}】策略 {type} 的最终得分: {final_score:.4f}, 占比: {percent:.4%}, 分配资金: {capital_no_leverage:.2f}")
    print(f"总的 final_score: {final_score_total:.4f}")
    return inst_map_info



if __name__ == "__main__":
    inst_map_info = init()
    processes = []
    for instr in INSTRUMENT_LIST:
        temp_inst_info = {}
        temp_inst_info['instrument'] = instr
        temp_inst_info['all'] = inst_map_info['all'][instr]
        temp_inst_info['all_short'] = inst_map_info['all_short'][instr]
        p = multiprocessing.Process(target=run_instrument, args=(temp_inst_info,))
        p.start()
        processes.append(p)
        time.sleep(10)  # 在启动下一个进程前暂停10秒

    for p in processes:
        p.join()