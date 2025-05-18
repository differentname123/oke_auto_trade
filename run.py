#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    "XRP-USDT-SWAP": 0.01,
    "PEPE-USDT-SWAP": 0.1
}


class InstrumentTrader:
    def __init__(self, instrument):
        self.instrument = instrument
        self.min_count = min_count_map.get(instrument, 0)
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
        if result:
            self.order_detail_map[signal_name] = {
                "price": price_val,
                "side": side,
                "pin_side": pin_side,
                "time": self.current_minute,
                "size": self.min_count,
            }
            print(
                f"开仓成功 {signal_name} for {self.instrument} 成交, 价格: {price_val}, 时间: {datetime.datetime.now()}"
            )
            self.save_order_detail_map()

    def close_order(self, signal_name, price_val):
        keys_to_remove = []
        for kai_key, order in list(self.order_detail_map.items()):
            if self.current_minute == order["time"]:
                continue
            pin_key = self.kai_pin_map.get(kai_key)
            if pin_key == signal_name:
                kai_price = order["price"]
                result = place_order(
                    self.instrument, order["pin_side"], order["size"], trade_action="close"
                )
                if result:
                    keys_to_remove.append(kai_key)
                    print(
                        f"【平仓】 {pin_key} for {self.instrument} {order['pin_side']} 成交, 价格: {price_val}, 开仓价格: {kai_price}, 时间: {datetime.datetime.now()}"
                    )
                else:
                    print(
                        f"❌ {pin_key} for {self.instrument} 平仓失败, 价格: {price_val}, 开仓价格: {kai_price}, 时间: {datetime.datetime.now()}"
                    )
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
                    if result:
                        self.order_detail_map[key] = {
                            "price": price_val,
                            "side": side,
                            "pin_side": pin_side,
                            "time": self.current_minute,
                            "size": self.min_count,
                        }
                        print(
                            f"开仓成功 {key} for {self.instrument} 成交, 价格: {price_val}, 时间: {datetime.datetime.now()} 最小价格: {min_price}, 最大价格: {max_price}"
                        )
                        self.save_order_detail_map()

    def process_close_orders(self, price_val):
        keys_to_remove = []
        for kai_key, order in list(self.order_detail_map.items()):
            if self.current_minute == order["time"]:
                continue
            pin_key = self.kai_pin_map.get(kai_key)
            if not pin_key:
                continue
            kai_price = order["price"]
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
                            print(
                                f"【平仓】 {pin_key} for {self.instrument} {order['pin_side']} 成交, 价格: {price_val}, 开仓价格: {kai_price}, 时间: {datetime.datetime.now()}"
                            )
                        else:
                            print(
                                f"❌ {pin_key} for {self.instrument} 平仓失败, 价格: {price_val}, 开仓价格: {kai_price}, 时间: {datetime.datetime.now()}"
                            )
        if keys_to_remove:
            for k in keys_to_remove:
                self.order_detail_map.pop(k, None)
            self.save_order_detail_map()

    async def fetch_new_data(self, max_period):
        kai_column_list = self.strategy_df["kai_column"].unique().tolist()
        pin_column_list = self.strategy_df["pin_column"].unique().tolist()
        print(
            f"【{self.instrument}】当前策略数据的开仓信号数量: {len(kai_column_list)} 平仓信号数量: {len(pin_column_list)}"
        )
        not_close_signal_key = ["abs", "relate", "donchian"]
        newest_data = LatestDataManager(max_period, self.instrument)
        max_attempts = 200
        previous_timestamp = None

        while True:
            try:
                now = datetime.datetime.now()
                if self.current_minute is None or now.minute != self.current_minute:
                    print(f"🕐 {now.strftime('%H:%M')} {self.instrument} 触发数据更新...")
                    attempt = 0
                    while attempt < max_attempts:
                        origin_df = newest_data.get_newnewest_data()
                        print(f"最新数据的时间{origin_df.iloc[-1]['timestamp']}")
                        df = origin_df[origin_df["confirm"] == "1"]
                        latest_timestamp = df.iloc[-1]["timestamp"] if not df.empty else None
                        if previous_timestamp is None or latest_timestamp != previous_timestamp:
                            print(
                                f"✅ {self.instrument} 数据已更新, 最新 timestamp: {latest_timestamp} 实时最新价格: {self.price}"
                            )
                            # 处理 close 类型的开仓和平仓
                            exist_kai_keys = list(self.order_detail_map.keys())
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
                            for kai in need_close_kai:
                                signal_flag, target_price = self.compute_last_signal(df, kai)
                                if signal_flag:
                                    self.open_order(kai, target_price)
                            print(
                                f"【{self.instrument}】  耗时: {datetime.datetime.now() - start_time} 需要close价格开仓的开仓信号:{len(need_close_kai)}  {need_close_kai} 不需要close价格开仓的开仓信号: {len(not_need_close_kai)} {not_need_close_kai}"
                            )

                            need_close_pin = []
                            not_need_close_pin = []
                            for pin in pin_column_list:
                                if any(k in pin for k in not_close_signal_key):
                                    not_need_close_pin.append(pin)
                                else:
                                    need_close_pin.append(pin)
                            start_time = datetime.datetime.now()
                            for pin in need_close_pin:
                                signal_flag, target_price = self.compute_last_signal(df, pin)
                                if signal_flag:
                                    self.close_order(pin, target_price)
                            print(
                                f"【{self.instrument}】  耗时: {datetime.datetime.now() - start_time} 需要close价格开仓的平仓信号:{len(need_close_pin)}  {need_close_pin} 不需要close价格开仓的平仓信号: {len(not_need_close_pin)} {not_need_close_pin}"
                            )

                            for kai in not_need_close_kai:
                                signal_flag, target_price = self.compute_last_signal(origin_df, kai)
                                if "long" in kai:
                                    self.kai_target_price_info_map[kai] = (target_price, 100000)
                                else:
                                    self.kai_target_price_info_map[kai] = (0, target_price)

                            for pin in not_need_close_pin:
                                signal_flag, target_price = self.compute_last_signal(origin_df, pin)
                                if "long" in pin:
                                    self.pin_target_price_info_map[pin] = (target_price, 100000)
                                else:
                                    self.pin_target_price_info_map[pin] = (0, target_price)

                            self.price_list.clear()

                            print(
                                f"{self.instrument} 开仓信号个数 {len(self.kai_target_price_info_map)}  详细结果：{self.kai_target_price_info_map} 平仓信号个数{len(self.pin_target_price_info_map)}  详细结果：{self.pin_target_price_info_map}"
                            )
                            self.is_new_minute = True
                            previous_timestamp = latest_timestamp
                            self.current_minute = now.minute
                            break
                        else:
                            print(
                                f"⚠️ {self.instrument} 数据未变化, 尝试重新获取 ({attempt + 1}/{max_attempts})"
                            )
                            attempt += 1
                    if attempt == max_attempts:
                        print(f"❌ {self.instrument} 多次尝试数据仍未更新，跳过本轮更新")
                await asyncio.sleep(1)
            except Exception as e:
                self.pin_target_price_info_map = {}
                self.kai_target_price_info_map = {}
                self.is_new_minute = True
                traceback.print_exc()

    async def subscribe_channel(self, ws):
        subscribe_msg = {
            "op": "subscribe",
            "args": [{"channel": "trades", "instId": self.instrument}],
        }
        await ws.send(json.dumps(subscribe_msg))
        print(f"📡 {self.instrument} 已订阅实时数据")

    async def websocket_listener(self):
        while True:
            try:
                async with websockets.connect(OKX_WS_URL) as ws:
                    print(f"✅ {self.instrument} 连接到 OKX WebSocket")
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
                            print(f"🔴 {self.instrument} WebSocket 连接断开，重连中...")
                            break
            except Exception as e:
                traceback.print_exc()

    def save_order_detail_map(self):
        try:
            if not os.path.exists("temp"):
                os.makedirs("temp")
            file_path = f"temp/order_detail_map_{self.instrument}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.order_detail_map, f)
        except Exception as e:
            traceback.print_exc()

    def load_order_detail_map(self):
        file_path = f"temp/order_detail_map_{self.instrument}.json"
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    self.order_detail_map.update(json.load(f))
                print(f"✅ {self.instrument} 已加载之前的订单信息")
            except Exception as e:
                print(f"❌ {self.instrument} 加载订单信息失败:", e)
        else:
            self.order_detail_map.clear()

    async def main_trading_loop(self):
        # 加载历史订单记录
        self.load_order_detail_map()

        # 加载策略数据（例如 parquet 文件）
        inst_id = self.instrument.split("-")[0]
        all_dfs = []
        for is_reverse in [True, False]:
            corr_path = f"temp/corr/{inst_id}_{is_reverse}_filter_similar_strategy.parquet_corr_weekly_net_profit_detail.parquet"
            origin_good_path = f"temp/corr/{inst_id}_{is_reverse}_filter_similar_strategy.parquet_origin_good_weekly_net_profit_detail.parquet"
            if os.path.exists(origin_good_path):
                temp_strategy_df = pd.read_parquet(origin_good_path)
                correlation_df = pd.read_parquet(corr_path)
                selected_strategies, selected_correlation_df = select_strategies_optimized(
                    temp_strategy_df,
                    correlation_df,
                    k=20,
                    penalty_scaler=0.1,
                    use_absolute_correlation=True,
                )
                all_dfs.append(selected_strategies)
                print(f"{self.instrument} final_good_df shape: {selected_strategies.shape[0]} 来自 {origin_good_path}")
        if all_dfs:
            self.strategy_df = pd.concat(all_dfs)
            self.strategy_df = self.strategy_df.drop_duplicates(subset=["kai_column"])
            self.strategy_df = self.strategy_df.drop_duplicates(subset=["pin_column"])
            print(f"【{self.instrument}】策略数据加载成功, 策略数量: {self.strategy_df.shape[0]}")
        else:
            print(f"❌ {self.instrument} 策略数据不存在!")
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
                print("Error parsing period from signal name:", ex)
        max_period = int(np.ceil(max(period_list) / 100) * 100) if period_list else 100
        max_period = max_period * 2
        print(f"【{self.instrument}】最大周期: {max_period}")

        # 同时启动数据更新任务和 WebSocket 监听任务
        await asyncio.gather(
            self.fetch_new_data(max_period),
            self.websocket_listener(),
        )


def run_instrument(instrument):
    print(f"【进程启动】开始处理 {instrument}")
    trader = InstrumentTrader(instrument)
    asyncio.run(trader.main_trading_loop())


if __name__ == "__main__":
    processes = []
    for instr in INSTRUMENT_LIST:
        p = multiprocessing.Process(target=run_instrument, args=(instr,))
        p.start()
        processes.append(p)
        time.sleep(10)  # 在启动下一个进程前暂停10秒

    for p in processes:
        p.join()