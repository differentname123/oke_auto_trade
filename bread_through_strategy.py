"""
突破策略的信号生成以及回测（稀疏存储版，不进行整数转换）
"""

import multiprocessing
import os
import pickle
import sys
import time
import traceback
from itertools import product

import numpy as np
import pandas as pd
from numba import njit


# 全局信号字典（只存储非零索引和对应的价格数据，均保持 float 类型）
GLOBAL_SIGNALS = {}


def iterative_search(func, base, step, max_range, tol):
    """
    在 [base - max_range, base + max_range] 内以步长 step 遍历，寻找使 func(candidate) 接近 0（|f(candidate)| < tol）的 candidate，
    若找到则返回该 candidate，否则返回 None。
    """
    candidates = np.arange(base - max_range, base + max_range + step, step)
    best_candidate = None
    best_val = float('inf')
    for candidate in candidates:
        val = abs(func(candidate))
        if val < tol:
            return candidate
        if val < best_val:
            best_val = val
            best_candidate = candidate
    return best_candidate if best_val < tol * 10 else None


def compute_future_signal(df, col_name):
    """
    根据历史行情数据 (df) 和信号名称 (col_name)，计算未来一个周期触发该信号所需的目标价格及比较方向。
    若上周期未满足前置条件，则返回 (None, None) 表示无效信号。

    参数:
      df: pandas.DataFrame，至少包含 "close", "high", "low" 列，
         数据按时间顺序排列（旧数据在前，新数据在后）。
      col_name: 信号名称字符串，格式为 "signalType_param1_param2_..._direction"，
                举例: "abs_20_2_long"、"boll_20_2_long"、"macross_12_26_long"、"macd_12_26_9_short"、
                "rsi_5_30_long"（RSI 设定在超卖 30）或 "cci_5_long"（CCI 长仓默认目标 -100）

    返回:
      tuple: (target_price, op)
         - target_price: float，未来周期需要达到/触发信号时的目标价格（四舍五入到小数点后4位）；若前置条件不满足，则为 None
         - op: str，对于多头信号，op 为 ">"，表示未来价格大于 target_price 时触发信号；
               对于空头信号，op 为 "<"；若前置条件不满足则返回 (None, None)。
    """
    parts = col_name.split('_')
    signal_type = parts[0]
    direction = parts[-1]

    # ------------------------ abs 信号 ------------------------
    if signal_type == 'abs':
        period = int(parts[1])
        abs_rate = float(parts[2]) / 100.0

        if len(df) < period + 1:
            raise ValueError("数据不足，无足够历史数据计算 abs 信号")

        if direction == "long":
            min_low = df['low'].iloc[-(period + 1):-1].min()
            target_price = round(min_low * (1 + abs_rate), 4)
            return target_price, ">"
        else:
            max_high = df['high'].iloc[-(period + 1):-1].max()
            target_price = round(max_high * (1 - abs_rate), 4)
            return target_price, "<"

    # ------------------------ relate 信号 ------------------------
    elif signal_type == 'relate':
        period = int(parts[1])
        percent = float(parts[2]) / 100.0

        if len(df) < period + 1:
            raise ValueError("数据不足，无足够历史数据计算 relate 信号")

        min_low = df['low'].iloc[-(period + 1):-1].min()
        max_high = df['high'].iloc[-(period + 1):-1].max()

        if direction == "long":
            target_price = round(min_low + percent * (max_high - min_low), 4)
            return target_price, ">"
        else:
            target_price = round(max_high - percent * (max_high - min_low), 4)
            return target_price, "<"

    # ------------------------ donchian 信号 ------------------------
    elif signal_type == 'donchian':
        period = int(parts[1])
        if len(df) < period + 1:
            raise ValueError("数据不足，无足够历史数据计算 donchian 信号")

        if direction == "long":
            highest_high = df['high'].iloc[-(period + 1):-1].max()
            target_price = round(highest_high, 4)
            return target_price, ">"
        else:
            lowest_low = df['low'].iloc[-(period + 1):-1].min()
            target_price = round(lowest_low, 4)
            return target_price, "<"

    # ------------------------ boll 信号 ------------------------
    elif signal_type == 'boll':
        period = int(parts[1])
        std_multiplier = float(parts[2])
        if len(df) < period:
            raise ValueError("数据不足，无足够历史数据计算 boll 信号")

        # 计算上周期 Bollinger 带（窗口取倒数 period+1 至倒数第1 行）
        hist_window = df['close'].iloc[-(period + 1):-1]
        if len(hist_window) < period:
            raise ValueError("数据不足，无足够历史数据计算 Bollinger 上周期指标")
        pre_ma = hist_window.mean()
        pre_std = hist_window.std(ddof=1)
        pre_lower = pre_ma - std_multiplier * pre_std
        pre_upper = pre_ma + std_multiplier * pre_std

        # 前置条件检查：多头要求上周期收盘价低于下轨；空头要求上周期收盘价高于上轨
        if direction == "long":
            if df['close'].iloc[-2] >= pre_lower:
                return None, None
        else:
            if df['close'].iloc[-2] <= pre_upper:
                return None, None

        # 模拟未来周期：新的窗口由最近 (period-1) 个收盘价加上未来价格 candidate 组成
        recent = df['close'].iloc[-(period - 1):]
        base = df['close'].iloc[-1]
        step = base * 0.001  # 0.1% 步长
        max_range = base * 0.1  # 最大 ±10%

        if direction == "long":
            # 对多头：要求 candidate = new_ma - std_multiplier * new_std
            def f(x):
                arr = np.append(recent.values, x)
                new_ma = arr.mean()
                new_std = np.std(arr, ddof=1)
                return x - (new_ma - std_multiplier * new_std)

            candidate = iterative_search(f, base, step, max_range, tol=1e-4)
            if candidate is None:
                return None, None
            return round(candidate, 4), ">"
        else:
            # 对空头：要求 candidate = new_ma + std_multiplier * new_std
            def f(x):
                arr = np.append(recent.values, x)
                new_ma = arr.mean()
                new_std = np.std(arr, ddof=1)
                return x - (new_ma + std_multiplier * new_std)

            candidate = iterative_search(f, base, step, max_range, tol=1e-4)
            if candidate is None:
                return None, None
            return round(candidate, 4), "<"

    # ------------------------ macross 信号 ------------------------
    elif signal_type == 'macross':
        fast_period = int(parts[1])
        slow_period = int(parts[2])
        if fast_period >= slow_period:
            raise ValueError("macross 信号中 fast_period 必须小于 slow_period")
        if len(df) < slow_period:
            raise ValueError("数据不足，无足够历史数据计算 macross 信号")

        # 计算上周期的均值
        fast_ma_prev = df['close'].iloc[-fast_period:].mean()
        slow_ma_prev = df['close'].iloc[-slow_period:].mean()
        # 前置条件：多头要求上周期 fast_ma < slow_ma；空头相反
        if direction == "long":
            if fast_ma_prev >= slow_ma_prev:
                return None, None
        else:
            if fast_ma_prev <= slow_ma_prev:
                return None, None

        # 模拟未来周期：新快均线 = (sum(最近 fast_period-1 个数据) + x)/fast_period，
        # 新慢均线 = (sum(最近 slow_period-1 个数据) + x)/slow_period，
        # 令两者相等求 x 的闭合解
        if fast_period > 1:
            fast_window = df['close'].iloc[-(fast_period - 1):]
            sum_fast = fast_window.sum()
        else:
            sum_fast = 0.0
        if slow_period > 1:
            slow_window = df['close'].iloc[-(slow_period - 1):]
            sum_slow = slow_window.sum()
        else:
            sum_slow = 0.0
        denominator = slow_period - fast_period
        x = (fast_period * sum_slow - slow_period * sum_fast) / denominator
        return round(x, 4), ">" if direction == "long" else "<"

    # ------------------------ macd 信号 ------------------------
    elif signal_type == 'macd':
        fast_period = int(parts[1])
        slow_period = int(parts[2])
        signal_period = int(parts[3])
        if len(df) < 1:
            raise ValueError("数据不足，无足够历史数据计算 macd 信号")

        fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        prev_macd = macd_line.iloc[-1]
        prev_signal = signal_line.iloc[-1]

        # 前置条件：对于多头，要求上周期 MACD < signal；空头相反
        if direction == "long":
            if prev_macd >= prev_signal:
                return None, None
        else:
            if prev_macd <= prev_signal:
                return None, None

        # 根据 EMA 的递推公式模拟未来一周期
        alpha_fast = 2 / (fast_period + 1)
        alpha_slow = 2 / (slow_period + 1)
        A = alpha_fast - alpha_slow
        B = fast_ema.iloc[-1] * (1 - alpha_fast) - slow_ema.iloc[-1] * (1 - alpha_slow)
        if A == 0:
            raise ValueError("macd 参数错误，导致除零")
        x = (prev_signal - B) / A
        return round(x, 4), ">" if direction == "long" else "<"

    # ------------------------ rsi 信号 ------------------------
    elif signal_type == 'rsi':
        period = int(parts[1])
        overbought = int(parts[2])
        oversold = int(parts[3])
        if len(df) < period + 1:
            raise ValueError("数据不足，无足够历史数据计算 rsi 信号")

        # 利用最近 period+1 个数据计算上周期 RSI
        window_prev = df['close'].iloc[-(period + 1):].values
        diffs = np.diff(window_prev)
        gains = np.maximum(diffs, 0)
        losses = -np.minimum(diffs, 0)
        avg_gain = gains.mean()
        avg_loss = losses.mean() if losses.mean() != 0 else 1e-6
        prev_rsi = 100 - 100 / (1 + avg_gain / avg_loss)

        # 前置条件：多头要求上周期 RSI < oversold；空头要求上周期 RSI > overbought
        if direction == "long":
            if prev_rsi >= oversold:
                return None, None
            # 未来窗口：取最近 period 个收盘价，加上 candidate 作为未来周期数据
            hist = df['close'].iloc[-period:].values
            base = df['close'].iloc[-1]
            step = base * 0.001
            max_range = base * 0.1

            def f(x):
                window_new = np.append(hist, x)
                d = np.diff(window_new)
                gains_new = np.maximum(d, 0)
                losses_new = -np.minimum(d, 0)
                avg_gain_new = gains_new.mean()
                avg_loss_new = losses_new.mean() if losses_new.mean() != 0 else 1e-6
                rsi_new = 100 - 100 / (1 + avg_gain_new / avg_loss_new)
                return rsi_new - oversold

            candidate = iterative_search(f, base, step, max_range, tol=1e-2)
            if candidate is None:
                return None, None
            return round(candidate, 4), ">"
        else:
            if prev_rsi <= overbought:
                return None, None
            hist = df['close'].iloc[-period:].values
            base = df['close'].iloc[-1]
            step = base * 0.001
            max_range = base * 0.1

            def f(x):
                window_new = np.append(hist, x)
                d = np.diff(window_new)
                gains_new = np.maximum(d, 0)
                losses_new = -np.minimum(d, 0)
                avg_gain_new = gains_new.mean()
                avg_loss_new = losses_new.mean() if losses_new.mean() != 0 else 1e-6
                rsi_new = 100 - 100 / (1 + avg_gain_new / avg_loss_new)
                return rsi_new - overbought

            candidate = iterative_search(f, base, step, max_range, tol=1e-2)
            if candidate is None:
                return None, None
            return round(candidate, 4), "<"

    # ------------------------ cci 信号 ------------------------
    elif signal_type == 'cci':
        period = int(parts[1])
        constant = 0.015
        tp = (df['high'] + df['low'] + df['close']) / 3
        if len(tp) < period:
            raise ValueError("数据不足，无足够历史数据计算 cci 信号")
        hist_window = tp.iloc[-period:]
        pre_ma = hist_window.mean()
        pre_md = hist_window.apply(lambda x: abs(x - pre_ma)).mean()
        pre_cci = (tp.iloc[-1] - pre_ma) / (constant * pre_md) if pre_md != 0 else 0

        # 前置条件：多头要求上周期 CCI < -100；空头要求上周期 CCI > 100
        if direction == "long":
            if pre_cci >= -100:
                return None, None
            hist_tp = tp.iloc[-(period - 1):].values  # 最近 period-1 个历史典型价格
            base = df['close'].iloc[-1]  # 以最后收盘价作为基准
            step = base * 0.001
            max_range = base * 0.1

            def f(x):
                # 假设未来周期 high, low, close 均等于 x, 则典型价格即为 x
                window_new = np.append(hist_tp, x)
                new_ma = window_new.mean()
                new_md = np.mean(np.abs(window_new - new_ma))
                new_cci = (x - new_ma) / (constant * new_md) if new_md != 0 else 0
                # 希望 new_cci 刚好达到 -100
                return new_cci + 100

            candidate = iterative_search(f, base, step, max_range, tol=1e-2)
            if candidate is None:
                return None, None
            return round(candidate, 4), ">"
        else:
            if pre_cci <= 100:
                return None, None
            hist_tp = tp.iloc[-(period - 1):].values
            base = df['close'].iloc[-1]
            step = base * 0.001
            max_range = base * 0.1

            def f(x):
                window_new = np.append(hist_tp, x)
                new_ma = window_new.mean()
                new_md = np.mean(np.abs(window_new - new_ma))
                new_cci = (x - new_ma) / (constant * new_md) if new_md != 0 else 0
                return new_cci - 100

            candidate = iterative_search(f, base, step, max_range, tol=1e-2)
            if candidate is None:
                return None, None
            return round(candidate, 4), "<"

    # ------------------------ atr 信号 ------------------------
    elif signal_type == 'atr':
        period = int(parts[1])
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        atr_ma = atr.rolling(period).mean()
        if direction == "long":
            if atr.iloc[-2] >= atr_ma.iloc[-2]:
                return None, None
        else:
            if atr.iloc[-2] <= atr_ma.iloc[-2]:
                return None, None
        target_price = round(df['close'].iloc[-1], 4)
        return target_price, ">" if direction == "long" else "<"

    else:
        raise ValueError(f"未知信号类型: {signal_type}")


def compute_signal(df, col_name):
    """
    根据历史行情数据(df)和指定信号名称(col_name)，生成交易信号和对应目标价格。

    当前支持的信号类型包括：
      - abs: 绝对百分比突破信号
          示例："abs_20_2_long" (20周期内最低价向上2%多头突破)
      - relate: 相对区间百分比突破信号
          示例："relate_20_50_short" (20周期区间顶部向下50%空头突破)
      - donchian: 唐奇安通道突破信号（实时价格触发优化）
          示例："donchian_20_long" (20周期最高价向上突破多头信号)
      - boll: 布林带信号
          示例："boll_20_2_long" 或 "boll_20_2_short"
      - macross: MACROSS 信号 (双均线交叉信号)
          示例："macross_10_20_long"
      - rsi: RSI 超买超卖反转信号
          示例："rsi_14_70_30_long"
      - macd: MACD交叉信号
          示例："macd_12_26_9_long"
      - cci: 商品通道指数超买超卖反转信号
          示例："cci_20_short"
      - atr: ATR波动率突破信号
          示例："atr_14_long"

    参数:
      df: pandas.DataFrame，必须包含这些列：
          "close": 收盘价
          "high": 最高价
          "low": 最低价
      col_name: 信号名称，格式如 "signalType_param1_param2_..._direction"

    返回:
      tuple:
        - signal_series: pandas.Series(bool), 当满足信号条件时置为 True
        - trade_price_series: pandas.Series(float), 信号触发时推荐执行的交易价格，
          此价格经过剪裁，确保一定处于当前 bar 的 low 和 high 之间。
    """

    parts = col_name.split('_')
    signal_type = parts[0]
    direction = parts[-1]

    if signal_type == 'abs':
        period = int(parts[1])
        abs_value = float(parts[2]) / 100
        if direction == "long":
            min_low_series = df['low'].shift(1).rolling(period).min()
            target_price = (min_low_series * (1 + abs_value)).round(4)
            signal_series = df['high'] > target_price
        else:
            max_high_series = df['high'].shift(1).rolling(period).max()
            target_price = (max_high_series * (1 - abs_value)).round(4)
            signal_series = df['low'] < target_price
        # 确保返回的价格位于当前 bar 的 low 与 high 之间
        trade_price_series = target_price.clip(lower=df['low'], upper=df['high'])
        return signal_series, trade_price_series

    elif signal_type == 'relate':
        period = int(parts[1])
        percent = float(parts[2]) / 100
        min_low_series = df['low'].shift(1).rolling(period).min()
        max_high_series = df['high'].shift(1).rolling(period).max()
        if direction == "long":
            target_price = (min_low_series + percent * (max_high_series - min_low_series)).round(4)
            signal_series = df['high'] > target_price
        else:
            target_price = (max_high_series - percent * (max_high_series - min_low_series)).round(4)
            signal_series = df['low'] < target_price
        trade_price_series = target_price.clip(lower=df['low'], upper=df['high'])
        return signal_series, trade_price_series

    elif signal_type == 'donchian':
        period = int(parts[1])
        if direction == "long":
            highest_high = df['high'].shift(1).rolling(period).max()
            signal_series = df['high'] > highest_high
            target_price = highest_high
        else:
            lowest_low = df['low'].shift(1).rolling(period).min()
            signal_series = df['low'] < lowest_low
            target_price = lowest_low
        trade_price_series = target_price.clip(lower=df['low'], upper=df['high']).round(4)
        return signal_series, trade_price_series

    elif signal_type == 'boll':
        period = int(parts[1])
        std_multiplier = float(parts[2])
        ma = df['close'].rolling(window=period, min_periods=period).mean()
        std_dev = df['close'].rolling(window=period, min_periods=period).std()
        upper_band = (ma + std_multiplier * std_dev).round(4)
        lower_band = (ma - std_multiplier * std_dev).round(4)
        if direction == "long":
            signal_series = (df['close'].shift(1) < lower_band.shift(1)) & (df['close'] >= lower_band)
        else:  # short
            signal_series = (df['close'].shift(1) > upper_band.shift(1)) & (df['close'] <= upper_band)
        # 这里交易价格使用收盘价；通常close位于low和high之间
        return signal_series, df["close"]

    elif signal_type == 'macross':
        fast_period = int(parts[1])
        slow_period = int(parts[2])
        fast_ma = df["close"].rolling(window=fast_period, min_periods=fast_period).mean().round(4)
        slow_ma = df["close"].rolling(window=slow_period, min_periods=slow_period).mean().round(4)
        if direction == "long":
            signal_series = (fast_ma.shift(1) < slow_ma.shift(1)) & (fast_ma >= slow_ma)
        else:
            signal_series = (fast_ma.shift(1) > slow_ma.shift(1)) & (fast_ma <= slow_ma)
        trade_price = df["close"]
        return signal_series, trade_price

    elif signal_type == 'rsi':
        period = int(parts[1])
        overbought = int(parts[2])
        oversold = int(parts[3])
        delta = df['close'].diff(1).astype(np.float32)
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        if direction == "long":
            signal_series = (rsi.shift(1) < oversold) & (rsi >= oversold)
        else:
            signal_series = (rsi.shift(1) > overbought) & (rsi <= overbought)
        return signal_series, df['close']

    elif signal_type == 'macd':
        fast_period, slow_period, signal_period = map(int, parts[1:4])
        fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        if direction == "long":
            signal_series = (macd_line.shift(1) < signal_line.shift(1)) & (macd_line >= signal_line)
        else:
            signal_series = (macd_line.shift(1) > signal_line.shift(1)) & (macd_line <= signal_line)
        return signal_series, df["close"]

    elif signal_type == 'cci':
        period = int(parts[1])
        constant = float(parts[2]) / 100
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma = tp.rolling(period).mean()
        md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        cci = (tp - ma) / (constant * md)
        if direction == "long":
            signal_series = (cci.shift(1) < -100) & (cci >= -100)
        else:
            signal_series = (cci.shift(1) > 100) & (cci <= 100)
        return signal_series, df['close']

    elif signal_type == 'atr':
        period = int(parts[1])
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        atr_ma = atr.rolling(period).mean()
        if direction == "long":
            signal_series = (atr.shift(1) < atr_ma.shift(1)) & (atr >= atr_ma)
        else:
            signal_series = (atr.shift(1) > atr_ma.shift(1)) & (atr <= atr_ma)
        return signal_series, df['close']

    else:
        raise ValueError(f"未知信号类型: {signal_type}")


@njit
def calculate_max_sequence_numba(series):
    n = series.shape[0]
    min_sum = 0.0
    cur_sum = 0.0
    start_idx = 0
    min_start = 0
    min_end = 0
    trade_count = 0
    max_trade_count = 0

    for i in range(n):
        if cur_sum == 0:
            start_idx = i
            trade_count = 0
        cur_sum += series[i]
        trade_count += 1
        if cur_sum < min_sum:
            min_sum = cur_sum
            min_start = start_idx
            min_end = i
            max_trade_count = trade_count
        if cur_sum > 0:
            cur_sum = 0
            trade_count = 0
    return min_sum, min_start, min_end, max_trade_count


@njit
def calculate_max_profit_numba(series):
    n = series.shape[0]
    max_sum = 0.0
    cur_sum = 0.0
    start_idx = 0
    max_start = 0
    max_end = 0
    trade_count = 0
    max_trade_count = 0

    for i in range(n):
        if cur_sum == 0:
            start_idx = i
            trade_count = 0
        cur_sum += series[i]
        trade_count += 1
        if cur_sum > max_sum:
            max_sum = cur_sum
            max_start = start_idx
            max_end = i
            max_trade_count = trade_count
        if cur_sum < 0:
            cur_sum = 0
            trade_count = 0
    return max_sum, max_start, max_end, max_trade_count


@njit
def compute_low_min_range(low_array, start_pos, end_pos):
    n = start_pos.shape[0]
    out = np.empty(n, dtype=low_array.dtype)
    for i in range(n):
        s = start_pos[i]
        e = end_pos[i] + 1  # 包含终点
        m = low_array[s]
        for j in range(s + 1, e):
            if low_array[j] < m:
                m = low_array[j]
        out[i] = m
    return out


@njit
def compute_high_max_range(high_array, start_pos, end_pos):
    n = start_pos.shape[0]
    out = np.empty(n, dtype=high_array.dtype)
    for i in range(n):
        s = start_pos[i]
        e = end_pos[i] + 1  # 包含终点
        m = high_array[s]
        for j in range(s + 1, e):
            if high_array[j] > m:
                m = high_array[j]
        out[i] = m
    return out


def optimize_parameters(df, tp_range=None, sl_range=None):
    if tp_range is None:
        tp_range = df['max_true_profit'].values
        tp_range = tp_range[tp_range > 0]
        tp_range = np.round(tp_range, 2)
        tp_range = np.unique(tp_range)
    if sl_range is None:
        sl_range = df['min_true_profit'].values
        sl_range = sl_range[sl_range < 0]
        sl_range = np.round(sl_range, 2)
        sl_range = np.unique(sl_range)

    true_profit = df['true_profit'].values
    max_true_profit = df['max_true_profit'].values
    min_true_profit = df['min_true_profit'].values
    n_trades = true_profit.shape[0]

    simulated_tp = np.where(
        max_true_profit[np.newaxis, :] >= tp_range[:, np.newaxis],
        tp_range[:, np.newaxis],
        true_profit[np.newaxis, :]
    )
    total_profits_tp = simulated_tp.sum(axis=1)
    loss_rates_tp = (simulated_tp < 0).sum(axis=1) / n_trades
    best_tp_index = np.argmax(total_profits_tp)
    best_tp = tp_range[best_tp_index]
    best_tp_profit = total_profits_tp[best_tp_index]
    best_tp_loss_rate = loss_rates_tp[best_tp_index]

    simulated_sl = np.where(
        min_true_profit[np.newaxis, :] <= sl_range[:, np.newaxis],
        sl_range[:, np.newaxis],
        true_profit[np.newaxis, :]
    )
    total_profits_sl = simulated_sl.sum(axis=1)
    loss_rates_sl = (simulated_sl < 0).sum(axis=1) / n_trades
    best_sl_index = np.argmax(total_profits_sl)
    best_sl = sl_range[best_sl_index]
    best_sl_profit = total_profits_sl[best_sl_index]
    best_sl_loss_rate = loss_rates_sl[best_sl_index]

    return {
        'max_optimal_value': best_tp,
        'max_optimal_profit': best_tp_profit,
        'max_optimal_loss_rate': best_tp_loss_rate,
        'min_optimal_value': best_sl,
        'min_optimal_profit': best_sl_profit,
        'min_optimal_loss_rate': best_sl_loss_rate
    }



def get_detail_backtest_result_op(df, kai_column, pin_column, is_filter=True, is_detail=False, is_reverse=False):
    """
    优化后的 get_detail_backtest_result_op 函数：
      1. 从全局预计算的稀疏信号数据中提取非零索引及对应价格；
      2. 利用向量化操作（缓存中间变量、减少重复转换）获取回测数据并计算各类指标；

    参数:
      df          : 原始行情 DataFrame，要求包含 timestamp, open, high, low, close 等列；
      kai_column  : 主信号列名，对应 GLOBAL_SIGNALS 中的 key；
      pin_column  : 对应匹配信号列名；
      is_filter   : 是否对结果过滤（按照 timestamp 排序并去重）；
      is_detail   : 是否执行详细计算（如 low_min、high_max 以及区间收益率计算等）。

    返回:
      kai_data_df : 最终包含匹配信号数据及计算结果的 DataFrame；
      statistic_dict : 包含回测统计指标的字典；
    """
    global GLOBAL_SIGNALS
    try:
        kai_idx, kai_prices = GLOBAL_SIGNALS[kai_column]
        pin_idx, pin_prices = GLOBAL_SIGNALS[pin_column]
    except KeyError:
        return None, None

    # 如果信号数量较少，则直接返回
    if kai_idx.size < 100 or pin_idx.size < 100:
        return None, None

    # 提取对应行并赋值价格数据
    kai_data_df = df.iloc[kai_idx].copy()
    pin_data_df = df.iloc[pin_idx].copy()
    kai_data_df["kai_price"] = kai_prices
    pin_data_df["pin_price"] = pin_prices

    # 判断两个信号的公共索引数（用来过滤不匹配的组合）
    common_index = kai_data_df.index.intersection(pin_data_df.index)
    same_count = len(common_index)
    kai_count = len(kai_data_df)
    pin_count = len(pin_data_df)
    same_count_rate = (100 * same_count / min(kai_count, pin_count)) if min(kai_count, pin_count) > 0 else 0
    # if same_count_rate > 80:
    #     return None, None

    # 使用 numpy 向量化查找匹配：对 kai_data_df 中的时间索引在 pin_data_df 中进行右侧查找
    kai_idx_arr = np.asarray(kai_data_df.index)
    pin_idx_arr = np.asarray(pin_data_df.index)
    pin_match_indices = np.searchsorted(pin_idx_arr, kai_idx_arr, side="right")
    valid_mask = pin_match_indices < len(pin_idx_arr)
    if valid_mask.sum() == 0:
        return None, None

    # 筛选有效数据，及对应的匹配结果
    kai_data_df = kai_data_df.iloc[valid_mask].copy()
    kai_idx_valid = kai_idx_arr[valid_mask]
    pin_match_indices_valid = pin_match_indices[valid_mask]
    matched_pin = pin_data_df.iloc[pin_match_indices_valid].copy()

    # 更新匹配数据：引入匹配的 pin_price 与 timestamp，并计算持仓时长（假设 index 为可直接相减的数值）
    kai_data_df["pin_price"] = matched_pin["pin_price"].values
    kai_data_df["pin_time"] = matched_pin["timestamp"].values
    # 利用匹配后 DataFrame 的索引值进行差值计算（采用 .values 避免重复转换）
    kai_data_df["hold_time"] = matched_pin.index.values - kai_idx_valid

    # 判断方向，仅判断一次，避免多处调用字符串查找
    if is_reverse:
        is_long = "short" in kai_column.lower()
    else:
        is_long = "long" in kai_column.lower()

    # 若要求详细计算，用已缓存的 NumPy 数组及向量化操作计算区间最低和最高价格，进而计算收益率区间
    if is_detail:
        df_index_arr = np.asarray(df.index)
        low_array = df["low"].values
        high_array = df["high"].values

        start_times = np.asarray(kai_data_df.index)
        end_times = np.asarray(matched_pin.index)
        start_pos = np.searchsorted(df_index_arr, start_times, side="left")
        end_pos = np.searchsorted(df_index_arr, end_times, side="right") - 1

        low_min_arr = compute_low_min_range(low_array, start_pos, end_pos)
        high_max_arr = compute_high_max_range(high_array, start_pos, end_pos)
        kai_data_df["low_min"] = low_min_arr
        kai_data_df["high_max"] = high_max_arr

        if is_long:
            kai_data_df["max_true_profit"] = (
                    ((kai_data_df["high_max"] - kai_data_df["kai_price"]) / kai_data_df["kai_price"] * 100) - 0.07
            ).round(4)
            kai_data_df["min_true_profit"] = (
                    ((kai_data_df["low_min"] - kai_data_df["kai_price"]) / kai_data_df["kai_price"] * 100) - 0.07
            ).round(4)
        else:
            kai_data_df["max_true_profit"] = (
                    ((kai_data_df["kai_price"] - kai_data_df["low_min"]) / kai_data_df["kai_price"] * 100) - 0.07
            ).round(4)
            kai_data_df["min_true_profit"] = (
                    ((kai_data_df["kai_price"] - kai_data_df["high_max"]) / kai_data_df["kai_price"] * 100) - 0.07
            ).round(4)

    # 若需要过滤，则对结果按 timestamp 排序，并根据 pin_time 去重
    if is_filter:
        kai_data_df = kai_data_df.sort_values("timestamp").drop_duplicates("pin_time", keep="first")
    # 基本统计指标
    trade_count = len(kai_data_df)
    total_count = len(df)

    # 根据 pin_time 建立映射，更新 kai_price，使得价格更准确
    pin_price_map = kai_data_df.set_index("pin_time")["pin_price"]
    mapped_prices = kai_data_df["timestamp"].map(pin_price_map)
    if same_count > 0 and not mapped_prices.isna().all():
        kai_data_df["kai_price"] = mapped_prices.combine_first(kai_data_df["kai_price"])
    modification_rate = round(100 * mapped_prices.notna().sum() / trade_count, 4) if trade_count else 0

    # 利用向量化方式计算收益率
    if is_long:
        profit_series = ((kai_data_df["pin_price"] - kai_data_df["kai_price"]) / kai_data_df["kai_price"] * 100).round(4)
    else:
        profit_series = ((kai_data_df["kai_price"] - kai_data_df["pin_price"]) / kai_data_df["kai_price"] * 100).round(4)
    kai_data_df["profit"] = profit_series
    kai_data_df["true_profit"] = profit_series - 0.07
    profit_sum = profit_series.sum()

    fix_profit = round(kai_data_df[mapped_prices.notna()]["true_profit"].sum(), 4)  # 收到影响的交易的收益，实盘交易时可以设置不得连续开平来避免。也就是将fix_profit减去就是正常的利润



    if is_detail and trade_count > 0:
        max_single_profit = kai_data_df["max_true_profit"].max()
        min_single_profit = kai_data_df["min_true_profit"].min()
        temp_dict = optimize_parameters(kai_data_df) if trade_count > 0 else {}
    else:
        max_single_profit = kai_data_df["true_profit"].max()
        min_single_profit = kai_data_df["true_profit"].min()
        temp_dict = {}

    true_profit_std = kai_data_df["true_profit"].std()
    true_profit_mean = kai_data_df["true_profit"].mean() * 100 if trade_count > 0 else 0

    profits_arr = kai_data_df["true_profit"].values
    max_loss, max_loss_start_idx, max_loss_end_idx, loss_trade_count = calculate_max_sequence_numba(profits_arr)
    max_profit, max_profit_start_idx, max_profit_end_idx, profit_trade_count = calculate_max_profit_numba(profits_arr)

    # 根据索引获取最大连续亏损的起止时间和持仓时长
    if trade_count > 0 and max_loss_start_idx < len(kai_data_df) and max_loss_end_idx < len(kai_data_df):
        max_loss_start_time = kai_data_df.iloc[max_loss_start_idx]["timestamp"]
        max_loss_end_time = kai_data_df.iloc[max_loss_end_idx]["timestamp"]
        max_loss_hold_time = kai_data_df.index[max_loss_end_idx] - kai_data_df.index[max_loss_start_idx]
    else:
        max_loss_start_time = max_loss_end_time = max_loss_hold_time = None

    # 同理，计算最大连续盈利的起止时间和持仓时长
    if trade_count > 0 and max_profit_start_idx < len(kai_data_df) and max_profit_end_idx < len(kai_data_df):
        max_profit_start_time = kai_data_df.iloc[max_profit_start_idx]["timestamp"]
        max_profit_end_time = kai_data_df.iloc[max_profit_end_idx]["timestamp"]
        max_profit_hold_time = kai_data_df.index[max_profit_end_idx] - kai_data_df.index[max_profit_start_idx]
    else:
        max_profit_start_time = max_profit_end_time = max_profit_hold_time = None

    # 计算盈利与亏损相关指标
    profit_df = kai_data_df[kai_data_df["true_profit"] > 0]
    loss_df = kai_data_df[kai_data_df["true_profit"] < 0]
    fu_profit_sum = loss_df["true_profit"].sum()
    fu_profit_mean = round(loss_df["true_profit"].mean() if not loss_df.empty else 0, 4)
    zhen_profit_sum = profit_df["true_profit"].sum()
    zhen_profit_mean = round(profit_df["true_profit"].mean() if not profit_df.empty else 0, 4)
    loss_rate = loss_df.shape[0] / trade_count if trade_count else 0
    loss_time = loss_df["hold_time"].sum() if not loss_df.empty else 0
    profit_time = profit_df["hold_time"].sum() if not profit_df.empty else 0
    loss_time_rate = loss_time / (loss_time + profit_time) if (loss_time + profit_time) else 0

    trade_rate = round(100 * trade_count / total_count, 4) if total_count else 0
    hold_time_mean = kai_data_df["hold_time"].mean() if trade_count else 0

    monthly_groups = kai_data_df["timestamp"].dt.to_period("M")
    monthly_agg = kai_data_df.groupby(monthly_groups)["true_profit"].agg(["sum", "mean", "count"])
    monthly_trade_std = float(monthly_agg["count"].std())
    active_months = monthly_agg.shape[0]
    total_months = 22
    active_month_ratio = active_months / total_months if total_months else 0
    monthly_net_profit_std = float(monthly_agg["sum"].std())
    monthly_avg_profit_std = float(monthly_agg["mean"].std())
    monthly_net_profit_min = monthly_agg["sum"].min()
    monthly_net_profit_max = monthly_agg["sum"].max()
    monthly_loss_rate = (monthly_agg["sum"] < 0).sum() / active_months if active_months else 0

    # 新增指标：每个月净利润和交易个数
    # monthly_net_profit_detail = {str(month): round(val, 4) for month, val in monthly_agg["sum"].to_dict().items()}
    # monthly_trade_count_detail = {str(month): int(val) for month, val in monthly_agg["count"].to_dict().items()}

    hold_time_std = kai_data_df["hold_time"].std()

    # 前10%盈利/亏损的比率计算
    if not profit_df.empty:
        top_profit_count = max(1, int(np.ceil(len(profit_df) * 0.1)))
        profit_sorted = profit_df.sort_values("true_profit", ascending=False)
        top_profit_sum = profit_sorted["true_profit"].iloc[:top_profit_count].sum()
        total_profit_sum = profit_df["true_profit"].sum()
        top_profit_ratio = top_profit_sum / total_profit_sum if total_profit_sum != 0 else 0
    else:
        top_profit_ratio = 0

    if not loss_df.empty:
        top_loss_count = max(1, int(np.ceil(len(loss_df) * 0.1)))
        loss_sorted = loss_df.sort_values("true_profit", ascending=True)
        top_loss_sum = loss_sorted["true_profit"].iloc[:top_loss_count].sum()
        total_loss_sum = loss_df["true_profit"].sum()
        top_loss_ratio = (abs(top_loss_sum) / abs(total_loss_sum)) if total_loss_sum != 0 else 0
    else:
        top_loss_ratio = 0

    statistic_dict = {
        "kai_side": "long" if is_long else "short",
        "kai_column": kai_column,
        "pin_column": pin_column,
        "kai_count": trade_count,
        "total_count": total_count,
        "trade_rate": trade_rate,
        "hold_time_mean": hold_time_mean,
        "hold_time_std": hold_time_std,
        "loss_rate": loss_rate,
        "loss_time_rate": loss_time_rate,
        'zhen_profit_sum': zhen_profit_sum,
        'zhen_profit_mean': zhen_profit_mean,
        'fu_profit_sum': fu_profit_sum,
        'fu_profit_mean': fu_profit_mean,
        "profit_rate": profit_sum,
        "max_profit": max_single_profit,
        "min_profit": min_single_profit,
        "cost_rate": trade_count * 0.07,
        "net_profit_rate": kai_data_df["true_profit"].sum(),
        "fix_profit":fix_profit,
        "avg_profit_rate": round(true_profit_mean, 4),
        "true_profit_std": true_profit_std,
        "max_consecutive_loss": round(max_loss, 4),
        "max_loss_trade_count": loss_trade_count,
        "max_loss_hold_time": max_loss_hold_time,
        "max_loss_start_time": max_loss_start_time,
        "max_loss_end_time": max_loss_end_time,
        "max_consecutive_profit": round(max_profit, 4),
        "max_profit_trade_count": profit_trade_count,
        "max_profit_hold_time": max_profit_hold_time,
        "max_profit_start_time": max_profit_start_time,
        "max_profit_end_time": max_profit_end_time,
        "same_count": same_count,
        "same_count_rate": round(same_count_rate, 4),
        "true_same_count_rate": modification_rate,
        "monthly_trade_std": round(monthly_trade_std, 4),
        "active_month_ratio": round(active_month_ratio, 4),
        "monthly_loss_rate": round(monthly_loss_rate, 4),
        "monthly_net_profit_min": round(monthly_net_profit_min, 4),
        "monthly_net_profit_max": round(monthly_net_profit_max, 4),
        "monthly_net_profit_std": round(monthly_net_profit_std, 4),
        "monthly_avg_profit_std": round(monthly_avg_profit_std, 4),
        "top_profit_ratio": round(top_profit_ratio, 4),
        "top_loss_ratio": round(top_loss_ratio, 4),
        'is_reverse':is_reverse
        # 新增的每月净利润和交易个数的详细数据
        # "monthly_net_profit_detail": monthly_net_profit_detail,
        # "monthly_trade_count_detail": monthly_trade_count_detail
    }
    statistic_dict.update(temp_dict)
    return kai_data_df, statistic_dict


def generate_numbers(start, end, number, even=True):
    if start > end or number <= 0:
        return []
    if number == 1:
        return []
    result = []
    if even:
        step = (end - start) / (number - 1)
        for i in range(number):
            result.append(int(round(start + i * step)))
    else:
        power = 2
        for i in range(number):
            normalized_index = i / (number - 1) if number > 1 else 0
            value = start + (end - start) * (normalized_index ** power)
            result.append(int(round(value)))
    final_result = []
    last_val = None
    for val in result:
        if start <= val <= end and val != last_val:
            final_result.append(val)
            last_val = val
    return final_result[:number]


def process_tasks(task_chunk, df, is_filter):
    start_time = time.time()
    results = []
    for long_column, short_column in task_chunk:
        _, stat_long = get_detail_backtest_result_op(df, long_column, short_column, is_filter)
        _, stat_long_reverse = get_detail_backtest_result_op(df, long_column, short_column, is_filter, is_reverse=True)
        results.append(stat_long)
        results.append(stat_long_reverse)
    print(f"处理 {len(task_chunk)} 个任务，耗时 {time.time() - start_time:.2f} 秒。")
    return results


def gen_ma_signal_name(start_period, end_period, step):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    long_columns = [f"ma_{period}_high_long" for period in period_list]
    short_columns = [f"ma_{period}_low_short" for period in period_list]
    key_name = f'ma_{start_period}_{end_period}_{step}'
    print(f"ma 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}")
    return long_columns, short_columns, key_name


def gen_rsi_signal_name(start_period, end_period, step):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    temp_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    long_columns = [f"rsi_{period}_{overbought}_{100 - overbought}_long"
                    for period in period_list for overbought in temp_list]
    short_columns = [f"rsi_{period}_{overbought}_{100 - overbought}_short"
                     for period in period_list for overbought in temp_list]
    key_name = f'rsi_{start_period}_{end_period}_{step}'
    print(f"rsi 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}")
    return long_columns, short_columns, key_name


def gen_macd_signal_name(start_period, end_period, step):
    period_list = generate_numbers(start_period, end_period, step, even=False)

    # 选择合理的 signal_period 值（通常较小）
    signal_list = [9, 12, 15, 40]  # 预定义 MACD 计算常见的信号周期

    long_columns = [
        f"macd_{fast}_{slow}_{signal}_long"
        for fast in period_list
        for slow in period_list if slow > fast  # 确保 slow_period > fast_period
        for signal in signal_list  # 限制 signal_period 取值
    ]

    short_columns = [
        f"macd_{fast}_{slow}_{signal}_short"
        for fast in period_list
        for slow in period_list if slow > fast  # 确保 slow_period > fast_period
        for signal in signal_list  # 限制 signal_period 取值
    ]

    key_name = f'macd_{start_period}_{end_period}_{step}'
    print(f"MACD 生成了 {len(long_columns)} 个信号列名，优化后减少了无效组合。")

    return long_columns, short_columns, key_name

def gen_donchian_signal_name(start_period, end_period, step):
    period_list = range(start_period, end_period, step)
    long_columns = [f"donchian_{period}_long" for period in period_list]
    short_columns = [f"donchian_{period}_short" for period in period_list]
    key_name = f'donchian_{start_period}_{end_period}_{step}'
    print(f"donchian 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}")
    return long_columns, short_columns, key_name

def gen_atr_signal_name(start_period, end_period, step):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    long_columns = [f"atr_{period}_long" for period in period_list]
    short_columns = [f"atr_{period}_short" for period in period_list]
    key_name = f'atr_{start_period}_{end_period}_{step}'
    print(f"atr 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}")
    return long_columns, short_columns, key_name


def gen_continue_signal_name(start_period, end_period, step):
    period_list = range(start_period, end_period, step)
    long_columns = [f"continue_{period}_high_long" for period in period_list]
    short_columns = [f"continue_{period}_low_short" for period in period_list]
    key_name = f'continue_{start_period}_{end_period}_{step}'
    print(f"continue 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}")
    return long_columns, short_columns, key_name


def gen_abs_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = range(start_period1, end_period1, step1)
    period_list1 = [x / 10 for x in period_list1]
    long_columns = [f"abs_{period}_{period1}_long"
                    for period in period_list for period1 in period_list1 if period >= period1]
    short_columns = [f"abs_{period}_{period1}_short"
                     for period in period_list for period1 in period_list1 if period >= period1]
    key_name = f'abs_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}'
    print(f"abs 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}, {start_period1}, {end_period1}, {step1}")
    return long_columns, short_columns, key_name


def gen_cci_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = range(start_period1, end_period1, step1)
    period_list1 = [x / 10 for x in period_list1]
    long_columns = [f"cci_{period}_{period1}_long"
                    for period in period_list for period1 in period_list1 if period >= period1]
    short_columns = [f"cci_{period}_{period1}_short"
                     for period in period_list for period1 in period_list1 if period >= period1]
    key_name = f'cci_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}'
    print(f"cci 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}, {start_period1}, {end_period1}, {step1}")
    return long_columns, short_columns, key_name

def gen_boll_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = range(start_period1, end_period1, step1)
    period_list1 = [x / 10 for x in period_list1]
    long_columns = [f"boll_{period}_{period1}_long"
                    for period in period_list for period1 in period_list1 if period >= period1]
    short_columns = [f"boll_{period}_{period1}_short"
                     for period in period_list for period1 in period_list1 if period >= period1]
    key_name = f'boll_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}'
    print(f"boll 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}, {start_period1}, {end_period1}, {step1}")
    return long_columns, short_columns, key_name

def gen_relate_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = range(start_period1, end_period1, step1)
    long_columns = [f"relate_{period}_{period1}_long"
                    for period in period_list for period1 in period_list1 if period >= period1]
    short_columns = [f"relate_{period}_{period1}_short"
                     for period in period_list for period1 in period_list1 if period >= period1]
    key_name = f'relate_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}'
    print(f"relate 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}, {start_period1}, {end_period1}, {step1}")
    return long_columns, short_columns, key_name


def gen_macross_signal_name(start_period, end_period, step, start_period1, end_period1, step1):
    period_list = generate_numbers(start_period, end_period, step, even=False)
    period_list1 = generate_numbers(start_period1, end_period1, step1, even=False)
    long_columns = [f"macross_{period}_{period1}_long"
                    for period in period_list for period1 in period_list1]
    short_columns = [f"macross_{period}_{period1}_short"
                     for period in period_list for period1 in period_list1]
    key_name = f'macross_{start_period}_{end_period}_{step}_{start_period1}_{end_period1}_{step1}'
    print(f"macross 一共生成 {len(long_columns)} 个信号列名。参数: {start_period}, {end_period}, {step}, {start_period1}, {end_period1}, {step1}")
    return long_columns, short_columns, key_name


def worker_func(args):
    chunk, df, is_filter = args
    return process_tasks(chunk, df, is_filter)


def init_worker(precomputed_signals):
    """
    进程池初始化函数，将预计算的稀疏信号数据设置为各进程的全局变量。
    """
    global GLOBAL_SIGNALS
    GLOBAL_SIGNALS = precomputed_signals


def init_worker1(dataframe):
    """
    在每个子进程中初始化全局变量 df，使得 compute_signal 能够访问到它。
    """
    global df
    df = dataframe

def process_signal(sig):
    """
    针对单个信号进行计算：
      - 调用 compute_signal 函数获得 s, p
      - 将 s, p 转换为 numpy 数组
      - 获取非零索引，如果数量不足 100 个则跳过
      - 返回 (sig, (indices, 对应的 p 值)) 的结果
    如果计算出错，则打印错误信息并返回 None。
    """
    try:
        s, p = compute_signal(df, sig)
        s_np = s.to_numpy(copy=False) if hasattr(s, "to_numpy") else np.asarray(s)
        p_np = p.to_numpy(copy=False) if hasattr(p, "to_numpy") else np.asarray(p)
        if p_np.dtype == np.float64:
            p_np = p_np.astype(np.float32)
        indices = np.nonzero(s_np)[0]
        if indices.size < 100:
            return None
        return (sig, (indices.astype(np.int32), p_np[indices]))
    except Exception as e:
        print(f"预计算 {sig} 时出错：{e}")
        return None

def process_batch(sig_batch):
    """
    针对一批信号进行处理：
      遍历列表并调用 process_signal, 如果返回非 None 则加入结果列表
    """
    batch_results = []
    for sig in sig_batch:
        res = process_signal(sig)
        if res is not None:
            batch_results.append(res)
    return batch_results


def generate_all_signals():
    """
    生成所有信号，分别调用各个信号生成函数，然后合并生成总信号列表及标识 key_name。
    返回:
      all_signals: 所有信号（列表）
      key_name   : 信号标识，用于后续文件命名
    """
    column_list = []

    abs_long_columns, abs_short_columns, abs_key_name = gen_abs_signal_name(1, 1000, 25, 1, 40, 2)
    column_list.append((abs_long_columns, abs_short_columns, abs_key_name))

    relate_long_columns, relate_short_columns, relate_key_name = gen_relate_signal_name(1, 1000, 50, 1, 30, 3)
    column_list.append((relate_long_columns, relate_short_columns, relate_key_name))

    donchian_long_columns, donchian_short_columns, donchian_key_name = gen_donchian_signal_name(1, 20, 1)
    column_list.append((donchian_long_columns, donchian_short_columns, donchian_key_name))

    boll_long_columns, boll_short_columns, boll_key_name = gen_boll_signal_name(1, 1000, 25, 1, 40, 2)
    column_list.append((boll_long_columns, boll_short_columns, boll_key_name))

    macross_long_columns, macross_short_columns, macross_key_name = gen_macross_signal_name(1, 1000, 20, 1, 1000, 25)
    column_list.append((macross_long_columns, macross_short_columns, macross_key_name))

    rsi_long_columns, rsi_short_columns, rsi_key_name = gen_rsi_signal_name(1, 1000, 50)
    column_list.append((rsi_long_columns, rsi_short_columns, rsi_key_name))

    macd_long_columns, macd_short_columns, macd_key_name = gen_macd_signal_name(1, 1000, 15)
    column_list.append((macd_long_columns, macd_short_columns, macd_key_name))

    cci_long_columns, cci_short_columns, cci_key_name = gen_cci_signal_name(1, 1000, 25, 1, 40, 2)
    column_list.append((cci_long_columns, cci_short_columns, cci_key_name))

    atr_long_columns, atr_short_columns, atr_key_name = gen_atr_signal_name(1, 1000, 100)
    column_list.append((atr_long_columns, atr_short_columns, atr_key_name))

    # 按多头信号数量升序排序
    column_list = sorted(column_list, key=lambda x: len(x[0]))
    all_signals = []
    key_name = ''
    for long_columns, short_columns, temp_key_name in column_list:
        temp = long_columns + short_columns
        key_name += temp_key_name + '_'
        all_signals.extend(temp)
    return all_signals, key_name


def split_into_batches(signal_list, batch_size):
    """
    按照指定的 batch_size 将 signal_list 划分为若干批次。
    """
    return [signal_list[i:i + batch_size] for i in range(0, len(signal_list), batch_size)]


def compute_precomputed_for_batch(signals, df):
    """
    计算并返回给定信号列表 signals 的预计算结果。
    将 signals 划分为小块（每10个信号）并利用多进程并行计算。
    """
    num_workers = multiprocessing.cpu_count()
    sub_batches = [signals[k:k + 10] for k in range(0, len(signals), 10)]
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker1, initargs=(df,)) as pool:
        sub_results = pool.map(process_batch, sub_batches)
    precomputed = {}
    for sub_res in sub_results:
        for sig, data in sub_res:
            precomputed[sig] = data
    return precomputed


def load_or_compute_batch(batch_index, signals, df, base_name, key_name):
    """
    根据文件是否存在，按批次加载或计算预计算结果。
    如果 pickle 文件存在则加载，否则计算后保存到磁盘。
    """
    batch_file = os.path.join("temp", f"{base_name}_{key_name}_batch_{batch_index}.pkl")
    if os.path.exists(batch_file):
        try:
            with open(batch_file, "rb") as f:
                precomputed = pickle.load(f)
            print(f"加载批次 {batch_index} 的预计算结果，信号数：{len(precomputed)}。")
            return precomputed
        except Exception as e:
            print(f"加载批次 {batch_index} 失败：{e}")
    print(f"开始计算批次 {batch_index} 的预计算结果：共 {len(signals)} 个信号。")
    start = time.time()
    precomputed = compute_precomputed_for_batch(signals, df)
    elapsed = time.time() - start
    print(f"批次 {batch_index} 完成预计算，共 {len(precomputed)} 个信号，耗时 {elapsed:.2f} 秒。")
    try:
        with open(batch_file, "wb") as f:
            pickle.dump(precomputed, f)
        print(f" {batch_index} 的预计算结果到 {batch_file}。")
    except Exception as e:
        print(f" {batch_index} 时出错：{e}")
    return precomputed


def get_precomputed(batch_index, batches, df, base_name, key_name, cache):
    """
    利用缓存按需加载或计算指定批次的预计算结果。
    """
    if batch_index not in cache:
        signals = batches[batch_index]
        cache[batch_index] = load_or_compute_batch(batch_index, signals, df, base_name, key_name)
    return cache[batch_index]


def process_batch_pair(i, j, batches, df, is_filter, base_name, key_name, pool_processes, batch_cache):
    """
    针对批次 i 与批次 j 的预计算结果生成任务，并按照大块（100000 个任务为一块）
    分块处理任务，再进一步小块拆分后利用多进程计算，最后保存结果到 CSV 文件中。

    对于批次组合：
      - 加载对应批次的预计算数据（按需加载，不全部加载到内存）。
      - 生成任务列表：即两个批次内信号的笛卡尔积，保证任务数量与直接对全信号计算一致。
      - 任务列表首先按照 100000 任务为一大块分块，再进一步细分为更小的块传入 worker 进行计算。
    """
    # 加载批次预计算结果（利用缓存）
    precomputed_i = get_precomputed(i, batches, df, base_name, key_name, batch_cache)
    if i == j:
        precomputed_j = precomputed_i
    else:
        precomputed_j = get_precomputed(j, batches, df, base_name, key_name, batch_cache)

    # 供 worker 使用的预计算数据（如果 i != j 则合并两个字典）
    combined_precomputed = precomputed_i if i == j else {**precomputed_i, **precomputed_j}

    total_size = sys.getsizeof(combined_precomputed)  # 计算字典对象本身的大小

    for sig, (s_np, p_np) in combined_precomputed.items():
        total_size += sys.getsizeof(sig)  # 计算键的大小（字符串）
        total_size += s_np.nbytes  # NumPy 数组的实际数据大小
        total_size += p_np.nbytes  # NumPy 数组的实际数据大小
    max_memory = 64
    # 以 MB 为单位打印内存占用
    print(f"precomputed_signals 占用内存总大小: {total_size / (1024 * 1024):.2f} MB")
    pool_processes = min(pool_processes, int(max_memory * 1024 * 1024 * 1024 / total_size))  # 限制进程数不超过 CPU 核心数
    print(f"进程数限制为 {pool_processes}，根据内存限制调整。")
    list_i = list(precomputed_i.keys())
    list_j = list(precomputed_j.keys())
    total_tasks = len(list_i) * len(list_j)
    print(f"处理批次对 ({i}, {j})：批次 {i} 信号数 {len(list_i)}，批次 {j} 信号数 {len(list_j)}，任务总数 {total_tasks}。")

    # 构造任务列表：所有信号对（有序）
    task_list = list(product(list_i, list_j))

    # 按照大块方式分块：每 100000 个任务为一块
    BIG_CHUNK_SIZE = 100000
    big_chunks = [task_list[k:k + BIG_CHUNK_SIZE] for k in range(0, len(task_list), BIG_CHUNK_SIZE)]

    # 针对每个大块进一步细分为小块后调用 worker_func
    for chunk_index, big_chunk in enumerate(big_chunks):
        start_time = time.time()
        output_path = os.path.join("temp", f"{base_name}_{key_name}_{i}_{j}_{chunk_index}.csv")
        if os.path.exists(output_path):
            print(f"{output_path} 批次对 ({i}, {j}) 大块 {chunk_index + 1} 结果文件已存在，跳过。")
            continue
        np.random.shuffle(big_chunk)

        # 根据当前大块任务数计算小块大小，保证每个小块至少有 50 个任务，
        # 同时根据 pool_processes 动态调整小块数（这里用 pool_processes*12 拆分小块）
        small_chunk_size = max(50, int(np.ceil(len(big_chunk) / (pool_processes * 12))))
        small_chunks = [big_chunk[l:l + small_chunk_size] for l in range(0, len(big_chunk), small_chunk_size)]
        print(f'当前时间 {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}  批次对 ({i}, {j}) - 大块 {chunk_index + 1}/{len(big_chunks)}：任务数 {len(big_chunk)}，拆分为 {len(small_chunks)} 个小块（每块约 {small_chunk_size} 个任务）。')

        statistic_dict_list = []
        # 多进程处理当前大块内的小块任务
        with multiprocessing.Pool(processes=pool_processes - 3, initializer=init_worker,
                                  initargs=(combined_precomputed,)) as pool:
            tasks_args = [(chunk, df, is_filter) for chunk in small_chunks]
            for result in pool.imap_unordered(worker_func, tasks_args, chunksize=1):
                statistic_dict_list.extend(result)
        statistic_dict_list = [x for x in statistic_dict_list if x is not None]

        # 保存当前大块的结果到 CSV 文件
        pd.DataFrame(statistic_dict_list).to_csv(output_path, index=False)
        print(f'当前时间 {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}  批次对 ({i}, {j}) 大块 {chunk_index + 1} 完成，耗时 {time.time() - start_time:.2f} 秒 {len(small_chunks)} 小块任务处理。结果保存至：{output_path}')

    # 可在合适时清理缓存，降低内存占用（示例中仅清理当前批次 i）
    if i in batch_cache:
        batch_cache.pop(i)


##########################################
# 主函数
##########################################
def backtest_breakthrough_strategy(df, base_name, is_filter, batch_size=500):
    """
    回测函数（重构版）：
      1. 依据不同策略函数生成各个信号，并合并为一个 all_signals 列表，同时构造 key_name 标识；
      2. 将所有信号按照 batch_size 进行分批，每批预计算结果独立保存（按需加载，降低内存压力）；
      3. 对所有批次两两组合，生成任务列表（即两个批次信号的笛卡尔积），任务总数与直接对所有信号求笛卡尔积完全一致；
      4. 对每个批次对的任务列表，先按照大块（100000 个任务为一大块）分块，再根据进程数细分为子块，
         然后进程池并行计算，结果保存为 CSV 文件。

    参数:
      df         : 参与计算的 DataFrame 数据
      base_name  : 文件保存的基础名称
      is_filter  : 回测时的过滤参数
      batch_size : 每个预计算批次内包含信号的数量（默认 500）
    """
    # 创建临时目录
    os.makedirs("temp", exist_ok=True)

    # 1. 生成所有信号和标识字符串
    all_signals, key_name = generate_all_signals()
    print(f"生成所有信号，共 {len(all_signals)} 个，信号标识: {key_name}")

    # 2. 按批次划分信号
    batches = split_into_batches(all_signals, batch_size)
    total_batches = len(batches)
    print(f"将信号划分为 {total_batches} 个批次，每批最多 {batch_size} 个信号。")

    # 3. 设置多进程使用的进程数
    pool_processes = max(1, multiprocessing.cpu_count())
    print(f"使用 {pool_processes} 个 CPU 核心。")

    # 使用缓存降低重复加载同一批次的数据
    batch_cache = {}
    print('预估的批次对数量为：', total_batches * total_batches)
    print('预估总体的任务数量为：', len(all_signals) * len(all_signals))
    # 4. 遍历所有批次两两组合，依次处理任务
    for i in range(total_batches):
        for j in range(total_batches):
            print(f"\n============== 开始处理批次对 ({i}, {j}) ==============")
            process_batch_pair(i, j, batches, df, is_filter, base_name, key_name, pool_processes, batch_cache)

    print("\n所有批次组合任务计算完成。")


def gen_breakthrough_signal(data_path='temp/TON_1m_2000.csv'):
    """
    主函数：
      1. 加载 CSV 数据（保留 timestamp, open, high, low, close）；
      2. 转换所有价格为 float 类型；
      3. 计算涨跌幅 chg，过滤月初与月末数据，然后启动回测。
    """
    base_name = os.path.basename(data_path)
    # 去除base_name中的-USDT-SWAP.csv
    base_name = base_name.replace('-USDT-SWAP.csv', '')
    base_name = base_name.replace('origin_data_', '')
    is_filter = True
    df = pd.read_csv(data_path)
    needed_columns = ['timestamp', 'high', 'low', 'close']
    df = df[needed_columns]
    jingdu = 'float32'

    df['chg'] = (df['close'].pct_change() * 100).astype('float16')
    df['high'] = df['high'].astype(jingdu)
    df['low'] = df['low'].astype(jingdu)
    df['close'] = df['close'].astype(jingdu)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_monthly = df['timestamp'].dt.to_period('M')
    min_df_month = df_monthly.min()
    max_df_month = df_monthly.max()
    df = df[(df_monthly != min_df_month) & (df_monthly != max_df_month)]
    print(f'开始回测 {base_name} ... 数据长度 {df.shape[0]} 当前时间 {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    backtest_breakthrough_strategy(df, base_name, is_filter)


def example():
    start_time = time.time()
    data_path_list = [
        'kline_data/origin_data_1m_10000000_SOL-USDT-SWAP.csv',
        'kline_data/origin_data_1m_10000000_BTC-USDT-SWAP.csv',
        # 'kline_data/origin_data_1m_10000000_ETH-USDT-SWAP.csv',
        # 'kline_data/origin_data_1m_10000000_TON-USDT-SWAP.csv',
        # 'kline_data/origin_data_1m_10000000_DOGE-USDT-SWAP.csv',
        # 'kline_data/origin_data_1m_10000000_XRP-USDT-SWAP.csv',
        # 'kline_data/origin_data_1m_10000000_PEPE-USDT-SWAP.csv',
    ]
    for data_path in data_path_list:
        try:
            gen_breakthrough_signal(data_path)
            print(f'{data_path} 总耗时 {time.time() - start_time:.2f} 秒。')
        except Exception as e:
            traceback.print_exc()
            print(f'处理 {data_path} 时出错：{e}')


if __name__ == '__main__':
    example()