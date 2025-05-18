import math
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd

def compute_signal(df, col_name):
    """
    根据历史行情数据(df)和指定信号名称(col_name)生成交易信号。
    支持的信号类型包括：abs, relate, donchian, boll, macross, rsi, macd, cci, atr。
    """
    parts = col_name.split("_")
    signal_type = parts[0]
    direction = parts[-1]

    if signal_type == "abs":
        period = int(parts[1])
        abs_value = float(parts[2]) / 100
        if direction == "long":
            min_low = df["low"].shift(1).rolling(period).min()
            target_price = (min_low * (1 + abs_value)).round(4)
            signal_series = df["high"] > target_price
        else:
            max_high = df["high"].shift(1).rolling(period).max()
            target_price = (max_high * (1 - abs_value)).round(4)
            signal_series = df["low"] < target_price

        valid_trade = (target_price >= df["low"]) & (target_price <= df["high"])
        signal_series = signal_series & valid_trade
        trade_price_series = target_price
        # 可选记录调试数据
        df["target_price"] = target_price
        df["signal_series"] = signal_series
        df["trade_price_series"] = trade_price_series
        return signal_series, trade_price_series

    elif signal_type == "relate":
        period = int(parts[1])
        percent = float(parts[2]) / 100
        min_low = df["low"].shift(1).rolling(period).min()
        max_high = df["high"].shift(1).rolling(period).max()
        if direction == "long":
            target_price = (min_low + percent * (max_high - min_low)).round(4)
            signal_series = df["high"] > target_price
        else:
            target_price = (max_high - percent * (max_high - min_low)).round(4)
            signal_series = df["low"] < target_price

        valid_trade = (target_price >= df["low"]) & (target_price <= df["high"])
        signal_series = signal_series & valid_trade
        return signal_series, target_price

    elif signal_type == "donchian":
        period = int(parts[1])
        if direction == "long":
            highest_high = df["high"].shift(1).rolling(period).max()
            signal_series = df["high"] > highest_high
            target_price = highest_high
        else:
            lowest_low = df["low"].shift(1).rolling(period).min()
            signal_series = df["low"] < lowest_low
            target_price = lowest_low
        valid_trade = (target_price >= df["low"]) & (target_price <= df["high"])
        signal_series = signal_series & valid_trade
        trade_price_series = target_price.round(4)
        return signal_series, trade_price_series

    elif signal_type == "boll":
        period = int(parts[1])
        std_multiplier = float(parts[2])
        ma = df["close"].rolling(window=period, min_periods=period).mean()
        std_dev = df["close"].rolling(window=period, min_periods=period).std()
        upper_band = (ma + std_multiplier * std_dev).round(4)
        lower_band = (ma - std_multiplier * std_dev).round(4)
        if direction == "long":
            signal_series = (df["close"].shift(1) < lower_band.shift(1)) & (df["close"] >= lower_band)
        else:
            signal_series = (df["close"].shift(1) > upper_band.shift(1)) & (df["close"] <= upper_band)
        return signal_series, df["close"]

    elif signal_type == "macross":
        fast_period = int(parts[1])
        slow_period = int(parts[2])
        fast_ma = df["close"].rolling(window=fast_period, min_periods=fast_period).mean().round(4)
        slow_ma = df["close"].rolling(window=slow_period, min_periods=slow_period).mean().round(4)
        if direction == "long":
            signal_series = (fast_ma.shift(1) < slow_ma.shift(1)) & (fast_ma >= slow_ma)
        else:
            signal_series = (fast_ma.shift(1) > slow_ma.shift(1)) & (fast_ma <= slow_ma)
        return signal_series, df["close"]

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
            signal_series = (rsi.shift(1) < oversold) & (rsi >= oversold)
        else:
            signal_series = (rsi.shift(1) > overbought) & (rsi <= overbought)
        return signal_series, df["close"]

    elif signal_type == "macd":
        fast_period, slow_period, signal_period = map(int, parts[1:4])
        fast_ema = df["close"].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df["close"].ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        if direction == "long":
            signal_series = (macd_line.shift(1) < signal_line.shift(1)) & (macd_line >= signal_line)
        else:
            signal_series = (macd_line.shift(1) > signal_line.shift(1)) & (macd_line <= signal_line)
        return signal_series, df["close"]

    elif signal_type == "cci":
        period = int(parts[1])
        tp = (df["high"] + df["low"] + df["close"]) / 3
        ma = tp.rolling(period).mean()
        md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        cci = (tp - ma) / (0.015 * md)
        if direction == "long":
            signal_series = (cci.shift(1) < -100) & (cci >= -100)
        else:
            signal_series = (cci.shift(1) > 100) & (cci <= 100)
        return signal_series, df["close"]

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
            signal_series = (atr.shift(1) < atr_ma.shift(1)) & (atr >= atr_ma)
        else:
            signal_series = (atr.shift(1) > atr_ma.shift(1)) & (atr <= atr_ma)
        return signal_series, df["close"]

    else:
        raise ValueError(f"未知信号类型: {signal_type}")


def compute_price_range_abs(df, col_name):

    parts = col_name.split("_")
    if len(parts) < 4:
        return None

    signal_type = parts[0]
    try:
        period = int(parts[1])
        abs_value = float(parts[2]) / 100
    except Exception:
        return None

    direction = parts[-1]
    if signal_type != "abs":
        return None  # 目前只支持 abs 类型

    # 对下一根K线计算使用历史数据窗口：最近 period 行
    if len(df) < period:
        return None
    latest_close = df.iloc[-1]["close"]

    if direction == "long":
        window_min = df["low"].tail(period).min()
        target_price = (window_min * (1 + abs_value))
        if latest_close < target_price:
            return (target_price, None)
    elif direction == "short":
        window_max = df["high"].tail(period).max()
        target_price = (window_max * (1 - abs_value))
        if latest_close > target_price:
            return (None, target_price)
    else:
        return None
    return None


def validate_signals_abs(
        df: pd.DataFrame,
        col_name: str,
        verbose: bool = True
    ):
    max_samples = 10000
    period = int(col_name.split("_")[1])

    # 1. 得到基准信号
    sig = compute_signal(df, col_name)[0].fillna(False)

    # 2. 分组取样
    true_idx  = sig[sig].index
    false_idx = sig[~sig].index

    rng = np.random.default_rng(42)   # 固定随机种子，方便复现
    sample_true  = rng.choice(true_idx,  size=min(max_samples,  len(true_idx)),  replace=False) if len(true_idx)  else []
    sample_false = rng.choice(false_idx, size=min(max_samples,  len(false_idx)), replace=False) if len(false_idx) else []
    sample_idx   = np.concatenate([sample_true, sample_false])

    mismatch = []

    for i in sample_idx:
        # compute_price_range_abs 只用到历史数据 => 取 i 之前所有行
        if i < period:        # 滚动窗口不足，跳过
            continue
        hist_df = df.iloc[:i]  # 不含当前行 i
        res = compute_price_range_abs(hist_df, col_name)

        if res is None:
            continue  # 理论上不会发生，保险起见

        price_high = df.loc[i, "high"]
        price_low  = df.loc[i, "low"]

        direction = col_name.split("_")[-1]
        if direction == "long":
            match = price_high >= res[0]
        else:  # short
            match = price_low  <= res[1]

        if match != sig.loc[i]:
            mismatch.append(int(i))
            if verbose:
                print(f"[Mismatch] idx={i}, "
                      f"expected {sig.loc[i]}, "
                      f"price_high={price_high:.4f}, price_low={price_low:.4f}, "
                      f"threshold={res}")

    total_checked = len(sample_idx)
    accuracy = 1 - len(mismatch) / total_checked if total_checked else 1.0

    if verbose:
        print(f"\nChecked {total_checked} samples "
              f"({len(sample_true)} True, {len(sample_false)} False)")
        print(f"Accuracy: {accuracy:.2%}  |  mismatches: {len(mismatch)}")

    # 返回结果，True表示验证通过
    return accuracy == 1.0


def validate_signals_relate(
        df: pd.DataFrame,
        col_name: str,
        verbose: bool = True
    ):
    max_samples = 10000
    period = int(col_name.split("_")[1])

    # 1. 得到基准信号
    sig = compute_signal(df, col_name)[0].fillna(False)

    # 2. 分组取样
    true_idx  = sig[sig].index
    false_idx = sig[~sig].index

    rng = np.random.default_rng(42)   # 固定随机种子，方便复现
    sample_true  = rng.choice(true_idx,  size=min(max_samples,  len(true_idx)),  replace=False) if len(true_idx)  else []
    sample_false = rng.choice(false_idx, size=min(max_samples,  len(false_idx)), replace=False) if len(false_idx) else []
    sample_idx   = np.concatenate([sample_true, sample_false])

    mismatch = []

    for i in sample_idx:
        # compute_price_range_abs 只用到历史数据 => 取 i 之前所有行
        if i < period:        # 滚动窗口不足，跳过
            continue
        hist_df = df.iloc[:i]  # 不含当前行 i
        res = compute_price_range_abs(hist_df, col_name)

        if res is None:
            continue  # 理论上不会发生，保险起见

        price_high = df.loc[i, "high"]
        price_low  = df.loc[i, "low"]

        direction = col_name.split("_")[-1]
        if direction == "long":
            match = price_high >= res[0]
        else:  # short
            match = price_low  <= res[1]

        if match != sig.loc[i]:
            mismatch.append(int(i))
            if verbose:
                print(f"[Mismatch] idx={i}, "
                      f"expected {sig.loc[i]}, "
                      f"price_high={price_high:.4f}, price_low={price_low:.4f}, "
                      f"threshold={res}")

    total_checked = len(sample_idx)
    accuracy = 1 - len(mismatch) / total_checked if total_checked else 1.0

    if verbose:
        print(f"\nChecked {total_checked} samples "
              f"({len(sample_true)} True, {len(sample_false)} False)")
        print(f"Accuracy: {accuracy:.2%}  |  mismatches: {len(mismatch)}")

    # 返回结果，True表示验证通过
    return accuracy == 1.0

def validate_signals_macd(df, col_name):
    """
    根据 compute_signal 的结果进行验证，包括：
      1. 对于所有信号为 True 的行，验证该行的收盘价满足由前一根数据计算得到的价格触发条件，
      2. 验证最后50行的信号都是 False。

    参数:
        df: 包含历史数据的 DataFrame，必须包含 'close' 列。
        col_name: 信号指标名称，例如 "macd_12_26_9_long" 或 "macd_12_26_9_short"。

    返回:
        如果所有验证均通过，则返回 True；否则返回 False。
    """
    # 先计算整个 DataFrame 的信号序列，作为最终的信号标记标准
    signal_series, _ = compute_signal(df, col_name)
    count = 0
    valid = True
    # 从 col_name 中提取信号方向，用于后续判断
    direction = col_name.split("_")[-1]

    # 从索引 1 开始遍历（依赖于 shift(1) 的数据，第0行没有前置数据）
    for i in range(1, len(df)):
        # 对于第 i 行，模拟“实盘”情况：只有 df.iloc[:i] 的历史数据可用
        hist_df = df.iloc[:i]
        # 调用 compute_signal_range 得到下一根 K 线需要满足的价格范围

        current_price = df.iloc[i]["close"]

        if signal_series.iloc[i]:
            range_result = compute_signal_range(hist_df, col_name)

            # 如果本行信号为 True，意味着历史数据满足前置条件且实际的收盘价应触发信号
            if range_result is None:
                print(f"Index {i}: Signal is True but compute_signal_range returned None.")
                valid = False
            else:
                lower, upper = range_result
                if direction == "long":
                    # long 信号要求：如果 lower_bound 存在，则收盘价必须 >= lower_bound
                    if lower is not None and current_price < lower:
                        print(f"Index {i}: long signal is True but close price {current_price} < lower bound {lower}.")
                        valid = False
                elif direction == "short":
                    # short 信号要求：如果 upper_bound 存在，则收盘价必须 <= upper_bound
                    if upper is not None and current_price > upper:
                        print(f"Index {i}: short signal is True but close price {current_price} > upper bound {upper}.")
                        valid = False
        else:
            range_result = compute_signal_range(hist_df, col_name)

            count += 1
            if count > 50:
                # 仅验证最后50行中标记为 False 的行
                continue
            # 对于信号为 False 的情况：
            # ① 如果 compute_signal_range 返回 None，则说明历史数据未满足触发条件，此时信号为 False 是合理的；
            # ② 如果返回了价格区间，说明历史上处于转换前置状态，但新K线未满足价格条件，则要求其不应满足触发要求：
            if range_result is not None:
                lower, upper = range_result
                if direction == "long":
                    # long 信号触发条件是：close >= lower_bound
                    if lower is not None and current_price >= lower:
                        print(
                            f"Index {i}: long signal is False but close price {current_price} >= lower bound {lower}.")
                        valid = False
                elif direction == "short":
                    # short 信号触发条件是：close <= upper_bound
                    if upper is not None and current_price <= upper:
                        print(
                            f"Index {i}: short signal is False but close price {current_price} <= upper bound {upper}.")
                        valid = False

    return valid

def compute_price_range_macd(df: pd.DataFrame, col_name: str):
    """
    根据 MACD 指标计算使下一个 K 线触发信号的价格范围.

    信号格式: "macd_fast_slow_signal_long" 或 "macd_fast_slow_signal_short"

    对于多头信号:
      - 要求上一周期 (macd_prev < signal_prev)，并返回 (lower_bound, None)
        表示只要收盘价 >= lower_bound 就能触发信号.
    对于空头信号:
      - 要求上一周期 (macd_prev > signal_prev)，并返回 (None, upper_bound)
        表示只要收盘价 <= upper_bound 就能触发信号.

    如果条件不满足或者参数错误，则返回 None.
    """
    parts = col_name.split("_")
    if parts[0] != "macd":
        return None

    direction = parts[-1]
    try:
        fast_period, slow_period, signal_period = map(int, parts[1:4])
    except Exception:
        return None

    # 计算 EMA 序列
    fast_ema = df["close"].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df["close"].ewm(span=slow_period, adjust=False).mean()
    macd_series = fast_ema - slow_ema
    signal_series = macd_series.ewm(span=signal_period, adjust=False).mean()

    fast_ema_prev = fast_ema.iloc[-1]
    slow_ema_prev = slow_ema.iloc[-1]
    macd_prev = macd_series.iloc[-1]
    signal_prev = signal_series.iloc[-1]

    # 计算 EMA 平滑系数
    alpha_fast = 2 / (fast_period + 1)
    alpha_slow = 2 / (slow_period + 1)
    # alpha_signal 并非直接用于定价计算，但保留计算记录便于理解
    alpha_signal = 2 / (signal_period + 1)

    # 对于新数据，假设下一个收盘价为 p，则
    # new_fast_ema = p * alpha_fast + fast_ema_prev * (1 - alpha_fast)
    # new_slow_ema = p * alpha_slow + slow_ema_prev * (1 - alpha_slow)
    # new_macd = new_fast_ema - new_slow_ema = C * p + D
    C = alpha_fast - alpha_slow
    D = fast_ema_prev * (1 - alpha_fast) - slow_ema_prev * (1 - alpha_slow)

    if direction == "long":
        if not (macd_prev < signal_prev):
            return None
        if C > 0:
            lower_bound = (signal_prev - D) / C
            return (lower_bound, None)
        elif C < 0:
            upper_bound = (signal_prev - D) / C
            return (None, upper_bound)
        else:  # C == 0
            if D >= signal_prev:
                return (None, None)
            else:
                return None

    elif direction == "short":
        if not (macd_prev > signal_prev):
            return None
        if C > 0:
            upper_bound = (signal_prev - D) / C
            return (None, upper_bound)
        elif C < 0:
            lower_bound = (signal_prev - D) / C
            return (lower_bound, None)
        else:  # C == 0
            if D <= signal_prev:
                return (None, None)
            else:
                return None
    else:
        return None


def compute_price_range_relate(df: pd.DataFrame, col_name: str):
    """
    快速给出“下一根 K 线若想触发该信号，可出现的价格区间”。

    返回
        (lower_bound, upper_bound)
        - lower_bound 为 None 表示区间下界为 -∞
        - upper_bound 为 None 表示区间上界为  +∞
        - 若历史长度不足或无法计算则返回 None
    """
    p = col_name.split('_')
    if p[0] != 'relate':
        raise ValueError('只支持 relate_*_*_long/short')

    period = int(p[1])
    percent = float(p[2]) / 100
    direc = p[-1].lower()

    if len(df) < period:  # 数据不足
        return None

    # 不再 shift(1) —— 因为要给“下一根”用
    min_low = df['low'].rolling(period).min().iloc[-1]
    max_high = df['high'].rolling(period).max().iloc[-1]

    latest_close = df['close'].iloc[-1]
    if np.isnan(min_low) or np.isnan(max_high):
        return None

    if direc == 'long':
        target = round(min_low + percent * (max_high - min_low), 4)
        if latest_close < target:
            return (target, None)
    else:  # short
        target = round(max_high - percent * (max_high - min_low), 4)
        if latest_close > target:
            return (None, target)
    return None


def compute_price_range_boll(df, col_name):
    """
    对于 boll 指标信号，计算下一根K线需要满足的价格范围，使得信号条件为真。

    参数:
      df: 包含历史数据的 DataFrame，必须含有 'close' 列
      col_name: 信号名称, 格式形如 "boll_20_2_long" 或 "boll_20_2_short"

    返回值:
      对于 long 信号，返回 (price_lower_bound, None) 表示 x 必须 ≥ price_lower_bound；
      对于 short 信号，返回 (None, price_upper_bound) 表示 x 必须 ≤ price_upper_bound；
      若条件无法满足，返回 None。

    说明:
      假设滚动窗口长度 period 中，下一根K线的 rolling 窗口由最近 (period-1) 根收盘价加上新价格 x 计算得出。
    """
    parts = col_name.split("_")
    if len(parts) < 3:
        raise ValueError("col_name 格式不正确，应为 boll_{period}_{multiplier}_{direction}")

    signal_type = parts[0]
    if signal_type != "boll":
        raise ValueError("目前仅支持 boll 信号")

    try:
        period = int(parts[1])
        k = float(parts[2])
    except Exception as e:
        raise ValueError("col_name 中 period 或 multiplier 格式错误") from e

    if parts[-1] not in ["long", "short"]:
        raise ValueError("方向需为 long 或 short")

    direction = parts[-1]

    # 至少需要 period 个数据来计算上一根完整的 Bollinger 带
    if len(df) < period:
        return None

    # 计算历史 boll 带（rolling计算），取最新一根数据的结果
    ma_series = df["close"].rolling(window=period, min_periods=period).mean()
    std_series = df["close"].rolling(window=period, min_periods=period).std()
    upper_band = (ma_series + k * std_series).round(4)
    lower_band = (ma_series - k * std_series).round(4)

    prev_close = df["close"].iloc[-1]

    if direction == "long":
        # 多头信号要求上根收盘价低于其下轨
        if not (prev_close < lower_band.iloc[-1]):
            return None
    else:  # short
        # 空头信号要求上根收盘价高于其上轨
        if not (prev_close > upper_band.iloc[-1]):
            return None

    # 计算下一根K线滚动窗口：最近 (period-1) 个收盘价
    window = df["close"].iloc[-(period - 1):]
    A_last = window.sum()  # 和
    T_last = (window ** 2).sum()  # 平方和
    n = period

    # 利用先前推导，构造二次方程的系数（两种情况实际上解同一个二次方程，
    # 但选择解时依据 long 对 x 应 ≤ A_last/(n-1)，而 short 要 x ≥ A_last/(n-1)）
    A_coef = (n - 1) * ((n - 1) ** 2 - k ** 2 * n)
    B_coef = -2 * A_last * (n - 1) ** 2 + 2 * k ** 2 * A_last * n
    C_coef = A_last ** 2 * ((n - 1) + k ** 2 * n) - k ** 2 * (n ** 2) * T_last

    # 如果 A_coef 非零则用二次公式，否则退化成线性问题
    if A_coef == 0:
        if B_coef == 0:
            # 无法确定
            return None
        x_fixed = -C_coef / B_coef
        valid_sol = x_fixed
    else:
        disc = B_coef ** 2 - 4 * A_coef * C_coef
        if disc < 0:
            return None
        sol1 = (-B_coef + np.sqrt(disc)) / (2 * A_coef)
        sol2 = (-B_coef - np.sqrt(disc)) / (2 * A_coef)
        valid_sol = None
        # 对于 long 信号，要求解满足 x <= A_last/(n-1)
        ref = A_last / (n - 1)
        if direction == "long":
            for sol in [sol1, sol2]:
                if sol is not None and sol <= ref:
                    valid_sol = sol
                    break
        else:  # short: 要求解满足 x >= A_last/(n-1)
            for sol in [sol1, sol2]:
                if sol is not None and sol >= ref:
                    valid_sol = sol
                    break

        if valid_sol is None:
            return None

    # 最终返回的范围：对于 long，下一根价格必须 >= 临界价；对于 short，必须<= 临界价
    valid_sol = round(valid_sol, 4)
    if direction == "long":
        return (valid_sol, None)
    else:
        return (None, valid_sol)


def validate_signals_boll(
    df: pd.DataFrame,
    col_name: str,
    max_per_class: int = 100000,
    random_state: int = 0,
):
    """
    验证 compute_price_range_boll 对给定 boll 信号列的计算是否正确。

    参数
    ----
    df : pd.DataFrame
        必须包含 'close' 列的 K 线数据，索引可以是日期或整数。
    col_name : str
        形如 'boll_20_2_long' / 'boll_20_2_short' 的信号名称。
    max_per_class : int, default 100
        True 类和 False 类各最多抽取多少条记录参与验证。
    random_state : int
        抽样随机种子，保证可重复。

    返回
    ----
    result_df : pd.DataFrame
        每条被验证样本的详细信息，含：idx、real_price、lower_bound、upper_bound、
        predict_signal、real_signal、is_correct 等列。
    """
    rng = np.random.default_rng(random_state)

    # 1. 基准信号
    real_signal, _ = compute_signal(df, col_name)
    direction = col_name.split('_')[-1]   # 'long' or 'short'

    # 2. True / False 行索引
    idx_true  = real_signal[real_signal].index.to_list()
    idx_false = real_signal[~real_signal].index.to_list()

    # 随机抽样
    if len(idx_true)  > max_per_class:
        idx_true  = rng.choice(idx_true,  size=max_per_class, replace=False)
    if len(idx_false) > max_per_class:
        idx_false = rng.choice(idx_false, size=max_per_class, replace=False)

    # 合并并按顺序
    indices = sorted(np.concatenate([idx_true, idx_false]))

    records = []

    for idx in indices:
        # 头一行无法计算上一根收盘价，跳过
        if idx == 0:
            continue

        # 取到 “上一根 K 线” 为止的数据
        df_hist = df.iloc[:idx]        # 截止到 idx-1 行
        try:
            res = compute_price_range_boll(df_hist, col_name)
        except Exception as e:
            # 任何异常均视为返回 None
            res = None

        lower_bound, upper_bound = (None, None)
        if res is not None:
            lower_bound, upper_bound = res

        # 若 compute_price_range_boll 返回 None，预测一定是 False
        if res is None:
            predict_signal = False
        else:
            price_now = df["close"].iloc[idx]
            if direction == "long":
                predict_signal = price_now >= lower_bound
            else:  # short
                predict_signal = price_now <= upper_bound

        records.append(
            {
                "idx": idx,
                "real_price": float(df["close"].iloc[idx]),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "predict_signal": predict_signal,
                "real_signal": bool(real_signal.iloc[idx]),
                "is_correct": predict_signal == bool(real_signal.iloc[idx]),
            }
        )

    result_df = pd.DataFrame(records)

    # 打印汇总
    total   = len(result_df)
    correct = result_df["is_correct"].sum()
    acc_all = correct / total if total else np.nan
    acc_true  = result_df[result_df.real_signal]["is_correct"].mean()
    acc_false = result_df[~result_df.real_signal]["is_correct"].mean()

    print(
        f"验证样本数: {total}\n"
        f"整体准确率: {acc_all:.3%}\n"
        f"  True 行准确率 : {acc_true:.3%}\n"
        f"  False 行准确率: {acc_false:.3%}"
    )

    # 返回最终的验证结果 True或者false
    if acc_all == 1.0:
        print(f"验证通过: {col_name} 的计算结果正确")
        return True
    else:
        print(f"验证失败: {col_name} 的计算结果不正确")
        return False


def compute_price_range_macross(df, col_name):
    """
    根据df和col_name提前计算出能够使下一个数据产生信号的价格范围。

    参数:
      df: 包含历史行情数据的DataFrame，必须含有"close"列。
      col_name: 信号的名称，例如 "macross_10_20_long" 或 "macross_10_20_short"。

    返回:
      当存在满足条件的价格范围时，返回一个tuple (lower_bound, upper_bound)。
        - 若下界无限制，则lower_bound返回None；若上界无限制，则upper_bound返回None。
      如果不能产生信号则返回 None。
    """
    parts = col_name.split("_")
    signal_type = parts[0]
    direction = parts[-1]

    if signal_type != "macross":
        raise ValueError("当前仅支持macross信号类型")

    try:
        fast_period = int(parts[1])
        slow_period = int(parts[2])
    except Exception as e:
        raise ValueError("col_name格式错误，示例格式：macross_10_20_long") from e

    # 检查数据是否足够计算滚动均线
    if len(df) < max(fast_period, slow_period):
        return None

    # 计算历史中的均线，并与原函数保持一致（这里使用round(4)）
    fast_ma = df["close"].rolling(window=fast_period, min_periods=fast_period).mean()
    slow_ma = df["close"].rolling(window=slow_period, min_periods=slow_period).mean()

    # 对于下一个信号，前置条件仍然使用最后一根已完成的数据
    prev_fast = fast_ma.iloc[-1]
    prev_slow = slow_ma.iloc[-1]

    if direction == "long":
        if not (prev_fast < prev_slow):
            # 前置条件不满足，则无论下一个价格怎样都无法触发信号
            return None
    elif direction == "short":
        if not (prev_fast > prev_slow):
            return None
    else:
        raise ValueError("direction 必须为 long 或 short")

    # 计算用于新均线计算的过去价格和它们的和
    # 对于 fast MA，需要取最后 (fast_period - 1) 个收盘价
    if fast_period - 1 > len(df):
        return None
    last_fast_prices = df["close"].iloc[-(fast_period - 1):]
    fast_sum = last_fast_prices.sum()

    # 对于 slow MA，需要取最后 (slow_period - 1) 个收盘价
    if slow_period - 1 > len(df):
        return None
    last_slow_prices = df["close"].iloc[-(slow_period - 1):]
    slow_sum = last_slow_prices.sum()

    # 设定参数:
    a = fast_period
    b = slow_period

    numerator = a * slow_sum - b * fast_sum
    denominator = b - a

    if denominator == 0:
        # 此时只要前置条件满足，则任意价格都能触发信号。
        return (-float("inf"), float("inf"))

    threshold = numerator / denominator

    # 根据 denominator 的正负和方向确定价格区间
    if denominator > 0:
        if direction == "long":
            # x >= threshold
            return (round(threshold, 10), None)  # 下限阈值，无上限
        else:  # short
            return (None, round(threshold, 10))  # 上限阈值，无下限
    else:  # denominator < 0 时，注意不等号方向反转
        if direction == "long":
            # x <= threshold
            return (None, round(threshold, 10))
        else:  # short
            # x >= threshold
            return (round(threshold, 10), None)

def validate_signals_macross(df, col_name,
                               sample_per_class: int = 100000,
                             tol: float = 1e-8,
                             verbose: bool = True):
    """
    校验 compute_price_range_macross 的正确性。

    参数
    ----
    df : DataFrame
        必须包含列 'close'。
    col_name : str
        形如 'macross_10_20_long' / 'macross_10_20_short'。
    sample_per_class : int, default 100
        对每一个类别（下一根 signal 为 True / False）
        最多抽取多少条样本进行验证。
    verbose : bool, default True
        是否打印进度。

    返回
    ----
    dict
        {
            'true_checked':  …,   # 已验证的 True 样本数量
            'false_checked': …,   # 已验证的 False 样本数量
            'all_passed':   bool  # 只要有一条不匹配就为 False
        }
    """
    # 1. 预计算整张表的信号
    signal_series, _ = compute_signal(df, col_name)

    true_cnt = false_cnt = 0
    true_fail = false_fail = 0
    errors = []

    # 为了随机挑样本，打乱行号（从 1 开始，行 0 没有“上一根”）
    idx_candidates = list(range(1, len(df)))

    for idx in idx_candidates:
        # 抽样完毕就退出循环
        if true_cnt >= sample_per_class and false_cnt >= sample_per_class:
            break

        next_signal = bool(signal_series.iloc[idx])

        # 若该类别已满足抽样量，则跳过
        if next_signal and true_cnt >= sample_per_class:
            continue
        if (not next_signal) and false_cnt >= sample_per_class:
            continue

        # 2. 以 idx-1 作为“上一根”，预测下一根价格区间
        df_before = df.iloc[:idx]  # up to idx-1
        price_range = compute_price_range_macross(df_before, col_name)
        next_price = float(df["close"].iloc[idx])

        # 3. 校验
        violated = False
        msg = ""

        if next_signal:
            # 理应触发信号
            if price_range is None:
                violated = True
                msg = "signal=True 但 price_range=None"
            else:
                lower, upper = price_range
                if lower is not None and next_price < lower - tol:
                    violated = True
                    msg = f"signal=True 应 >= {lower}, 实际 {next_price}"
                if upper is not None and next_price > upper + tol:
                    violated = True
                    msg = f"signal=True 应 <= {upper}, 实际 {next_price}"
        else:
            # 理应不触发信号
            if price_range is not None:
                lower, upper = price_range
                in_lower = (lower is None) or (next_price >= lower - tol)
                in_upper = (upper is None) or (next_price <= upper + tol)
                if in_lower and in_upper:
                    violated = True
                    msg = (f"signal=False 但价格 {next_price} 落在区间 "
                           f"({lower}, {upper}) 内")

        # 4. 统计
        if next_signal:
            true_cnt += 1
            if violated:
                true_fail += 1
        else:
            false_cnt += 1
            if violated:
                false_fail += 1

        if violated:
            if len(errors) < 20:  # 报告里只保留前 20 条详情
                errors.append({
                    "idx": idx,
                    "price": next_price,
                    "price_range": price_range,
                    "signal_expected": next_signal,
                    "msg": msg
                })
    # 计算通过率
    true_pass = true_cnt - true_fail
    false_pass = false_cnt - false_fail
    # 计算通过率
    true_pass_rate = true_pass / true_cnt if true_cnt else 1.0
    false_pass_rate = false_pass / false_cnt if false_cnt else 1.0
    # 打印结果
    if verbose:
        print(f"验证样本数: {true_cnt + false_cnt}")
        print(f"  True 行准确率 : {true_pass_rate:.3%} ({true_pass}/{true_cnt})")
        print(f"  False 行准确率: {false_pass_rate:.3%} ({false_pass}/{false_cnt})")

    report = {
        "true_checked": true_cnt,
        "false_checked": false_cnt,
        "true_failed": true_fail,
        "false_failed": false_fail,
        "errors": errors,
        "all_passed": (true_fail == 0 and false_fail == 0)
    }

    if report['all_passed']:
        print(f"验证通过: {col_name} 的计算结果正确")
        return True
    else:
        print(f"验证失败: {col_name} 的计算结果不正确")
        return False


def compute_price_range_rsi(df, col_name):
    """
    根据历史数据 df 和信号字符串 col_name（如 "rsi_14_70_30_long" 或 "rsi_14_70_30_short"），
    计算下一根K线以什么价格才能使 RSI 信号发生穿越（由下向上穿越 oversold 或由上向下穿越 overbought）。

    参数：
      df: 包含至少 "close" 列的 pandas.DataFrame，要求至少有 period+1 个数据点
      col_name: 信号字符串，格式 "rsi_{period}_{overbought}_{oversold}_{direction}"
                其中 direction 为 "long" 或 "short"

    返回值：
      如果存在满足条件的价格区间，则返回一个二元元组 (low_bound, high_bound)：
        - 对于 long 信号，要求前一周期 RSI < oversold，新周期 RSI >= oversold，
          有效区间为 [x*, +∞)，此处返回 (x*, None)；但是有些情况下，新 RSI 可从下跌的 x 区间“上穿”，
          那么 x∈ [x*, 当前收盘价] 也可以，此时返回 (x*, current_close)。
        - 对于 short 信号，同理返回 (-∞, x*] 或 (current_close, x*]。
      如果无任何价格能产生信号，则返回 None。
    """
    parts = col_name.split("_")
    signal_type = parts[0]
    if signal_type != "rsi":
        raise ValueError("目前仅支持 RSI 信号")
    try:
        period = int(parts[1])
        overbought = int(parts[2])
        oversold = int(parts[3])
    except Exception as e:
        raise ValueError("col_name 格式错误，应为 'rsi_{period}_{overbought}_{oversold}_{direction}'") from e

    direction = parts[-1].lower()  # "long" 或 "short"

    # 检查数据量是否足够
    if len(df) < period + 1:
        return None

    # 计算历史 RSI：这里与原函数一致
    delta = df["close"].diff(1).astype(np.float32)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    # 注意避免除 0
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    rsi_last = rsi.iloc[-1]

    # 对于信号穿越要求：long 信号要求上一周期 < oversold，新周期 >= oversold；
    # short 信号要求上一周期 > overbought，新周期 <= overbought。
    if direction == "long" and rsi_last >= oversold:
        return None
    if direction == "short" and rsi_last <= overbought:
        return None

    # 取得最新收盘价
    current_close = df["close"].iloc[-1]

    # 为“新周期”RSI计算准备——取最后 period 个 diff 值
    # 注意：df["close"].diff() 得到的是从第二行开始的 diff，这里取最后 period 个差值
    last_deltas = df["close"].diff(1).iloc[-period:]
    d_vals = last_deltas.values  # 长度为 period
    # 计算完整窗口内的累计正负收益
    G_full = sum(max(d, 0) for d in d_vals)
    L_full = sum(max(-d, 0) for d in d_vals)
    # 当新增数据后，新窗口将舍去最旧的 diff（即 d_vals[0]），保留后 period-1 个历史 diff，
    # 再加上新涨跌幅 new_delta = (new_price - current_close)
    d_old = d_vals[0]
    G = G_full - (max(d_old, 0))
    L = L_full - (max(-d_old, 0))
    n = period  # 窗口长度

    # 注意：解析解依赖于对以下两种情形分支讨论：
    # 1. 若新价格 x >= current_close，则新 avg_gain = (G + (x - current_close))/n, new avg_loss = L/n，
    #    则 RSI_new = 100 - 100/(1 + (G + (x - current_close)) / L).
    # 2. 若 x < current_close，则新 avg_gain = G/n, new avg_loss = (L + (current_close - x))/n，
    #    则 RSI_new = 100 - 100/(1 + G / (L + (current_close - x))).
    #
    # 定义一个辅助函数 new_rsi(x)：
    def new_rsi(x):
        new_delta = x - current_close
        if x >= current_close:
            new_G = G + new_delta
            new_L = L
        else:
            new_G = G
            new_L = L + (current_close - x)
        # 避免除零问题
        if new_L == 0:
            return 100.0
        rs_new = new_G / new_L
        return 100 - 100 / (1 + rs_new)

    # 分析信号逻辑：
    # long 信号要求新 RSI >= oversold，
    # 因为 new_rsi(x) 随 x 单调增加，故存在唯一解 x* 使得 new_rsi(x*) = oversold.
    # 同理，short 信号要求 new_rsi(x) <= overbought，存在最大 x* 使得 new_rsi(x*) = overbought.
    #
    # 为了求解解析表达式，我们令 A = threshold/(100 - threshold)，其中 threshold 分别取 oversold（long）或 overbought（short）。
    # 对于 x >= current_close（正涨幅分支），令 new_rsi(x)=threshold 得：
    #   100 - 100/(1 + (G + (x-current_close))/L) = threshold
    # =>
    #   (G + (x-current_close))/L = 100/(100-threshold) - 1 = A.
    # 因此解得：
    #   x = current_close + L*A - G.
    #
    # 对于 x < current_close（负涨幅分支），令 new_rsi(x)=threshold 得：
    #   100 - 100/(1 + G/(L+(current_close-x))) = threshold
    # =>
    #   G/(L + (current_close-x)) = A
    # =>
    #   x = current_close + L - G/A.
    #
    # 不同信号方向下，还要结合“前一期 RSI 是否在信号边界一侧”来判断究竟新窗口的 RSI 在 current_close 处
    # 是低于还是高于阈值，从而决定使用正分支还是负分支的解。
    #
    # 先计算 new_rsi 在当前收盘价处，即 f(current_close)：
    f_at_current = new_rsi(current_close)

    if direction == "long":
        threshold = oversold
        A = threshold / (100 - threshold)
        # 若在 current_close 处，新 RSI < oversold，则只有增大 x（x>=current_close）的解有效；
        # 否则，若 f(current_close) >= oversold，则说明在 current_close 已经“过界”，仅可能 x < current_close 使新 RSI 刚好达到阈值。
        if f_at_current < threshold:
            # 使用 x >= current_close 分支：
            x_candidate = current_close + L * A - G
            # 对于 long 信号，满足条件需“上穿”阈值，因此有效区间为 [x_candidate, +∞)
            return (x_candidate, None)
        else:
            # 使用 x < current_close 分支：
            x_candidate = current_close + L - G / A
            # 此时有效区间为 [x_candidate, current_close]（即向下，低价行情“拉低” RSI，再上穿时信号才成立）
            return (x_candidate, current_close)
    elif direction == "short":
        threshold = overbought
        A = threshold / (100 - threshold)
        # 对于 short 信号，要求新 RSI <= overbought。
        # 如果当前点 f(current_close) > threshold，则说明在 current_close，新 RSI 较高，只有减少 x (x<current_close) 才能降至阈值之下。
        if f_at_current > threshold:
            x_candidate = current_close + L - G / A
            # 空仓信号有效区间为 (-∞, x_candidate]
            return (None, x_candidate)
        else:
            # 否则，若 f(current_close) <= threshold，则当前价格已经较低，
            # 需要 x 增加（x>=current_close）才能维持新 RSI 不大于阈值（过高）—不过 RSI 随 x 增大是上升的，所以上界就是 x_candidate
            x_candidate = current_close + L * A - G
            # 有效区间为 [current_close, x_candidate]
            return (current_close, x_candidate)
    else:
        raise ValueError("direction 必须为 'long' 或 'short'")


# ---------- 验证函数 -----------------
def validate_signals_rsi(
        df: pd.DataFrame,
        col_name: str,
        max_samples_per_class: int = 100,
        price_range_func=compute_price_range_rsi,
        atol: float = 1e-8
) -> dict:
    """
    随机抽样验证 compute_price_range_rsi 是否正确。

    参数
    ----
    df : 完整历史 K 线（必须有 "close" 列）
    col_name : 形如 "rsi_14_70_30_long"
    max_samples_per_class : True 类 / False 类 各抽多少行做验证
    price_range_func : 要验证的 “区间求解” 函数
    atol : 数值容忍误差

    返回
    ----
    result : dict 统计信息
    """
    period = int(col_name.split("_")[1])

    # 1. 基准答案
    signal_series, _ = compute_signal(df, col_name)
    signal_series = signal_series.astype(bool)

    # 只有从 period+1 行开始才有“上一窗口 + 下一根 K 线”可比
    valid_idx = np.arange(period + 1, len(df))
    true_idx = valid_idx[signal_series.iloc[valid_idx].values]
    false_idx = valid_idx[~signal_series.iloc[valid_idx].values]

    # 2. 随机抽样
    rng = np.random.default_rng(20240518)
    if len(true_idx) > max_samples_per_class:
        true_idx = rng.choice(true_idx, size=max_samples_per_class, replace=False)
    if len(false_idx) > max_samples_per_class:
        false_idx = rng.choice(false_idx, size=max_samples_per_class, replace=False)

    def _in_range(price: float, prange: Optional[Tuple[Optional[float], Optional[float]]]) -> bool:
        if prange is None:
            return False
        low, high = prange
        if (low is not None) and (price < low - atol):
            return False
        if (high is not None) and (price > high + atol):
            return False
        return True

    # 3. 逐行验证
    ok_true = 0
    ok_false = 0

    for i in true_idx:
        prange = price_range_func(df.iloc[:i], col_name)
        price = df["close"].iloc[i]
        if _in_range(price, prange):
            ok_true += 1
        else:
            raise AssertionError(
                f"[True 例失败] 行号 {i} 真实 close={price:.6f} 不在求得区间 {prange}"
            )

    for i in false_idx:
        prange = price_range_func(df.iloc[:i], col_name)
        price = df["close"].iloc[i]
        if (prange is None) or (not _in_range(price, prange)):
            ok_false += 1
        else:
            raise AssertionError(
                f"[False 例失败] 行号 {i} 真实 close={price:.6f} 却落在区间 {prange}"
            )

    result = {
        "checked_true": len(true_idx),
        "passed_true": ok_true,
        "checked_false": len(false_idx),
        "passed_false": ok_false,
        "overall_pass": (ok_true + ok_false) == (len(true_idx) + len(false_idx))
    }
    return result

def validate_price_range(df, col_name) -> bool:

    if col_name.startswith("abs_"):
        return validate_signals_abs(df, col_name)
    if col_name.startswith("relate_"):
        return validate_signals_relate(df, col_name)
    if col_name.startswith("macd_"):
        return validate_signals_macd(df, col_name)
    if col_name.startswith("boll_"):
        return validate_signals_boll(df, col_name)
    if col_name.startswith("macross_"):
        return validate_signals_macross(df, col_name)
    if col_name.startswith("rsi_"):
        return validate_signals_rsi(df, col_name)
    return '不支持'


def compute_signal_range(df: pd.DataFrame, signal_name: str):
    if signal_name.startswith("abs_"):
        return compute_price_range_abs(df, signal_name)
    if signal_name.startswith("relate_"):
        return compute_price_range_relate(df, signal_name)
    if signal_name.startswith("macd_"):
        return compute_price_range_macd(df, signal_name)
    if signal_name.startswith("boll_"):
        return compute_price_range_boll(df, signal_name)
    if signal_name.startswith("macross_"):
        return compute_price_range_macross(df, signal_name)
    if signal_name.startswith("rsi_"):
        return compute_price_range_rsi(df, signal_name)
    else:
        return None


def safe_compare_num(a, b, tol=1e-6):
    """
    对于数值型数据，使用容差 tol 做比较，解决浮点数精度问题；
    若两者均为 NaN 则返回 True，否则返回两者是否在 tol 范围内相等。
    """
    try:
        if np.isnan(a) and np.isnan(b):
            return True
    except Exception:
        pass
    return abs(a - b) < tol


import numpy as np
import pandas as pd


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
        # 由 .shift(1).rolling(window=period) 得到最后一行对应窗口数据在原始序列中为：[N - period - 1, N - 1)
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
        # 当前行：滚动窗口取最后 period 个收盘价
        current_window = df["close"].iloc[N - period: N]
        # 前一行：窗口向前平移1个周期
        prev_window = df["close"].iloc[N - period - 1: N - 1]
        current_ma = current_window.mean()
        # 使用 ddof=1 模拟 pandas 默认的 rolling.std() 计算方式
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
        # 为计算 RSI 需要 period+1 个收盘价以得到 period 个差分数据
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
        # 利用 pandas 的 ewm 在 C 层面计算 EMA，取最后两行数值即可
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
        # 取前一天与当天的 atr 与 atr 均线值
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

def validate_signal_functions(df, col_name):
    """
    验证原始函数 compute_signal 与优化后的函数 compute_last_signal 在各时刻的计算结果是否一致。

    验证逻辑：
      1. 遍历 df 不同截取（子 DataFrame），即从头开始取前 i+1 行数据；
      2. 对每个子 DataFrame，调用原始函数 compute_signal 计算信号序列和价格序列，
         再取序列最后一行作为原始函数在“当前时刻”的计算结果；
      3. 分别调用新函数 compute_last_signal 计算最后一行的信号结果和价格；
      4. 将两者进行比较，不匹配时打印详细信息。

    参数：
      df: 包含历史行情数据的 DataFrame，至少需要包含列 "high", "low", "close"（以及其它指标所需列）。
      col_name: 指定信号类型和参数的字符串，如："abs_3_10_long"。

    注意：
      由于部分指标对数据行数有要求（例如 rolling 窗口数据量等），验证时建议从足够长的子 DataFrame 开始。
    """
    start_time = time.time()
    mismatches = []
    total_cases = 0
    n = len(df)

    # 根据不同指标，可能需要足够的数据行才能计算出结果
    # 例如：abs/relate/donchian 等指标至少需要 period + 1 行数据，
    # 最简单地：从第二行开始进行验证
    for i in range(1, n):
        sub_df = df.iloc[:i + 1]
        total_cases += 1

        try:
            # 使用原始函数计算完整序列，再取最后一行结果
            orig_signal_series, orig_price_series = compute_signal(sub_df, col_name)
            if hasattr(orig_signal_series, "iloc"):
                orig_signal = orig_signal_series.iloc[-1]
            else:
                orig_signal = orig_signal_series

            if hasattr(orig_price_series, "iloc"):
                orig_price = orig_price_series.iloc[-1]
            else:
                orig_price = orig_price_series

            # 使用新函数仅计算最后一行结果
            new_signal, new_price = compute_last_signal(sub_df, col_name)
        except Exception as e:
            mismatches.append((i, f"异常: {e}"))
            print(f"子 DataFrame（前 {i + 1} 行）计算异常: {e}")
            continue

        # 比较信号（布尔值）和价格（数字或其它类型，采用安全比较）
        signal_match = (orig_signal == new_signal)
        # 如果 trade_price 为数值类型，则用 safe_compare_num 进行比较，否则直接比较
        if isinstance(orig_price, (int, float)) and isinstance(new_price, (int, float)):
            price_match = safe_compare_num(orig_price, new_price)
        else:
            price_match = (orig_price == new_price)

        if not (signal_match):
            mismatches.append((i, orig_signal, new_signal, orig_price, new_price))
            print(f"不匹配的索引 {i}: 原始信号 = {orig_signal}, 新信号 = {new_signal}, "
                  f"原始价格 = {orig_price}, 新价格 = {new_price}")
    print(f"验证耗时: {time.time() - start_time:.2f} 秒 当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} 平均验证耗时: {total_cases / (time.time() - start_time):.2f} 次/秒")
    if mismatches:
        print(f"共验证 {total_cases} 个样本，存在 {len(mismatches)} 处不匹配。")
        return False
    else:
        print(f"共验证 {total_cases} 个样本，所有结果均匹配！")
        return True

# ---------------- demo ----------------
def example():
    key_name = ''
    inst_id_list = ['BTC', 'ETH', 'SOL', 'TON', 'DOGE', 'XRP']
    is_reverse = True

    for inst_id in inst_id_list:
        df = pd.read_csv(f"kline_data/origin_data_1m_100000_{inst_id}-USDT-SWAP_2025-05-14.csv")
        good_df = pd.read_parquet(f'temp_back/{inst_id}_{is_reverse}_filter_similar_strategy.parquet')
        exist_key = []
        # 获取good_df中所有的kai_column这列的不重复值
        kai_column_list = good_df['kai_column'].unique()
        pin_column_list = good_df['pin_column'].unique()
        all_column = list(set(kai_column_list) | set(pin_column_list))
        final_column = []
        for col in all_column:
            target_key = col.split("_")[0] + col.split("_")[-1]
            if target_key not in exist_key:
                exist_key.append(target_key)
                final_column.append(col)
        for signal_name in final_column:
            if key_name in signal_name:
                # signal_name = 'boll_1722_0.3_short'

                result = validate_signal_functions(df, signal_name)
                print(f"{inst_id} Signal: {signal_name}, Validation Result: {result}")

                # # df = df.head(31000)
                # start_time = time.time()
                # threshold = compute_signal_range(df, signal_name)
                # signal_series, price = compute_signal(df, signal_name)
                # df["signal"] = signal_series
                # df["price"] = price
                # print("Elapsed time:", time.time() - start_time)
                # print("Refined threshold:", threshold)


if __name__ == "__main__":
    example()