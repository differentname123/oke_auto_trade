import math
import time

import numpy as np
import pandas as pd
from functools import lru_cache

# from bread_through_deal_data import compute_signal

def compute_signal(df, col_name):
    """
    根据历史行情数据(df)和指定信号名称(col_name)生成交易信号。
    """
    parts = col_name.split("_")
    signal_type = parts[0]
    direction = parts[-1]
    if signal_type == "boll":
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
    if signal_type == "macross":
        fast_period = int(parts[1])
        slow_period = int(parts[2])
        fast_ma = df["close"].rolling(window=fast_period, min_periods=fast_period).mean().round(4)
        slow_ma = df["close"].rolling(window=slow_period, min_periods=slow_period).mean().round(4)
        if direction == "long":
            signal_series = (fast_ma.shift(1) < slow_ma.shift(1)) & (fast_ma >= slow_ma)
        else:
            signal_series = (fast_ma.shift(1) > slow_ma.shift(1)) & (fast_ma <= slow_ma)
        return signal_series, df["close"]

def compute_live_signal_price_range_boll(df, col_name):
    """
    在实盘中，根据历史数据(df)提前计算出让最新数据满足信号要求的价格范围。
    对于Bollinger信号：
      - 多头信号返回：只要最新价格 >= X_min，则满足条件。
      - 空头信号返回：只要最新价格 <= X_max，则满足条件。

    注意：此处使用的滚动窗口构造为最新窗口，由最近 period-1 根已知收盘价和未来未知价格组成，
    需提前满足前一根bar的信号条件才能计算出有效的价格水平。
    """
    parts = col_name.split("_")
    signal_type = parts[0]
    direction = parts[-1]

    if signal_type != "boll":
        raise NotImplementedError("目前仅支持boll信号类型")

    period = int(parts[1])
    std_multiplier = float(parts[2])

    # 保证历史数据足够构造新的窗口（需要 period-1 个历史收盘价）
    if len(df) < period - 1:
        raise ValueError("历史数据不足，至少需要 {} 个数据点".format(period - 1))

    # 计算历史的滚动Bollinger带（用于验证前一根bar的条件）
    ma = df["close"].rolling(window=period, min_periods=period).mean()
    std_dev = df["close"].rolling(window=period, min_periods=period).std()
    upper_band = (ma + std_multiplier * std_dev).round(4)
    lower_band = (ma - std_multiplier * std_dev).round(4)

    # 检查上一根bar是否满足信号条件
    if len(df) < 2:
        raise ValueError("至少需要2根数据才能验证前一根bar的信号条件")

    if direction == "long":
        # 前一根bar的条件：收盘价 < 下轨
        if not (df["close"].iloc[-2] < lower_band.iloc[-2]):
            # 如果不满足，说明当前时刻不满足生成信号的条件，返回None
            return None
    else:  # short
        # 前一根bar的条件：收盘价 > 上轨
        if not (df["close"].iloc[-2] > upper_band.iloc[-2]):
            return None

    # 取最近 period-1 根已知的收盘价组成新的滚动窗口数据
    historical_prices = df["close"].iloc[-(period - 1):]
    S1 = historical_prices.sum()  # sum(x_i)
    S2 = (historical_prices ** 2).sum()  # sum(x_i^2)
    n = period  # 窗口长度(历史期数 + 新价格 X)

    # 定义用于求根的函数
    if direction == "long":
        def f_long(X):
            mean = (S1 + X) / n
            # 计算加入X后的样本方差（ddof=1）
            variance = (S2 + X ** 2 - (S1 + X) ** 2 / n) / (n - 1)
            sigma = math.sqrt(variance) if variance > 0 else 0
            lower = mean - std_multiplier * sigma
            return X - lower  # 当 X - lower == 0 时，即为临界价格

        # 找到 f_long(X)=0 的根，即最低价格要求 X_min
        # 采用二分法。首先构造一个区间保证 f_long(lower_bound) <= 0， f_long(upper_bound) >= 0
        low = 0.0
        # 初始取当前最后一根bar的价格作为高估计，也可以根据实际情况调整
        high = float(df["close"].iloc[-1])
        # 如果 high 处 f_long 函数仍小于 0，则不断扩大 high
        while f_long(high) < 0:
            high *= 2

        tol = 1e-6
        max_iter = 100
        for i in range(max_iter):
            mid = (low + high) / 2
            f_mid = f_long(mid)
            if abs(f_mid) < tol:
                break
            if f_long(low) * f_mid < 0:
                high = mid
            else:
                low = mid
        price_threshold = mid
        # 对多头来说，只要未来价格 >= price_threshold 即可触发信号
        return {"long": {"min_price": round(price_threshold, 4)}}

    else:  # direction == "short"
        def f_short(X):
            mean = (S1 + X) / n
            variance = (S2 + X ** 2 - (S1 + X) ** 2 / n) / (n - 1)
            sigma = math.sqrt(variance) if variance > 0 else 0
            upper = mean + std_multiplier * sigma
            return upper - X  # 当 upper - X == 0 时，即为临界价格

        # 同样采用二分法求解 f_short(X)=0，此处希望找到 X_max，使得价格必须 <= X_max 才能触发信号
        low = 0.0
        high = float(df["close"].iloc[-1])
        # 需要保证 low 处 f_short >= 0，而 high 处 f_short <= 0
        while f_short(low) < 0:
            low = max(low - 10, 0)
        while f_short(high) > 0:
            high *= 2

        tol = 1e-6
        max_iter = 100
        for i in range(max_iter):
            mid = (low + high) / 2
            f_mid = f_short(mid)
            if abs(f_mid) < tol:
                break
            if f_short(low) * f_mid < 0:
                high = mid
            else:
                low = mid
        price_threshold = mid
        # 对空头来说，只要未来价格 <= price_threshold 即可触发信号
        return {"short": {"max_price": round(price_threshold, 4)}}



def validate_live_signal_threshold_all_boll(df, signal_name, max_false_samples=5):
    """
    验证历史信号和实时计算阈值的一致性：

    步骤：
      1. 使用 compute_signal 得到所有历史行的信号（True/False）。
      2. 对于满足滚动窗口要求的数据（至少 period 个数据），逐行模拟实时数据（df[0:idx+1]），
         调用 compute_live_signal_price_range_boll 得到实时阈值。
      3. 如果生成信号 (True)：
             long: 要求 close >= min_price；
             short: 要求 close <= max_price。
         如果未生成信号 (False)：
             long: 要求 close < min_price；
             short: 要求 close > max_price。
      4. 如果 compute_live_signal_price_range_boll 返回 None，
         表示前一根bar条件不满足，此时只要 signal_generated 为 False 就认为验证成功。

    为了减少对 False 情况的计算，当 False 的验证超过 max_false_samples 后，后续 False 不再验证。

    返回一个列表，每个元素为一个字典，包含：
      - index：行索引
      - signal_generated：compute_signal生成的信号 (True/False)
      - close：当时的收盘价
      - threshold：计算得到的价格阈值（如果有）
      - valid：该行是否满足阈值要求 (True/False)
      - difference：（对于 long 为 close - threshold，对于 short 为 threshold - close）
      - note：其他说明信息
    """
    signal_series, _ = compute_signal(df, signal_name)
    results = []
    period = int(signal_name.split("_")[1])
    direction = signal_name.split("_")[-1]

    false_count = 0  # 用于采样 False 的计数
    for idx in range(len(df)):
        # 如果数据不足以构成滚动窗口，则跳过
        if idx + 1 < period:
            continue

        signal_flag = signal_series.iloc[idx]
        # 如果当前信号为 False 且 False 验证样本数量超过上限，则跳过
        if not signal_flag and false_count >= max_false_samples:
            continue

        df_sub = df.iloc[:idx + 1].copy()  # 模拟"实时"数据
        try:
            threshold_result = compute_live_signal_price_range_boll(df_sub, signal_name)
        except Exception as e:
            threshold_result = None
            note = f"计算阈值出错: {e}"
        else:
            note = ""

        row_close = df.iloc[idx]["close"]

        # 当返回 threshold_result 为 None 时，
        # 如果 signal_generated 为 False，我们认为验证成功
        if threshold_result is None:
            if not signal_flag:
                valid = True
                note += " (前一根bar条件不满足，因此未生成信号, 验证成功)"
            else:
                valid = False
                note += " (前一根bar条件不满足，但信号仍为True)"
            result = {
                "index": idx,
                "signal_generated": signal_flag,
                "close": row_close,
                "threshold": None,
                "valid": valid,
                "difference": None,
                "note": note
            }
        else:
            if direction == "long":
                threshold = threshold_result["long"]["min_price"]
                if signal_flag:
                    valid = row_close >= threshold
                    diff = row_close - threshold
                else:
                    valid = row_close < threshold
                    diff = row_close - threshold
                result = {
                    "index": idx,
                    "signal_generated": signal_flag,
                    "close": row_close,
                    "threshold": threshold,
                    "valid": valid,
                    "difference": diff,
                    "note": note
                }
            else:  # short
                threshold = threshold_result["short"]["max_price"]
                if signal_flag:
                    valid = row_close <= threshold
                    diff = threshold - row_close
                else:
                    valid = row_close > threshold
                    diff = threshold - row_close
                result = {
                    "index": idx,
                    "signal_generated": signal_flag,
                    "close": row_close,
                    "threshold": threshold,
                    "valid": valid,
                    "difference": diff,
                    "note": note
                }
        if not signal_flag:
            false_count += 1
        results.append(result)
    return results


def compute_next_signal_price_range(df, col_name):
    """
    根据历史行情数据(df)和信号名称(col_name)计算使下一个数据满足信号条件的价格范围。
    如果数据不足或无法计算，也返回 None。
    """
    parts = col_name.split("_")
    if len(parts) < 2:
        return None

    signal_type = parts[0]
    direction = parts[-1]

    # ---------- abs 信号 -----------
    if signal_type == "abs":
        try:
            period = int(parts[1])
            abs_value = float(parts[2]) / 100
        except (IndexError, ValueError):
            return None
        if direction == "long":
            # 使用前一根K线数据计算 rolling 的最小值
            historical_min_low = df["low"].rolling(period).min().iloc[-1]
            if np.isnan(historical_min_low):
                return None
            target = round(historical_min_low * (1 + abs_value), 4)
            # 对于下根K线，条件为：low <= target <= high 且 high > target
            return (target, 10000000)
        else:  # direction == "short"
            historical_max_high = df["high"].rolling(period).max().iloc[-1]
            if np.isnan(historical_max_high):
                return None
            target = round(historical_max_high * (1 - abs_value), 4)
            return (0, target)

    # ---------- relate 信号 -----------
    elif signal_type == "relate":
        try:
            period = int(parts[1])
            percent = float(parts[2]) / 100
        except (IndexError, ValueError):
            return None
        historical_min = df["low"].rolling(period).min().iloc[-1]
        historical_max = df["high"].rolling(period).max().iloc[-1]
        if np.isnan(historical_min) or np.isnan(historical_max):
            return None
        if direction == "long":
            target = round(historical_min + percent * (historical_max - historical_min), 4)
            return (target, 10000000)
        else:
            target = round(historical_max - percent * (historical_max - historical_min), 4)
            return (0, target)

    # ---------- donchian 信号 -----------
    elif signal_type == "donchian":
        try:
            period = int(parts[1])
        except (IndexError, ValueError):
            return None
        if direction == "long":
            historical_highest = df["high"].rolling(period).max().iloc[-1]
            if np.isnan(historical_highest):
                return None
            target = round(historical_highest, 4)
            return (target, 10000000)
        else:
            historical_lowest = df["low"].rolling(period).min().iloc[-1]
            if np.isnan(historical_lowest):
                return None
            target = round(historical_lowest, 4)
            return (0, target)

    elif signal_type == "macross":
        try:
            fast_period = int(parts[1])
            slow_period = int(parts[2])
        except (IndexError, ValueError):
            return None

        closes = df["close"]
        if len(closes) < max(fast_period, slow_period):
            return None
        if (fast_period - 1) <= 0 or (slow_period - 1) <= 0:
            return None

        S_f = closes.iloc[-(fast_period - 1):].sum()
        S_s = closes.iloc[-(slow_period - 1):].sum()
        try:
            threshold = (fast_period * S_s - slow_period * S_f) / (slow_period - fast_period)
        except ZeroDivisionError:
            return None
        threshold = round(threshold, 4)
        if direction == "long":
            return (threshold, 1000000)
        else:
            return (0, threshold)




def validate_compute_next_signal_price_range(df, col_name):
    """
    验证 compute_next_signal_price_range 的计算是否正确。

    验证逻辑：
    1. 使用 compute_signal 计算整个历史数据的信号。
    2. 对于每个信号触发的行（True），取该行之前的历史数据（df.iloc[:i]），
       计算出使下一个信号触发的价格范围 predicted_range，
       并验证新行是否满足条件（abs/relate/donchian 类型要求新行价格区间包含目标价，
       macross 类型要求收盘价达到或不超过阈值）。
    3. 同时对未触发的（False）信号，也做同样操作，但仅验证最后 100 条 False 信号，
       并判断其价格是否“不满足”应触发该信号的条件。
    4. 如果检测到问题，则打印出来，否则说明验证通过。
    """
    # 目前只有 abs、relate、donchian、macross 信号支持提前预测验证
    verifiable_types = ["abs", "relate", "donchian", "macross"]
    if not any(col_name.startswith(v) for v in verifiable_types):
        print(f"信号 {col_name} 暂不支持提前预测条件验证。")
        return

    signal_series, _ = compute_signal(df, col_name)
    issues = []

    # 验证 True 信号
    for i in range(1, len(df)):
        if not signal_series.iloc[i]:
            continue

        hist_df = df.iloc[:i]
        predicted_range = compute_next_signal_price_range(hist_df, col_name)
        if predicted_range is None:
            issues.append(f"索引 {i} (True): 预测价格范围为 None，但信号触发。")
            continue

        row = df.iloc[i]
        if col_name.startswith("abs") or col_name.startswith("relate") or col_name.startswith("donchian"):
            target = predicted_range[0]  # 预测结果为 (target, target)
            if col_name.endswith("long"):
                # 多头：应满足 new_row.low <= target <= new_row.high 且 new_row.high > target
                if not (row["low"] <= target <= row["high"] and row["high"] > target):
                    issues.append(
                        f"索引 {i} (True, long): 预测目标 {target} 不在新行价格区间 [{row['low']}, {row['high']}] 内。")
            else:
                # 空头：应满足 new_row.low <= target <= new_row.high 且 new_row.low < target
                if not (row["low"] <= target <= row["high"] and row["low"] < target):
                    issues.append(
                        f"索引 {i} (True, short): 预测目标 {target} 不在新行价格区间 [{row['low']}, {row['high']}] 内。")
        elif col_name.startswith("macross"):
            new_close = row["close"]
            if col_name.endswith("long"):
                if new_close < predicted_range[0]:
                    issues.append(
                        f"索引 {i} (True, macross long): 预测阈值 {predicted_range[0]} 未达成，新收盘价 {new_close}。")
            else:
                if new_close > predicted_range[1]:
                    issues.append(
                        f"索引 {i} (True, macross short): 预测阈值 {predicted_range[1]} 未满足，新收盘价 {new_close}。")

    # 验证 False 信号，仅验证最后100个 False 的样本
    false_indices = [i for i in range(1, len(df)) if not signal_series.iloc[i]]
    false_indices = false_indices[-100:]
    for i in false_indices:
        hist_df = df.iloc[:i]
        predicted_range = compute_next_signal_price_range(hist_df, col_name)
        if predicted_range is None:
            issues.append(f"索引 {i} (False): 预测价格范围为 None，但信号为 False。")
            continue

        row = df.iloc[i]
        if col_name.startswith("abs") or col_name.startswith("relate") or col_name.startswith("donchian"):
            target = predicted_range[0]
            if col_name.endswith("long"):
                # 若满足条件则应触发信号，但此处 signal 为 False
                if row["low"] <= target <= row["high"] and row["high"] > target:
                    issues.append(
                        f"索引 {i} (False, long): 预测目标 {target} 却满足新行价格区间 [{row['low']}, {row['high']}]。")
            else:
                if row["low"] <= target <= row["high"] and row["low"] < target:
                    issues.append(
                        f"索引 {i} (False, short): 预测目标 {target} 却满足新行价格区间 [{row['low']}, {row['high']}]。")
        elif col_name.startswith("macross"):
            new_close = row["close"]
            if col_name.endswith("long"):
                # 对于多头，真信号要求 new_close >= threshold，所以 false 时应满足 new_close < threshold
                if new_close >= predicted_range[0]:
                    issues.append(
                        f"索引 {i} (False, macross long): 预测阈值 {predicted_range[0]} 达成，新收盘价 {new_close}。")
            else:
                # 对于空头，真信号要求 new_close <= threshold，所以 false 时应满足 new_close > threshold
                if new_close <= predicted_range[1]:
                    issues.append(
                        f"索引 {i} (False, macross short): 预测阈值 {predicted_range[1]} 满足，新收盘价 {new_close}。")

    if issues:
        print("验证中发现以下问题：")
        for issue in issues:
            print(issue)
    else:
        print("所有触发和未触发的信号均满足提前预测的价格范围条件。")


# ---------------- demo ----------------
def example():
    df = pd.read_csv(
        "kline_data/origin_data_1m_100000_BTC-USDT-SWAP_2025-05-14.csv"
    )
    signal_name = "macross_258_2648_short"

    df = df.head(1100)
    # results = validate_live_signal_threshold_all(df, signal_name)
    # df = df.head(96985)
    for i in range(10):
        start_time = time.time()
        threshold = compute_price_range(df, signal_name)
        signal_series, price = compute_signal(df, signal_name)
        df["signal"] = signal_series
        df["price"] = price
        print("Elapsed time:", time.time() - start_time)
        print("Refined threshold:", threshold)


if __name__ == "__main__":
    example()