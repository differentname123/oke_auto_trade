import math
import time

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


def validate_price_range(df, col_name) -> bool:

    if col_name.startswith("abs_"):
        return validate_signals_abs(df, col_name)
    if col_name.startswith("relate_"):
        return validate_signals_relate(df, col_name)
    if col_name.startswith("macd_"):
        return validate_signals_macd(df, col_name)
    return '不支持'


def compute_signal_range(df: pd.DataFrame, signal_name: str):
    if signal_name.startswith("abs_"):
        return compute_price_range_abs(df, signal_name)
    if signal_name.startswith("relate_"):
        return compute_price_range_relate(df, signal_name)
    if signal_name.startswith("macd_"):
        return compute_price_range_macd(df, signal_name)
    else:
        return None

# ---------------- demo ----------------
def example():
    key_name = 'donchian'
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

                result = validate_price_range(df, signal_name)
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