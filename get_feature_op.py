import datetime
import traceback

import pandas as pd
import numpy as np
import os
import os
import pandas as pd
import time
import okx.MarketData as Market
from common_utils import get_config
import json
# 设置代理（如果需要）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# OKX API初始化
flag = "0"  # 模拟盘: 1, 实盘: 0
apikey = get_config('api_key')
secretkey = get_config('secret_key')
passphrase = get_config('passphrase')
marketAPI = Market.MarketAPI(apikey, secretkey, passphrase, False, flag)
output_path = "kline_data"

def get_kline_data(inst_id, bar="1m", limit=100, max_candles=1000):
    """
    从OKX获取历史K线数据，并返回DataFrame。

    :param inst_id: 产品ID，例如 "BTC-USDT"
    :param bar: 时间粒度，例如 "1m", "5m", "1H" 等
    :param limit: 单次请求的最大数据量，默认100，最大100
    :param max_candles: 请求的最大K线数量，默认1000
    :return: pandas.DataFrame，包含K线数据
    """
    all_data = []
    after = ''  # 初始值为None，获取最新数据
    fetched_candles = 0  # 已获取的K线数量
    fail_count = 0  # 失败次数
    max_retries = 20  # 最大重试次数

    while fetched_candles < max_candles:
        try:
            # 调用OKX API获取历史K线数据
            response = marketAPI.get_history_candlesticks(instId=inst_id, bar=bar, after=after, limit=limit)

            if response.get("code") != "0":
                print(f"获取K线数据失败，错误代码：{response.get('code')}，错误消息：{response.get('msg')}")
                fail_count += 1
                if fail_count >= max_retries:
                    print(f"连续失败 {max_retries} 次，停止获取。")
                    break
                time.sleep(1)
                continue  # 继续下一轮尝试

            fail_count = 0  # 成功获取后重置失败计数

            # 提取返回数据
            data = response.get("data", [])
            if not data:
                print("无更多数据，已全部获取。")
                break

            # 解析数据并添加到总数据中
            all_data.extend(data)
            fetched_candles += len(data)
            if fetched_candles % 10000 == 0:
                print(f"已获取 {fetched_candles} 根K线数据...")

            # 更新 `after` 参数为当前返回数据的最早时间戳，用于获取更早的数据
            after = data[-1][0]

            # 如果获取的数据量小于limit，说明数据已获取完毕
            if len(data) < limit:
                break

            # 短暂延迟，避免触发API限频
            time.sleep(0.2)

        except Exception as e:
            traceback.print_exc()
            print(f"获取K线数据时出现异常：{e}")
            fail_count += 1
            if fail_count >= max_retries:
                print(f"连续异常 {max_retries} 次，停止获取。")
                break
            time.sleep(1)  # 等待后重试
            continue

    # 将所有数据转换为DataFrame，即使all_data为空也能处理
    if all_data:
        df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", "volCcy", "volCcyQuote",
                                             "confirm"])

        # 数据类型转换
        # 将时间戳转换为 datetime 对象，并将其设置为 UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float) / 1000, unit="s", utc=True)

        # 将 UTC 时间转换为北京时间（Asia/Shanghai 时区，UTC+8）
        df["timestamp"] = df["timestamp"].dt.tz_convert('Asia/Shanghai')
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

        # 按时间排序
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    else:
        print("未能获取到任何K线数据。")
        return pd.DataFrame() # 返回一个空的 DataFrame

def get_kline_data_newest(inst_id, bar="1m", limit=100, max_candles=1000):
    """
    从OKX获取历史K线数据，并返回DataFrame。

    :param inst_id: 产品ID，例如 "BTC-USDT"
    :param bar: 时间粒度，例如 "1m", "5m", "1H" 等
    :param limit: 单次请求的最大数据量，默认100，最大100
    :param max_candles: 请求的最大K线数量，默认1000
    :return: pandas.DataFrame，包含K线数据
    """
    all_data = []
    after = ''  # 初始值为None，获取最新数据
    fetched_candles = 0  # 已获取的K线数量
    fail_count = 0  # 失败次数
    max_retries = 3  # 最大重试次数

    while fetched_candles < max_candles:
        try:
            # 调用OKX API获取历史K线数据
            response = marketAPI.get_candlesticks(instId=inst_id, bar=bar, after=after, limit=limit)

            if response.get("code") != "0":
                print(f"获取K线数据失败，错误代码：{response.get('code')}，错误消息：{response.get('msg')}")
                time.sleep(1)
                fail_count += 1
                if fail_count >= max_retries:
                    print(f"连续失败 {max_retries} 次，停止获取。")
                    break
            else:
                fail_count = 0
                # 提取返回数据
                data = response.get("data", [])
                if not data:
                    print("无更多数据，已全部获取。")
                    break

                # 解析数据并添加到总数据中
                all_data.extend(data)
                fetched_candles += len(data)

                # 更新 `after` 参数为当前返回数据的最早时间戳，用于获取更早的数据
                after = data[-1][0]

                # 如果获取的数据量小于limit，说明数据已获取完毕
                if len(data) < limit:
                    break

                # 短暂延迟，避免触发API限频
                time.sleep(0.2)

        except Exception as e:
            print(f"获取K线数据时出现异常：{e}")
            break

    # 将所有数据转换为DataFrame，即使all_data为空也能处理
    if all_data:
        df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", "volCcy", "volCcyQuote",
                                             "confirm"])

        # 数据类型转换
        # 将时间戳转换为 datetime 对象，并将其设置为 UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float) / 1000, unit="s", utc=True)

        # 将 UTC 时间转换为北京时间（Asia/Shanghai 时区，UTC+8）
        df["timestamp"] = df["timestamp"].dt.tz_convert('Asia/Shanghai')
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

        # 按时间排序
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    else:
        print("未能获取到任何K线数据。")
        return pd.DataFrame() # 返回一个空的 DataFrame

def calculate_ma(data, period):
    """计算移动平均线 (MA)"""
    return data.rolling(window=period).mean()


def calculate_ema(data, period):
    """计算指数移动平均线 (EMA)"""
    return data.ewm(span=period, adjust=False).mean()


def calculate_rsi(data, period):
    """计算相对强弱指数 (RSI)"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data, fast_period, slow_period, signal_period):
    """计算 MACD 指标"""
    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)
    macd = ema_fast - ema_slow
    signal = calculate_ema(macd, signal_period)
    hist = macd - signal
    return macd, signal, hist


def calculate_bollinger_bands(data, period, num_std):
    """计算布林带"""
    ma = calculate_ma(data, period)
    std = data.rolling(window=period).std()
    upper_band = ma + (std * num_std)
    lower_band = ma - (std * num_std)
    return upper_band, ma, lower_band


def calculate_stoch(high, low, close, k_period, d_period):
    """计算随机指标 (KDJ)"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return k, d


def calculate_atr(high, low, close, period):
    """计算平均真实波幅 (ATR)"""
    high_low = high - low
    high_close_prev = np.abs(high - close.shift())
    low_close_prev = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_cci(high, low, close, period):
    """计算商品通道指数 (CCI)"""
    tp = (high + low + close) / 3
    ma = calculate_ma(tp, period)

    # 使用 mean absolute deviation 代替 mad
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())

    cci = (tp - ma) / (0.015 * mad)
    return cci


def calculate_adx(high, low, close, period):
    """计算平均趋向指数 (ADX)"""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = (up_move > down_move) & (up_move > 0)
    plus_dm = pd.Series(np.where(plus_dm, up_move, 0), index=high.index)
    minus_dm = (down_move > up_move) & (down_move > 0)
    minus_dm = pd.Series(np.where(minus_dm, down_move, 0), index=low.index)

    tr = calculate_atr(high, low, close, period)  # 使用前面定义的 ATR 函数
    atr = tr.rolling(window=period).mean()

    plus_di = 100 * calculate_ema(plus_dm, period) / atr
    minus_di = 100 * calculate_ema(minus_dm, period) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = calculate_ema(dx, period)
    return adx


def calculate_momentum(data, period):
    """计算动量 (Momentum)"""
    return data.diff(period)


def feature_engineering(df):
    """
    对原始数据进行特征工程，生成用于模型训练的特征数据。

    Args:
        df: pandas.DataFrame, 原始数据，包含 "timestamp", "open", "high", "low", "close", "volume" 列
            其中 "timestamp" 列为 datetime 类型

    Returns:
        pandas.DataFrame: 包含特征的 DataFrame
    """
    # 保留原始时间戳列
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 设置时间戳为索引
    df = df.set_index('timestamp')

    # 创建一个空的 DataFrame 用于存储特征
    df_features = pd.DataFrame(index=df.index)

    # 将时间戳列添加到 df_features 中
    df_features['timestamp'] = df.index

    # 现在 df_features 中不仅包含特征数据，还包含时间戳信息

    # 1. 基础特征
    df_features["range"] = df["high"] - df["low"]
    df_features["avg_price"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

    # 2. 技术指标
    # 不使用 TA-Lib 计算技术指标
    periods = [5, 10, 15, 30]  # 由于计算复杂度，减少了时间窗口
    short_periods = [5, 10]
    long_periods = [15, 30]

    for period in short_periods:
        df_features[f"MA_{period}"] = calculate_ma(df["close"], period)
        df_features[f"EMA_{period}"] = calculate_ema(df["close"], period)
        df_features[f"RSI_{period}"] = calculate_rsi(df["close"], period)
        df_features[f"MACD_{period}"], df_features[f"MACDSignal_{period}"], df_features[
            f"MACDHist_{period}"] = calculate_macd(
            df["close"], period, period * 2, 9
        )
        df_features[f"BBANDS_upper_{period}"], df_features[f"BBANDS_middle_{period}"], df_features[
            f"BBANDS_lower_{period}"] = calculate_bollinger_bands(
            df["close"], period, 2
        )
        df_features[f"K_{period}"], df_features[f"D_{period}"] = calculate_stoch(
            df["high"], df["low"], df["close"], period, 3
        )
        df_features[f"ATR_{period}"] = calculate_atr(df["high"], df["low"], df["close"], period)

    for period in long_periods:
        df_features[f"CCI_{period}"] = calculate_cci(df["high"], df["low"], df["close"], period)
        df_features[f"ADX_{period}"] = calculate_adx(df["high"], df["low"], df["close"], period)
        df_features[f"MOM_{period}"] = calculate_momentum(df["close"], period)
        df_features[f"MA_{period}"] = calculate_ma(df["close"], period)

    # 3. 波动率特征
    for period in short_periods:
        df_features[f"std_{period}"] = df["close"].rolling(window=period).std()
        df_features[f"return_{period}"] = df["close"].pct_change(periods=period)
        df_features[f"range_pct_{period}"] = (df["high"] - df["low"]) / df["low"] * 100  # 波动幅度百分比

    # 4. 交易量特征
    for period in long_periods + short_periods:
        if period in short_periods:
            df_features[f"volume_ma_{period}"] = calculate_ma(df["volume"], period)
            df_features[f"volume_change_{period}"] = df["volume"].pct_change(periods=period)
        elif period in long_periods:
            df_features[f"volume_ma_{period}"] = calculate_ma(df["volume"], period)
        # 然后再计算 volume_ratio
    for period in short_periods:
        df_features[f"volume_ratio_{period}"] = df_features[f"volume_ma_{period}"] / df_features[
            f"volume_ma_{short_periods[-1]}"]

    # 5. 时间特征
    df_features["minute_of_day"] = df.index.hour * 60 + df.index.minute
    df_features["hour_of_day"] = df.index.hour

    # 6. 特征交叉
    # 针对短线交易，更关注短期指标的交叉
    df_features["MA_diff_5_10"] = df_features["MA_5"] - df_features["MA_10"]
    df_features["EMA_diff_5_10"] = df_features["EMA_5"] - df_features["EMA_10"]
    df_features["close_above_MA_5"] = (df["close"] > df_features["MA_5"]).astype(int)
    df_features["close_above_MA_10"] = (df["close"] > df_features["MA_10"]).astype(int)
    df_features["RSI_gt_70_5"] = (df_features["RSI_5"] > 70).astype(int)
    df_features["RSI_lt_30_5"] = (df_features["RSI_5"] < 30).astype(int)
    df_features["RSI_gt_70_10"] = (df_features["RSI_10"] > 70).astype(int)
    df_features["RSI_lt_30_10"] = (df_features["RSI_10"] < 30).astype(int)
    df_features["MACD_diff_5"] = df_features["MACD_5"] - df_features["MACDSignal_5"]
    df_features["MACD_cross_up_5"] = (
                (df_features["MACD_diff_5"] > 0) & (df_features["MACD_diff_5"].shift(1) < 0)).astype(int)
    df_features["MACD_cross_down_5"] = (
                (df_features["MACD_diff_5"] < 0) & (df_features["MACD_diff_5"].shift(1) > 0)).astype(int)
    df_features["MACD_diff_10"] = df_features["MACD_10"] - df_features["MACDSignal_10"]
    df_features["MACD_cross_up_10"] = (
                (df_features["MACD_diff_10"] > 0) & (df_features["MACD_diff_10"].shift(1) < 0)).astype(int)
    df_features["MACD_cross_down_10"] = (
                (df_features["MACD_diff_10"] < 0) & (df_features["MACD_diff_10"].shift(1) > 0)).astype(int)
    df_features['vol_price_corr_5'] = df['volume'].rolling(5).corr(df['close'])
    df_features['vol_price_corr_10'] = df['volume'].rolling(10).corr(df['close'])
    df_features['high_low_diff'] = df['high'] - df['low']
    df_features['close_open_diff'] = df['close'] - df['open']
    df_features['MA_5_slope'] = df_features['MA_5'] - df_features['MA_5'].shift(1)
    df_features['MA_10_slope'] = df_features['MA_10'] - df_features['MA_10'].shift(1)
    df_features['price_above_MA_5'] = (df['close'] > df_features['MA_5']).astype(int)
    df_features['price_above_MA_10'] = (df['close'] > df_features['MA_10']).astype(int)
    df_features["K_5_gt_80"] = (df_features["K_5"] > 80).astype(int)
    df_features["K_5_lt_20"] = (df_features["K_5"] < 20).astype(int)
    df_features["K_10_gt_80"] = (df_features["K_10"] > 80).astype(int)
    df_features["K_10_lt_20"] = (df_features["K_10"] < 20).astype(int)
    df_features["D_5_gt_80"] = (df_features["D_5"] > 80).astype(int)
    df_features["D_5_lt_20"] = (df_features["D_5"] < 20).astype(int)
    df_features["D_10_gt_80"] = (df_features["D_10"] > 80).astype(int)
    df_features["D_10_lt_20"] = (df_features["D_10"] < 20).astype(int)
    df_features["K_5_D_5_diff"] = df_features["K_5"] - df_features["D_5"]
    df_features["K_10_D_10_diff"] = df_features["K_10"] - df_features["D_10"]
    df_features["K_5_cross_up_D_5"] = (
                (df_features["K_5_D_5_diff"] > 0) & (df_features["K_5_D_5_diff"].shift(1) < 0)).astype(int)
    df_features["K_5_cross_down_D_5"] = (
                (df_features["K_5_D_5_diff"] < 0) & (df_features["K_5_D_5_diff"].shift(1) > 0)).astype(int)
    df_features["K_10_cross_up_D_10"] = (
                (df_features["K_10_D_10_diff"] > 0) & (df_features["K_10_D_10_diff"].shift(1) < 0)).astype(int)
    df_features["K_10_cross_down_D_10"] = (
                (df_features["K_10_D_10_diff"] < 0) & (df_features["K_10_D_10_diff"].shift(1) > 0)).astype(int)

    # 合并原始数据和生成的特征数据
    df_combined = pd.concat([df, df_features], axis=1)

    # 去除包含 NaN 的行
    df_combined.dropna(inplace=True)

    return df_combined


# 示例用法 (假设你已经获取了数据并存储在 df 变量中, 且 df["timestamp"] 列为 datetime 类型)
# df = pd.DataFrame(...)  # 你的数据
# features_df = feature_engineering(df)
# print(features_df)
def get_dist(data_path):
    """
    获取目标变量的分布信息并存储到CSV文件
    :param data_path: 输入数据的路径
    :param output_csv_path: 输出CSV文件的路径
    :return: None
    """
    output_path = f'{data_path[:-4]}_distribution.csv'
    if os.path.exists(output_path):
        data_df = pd.read_csv(output_path)
        data_df['score*'] = 10*(data_df['ratio'] - 0.05)*10*(data_df['ratio'] - 0.05) / (data_df['period']) * data_df['1'] * data_df['1']
        # 将data_df['score']归一化
        data_df['score*'] = (data_df['score*'] - data_df['score*'].min()) / (data_df['score*'].max() - data_df['score*'].min())
        data_df['score'] = 10*(data_df['ratio'] - 0.05)*10*(data_df['ratio'] - 0.05) / (data_df['period']) * data_df['1'] * data_df['1']
        # 将data_df['score']归一化
        data_df['score'] = (data_df['score'] - data_df['score'].min()) / (data_df['score'].max() - data_df['score'].min())

        return data_df

    df = pd.read_csv(data_path)
    feature_cols = [col for col in df.columns if 'close_down' in col or 'close_up' in col]

    # 用于存储分布信息的列表
    distribution_data = []

    for col in feature_cols:
        value_counts = df[col].value_counts(normalize=True)  # 计算分布
        row = [col]  # 初始化行，包含列名
        forward = col.split('_')[1]
        ratio = float(col.split('_')[2])
        period = col.split('_')[3]
        period = int(period[1:])  # 去掉前缀 t
        row.append(forward)
        row.append(period)
        row.append(ratio)
        # 按照 0 和 1 顺序添加分布值
        row.append(value_counts.get(0, 0))  # 如果没有值，返回 0
        row.append(value_counts.get(1, 0))  # 如果没有值，返回 0
        distribution_data.append(row)

    # 创建 DataFrame 并添加列名
    dist_df = pd.DataFrame(distribution_data, columns=['col', 'forward', 'period', 'ratio', '0', '1'])
    # 将dist_df保存到文件
    dist_df.to_csv(output_path, index=False)
    return dist_df


def gen_feature(origin_name):
    # 如果origin_name已经是dataframe，直接使用
    if isinstance(origin_name, pd.DataFrame):
        data = origin_name

    else:
        data = pd.read_csv(origin_name)
    # data = data.tail(1000)
    df = feature_engineering(data)
    # # 将处理后的数据保存到文件
    df.to_csv(f'{output_path}/df_features.csv', index=False)
    # # 并且保留最新的100000条数据到文件
    # df.tail(100000).to_csv(f'{origin_name[:-4]}_features_tail.csv', index=False)
    return df

def add_target_variables_op(df, step_list=[10,100,1000,10000,10000]):
    """
    添加目标变量，包括未来多个时间步的涨幅和跌幅，针对close、high、low的多阈值计算。

    :param df: 数据集
    :param thresholds: 阈值列表，默认[0.1, 0.2, ..., 1.0]
    :param max_decoder_length: 预测未来的时间步长，默认5
    :return: 添加了目标变量的DataFrame
    """
    # 创建一个新的字典，用于存储所有新增列
    new_columns = {}


    for col in ["close"]:
        for step in step_list:  # 未来 1 到 max_decoder_length 分钟
            # 获取未来 step 个时间窗口内的最高价和最低价
            future_max_high = df['high'].rolling(window=step, min_periods=1).max().shift(-step)
            future_min_low = df['low'].rolling(window=step, min_periods=1).min().shift(-step)

            # 计算未来 step 个时间窗口内的最大涨幅和跌幅 (修正部分)
            new_columns[f"{col}_max_up_t{step}"] = (future_max_high - df[col]) / df[col] * 100 #最大涨幅用最高价
            new_columns[f"{col}_max_down_t{step}"] = (df[col] - future_min_low) / df[col] * 100 #最大跌幅用最低价
    # 使用 pd.concat 一次性将所有新列添加到原数据框
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    return df

def get_train_data(inst_id="BTC-USDT-SWAP", bar="1m", limit=100, max_candles=1000, is_newest=False):
    # inst_id = "BTC-USDT-SWAP"
    # bar = "1m"
    # limit = 100
    # max_candles = 60 * 24

    # 获取数据
    if is_newest:
        kline_data = get_kline_data_newest(inst_id=inst_id, bar=bar, limit=limit, max_candles=max_candles)
    else:
        kline_data = get_kline_data(inst_id=inst_id, bar=bar, limit=limit, max_candles=max_candles)

    if not kline_data.empty:
        # print("成功获取K线数据，开始处理...")

        # 添加时间特征
        # kline_data = add_time_features(kline_data)

        # 添加目标变量
        kline_data = add_target_variables_op(kline_data)

        # 重置索引
        kline_data.reset_index(drop=True, inplace=True)
        return kline_data
    else:
        print("未能获取到任何K线数据。")
        return pd.DataFrame()

def get_latest_data(max_candles=1000):
    origin_data = get_train_data(max_candles=max_candles)
    feature_df = gen_feature(origin_data)
    return feature_df

def generate_body_wick_signals(df):
    """
    根据 K 线实体和影线的关系生成多种参数的交易信号。
    """
    df_signals = pd.DataFrame(index=df.index)
    body_ratios = [0.6, 0.7, 0.8]  # 实体相对于 K 线范围的比例
    wick_ratios = [0.2, 0.3, 0.4]  # 影线相对于实体大小的比例

    body = abs(df['close'] - df['open'])
    candle_range = df['high'] - df['low']
    upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
    lower_wick = df[['close', 'open']].min(axis=1) - df['low']

    for body_ratio in body_ratios:
        for wick_ratio in wick_ratios:
            # 看涨信号
            buy_condition = (
                (body > candle_range * body_ratio) &
                (upper_wick < body * wick_ratio) &
                (lower_wick > body * wick_ratio)
            )
            df_signals[f'Bullish_BodyWick_B{int(body_ratio*100)}_W{int(wick_ratio*100)}_Buy'] = np.where(buy_condition, 1, 0)

            # 看跌信号
            sell_condition = (
                (body > candle_range * body_ratio) &
                (lower_wick < body * wick_ratio) &
                (upper_wick > body * wick_ratio)
            )
            df_signals[f'Bearish_BodyWick_B{int(body_ratio*100)}_W{int(wick_ratio*100)}_Sell'] = np.where(sell_condition, 1, 0)

    return pd.concat([df, df_signals], axis=1)


def generate_volume_spike_signals(df):
    """
    优化后的函数，使用向量化操作生成基于多种周期成交量异动的交易信号。
    通过一次性连接列来避免DataFrame碎片化。
    """
    start_time = time.time()
    signal_cols = []  # 用于存储生成的信号列

    n_periods_list = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]
    volume_multipliers = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10]

    for n_periods in n_periods_list:
        average_volume = df['volume'].rolling(window=n_periods).mean().shift(1)
        for multiplier in volume_multipliers:
            # 向量化计算买入和卖出条件
            buy_condition = (df['volume'] > average_volume * multiplier) & (df['close'] > df['open'])
            sell_condition = (df['volume'] > average_volume * multiplier) & (df['close'] < df['open'])

            # 使用向量化赋值创建信号列，并添加到列表中
            signal_cols.append(pd.Series(np.where(buy_condition, 1, 0), index=df.index, name=f'Volume_Spike_Up_{n_periods}_M{int(multiplier*10)}_Buy'))
            signal_cols.append(pd.Series(np.where(sell_condition, 1, 0), index=df.index, name=f'Volume_Spike_Down_{n_periods}_M{int(multiplier*10)}_Sell'))

    # 一次性连接所有信号列
    df_signals = pd.concat(signal_cols, axis=1)

    print(f"Volume spike signals generated in {time.time() - start_time:.2f} seconds.")
    return pd.concat([df, df_signals], axis=1)

def generate_signals(df):
    """
    融合多种方法生成的买入和卖出信号。

    Args:
        df: 包含 'timestamp', 'open', 'high', 'low', 'close', 'volume' 列的 Pandas DataFrame。

    Returns:
        DataFrame:  包含原始数据和生成的信号列的 DataFrame。
    """

    df = df.copy()  # 避免修改原始 DataFrame
    # df = generate_volume_spike_signals(df)
    # df = generate_price_extremes_reverse_signals(df, [x for x in range(10, 10000, 10)])
    df = generate_price_extremes_signals(df, [x for x in range(10, 10000, 100)])
    # 生成单一 MA 信号
    df = generate_signals_single_ma(df, [x for x in range(10, 10000, 100)])

    # 生成双 MA 信号
    df = generate_signals_double_ma(df, [x for x in range(10, 2500, 100)], [x for x in range(10, 2500, 100)])
    # df = generate_body_wick_signals(df)
    # df = generate_consecutive_and_large_candle_signals(df)
    # df = generate_ichimoku_signals(df)

    # df = generate_volume_signals(df)
    # df = generate_price_signals(df)
    # df = generate_moving_average_signals(df)
    # df = generate_rsi_signals(df)
    # df = generate_macd_signals(df)
    # df = generate_bollinger_band_signals(df)
    # df = generate_ema_cross_signals(df)
    # df = generate_stochastic_signals(df)
    # df = generate_cci_signals(df)
    # df = generate_adx_signals(df)
    # df = generate_obv_signals(df)
    # df = generate_williams_r_signals(df)
    # df = generate_cmf_signals(df)
    # df = generate_mfi_signals(df)
    # df = generate_roc_signals(df)
    # df = generate_donchian_channel_signals(df)
    # df = generate_keltner_channel_signals(df)

    return df


def generate_volume_signals(df, short_window=5, long_window=20):
    """
    基于成交量和其他指标生成买卖信号。

    Args:
        df (pd.DataFrame): 包含 'close' 和 'volume' 列的 DataFrame。
        short_window (int): 短期均线的窗口大小。
        long_window (int): 长期均线的窗口大小。

    Returns:
        pd.DataFrame: 包含信号列的 DataFrame。
    """
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame

    # 1. 基于成交量和价格变化的信号
    price_change = df['close'].diff()
    volume_change = df['volume'].diff()

    df['Volume_Buy'] = np.where((price_change > 0) & (volume_change > 0), 1, 0)
    df['Volume_Sell'] = np.where((price_change < 0) & (volume_change > 0), 1, 0)

    # 2. 基于成交量移动平均线的信号
    df['volume_short_ma'] = df['volume'].rolling(window=short_window).mean()
    df['volume_long_ma'] = df['volume'].rolling(window=long_window).mean()

    df['Volume_MA_Buy'] = np.where(df['volume_short_ma'] > df['volume_long_ma'], 1, 0)
    df['Volume_MA_Sell'] = np.where(df['volume_short_ma'] < df['volume_long_ma'], 1, 0)

    # 3. 基于价格和成交量移动平均线的信号(金叉死叉)
    df['close_short_ma'] = df['close'].rolling(window=short_window).mean()
    df['close_long_ma'] = df['close'].rolling(window=long_window).mean()

    df['Price_Volume_GoldenCross_Buy'] = np.where(
        (df['close_short_ma'] > df['close_long_ma']) & (df['volume_short_ma'] > df['volume_long_ma']), 1, 0)
    df['Price_Volume_DeathCross_Sell'] = np.where(
        (df['close_short_ma'] < df['close_long_ma']) & (df['volume_short_ma'] < df['volume_long_ma']), 1, 0)

    # 4. 价格上涨且成交量大于其短期均线
    df['PriceUp_VolumeAboveSMA_Buy'] = np.where((price_change > 0) & (df['volume'] > df['volume_short_ma']), 1, 0)

    # 5. 价格下跌且成交量大于其短期均线
    df['PriceDown_VolumeAboveSMA_Sell'] = np.where((price_change < 0) & (df['volume'] > df['volume_short_ma']), 1, 0)

    # --- 可以添加更多信号 ---

    # 删除中间生成的列
    df.drop(columns=['volume_short_ma', 'volume_long_ma', 'close_short_ma', 'close_long_ma'], inplace=True)

    return df

def generate_price_signals(df):
    """
    生成多种价格突破相关的价格信号，买入信号包含_Buy，卖出信号包含_Sell。

    Args:
        df: 包含 'open', 'high', 'low', 'close' 列的 Pandas DataFrame。

    Returns:
        Pandas DataFrame，包含原始数据和生成的信号列。
    """
    df_signals = df.copy() # 创建副本，避免修改原始df

    # 1. 简单价格突破
    df_signals['Price_Breakout_Buy'] = np.where(df_signals['close'] > df_signals['high'].shift(1), 1, 0)
    df_signals['Price_Breakout_Sell'] = np.where(df_signals['close'] < df_signals['low'].shift(1), 1, 0)

    # 2. 更高周期的价格突破 (例如，突破前N天的最高价/最低价)
    n_days = 5  # 可以根据需要调整周期
    df_signals[f'High_Breakout_{n_days}D_Buy'] = np.where(df_signals['close'] > df_signals['high'].rolling(window=n_days).max().shift(1), 1, 0)
    df_signals[f'Low_Breakout_{n_days}D_Sell'] = np.where(df_signals['close'] < df_signals['low'].rolling(window=n_days).min().shift(1), 1, 0)

    # 3. 开盘价突破
    df_signals['Open_Breakout_High_Buy'] = np.where(df_signals['close'] > df_signals['open'].shift(1), 1, 0)
    df_signals['Open_Breakout_Low_Sell'] = np.where(df_signals['close'] < df_signals['open'].shift(1), 1, 0)

    # 4. 组合突破 (例如，同时突破前一天的最高价和开盘价)
    df_signals['Combined_HighOpen_Breakout_Buy'] = np.where((df_signals['close'] > df_signals['high'].shift(1)) & (df_signals['close'] > df_signals['open'].shift(1)), 1, 0)
    df_signals['Combined_LowOpen_Breakout_Sell'] = np.where((df_signals['close'] < df_signals['low'].shift(1)) & (df_signals['close'] < df_signals['open'].shift(1)), 1, 0)

    return df_signals

def generate_moving_average_signals(df):
    """
    生成基于移动平均线的买卖信号，并探索不同参数下的信号。

    Args:
        df: 包含 'close' 列的 DataFrame。

    Returns:
        包含原始数据和生成信号的 DataFrame。
    """

    # 参数空间，用于探索不同的均线周期
    short_windows = [3, 5, 7, 10]
    long_windows = [15, 20, 30, 50]

    for short_window in short_windows:
        for long_window in long_windows:
            if short_window >= long_window:
                continue  # 确保短周期小于长周期

            # 计算移动平均线
            sma_short_col = f'SMA_{short_window}'
            sma_long_col = f'SMA_{long_window}'
            df[sma_short_col] = df['close'].rolling(window=short_window, min_periods=1).mean()
            df[sma_long_col] = df['close'].rolling(window=long_window, min_periods=1).mean()

            # 生成金叉和死叉信号
            golden_cross_buy_signal = f'MA_Golden_Cross_{short_window}_{long_window}_Buy'
            death_cross_sell_signal = f'MA_Death_Cross_{short_window}_{long_window}_Sell'
            df[golden_cross_buy_signal] = np.where(
                (df[sma_short_col] > df[sma_long_col]) & (df[sma_short_col].shift(1) <= df[sma_long_col].shift(1)), 1, 0)
            df[death_cross_sell_signal] = np.where(
                (df[sma_short_col] < df[sma_long_col]) & (df[sma_short_col].shift(1) >= df[sma_long_col].shift(1)), 1, 0)

            # 生成价格突破均线信号
            price_above_buy_signal = f'Price_Above_SMA_{short_window}_Buy'
            price_below_sell_signal = f'Price_Below_SMA_{short_window}_Sell'
            df[price_above_buy_signal] = np.where(
                (df['close'] > df[sma_short_col]) & (df['close'].shift(1) <= df[sma_short_col].shift(1)), 1, 0)
            df[price_below_sell_signal] = np.where(
                (df['close'] < df[sma_short_col]) & (df['close'].shift(1) >= df[sma_short_col].shift(1)), 1, 0)

            # 删除过程中生成的均线列，保留信号列
            df.drop(columns=[sma_short_col, sma_long_col], inplace=True)

    return df
def generate_rsi_signals(df):
    rsi_period = 14
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = down.rolling(window=rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['rsi'] = rsi  # 保留 RSI 用于后续信号生成

    signals = {}
    lower_threshold = 30
    upper_threshold = 70
    mid_level = 50

    # RSI 阈值信号
    signals['RSI_Threshold_Buy'] = np.where(df['rsi'] < lower_threshold, 1, 0)
    signals['RSI_Threshold_Sell'] = np.where(df['rsi'] > upper_threshold, 1, 0)

    # RSI 突破信号
    signals['RSI_Break_Buy'] = np.where(
        (df['rsi'] >= lower_threshold) & (df['rsi'].shift(1) < lower_threshold), 1, 0)
    signals['RSI_Break_Sell'] = np.where(
        (df['rsi'] <= upper_threshold) & (df['rsi'].shift(1) > upper_threshold), 1, 0)

    # RSI 中线交叉信号
    signals['RSI_Mid_Cross_Buy'] = np.where(
        (df['rsi'] > mid_level) & (df['rsi'].shift(1) <= mid_level), 1, 0)
    signals['RSI_Mid_Cross_Sell'] = np.where(
        (df['rsi'] < mid_level) & (df['rsi'].shift(1) >= mid_level), 1, 0)

    # RSI 背离信号 (简化版 - 仅考虑与价格的简单对比，更复杂的背离需要更多逻辑)
    # 注意：这只是一个简化的示例，实际背离判断需要更精细的逻辑
    # 找到局部高点和低点，并比较价格和 RSI 的趋势
    # 这里仅作为示例，实际应用中可能需要更复杂的算法
    # signals['RSI_Divergence_Buy'] = np.where(
    #     (df['close'].diff(2) > 0) & (df['rsi'].diff(2) < 0), 1, 0)
    # signals['RSI_Divergence_Sell'] = np.where(
    #     (df['close'].diff(2) < 0) & (df['rsi'].diff(2) > 0), 1, 0)

    # 将信号添加到 DataFrame
    for col, signal in signals.items():
        df[col] = signal

    df.drop(columns=['rsi'], inplace=True) # 删除中间计算的 'rsi' 列
    return df

def generate_macd_signals(df):
    # MACD 参数
    short_ema = 12
    long_ema = 26
    signal_ema = 9

    # 计算 EMA
    ema_short = df['close'].ewm(span=short_ema, adjust=False).mean()
    ema_long = df['close'].ewm(span=long_ema, adjust=False).mean()

    # 计算 MACD 和信号线
    macd = ema_short - ema_long
    macd_signal = macd.ewm(span=signal_ema, adjust=False).mean()
    macd_histogram = macd - macd_signal

    signals = {}

    # 1. MACD 金叉死叉信号
    signals['MACD_Cross_Buy'] = np.where(
        (macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1)), 1, 0)
    signals['MACD_Cross_Sell'] = np.where(
        (macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1)), 1, 0)

    # 2. MACD Histogram 信号
    signals['MACD_Histogram_Buy'] = np.where(
        (macd_histogram > 0) & (macd_histogram.shift(1) <= 0), 1, 0)
    signals['MACD_Histogram_Sell'] = np.where(
        (macd_histogram < 0) & (macd_histogram.shift(1) >= 0), 1, 0)

    # 3. MACD Histogram 零轴上金叉
    signals['MACD_Histogram_AboveZero_Buy'] = np.where(
        (macd_histogram > 0) & (macd_histogram > macd_histogram.shift(1)), 1, 0)

    # 4. MACD Histogram 零轴下死叉
    signals['MACD_Histogram_BelowZero_Sell'] = np.where(
        (macd_histogram < 0) & (macd_histogram < macd_histogram.shift(1)), 1, 0)

    # 5. MACD 向上穿越零轴
    signals['MACD_ZeroCross_Up_Buy'] = np.where(
        (macd > 0) & (macd.shift(1) <= 0), 1, 0)

    # 6. MACD 向下穿越零轴
    signals['MACD_ZeroCross_Down_Sell'] = np.where(
        (macd < 0) & (macd.shift(1) >= 0), 1, 0)

    # 7. MACD 信号线向上穿越零轴
    signals['MACD_Signal_ZeroCross_Up_Buy'] = np.where(
        (macd_signal > 0) & (macd_signal.shift(1) <= 0), 1, 0)

    # 8. MACD 信号线向下穿越零轴
    signals['MACD_Signal_ZeroCross_Down_Sell'] = np.where(
        (macd_signal < 0) & (macd_signal.shift(1) >= 0), 1, 0)

    df = df.assign(**signals)

    # 删除过程生成的字段
    columns_to_drop = ['ema_short', 'ema_long', 'macd', 'macd_signal', 'macd_histogram']
    # 注意：这里只删除临时计算列，不删除原始df中的数据
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    return df

def generate_bollinger_band_signals(df):
    # 布林带参数
    bollinger_window = 20
    bollinger_std = 2

    # 计算布林带
    rolling_mean = df['close'].rolling(window=bollinger_window).mean()
    rolling_std = df['close'].rolling(window=bollinger_window).std(ddof=0)  # ddof=0 保持与其他软件一致
    bollinger_upper = rolling_mean + bollinger_std * bollinger_std
    bollinger_lower = rolling_mean - bollinger_std * bollinger_std

    # 初始化信号字典
    signals = {}

    # 布林带突破信号
    signals['Bollinger_Breakout_Buy'] = np.where(df['close'] > bollinger_upper, 1, 0)
    signals['Bollinger_Breakout_Sell'] = np.where(df['close'] < bollinger_lower, 1, 0)

    # 布林带触及信号
    signals['Bollinger_Lower_Band_Touch_Buy'] = np.where(df['close'] <= bollinger_lower, 1, 0)
    signals['Bollinger_Upper_Band_Touch_Sell'] = np.where(df['close'] >= bollinger_upper, 1, 0)

    # 布林带反转突破信号
    signals['Bollinger_Lower_Band_Reversal_Buy'] = np.where(
        (df['close'] > bollinger_lower) & (df['close'].shift(1) <= bollinger_lower.shift(1)), 1, 0)
    signals['Bollinger_Upper_Band_Reversal_Sell'] = np.where(
        (df['close'] < bollinger_upper) & (df['close'].shift(1) >= bollinger_upper.shift(1)), 1, 0)

    # 布林带中线交叉信号
    signals['Bollinger_Middle_Cross_Above_Buy'] = np.where(
        (df['close'] > rolling_mean) & (df['close'].shift(1) <= rolling_mean.shift(1)), 1, 0)
    signals['Bollinger_Middle_Cross_Below_Sell'] = np.where(
        (df['close'] < rolling_mean) & (df['close'].shift(1) >= rolling_mean.shift(1)), 1, 0)

    # 布林带宽度收缩信号 - 买入信号（布林带变窄且价格上穿中线）
    bollinger_band_width = bollinger_upper - bollinger_lower
    bb_width_shrinking = bollinger_band_width < bollinger_band_width.rolling(window=5).mean()
    signals['Bollinger_Width_Shrinking_Buy'] = np.where(
        bb_width_shrinking & (df['close'] > rolling_mean), 1, 0)

    # 布林带宽度收缩信号 - 卖出信号（布林带变窄且价格下穿中线）
    signals['Bollinger_Width_Shrinking_Sell'] = np.where(
        bb_width_shrinking & (df['close'] < rolling_mean), 1, 0)

    # 将信号添加到 DataFrame，并删除中间过程字段
    df_with_signals = df.assign(**signals)

    return df_with_signals

def generate_ema_cross_signals(df):
    """
    生成基于 EMA 交叉和价格与 EMA 关系的交易信号。

    Args:
        df: 包含 'close' 列的 Pandas DataFrame。

    Returns:
        包含新增信号列的 Pandas DataFrame。
    """
    ema_short_window = 12
    ema_long_window = 26

    ema_short = df['close'].ewm(span=ema_short_window, adjust=False).mean()
    ema_long = df['close'].ewm(span=ema_long_window, adjust=False).mean()

    signals = {}

    # 1. 指数移动平均线 (EMA) 黄金交叉买入信号
    signals['EMA_Golden_Cross_Buy'] = np.where(
        (ema_short > ema_long) & (ema_short.shift(1) <= ema_long.shift(1)), 1, 0)

    # 2. 指数移动平均线 (EMA) 死亡交叉卖出信号
    signals['EMA_Death_Cross_Sell'] = np.where(
        (ema_short < ema_long) & (ema_short.shift(1) >= ema_long.shift(1)), 1, 0)

    # 3. 价格上穿短期 EMA 买入信号
    signals['Price_Cross_Up_EMA_Short_Buy'] = np.where(
        (df['close'] > ema_short) & (df['close'].shift(1) <= ema_short.shift(1)), 1, 0)

    # 4. 价格下穿短期 EMA 卖出信号
    signals['Price_Cross_Down_EMA_Short_Sell'] = np.where(
        (df['close'] < ema_short) & (df['close'].shift(1) >= ema_short.shift(1)), 1, 0)

    # 5. 价格上穿长期 EMA 买入信号 (可以根据策略选择是否添加)
    signals['Price_Cross_Up_EMA_Long_Buy'] = np.where(
        (df['close'] > ema_long) & (df['close'].shift(1) <= ema_long.shift(1)), 1, 0)

    # 6. 价格下穿长期 EMA 卖出信号 (可以根据策略选择是否添加)
    signals['Price_Cross_Down_EMA_Long_Sell'] = np.where(
        (df['close'] < ema_long) & (df['close'].shift(1) >= ema_long.shift(1)), 1, 0)

    df = df.assign(**signals)
    return df

def generate_stochastic_signals(df):
    """
    生成基于随机指标的更多买卖信号。

    Args:
        df: 包含 'high', 'low', 'close' 列的 Pandas DataFrame。

    Returns:
        Pandas DataFrame，新增了包含买卖信号的列。
    """
    stoch_window = 14
    low_min = df['low'].rolling(window=stoch_window, min_periods=1).min()
    high_max = df['high'].rolling(window=stoch_window, min_periods=1).max()
    high_low_range = high_max - low_min
    high_low_range = high_low_range.replace(0, np.nan) # 避免除以零

    percent_k = ((df['close'] - low_min) / high_low_range) * 100
    percent_d = percent_k.rolling(window=3, min_periods=1).mean()

    signals = {}

    # 1. 经典买入信号 (低于 20 并上穿)
    signals['Stochastic_Buy_Classic'] = np.where(
        (percent_k > percent_d) & (percent_k.shift(1) <= percent_d.shift(1)) & (percent_k < 20), 1, 0)

    # 2. 经典卖出信号 (高于 80 并下穿)
    signals['Stochastic_Sell_Classic'] = np.where(
        (percent_k < percent_d) & (percent_k.shift(1) >= percent_d.shift(1)) & (percent_k > 80), 1, 0)

    # 3. 激进买入信号 (低于 30 并上穿)
    signals['Stochastic_Buy_Aggressive'] = np.where(
        (percent_k > percent_d) & (percent_k.shift(1) <= percent_d.shift(1)) & (percent_k < 30), 1, 0)

    # 4. 激进卖出信号 (高于 70 并下穿)
    signals['Stochastic_Sell_Aggressive'] = np.where(
        (percent_k < percent_d) & (percent_k.shift(1) >= percent_d.shift(1)) & (percent_k > 70), 1, 0)

    # 5. 买入信号 (快速线上穿慢速线，不考虑超卖区)
    signals['Stochastic_Buy_Crossover'] = np.where(
        (percent_k > percent_d) & (percent_k.shift(1) <= percent_d.shift(1)), 1, 0)

    # 6. 卖出信号 (快速线下穿慢速线，不考虑超买区)
    signals['Stochastic_Sell_Crossover'] = np.where(
        (percent_k < percent_d) & (percent_k.shift(1) >= percent_d.shift(1)), 1, 0)

    # 7. 超卖区买入信号 (进入超卖区后，%K 上穿某个阈值，例如 10)
    oversold_threshold = 10
    signals['Stochastic_Buy_Oversold'] = np.where(
        (percent_k > oversold_threshold) & (percent_k.shift(1) <= oversold_threshold) & (percent_k < percent_d), 1, 0)

    # 8. 超买区卖出信号 (进入超买区后，%K 下穿某个阈值，例如 90)
    overbought_threshold = 90
    signals['Stochastic_Sell_Overbought'] = np.where(
        (percent_k < overbought_threshold) & (percent_k.shift(1) >= overbought_threshold) & (percent_k > percent_d), 1, 0)

    df = df.assign(**signals)
    return df


def generate_ichimoku_signals(df):
    # 计算 Ichimoku 指标的各条线
    high9 = df['high'].rolling(window=9).max()
    low9 = df['low'].rolling(window=9).min()
    df['tenkan_sen'] = (high9 + low9) / 2

    high26 = df['high'].rolling(window=26).max()
    low26 = df['low'].rolling(window=26).min()
    df['kijun_sen'] = (high26 + low26) / 2

    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    high52 = df['high'].rolling(window=52).max()
    low52 = df['low'].rolling(window=52).min()
    df['senkou_span_b'] = ((high52 + low52) / 2).shift(26)

    df['chikou_span'] = df['close'].shift(-26)

    # 生成买卖信号
    df['Ichimoku_Buy'] = np.where((df['tenkan_sen'] > df['kijun_sen']) & (df['tenkan_sen'].shift(1) <= df['kijun_sen'].shift(1)), 1, 0)
    df['Ichimoku_Sell'] = np.where((df['tenkan_sen'] < df['kijun_sen']) & (df['tenkan_sen'].shift(1) >= df['kijun_sen'].shift(1)), 1, 0)

    return df

def generate_consecutive_and_large_candle_signals(df):
    """
    生成基于连续上涨/下跌天数和大阳线/大阴线的交易信号，并考虑多个参数。

    Args:
        df (pd.DataFrame): 包含 'close' 列的 DataFrame。

    Returns:
        pd.DataFrame: 添加了信号列的 DataFrame。
    """
    df = df.copy()

    # 连续上涨/下跌天数
    n_values = [3, 5, 10]
    df['Price_Change'] = df['close'].diff()
    for n in n_values:
        df[f'Consecutive_Up_{n}'] = (df['Price_Change'] > 0).rolling(window=n).sum() == n
        df[f'Consecutive_Down_{n}'] = (df['Price_Change'] < 0).rolling(window=n).sum() == n
        df[f'Consecutive_Up_{n}_Buy'] = np.where(df[f'Consecutive_Up_{n}'], 1, 0)
        df[f'Consecutive_Down_{n}_Sell'] = np.where(df[f'Consecutive_Down_{n}'], 1, 0)

    # 大阳线/大阴线
    percentage_thresholds = [0.01, 0.02, 0.04]  # 1%, 2%, 4%
    df['Pct_Change'] = df['close'].pct_change()
    for threshold in percentage_thresholds:
        df[f'Big_Up_{int(threshold*100)}_Buy'] = np.where(df['Pct_Change'] > threshold, 1, 0)
        df[f'Big_Down_{int(threshold*100)}_Sell'] = np.where(df['Pct_Change'] < -threshold, 1, 0)

    # 删除中间计算列，保持输出的整洁
    df = df.drop(columns=['Price_Change', 'Pct_Change'], errors='ignore')

    return df

def generate_price_extremes_reverse_signals(df,periods=[20]):
    """
    生成价格极值反转信号：
    如果上一个时间点创造了最高或最低价，并且当前价格反转，则生成信号。

    Args:
        df (pd.DataFrame): 包含 'close' 列的 DataFrame。

    Returns:
        pd.DataFrame: 添加了价格极值信号列的 DataFrame。
    """
    signals = {}
    for period in periods:
        # 计算指定周期内的最高价和最低价
        highest_close = df['close'].rolling(window=period).max()
        lowest_close = df['close'].rolling(window=period).min()

        # 卖出信号：上一时间点创造了最高价，且当前价格下跌
        signals[f'Highest_{period}_reverse_Sell'] = np.where(
            (df['close'].shift(1) == highest_close.shift(1)) & (df['close'] < df['close'].shift(1)),
            1,
            0
        )

        # 买入信号：上一时间点创造了最低价，且当前价格上涨
        signals[f'Lowest_{period}_reverse_Buy'] = np.where(
            (df['close'].shift(1) == lowest_close.shift(1)) & (df['close'] > df['close'].shift(1)),
            1,
            0
        )

    # 将信号列添加到原始 DataFrame 中
    df = df.assign(**signals)
    return df

def generate_signals_single_ma(df, periods=[10, 20, 30, 50, 100, 200]):
    """
    使用单一 MA 生成交易信号。

    Args:
        df (pd.DataFrame): 包含 'close' 列的 DataFrame。
        periods (list): 周期列表。

    Returns:
        pd.DataFrame: 添加了单一 MA 信号列的 DataFrame。
    """
    signals = {}
    for period in periods:
        ma = calculate_ma(df['close'], period)
        # 价格上穿 MA
        signals[f'SingleMA_{period}_Up_Buy'] = np.where((df['close'] > ma) & (df['close'].shift(1) <= ma.shift(1)), 1,
                                                        0)
        # 价格下穿 MA
        signals[f'SingleMA_{period}_Down_Sell'] = np.where((df['close'] < ma) & (df['close'].shift(1) >= ma.shift(1)),
                                                           1, 0)

        # 基于当前价格与MA值关系判断
        signals[f'SingleMA_{period}_Above_Buy'] = np.where(df['close'] > ma, 1, 0)
        signals[f'SingleMA_{period}_Below_Sell'] = np.where(df['close'] < ma, 1, 0)

    df = df.assign(**signals)
    return df


def generate_signals_double_ma(df, short_periods=[10, 20, 30], long_periods=[50, 100, 200]):
    """
    使用双 MA 生成交易信号。

    Args:
        df (pd.DataFrame): 包含 'close' 列的 DataFrame。
        short_periods (list): 短周期列表。
        long_periods (list): 长周期列表。

    Returns:
        pd.DataFrame: 添加了双 MA 信号列的 DataFrame。
    """
    signals = {}
    for short_period in short_periods:
        for long_period in long_periods:
            if short_period >= long_period:
                continue  # 确保短周期小于长周期

            short_ma = calculate_ma(df['close'], short_period)
            long_ma = calculate_ma(df['close'], long_period)

            # 金叉
            signals[f'DoubleMA_{short_period}_{long_period}_GoldenCross_Buy'] = np.where(
                (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1)), 1, 0)
            # 死叉
            signals[f'DoubleMA_{short_period}_{long_period}_DeathCross_Sell'] = np.where(
                (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1)), 1, 0)

            # 基于当前快慢均线关系判断
            signals[f'DoubleMA_{short_period}_{long_period}_Above_Buy'] = np.where(short_ma > long_ma, 1, 0)
            signals[f'DoubleMA_{short_period}_{long_period}_Below_Sell'] = np.where(short_ma < long_ma, 1, 0)

    df = df.assign(**signals)
    return df

def generate_price_extremes_signals(df, param_info={"periods": [20]}):
    """
    根据指定周期内的最高价和最低价生成买入和卖出信号。

    Args:
        df (pd.DataFrame): 包含 'close' 列的 DataFrame。
        **kwargs: 可变关键字参数，用于传递不同的配置选项。

    Returns:
        pd.DataFrame: 添加了价格极值信号列的 DataFrame。
    """
    periods = param_info.get("periods", [20])
    signals = {}
    for period in periods:
        # 检查是否为指定周期内的最高价
        highest_close = df['close'].rolling(window=period).max()
        signals[f'Highest_{period}_Sell'] = np.where(df['close'] == highest_close, 1, 0)

        # 检查是否为指定周期内的最低价
        lowest_close = df['close'].rolling(window=period).min()
        signals[f'Lowest_{period}_Buy'] = np.where(df['close'] == lowest_close, 1, 0)

    df = df.assign(**signals)
    return df

def generate_trend_signals1(df, param_info={"period": 1}):
    """
    根据指定周期的数据价格变化生成买入和卖出信号。

    Args:
        df (pd.DataFrame): 包含 'close' 列的 DataFrame。
        period (int, optional):  用于分组数据的周期数。默认为 1，即每行数据单独比较。

    Returns:
        pd.DataFrame: 添加了 'buy_signal' 和 'sell_signal' 列的 DataFrame。
    """
    period = param_info.get("period", 1)

    if period < 1:
        raise ValueError("Period must be at least 1.")

    # 计算指定周期的价格变化
    price_change = df['close'].diff(periods=period)

    # 生成买入信号：涨幅大于 0
    df['change_Buy'] = np.where(price_change > 0, 1, 0)

    # 生成卖出信号：涨幅小于 0
    df['change_Sell'] = np.where(price_change < 0, 1, 0)

    return df


def generate_trend_signals(df, param_info={"period": 1}):
    """
    根据指定周期的数据价格变化生成买入和卖出信号。

    Args:
        df (pd.DataFrame): 包含 'close' 列的 DataFrame。
        period (int, optional):  用于分组数据的周期数。默认为 1，即每行数据单独比较。

    Returns:
        pd.DataFrame: 添加了 'buy_signal' 和 'sell_signal' 列的 DataFrame。
    """
    period = param_info.get("period", 1)
    theshold = param_info.get("theshold", 0.1)
    window = 2000

    # 计算指定周期的价格变化
    price_change = df['close'].diff(periods=period)

    # 计算最近100周期的最大值和最小值
    df['roll_max_100'] = df['close'].rolling(window=window, min_periods=1).max()
    df['roll_min_100'] = df['close'].rolling(window=window, min_periods=1).min()
    df['cha'] = df['roll_max_100'] - df['roll_min_100']

    # 生成买入信号：涨幅大于 0 且 close 不大于最近100周期最大值的90%
    df['change_Buy'] = np.where((price_change > 0) & (df['close'] < (theshold * df['cha'] + df['roll_min_100'])), 1, 0)

    # 生成卖出信号：涨幅小于 0 且 close 的 90% 不小于最近100周期最小值
    df['change_Sell'] = np.where((price_change < 0) & (df['close'] > (df['roll_max_100'] - theshold * df['cha'])), 1, 0)

    # 删除辅助列
    # df.drop(columns=['roll_max_100', 'roll_min_100'], inplace=True)

    return df

def generate_price_unextremes_signals(df, param_info={"periods": [20]}):
    """
    根据指定周期内的最高价和最低价生成买入和卖出信号。

    Args:
        df (pd.DataFrame): 包含 'close' 列的 DataFrame。

    Returns:
        pd.DataFrame: 添加了价格极值信号列的 DataFrame。
    """
    periods = param_info.get("periods", [20])
    signals = {}
    for period in periods:
        # 检查是否为指定周期内的最高价
        highest_close = df['close'].rolling(window=period).max()
        signals[f'Highest_{period}_Buy'] = np.where(df['close'] == highest_close, 1, 0)

        # 检查是否为指定周期内的最低价
        lowest_close = df['close'].rolling(window=period).min()
        signals[f'Lowest_{period}_Sell'] = np.where(df['close'] == lowest_close, 1, 0)

    df = df.assign(**signals)
    return df

def generate_cci_signals(df, cci_periods=[20], buy_threshold=-100, sell_threshold=100):
    """
    生成 CCI 买卖信号。

    Args:
        df: 包含 'high', 'low', 'close' 列的 Pandas DataFrame。
        cci_periods: CCI 周期列表, 默认为 [20]。
        buy_threshold: 买入阈值，默认为 -100。
        sell_threshold: 卖出阈值，默认为 100。

    Returns:
        包含 CCI 买卖信号的 Pandas DataFrame。
    """
    df_copy = df.copy()  # 创建 df 的副本以避免修改原始数据

    for cci_period in cci_periods:
        tp = (df_copy['high'] + df_copy['low'] + df_copy['close']) / 3
        ma_tp = tp.rolling(window=cci_period, min_periods=1).mean()
        mad = tp.rolling(window=cci_period, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        cci = (tp - ma_tp) / (0.015 * mad)

        # CCI 买卖信号
        df_copy = df_copy.assign(
            **{
                f'CCI_{cci_period}_Buy': np.where((cci > buy_threshold) & (cci.shift(1) <= buy_threshold), 1, 0),
                f'CCI_{cci_period}_Sell': np.where((cci < sell_threshold) & (cci.shift(1) >= sell_threshold), 1, 0)
            }
        )

    return df_copy


def generate_adx_signals(df):
    """
    生成 ADX 相关的买卖信号。

    Args:
        df: 包含 'high', 'low', 'close' 列的 Pandas DataFrame。

    Returns:
        Pandas DataFrame，包含原始数据和生成的信号列。
    """
    adx_period = 14

    # 计算 +DM, -DM 和 TR
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # 平滑 +DM, -DM 和 TR
    plus_dm_smooth = pd.Series(plus_dm, index=df.index).rolling(window=adx_period, min_periods=1).sum()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).rolling(window=adx_period, min_periods=1).sum()
    tr_smooth = tr.rolling(window=adx_period, min_periods=1).sum()

    # 计算 +DI, -DI
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)

    # 计算 DX 和 ADX
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=adx_period, min_periods=1).mean()

    # 添加 ADX, +DI, -DI 到 DataFrame (便于后续分析，不用于生成信号可以注释掉)
    # df['ADX'] = adx
    # df['Plus_DI'] = plus_di
    # df['Minus_DI'] = minus_di

    # --- 生成信号 ---
    # 1. 基于 +DI 和 -DI 交叉的信号
    df['ADX_Cross_Buy'] = np.where((plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1)), 1, 0)
    df['ADX_Cross_Sell'] = np.where((plus_di < minus_di) & (plus_di.shift(1) >= minus_di.shift(1)), 1, 0)

    # 2. 基于 ADX 趋势的信号 (ADX > 25 表示趋势存在)
    df['ADX_Trend_Buy'] = np.where((adx > 25) & (plus_di > minus_di), 1, 0)
    df['ADX_Trend_Sell'] = np.where((adx > 25) & (plus_di < minus_di), 1, 0)

    # 3. 结合 ADX 趋势和 +DI, -DI 交叉的信号
    df['ADX_Combined_Buy'] = np.where((adx > 25) & (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1)), 1, 0)
    df['ADX_Combined_Sell'] = np.where((adx > 25) & (plus_di < minus_di) & (plus_di.shift(1) >= minus_di.shift(1)), 1,
                                       0)

    # 4. 基于ADX本身趋势的信号
    df['ADX_Itself_Buy'] = np.where((adx > adx.shift(1)) & (adx.shift(1) < adx.shift(2)) & (plus_di > minus_di), 1, 0)
    df['ADX_Itself_Sell'] = np.where((adx < adx.shift(1)) & (adx.shift(1) > adx.shift(2)) & (plus_di < minus_di), 1, 0)

    return df


def generate_obv_signals(df):
    """
    生成多种基于OBV的交易信号，并删除过程中生成的临时字段。

    Args:
        df: 包含'close'和'volume'列的DataFrame。

    Returns:
        添加了信号列的DataFrame。
    """

    # 计算OBV
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv'] = obv  # 将OBV添加到DataFrame，方便后续计算，最后会删除

    # 1. OBV 简单移动平均 (SMA) 信号
    obv_sma_short = obv.rolling(window=5).mean()
    obv_sma_long = obv.rolling(window=20).mean()
    df['OBV_SMA_Buy'] = np.where((obv_sma_short > obv_sma_long) & (obv_sma_short.shift(1) <= obv_sma_long.shift(1)), 1, 0)
    df['OBV_SMA_Sell'] = np.where((obv_sma_short < obv_sma_long) & (obv_sma_short.shift(1) >= obv_sma_long.shift(1)), 1, 0)

    # 2. OBV 指数移动平均 (EMA) 信号 (原代码中的信号)
    obv_ema = obv.ewm(span=20, adjust=False).mean()
    df['OBV_EMA_Buy'] = np.where((obv > obv_ema) & (obv.shift(1) <= obv_ema.shift(1)), 1, 0)
    df['OBV_EMA_Sell'] = np.where((obv < obv_ema) & (obv.shift(1) >= obv_ema.shift(1)), 1, 0)

    # 3. OBV 斜率信号
    obv_slope = obv.diff(5)  # 计算5周期OBV差值，可以调整周期
    df['OBV_Slope_Buy'] = np.where((obv_slope > 0) & (obv_slope.shift(1) <= 0), 1, 0)
    df['OBV_Slope_Sell'] = np.where((obv_slope < 0) & (obv_slope.shift(1) >= 0), 1, 0)

    # 4. OBV 与价格背离信号
    price_sma = df['close'].rolling(window=20).mean()
    df['OBV_Divergence_Buy'] = np.where((df['close'] < price_sma) & (obv > obv_ema), 1, 0) # 价格下跌，OBV上升
    df['OBV_Divergence_Sell'] = np.where((df['close'] > price_sma) & (obv < obv_ema), 1, 0) # 价格上涨，OBV下跌

    # 删除临时字段
    df.drop(columns=['obv'], inplace=True)

    return df



def generate_williams_r_signals(df, wr_period=14, overbought_threshold=-20, oversold_threshold=-80,
                                generate_continuous_signal=False, generate_multi_level_signal=False,
                                use_trend_filter=False, trend_ma_period=50):
    """
    生成基于威廉指标 (Williams %R) 的交易信号。

    参数:
        df: 包含 'high', 'low', 'close' 列的 DataFrame。
        wr_period: 威廉指标的计算周期。
        overbought_threshold: 超买阈值。
        oversold_threshold: 超卖阈值。
        generate_continuous_signal: 是否生成连续信号 (Williams R 值)。
        generate_multi_level_signal: 是否生成多级信号。
        use_trend_filter: 是否使用趋势过滤。
        trend_ma_period: 用于趋势过滤的移动平均线周期。

    返回:
        包含原始数据和信号列的 DataFrame。
    """

    # 计算威廉指标
    highest_high = df['high'].rolling(window=wr_period, min_periods=1).max()
    lowest_low = df['low'].rolling(window=wr_period, min_periods=1).min()
    williams_r = (highest_high - df['close']) / (highest_high - lowest_low) * -100

    # 趋势过滤 (可选)
    if use_trend_filter:
        trend_ma = df['close'].rolling(window=trend_ma_period, min_periods=1).mean()
        uptrend = df['close'] > trend_ma
        downtrend = df['close'] < trend_ma
    else:
        uptrend = downtrend = pd.Series([True]*len(df), index=df.index)

    # 初始化信号字典
    signals = {}

    # 原始买卖信号
    buy_signal = (williams_r > oversold_threshold) & (williams_r.shift(1) <= oversold_threshold) & uptrend
    sell_signal = (williams_r < overbought_threshold) & (williams_r.shift(1) >= overbought_threshold) & downtrend
    signals['WilliamsR_Buy'] = np.where(buy_signal, 1, 0)
    signals['WilliamsR_Sell'] = np.where(sell_signal, 1, 0)

    # 多级信号 (可选)
    if generate_multi_level_signal:
        signals['WilliamsR_MultiLevel_Buy'] = np.where(williams_r < oversold_threshold, 1,
                                                       np.where(williams_r < oversold_threshold + 10, 0.5, 0))
        signals['WilliamsR_MultiLevel_Sell'] = np.where(williams_r > overbought_threshold, 1,
                                                        np.where(williams_r > overbought_threshold - 10, 0.5, 0))

    # 如果需要生成连续的威廉指标信号
    if generate_continuous_signal:
        signals['WilliamsR_Value'] = williams_r

    # 将信号添加到 DataFrame
    df = df.assign(**signals)

    return df


def generate_cmf_signals(df, cmf_periods=[20, 40], cmf_thresholds=[0, 0.1, -0.1]):
    """
    生成蔡金钱流量指标 (Chaikin Money Flow, CMF) 信号，并添加到 DataFrame 中。

    参数:
        df: 包含 'high', 'low', 'close', 'volume' 列的 DataFrame。
        cmf_periods: CMF 计算周期的列表，例如 [20, 40]。
        cmf_thresholds: CMF 阈值的列表，用于生成买卖信号，例如 [0, 0.05, -0.05]。

    返回:
        包含 CMF 信号的 DataFrame。
    """

    df = df.copy()  # Create a copy to avoid modifying the original DataFrame

    for cmf_period in cmf_periods:
        # 1. 计算 CMF
        hl_diff = df['high'] - df['low']
        hl_diff = hl_diff.replace(0, np.nan)  # 避免除以零错误
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / hl_diff
        mfm = mfm.fillna(0)  # 填充 NaN 值为 0
        mfv = mfm * df['volume']
        cmf = mfv.rolling(window=cmf_period, min_periods=1).sum() / df['volume'].rolling(window=cmf_period,
                                                                                         min_periods=1).sum()

        # 2. 生成信号
        for threshold in cmf_thresholds:
            # 2.1 CMF 上穿阈值买入信号
            df[f'CMF_{cmf_period}_Above_{threshold}_Buy'] = np.where(
                (cmf > threshold) & (cmf.shift(1) <= threshold), 1, 0
            )
            # 2.2 CMF 下穿阈值卖出信号
            df[f'CMF_{cmf_period}_Below_{threshold}_Sell'] = np.where(
                (cmf < threshold) & (cmf.shift(1) >= threshold), 1, 0
            )

        # 2.3 CMF 持续在零线上方的买入信号
        df[f'CMF_{cmf_period}_Above_Zero_Buy'] = np.where(cmf > 0, 1, 0)
        # 2.4 CMF 持续在零线下方的卖出信号
        df[f'CMF_{cmf_period}_Below_Zero_Sell'] = np.where(cmf < 0, 1, 0)

        # 3. 删除中间计算字段，只在最后一次循环删除
        if cmf_period == cmf_periods[-1]:
            df.drop(columns=['hl_diff', 'mfm', 'mfv'], errors='ignore', inplace=True)

    return df

def generate_mfi_signals(df):
    # 14. 资金流量指标 (MFI) 信号
    signals = {}
    mfi_period = 14
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_money_flow = np.where(typical_price > typical_price.shift(1),
                                   raw_money_flow, 0)
    negative_money_flow = np.where(typical_price < typical_price.shift(1),
                                   raw_money_flow, 0)

    positive_mf_sum = pd.Series(positive_money_flow).rolling(window=mfi_period, min_periods=1).sum()
    negative_mf_sum = pd.Series(negative_money_flow).rolling(window=mfi_period, min_periods=1).sum()

    mfr = positive_mf_sum / negative_mf_sum
    mfi = 100 - (100 / (1 + mfr))

    # 原始 MFI 买卖信号
    signals['MFI_Buy'] = np.where((mfi < 20) & (mfi.shift(1) >= 20), 1, 0)
    signals['MFI_Sell'] = np.where((mfi > 80) & (mfi.shift(1) <= 80), 1, 0)

    # 基于不同阈值的 MFI 买卖信号
    signals['MFI_Buy_30'] = np.where((mfi < 30) & (mfi.shift(1) >= 30), 1, 0)
    signals['MFI_Sell_70'] = np.where((mfi > 70) & (mfi.shift(1) <= 70), 1, 0)

    # MFI 突破水平信号
    signals['MFI_Cross_Above_20_Buy'] = np.where(mfi >= 20, 1, 0)
    signals['MFI_Cross_Below_80_Sell'] = np.where(mfi <= 80, 1, 0)

    #  MFI 持续在超卖/超买区间的信号
    signals['MFI_Below_20_Buy'] = np.where(mfi < 20, 1, 0)
    signals['MFI_Above_80_Sell'] = np.where(mfi > 80, 1, 0)

    df = df.assign(**signals)

    # 删除过程生成的字段 (这里实际上没有生成额外的中间字段)
    # 如果有中间计算字段，可以使用类似下面的代码删除
    # columns_to_drop = ['typical_price', 'raw_money_flow', ...]
    # df = df.drop(columns=columns_to_drop, errors='ignore') # errors='ignore' 防止列不存在时报错

    return df
def generate_roc_signals(df):
    # 15. 变动速率 (ROC) 信号
    signals = {}
    roc_periods = [5, 12, 20]  # 尝试不同的 ROC 周期

    for period in roc_periods:
        roc = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        roc_name = f'ROC_{period}'

        # 基本的 ROC 零轴交叉信号
        signals[f'{roc_name}_Buy'] = np.where((roc > 0) & (roc.shift(1) <= 0), 1, 0)
        signals[f'{roc_name}_Sell'] = np.where((roc < 0) & (roc.shift(1) >= 0), 1, 0)

        # 基于 ROC 数值的信号 (例如，高于/低于特定阈值)
        signals[f'{roc_name}_Above_10_Buy'] = np.where(roc > 10, 1, 0)
        signals[f'{roc_name}_Below_minus_10_Sell'] = np.where(roc < -10, 1, 0)

        # 基于 ROC 变化的信号 (例如，ROC 加速上涨/下跌)
        signals[f'{roc_name}_Accelerate_Buy'] = np.where((roc > 0) & (roc > roc.shift(1)), 1, 0)
        signals[f'{roc_name}_Accelerate_Sell'] = np.where((roc < 0) & (roc < roc.shift(1)), 1, 0)

    df = df.assign(**signals)

    # 删除过程中生成的字段 (例如，单独的 ROC 列，如果存在的话)
    # 这里我们直接生成信号，没有单独的 ROC 列，所以不需要删除

    return df

def generate_donchian_channel_signals(df):
    # 16. 唐奇安通道 (Donchian Channels) 信号
    signals = {}

    # 定义不同的唐奇安通道周期
    donchian_periods = [10, 20, 30]

    for period in donchian_periods:
        donchian_high = df['high'].rolling(window=period, min_periods=1).max()
        donchian_low = df['low'].rolling(window=period, min_periods=1).min()

        # 唐奇安通道突破信号 (不同周期)
        signals[f'Donchian_{period}_Breakout_Buy'] = np.where(df['close'] > donchian_high.shift(1), 1, 0)
        signals[f'Donchian_{period}_Breakout_Sell'] = np.where(df['close'] < donchian_low.shift(1), 1, 0)

        # 结合唐奇安通道的信号 (例如，价格接近通道边界)
        close_to_high = donchian_high.shift(1) - df['close']
        close_to_low = df['close'] - donchian_low.shift(1)

        signals[f'Donchian_{period}_NearHigh_Buy'] = np.where((df['close'] < donchian_high.shift(1)) & (close_to_high < df['close'] * 0.01), 1, 0) # 接近上限一定比例时
        signals[f'Donchian_{period}_NearLow_Sell'] = np.where((df['close'] > donchian_low.shift(1)) & (close_to_low < df['close'] * 0.01), 1, 0)  # 接近下限一定比例时

    df = df.assign(**signals)

    # 清理过程生成的字段 (这里不需要额外清理，因为我们直接将信号添加到 signals 字典)

    return df

def generate_keltner_channel_signals(df):
    # 17. 肯特纳通道 (Keltner Channels) 信号
    signals = {}
    kc_ema_period = 20
    kc_multiplier = 2
    kc_atr_period = 10

    keltner_ema = df['close'].ewm(span=kc_ema_period, adjust=False).mean()

    # 计算真实波幅 (TR)
    tr_kc = pd.DataFrame({
        'high_low': df['high'] - df['low'],
        'high_close': abs(df['high'] - df['close'].shift(1)),
        'low_close': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)

    # 计算平均真实波幅 (ATR)
    atr_kc = tr_kc.rolling(window=kc_atr_period, min_periods=1).mean()

    keltner_upper = keltner_ema + kc_multiplier * atr_kc
    keltner_lower = keltner_ema - kc_multiplier * atr_kc

    # 肯特纳通道买卖信号
    signals['Keltner_Breakout_Buy'] = np.where(df['close'] > keltner_upper, 1, 0)
    signals['Keltner_Breakout_Sell'] = np.where(df['close'] < keltner_lower, 1, 0)

    # 新增信号：价格触及上轨后回落卖出
    signals['Keltner_TouchUpper_Sell'] = np.where((df['close'].shift(1) > keltner_upper.shift(1)) & (df['close'] <= keltner_upper), 1, 0)

    # 新增信号：价格触及下轨后反弹买入
    signals['Keltner_TouchLower_Buy'] = np.where((df['close'].shift(1) < keltner_lower.shift(1)) & (df['close'] >= keltner_lower), 1, 0)

    # 新增信号：价格从上轨内跌破中轨卖出
    signals['Keltner_BreakMidFromUpper_Sell'] = np.where((df['close'].shift(1) > keltner_ema.shift(1)) & (df['close'] <= keltner_ema) & (df['close'].shift(1) >= keltner_upper.shift(1)), 1, 0)

    # 新增信号：价格从下轨内突破中轨买入
    signals['Keltner_BreakMidFromLower_Buy'] = np.where((df['close'].shift(1) < keltner_ema.shift(1)) & (df['close'] >= keltner_ema) & (df['close'].shift(1) <= keltner_lower.shift(1)), 1, 0)

    df = df.assign(**signals)
    return df

# 回测策略
def backtest_strategy(signal_data_df):
    """
    回测策略，生成适合保存为 CSV 的扁平化结果数据。

    Args:
        signal_data_df (pd.DataFrame): 包含信号列和目标列的 DataFrame。
                                        需要包含 '_Buy' 或 '_Sell' 结尾的列作为买入/卖出信号，
                                        以及包含 'max' 的列作为目标列（未来涨跌幅）。

    Returns:
        pd.DataFrame: 包含扁平化回测结果的 DataFrame。
    """
    start_time = time.time()

    # 提取买入和卖出信号列，以及目标列
    buy_signals = [col for col in signal_data_df.columns if col.endswith('_Buy')]
    sell_signals = [col for col in signal_data_df.columns if col.endswith('_Sell')]
    target_cols = [col for col in signal_data_df.columns if 'max' in col]
    signals = buy_signals + sell_signals

    if not target_cols:
        print("警告：未找到包含 'max' 的目标列。")
        return pd.DataFrame()

    results = []
    thresholds = [0.1, 0.15, 0.2]
    total_samples = len(signal_data_df)

    # -----------------------------
    # 优化点 1：预先计算并缓存 baseline_performance
    # -----------------------------
    # 用于存储 baseline_performance，减少重复计算
    baseline_performances = {}

    for target_col in target_cols:
        key_target_col = target_col.replace('close_max_', '')

        # 对于每个目标列，预先计算并缓存不同阈值下的性能指标
        for threshold in thresholds:
            # 计算满足条件的样本数量
            count = (signal_data_df[target_col] > threshold).sum()
            # 基准线性能 = 满足条件的样本数量 / 总样本数量
            baseline_performance = round(count / total_samples, 4)
            # 缓存结果
            baseline_performances[(target_col, threshold)] = baseline_performance

            # 将结果添加到 results 列表中
            results.append({
                "signal": "Baseline",
                "target": key_target_col,
                "threshold": threshold,
                "signal_performance": baseline_performance,
                "diff_vs_baseline": 0.0,
                "count": total_samples,
                "ratio": 1.0
            })

    # -----------------------------
    # 优化点 2：使用布尔掩码和向量化计算
    # -----------------------------
    # 预先计算所有信号的布尔掩码和计数
    signal_masks = {}
    signal_counts = {}
    signal_ratios = {}

    for signal in signals:
        # 计算信号掩码
        signal_mask = (signal_data_df[signal] == 1)
        signal_masks[signal] = signal_mask
        # 计算信号对应的样本数量和比例
        signal_count = signal_mask.sum()
        signal_counts[signal] = signal_count
        signal_ratio = round(signal_count / total_samples, 4) if total_samples > 0 else 0
        signal_ratios[signal] = signal_ratio

    # 预先计算目标列和阈值的布尔掩码
    target_masks = {}
    for target_col in target_cols:
        target_masks[target_col] = {}
        for threshold in thresholds:
            # 计算目标列在不同阈值下的掩码
            target_mask = (signal_data_df[target_col] > threshold)
            target_masks[target_col][threshold] = target_mask

    # -----------------------------
    # 优化点 3：减少嵌套循环层次
    # -----------------------------
    # 遍历所有信号，目标列和阈值，计算结果
    for signal in signals:
        signal_mask = signal_masks[signal]
        signal_count = signal_counts[signal]
        signal_ratio = signal_ratios[signal]

        for target_col in target_cols:
            key_target_col = target_col.replace('close_max_', '')
            for threshold in thresholds:
                baseline_performance = baseline_performances[(target_col, threshold)]

                if signal_count > 0:
                    # 使用布尔掩码计算满足条件的数量
                    above_threshold_count = (signal_mask & target_masks[target_col][threshold]).sum()
                    # 当前信号性能 = 满足条件的样本数量 / 信号样本数量
                    current_performance = round(above_threshold_count / signal_count, 4)
                else:
                    current_performance = 0.0

                # 信号性能与基准线性能的差异
                diff_vs_baseline = round(current_performance - baseline_performance, 4)

                # 将结果添加到 results 列表中
                results.append({
                    "signal": signal,
                    "target": key_target_col,
                    "threshold": threshold,
                    "signal_performance": current_performance,
                    "diff_vs_baseline": diff_vs_baseline,
                    "count": signal_count,
                    "ratio": signal_ratio
                })

    # 将 results 转化为 DataFrame 格式
    results_df = pd.DataFrame(results)
    print(f"回测策略完成，耗时 {time.time() - start_time:.2f} 秒。")
    return results_df


def analyze_backtest_results():
    """
     分析回测结果，生成信号与目标列表的映射字典，其中每个信号对应一个目标列表。
     并将映射字典保存到 best_signal_list.json 文件中。
     """
    head_count = 50
    backtest_path = 'kline_data'
    if not os.path.exists(backtest_path):
        os.makedirs(backtest_path)
    data_list = []
    signal_to_target_map = {}  # 初始化信号到目标列表的映射字典

    # 加载 backtest_path 下面所有的 statistic_*.csv 文件
    files = os.listdir(backtest_path)
    files = [file for file in files if file.startswith('statistic_')]
    if len(files) > 0:
        print('已经存在该文件，直接读取')

    for file in files:
        if '1m' in file and '1000000' in file:
            data = pd.read_csv(f'{backtest_path}/{file}')
            data[file] = file
            data_list.append(data)
            # 删除target中包含next的行
            data = data[~data['target'].str.contains('next')]

            data['score'] = data['diff_vs_baseline'] / (1 - data['signal_performance']) * data['ratio']
            # 将 data 按照 signal_performance 降序排序
            data = data.sort_values(by='signal_performance', ascending=False)

            # # 获取 signal_performance 大于 0.8 且 diff_vs_baseline 大于 0.0 的数据
            signal_list = data[(data['signal_performance'] > 0.8) & (data['diff_vs_baseline'] > -0.0)]


            # # 分别获取target包含up和down的前10个
            # up_signal_list = data[(data['target'].str.contains('up'))].head(head_count)
            # down_signal_list = data[(data['target'].str.contains('down'))].head(head_count)
            # signal_list = pd.concat([up_signal_list, down_signal_list])

            # 遍历 signal_list，构建 signal_to_target_map
            for index, row in signal_list.iterrows():
                signal = row['signal']
                target = row['target']
                if signal not in signal_to_target_map:
                    signal_to_target_map[signal] = []
                if target not in signal_to_target_map[signal]:  # 避免重复添加 target
                    signal_to_target_map[signal].append(target)

    # 对每个 signal 的 target 列表进行排序
    for signal in signal_to_target_map:
        signal_to_target_map[signal].sort()

    # 保存 signal_to_target_map 到 best_signal_list.json 文件
    best_signal_list_path = os.path.join(backtest_path, 'best_signal_list.json')
    with open(best_signal_list_path, 'w') as f:
        json.dump(signal_to_target_map, f, indent=4)  # 使用 indent=4 进行美化输出

    return signal_to_target_map

def run():
    """
      正式运行，得到买入或者卖出信号
      :return:
      """
    bar = '1m'
    max_candles = 3000
    backtest_path = 'kline_data'
    best_signal_list_path = os.path.join(backtest_path, 'best_signal_list.json')

    # 加载 signal_to_target_map
    if os.path.exists(best_signal_list_path):
        with open(best_signal_list_path, 'r') as f:
            signal_to_target_map = json.load(f)
    else:
        signal_to_target_map = analyze_backtest_results()  # 如果不存在则运行

    best_signal_list = list(signal_to_target_map.keys())

    # 获取最新数据
    data = get_train_data(max_candles=max_candles, bar=bar)
    signal_data = generate_signals(data)
    # 只保留signal_data最后的60行
    signal_data = signal_data[-1000:]



    # 获取每一行中值为1的列名
    active_signals = []
    for index, row in signal_data.iterrows():
        active_signals_row = [col for col in best_signal_list if col in row and row[col] == 1]
        active_signals.append(active_signals_row)

    # 将active_signals添加到signal_data中
    signal_data['active_signals'] = active_signals

    # 根据active_signals找到对应的target
    corresponding_targets = []
    for signals in signal_data['active_signals']:

        if signals == ['Donchian_30_Breakout_Sell', 'Bollinger_Breakout_Sell', 'BB_Lower_Band_Touch_Buy', 'Donchian_20_Breakout_Sell', 'Keltner_Breakout_Sell', 'Donchian_10_Breakout_Sell', 'Stochastic_Buy', 'Price_Breakout_Sell', 'Volume_Sell']:
            print('yes')
        targets = []
        targets_map = {}
        # 将signals进行排序
        signals = sorted(signals)
        for signal in signals:
            signal_key = signal.split('_')[0]
            if signal_key not in targets_map.keys():
                targets_map[signal_key] = []
            if signal in signal_to_target_map:
                targets_map[signal_key].extend(signal_to_target_map[signal])
        for key in targets_map:
            targets.extend(list(set(targets_map[key])))
        corresponding_targets.append(targets)  # 将集合转换为列表

    signal_data['corresponding_targets'] = corresponding_targets


    # 统计 Buy 和 Sell 信号的数量
    signal_data['buy_count'] = signal_data['corresponding_targets'].apply(lambda x: sum('up' in s for s in x))
    signal_data['sell_count'] = signal_data['corresponding_targets'].apply(lambda x: sum('down' in s for s in x))
    # 计算差值
    signal_data['diff'] = signal_data['buy_count'] - signal_data['sell_count']
    # 计算和
    signal_data['sum'] = signal_data['buy_count'] + signal_data['sell_count']
    # 调整列的顺序
    cols = signal_data.columns.tolist()
    cols = ['active_signals','corresponding_targets', 'buy_count', 'sell_count', 'diff', 'sum', 'close_max_up_t30', 'close_max_down_t30'] + [col for col in cols if col not in ['active_signals', 'corresponding_targets','buy_count', 'sell_count', 'diff', 'sum', 'close_max_down_t30', 'close_max_up_t30']]
    signal_data = signal_data[cols]

    # 在这里可以根据active_signals, buy_count, sell_count进一步处理，例如生成买入卖出信号

    print(signal_data)
    return signal_data


def download_data():
    # df = get_train_data(inst_id='TON-USDT-SWAP', bar='5m', max_candles=1000)
    # df.to_csv('temp/TON_5m_1000.csv', index=False)

    backtest_path = 'kline_data'
    base_file_path = 'origin_data.csv'
    is_reload = True
    inst_id_list = ['BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'SOL-USDT-SWAP', 'TON-USDT-SWAP', 'DOGE-USDT-SWAP', 'XRP-USDT-SWAP', 'OKB-USDT']
    if not os.path.exists(backtest_path):
        os.makedirs(backtest_path)
    bar_list = ['1m']
    #获取当前时间，精确到天
    now = datetime.datetime.now()
    # 转换为可读性格式
    readable_time = now.strftime("%Y-%m-%d")
    max_candles_list = [100000]

    for max_candles in max_candles_list:
        for bar in bar_list:
            for inst_id in inst_id_list:
                final_file_path = f'{backtest_path}/{base_file_path[:-4]}_{bar}_{max_candles}_{inst_id}_{readable_time}.csv'
                # 判断文件是否存在，并且有一定的大小
                if not is_reload and os.path.exists(final_file_path) and os.path.getsize(final_file_path) > 1024:
                    print('已经存在该文件，直接读取')
                else:
                    print(f'不存在该文件，开始获取 {final_file_path}')
                    data = get_train_data(inst_id=inst_id, bar=bar, max_candles=max_candles)
                    data.to_csv(final_file_path, index=False)

def example():
    download_data()
    # run_backtest()
    # analyze_backtest_results()
    # run()


def backtest_strategy_op(signal_data_df):
    """
    回测策略，生成适合保存为 CSV 的扁平化结果数据。

    Args:
        signal_data_df (pd.DataFrame): 包含信号列和目标列的 DataFrame。
                                        需要包含 '_Buy' 或 '_Sell' 结尾的列作为买入/卖出信号，
                                        以及包含 'max' 的列作为目标列（未来涨跌幅）。

    Returns:
        pd.DataFrame: 包含扁平化回测结果的 DataFrame。
    """
    start_time = time.time()

    # 提取买入和卖出信号列，以及目标列
    buy_signals = [col for col in signal_data_df.columns if col.endswith('_Buy')]
    sell_signals = [col for col in signal_data_df.columns if col.endswith('_Sell')]
    target_cols = [col for col in signal_data_df.columns if 'max' in col]
    signals = buy_signals + sell_signals

    if not target_cols:
        print("警告：未找到包含 'max' 的目标列。")
        return pd.DataFrame()

    results = []
    quantiles = [0.01, 0.02, 0.03, 0.04, 0.05]
    total_samples = len(signal_data_df)

    # -----------------------------
    # 优化点 1：预先计算并缓存 baseline_quantile_values
    # -----------------------------
    # 用于存储 baseline_quantile_values，减少重复计算
    baseline_quantile_values = {}
    baseline_avg_values = {}

    for target_col in target_cols:
        key_target_col = target_col.replace('close_max_', '')
        avg_value = signal_data_df[target_col].mean()
        baseline_avg_values[target_col] = avg_value

        # 对于每个目标列，预先计算并缓存不同分位数的值
        for q in quantiles:
            # 计算分位数值
            quantile_value = signal_data_df[target_col].quantile(q)
            # 缓存结果
            baseline_quantile_values[(target_col, q)] = quantile_value

            # 将结果添加到 results 列表中
            results.append({
                "signal": "Baseline",
                "target": key_target_col,
                "quantile": q,
                "quantile_value": quantile_value,
                "diff_vs_baseline": 0.0,  # 基准策略与自身的差异为 0
                "avg_value": avg_value,
                "diff_vs_baseline_avg": 0.0,  # 基准策略与自身的差异为 0
                "count": total_samples,
                "ratio": 1.0
            })

    # -----------------------------
    # 优化点 2：使用布尔掩码和向量化计算
    # -----------------------------
    # 预先计算所有信号的布尔掩码和计数
    signal_masks = {}
    signal_counts = {}
    signal_ratios = {}
    signal_avgs = {}

    for signal in signals:
        # 计算信号掩码
        signal_mask = (signal_data_df[signal] == 1)
        signal_masks[signal] = signal_mask
        # 计算信号对应的样本数量和比例
        signal_count = signal_mask.sum()
        signal_counts[signal] = signal_count
        signal_ratio = round(signal_count / total_samples, 4) if total_samples > 0 else 0
        signal_ratios[signal] = signal_ratio
        for target_col in target_cols:
            signal_avgs[(signal,target_col)] = signal_data_df.loc[signal_mask, target_col].mean()

    # -----------------------------
    # 优化点 3：减少嵌套循环层次
    # -----------------------------
    # 遍历所有信号，目标列和分位数，计算结果
    for signal in signals:
        signal_mask = signal_masks[signal]
        signal_count = signal_counts[signal]
        signal_ratio = signal_ratios[signal]
        for target_col in target_cols:
            avg_value = signal_avgs[(signal, target_col)]
            baseline_avg_value = baseline_avg_values[target_col]
            key_target_col = target_col.replace('close_max_', '')
            diff_vs_baseline_avg = avg_value - baseline_avg_value
            for q in quantiles:
                baseline_quantile_value = baseline_quantile_values[(target_col, q)]

                if signal_count > 0:
                    # 使用布尔掩码筛选数据并计算分位数值
                    quantile_value = signal_data_df.loc[signal_mask, target_col].quantile(q)
                    # 计算与基准策略的差异
                    diff_vs_baseline = quantile_value - baseline_quantile_value
                else:
                    quantile_value = np.nan
                    diff_vs_baseline = np.nan

                # 将结果添加到 results 列表中
                results.append({
                    "signal": signal,
                    "target": key_target_col,
                    "quantile": q,
                    "quantile_value": quantile_value,
                    "diff_vs_baseline": diff_vs_baseline,
                    "avg_value": avg_value,
                    "diff_vs_baseline_avg": diff_vs_baseline_avg,
                    "count": signal_count,
                    "ratio": signal_ratio
                })
    # 将 results 转化为 DataFrame 格式
    results_df = pd.DataFrame(results)
    print(f"回测策略完成，耗时 {time.time() - start_time:.2f} 秒。")
    return results_df


def run_backtest():
    backtest_path = 'kline_data'
    if not os.path.exists(backtest_path):
        os.makedirs(backtest_path)
    inst_id_list = ['BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'SOL-USDT-SWAP', 'TON-USDT-SWAP', 'DOGE-USDT-SWAP', 'XRP-USDT-SWAP', 'PEPE-USDT-SWAP']

    base_file_path = 'origin_data.csv'
    bar_list = ['1m', '1m', '5m', '15m', '30m', '1h', '4h']
    max_candles_list = [10000000, 1000000]

    for inst_id in inst_id_list:
        for bar in bar_list:
            for max_candles in max_candles_list:
                full_final_file_path = f'{backtest_path}/{base_file_path[:-4]}_{bar}_{max_candles}_{inst_id}.csv'
                full_data = pd.read_csv(full_final_file_path)
                try:
                    final_output_file_path = f'{backtest_path}/statistic_{base_file_path[:-4]}_{bar}_{max_candles}.csv'
                    final_op_output_file_path = f'{backtest_path}/statistic_op_{base_file_path[:-4]}_{bar}_{max_candles}.csv'
                    # 去除后60行
                    data = full_data[:-10000]
                    signal_data = generate_signals(data)
                    # backtest_df = backtest_strategy(signal_data)
                    # backtest_df_op = backtest_strategy_op(signal_data)
                    backtest_df = pd.read_csv(final_output_file_path)
                    backtest_df_op = pd.read_csv(final_op_output_file_path)
                    backtest_df_op['quantile_score1'] = backtest_df_op['quantile_value'] / backtest_df_op['quantile'] * backtest_df_op['ratio'] * backtest_df_op['diff_vs_baseline']
                    backtest_df.to_csv(final_output_file_path, index=False)
                    backtest_df_op.to_csv(final_op_output_file_path, index=False)
                    print(f'已经保存至{final_output_file_path}')
                    return
                except Exception as e:
                    print(e)
                    continue


if __name__ == '__main__':
    example()
    # file_name = 'BTC-USDT-SWAP_1m_20230124_20241218.csv'
    # long_df = get_dist(file_name)
    # # short_df = get_dist('BTC-USDT-SWAP_1m_20240627_20241212.csv')
    # # pass
    # # gen_feature('BTC-USDT-SWAP_1m_20241219_20241220.csv')
    # get_latest_data()