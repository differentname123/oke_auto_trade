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

    while fetched_candles < max_candles:
        try:
            # 调用OKX API获取历史K线数据
            response = marketAPI.get_history_candlesticks(instId=inst_id, bar=bar, after=after, limit=limit)

            if response["code"] != "0":
                print(f"获取K线数据失败，错误代码：{response['code']}，错误消息：{response['msg']}")
                time.sleep(1)
                fail_count += 1
                if fail_count >= 3:
                    print("连续失败次数过多，停止获取。")
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
                # 将时间戳转换为可读性更好的格式
                # print(f"已获取 {fetched_candles} 条K线数据，最新时间：{pd.to_datetime(after, unit='ms')}")

                # 如果获取的数据量小于limit，说明数据已获取完毕
                if len(data) < limit:
                    break

                # 短暂延迟，避免触发API限频
                time.sleep(0.2)

        except Exception as e:
            print(f"获取K线数据时出现异常：{e}")
            break

    # 将所有数据转换为DataFrame
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", "volCcy", "volCcyQuote",
                                         "confirm"])

    # 数据类型转换
    # 将时间戳转换为 datetime 对象，并将其设置为 UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float) / 1000, unit="s", utc=True)

    # 将 UTC 时间转换为北京时间（Asia/Shanghai 时区，UTC+8）
    df["timestamp"] = df["timestamp"].dt.tz_convert('Asia/Shanghai')
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

    # 按时间排序
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df

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

def add_target_variables_op(df, max_decoder_length=30):
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
        for step in range(10, max_decoder_length + 1, 10):  # 未来 1 到 max_decoder_length 分钟
            # 获取未来 step 个时间窗口内的最高价和最低价
            future_max_high = df['high'].shift(-step).rolling(window=step, min_periods=1).max()
            future_min_low = df['low'].shift(-step).rolling(window=step, min_periods=1).min()

            # 计算未来 step 个时间窗口内的最大涨幅和跌幅 (修正部分)
            new_columns[f"{col}_max_up_t{step}"] = (future_max_high - df[col]) / df[col] * 100 #最大涨幅用最高价
            new_columns[f"{col}_max_down_t{step}"] = (df[col] - future_min_low) / df[col] * 100 #最大跌幅用最低价
    # 使用 pd.concat 一次性将所有新列添加到原数据框
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    return df

def get_train_data(inst_id="BTC-USDT-SWAP", bar="1m", limit=100, max_candles=1000):
    # inst_id = "BTC-USDT-SWAP"
    # bar = "1m"
    # limit = 100
    # max_candles = 60 * 24

    # 获取数据
    kline_data = get_kline_data(inst_id=inst_id, bar=bar, limit=limit, max_candles=max_candles)

    if not kline_data.empty:
        # print("成功获取K线数据，开始处理...")

        # 添加时间特征
        # kline_data = add_time_features(kline_data)

        # 添加目标变量
        kline_data = add_target_variables_op(kline_data)

        # 重置索引
        kline_data.reset_index(drop=True, inplace=True)

        # 保存文件
        start_date = kline_data["timestamp"].iloc[0].strftime("%Y%m%d")
        end_date = kline_data["timestamp"].iloc[-1].strftime("%Y%m%d")
        filename = f"{inst_id}_{bar}_{start_date}_{end_date}.csv"

        # kline_data.to_csv(filename, index=False)
        # print(f"数据已保存至文件：{filename}")
        return kline_data
    else:
        print("未能获取到任何K线数据。")

def get_latest_data(max_candles=1000):
    origin_data = get_train_data(max_candles=max_candles)
    feature_df = gen_feature(origin_data)
    return feature_df

def generate_signals(df):
    """
    融合多种方法生成的买入和卖出信号。

    Args:
        df: 包含 'timestamp', 'open', 'high', 'low', 'close', 'volume' 列的 Pandas DataFrame。

    Returns:
        DataFrame:  包含原始数据和生成的信号列的 DataFrame。
    """

    df = df.copy()  # 避免修改原始 DataFrame

    df = generate_volume_signals(df)
    df = generate_price_signals(df)
    df = generate_moving_average_signals(df)
    df = generate_rsi_signals(df)
    df = generate_macd_signals(df)
    df = generate_bollinger_band_signals(df)
    df = generate_ema_cross_signals(df)
    df = generate_stochastic_signals(df)
    df = generate_cci_signals(df)
    df = generate_adx_signals(df)
    df = generate_obv_signals(df)
    df = generate_williams_r_signals(df)
    df = generate_cmf_signals(df)
    df = generate_mfi_signals(df)
    df = generate_roc_signals(df)
    df = generate_donchian_channel_signals(df)
    df = generate_keltner_channel_signals(df)

    return df

def generate_volume_signals(df):
    # 1. 基于成交量的信号
    signals = {}
    signals['Volume_Buy'] = np.where((df['close'].diff() > 0) & (df['volume'].diff() > 0), 1, 0)
    signals['Volume_Sell'] = np.where((df['close'].diff() < 0) & (df['volume'].diff() > 0), 1, 0)
    df = df.assign(**signals)
    return df

def generate_price_signals(df):
    # 2. 基于价格的信号 (简单价格突破)
    signals = {}
    signals['Price_Breakout_Buy'] = np.where(df['close'] > df['high'].shift(1), 1, 0)
    signals['Price_Breakout_Sell'] = np.where(df['close'] < df['low'].shift(1), 1, 0)
    df = df.assign(**signals)
    return df

def generate_moving_average_signals(df):
    # 3. 基于移动平均线的信号 (简单均线交叉)
    short_window = 5
    long_window = 20

    signals = {}
    df['SMA_Short'] = df['close'].rolling(window=short_window, min_periods=1).mean()
    df['SMA_Long'] = df['close'].rolling(window=long_window, min_periods=1).mean()

    signals['MA_Golden_Cross_Buy'] = np.where(
        (df['SMA_Short'] > df['SMA_Long']) & (df['SMA_Short'].shift(1) <= df['SMA_Long'].shift(1)), 1, 0)
    signals['MA_Death_Cross_Sell'] = np.where(
        (df['SMA_Short'] < df['SMA_Long']) & (df['SMA_Short'].shift(1) >= df['SMA_Long'].shift(1)), 1, 0)

    signals['Price_Above_SMA_Short_Buy'] = np.where(
        (df['close'] > df['SMA_Short']) & (df['close'].shift(1) <= df['SMA_Short'].shift(1)), 1, 0)
    signals['Price_Below_SMA_Short_Sell'] = np.where(
        (df['close'] < df['SMA_Short']) & (df['close'].shift(1) >= df['SMA_Short'].shift(1)), 1, 0)

    df = df.assign(**signals)
    df.drop(columns=['SMA_Short', 'SMA_Long'], inplace=True)
    return df

def generate_rsi_signals(df):
    # 4. 相对强弱指数 (RSI) 信号
    rsi_period = 14
    signals = {}
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = down.rolling(window=rsi_period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi

    # RSI 超买超卖信号
    signals['RSI_Threshold_Buy'] = np.where(df['RSI'] < 30, 1, 0)
    signals['RSI_Threshold_Sell'] = np.where(df['RSI'] > 70, 1, 0)

    # RSI 突破信号
    lower_threshold = 30
    upper_threshold = 70
    signals['RSI_Buy'] = np.where(
        (df['RSI'] < lower_threshold) & (df['RSI'].shift(1) >= lower_threshold), 1, 0)
    signals['RSI_Sell'] = np.where(
        (df['RSI'] > upper_threshold) & (df['RSI'].shift(1) <= upper_threshold), 1, 0)

    df = df.assign(**signals)
    df.drop(columns=['RSI'], inplace=True)
    return df

def generate_macd_signals(df):
    # 5. 平滑异同移动平均线 (MACD) 信号
    signals = {}
    short_ema = 12
    long_ema = 26
    signal_ema = 9

    ema_short = df['close'].ewm(span=short_ema, adjust=False).mean()
    ema_long = df['close'].ewm(span=long_ema, adjust=False).mean()
    macd = ema_short - ema_long
    macd_signal = macd.ewm(span=signal_ema, adjust=False).mean()
    macd_histogram = macd - macd_signal

    # MACD 金叉死叉信号
    signals['MACD_Cross_Buy'] = np.where(
        (macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1)), 1, 0)
    signals['MACD_Cross_Sell'] = np.where(
        (macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1)), 1, 0)

    # MACD Histogram 信号
    signals['MACD_Histogram_Buy'] = np.where(
        (macd_histogram > 0) & (macd_histogram.shift(1) <= 0), 1, 0)
    signals['MACD_Histogram_Sell'] = np.where(
        (macd_histogram < 0) & (macd_histogram.shift(1) >= 0), 1, 0)

    df = df.assign(**signals)
    return df

def generate_bollinger_band_signals(df):
    # 6. 布林带信号
    signals = {}
    bollinger_window = 20
    bollinger_std = 2

    rolling_mean = df['close'].rolling(window=bollinger_window).mean()
    rolling_std = df['close'].rolling(window=bollinger_window).std()
    bollinger_upper = rolling_mean + bollinger_std * rolling_std
    bollinger_lower = rolling_mean - bollinger_std * rolling_std

    # 布林带突破信号
    signals['Bollinger_Breakout_Buy'] = np.where(df['close'] > bollinger_upper, 1, 0)
    signals['Bollinger_Breakout_Sell'] = np.where(df['close'] < bollinger_lower, 1, 0)

    # 布林带触及和突破信号
    signals['BB_Lower_Band_Touch_Buy'] = np.where(df['close'] <= bollinger_lower, 1, 0)
    signals['BB_Upper_Band_Touch_Sell'] = np.where(df['close'] >= bollinger_upper, 1, 0)
    signals['BB_Lower_Band_Break_Buy'] = np.where(
        (df['close'] > bollinger_lower) & (df['close'].shift(1) <= bollinger_lower.shift(1)), 1, 0)
    signals['BB_Upper_Band_Break_Sell'] = np.where(
        (df['close'] < bollinger_upper) & (df['close'].shift(1) >= bollinger_upper.shift(1)), 1, 0)

    df = df.assign(**signals)
    return df

def generate_ema_cross_signals(df):
    # 7. 指数移动平均线 (EMA) 交叉信号
    signals = {}
    ema_short_window = 12
    ema_long_window = 26

    ema_short = df['close'].ewm(span=ema_short_window, adjust=False).mean()
    ema_long = df['close'].ewm(span=ema_long_window, adjust=False).mean()

    signals['EMA_Golden_Cross_Buy'] = np.where(
        (ema_short > ema_long) & (
            ema_short.shift(1) <= ema_long.shift(1)), 1, 0)
    signals['EMA_Death_Cross_Sell'] = np.where(
        (ema_short < ema_long) & (
            ema_short.shift(1) >= ema_long.shift(1)), 1, 0)

    df = df.assign(**signals)
    return df

def generate_stochastic_signals(df):
    # 8. 随机指标 (Stochastic Oscillator) 信号
    signals = {}
    stoch_window = 14
    low_min = df['low'].rolling(window=stoch_window, min_periods=1).min()
    high_max = df['high'].rolling(window=stoch_window, min_periods=1).max()
    high_low_range = high_max - low_min
    high_low_range.replace(0, np.nan, inplace=True)
    percent_k = ((df['close'] - low_min) / high_low_range) * 100
    percent_d = percent_k.rolling(window=3, min_periods=1).mean()

    # 随机指标买卖信号
    signals['Stochastic_Buy'] = np.where(
        (percent_k > percent_d) & (percent_k.shift(1) <= percent_d.shift(1)) & (percent_k < 20), 1, 0)
    signals['Stochastic_Sell'] = np.where(
        (percent_k < percent_d) & (percent_k.shift(1) >= percent_d.shift(1)) & (percent_k > 80), 1, 0)

    df = df.assign(**signals)
    return df

def generate_cci_signals(df):
    # 9. 商品通道指数 (CCI) 信号
    signals = {}
    cci_period = 20
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma_tp = tp.rolling(window=cci_period, min_periods=1).mean()
    mad = tp.rolling(window=cci_period, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - ma_tp) / (0.015 * mad)

    # CCI 买卖信号
    signals['CCI_Buy'] = np.where(
        (cci > -100) & (cci.shift(1) <= -100), 1, 0)
    signals['CCI_Sell'] = np.where(
        (cci < 100) & (cci.shift(1) >= 100), 1, 0)

    df = df.assign(**signals)
    return df
def generate_adx_signals(df):
    # 10. 平均方向性指数 (ADX) 信号
    signals = {}
    adx_period = 14
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0), down_move, 0)

    tr_tmp1 = df['high'] - df['low']
    tr_tmp2 = abs(df['high'] - df['close'].shift(1))
    tr_tmp3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr_tmp1, tr_tmp2, tr_tmp3], axis=1).max(axis=1)

    # 将 plus_dm 和 minus_dm 转换为 pandas Series
    plus_dm_series = pd.Series(plus_dm, index=df.index)
    minus_dm_series = pd.Series(minus_dm, index=df.index)

    tr_n = tr.rolling(window=adx_period, min_periods=1).sum()
    plus_dm_n = plus_dm_series.rolling(window=adx_period, min_periods=1).sum()
    minus_dm_n = minus_dm_series.rolling(window=adx_period, min_periods=1).sum()

    plus_di = 100 * (plus_dm_n / tr_n)
    minus_di = 100 * (minus_dm_n / tr_n)

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=adx_period, min_periods=1).mean()

    # ADX 买卖信号
    signals['ADX_Buy'] = np.where(
        (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1)), 1, 0)
    signals['ADX_Sell'] = np.where(
        (plus_di < minus_di) & (plus_di.shift(1) >= minus_di.shift(1)), 1, 0)

    df = df.assign(**signals)
    return df


def generate_obv_signals(df):
    # 11. 能量潮指标 (OBV) 信号
    signals = {}
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    obv_ema = obv.ewm(span=20, adjust=False).mean()

    signals['OBV_Buy'] = np.where(
        (obv > obv_ema) & (obv.shift(1) <= obv_ema.shift(1)), 1, 0)
    signals['OBV_Sell'] = np.where(
        (obv < obv_ema) & (obv.shift(1) >= obv_ema.shift(1)), 1, 0)

    df = df.assign(**signals)
    return df

def generate_williams_r_signals(df):
    # 12. 威廉指标 (Williams %R) 信号
    signals = {}
    wr_period = 14
    highest_high = df['high'].rolling(window=wr_period, min_periods=1).max()
    lowest_low = df['low'].rolling(window=wr_period, min_periods=1).min()
    williams_r = (highest_high - df['close']) / (highest_high - lowest_low) * -100

    # Williams %R 买卖信号
    signals['Williams_%R_Buy'] = np.where(
        (williams_r > -80) & (williams_r.shift(1) <= -80), 1, 0)
    signals['Williams_%R_Sell'] = np.where(
        (williams_r < -20) & (williams_r.shift(1) >= -20), 1, 0)

    df = df.assign(**signals)
    return df

def generate_cmf_signals(df):
    # 13. 蔡金钱流量指标 (Chaikin Money Flow, CMF) 信号
    signals = {}
    cmf_period = 20
    hl_diff = df['high'] - df['low']
    hl_diff = hl_diff.replace(0, np.nan)  # 避免除以零错误
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / hl_diff
    mfm = mfm.fillna(0)  # 填充 NaN 值为 0
    mfv = mfm * df['volume']
    cmf = mfv.rolling(window=cmf_period, min_periods=1).sum() / df['volume'].rolling(window=cmf_period, min_periods=1).sum()

    # CMF 买卖信号
    signals['CMF_Buy'] = np.where(
        (cmf > 0) & (cmf.shift(1) <= 0), 1, 0)
    signals['CMF_Sell'] = np.where(
        (cmf < 0) & (cmf.shift(1) >= 0), 1, 0)

    df = df.assign(**signals)
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

    # MFI 买卖信号
    signals['MFI_Buy'] = np.where((mfi < 20) & (mfi.shift(1) >= 20), 1, 0)
    signals['MFI_Sell'] = np.where((mfi > 80) & (mfi.shift(1) <= 80), 1, 0)

    df = df.assign(**signals)
    return df

def generate_roc_signals(df):
    # 15. 变动速率 (ROC) 信号
    signals = {}
    roc_period = 12
    roc = ((df['close'] - df['close'].shift(roc_period)) / df['close'].shift(roc_period)) * 100

    # ROC 买卖信号
    signals['ROC_Buy'] = np.where((roc > 0) & (roc.shift(1) <= 0), 1, 0)
    signals['ROC_Sell'] = np.where((roc < 0) & (roc.shift(1) >= 0), 1, 0)

    df = df.assign(**signals)
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
        list of dict: 包含扁平化回测结果的字典列表。
    """

    buy_signals = [col for col in signal_data_df.columns if col.endswith('_Buy')]
    sell_signals = [col for col in signal_data_df.columns if col.endswith('_Sell')]
    target_cols = [col for col in signal_data_df.columns if 'max' in col]
    signals = buy_signals + sell_signals

    if not target_cols:
        print("警告：未找到包含 'max' 的目标列。")
        return []

    results = []
    thresholds = [0.1, 0.2, 0.3]
    total_samples = len(signal_data_df)

    # 处理 Baseline
    for target_col in target_cols:
        key_target_col = target_col.replace('close_max_', '')
        for threshold in thresholds:
            count = signal_data_df[signal_data_df[target_col] > threshold].shape[0]
            baseline_performance = round(count / total_samples, 4)
            results.append({
                "signal": "Baseline",
                "target": key_target_col,
                "threshold": threshold,
                "signal_performance": baseline_performance,
                "diff_vs_baseline": 0.0,
                "count": total_samples,
                "ratio": 1.0
            })

    # 处理买入/卖出信号
    for signal in signals:
        signal_data = signal_data_df[signal_data_df[signal] == 1]
        signal_count = len(signal_data)
        signal_ratio = round(signal_count / total_samples, 4) if total_samples > 0 else 0

        for target_col in target_cols:
            key_target_col = target_col.replace('close_max_', '')
            for threshold in thresholds:
                baseline_count = signal_data_df[signal_data_df[target_col] > threshold].shape[0]
                baseline_performance = round(baseline_count / total_samples, 4)

                if signal_count > 0:
                    above_threshold_count = signal_data[signal_data[target_col] > threshold].shape[0]
                    current_performance = round(above_threshold_count / signal_count, 4)
                    diff_vs_baseline = round(current_performance - baseline_performance, 4)
                else:
                    current_performance = 0.0
                    diff_vs_baseline = round(current_performance - baseline_performance, 4)

                results.append({
                    "signal": signal,
                    "target": key_target_col,
                    "threshold": threshold,
                    "signal_performance": current_performance,
                    "diff_vs_baseline": diff_vs_baseline,
                    "count": signal_count,
                    "ratio": signal_ratio
                })
    # 将results转化为dataframe格式
    results_df = pd.DataFrame(results)
    return results_df


def analyze_backtest_results():
    best_signal_list = []
    backtest_path = 'backtest_result'
    if not os.path.exists(backtest_path):
        os.makedirs(backtest_path)
    data_list = []
    # 加载   backtest_path下面所有的 statistic_*.csv文件
    files = os.listdir(backtest_path)
    files = [file for file in files if file.startswith('statistic_')]
    if len(files) > 0:
        print('已经存在该文件，直接读取')
    for file in files:
        if '3m' in file and '100000' in file:
            data = pd.read_csv(f'{backtest_path}/{file}')
            data[file] = file
            data_list.append(data)
            # 将data按照signal_performance降序排序
            data = data.sort_values(by='signal_performance', ascending=False)
            # # 获取前10个数据的signal
            # signal_list = data['signal'].head(32).tolist()
            # 获取signal_performance大于0.8 且 diff_vs_baseline大于0.05的数据
            signal_list = data[(data['signal_performance'] > 0.8) & (data['diff_vs_baseline'] > 0.0)]['signal'].tolist()

            best_signal_list.extend(signal_list)
    # 对best_signal_list进行去重
    best_signal_list = list(set(best_signal_list))
    # 保存best_signal_list
    with open(f'{backtest_path}/best_signal_list.txt', 'w') as f:
        f.write('\n'.join(best_signal_list))

    return best_signal_list

def run():
    """
    正式运行，得到买入或者卖出信号
    :return:
    """
    bar = '3m'
    max_candles = 1000
    backtest_path = 'backtest_result'
    best_signal_path = f'{backtest_path}/best_signal_list.txt'
    with open(best_signal_path, 'r') as f:
        best_signal_list = f.read().split('\n')
    # 获取最新数据
    data = get_train_data(max_candles=max_candles, bar=bar)
    signal_data = generate_signals(data)
    # 只保留signal_data最后的60行
    signal_data = signal_data[-60:]

    # 获取每一行中值为1的列名
    active_signals = []
    for index, row in signal_data.iterrows():
        active_signals_row = [col for col in best_signal_list if col in row and row[col] == 1]
        active_signals.append(active_signals_row)

    # 将active_signals添加到signal_data中
    signal_data['active_signals'] = active_signals

    # 统计 Buy 和 Sell 信号的数量
    signal_data['buy_count'] = signal_data['active_signals'].apply(lambda x: sum('Buy' in s for s in x))
    signal_data['sell_count'] = signal_data['active_signals'].apply(lambda x: sum('Sell' in s for s in x))
    # 计算差值
    signal_data['diff'] = signal_data['buy_count'] - signal_data['sell_count']
    # 计算和
    signal_data['sum'] = signal_data['buy_count'] + signal_data['sell_count']
    # 调整列的顺序
    cols = signal_data.columns.tolist()
    cols = ['active_signals', 'buy_count', 'sell_count', 'diff', 'sum', 'close_max_down_t30', 'close_max_up_t30'] + [col for col in cols if col not in ['active_signals', 'buy_count', 'sell_count', 'diff', 'sum', 'close_max_down_t30', 'close_max_up_t30']]
    signal_data = signal_data[cols]

    # 在这里可以根据active_signals, buy_count, sell_count进一步处理，例如生成买入卖出信号

    print(signal_data)
    return signal_data


def example():
    # run_backtest()
    analyze_backtest_results()
    run()




def run_backtest():
    backtest_path = 'backtest_result'
    if not os.path.exists(backtest_path):
        os.makedirs(backtest_path)
    base_file_path = 'origin_data.csv'
    is_reload = False
    bar_list = ['3m']
    max_candles_list = [100000]
    for max_candles in max_candles_list:
        for bar in bar_list:
            final_file_path = f'{backtest_path}/{base_file_path[:-4]}_{bar}_{max_candles}.csv'
            final_output_file_path = f'{backtest_path}/statistic_{base_file_path[:-4]}_{bar}_{max_candles}.csv'
            if os.path.exists(final_file_path) and not is_reload:
                data = pd.read_csv(final_file_path)
                # 获取最后一行的timestamp
                last_timestamp = data['timestamp'].iloc[-1]
                print(f'已经存在该文件，直接读取 {last_timestamp}')
            else:
                print('不存在该文件，开始获取')
                data = get_train_data(max_candles=max_candles, bar=bar)
                data.to_csv(final_file_path, index=False)

            # 去除后60行
            data = data[:-60]
            signal_data = generate_signals(data)
            backtest_df = backtest_strategy(signal_data)
            backtest_df.to_csv(final_output_file_path, index=False)
            print(f'已经保存至{final_output_file_path}')


if __name__ == '__main__':
    example()
    # file_name = 'BTC-USDT-SWAP_1m_20230124_20241218.csv'
    # long_df = get_dist(file_name)
    # # short_df = get_dist('BTC-USDT-SWAP_1m_20240627_20241212.csv')
    # # pass
    # # gen_feature('BTC-USDT-SWAP_1m_20241219_20241220.csv')
    # get_latest_data()