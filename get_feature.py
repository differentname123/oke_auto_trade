import pandas as pd
import numpy as np


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
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df_features = pd.DataFrame(index=df.index)  # 创建一个空的 DataFrame 用于存储特征


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
        row.append(forward)
        row.append(period)
        row.append(ratio)
        # 按照 0 和 1 顺序添加分布值
        row.append(value_counts.get(0, 0))  # 如果没有值，返回 0
        row.append(value_counts.get(1, 0))  # 如果没有值，返回 0
        distribution_data.append(row)

    # 创建 DataFrame 并添加列名
    dist_df = pd.DataFrame(distribution_data, columns=['col', 'forward', 'period', 'ratio', '0', '1'])
    return dist_df


def gen_feature(origin_name):
    data = pd.read_csv(origin_name)
    # data = data.tail(1000)
    df = feature_engineering(data)
    # 将处理后的数据保存到文件
    df.to_csv(f'{origin_name[:-4]}_features.csv', index=False)

if __name__ == '__main__':
    file_name = 'BTC-USDT-SWAP_1m_20241212_20241214.csv'
    long_df = get_dist(file_name)
    short_df = get_dist('BTC-USDT-SWAP_1m_20240627_20241212.csv')
    pass
    # gen_feature(file_name)