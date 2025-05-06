import numpy as np
import pandas as pd

# 假设你的原始数据存储在一个 CSV 文件中
# 并且文件中只有 ["timestamp", "open", "high", "low", "close", "volume"] 这几列
df = pd.read_csv("kline_data/origin_data_1m_5000000_BTC-USDT-SWAP_2025-05-06.csv")
output_path = "kline_data/origin_data_1m_5000000_BTC-USDT-SWAP_2025-05-06_with_feature.parquet"
# 只保留["timestamp", "open", "high", "low", "close", "volume"]
df = df[["timestamp", "open", "high", "low", "close", "volume"]]

# ---------------------------
# 1. 计算对数收益率和对数成交量
# ---------------------------
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
df['log_vol'] = np.log(df['volume'].replace(0, np.nan))  # 避免成交量为0时报错

# ---------------------------
# 2. 计算简单移动平均线 (SMA)
# ---------------------------
df['sma20']  = df['close'].rolling(window=20, min_periods=1).mean()
df['sma60']  = df['close'].rolling(window=60, min_periods=1).mean()
df['sma120'] = df['close'].rolling(window=120, min_periods=1).mean()

# ---------------------------
# 3. 计算 RSI（14周期）
# ---------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # 简单移动平均计算
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['rsi14'] = compute_rsi(df['close'], period=14)

# ---------------------------
# 4. 计算 ATR（14周期）
# ---------------------------
def compute_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

df['atr14'] = compute_atr(df, period=14)

# ---------------------------
# 5. 计算布林带宽度 (bollw)
# ---------------------------
rolling_std = df['close'].rolling(window=20, min_periods=1).std()
upper_band = df['sma20'] + 2 * rolling_std
lower_band = df['sma20'] - 2 * rolling_std
# 布林带宽度取上下轨之差
df['bollw'] = upper_band - lower_band
# 若需要归一化，也可以计算：df['bollw_norm'] = (upper_band - lower_band) / df['sma20']

# ---------------------------
# 6. 计算60周期波动率、偏度和峰度
# ---------------------------
df['vol60'] = df['log_ret'].rolling(window=60, min_periods=1).std()
df['skew60'] = df['log_ret'].rolling(window=60, min_periods=1).skew()
df['kurt60'] = df['log_ret'].rolling(window=60, min_periods=1).kurt()

# ---------------------------
# 7. 市场阶段指标：bull, bear, side
# ---------------------------
# 设置一个阈值，用于判断是否“显著”偏离均线，例如 0.1%
threshold = 0.001

# 如果收盘价高于 sma20*(1+阈值) 认为是多头市场
df['bull'] = (df['close'] > df['sma20'] * (1 + threshold)).astype(int)
# 如果收盘价低于 sma20*(1-阈值) 认为是空头市场
df['bear'] = (df['close'] < df['sma20'] * (1 - threshold)).astype(int)
# 剩余情况定义为横盘阶段
df['side'] = 1 - df['bull'] - df['bear']

# ---------------------------
# 8. 添加占位列：dummy_pos, dummy_pnl, dummy_margin
# ---------------------------
df['dummy_pos'] = 0        # 用于记录持仓情况
df['dummy_pnl'] = 0.0      # 用于记录累计收益
df['dummy_margin'] = 0.0   # 用于记录保证金或风险度量

# ---------------------------
# 整理数据
# ---------------------------
# 许多技术指标在初始若干行由于采样窗口不足会出现 NaN，可根据需要进行填充或删除
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
# 保存处理后的数据
df.to_parquet(output_path, index=False)
# 查看生成的特征列
print(df.columns.tolist())