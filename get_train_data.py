import os
import pandas as pd
import time
import okx.MarketData as Market
from common_utils import get_config

# 设置代理（如果需要）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# OKX API初始化
flag = "0"  # 模拟盘: 1, 实盘: 0
apikey = get_config('api_key')
secretkey = get_config('secret_key')
passphrase = get_config('passphrase')
marketAPI = Market.MarketAPI(apikey, secretkey, passphrase, False, flag)


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

    while fetched_candles < max_candles:
        try:
            # 调用OKX API获取历史K线数据
            response = marketAPI.get_history_candlesticks(instId=inst_id, bar=bar, after=after, limit=limit)

            if response["code"] != "0":
                print(f"获取K线数据失败，错误代码：{response['code']}，错误消息：{response['msg']}")
                break

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
            print(f"已获取 {fetched_candles} 条K线数据，最新时间：{pd.to_datetime(after, unit='ms')}")

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
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float) / 1000, unit="s")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

    # 按时间排序
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def add_time_features(df):
    """
    添加时间特征，包括小时、分钟、星期几、是否周末等。
    """
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 星期几（0=周一, 6=周日）
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)  # 是否为周末
    df["day_of_month"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    return df


def add_target_variables(df, thresholds=None, max_decoder_length=5):
    """
    添加目标变量，包括未来多个时间步的涨幅和跌幅，针对close、high、low的多阈值计算。

    :param df: 数据集
    :param thresholds: 阈值列表，默认[0.1, 0.2, ..., 1.0]
    :param max_decoder_length: 预测未来的时间步长，默认5
    :return: 添加了目标变量的DataFrame
    """
    if thresholds is None:
        thresholds = [round(x * 0.1, 2) for x in range(1, 11)]  # [0.1, 0.2, ..., 1.0]

    # 创建一个新的字典，用于存储所有新增列
    new_columns = {}

    for col in ["close", "high", "low"]:
        for step in range(1, max_decoder_length + 1):  # 未来 1 到 max_decoder_length 分钟
            for threshold in thresholds:
                # 计算涨幅超过阈值的列
                new_columns[f"{col}_up_{threshold}_t{step}"] = (
                    (df[col].shift(-step) - df[col]) / df[col] > threshold / 100
                ).astype(int)

                # 计算跌幅超过阈值的列
                new_columns[f"{col}_down_{threshold}_t{step}"] = (
                    (df[col].shift(-step) - df[col]) / df[col] < -threshold / 100
                ).astype(int)

    # 使用 pd.concat 一次性将所有新列添加到原数据框
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    return df


if __name__ == "__main__":
    # 示例：获取BTC-USDT的1分钟K线数据，最大获取1000条
    inst_id = "BTC-USDT-SWAP"
    bar = "1m"
    limit = 100
    max_candles = 50000

    # 获取数据
    kline_data = get_kline_data(inst_id=inst_id, bar=bar, limit=limit, max_candles=max_candles)

    if not kline_data.empty:
        print("成功获取K线数据，开始处理...")

        # 添加时间特征
        kline_data = add_time_features(kline_data)

        # 添加目标变量
        kline_data = add_target_variables(kline_data)

        # 重置索引
        kline_data.reset_index(drop=True, inplace=True)

        # 保存文件
        start_date = kline_data["timestamp"].iloc[0].strftime("%Y%m%d")
        end_date = kline_data["timestamp"].iloc[-1].strftime("%Y%m%d")
        filename = f"{inst_id}_{bar}_{start_date}_{end_date}.csv"

        kline_data.to_csv(filename, index=False)
        print(f"数据已保存至文件：{filename}")
    else:
        print("未能获取到任何K线数据。")