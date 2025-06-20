import json
import os
import time
from datetime import datetime

import okx.Trade as Trade
import okx.MarketData as Market
import okx.Account as Account
import pandas as pd
from common_utils import get_config

# 设置代理（如果需要）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

flag = "0"  # 实盘: 0, 模拟盘: 1

# API 初始化
if flag == "1":
    apikey = get_config('api_key')
    secretkey = get_config('secret_key')
    passphrase = get_config('passphrase')
else:
    apikey = get_config('true_api_key')
    secretkey = get_config('true_secret_key')
    passphrase = get_config('true_passphrase')

tradeAPI = Trade.TradeAPI(apikey, secretkey, passphrase, False, flag)
marketAPI = Market.MarketAPI(apikey, secretkey, passphrase, False, flag)
accountAPI = Account.AccountAPI(apikey, secretkey, passphrase, False, flag)

class LatestDataManager:
    def __init__(self, capacity, INST_ID):
        self.capacity = capacity
        self.inst_id = INST_ID
        self.max_single_size = 100 # 底层数据最大单次获取数量
        self.data_df = get_train_data(max_candles=self.capacity, inst_id=self.inst_id)

    def get_newest_data(self):
        recent_data_df = get_train_data(max_candles=self.max_single_size, is_newest=False, inst_id=self.inst_id)
        if recent_data_df.empty:
            print("获取最新数据失败，返回空数据")
            return self.data_df
        # 判断recent_data_df第一个时间戳（timestamp）是否在data_df中
        if recent_data_df['timestamp'].iloc[0] in self.data_df['timestamp'].values:
            # print('历史数据在最新数据中，更新数据')
            # 合并两个df
            self.data_df = pd.concat([recent_data_df, self.data_df], ignore_index=True)
            # 去重
            self.data_df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
            # 排序
            self.data_df.sort_values(by='timestamp', ascending=True, inplace=True)
            self.data_df = self.data_df[self.data_df['confirm'] == '1']  # 只保留confirm为1的数据

            # 保留最新的capacity条数据
            self.data_df = self.data_df.iloc[-self.capacity:]
            return self.data_df

        else:
            print('历史数据不在最新数据中，重新获取数据')
            self.data_df = get_train_data(max_candles=self.capacity)
            self.data_df = self.data_df[self.data_df['confirm'] == '1']  # 只保留confirm为1的数据
            return self.data_df

    def get_newnewest_data(self):
        recent_data_df = get_train_data(max_candles=self.max_single_size, is_newest=True, inst_id=self.inst_id)
        if recent_data_df.empty:
            print("获取最新数据失败，返回空数据")
            return self.data_df
        # 判断recent_data_df第一个时间戳（timestamp）是否在data_df中
        if recent_data_df['timestamp'].iloc[0] in self.data_df['timestamp'].values:
            # print('历史数据在最新数据中，更新数据')
            # 合并两个df
            self.data_df = pd.concat([recent_data_df, self.data_df], ignore_index=True)
            # 去重
            self.data_df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
            # 排序
            self.data_df.sort_values(by='timestamp', ascending=True, inplace=True)
            # 保留最新的capacity条数据
            self.data_df = self.data_df.iloc[-self.capacity:]
            return self.data_df

        else:
            print('历史数据不在最新数据中，重新获取数据')
            self.data_df = get_train_data(max_candles=self.capacity)
            return self.data_df

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
                print(f"获取K线数据失败，错误代码：{response.get('code')}，错误消息：{response.get('msg')} {inst_id}")
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
    max_retries = 3  # 最大重试次数

    while fetched_candles < max_candles:
        try:
            # 调用OKX API获取历史K线数据
            response = marketAPI.get_history_candlesticks(instId=inst_id, bar=bar, after=after, limit=limit)

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


def get_train_data(inst_id="BTC-USDT-SWAP", bar="1m", limit=100, max_candles=1000, is_newest=False):
    # 获取数据
    if is_newest:
        kline_data = get_kline_data_newest(inst_id=inst_id, bar=bar, limit=limit, max_candles=max_candles)
    else:
        kline_data = get_kline_data(inst_id=inst_id, bar=bar, limit=limit, max_candles=max_candles)

    if not kline_data.empty:

        # 重置索引
        kline_data.reset_index(drop=True, inplace=True)
        return kline_data
    else:
        print("未能获取到任何K线数据。")
        return pd.DataFrame()


def log_order(log_item):
    """
    将订单日志追加到temp/all_order.json中，日志以JSON数组的形式保存。
    """
    file_path = "temp/all_order.json"
    # 确保temp目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 尝试读取已有的日志内容
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
    except Exception:
        data = []

    # 追加日志
    data.append(log_item)

    # 写入文件（带缩进，保证JSON格式的可读性）
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def place_order(inst_id, side, size, trade_action="open"):
    """
    以最优价格下单（市价单），支持开仓或平仓（双向持仓模式）。
    每次调用都会将详细的订单信息保存到 temp/all_order.json 中。

    :param inst_id: 交易对 (如 "BTC-USDT-SWAP")
    :param side: 交易方向 ("buy" = 买入 / "sell" = 卖出)
    :param size: 下单数量
    :param trade_action: "open" (开仓) 或 "close" (平仓)
    :return: 下单成功返回 True，否则返回 False
    """
    try:
        if trade_action not in ["open", "close"]:
            raise ValueError("❌ trade_action 必须是 'open' 或 'close'")

        # 确定持仓方向（posSide）
        if trade_action == "open":
            pos_side = "long" if side == "buy" else "short"
            reduce_only = "false"  # 开仓不需要 reduceOnly
        else:  # 平仓
            pos_side = "long" if side == "sell" else "short"
            reduce_only = "true"  # 平仓需要 reduceOnly

        # 构建市价单下单参数
        order_params = {
            "instId": inst_id,  # 交易对
            "tdMode": "cross",  # 全仓模式
            "side": side,  # 交易方向
            "ordType": "market",  # 市价单（最优价格）
            "sz": str(size),  # 下单量
            "posSide": pos_side,  # 持仓方向（long = 多 / short = 空）
            "reduceOnly": reduce_only  # 是否为平仓单
        }

        # 调用 OKX 下单 API
        order = tradeAPI.place_order(**order_params)

        # 构造日志记录
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "inst_id": inst_id,
            "side": side,
            "size": size,
            "trade_action": trade_action,
            "order_params": order_params,
            "order_response": order,
        }

        # 判断下单结果
        if order.get("code") != "0":
            print(f"❌ {trade_action.upper()} {side.upper()} 市价单下单失败，错误信息：", order)
            log_entry["result"] = False
            log_order(log_entry)
            return False

        log_entry["result"] = True
        log_order(log_entry)
        return True

    except Exception as e:
        # 在异常时记录详细的错误日志
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "inst_id": inst_id,
            "side": side,
            "size": size,
            "trade_action": trade_action,
            "order_params": order_params if 'order_params' in locals() else None,
            "error": str(e),
            "result": False
        }
        print(f"❌ {trade_action.upper()} {side.upper()} 市价单下单失败，错误信息：", e)
        log_order(log_entry)
        return False
def get_total_usdt_equity():
    """
    获取账户总权益的USDT等值。
    该函数会抓取账户所有币种的余额，并根据当前市价将其换算成USDT价值，最终汇总。

    :param accountAPI: OKX账户API实例
    :param marketAPI: OKX行情API实例
    :return: 浮点数，表示账户总权益的USDT价值；如果出错则返回None。
    """
    total_usdt_equity = 0.0
    try:
        # 1. 获取账户余额信息
        balance_res = accountAPI.get_account_balance()
        if balance_res.get("code") != "0":
            print(f"❌ 获取账户余额失败: {balance_res.get('msg')}")
            return None

        # 提取账户资产详情
        details = balance_res.get("data", [{}])[0].get("details", [])
        if not details:
            print("ℹ️ 账户中没有资产。")
            return 0.0

        print("正在计算账户总USDT价值...")
        # 2. 遍历所有资产并计算其USDT价值
        for asset in details:
            currency = asset.get("ccy")
            equity = float(asset.get("eq", 0))

            # 如果该币种权益为0，则跳过
            if equity == 0:
                continue

            # 3. 如果是USDT，直接累加
            if currency.upper() == "USDT":
                total_usdt_equity += equity
                print(f"  - {currency}: {equity:.2f} USDT")
            else:
                # 4. 如果是其他币种，获取其对USDT的最新价格并换算
                inst_id_spot = f"{currency}-USDT"
                inst_id_swap = f"{currency}-USDT-SWAP"
                last_price = 0

                try:
                    # 优先查询现货价格
                    ticker_res = marketAPI.get_ticker(instId=inst_id_spot)
                    if ticker_res.get("code") == "0" and ticker_res.get("data"):
                        last_price = float(ticker_res.get("data")[0].get("last", 0))
                    else:
                        # 如果现货价格获取失败，尝试查询永续合约价格
                        ticker_res_swap = marketAPI.get_ticker(instId=inst_id_swap)
                        if ticker_res_swap.get("code") == "0" and ticker_res_swap.get("data"):
                            last_price = float(ticker_res_swap.get("data")[0].get("last", 0))
                        else:
                            print(f"⚠️ 无法获取 {inst_id_spot} 或 {inst_id_swap} 的价格，跳过 {currency}。")
                            continue

                    if last_price > 0:
                        usdt_value = equity * last_price
                        total_usdt_equity += usdt_value
                        print(f"  - {currency}: {equity} * ${last_price:.4f} ≈ {usdt_value:.2f} USDT")

                except Exception as e:
                    print(f"获取 {currency} 价格时发生错误: {e}")

        return total_usdt_equity

    except Exception as e:
        print(f"❌ 计算总权益时发生未知错误: {e}")
        return None


if __name__ == '__main__':
    # 获取最新数据
    # place_order("BTC-USDT-SWAP", "sell", 1, trade_action="close")  # 以最优价格开多 0.01 BTC

    total_equity = get_total_usdt_equity()
    if total_equity is not None:
        print(f"\n✅ 账户总权益 ≈ {total_equity:.2f} USDT")