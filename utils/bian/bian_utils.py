import os

import requests
import pandas as pd
import datetime

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'


def get_binance_futures_info():
    """
    拉取币安所有U本位合约的当前资金费率及24小时涨跌幅，并保存为CSV
    """
    base_url = "https://fapi.binance.com"
    ticker_endpoint = "/fapi/v1/ticker/24hr"
    funding_endpoint = "/fapi/v1/premiumIndex"

    try:
        print("正在从币安拉取数据，请稍候...")

        # 1. 获取24小时行情数据
        # proxies = {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}
        # ticker_res = requests.get(base_url + ticker_endpoint, proxies=proxies, timeout=10)
        ticker_res = requests.get(base_url + ticker_endpoint, timeout=10)
        ticker_res.raise_for_status()
        ticker_data = ticker_res.json()

        # 2. 获取溢价指数数据 (资金费率)
        funding_res = requests.get(base_url + funding_endpoint, timeout=10)
        funding_res.raise_for_status()
        funding_data = funding_res.json()

        # 3. 数据处理与合并
        merged_data = {}
        for item in ticker_data:
            symbol = item['symbol']
            merged_data[symbol] = {
                '合约 (Symbol)': symbol,
                '最新价格': float(item['lastPrice']),
                '24H涨跌幅 (%)': float(item['priceChangePercent'])
            }

        for item in funding_data:
            symbol = item['symbol']
            if symbol in merged_data:
                if 'lastFundingRate' in item and item['lastFundingRate']:
                    funding_rate = float(item['lastFundingRate']) * 100
                else:
                    funding_rate = 0.0
                merged_data[symbol]['当前资金费率 (%)'] = funding_rate

        # 4. 转换为 Pandas DataFrame
        data_list = list(merged_data.values())
        df = pd.DataFrame(data_list)

        # 5. 数据清洗：只保留USDT永续合约并排序
        df = df[df['合约 (Symbol)'].str.endswith('USDT')]
        df = df.sort_values(by='当前资金费率 (%)', ascending=False)
        df = df.reset_index(drop=True)

        return df

    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
        return None
    except Exception as e:
        print(f"处理数据时发生错误: {e}")
        return None


if __name__ == "__main__":
    result_df = get_binance_futures_info()

    if result_df is not None:
        # 获取当前时间，用于生成带时间戳的文件名（可选）
        # current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # filename = f"binance_futures_data_{current_time}.csv"

        # 默认保存文件名
        filename = r"W:\project\python_project\oke_auto_trade\kline_data/binance_futures_data.csv"

        # 保存为 CSV 文件
        # 使用 utf-8-sig 编码可以防止在使用 Excel 打开 CSV 时出现中文乱码
        result_df.to_csv(filename, index=False, encoding='utf-8-sig')

        print(f"\n✅ 数据拉取成功！已自动保存为: {filename}")

        # 在终端预览前5条数据
        print("\n--- 数据预览 (前5条) ---")
        print(result_df.head(5).to_string())