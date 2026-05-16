import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile  # <--- 新增这行导入，用于校验压缩包完整性
# ================= 配置区域 =================
TARGET_FILES = ["kline_data/BTC_ETH_1m.csv", "kline_data/DOGE_SOL_1m.csv", "kline_data/TON_XRP_1m.csv"]
# 自动解析列表涉及到的币，并拼接成 USDT 交易对 (例如: BTCUSDT, ETHUSDT)
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"
]

# for f in TARGET_FILES:
#     # 提取文件名如 'BTC_ETH_1m.csv' -> 'BTC_ETH' -> ['BTC', 'ETH']
#     base_name = f.split('/')[-1].replace('_1m.csv', '')
#     for coin in base_name.split('_'):
#         sym = f"{coin}USDT"
#         if sym not in SYMBOLS:
#             SYMBOLS.append(sym)

INTERVAL = "1m"  # K线级别
MARKET = "futures/um"  # 市场类型：现货填"spot"，U本位填"futures/um"
START_DATE = "2025-01-01"  # 开始日期 (YYYY-MM-DD)
END_DATE = "2026-05-26"  # 结束日期 (YYYY-MM-DD)
SAVE_DIR = "./binance_data"  # ZIP文件临时保存路径
MAX_WORKERS = 20  # 并发下载线程数（建议 5-10）
# ==========================================

# 币安官方 K线 CSV 的标准列名
COLUMNS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'count',
    'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
]

def download_single_day(symbol, date_str):
    """下载单日数据的核心函数"""
    base_url = f"https://data.binance.vision/data/{MARKET}/daily/klines/{symbol}/{INTERVAL}/"
    file_name = f"{symbol}-{INTERVAL}-{date_str}.zip"
    download_url = base_url + file_name
    save_path = os.path.join(SAVE_DIR, file_name)

    # 增强版检测：如果本地已经有这个zip了，检查它是否完整
    if os.path.exists(save_path):
        # 既要有大小，且能通过 zip 格式校验
        if os.path.getsize(save_path) > 0 and zipfile.is_zipfile(save_path):
            return save_path, True, "已存在且有效"
        else:
            # 如果是损坏的残骸，将其删除，让下方的 try 继续执行重新下载逻辑
            try:
                os.remove(save_path)
            except Exception as e:
                pass # 忽略删除报错（例如文件正被占用）

    try:
        response = requests.get(download_url, stream=True, timeout=15)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return save_path, True, f"下载成功{symbol}"
        elif response.status_code == 404:
            return save_path, False, "404 文件不存在(可能官网还未生成)"
        else:
            return save_path, False, f"状态码: {response.status_code}"
    except Exception as e:
        return save_path, False, f"异常: {e}"


def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")

    # 生成需要下载的日期列表
    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    # 遍历解析出的所有交易对
    for symbol in SYMBOLS:
        merged_file = f"kline_data/{symbol}_{INTERVAL}_{START_DATE}_merged.csv"
        # 确保最终输出目录存在
        os.makedirs(os.path.dirname(merged_file), exist_ok=True)

        print(f"\n=============================================")
        print(f"=== 开始并行下载 {symbol} {INTERVAL} 数据 ===")
        print(f"=============================================")
        print(f"时间范围: {START_DATE} 到 {END_DATE} (共 {len(date_list)} 天)")

        downloaded_zips = []

        # 1. 使用多线程并行下载
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务 (传入当前的 symbol)
            future_to_date = {executor.submit(download_single_day, symbol, date): date for date in date_list}

            for future in as_completed(future_to_date):
                date = future_to_date[future]
                save_path, success, msg = future.result()
                if success:
                    downloaded_zips.append(save_path)
                    print(f"[OK] {date} : {msg}")
                else:
                    print(f"[FAIL] {date} : {msg}")

        if not downloaded_zips:
            print(f"没有下载到 {symbol} 的任何数据，跳过。\n")
            continue

        # 2. 读取、解压与合并 (Pandas 可以直接读取 ZIP 里的 CSV)
        print("\n=== 开始读取并合并数据 ===")
        dfs = []
        for zip_file in downloaded_zips:
            try:
                # header=None 并强制使用我们定义的列名，避免官方个别天数混入表头
                df = pd.read_csv(zip_file, compression='zip', names=COLUMNS, header=None, low_memory=False)

                # 清洗：如果第一行是表头(包含英文字母)，转成数字时会变成 NaN，将其剔除
                df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
                df.dropna(subset=['open_time'], inplace=True)
                df['open_time'] = df['open_time'].astype('int64')

                dfs.append(df)
            except Exception as e:
                print(f"读取 {zip_file} 失败: {e}")

        # 合并所有 DataFrame
        final_df = pd.concat(dfs, ignore_index=True)

        # 3. 数据校验：排序与去重
        print("\n=== 开始数据清洗与校验 ===")
        # 强制按时间戳升序排列
        final_df.sort_values('open_time', ascending=True, inplace=True)

        # 去除完全重复的时间戳，保留第一条
        initial_len = len(final_df)
        final_df.drop_duplicates(subset=['open_time'], keep='first', inplace=True)
        drop_count = initial_len - len(final_df)
        print(f"去重完成: 删除了 {drop_count} 条时间戳重复的数据。")

        # 4. 检测时间断层 (1分钟 = 60,000 毫秒)
        INTERVAL_MS = 60000

        # 计算相邻两行的 open_time 差值
        diffs = final_df['open_time'].diff()

        # 差值大于 60000 毫秒的地方就是断层
        gaps = final_df[diffs > INTERVAL_MS]

        if not gaps.empty:
            print(f"\n⚠️ 警告：检测到 {len(gaps)} 处时间断层！")
            for idx, row in gaps.iterrows():
                # 找到断层开始和结束的具体人类可读时间
                gap_end_time = pd.to_datetime(row['open_time'], unit='ms')
                gap_start_time = pd.to_datetime(final_df.loc[idx - 1, 'open_time'], unit='ms')
                missing_minutes = (row['open_time'] - final_df.loc[idx - 1, 'open_time']) / 60000 - 1
                print(f" -> 缺失时间段: {gap_start_time} 至 {gap_end_time}，约缺失 {int(missing_minutes)} 分钟数据")
        else:
            print("\n✅ 数据完整度检查通过：完美连续，没有发现任何时间断层。")

        # 5. 保存最终 CSV
        final_df.to_csv(merged_file, index=False)
        print(f"\n=== {symbol} 全部完成！合并后的文件已保存为: {merged_file} ===")
        print(f"总数据量: {len(final_df)} 行\n")

if __name__ == "__main__":
    main()