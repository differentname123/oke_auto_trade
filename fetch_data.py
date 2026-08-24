import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile  # <--- 用于校验压缩包完整性

# ================= 配置区域 =================
TARGET_FILES = ["kline_data/BTC_ETH_1m.csv", "kline_data/DOGE_SOL_1m.csv", "kline_data/TON_XRP_1m.csv"]
# 自动解析列表涉及到的币，并拼接成 USDT 交易对 (例如: BTCUSDT, ETHUSDT)
SYMBOLS = [
    "AAVEUSDT",
    "AVAXUSDT",
    "BNBUSDT",
    "BTCUSDT",
    "DOGEUSDT",
    "ETHUSDT",
    "GMXUSDT",
    "JUPUSDT",
    "KASUSDT",
    "LDOUSDT",
    "LINKUSDT",
    "MKRUSDT",
    "NEARUSDT",
    "ONDOUSDT",
    "PENDLEUSDT",
    "PYTHUSDT",
    "RAYUSDT",
    "RENDERUSDT",
    "RNDRUSDT",
    "RUNEUSDT",
    "SKYUSDT",
    "SOLUSDT",
    "STXUSDT",
    "TAOUSDT",
    "TONUSDT",
    "TRXUSDT",
    "UNIUSDT"
]
SYMBOLS = []
# 新增：单独拉取 1s 级别数据的币种列表
SYMBOLS_1s = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT"
]

crypto_scores = {
    "BTCUSDT": 10.0,
    "ETHUSDT": 9.0,
    "SOLUSDT": 8.5,
    "AAVEUSDT": 7.8,
    "LINKUSDT": 7.5,
    "BNBUSDT": 7.2,
    "UNIUSDT": 7.0,
    "SKYUSDT": 7.0,
    "STXUSDT": 6.8,
    "KASUSDT": 6.5,
    "DOGEUSDT": 6.5,
    "RENDERUSDT": 6.2,
    "RUNEUSDT": 6.0,
    "TONUSDT": 5.5,
    "TRXUSDT": 5.5,
    "PENDLEUSDT": 5.0
}

INTERVAL = "1m"  # 原有 K线级别
MARKET = "futures/um"  # 原有 市场类型：现货填"spot"，U本位填"futures/um"

# 新增：1s 级别对应的配置
INTERVAL_1s = "1s"
MARKET_1s = "spot"  # 1s 数据通常存在于现货市场

START_DATE = "2021-01-01"  # 开始日期 (YYYY-MM-DD)
END_DATE = "2026-08-30"  # 结束日期 (YYYY-MM-DD)
SAVE_DIR = "./binance_data"  # ZIP文件临时保存路径
MAX_WORKERS = 20  # 并发下载线程数（建议 5-10）
# ==========================================

# 币安官方 K线 CSV 的标准列名
COLUMNS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'count',
    'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
]


def download_single_day(symbol, date_str, market, interval):
    """下载单日数据的核心函数 (已适配动态 market 和 interval)"""
    base_url = f"https://data.binance.vision/data/{market}/daily/klines/{symbol}/{interval}/"
    file_name = f"{symbol}-{interval}-{date_str}.zip"
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
                pass  # 忽略删除报错（例如文件正被占用）

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


def process_symbols(symbol_list, interval, market, date_list):
    """提取出的核心处理逻辑，支持不同列表、级别和市场"""
    for symbol in symbol_list:
        merged_file = f"kline_data/{symbol}_{interval}_{START_DATE}_merged.csv"
        # 确保最终输出目录存在
        os.makedirs(os.path.dirname(merged_file), exist_ok=True)

        print(f"\n=============================================")
        print(f"=== 开始并行下载 {symbol} {interval} 数据 ===")
        print(f"=============================================")
        print(f"时间范围: {START_DATE} 到 {END_DATE} (共 {len(date_list)} 天)")

        downloaded_zips = []

        # 1. 使用多线程并行下载
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务 (传入当前的 symbol, market, interval)
            future_to_date = {executor.submit(download_single_day, symbol, date, market, interval): date for date in
                              date_list}

            for future in as_completed(future_to_date):
                date = future_to_date[future]
                save_path, success, msg = future.result()
                if success:
                    downloaded_zips.append(save_path)
                else:
                    pass

        if not downloaded_zips:
            print(f"没有下载到 {symbol} 的任何数据，跳过。\n")
            continue

        # 2. 读取、解压与合并 (Pandas 可以直接读取 ZIP 里的 CSV)
        print("\n=== 开始读取并合并数据 ===")
        dfs = []
        for zip_file in downloaded_zips:
            try:
                # header=None 并强制使用我们定义的列名，完美适配 1s 的无表头数据
                df = pd.read_csv(zip_file, compression='zip', names=COLUMNS, header=None, low_memory=False)

                # 清洗：如果第一行是表头(包含英文字母)，转成数字时会变成 NaN，将其剔除
                df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
                df.dropna(subset=['open_time'], inplace=True)
                df['open_time'] = df['open_time'].astype('int64')

                dfs.append(df)
            except Exception as e:
                print(f"读取 {zip_file} 失败: {e}")

        if not dfs:
            print(f"{symbol} 没有有效数据可供合并。\n")
            continue

        # 合并所有 DataFrame
        final_df = pd.concat(dfs, ignore_index=True)

        # 3. 数据校验：排序与去重
        print("\n=== 开始数据清洗与校验 ===")
        # 强制按时间戳升序排列
        final_df.sort_values('open_time', ascending=True, inplace=True)

        # 去除完全重复的时间戳，保留第一条
        initial_len = len(final_df)
        final_df.drop_duplicates(subset=['open_time'], keep='first', inplace=True)

        # ！！！关键修复：去重和排序后，必须重置索引，否则会导致下方的 .loc[idx-1] 找不到旧索引报错
        final_df.reset_index(drop=True, inplace=True)

        drop_count = initial_len - len(final_df)
        print(f"去重完成: 删除了 {drop_count} 条时间戳重复的数据。")

        # 4. 检测时间断层 (动态适配 1s 和 1m)
        if interval == "1s":
            INTERVAL_MS = 1000
            time_unit_str = "秒"
        else:
            INTERVAL_MS = 60000
            time_unit_str = "分钟"

        # 计算相邻两行的 open_time 差值
        diffs = final_df['open_time'].diff()

        # 差值大于 INTERVAL_MS 毫秒的地方就是断层
        gaps = final_df[diffs > INTERVAL_MS]

        if not gaps.empty:
            print(f"\n⚠️ 警告：检测到 {len(gaps)} 处时间断层！")

            # 增加安全限制：如果断层太多，只打印前 30 条，防止控制台卡死
            max_print = 30
            for i, (idx, row) in enumerate(gaps.iterrows()):
                if i >= max_print:
                    print(f" ... 剩余 {len(gaps) - max_print} 处断层已省略打印，防止日志刷屏卡死 ...")
                    break

                # 找到断层开始和结束的具体人类可读时间
                gap_end_time = pd.to_datetime(row['open_time'], unit='ms')
                gap_start_time = pd.to_datetime(final_df.loc[idx - 1, 'open_time'], unit='ms')
                missing_count = (row['open_time'] - final_df.loc[idx - 1, 'open_time']) / INTERVAL_MS - 1
                print(
                    f" -> 缺失时间段: {gap_start_time} 至 {gap_end_time}，约缺失 {int(missing_count)} {time_unit_str}数据")
        else:
            print("\n✅ 数据完整度检查通过：完美连续，没有发现任何时间断层。")

        # 5. 保存最终 CSV
        final_df.to_csv(merged_file, index=False)
        print(f"\n=== {symbol} 全部完成！合并后的文件已保存为: {merged_file} ===")
        print(f"总数据量: {len(final_df)} 行\n")


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

    # === 执行任务 1：拉取原来的 1m U本位合约数据 ===
    if SYMBOLS:
        print("\n>>> 开始执行 1m 级别数据拉取任务 <<<")
        process_symbols(SYMBOLS, INTERVAL, MARKET, date_list)

    # === 执行任务 2：拉取新增的 1s 现货数据 ===
    if SYMBOLS_1s:
        print("\n>>> 开始执行 1s 级别现货数据拉取任务 <<<")
        process_symbols(SYMBOLS_1s, INTERVAL_1s, MARKET_1s, date_list)


if __name__ == "__main__":
    main()