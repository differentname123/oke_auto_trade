#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import datetime
import traceback
from concurrent.futures import ThreadPoolExecutor
from trade_common import LatestDataManager

INSTRUMENT_PAIR_LIST = [["BTC-USDT-SWAP", "ETH-USDT-SWAP"]]

# 获取数据的 K 线数量
MAX_DATA_PERIOD = 100


# ================= 工具函数 =================

def log_info(inst_id, msg):
    """带时间戳和币种标签的日志打印"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [{inst_id}] {msg}")


def log_error(inst_id, msg):
    """错误日志"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [{inst_id}] [ERROR] {msg}")


# ================= 核心逻辑 =================

def fetch_latest_confirmed_data(data_manager, inst_id):
    """
    尝试获取最新的、已确认的（confirm=1）K线数据。
    如果获取不到或者数据未确认，返回 None。
    """
    try:
        # 1. 调用 trade_common 获取原始数据

        if origin_df is None or origin_df.empty:
            return None

        # 2. 核心逻辑：必须筛选 confirm == '1' 的数据
        # 意味着这根 K 线已经走完，交易所归档了数据
        df_confirmed = origin_df[origin_df["confirm"] == "1"]

        if df_confirmed.empty:
            return None

        # 返回最后一行（最新的那根已完成 K 线）
        return df_confirmed.iloc[-1]

    except Exception as e:
        log_error(inst_id, f"获取数据异常: {e}")
        return None


def monitor_worker(inst_id_list):
    """
    【工作线程】
    针对【一组】币种的无限循环监控任务。
    """
    # 打印列表内容，用 join 拼接成字符串更直观
    insts_str = ", ".join(inst_id_list)
    log_info("System", f"开始监控以下币种: {insts_str}")

    # ==========================================
    # 1. 初始化阶段：为列表中的每个币种建立独立的档案
    # ==========================================
    managers = {}  # 存放每个币种的数据管理器
    last_timestamps = {}  # 存放每个币种上一次的时间戳

    for inst_id in inst_id_list:
        # 为每个 inst_id 创建专属的管理器
        managers[inst_id] = LatestDataManager(MAX_DATA_PERIOD, inst_id)
        # 初始化该币种的时间戳为 None
        last_timestamps[inst_id] = None

    inst_id_df = {}
    while True:
        try:
            # 遍历列表，轮询每一个币种
            for inst_id in inst_id_list:
                current_manager = managers[inst_id]
                origin_df = current_manager.get_newest_data()
                inst_id_df[inst_id] = origin_df
        except Exception:
            # 打印错误，这里可以用 traceback 看是哪个环节出的错
            log_error("System", "监控循环发生异常")
            traceback.print_exc()
            time.sleep(5)  # 出错后多休息一会再重试


def main():
    """
    主程序入口：使用线程池并发启动所有币种的监控
    """
    print(f"--- 启动监控程序，共 {len(INSTRUMENT_PAIR_LIST)} 个交易对 ---")

    # 使用 ThreadPoolExecutor 来替代 asyncio 和 multiprocessing
    # max_workers 设置为交易对的数量，保证每个币种有一个独立的线程
    with ThreadPoolExecutor(max_workers=len(INSTRUMENT_PAIR_LIST)) as executor:
        # 将 monitor_worker 函数映射到每一个 instrument 上
        executor.map(monitor_worker, INSTRUMENT_PAIR_LIST)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序已手动停止")