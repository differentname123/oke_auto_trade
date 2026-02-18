#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import datetime
import traceback
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time

import okx.Trade as Trade
import okx.MarketData as Market
import okx.Account as Account
import pandas as pd
from common_utils import get_config
import pandas as pd

from run_pair_backest import parse_backtest_filename, generate_pair_trading_signals
from trade_common import LatestDataManager

INSTRUMENT_PAIR_LIST = [["BTC-USDT-SWAP", "ETH-USDT-SWAP"]]

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


# 获取数据的 K 线数量
MAX_DATA_PERIOD = 10000


# ================= 工具函数 =================

def log_info(inst_id, msg):
    """带时间戳和币种标签的日志打印"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [{inst_id}] {msg}")


def log_error(inst_id, msg):
    """错误日志"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [{inst_id}] [ERROR] {msg}")



def get_good_param(final_df_path='kline_data/result_df.csv'):
    df = pd.read_csv(final_df_path)
    df = df[df['avg_profit_per_trade'] > 0.1]
    df = df[df['total_trades'] > 100]
    df['profit'] = (df['avg_profit_per_trade'] - 0.1) * df['total_trades']
    df['score'] = df['profit'] / -(df['Max Drawdown'] - 0.5) / -(df['Max Drawdown'] - 0.5)
    df_filtered = df.sort_values(by='score', ascending=False).head(150)
    params_list = []
    for index, row in df_filtered.iterrows():
        # 从该行的 'file_name' 列解析参数
        params = parse_backtest_filename(row['file_name'])
        params_list.append(params)
    return params_list


def monitor_worker(inst_id_list):
    """
    【工作线程】
    针对【一组】币种的无限循环监控任务。
    """
    try:
        # 打印列表内容，用 join 拼接成字符串更直观
        insts_str = ", ".join(inst_id_list)
        log_info("System", f"开始监控以下币种: {insts_str}")

        params_list = get_good_param()
        # ==========================================
        # 1. 初始化阶段：为列表中的每个币种建立独立的档案
        # ==========================================
        managers = {}  # 存放每个币种的数据管理器

        for inst_id in inst_id_list:
            # 为每个 inst_id 创建专属的管理器
            managers[inst_id] = LatestDataManager(MAX_DATA_PERIOD, inst_id)
    except Exception as e:
        traceback.print_exc()

    while True:
        try:
            origin_df_list = []
            ok_signal_df_list = []
            all_signal_df_list = []
            has_signal_df_list = []
            for inst_id in inst_id_list:
                current_manager = managers[inst_id]
                origin_df = current_manager.get_newnewest_data()
                if origin_df is not None:
                    origin_df['open_time'] = pd.to_datetime(origin_df['timestamp'])
                    origin_df_list.append(origin_df)
            if len(origin_df_list) == 2:
                main_df = origin_df_list[0]
                sub_df = origin_df_list[1]
                main_df = main_df.sort_values('open_time').set_index('open_time')
                sub_df = sub_df.sort_values('open_time').set_index('open_time')

                # 合并数据，只保留两者都有的时间点 (Inner Join)
                merged_df = pd.merge(main_df[['close']], sub_df[['close']], left_index=True, right_index=True, suffixes=('_main', '_sub'))
                for params in params_list:
                    full_df = generate_pair_trading_signals(merged_df=merged_df, main_col='close_main', sub_col='close_sub', window=60, z_entry=params['z_entry'], z_exit=params['z_exit'], delta=params['delta'], ve=params['ve'])
                    full_df['params'] = [copy.deepcopy(params) for _ in range(len(full_df))]
                    all_signal_df_list.append(full_df)
                    if len(full_df) > 0 and (full_df['signal'] != 0).any():
                        has_signal_df_list.append(full_df)
                    # 搜集最后一行signal字段和倒数第二行signal字段不一样的
                    if len(full_df) >= 1:
                        last_signal = full_df.iloc[-1]['signal']

                        signal_changed = False
                        if len(full_df) >= 2:
                            prev_signal = full_df.iloc[-2]['signal']
                            signal_changed = last_signal != prev_signal

                        signal_active = last_signal != 0

                        if signal_changed or signal_active:
                            ok_signal_df_list.append(full_df)
                print()

        except Exception:
            # 打印错误，这里可以用 traceback 看是哪个环节出的错
            log_error("System", "监控循环发生异常")
            traceback.print_exc()
            time.sleep(5)  # 出错后多休息一会再重试




def main():
    """
    主程序入口：使用线程池并发启动所有币种的监控
    """
    # balance_res = accountAPI.get_account_balance()

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