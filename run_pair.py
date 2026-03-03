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

import numpy as np
import okx.Trade as Trade
import okx.MarketData as Market
import okx.Account as Account
import pandas as pd
from common_utils import get_config, read_json, save_json
import pandas as pd

from run_pair_backest import parse_backtest_filename, generate_pair_trading_signals
from trade_common import LatestDataManager

INSTRUMENT_PAIR_LIST = [["BTC-USDT-SWAP", "ETH-USDT-SWAP"]]
base_order_file = "history_order/order_history.json"

# 设置代理（如果需要）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

flag = "1"  # 实盘: 0, 模拟盘: 1

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


def calculate_score_and_target_str(df):
    """
    从 file_name 计算 target_str（按 'w60' 分割取第二个值），
    并计算 score（要求前面的 filter、avg_profit_per_trade 调整、profit 计算已完成）
    """
    # 计算 target_str
    df['target_str'] = df['file_name'].str.split('w60').str[1]
    df = df[df['Max Drawdown'] > -20]
    df = df[df['最长持仓时间'] < 10000]

    # 计算 score（保持原公式不变）
    df['score'] = (
            np.log(df['avg_profit_per_trade'] + 1)
            * df['profit']
            * df['avg_profit_per_trade']
            * df['total_trades']
            / -(df['Max Drawdown'] - 0.5)
            / np.log(df['最长持仓时间'] + 1)
    )
    return df


def get_good_param():
    key_word = 'BTC_ETH_1m'
    final_df_path = f'kline_data/result_df_{key_word}.csv'
    df = pd.read_csv(final_df_path)
    df = df[df['avg_profit_per_trade'] > 0.2]
    df = df[df['Max Drawdown'] > -20]

    df['avg_profit_per_trade'] = df['avg_profit_per_trade'] - 0.1
    df['profit'] = (df['avg_profit_per_trade']) * df['total_trades']
    df = df[df['profit'] > 20]

    df = df[df['trades_this_year'] > 0]

    df['score'] = np.log(df['avg_profit_per_trade'] + 1) * df['profit'] * (df['avg_profit_per_trade']) * df['total_trades'] / -(df['Max Drawdown'] - 0.5)


    # 按照 score 降序排序取前100
    df = df.sort_values(by='score', ascending=False).head(100)
    params_list = []
    print(f"筛选后剩余 {len(df)} 个参数组合")
    for index, row in df.iterrows():
        params = parse_backtest_filename(row['file_name'])
        params_list.append(params)
    return params_list

def get_good_param_new(final_df_path='kline_data/result_df.csv',
                   final_df_path_1s='kline_data/result_df_1s.csv'):
    # ====================== 1. 处理 1min 数据 ======================
    df = pd.read_csv(final_df_path)
    df = df[df['avg_profit_per_trade'] > 0.1].copy()
    df['avg_profit_per_trade'] = df['avg_profit_per_trade'] - 0.1
    df['profit'] = df['avg_profit_per_trade'] * df['total_trades']
    df = calculate_score_and_target_str(df)

    # ====================== 2. 处理 1s 数据 ======================
    df_1s = pd.read_csv(final_df_path_1s)
    df_1s = df_1s[df_1s['avg_profit_per_trade'] > 0.1].copy()
    df_1s['avg_profit_per_trade'] = df_1s['avg_profit_per_trade'] - 0.1
    df_1s['profit'] = df_1s['avg_profit_per_trade'] * df_1s['total_trades']
    df_1s = calculate_score_and_target_str(df_1s)

    # ====================== 3. 给所有冲突字段加上后缀（除 target_str 外全部加） ======================
    # 1min 数据全部加上 _1min
    rename_1min = {col: f"{col}_1min" for col in df.columns if col != 'target_str'}
    df = df.rename(columns=rename_1min)

    # 1s 数据全部加上 _1s
    rename_1s = {col: f"{col}_1s" for col in df_1s.columns if col != 'target_str'}
    df_1s = df_1s.rename(columns=rename_1s)

    # ====================== 4. 合并成一行一个 target_str ======================
    # outer 合并，任何一个 df 有的 target_str 都会保留
    result = pd.merge(df, df_1s, on='target_str', how='inner')

    # ====================== 5. 计算同一个 target_str 下两个 df 的聚合值 ======================
    result['score_sum'] = result['score_1min'].fillna(0) + result['score_1s'].fillna(0)
    result['score_mul'] = result['score_1min'].fillna(0) * result['score_1s'].fillna(0)

    result['min_score'] = result[['score_1min', 'score_1s']].min(axis=1)

    # ====================== 6. 按 score_sum 降序排序（推荐） ======================
    result = result.sort_values(by='score_sum', ascending=False).reset_index(drop=True)

    return result

def generate_trade_instruction(action, row, signal_dir, inst_id_list, is_recovery=False):
    """
    生成标准化的交易指令数据字典
    """
    main_inst = inst_id_list[0] if len(inst_id_list) > 0 else 'MAIN'
    sub_inst = inst_id_list[1] if len(inst_id_list) > 1 else 'SUB'
    beta = row.get('beta', 1.0)

    # 确定文字描述
    if action == 'open':
        direction = "做多" if signal_dir == -1 else "做空"
        desc = f"{direction}价差 (Main: {main_inst}, Sub: {beta:.3f} {sub_inst})"
    else:
        desc = "平仓 (Close Positions)"

    if is_recovery:
        desc = f"[恢复模式补发] {desc}"

    return {
        "time": str(row['open_time']),
        "action": action,
        "signal_dir": int(signal_dir),
        "main_inst": main_inst,
        "sub_inst": sub_inst,
        "beta": float(beta),
        "close_main": float(row.get('close_main', 0)),
        "close_sub": float(row.get('close_sub', 0)),
        "description": desc,
        "status": "pending_execution"
    }


def process_signal_df(signal_df, params, inst_id_list, base_order_file="orders.json", recovery_mode=False):
    """
    根据信号数据进行开平仓判断，生成并记录交易指令，包含异常恢复机制。
    """
    # ---------------- Step 1: 数据校验与预处理 ----------------
    if len(signal_df) < 2:
        return None

    df = signal_df.copy()
    if 'open_time' not in df.columns:
        if 'index' in df.columns:
            df.rename(columns={'index': 'open_time'}, inplace=True)
        elif 'timestamp' in df.columns:
            df.rename(columns={'timestamp': 'open_time'}, inplace=True)

    # ---------------- Step 2: 准备文件路径与读取历史状态 ----------------
    params_key = "_".join([f"{k}_{v}" for k, v in params.items()]) + "_"
    insts_str = "_".join(inst_id_list)
    order_file = base_order_file.replace(".json", f"_{insts_str}.json")

    # 直接使用你外部的 read_json
    order_info = read_json(order_file)
    if params_key not in order_info:
        order_info[params_key] = []

    history_orders = order_info[params_key]

    # 判断当前本地记录中的真实持仓状态：如果有历史记录且最后一次是 open，则认为正在持仓
    is_holding = bool(history_orders) and history_orders[-1].get('action') == 'open'

    # ---------------- Step 3: 获取信号数据 ----------------
    latest_row = df.iloc[-1]
    prev_row = df.iloc[-2]

    curr_signal = latest_row['signal']
    prev_signal = prev_row['signal']

    instruction = None

    # ---------------- Step 4: 核心逻辑决策树 ----------------
    # 场景 A: 正常跳变 - 触发开仓
    if curr_signal != 0 and prev_signal == 0:
        if not is_holding:
            instruction = generate_trade_instruction('open', latest_row, curr_signal, inst_id_list)

    # 场景 B: 正常跳变 - 触发平仓
    elif curr_signal == 0 and prev_signal != 0:
        if is_holding:
            instruction = generate_trade_instruction('close', latest_row, 0, inst_id_list)

    # 场景 C: 信号延续 - 检查是否需要恢复补单
    elif recovery_mode:
        if curr_signal != 0 and prev_signal != 0 and not is_holding:
            # 有信号但本地无持仓：漏单补开
            instruction = generate_trade_instruction('open', latest_row, curr_signal, inst_id_list, is_recovery=True)

        elif curr_signal == 0 and prev_signal == 0 and is_holding:
            # 没信号但本地有持仓：漏单补平
            instruction = generate_trade_instruction('close', latest_row, 0, inst_id_list, is_recovery=True)

    # ---------------- Step 5: 输出与持久化 ----------------
    if instruction:
        print(
            f"[{instruction['time']}] 产生新的交易指令: {instruction['action'].upper()} -> {instruction['description']}")

        # 顺序追加并保存
        order_info[params_key].append(instruction)
        save_json(order_file, order_info)  # 直接使用你外部的 save_json

        return instruction

    return None



def monitor_worker(inst_id_list):
    """
    【工作线程】
    针对【一组】币种的无限循环监控任务。
    """
    try:
        # 打印列表内容，用 join 拼接成字符串更直观
        insts_str = "_".join(inst_id_list)
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
                    signal_df = generate_pair_trading_signals(merged_df=merged_df, main_col='close_main', sub_col='close_sub', window=60, z_entry=params['z_entry'], z_exit=params['z_exit'], delta=params['delta'], ve=params['ve'])
                    signal_df['params'] = [copy.deepcopy(params) for _ in range(len(signal_df))]
                    # 开始处理有交易信号的数据
                    process_signal_df(signal_df, params, inst_id_list)
                    # # 检测signal_df中是否有信号出现（即 signal 列中非 0 的行）
                    # if (signal_df['signal'] != 0).any():
                    #     log_info(inst_id_list[0], f"检测到交易信号，正在处理... (Params: {params})")
                    #     has_signal_df_list.append(signal_df)


                # print()

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
    # get_good_param()

    try:
        main()
    except KeyboardInterrupt:
        print("\n程序已手动停止")