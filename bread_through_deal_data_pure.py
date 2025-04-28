import os
import itertools
import json
import math
import re
from multiprocessing import Pool
import multiprocessing as mp

import multiprocessing
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numba import njit
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
import ast

from tqdm import tqdm


def check_array_len(df: pd.DataFrame, column_list: list) -> None:
    """
    检查数据框中指定列的每一行的长度是否相同

    对于指定的每一列，如果所有行的长度一致，则打印统一的长度；
    否则，打印所有不同的长度值。

    :param df: pandas DataFrame 对象
    :param column_list: 要检查的列名称列表
    """
    for col in column_list:
        if col not in df.columns:
            print(f"列 '{col}' 不存在于数据框中。")
            continue

        # 使用向量化操作计算每行的长度
        lengths = df[col].apply(len)
        unique_lengths = lengths.unique()

        if len(unique_lengths) == 1:
            pass
            # print(f"每一行的 '{col}' 的长度都相同，为：{unique_lengths[0]}")
        else:
            print(f"列 '{col}' 的不同长度值有: {set(unique_lengths)}")

def process_load_filter_data(file):
    """处理单个文件的函数，并根据条件过滤数据"""
    try:
        df = pd.read_parquet(file)
        check_array_len(df, ['monthly_kai_count_detail', 'monthly_net_profit_detail', 'weekly_kai_count_detail', 'weekly_net_profit_detail'])

        # 预计算 net_profit_rate 的平方，简化 profit_risk_score 的计算
        npr = df['net_profit_rate']
        df['profit_risk_score_con'] = -(npr * npr) / df['max_consecutive_loss']
        df['profit_risk_score'] = -(npr * npr) / df['fu_profit_sum']
        df['profit_risk_score_pure'] = -npr / df['fu_profit_sum']

        # 根据筛选条件过滤数据
        df = df[
            (df['max_consecutive_loss'] >= -20) &     # 最大连续亏损必须大于等于 -20
            (df['net_profit_rate'] >= 100) &           # 净盈利率至少为 100
            (df['kai_count'] >= 100) &                # 交易次数（kai_count）不少于 100
            (df['active_month_ratio'] >= 0.8) &         # 活跃月份比率至少为 0.8
            (df['weekly_loss_rate'] <= 0.2) &           # 每周亏损率不超过 0.2
            (df['monthly_loss_rate'] <= 0.2) &          # 每月亏损率不超过 0.2
            (df['top_profit_ratio'] <= 0.5) &           # 盈利峰值比率不超过 0.5
            (df['hold_time_mean'] <= 3000) &            # 平均持仓时间不超过 3000
            (df['avg_profit_rate'] >= 10) &             # 平均盈利率至少为 10
            (df['max_hold_time'] <= 10000)              # 最大持仓时间不超过 10000
        ]

        return df
    except Exception as e:
        print(f"处理文件 {file} 时报错: {e}")
        return None


def load_and_merger_data(inst_id):
    """
    加载并合并ga算法回测得到的数据，需要过滤部分数据
    :return:
    """
    file_list = os.listdir('temp')
    file_list = [file for file in file_list if inst_id in file and
                 'donchian_1_20_1_relate_400_1000_100_1_40_6_cci_1_2000_1000_1_2_1_atr_1_3000_3000_boll_1_3000_100_1_50_2_rsi_1_1000_500_abs_1_100_100_40_100_1_macd_300_1000_50_macross_1_3000_100_1_3000_100_' in file and
                 'pkl' not in file and 'sta' in file and '1m' in file]
    print(f"找到 {len(file_list)} 个文件")

    # 完善的文件路径
    file_list = [os.path.join('temp', file) for file in file_list]

    # 使用多进程池并行处理文件
    with mp.Pool(processes=1) as pool:
        df_list = pool.map(process_load_filter_data, file_list)

    # 过滤掉 None 值
    df_list = [df for df in df_list if df is not None]
    # 合并所有 DataFrame
    if df_list:
        result_df = pd.concat(df_list, ignore_index=True)
        result_df = result_df.drop_duplicates(subset=['kai_column', 'pin_column'])
        return result_df
    else:
        print("没有符合条件的数据")
        return pd.DataFrame()


def example():
    inst_id_list = ['BTC', 'ETH', 'SOL', 'TON', 'DOGE', 'XRP', 'PEPE']
    for inst_id in inst_id_list:
        load_and_merger_data(inst_id)


if __name__ == '__main__':
    example()
