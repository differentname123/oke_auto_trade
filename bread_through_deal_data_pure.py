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
    """处理单个文件的函数，并根据条件过滤数据；
    如果过滤结果不为空，则返回过滤后的整个 DataFrame，
    如果过滤后为空，则返回备份数据中 score_score 最大的那一行。
    """
    try:
        # 读取数据
        df = pd.read_parquet(file)

        origin_len = len(df)

        # 检查必要列的数组长度
        check_array_len(
            df,
            [
                'monthly_kai_count_detail',
                'monthly_net_profit_detail',
                'weekly_kai_count_detail',
                'weekly_net_profit_detail'
            ]
        )

        # # 预计算各项 profit_risk_score 指标
        # npr = df['net_profit_rate']
        # df['profit_risk_score_con'] = -(npr * npr) / df['max_consecutive_loss']
        # df['profit_risk_score'] = -(npr * npr) / df['fu_profit_sum']
        # df['profit_risk_score_pure'] = -npr / df['fu_profit_sum']

        # 计算奖励与惩罚相关数据
        df = compute_rewarded_penalty_from_flat_df(df)

        # 备份处理前的数据，后续用于 fallback（确保原始数据完整性）
        df_backup = df.copy()

        # 根据 score_final 过滤
        # df = df[df['score_final'] > -1000]
        # 添加原始差值相关列，并根据 norm_diff_score 过滤
        df = add_raw_diff_columns(df)
        # df = df[df['norm_diff_score'] > -1000]

        # 计算 score_score，并对负值情况进行调整
        df['score_score'] = df['score_final'] * df['norm_diff_score']
        df['score_score'] = np.where(
            (df['score_final'] < 0) | (df['norm_diff_score'] < 0),
            -abs(df['score_score']),
            df['score_score']
        )

        # 如果经过过滤后结果不为空，正常返回整个 DataFrame
        if not df.empty:
            return df
        else:
            # 当过滤结果为空时，对备份数据进行处理：
            # 为备份数据添加原始差值相关列，并计算 score_score
            df_backup = add_raw_diff_columns(df_backup)
            df_backup['score_score'] = df_backup['score_final'] * df_backup['norm_diff_score']
            df_backup['score_score'] = np.where(
                (df_backup['score_final'] < 0) | (df_backup['norm_diff_score'] < 0),
                -abs(df_backup['score_score']),
                df_backup['score_score']
            )
            # 取 score_score 最大的那一行
            max_idx = df_backup['score_score'].idxmax()
            return df_backup.loc[[max_idx]]
    except Exception as e:
        print(f"处理文件 {file} 时出错: {e}")
        return None


def load_and_merger_data(inst_id, is_reverse):
    """
    加载并合并ga算法回测得到的数据，需要过滤部分数据
    :return:
    """
    start_time = time.time()
    file_list = os.listdir('temp')
    file_list = [file for file in file_list if inst_id in file and
                 'donchian_1_20_1_relate_400_1000_100_1_40_6_cci_1_2000_1000_1_2_1_atr_1_3000_3000_boll_1_3000_100_1_50_2_rsi_1_1000_500_abs_1_100_100_40_100_1_macd_300_1000_50_macross_1_3000_100_1_3000_100_' in file and
                 'pkl' not in file and '_stats' in file and '1m' in file and str(is_reverse) in file]
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

    else:
        print("没有符合条件的数据")
        result_df = pd.DataFrame()

    print(f"{inst_id} 合并后的数据行数: {len(result_df)} 耗时: {time.time() - start_time:.2f}秒")
    result_df = calculate_downside_metrics(result_df, ['weekly_net_profit_detail', 'monthly_net_profit_detail'],
                                           threshold=1)
    result_df = compute_rewarded_penalty_from_flat_df(result_df)
    df = add_raw_diff_columns(result_df)
    df = df[df['norm_diff_score'] > -1000]

    # 计算 score_score，并对负值情况进行调整
    df['score_score'] = df['score_final'] * df['norm_diff_score']
    df['score_score'] = np.where(
        (df['score_final'] < 0) | (df['norm_diff_score'] < 0),
        -abs(df['score_score']),
        df['score_score']
    )

    return df


def _single_downside_metrics(data_array, threshold=0):
    """
    备用的单条数据计算函数，供 fallback 用
    """
    try:
        arr = np.asarray(data_array, dtype=float)
    except Exception:
        return 0.0, 0
    mask = arr < threshold
    cnt = int(mask.sum())
    if cnt >= 2:
        std = float(np.std(arr[mask], ddof=1))
    else:
        std = 0.0
    return std, cnt


def calculate_downside_metrics(df: pd.DataFrame,
                               column_list: list,
                               threshold: float = 0.0) -> pd.DataFrame:
    """
    对 df 中指定的列（每个单元格是数值 list/ndarray），
    一次性向量化计算 DownsideStdDev 与 DownsideCount。
    """
    if not isinstance(df, pd.DataFrame):
        print("Warning: Input 'df' is not a pandas DataFrame. Returning original input.")
        return df
    if not isinstance(column_list, list):
        print("Warning: 'column_list' is not a list. Returning original DataFrame.")
        return df

    df_result = df.copy()
    n = len(df)

    for col in column_list:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found. Skipping.")
            continue

        series = df[col]
        # 判定哪些行是“统一长度、可向量化”的
        good_idx = series.map(lambda x: isinstance(x, (list, np.ndarray)))
        if good_idx.any():
            # 取所有有效的、且长度相同的
            arrs = [np.asarray(x, dtype=float) for x in series[good_idx]]
            lengths = [a.shape[0] for a in arrs]
            if len(set(lengths)) == 1:
                # 全部同长度，走向量化
                arr2d = np.stack(arrs, axis=0)  # shape = (n_good, L)
                mask = arr2d < threshold  # shape = (n_good, L)
                counts = mask.sum(axis=1)  # shape = (n_good,)
                # 在不满足阈值的位置置 NaN，再做 nanstd
                tmp = np.where(mask, arr2d, np.nan)
                stds = np.nanstd(tmp, axis=1, ddof=1)
                # 少于 2 个样本时 np.nanstd 会给 nan，我们把它置 0
                stds = np.where(counts >= 2, stds, 0.0)

                # 准备好两个全零/零数组，然后填入向量化结果
                down_count = np.zeros(n, dtype=int)
                down_std = np.zeros(n, dtype=float)
                idx_good = np.flatnonzero(good_idx.values)
                down_count[idx_good] = counts
                down_std[idx_good] = stds
            else:
                # 长度不一，退回到单条计算
                down_std = np.zeros(n, dtype=float)
                down_count = np.zeros(n, dtype=int)
                for i, x in enumerate(series):
                    s, c = _single_downside_metrics(x, threshold)
                    down_std[i] = s
                    down_count[i] = c
        else:
            # 全部非 list/ndarray，全部退回
            down_std = np.zeros(n, dtype=float)
            down_count = np.zeros(n, dtype=int)
            for i, x in enumerate(series):
                s, c = _single_downside_metrics(x, threshold)
                down_std[i] = s
                down_count[i] = c

        # 直接赋两列
        df_result[f'{col}_DownsideStdDev'] = down_std
        df_result[f'{col}_DownsideCount'] = down_count

    return df_result


def compute_rewarded_penalty_from_flat_df(df: pd.DataFrame) -> pd.Series:
    """
    向量化地计算每行的(奖励 - 惩罚*100)得分，返回一个 pd.Series，可直接赋值给 df["score"]。
    """
    # 1. 初始化 penalty 和 reward
    idx = df.index
    penalty = pd.Series(0.0, index=idx)
    reward = pd.Series(0.0, index=idx)

    features = [
        dict(col='max_consecutive_loss', thr=-20, sign=1, pf=10, power=2, rf=1, na=-10000),
        dict(col='net_profit_rate', thr=50, sign=1, pf=10, power=2, rf=1 / 100, na=-10000),
        dict(col='kai_count', thr=50, sign=1, pf=10, power=2, rf=1 / 100, na=-10000),
        dict(col='active_month_ratio', thr=0.8, sign=1, pf=10000, power=2, rf=2, na=-10000),
        dict(col='weekly_loss_rate', thr=0.2, sign=-1, pf=1000, power=2, rf=5, na=10000),
        dict(col='monthly_loss_rate', thr=0.2, sign=-1, pf=1000, power=2, rf=5, na=10000),
        dict(col='top_profit_ratio', thr=0.5, sign=-1, pf=1000, power=2, rf=2, na=10000),
        dict(col='hold_time_mean', thr=3000, sign=-1, pf=1 / 10000, power=2, rf=1 / 3000, na=100000),
        dict(col='max_hold_time', thr=10000, sign=-1, pf=1 / 10000, power=2, rf=1 / 10000, na=100000),
        dict(col='avg_profit_rate', thr=10, sign=1, pf=1, power=2, rf=1 / 100, na=-10000),
    ]

    # 3. 逐特征向量化累加
    for f in features:
        # 批量取值、填缺失、转浮点
        vals = df[f['col']].fillna(f['na']).astype(float)

        if f['sign'] == 1:
            diff_pen = (f['thr'] - vals).clip(lower=0)
            diff_rev = (vals - f['thr']).clip(lower=0)
        else:
            diff_pen = (vals - f['thr']).clip(lower=0)
            diff_rev = (f['thr'] - vals).clip(lower=0)

        penalty += diff_pen.pow(f['power']) * f['pf']
        reward += diff_rev * f['rf']

    # 4. 合成最终得分
    score = reward - penalty * 100
    df['score_final'] = score
    return df


def add_raw_diff_columns(df):
    """
    计算并只在原 df 上新增综合得分 norm_diff_score：
      - norm_diff_score：所有指标的 norm_diff 加权求和（负数乘以100）

    norm_diff 的计算方式是：
      - diff = 原始值 - 阈值 (min型) 或 阈值 - 原始值 (max型)，保留正负
      - norm_diff = diff / max(abs(diff))，保留正负

    使用向量化计算 norm_diff_score 提高效率。

    返回值：原 df（已被修改，只新增了 norm_diff_score 列）
    """
    metrics = [
        ("max_consecutive_loss", -20,  "min"),
        ("net_profit_rate",      50,  "min"),
        ("kai_count",            50,  "min"),
        ("active_month_ratio",   0.8,  "min"),
        ("weekly_loss_rate",     0.2,  "max"),
        ("monthly_loss_rate",    0.2,  "max"),
        ("top_profit_ratio",     0.5,  "max"),
        ("hold_time_mean",       3000, "max"),
        ("max_hold_time",        10000,"max"),
        ("avg_profit_rate",      10,   "min"),
    ]

    # List to store the calculated norm_diff Series for each metric
    temp_norm_diff_series_list = []

    for col, thresh, bound in metrics:
        # Check if the metric column exists, use NaN series if not
        if col in df.columns:
            s = df[col]
        else:

            s = pd.Series(np.nan, index=df.index, name=col)




        # 计算原始 diff（保留正负）
        if bound == "min":
            diff = s - thresh
        else: # bound == "max"
            diff = thresh - s

        max_abs = diff.abs().max()

        if pd.notna(max_abs) and max_abs != 0:
             norm_diff = diff / max_abs
        else:

             norm_diff = pd.Series(np.nan, index=df.index, name=col) # Give it a name

        temp_norm_diff_series_list.append(norm_diff)


    if not temp_norm_diff_series_list:
         df["norm_diff_score"] = np.nan # Or 0, depending on desired default
    else:
        temp_norm_diff_df = pd.concat(temp_norm_diff_series_list, axis=1)
        weighted_norm_diff_values = np.where(temp_norm_diff_df < 0, temp_norm_diff_df * 100, temp_norm_diff_df)
        df["norm_diff_score"] = weighted_norm_diff_values.sum(axis=1)

    return df

@njit
def kadane_min(arr):
    current_sum = arr[0]
    min_sum = arr[0]
    for i in range(1, arr.shape[0]):
        x = arr[i]
        current_sum = x if x < current_sum + x else current_sum + x
        min_sum = current_sum if current_sum < min_sum else min_sum
    return min_sum

def process_df_numba(df):
    origin_len = len(df)
    arrs = df['weekly_net_profit_detail_new20'].values

    # 第一次调用时有编译开销，随后即享受 JIT 加速
    min_contig = [kadane_min(a) for a in arrs]
    min_vals   = [a.min() if a.size>0 else np.nan for a in arrs]

    df = df.copy()
    df['min_contiguous_sum'] = min_contig
    df['min_value']          = min_vals

    mask = (df['min_value'] > df['weekly_net_profit_min']) & \
           (df['min_contiguous_sum'] > df['max_consecutive_loss'])
    df = df.loc[mask]
    print(f"处理后 {len(df)} 行 ← 原始 {origin_len} 行")
    return df

def merge_df(inst_id):
    is_reverse_list = [False, True]
    merge_columns = ["kai_column", "pin_column"]
    target_columns = [
        "kai_count", "hold_time_mean", "net_profit_rate",
        "fix_profit", "avg_profit_rate", "same_count",
        "weekly_net_profit_detail"
    ]
    columns_to_read = merge_columns + target_columns
    for is_reverse in is_reverse_list:
        output_file = f'temp_back/{inst_id}_{is_reverse}_pure_data_with_future.parquet'
        origin_file = f'temp_back/{inst_id}_{is_reverse}_pure_data.parquet'
        origin_good_df = pd.read_parquet(origin_file)

        # 读取 Parquet 文件时仅读取需要的列，指定 engine 可根据实际情况选择
        new_df = pd.read_parquet(
            f'temp_back/{inst_id}_{is_reverse}_pure_data.parquet_1m_200000_{inst_id}-USDT-SWAP_2025-05-01.csvstatistic_results_final.parquet',
            columns=columns_to_read,
            engine='pyarrow'
        )

        # 选择需要的列，并重命名以区分
        new_df_selected = new_df.copy()
        new_df_selected = new_df_selected.rename(
            columns={col: col + "_new20" for col in target_columns}
        )

        origin_good_df = origin_good_df.set_index(merge_columns)
        new_df_selected = new_df_selected.set_index(merge_columns)

        # 使用 join 进行合并（左连接），然后重置索引
        origin_good_df = origin_good_df.join(new_df_selected, how="left").reset_index()
        origin_good_df = process_df_numba(origin_good_df)
        origin_good_df.to_parquet(output_file, index=False)


def group_statistics_and_inst_details(df: pd.DataFrame,
                                                group_cols,
                                                target_cols) -> pd.DataFrame:
    """
    对 df 按 group_cols 分组，对 target_cols 计算 max, min, mean,
    positive_ratio, group_count（分组大小），并收集每个 target_col 的
    {inst_id: value} 字典。

    Arguments:
        df: 输入的 Pandas DataFrame。
        group_cols: 用于分组的列名列表。
        target_cols: 需要计算统计信息和收集值的列名列表。

    Returns:
        一个 Pandas DataFrame，包含分组键、统计结果以及每个 target_col 的
        {inst_id: value} 字典。
    """
    inst_id_col = 'inst_id'
    if inst_id_col not in df.columns:
        raise ValueError(f"inst_id_col '{inst_id_col}' not found in DataFrame columns.")

    # 按分组键分组，保留 NaN
    grouped = df.groupby(group_cols, dropna=False)

    # 计算分组大小，使用内置 size() 方法（C层级优化）
    group_count = grouped.size().rename("group_count")
    result_df = group_count.to_frame()

    for col in target_cols:
        # 计算 max, min, mean，使用内置聚合函数agg
        agg_stats = grouped[col].agg(['max', 'min', 'mean'])
        agg_stats.columns = [f"{col}_{stat}" for stat in ['max', 'min', 'mean']]
        result_df = result_df.join(agg_stats)

        # 计算正例比例：
        # 先将整列转换为数值，再生成布尔标记列（大于 0 为1，否则为0），
        # 然后对该布尔列用内置聚合函数求和，实现向量化操作。
        tmp_flag_col = f"_{col}_pos_flag"
        df[tmp_flag_col] = pd.to_numeric(df[col], errors='coerce').gt(0).astype(int)
        pos_ratio = grouped[tmp_flag_col].sum() / group_count
        result_df[f"{col}_positive_ratio"] = pos_ratio
        df.drop(columns=[tmp_flag_col], inplace=True)

        # 生成 {inst_id: value} 字典：
        # 先将 inst_id 与目标列聚合为列表，再利用 apply 对每个分组
        # 用 zip 构造字典（减少了在每个循环内部的纯 Python 操作次数）
        details_lists = grouped.agg({inst_id_col: list, col: list})
        result_df[f"{col}_details"] = details_lists.apply(lambda row: dict(zip(row[inst_id_col], row[col])), axis=1)

    return result_df.reset_index()


def get_common_data():
    inst_id_list = ['BTC', 'ETH', 'SOL', 'TON', 'DOGE', 'XRP', 'OKB']
    combinations_list = []

    for inst_id in inst_id_list:
        file_path = f'temp_back\statistic_results_final_{inst_id}_False.parquet'
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            df['inst_id'] = inst_id
            df = compute_rewarded_penalty_from_flat_df(df)

            df = df[df['max_consecutive_loss'] > -30]
            combinations_list.append(df)
    all_df = pd.concat(combinations_list, ignore_index=True)
    # 将all_df按照['kai_column', 'pin_column']分组，删除只有一行的组
    group_sizes = all_df.groupby(['kai_column', 'pin_column'])['kai_column'].transform('size')
    all_df = all_df[group_sizes > 2]
    result = group_statistics_and_inst_details(
        all_df,
        group_cols=['kai_column', 'pin_column'],
        target_cols=[
            'max_consecutive_loss',
            'net_profit_rate',
            'kai_count',
            'score_final'
        ]
    )
    return result


def example():
    get_common_data()
    inst_id_list = ['BTC', 'ETH', 'SOL', 'TON', 'DOGE', 'XRP', 'OKB']
    is_reverse = False
    # pd.read_parquet(f'temp/final_good_BTC_True_filter_all.parquet')

    for inst_id in inst_id_list:
        output_path = f'temp_back/{inst_id}_{is_reverse}_pure_data.parquet'
        if os.path.exists(output_path):
            result_df = pd.read_parquet(output_path)
            df = pd.read_parquet(f'temp_back\statistic_results_final_{inst_id}_False.parquet')
            df = compute_rewarded_penalty_from_flat_df(df)

            result_df = add_raw_diff_columns(result_df)
        result_df = load_and_merger_data(inst_id, is_reverse)
        result_df.to_parquet(output_path, index=False)

    # for inst_id in inst_id_list:
    #     output_file = f'temp_back/{inst_id}_{is_reverse}_pure_data_with_future.parquet'
    #     if os.path.exists(output_file):
    #         result_df = pd.read_parquet(output_file)
    #
    #     merge_df(inst_id)



if __name__ == '__main__':
    example()
