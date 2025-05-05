#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的优化方案代码示例

优化重点：
1. 避免重复创建进程池，将任务提交到全局进程池中并行处理；
2. 在 process_single_feature 中使用向量化操作对分箱聚合进行加速；
3. 派生特征的生成按批量进行，减少每批次创建进程池的次数；

步骤：
1. 数据加载与预处理；
2. 原始和派生特征分别并行执行分箱统计，计算每个箱内指标以及 Spearman 单调性；
3. 合并所有分箱结果并保存至 CSV 文件；
4. 提供辅助函数用于内存占用优化、DataFrame 合并等。
"""
from typing import List, Dict, Any, Tuple

import math
import os
import time
from functools import reduce
import concurrent.futures
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 过滤 Pandas 性能警告
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# 设置中文显示，防止中文乱码（根据系统环境适当调整字体）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from scipy.stats import spearmanr


def merge_and_compute(df_list, merge_on=["feature", "bin_seq"], metric_cols=None):
    """
    合并 DataFrame 列表，并按 merge_on 键计算多个指标列的最大、最小和均值。
    同时计算两种打分指标：
      - score: 基于指标最小值的乘积
      - score1: 基于均值的乘积
    """
    if not df_list:
        return pd.DataFrame()
    if metric_cols is None:
        metric_cols = [
            'avg_net_profit_rate_new20', 'pos_ratio',
            'spearman_avg_net_profit_rate', 'spearman_pos_ratio', 'spearman_score'
        ]

    df_renamed_list = []
    for idx, df in enumerate(df_list):
        df = df.copy()
        df['spearman_score'] = df['spearman_avg_net_profit_rate'] * df['spearman_pos_ratio']
        required_cols = merge_on + metric_cols
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame at index {idx} 缺少必要的列: {missing}")
        sub_df = df[required_cols].copy()
        rename_dict = {col: f"{col}_{idx}" for col in metric_cols}
        sub_df.rename(columns=rename_dict, inplace=True)
        df_renamed_list.append(sub_df)

    merged_df = reduce(lambda left, right: pd.merge(left, right, on=merge_on, how='inner'), df_renamed_list)
    for col in metric_cols:
        col_names = [f"{col}_{i}" for i in range(len(df_renamed_list))]
        merged_df[f"{col}_max"] = merged_df[col_names].max(axis=1)
        merged_df[f"{col}_min"] = merged_df[col_names].min(axis=1)
        merged_df[f"{col}_mean"] = merged_df[col_names].mean(axis=1)
    merged_df['score'] = merged_df['avg_net_profit_rate_new20_min'] * merged_df['pos_ratio_min']
    merged_df['score1'] = merged_df['avg_net_profit_rate_new20_mean'] * merged_df['pos_ratio_mean']

    output_cols = merge_on.copy()
    for col in metric_cols:
        output_cols.extend([f"{col}_max", f"{col}_min", f"{col}_mean"])
    output_cols.extend(['score', 'score1'])
    return merged_df[output_cols]


def process_single_feature(feature, feat_values, target_values, n_bins=50):
    """
    对单个特征进行分箱统计，并计算箱内指标的 Spearman 单调性。
    如果特征的唯一值数量不足 n_bins，则返回 None。

    优化：对分箱聚合采用先排序后利用 np.unique 获取分箱索引分组信息，减少 Python 循环次数。
    """
    # start_time = time.time()
    try:
        if feat_values.size == 0:
            return None
        # 如果唯一值不足 n_bins，则不分箱
        unique_vals = np.unique(feat_values)
        if unique_vals.size < n_bins:
            return None

        quantiles = np.percentile(feat_values, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(quantiles)
        if bins.size < 2:
            return None

        # 计算每个数据对应的分箱索引
        bin_indices = np.searchsorted(bins, feat_values, side='right') - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
        total_count = feat_values.shape[0]

        # 对分箱索引排序，并利用 np.unique 获得每个分箱的起始位置和计数
        order = np.argsort(bin_indices)
        sorted_bins = bin_indices[order]
        unique_bins, start_idxs, counts = np.unique(sorted_bins, return_index=True, return_counts=True)

        # 预分配存储聚合结果的数组
        avg_net_profit = np.empty(unique_bins.shape[0], dtype=np.float32)
        min_net_profit = np.empty(unique_bins.shape[0], dtype=np.float32)
        max_net_profit = np.empty(unique_bins.shape[0], dtype=np.float32)
        pos_ratio = np.empty(unique_bins.shape[0], dtype=np.float32)

        # 对每个分箱分组计算统计指标
        for i, (start, cnt) in enumerate(zip(start_idxs, counts)):
            indices = order[start:start + cnt]
            cur_targets = target_values[indices]
            avg_net_profit[i] = np.mean(cur_targets)
            min_net_profit[i] = np.min(cur_targets)
            max_net_profit[i] = np.max(cur_targets)
            pos_ratio[i] = np.mean(cur_targets > 0)

        # 生成分箱区间字符串信息
        intervals = [f"[{bins[bin_val]:.4f}, {bins[bin_val + 1]:.4f})" for bin_val in unique_bins]
        result_df = pd.DataFrame({
            "feature": feature,
            "bin_seq": unique_bins + 1,
            "bin_interval": intervals,
            "min_net_profit": min_net_profit,
            "max_net_profit": max_net_profit,
            "avg_net_profit_rate_new20": avg_net_profit,
            "pos_ratio": pos_ratio,
            "count": counts,
            "bin_ratio": counts / total_count
        })

        if result_df.shape[0] > 1:
            spearman_avg, _ = spearmanr(result_df["bin_seq"], result_df["avg_net_profit_rate_new20"])
            spearman_pos, _ = spearmanr(result_df["bin_seq"], result_df["pos_ratio"])
        else:
            spearman_avg = spearman_pos = np.nan
        result_df["spearman_avg_net_profit_rate"] = spearman_avg
        result_df["spearman_pos_ratio"] = spearman_pos

        # print(f"特征 {feature} 处理完成，耗时 {time.time() - start_time:.2f} 秒")
        return result_df
    except Exception as e:
        print(f"处理特征 {feature} 时遇到异常: {e}")
        return None


def process_features_parallel(feature_names, feature_data, target_values, n_bins=50, max_workers=30):
    """
    利用全局进程池并行处理特征，避免在每个批次中重复创建进程池。

    参数：
      - feature_names: 需要处理的特征名称列表
      - feature_data: 字典，键为特征名称，值为对应的数值数组
      - target_values: 目标变量的数值数组
      - n_bins: 分箱数量
      - max_workers: 最大并行进程数

    返回处理后的结果列表。
    """
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_feature, feature, feature_data[feature], target_values, n_bins): feature
            for feature in feature_names
        }
        for future in concurrent.futures.as_completed(futures):
            feat = futures[future]
            try:
                res = future.result()
                if res is not None:
                    results.append(res)
            except Exception as e:
                print(f"【process_features_parallel】：特征 {feat} 处理出错: {e}")
    return results


def process_data_flat(data_df, target_column_list):
    """
    对每个目标指标：
      1. 按目标指标倒序排序 DataFrame；
      2. 取排序后前 1% 至 10% 的数据（至少一行）；
      3. 针对每个子集，对 'feature' 列做拆分（'-' 分隔）并统计出现次数。
    返回拼接后的 DataFrame，包含目标指标、百分比、阈值、特征名称及计数。
    """
    data_df = data_df[(data_df['bin_seq'] > 90) | (data_df['bin_seq'] < 10)]
    output_rows = []
    percentage_list = [x / 100 for x in range(1, 11)]
    for target in target_column_list:
        if target not in data_df.columns:
            print(f"列 {target} 不存在于 DataFrame 中，跳过。")
            continue
        sorted_df = data_df.sort_values(by=target, ascending=False).reset_index(drop=True)
        total_rows = sorted_df.shape[0]
        for perc in percentage_list:
            n = max(1, int(math.ceil(total_rows * perc)))
            subset_df = sorted_df.head(n)
            threshold_value = sorted_df[target].iloc[n - 1]
            feature_list = []
            for feat in subset_df['feature']:
                if '-' in feat:
                    feature_list.extend(feat.split('-')[:2])
                else:
                    feature_list.append(feat)
            counter = {}
            for f in feature_list:
                counter[f] = counter.get(f, 0) + 1
            for feature, count in counter.items():
                output_rows.append({
                    'target_column': target,
                    'percentage': f"{int(perc * 100)}",
                    '临界值': threshold_value,
                    'feature': feature,
                    'count': count
                })
    result_df = pd.DataFrame(output_rows)
    result_df = result_df.sort_values(
        by=['target_column', 'percentage', 'count'],
        ascending=[True, True, False]
    ).reset_index(drop=True)
    return result_df

def main(n_bins=50, batch_size=10):
    """
    主流程：
      1. 加载数据；
      2. 并行处理原始特征与派生（交叉）特征的分箱；
      3. 合并所有分箱结果，并保存 CSV 文件；
      4. 可根据需要调用 process_data_flat、merge_and_compute、auto_reduce_precision 进行后续处理。
    """
    print("【主流程】：开始处理数据")
    inst_id_list = ['BTC', 'ETH', 'SOL', 'TON', 'DOGE', 'XRP', 'PEPE']
    images_dir = "temp_back"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    total_inst = len(inst_id_list)
    for inst_index, inst_id in enumerate(inst_id_list):
        print(f"\n【处理数据】：开始处理 {inst_id} ({inst_index+1}/{total_inst})")
        data_file = f'temp/corr/final_good_{inst_id}_True_filter_all.parquet_origin_good_weekly_net_profit_detail.parquet'
        data_df = pd.read_parquet(data_file)
        data_df = data_df[data_df['hold_time_mean'] < 5000]
        # data_df = data_df[data_df['kai_column'].str.contains('long', na=False)]
        if data_df.shape[0] < 50:
            print(f"【提示】：数据行数不足，跳过 {inst_id} 的处理")
            continue
        print(f"【提示】：数据加载完成，数据行数：{data_df.shape[0]}")

        # 1. 筛选原始数值特征（排除 'timestamp'、'net_profit_rate_new20' 和包含 'new' 的字段）
        all_numeric_columns = data_df.select_dtypes(include=np.number).columns.tolist()
        orig_feature_cols = [
            col for col in all_numeric_columns
            if col not in ['timestamp', 'net_profit_rate_new20'] and 'new' not in col
        ]
        print(f"【提示】：待处理的原始特征数量: {len(orig_feature_cols)}")
        orig_feature_data = {feature: data_df[feature].values for feature in orig_feature_cols}
        target_values = data_df['net_profit_rate_new20'].values
        orig_values = data_df[orig_feature_cols].to_numpy(dtype=np.float32)
        num_features = orig_values.shape[1]

        all_bin_analyses = []

        # 2. 原始特征分箱处理
        print("【提示】：开始处理原始特征并行任务...")
        original_results = process_features_parallel(
            orig_feature_cols, orig_feature_data, target_values, n_bins, max_workers=30
        )
        all_bin_analyses.extend(original_results)
        print("【原始特征】：所有原始特征处理完成")

        # 3. 单特征变换处理
        print("【提示】：开始处理单特征变换任务...")
        single_feature_names = []
        single_feature_data = {}
        single_batch_counter = 0
        single_transform_results = []
        expected_single_transform_features = 0

        for idx, col in enumerate(orig_feature_cols):
            print(f"【单特征变换】：处理原始特征 {col} ({idx+1}/{len(orig_feature_cols)})")
            a = orig_feature_data[col]
            transforms = {}

            # 对数变换：仅当所有值均大于0时
            if np.all(a > 0):
                transforms[f'{col}-log'] = np.log1p(a).astype(np.float32)
            # 平方变换
            transforms[f'{col}-square'] = (a ** 2).astype(np.float32)
            # 倒数变换（处理除 0 问题）
            transforms[f'{col}-reciprocal'] = np.where(a == 0, np.nan, 1 / a).astype(np.float32)
            # 立方变换
            transforms[f'{col}-cube'] = (a ** 3).astype(np.float32)
            # 平方根变换：仅当所有值均不为负时
            if np.all(a >= 0):
                transforms[f'{col}-sqrt'] = np.sqrt(a).astype(np.float32)
            # 双曲正切变换
            transforms[f'{col}-tanh'] = np.tanh(a).astype(np.float32)
            # 标准化 zscore： (a - mean) / std，当 std==0 时返回 NaN
            std_val = np.std(a)
            mean_val = np.mean(a)
            if std_val == 0:
                zscore = np.full_like(a, np.nan)
            else:
                zscore = ((a - mean_val) / std_val).astype(np.float32)
            transforms[f'{col}-zscore'] = zscore

            # 将每个转化后的特征添加到任务队列中
            for trans_name, trans_values in transforms.items():
                single_feature_names.append(trans_name)
                single_feature_data[trans_name] = trans_values
                expected_single_transform_features += 1
                single_batch_counter += 1

                if single_batch_counter >= batch_size:
                    print(f"【单特征变换】：提交批次任务，共 {len(single_feature_names)} 特征（已累计处理 {expected_single_transform_features} 个单特征变换）")
                    batch_results = process_features_parallel(
                        single_feature_names, single_feature_data, target_values, n_bins, max_workers=30
                    )
                    single_transform_results.extend(batch_results)
                    single_feature_names, single_feature_data = [], {}
                    single_batch_counter = 0

        # 处理最后不足 batch_size 的剩余任务
        if single_feature_names:
            print(f"【单特征变换】：提交最后剩余批次任务，共 {len(single_feature_names)} 特征")
            batch_results = process_features_parallel(
                single_feature_names, single_feature_data, target_values, n_bins, max_workers=30
            )
            single_transform_results.extend(batch_results)
        all_bin_analyses.extend(single_transform_results)
        print(f"【提示】：预计新增单特征变换数量为: {expected_single_transform_features}")
        print("【单特征变换】：所有单特征变换处理完成")

        # 4. 两两组合派生特征处理
        expected_derived_features = (num_features * (num_features - 1) // 2) * 6
        print(f"【提示】：预计通过两两组合生成的派生特征数量为: {expected_derived_features}")

        derived_feature_names = []
        derived_feature_data = {}
        derived_batch_counter = 0
        derived_results = []

        total_pairs = num_features * (num_features - 1) // 2
        pair_counter = 0

        for i in range(num_features):
            a = orig_values[:, i]
            col1 = orig_feature_cols[i]
            for j in range(i + 1, num_features):
                pair_counter += 1
                # 每处理100个组合打印一次进度
                if pair_counter % 100 == 0 or pair_counter == total_pairs:
                    print(f"【派生特征】：已处理派生特征组合 {pair_counter}/{total_pairs}")
                b = orig_values[:, j]
                col2 = orig_feature_cols[j]
                features = {
                    f'{col1}-{col2}-ratio': np.where(b == 0, np.nan, a / b).astype(np.float32),
                    f'{col1}-{col2}-diff': (a - b).astype(np.float32),
                    f'{col1}-{col2}-abs_diff': np.abs(a - b).astype(np.float32),
                    f'{col1}-{col2}-mean': ((a + b) / 2).astype(np.float32),
                    f'{col1}-{col2}-prod': np.where((a < 0) & (b < 0), - (np.abs(a) * np.abs(b)), a * b).astype(np.float32),
                    f'{col1}-{col2}-norm_diff': np.where((a + b) == 0, np.nan, (a - b) / (a + b + 1e-8)).astype(np.float32)
                }
                for feature_name, values in features.items():
                    derived_feature_names.append(feature_name)
                    derived_feature_data[feature_name] = values
                    derived_batch_counter += 1
                    if derived_batch_counter >= batch_size:
                        print(f"【派生特征】：提交批次任务，共 {len(derived_feature_names)} 派生特征")
                        batch_results = process_features_parallel(
                            derived_feature_names, derived_feature_data, target_values, n_bins, max_workers=30
                        )
                        derived_results.extend(batch_results)
                        derived_feature_names, derived_feature_data = [], {}
                        derived_batch_counter = 0

        if derived_feature_names:
            print(f"【派生特征】：提交最后剩余批次任务，共 {len(derived_feature_names)} 派生特征")
            batch_results = process_features_parallel(
                derived_feature_names, derived_feature_data, target_values, n_bins, max_workers=30
            )
            derived_results.extend(batch_results)
        all_bin_analyses.extend(derived_results)
        print("【派生特征】：所有派生特征处理完成")

        # 5. 合并所有分箱结果，并保存为 Parquet 文件
        if all_bin_analyses:
            combined_bin_analysis_df = pd.concat(all_bin_analyses, ignore_index=True)
            combined_file = os.path.join(images_dir, f"combined_bin_analysis_{inst_id}_false_new_1000.parquet")
            combined_bin_analysis_df.to_parquet(combined_file, index=False, compression='snappy')
            print(f"【提示】：合并后的 bin_analysis 已保存为 Parquet 文件：{combined_file}")
        else:
            print("【提示】：未生成任何 bin_analysis 结果。")

        print(f"【处理数据】：{inst_id} 处理完成")

    print("【主流程】：所有数据处理完成")



def group_statistics_fast(df: pd.DataFrame,
                          group_cols: list[str],
                          target_cols: list[str]) -> pd.DataFrame:
    """
    对 df 按 group_cols 分组，对 target_cols 计算 max, min, mean,
    positive_ratio 和 group_count（分组大小）。
    """
    # 1) 计算正例标志列
    for col in target_cols:
        df[col + '_pos'] = (df[col] > 0).astype('uint8')

    # 2) 构造 named aggregation dict
    agg_dict: dict[str, tuple[str, str]] = {}
    for col in target_cols:
        agg_dict[f'{col}_max']      = (col, 'max')
        agg_dict[f'{col}_min']      = (col, 'min')
        agg_dict[f'{col}_mean']     = (col, 'mean')
        agg_dict[f'{col}_pos_sum']  = (col + '_pos', 'sum')
    # group_count 用任意一个分组列的 'size'
    agg_dict['group_count'] = (group_cols[0], 'size')

    # 3) 一次性 groupby.agg
    g = df.groupby(group_cols, dropna=False).agg(**agg_dict)

    # 4) 计算正例比例，并一次性删除中间列
    for col in target_cols:
        g[f'{col}_positive_ratio'] = g[f'{col}_pos_sum'] / g['group_count']
    g.drop(columns=[f'{c}_pos_sum' for c in target_cols], inplace=True)

    # 5) 重置索引返回
    return g.reset_index()


def group_statistics_and_inst_details(df: pd.DataFrame,
                                      group_cols: List[str],
                                      target_cols: List[str]) -> pd.DataFrame:
    """
    对 df 按 group_cols 分组，对 target_cols 计算 max, min, mean,
    positive_ratio, group_count（分组大小），并收集每个 target_col 的
    {inst_id: value} 字典。

    Args:
        df: 输入的 Pandas DataFrame。
        group_cols: 用于分组的列名列表。
        target_cols: 需要计算统计信息和收集值的列名列表。
        inst_id_col: 用于创建详细字典 key 的实例 ID 列名。

    Returns:
        一个 Pandas DataFrame，包含分组键、统计结果以及每个 target_col 的
        {inst_id: value} 字典。

    Raises:
        ValueError: 如果 inst_id_col 不在 df 的列中。
    """
    inst_id_col = 'inst_id'
    if inst_id_col not in df.columns:
        raise ValueError(f"inst_id_col '{inst_id_col}' not found in DataFrame columns.")

    # 定义一个函数，该函数将应用于每个分组 (sub-DataFrame)
    def aggregate_group(group_df: pd.DataFrame) -> pd.Series:
        results = {}
        group_count = len(group_df)
        results['group_count'] = group_count

        if group_count == 0:
            # Handle empty groups if they somehow occur
            for col in target_cols:
                results[f'{col}_max'] = pd.NA
                results[f'{col}_min'] = pd.NA
                results[f'{col}_mean'] = pd.NA
                results[f'{col}_positive_ratio'] = pd.NA
                results[f'{col}_details'] = {}
            return pd.Series(results)

        # Prepare inst_id list once per group
        inst_ids = group_df[inst_id_col].tolist()

        for col in target_cols:
            # --- 计算统计指标 ---
            col_data = group_df[col]
            results[f'{col}_max'] = col_data.max()
            results[f'{col}_min'] = col_data.min()
            results[f'{col}_mean'] = col_data.mean()

            # --- 计算正例比例 ---
            try:
                # Attempt numeric conversion for comparison
                numeric_col = pd.to_numeric(col_data, errors='coerce')
                # .sum() on boolean True/False counts True as 1
                positive_sum = (numeric_col > 0).sum()
                # Calculate ratio, handle division by zero
                results[f'{col}_positive_ratio'] = positive_sum / group_count if group_count > 0 else pd.NA
            except TypeError:
                 # Handle cases where the column is fundamentally non-numeric
                 print(f"Warning: Column '{col}' could not be coerced to numeric for positive check. Positive ratio set to NA.")
                 results[f'{col}_positive_ratio'] = pd.NA


            # --- 创建 {inst_id: value} 字典 ---
            target_values = col_data.tolist()
            # Ensure lengths match (should always if coming from same group_df)
            if len(inst_ids) == len(target_values):
                 results[f'{col}_details'] = dict(zip(inst_ids, target_values))
            else:
                 # This case should ideally not happen with groupby().apply()
                 print(f"Warning: Mismatch length between inst_id and {col} in a group. Details dictionary might be incomplete.")
                 results[f'{col}_details'] = {} # Assign empty dict on error

        # 返回一个 Series，其索引将成为结果 DataFrame 的列
        return pd.Series(results)

    # 使用 groupby().apply()
    # dropna=False 保留分组键中的 NaN
    grouped = df.groupby(group_cols, dropna=False)
    # Apply the custom function to each group
    result_df = grouped.apply(aggregate_group) # include_groups=False since pandas 2.2.0 by default but explicit here

    # 重置索引以将分组键变回列
    return result_df.reset_index()

def merge_data_optimized(
    images_dir: str = "temp_back",
    inst_id_list: list[str] = ('BTC', 'ETH', 'LTC', 'XRP', 'BCH', 'EOS', 'TRX', 'ZRX', 'QTUM', 'ETC')
) -> list[pd.DataFrame]:
    """
    1) 分别读取每个 inst 的 parquet，只保留 feature/bin_seq 和目标列
    2) 按 feature 筛出 bin_seq 最大且等于 1000 的 rows
    3) 按 count 范围再筛一次
    4) 求各 inst 共有的 (feature, bin_seq) 对
    5) 合并所有 inst 的数据，并做分组统计
    6) 输出 parquet 文件，返回各 inst 筛选后的 DataFrame 列表
    """
    need_cols = [
        'feature', 'bin_seq',
        'min_net_profit', 'avg_net_profit_rate_new20',
        'pos_ratio', 'count'
    ]
    positive_ratio_threshold = 0.5
    inst_id_list = ['BTC', 'ETH', 'SOL', 'TON', 'DOGE', 'XRP', 'PEPE']
    # 1) 读入并初步筛选
    dfs: list[pd.DataFrame] = []
    for inst in inst_id_list:
        path = os.path.join(images_dir, f"combined_bin_analysis_{inst}_false_new_1000.parquet")
        df = pd.read_parquet(path, columns=need_cols)
        df['inst_id'] = inst

        # a) feature 分组取 max bin_seq
        feat_max = (
            df.groupby('feature')['bin_seq']
              .max()
              .reset_index()
              .query('bin_seq == 10')
              ['feature']
        )
        df = df[df['feature'].isin(feat_max)]
        # b) count 范围筛选
        df = df.query('3 < count < 20000')
        # df = df[df['avg_net_profit_rate_new20'] > -30]
        dfs.append(df)

    # 2) 求所有 inst 共有的 (feature, bin_seq)
    #    先取第一个，依次 merge 取 inner
    common = reduce(
        lambda left, right: pd.merge(
            left[['feature', 'bin_seq']].drop_duplicates(),
            right[['feature', 'bin_seq']].drop_duplicates(),
            on=['feature', 'bin_seq'],
            how='inner'
        ),
        dfs[1:], dfs[0]
    )

    # 3) 拼接所有 inst 数据，并只保留共有的组
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.merge(common, on=['feature', 'bin_seq'], how='inner')

    # ───────────────────────────────────────────────────────────────
    # 4) 预先针对 avg_net_profit_rate_new20 做一次“分组正例比例”过滤
    #    这样后续 heavy aggregator 只跑在符合比例 > threshold 的组上
    # 4a) 生成一列 pos_flag
    all_df['_avg_pos'] = (all_df['avg_net_profit_rate_new20'] > 0).astype('uint8')
    # 4b) groupby.transform 计算各组 size 和 pos_sum
    grp_cols = ['feature', 'bin_seq']
    all_df['_grp_size'] = all_df.groupby(grp_cols)['_avg_pos'].transform('size')
    all_df['_avg_pos_sum'] = all_df.groupby(grp_cols)['_avg_pos'].transform('sum')
    # 4c) 只保留 pos_sum/size > threshold 的整组
    mask = all_df['_avg_pos_sum'] > positive_ratio_threshold * all_df['_grp_size']
    # 拿出所有满足条件组的 key
    valid_keys = (
        all_df.loc[mask, grp_cols]
              .drop_duplicates()
    )
    # 4d) 过滤出这些组的全部行
    filtered_df = all_df.merge(valid_keys, on=grp_cols, how='inner')

    # 清理临时列
    filtered_df.drop(columns=['_avg_pos', '_grp_size', '_avg_pos_sum'],
                     inplace=True)
    # ───────────────────────────────────────────────────────────────

    # 4) 分组统计
    result = group_statistics_and_inst_details(
        filtered_df,
        group_cols=['feature', 'bin_seq'],
        target_cols=[
            'min_net_profit',
            'avg_net_profit_rate_new20',
            'pos_ratio',
            'count'
        ]
    )

    # 5) 写盘
    os.makedirs('temp', exist_ok=True)
    result.to_parquet(
        'temp/all_result_df_new_1000.parquet',
        index=False,
        compression='snappy'
    )

    return dfs


if __name__ == '__main__':
    # main(n_bins=10, batch_size=12000)
    merge_data_optimized()