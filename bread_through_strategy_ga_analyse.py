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
            'avg_net_profit_rate_new', 'pos_ratio',
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
    merged_df['score'] = merged_df['avg_net_profit_rate_new_min'] * merged_df['pos_ratio_min']
    merged_df['score1'] = merged_df['avg_net_profit_rate_new_mean'] * merged_df['pos_ratio_mean']

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
            "avg_net_profit_rate_new": avg_net_profit,
            "pos_ratio": pos_ratio,
            "count": counts,
            "bin_ratio": counts / total_count
        })

        if result_df.shape[0] > 1:
            spearman_avg, _ = spearmanr(result_df["bin_seq"], result_df["avg_net_profit_rate_new"])
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
    inst_id_list = ['SOL', 'TON', 'DOGE', 'XRP', 'PEPE']
    images_dir = "temp_back"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    total_inst = len(inst_id_list)
    for inst_index, inst_id in enumerate(inst_id_list):
        print(f"\n【处理数据】：开始处理 {inst_id} ({inst_index+1}/{total_inst})")
        data_file = f'temp/final_good_{inst_id}_false.parquet'
        data_df = pd.read_parquet(data_file)
        data_df = data_df[data_df['hold_time_mean'] < 5000]
        data_df = data_df[data_df['kai_column'].str.contains('long', na=False)]
        print(f"【提示】：数据加载完成，数据行数：{data_df.shape[0]}")

        # 1. 筛选原始数值特征（排除 'timestamp'、'net_profit_rate_new' 和包含 'new' 的字段）
        all_numeric_columns = data_df.select_dtypes(include=np.number).columns.tolist()
        orig_feature_cols = [
            col for col in all_numeric_columns
            if col not in ['timestamp', 'net_profit_rate_new'] and 'new' not in col
        ]
        print(f"【提示】：待处理的原始特征数量: {len(orig_feature_cols)}")
        orig_feature_data = {feature: data_df[feature].values for feature in orig_feature_cols}
        target_values = data_df['net_profit_rate_new'].values
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
            combined_file = os.path.join(images_dir, f"combined_bin_analysis_{inst_id}_false.parquet")
            combined_bin_analysis_df.to_parquet(combined_file, index=False, compression='snappy')
            print(f"【提示】：合并后的 bin_analysis 已保存为 Parquet 文件：{combined_file}")
        else:
            print("【提示】：未生成任何 bin_analysis 结果。")

        print(f"【处理数据】：{inst_id} 处理完成")

    print("【主流程】：所有数据处理完成")


if __name__ == '__main__':
    main(n_bins=1000, batch_size=100)