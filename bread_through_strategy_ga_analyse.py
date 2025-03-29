#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的分析方案代码示例

步骤：
1. 数据加载与预处理
2. 探索性数据分析（对每个特征分箱，并统计每个箱的平均净利润率、正向比例、样本数量及占比），
   将图像保存至 images 文件夹，同时合并所有分箱结果为一个 DataFrame
3. 建立模型（线性回归和决策树回归），用于评价各特征对目标变量的影响
4. 输出模型结果，帮助确定关键特征及其最优取值范围

注：如果你已有 data_df 数据集和 feature_column_list 列表，
    请注释掉示例数据生成部分，采用你自己的数据。
"""

import math
import os
from collections import Counter
from functools import reduce
import itertools
import concurrent.futures
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示，防止中文乱码（视系统环境而调整字体）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入机器学习模型和评估指标相关模块
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr


def merge_and_compute(df_list, merge_on=["feature", "bin_seq"], metric_cols=None):
    """
    合并一个 DataFrame 列表，并按 merge_on 键计算多个指标列的最大值、最小值和平均值。

    参数:
        df_list (list of pd.DataFrame): 每个 DataFrame 必须包含 merge_on 中指定的列和指标列。
        merge_on (list): 指定用于合并的键（默认: ["feature", "bin_seq"]）。
        metric_cols (list): 指定需要计算统计值的指标列（默认: ['avg_net_profit_rate_new', 'pos_ratio']）。

    返回:
        pd.DataFrame: 合并后的 DataFrame，包含 merge_on 列及各指标的统计列。
                      同时计算了两个打分指标: 一种使用指标最小值的乘积（score），一种使用指标均值的乘积（score1）。
    """
    if not df_list:
        return pd.DataFrame()

    if metric_cols is None:
        metric_cols = ['avg_net_profit_rate_new', 'pos_ratio', 'spearman_avg_net_profit_rate',
                       'spearman_pos_ratio', 'spearman_score']

    # 对每个 DataFrame 检查必备的列并重命名指标列
    df_renamed_list = []
    for idx, df in enumerate(df_list):
        df['spearman_score'] = df['spearman_avg_net_profit_rate'] * df['spearman_pos_ratio']

        # 检查必要列是否存在
        required_cols = merge_on + metric_cols
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame at index {idx} 缺少必要的列: {missing}")

        # 只保留必要列，复制一份数据
        sub_df = df[required_cols].copy()
        # 按指标列重命名，添加后缀 idx
        rename_dict = {col: f"{col}_{idx}" for col in metric_cols}
        sub_df.rename(columns=rename_dict, inplace=True)
        df_renamed_list.append(sub_df)

    # 使用内连接依次合并多个 DataFrame，确保只有共同的 merge_on 组合会保留
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=merge_on, how='inner'), df_renamed_list)

    # 对每个指标列分别计算最大值、最小值与均值
    for col in metric_cols:
        col_names = [f"{col}_{i}" for i in range(len(df_renamed_list))]
        merged_df[f"{col}_max"] = merged_df[col_names].max(axis=1)
        merged_df[f"{col}_min"] = merged_df[col_names].min(axis=1)
        merged_df[f"{col}_mean"] = merged_df[col_names].mean(axis=1)

    # 根据需求，可以调整或者加入额外的打分逻辑
    merged_df['score'] = merged_df['avg_net_profit_rate_new_min'] * merged_df['pos_ratio_min']
    merged_df['score1'] = merged_df['avg_net_profit_rate_new_mean'] * merged_df['pos_ratio_mean']

    # 统一整理输出列：merge_on 列 + 所有指标的统计列 + 打分列
    output_cols = merge_on.copy()
    for col in metric_cols:
        output_cols.extend([f"{col}_max", f"{col}_min", f"{col}_mean"])
    output_cols.extend(['score', 'score1'])

    return merged_df[output_cols]


def process_single_feature(feature, feat_values, target_values, n_bins=50):
    """
    处理单个特征的分箱统计，同时计算特征的单调性指标。
    利用分箱序号与该箱内平均净利润率和正向比例之间的 Spearman 相关系数来衡量单调效果。

    参数:
      feature      - 特征名称
      feat_values  - 该特征对应的 numpy 数组
      target_values- 目标指标 numpy 数组（net_profit_rate_new）
      n_bins       - 分箱的个数，默认是50
    返回:
      包含分箱统计结果和单调性指标的 DataFrame，如果特征无法分箱则返回 None
    """
    try:
        # 如果特征的唯一值数量不足 n_bins，则不做分箱分析
        if np.unique(feat_values).size < n_bins:
            return None

        # 利用 np.percentile 快速计算分位数
        quantiles = np.percentile(feat_values, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(quantiles)
        if len(bins) - 1 < 1:
            return None

        # 利用 np.searchsorted 为每个值分箱
        bin_indices = np.searchsorted(bins, feat_values, side='right') - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
        total_count = len(feat_values)

        bins_info = []
        for bin_idx in np.unique(bin_indices):
            mask = (bin_indices == bin_idx)
            count = int(np.sum(mask))
            avg_net_profit = np.mean(target_values[mask])
            min_net_profit = np.min(target_values[mask])
            max_net_profit = np.max(target_values[mask])
            pos_ratio = np.mean(target_values[mask] > 0)
            bin_seq = int(bin_idx) + 1
            bin_ratio = count / total_count
            if bin_idx < len(bins) - 1:
                interval_str = f"[{bins[bin_idx]:.4f}, {bins[bin_idx + 1]:.4f})"
            else:
                interval_str = f"[{bins[bin_idx]:.4f}, {bins[bin_idx]:.4f}]"
            bins_info.append({
                "feature": feature,
                "bin_seq": bin_seq,
                "bin_interval": interval_str,
                'min_net_profit': min_net_profit,
                'max_net_profit': max_net_profit,
                "avg_net_profit_rate_new": avg_net_profit,
                "pos_ratio": pos_ratio,
                "count": count,
                "bin_ratio": bin_ratio
            })
        result_df = pd.DataFrame(bins_info).sort_values(by="bin_seq")

        if result_df.shape[0] > 1:
            spearman_avg, _ = spearmanr(result_df["bin_seq"], result_df["avg_net_profit_rate_new"])
            spearman_pos, _ = spearmanr(result_df["bin_seq"], result_df["pos_ratio"])
        else:
            spearman_avg = np.nan
            spearman_pos = np.nan

        result_df["spearman_avg_net_profit_rate"] = spearman_avg
        result_df["spearman_pos_ratio"] = spearman_pos

        return result_df
    except Exception as e:
        print(f"处理特征 {feature} 时遇到异常: {e}")
        return None


def process_feature_batch(batch_features, batch_feature_data, target_values, n_bins=50):
    """
    处理一批特征，每批包含一定数量的特征。
    使用多进程（20进程）并行计算每个特征的分箱统计。

    参数:
      batch_features      - 批次中特征名称的列表
      batch_feature_data  - 该批次中每个特征对应的 numpy 数组字典
      target_values       - 目标指标数组
      n_bins              - 分箱的个数，默认是50
    返回:
      当前批次所有成功分箱后的 DataFrame列表
    """
    batch_results = []
    print(f"【process_feature_batch】：开始处理批次，共计 {len(batch_features)} 个特征")
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(process_single_feature, feature, batch_feature_data[feature], target_values,
                            n_bins): feature
            for feature in batch_features
        }
        for future in concurrent.futures.as_completed(futures):
            feat = futures[future]
            try:
                result = future.result()
                if result is not None:
                    batch_results.append(result)
                # print(f"【process_feature_batch】：完成 {feat} 分箱处理")
            except Exception as e:
                print(f"【process_feature_batch】：特征 {feat} 处理出错: {e}")
    # print("【process_feature_batch】：批次处理完成")
    return batch_results


def process_data_flat(data_df, target_column_list):
    """
    参数:
      data_df: 输入的 pandas DataFrame，必须包含 'feature' 列
      target_column_list: 需要进行倒序排序和统计的目标列名称列表

    功能:
      对于每个 target 列：
        1. 按 target 列倒序排序 DataFrame。
        2. 分别取排序后前 1%、2%、3% 的行（确保至少一行）。
        3. 对每个子集的 'feature' 列进行如下处理：
             - 如果包含 '-' 则拆分，并分别取拆分结果的第1和第2个元素
             - 如果不含 '-' 则直接加入原字符串
        4. 统计每个特征出现的次数
      最终返回一个扁平的 DataFrame，包含：
         - target_column：目标列名称
         - percentage：百分比标识（如 "1%"、"2%"、"3%"）
         - feature：特征名称
         - count：对应特征在该子集中的出现次数
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
                        parts = feat.split('-')
                        if len(parts) >= 2:
                            feature_list.append(parts[0])
                            feature_list.append(parts[1])
                        else:
                            feature_list.append(feat)
                    else:
                        feature_list.append(feat)
                counter = Counter(feature_list)
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


def auto_reduce_precision(df, verbose=True):
    """
    自动降低DataFrame中数值列的精度：
      - 对于浮点列，若数据范围在float16可表示范围内，则转换为float16，否则转换为float32。
      - 对于整数列，尝试使用downcast降低其位宽。

    参数:
      df: 需要降低精度的DataFrame
      verbose: 是否打印内存占用变化信息，默认为True

    返回:
      转换精度后的DataFrame副本
    """
    df_new = df.copy()
    start_mem = df_new.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df_new.columns:
        col_dtype = df_new[col].dtype
        if pd.api.types.is_numeric_dtype(col_dtype):
            if pd.api.types.is_float_dtype(col_dtype):
                col_min = df_new[col].min()
                col_max = df_new[col].max()
                if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:
                    df_new[col] = df_new[col].astype(np.float16)
                else:
                    df_new[col] = df_new[col].astype(np.float32)
            elif pd.api.types.is_integer_dtype(col_dtype):
                df_new[col] = pd.to_numeric(df_new[col], downcast='integer')

    end_mem = df_new.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        print(f"内存占用从 {start_mem:.2f} MB 降低到 {end_mem:.2f} MB，减少 {reduction:.1f}%")
    return df_new


def debug():
    inst_id_list = ['SOL']
    df_list = []
    df_list.append(pd.read_csv(f'images/combined_bin_analysis_SOL_false_short.csv').drop_duplicates(subset=["feature"]))
    for inst_id in inst_id_list:
        file_path = f'images/combined_bin_analysis_{inst_id}_false.csv'
        temp_df = pd.read_csv(file_path)
        temp_df['file'] = inst_id
        temp_df = temp_df.drop_duplicates(subset=["feature"])
        temp_df['spearman_score'] = temp_df['spearman_avg_net_profit_rate'] * temp_df['spearman_pos_ratio']
        df_list.append(temp_df)
    result_df = merge_and_compute(df_list)
    all_result_df = pd.concat(df_list, ignore_index=True)
    result_df1 = process_data_flat(result_df, ['avg_net_profit_rate_new_max', 'avg_net_profit_rate_new_min',
                                               'avg_net_profit_rate_new_mean', 'pos_ratio_max', 'pos_ratio_min',
                                               'pos_ratio_mean', 'score', 'score1'])
    data_df = result_df[(result_df['bin_seq'] > 90) | (result_df['bin_seq'] < 10)]
    print(result_df)


def main(n_bins=50, batch_size=10):
    """
    主流程：加载数据 -> 处理原始特征分箱 -> 计算派生特征分箱（分批计算以降低内存占用）-> 合并结果保存
    参数:
        n_bins: 分箱个数
        batch_size: 每一批次的特征数量（原始或者派生特征均适用）
    """
    print("【主流程】：开始处理数据")
    inst_id_list = ['SOL']
    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    for inst_id in inst_id_list:
        print(f"\n【处理数据】：开始处理 {inst_id}")
        data_file = f'temp/final_good_{inst_id}_false.csv'
        data_df = pd.read_csv(data_file)
        data_df = auto_reduce_precision(data_df)

        # 获取所有数值型特征，排除 'timestamp', 'net_profit_rate_new' 以及包含 'new' 的特征
        all_numeric_columns = data_df.select_dtypes(include=np.number).columns.tolist()
        orig_feature_cols = [col for col in all_numeric_columns
                             if col not in ['timestamp', 'net_profit_rate_new'] and 'new' not in col]

        # 将原始特征转换为 numpy 数组（float32）
        orig_values = data_df[orig_feature_cols].to_numpy(dtype=np.float32)
        target_values = data_df['net_profit_rate_new'].values
        num_features = orig_values.shape[1]

        all_bin_analyses = []

        # --- 原始特征逐批处理 ---
        print("【提示】：开始处理原始特征批次...")
        orig_feature_data = {feature: data_df[feature].values for feature in orig_feature_cols}
        # 利用多进程（批次数量通常较少，可以并行加速）
        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            orig_futures = []
            orig_batch_ids = []
            for i in range(0, len(orig_feature_cols), batch_size):
                batch_features = orig_feature_cols[i:i + batch_size]
                batch_feature_data = {feature: orig_feature_data[feature] for feature in batch_features}
                batch_id = i // batch_size + 1
                print(f"【原始批次】：提交批次 {batch_id}，处理特征数：{len(batch_features)}")
                orig_futures.append(
                    executor.submit(process_feature_batch, batch_features, batch_feature_data, target_values, n_bins))
                orig_batch_ids.append(batch_id)
            batch_no = 0
            for future in concurrent.futures.as_completed(orig_futures):
                batch_no += 1
                batch_results = future.result()
                if batch_results:
                    all_bin_analyses.extend(batch_results)
                print(f"【原始批次】：已完成 {batch_no} / {len(orig_futures)} 批次")
        print("【原始批次】：所有原始特征批次完成")

        # --- 派生（交叉）特征逐批处理 ---
        print("【提示】：开始处理派生特征批次...")
        derived_batch_features = []
        derived_batch_data = {}
        derived_total_count = 0  # 记录累计的派生特征数
        derived_batch_count = 0  # 记录已提交的派生批次数
        for i in range(num_features):
            a = orig_values[:, i]
            col1 = orig_feature_cols[i]
            for j in range(i + 1, num_features):
                b = orig_values[:, j]
                col2 = orig_feature_cols[j]

                # 交叉特征1：和
                feature_name = f'{col1}-{col2}-sum'
                derived_batch_features.append(feature_name)
                derived_batch_data[feature_name] = (a + b).astype(np.float32)

                # 交叉特征2：差值
                feature_name = f'{col1}-{col2}-diff'
                derived_batch_features.append(feature_name)
                derived_batch_data[feature_name] = (a - b).astype(np.float32)

                # 交叉特征3：乘积（考虑负数情况）
                feature_name = f'{col1}-{col2}-prod'
                prod_val = np.where((a < 0) & (b < 0), - (np.abs(a) * np.abs(b)), a * b)
                derived_batch_features.append(feature_name)
                derived_batch_data[feature_name] = prod_val.astype(np.float32)

                # 交叉特征4：比值（考虑负数情况）
                feature_name = f'{col1}-{col2}-ratio'
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where(b == 0, np.nan, a / b)
                    ratio = np.where((a < 0) & (b < 0), -np.abs(ratio), ratio)
                derived_batch_features.append(feature_name)
                derived_batch_data[feature_name] = ratio.astype(np.float32)

                derived_total_count += 4

                # 当当前批次派生特征数达到或超过 batch_size 时，立即处理当前批次
                if len(derived_batch_features) >= batch_size:
                    derived_batch_count += 1
                    print(f"【派生批次】：提交批次 {derived_batch_count}，当前派生特征数：{len(derived_batch_features)}")
                    batch_result = process_feature_batch(derived_batch_features, derived_batch_data, target_values,
                                                         n_bins)
                    if batch_result:
                        all_bin_analyses.extend(batch_result)
                    print(f"【派生批次】：完成批次 {derived_batch_count}")
                    # 清空当前批次数据
                    derived_batch_features = []
                    derived_batch_data = {}

        # 处理剩余未满一批的派生特征
        if derived_batch_features:
            derived_batch_count += 1
            print(f"【派生批次】：提交剩余批次 {derived_batch_count}，当前派生特征数：{len(derived_batch_features)}")
            batch_result = process_feature_batch(derived_batch_features, derived_batch_data, target_values, n_bins)
            if batch_result:
                all_bin_analyses.extend(batch_result)
            print(f"【派生批次】：完成剩余批次 {derived_batch_count}")
            derived_batch_features = []
            derived_batch_data = {}

        print(f"{inst_id}【提示】：原始批次与派生批次全部处理完成，共计派生特征数: {derived_total_count}")

        # 合并所有结果并保存
        if all_bin_analyses:
            combined_bin_analysis_df = pd.concat(all_bin_analyses, ignore_index=True)
            combined_csv_file = os.path.join(images_dir, f"combined_bin_analysis_{inst_id}_false.csv")
            combined_bin_analysis_df.to_csv(combined_csv_file, index=False)
            print(f"【提示】：合并后的 bin_analysis 已保存为 CSV 文件：{combined_csv_file}")
        else:
            print("【提示】：未生成任何 bin_analysis 结果。")
    print("【主流程】：所有数据处理完成")


if __name__ == '__main__':
    main(n_bins=100, batch_size=100)