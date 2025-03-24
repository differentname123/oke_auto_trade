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
import pandas as pd
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
        metric_cols = ['avg_net_profit_rate_new', 'pos_ratio', 'spearman_avg_net_profit_rate', 'spearman_pos_ratio']

    # 对每个 DataFrame 检查必备的列并重命名指标列
    df_renamed_list = []
    for idx, df in enumerate(df_list):
        # 检查必要列是否存在
        required_cols = merge_on + metric_cols
        missing = set(required_cols) - set(df.columns)
        # # 将metric_cols的值都取绝对值
        # for col in metric_cols:
        #     df[col] = df[col].abs()

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


def process_single_feature(feature, feat_values, target_values):
    """
    处理单个特征的分箱统计，同时计算特征的单调性指标。
    利用分箱序号与该箱内平均净利润率和正向比例之间的 Spearman 相关系数来衡量单调效果。

    参数:
      feature      - 特征名称
      feat_values  - 该特征对应的 numpy 数组
      target_values- 目标指标 numpy 数组（net_profit_rate_new）
    返回:
      包含分箱统计结果和单调性指标的 DataFrame，如果特征无法分箱则返回 None
    """
    try:
        # 如果特征的唯一值数量不足 100，则不做分箱分析
        if np.unique(feat_values).size < 100:
            # print(f"【提示】：特征 {feature} 的唯一值不足 100（当前 {np.unique(feat_values).size} 个），跳过。")
            return None

        # 利用 np.percentile 快速计算分位数（分为 101 个点，即 100 箱）
        quantiles = np.percentile(feat_values, np.linspace(0, 100, 101))
        # 去重后可能导致箱的数量减少
        bins = np.unique(quantiles)
        if len(bins) - 1 < 1:
            # print(f"【提示】：特征 {feature} 分箱数量不够，跳过。")
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
                "avg_net_profit_rate_new": avg_net_profit,
                "pos_ratio": pos_ratio,
                "count": count,
                "bin_ratio": bin_ratio
            })
        result_df = pd.DataFrame(bins_info).sort_values(by="bin_seq")

        # 计算分箱序号与各指标间的 Spearman 相关系数作为单调性指标
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
        # print(f"【提示】：特征 {feature} 分箱处理出错，错误信息：{e}")
        return None


def process_feature_batch(batch_features, batch_feature_data, target_values):
    """
    处理一批特征，每批包含 10 个特征。
    参数:
      batch_features      - 批次中特征名称的列表
      batch_feature_data  - 该批次中每个特征对应的 numpy 数组字典
      target_values       - 目标指标数组
    返回:
      当前批次所有成功分箱后的 DataFrame列表
    """
    batch_results = []
    for feature in batch_features:
        result = process_single_feature(feature, batch_feature_data[feature], target_values)
        if result is not None:
            batch_results.append(result)
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
    # 只保留data_df中bin_seq大于90或者小于10的数据
    data_df = data_df[(data_df['bin_seq'] > 90) | (data_df['bin_seq'] < 10)]

    output_rows = []
    percentage_list = [x / 100 for x in range(1, 11)]

    for target in target_column_list:
        # 如果目标列不存在，则跳过
        if target not in data_df.columns:
            print(f"列 {target} 不存在于 DataFrame 中，跳过。")
            continue

        # 按目标列倒序排序
        sorted_df = data_df.sort_values(by=target, ascending=False).reset_index(drop=True)
        total_rows = sorted_df.shape[0]

        for target in target_column_list:
            # 如果目标列不存在，则跳过
            if target not in data_df.columns:
                print(f"列 {target} 不存在于 DataFrame 中，跳过。")
                continue

            # 按目标列倒序排序
            sorted_df = data_df.sort_values(by=target, ascending=False).reset_index(drop=True)
            total_rows = sorted_df.shape[0]

            # 分别处理前1%、2%、3%
            for perc in percentage_list:
                # 计算行数，至少取一行
                n = max(1, int(math.ceil(total_rows * perc)))
                subset_df = sorted_df.head(n)

                # 记录截止值，即该子集 target 列的最后一行值
                threshold_value = sorted_df[target].iloc[n - 1]

                feature_list = []
                # 遍历子集中的每个 'feature' 值处理特征
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

                # 统计出现次数
                counter = Counter(feature_list)
                # 将计数结果存入列表，每个元素一行记录
                for feature, count in counter.items():
                    output_rows.append({
                        'target_column': target,
                        'percentage': f"{int(perc * 100)}",
                        '临界值': threshold_value,
                        'feature': feature,
                        'count': count
                    })

        # 转换为 DataFrame，并按 target_column, percentage 和 count 降序排列（方便分析）
        result_df = pd.DataFrame(output_rows)
        result_df = result_df.sort_values(
            by=['target_column', 'percentage', 'count'],
            ascending=[True, True, False]
        ).reset_index(drop=True)

        return result_df


def debug():
    inst_id_list = ['BTC', 'ETH', 'SOL', 'TON', 'DOGE', 'XRP', 'PEPE']
    df_list = []
    for inst_id in inst_id_list:
        file_path = f'images/combined_bin_analysis_{inst_id}.csv'
        temp_df = pd.read_csv(file_path)
        temp_df = temp_df.drop_duplicates(subset=["feature"])
        df_list.append(temp_df)
    result_df = merge_and_compute(df_list)
    result_df1 = process_data_flat(result_df, ['avg_net_profit_rate_new_max', 'avg_net_profit_rate_new_min',
                                               'avg_net_profit_rate_new_mean', 'pos_ratio_max', 'pos_ratio_min',
                                               'pos_ratio_mean', 'score', 'score1'])
    data_df = result_df[(result_df['bin_seq'] > 90) | (result_df['bin_seq'] < 10)]
    print(result_df)


def main():
    # debug()
    # ===== 数据加载与预处理 =====
    inst_id_list = ['BTC', 'ETH', 'SOL', 'TON', 'DOGE', 'XRP', 'PEPE']
    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    for inst_id in inst_id_list:
        print(f"\n【处理数据】：{inst_id}")
        data_file = f'temp/final_good_{inst_id}.csv'
        data_df = pd.read_csv(data_file)
        # data_df = data_df[(data_df['net_profit_rate'] > 100)]

        # 获取所有数值型特征，排除 'timestamp'、'net_profit_rate_new' 和包含 'new' 的特征
        all_numeric_columns = data_df.select_dtypes(include=np.number).columns.tolist()
        orig_feature_cols = [col for col in all_numeric_columns
                             if col not in ['timestamp', 'net_profit_rate_new'] and 'new' not in col]

        # -------------------------------
        # Step 2：边计算边分批合并派生特征（交叉特征）
        # 将原始特征数据转换为 numpy 数组，确保为 float32
        orig_values = data_df[orig_feature_cols].to_numpy(dtype=np.float32)
        num_features = orig_values.shape[1]

        # 记录所有新加入的特征名称，便于后续使用
        derived_feature_names = []

        for i in range(num_features):
            a = orig_values[:, i]
            col1 = orig_feature_cols[i]
            for j in range(i + 1, num_features):
                b = orig_values[:, j]
                col2 = orig_feature_cols[j]

                # 交叉特征1：和
                col_name = f'{col1}-{col2}-sum'
                data_df[col_name] = a + b
                derived_feature_names.append(col_name)

                # 交叉特征2：乘积
                col_name = f'{col1}-{col2}-prod'
                data_df[col_name] = a * b
                derived_feature_names.append(col_name)

                # 交叉特征3：差值
                col_name = f'{col1}-{col2}-diff'
                data_df[col_name] = a - b
                derived_feature_names.append(col_name)

                # 交叉特征4：比值（使用 np.errstate 屏蔽除 0 警告）
                col_name = f'{col1}-{col2}-ratio'
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where(b == 0, np.nan, a / b)
                data_df[col_name] = ratio
                derived_feature_names.append(col_name)

        # 更新完整的特征列表：原始特征 + 派生特征
        all_feature_cols = orig_feature_cols + derived_feature_names
        print(f"{inst_id}【提示】：已添加派生特征，总特征数为 {len(all_feature_cols)}")

        # ===== 准备多进程处理的数据 =====
        # 提取目标数组（net_profit_rate_new）
        target_values = data_df['net_profit_rate_new'].values

        # 预先提取所有特征的 numpy 数组，形成字典，避免每个子进程传送整个 DataFrame
        feature_data = {feature: data_df[feature].values for feature in all_feature_cols}

        # 将所有特征按每 10 个特征分为一批
        batches = []
        for i in range(0, len(all_feature_cols), 10):
            batch_features = all_feature_cols[i:i + 10]
            batch_feature_data = {feature: feature_data[feature] for feature in batch_features}
            batches.append((batch_features, batch_feature_data))

        all_bin_analyses = []
        # 多进程并发（最多20个进程）
        with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(process_feature_batch, batch_features, batch_feature_data, target_values)
                for (batch_features, batch_feature_data) in batches
            ]
            for future in concurrent.futures.as_completed(futures):
                batch_results = future.result()
                if batch_results:
                    all_bin_analyses.extend(batch_results)

        # ===== 合并结果并保存 =====
        if all_bin_analyses:
            combined_bin_analysis_df = pd.concat(all_bin_analyses, ignore_index=True)
            combined_csv_file = os.path.join(images_dir, f"combined_bin_analysis_{inst_id}.csv")
            combined_bin_analysis_df.to_csv(combined_csv_file, index=False)
            print(f"【提示】：合并后的 bin_analysis 已保存为 CSV 文件：{combined_csv_file}")


if __name__ == '__main__':
    main()