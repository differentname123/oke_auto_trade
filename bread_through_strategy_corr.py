import os
import time
import pandas as pd
import numpy as np
from multiprocessing import Pool

# 常量定义
THRESHOLD_LIST = list(range(-40, 61, 10))
VALID_MODES = ["low", "middle", "high"]
EXCLUDE_COLUMNS = ['timestamp', 'net_profit_rate_new']


def build_corr_dict(corr_df):
    """
    将相关性 DataFrame 转换为字典：
      键为 (min(strategy1, strategy2), max(strategy1, strategy2))
      值为对应的相关性
    """
    keys = list(zip(
        np.minimum(corr_df['Row1'], corr_df['Row2']),
        np.maximum(corr_df['Row1'], corr_df['Row2'])
    ))
    return dict(zip(keys, corr_df['Correlation'].values))


def get_candidate_columns(good_df):
    """
    从 good_df 中筛选出所有数值型候选列，
    排除 'timestamp'、'net_profit_rate_new' 以及包含 'new' 或 'index' 的列。
    """
    numeric_cols = good_df.select_dtypes(include=np.number).columns.tolist()
    return [
        col for col in numeric_cols
        if col not in EXCLUDE_COLUMNS
           and 'new' not in col
           and 'index' not in col
    ]


def create_sorted_df_dict(good_df, candidate_columns):
    """
    针对每个候选列，生成按该列升序和降序排序后的 DataFrame 字典，
    键为 (列名, 排序方式) ，排序方式 "asc" 为升序，"desc" 为降序。
    """
    return {
        (col, order): good_df.sort_values(by=col, ascending=(order == "asc"))
        for col in candidate_columns for order in ["asc", "desc"]
    }


def is_valid_candidate(candidate, selected_ids, mode, threshold, corr_dict):
    """
    判断候选策略 candidate 是否满足与所有已选策略的相关性要求。

    参数:
      candidate: 当前候选策略，通常是其 id
      selected_ids: 已经选中的策略 id 列表
      mode: 筛选模式 ("low", "middle", "high")
      threshold: 当前相关性阈值
      corr_dict: 策略对相关性字典

    返回:
      如果 candidate 满足条件返回 True，否则返回 False。
    """
    for selected in selected_ids:
        key = (candidate, selected) if candidate < selected else (selected, candidate)
        corr_value = corr_dict.get(key, -np.inf)
        if mode == "low" and corr_value >= threshold:
            return False
        elif mode == "high" and corr_value < threshold:
            return False
        elif mode == "middle" and not (threshold <= corr_value < threshold + 10):
            return False
    return True


def select_valid_candidates_from_sorted(sorted_df, mode, threshold, corr_dict):
    """
    根据已经排序的 DataFrame 从中筛选出满足相关性规则的候选策略行。

    参数:
      sorted_df: 按特定候选列与排序方向排序后的 DataFrame
      mode: 筛选模式 ("low", "middle", "high")
      threshold: 当前相关性阈值
      corr_dict: 策略对相关性字典

    返回:
      满足条件的策略行（字典形式的列表）。
    """
    selected_ids = []
    selected_rows = []
    for row in sorted_df.itertuples(index=False):
        # 这里假设 good_df 中存在名为 'index' 的列作为策略标识
        candidate_id = getattr(row, 'index')
        if is_valid_candidate(candidate_id, selected_ids, mode, threshold, corr_dict):
            selected_ids.append(candidate_id)
            selected_rows.append(row._asdict())
    return selected_rows


def select_and_aggregate(corr_df, good_df, target_column_list):
    """
    处理流程：
      1. 构造相关性字典。
      2. 筛选数值型候选列（排除 timestamp、net_profit_rate_new 及包含 'new' 或 'index' 的列）。
      3. 针对每个候选列及排序方向，
         对不同阈值和相关性筛选模式（low, middle, high）进行贪心式策略筛选，
         并计算 target_column_list 指标的均值、最大值和最小值。
      4. 计算全量 good_df 数据的基础统计信息，进而计算差异值。
    """
    corr_dict = build_corr_dict(corr_df)
    candidate_columns = get_candidate_columns(good_df)
    sorted_df_dict = create_sorted_df_dict(good_df, candidate_columns)
    results = []

    # 遍历每个候选列及其排序（升序、降序）
    for (col, order), sorted_df in sorted_df_dict.items():
        for threshold in THRESHOLD_LIST:
            for mode in VALID_MODES:
                # 根据排序后的 DataFrame 筛选候选策略
                candidates = select_valid_candidates_from_sorted(sorted_df, mode, threshold, corr_dict)
                if not candidates:
                    continue

                selected_df = pd.DataFrame(candidates)
                stats = {}

                # 对每个目标指标计算均值、最大值、最小值
                for metric in target_column_list:
                    stats[f"{metric}_mean"] = selected_df[metric].mean()
                    stats[f"{metric}_max"] = selected_df[metric].max()
                    stats[f"{metric}_min"] = selected_df[metric].min()

                # 记录筛选参数信息
                stats["sort_column"] = col
                stats["threshold"] = threshold
                stats["sort_side"] = order
                stats["n_selected"] = len(selected_df)
                stats["corr_select_mode"] = mode

                results.append(stats)

    final_df = pd.DataFrame(results)

    # 计算全量 good_df 数据的基础统计信息，并添加差异列
    base_stats = {}
    for metric in target_column_list:
        base_stats[f"{metric}_mean"] = good_df[metric].mean()
        base_stats[f"{metric}_max"] = good_df[metric].max()
        base_stats[f"{metric}_min"] = good_df[metric].min()

        final_df[f"{metric}_mean_diff"] = final_df[f"{metric}_mean"] - base_stats[f"{metric}_mean"]
        final_df[f"{metric}_max_diff"] = final_df[f"{metric}_max"] - base_stats[f"{metric}_max"]
        final_df[f"{metric}_min_diff"] = final_df[f"{metric}_min"] - base_stats[f"{metric}_min"]

    return final_df


def process_pair(file_pair):
    """
    处理单对文件，file_pair 为 (corr_file_path, good_file_path)

    流程：
      1. 根据文件名解析获取机构ID与特征信息，并生成输出文件路径。
      2. 跳过已存在的输出文件。
      3. 读取 parquet 文件，过滤不符合条件的数据。
      4. 调用 select_and_aggregate 进行候选策略筛选与统计聚合。
      5. 保存结果并返回处理状态。
    """
    file_path, good_file_path = file_pair
    start_time = time.time()
    try:
        base_name = os.path.basename(file_path)
        inst_id = base_name.split('_')[0]
        feature = base_name.split('feature_')[1].split('.parquet')[0]
        base_dir = os.path.dirname(file_path)
        output_path = os.path.join(base_dir, f"{inst_id}_{feature}_good_corr_agg.parquet")
        if os.path.exists(output_path):
            print(f"文件已存在，跳过处理：{output_path}")
            return (file_path, "skipped")

        corr_df = pd.read_parquet(file_path)
        good_df = pd.read_parquet(good_file_path)

        # 过滤 hold_time_mean 超过阈值的记录
        good_df = good_df[good_df['hold_time_mean'] < 5000]

        # 保留 corr_df 中 Row1 和 Row2 均存在于 good_df 的 'index' 列中的数据
        index_list = good_df['index'].unique()
        corr_df = corr_df[
            corr_df['Row1'].isin(index_list) & corr_df['Row2'].isin(index_list)
        ]

        result_df = select_and_aggregate(corr_df, good_df, ['net_profit_rate_new'])
        result_df['inst_id'] = inst_id
        result_df['feature'] = feature

        result_df.to_parquet(output_path, index=False)
        elapsed_time = time.time() - start_time
        print(f"文件已保存：{output_path} ，耗时：{elapsed_time:.2f}秒")
        return (file_path, "success")
    except Exception as e:
        print(f"读取文件失败：{file_path}，错误信息：{e}")
        return (file_path, "failed")


def debug1():
    """
    遍历 temp/corr 目录下所有符合条件的文件，
    对每个 (相关性文件, 候选数据文件) 文件对进行处理，
    并保存聚合结果至新的 parquet 文件中。
    使用 30 个进程并发处理文件对。
    """
    base_dir = 'temp/corr'
    if not os.path.exists(base_dir):
        print(f"目录不存在：{base_dir}")
        return

    # 筛选出文件名中同时包含 '_feature_' 和 'corr' 且不包含 'good' 的文件
    file_list = [
        file for file in os.listdir(base_dir)
        if '_feature_' in file and 'corr' in file and 'good' not in file
    ]

    file_path_pairs = []
    for file_name in file_list:
        good_file_name = file_name.replace('corr', 'origin_good')
        file_path = os.path.join(base_dir, file_name)
        good_file_path = os.path.join(base_dir, good_file_name)
        if os.path.exists(good_file_path):
            file_path_pairs.append((file_path, good_file_path))
        else:
            print(f"文件不存在：{good_file_path}")

    print(f"共找到 {len(file_path_pairs)} 个待处理文件对。")
    if not file_path_pairs:
        return

    with Pool(processes=30) as pool:
        results = list(pool.imap_unordered(process_pair, file_path_pairs))

    # 统计处理成功与失败的文件数量
    success_count = sum(1 for _, status in results if status == "success")
    failed_count = sum(1 for _, status in results if status == "failed")
    print(f"处理完成，成功：{success_count} 个，失败：{failed_count} 个。")


def group_statistics(df, group_column_list, target_column_list):
    """
    根据 group_column_list 对 DataFrame 进行分组，并对 target_column_list 中的每个列依次计算：
      - 最大值、最小值、均值、以及正值比例
    参数:
      df: 输入的 DataFrame
      group_column_list: 用于分组的列名列表
      target_column_list: 待统计的目标列名列表
    返回:
      分组聚合后的 DataFrame
    """
    def pos_ratio(s):
        count = s.count()  # 非 NA 数的个数
        return (s > 0).sum() / count if count > 0 else np.nan

    agg_funcs = {}
    for col in target_column_list:
        agg_funcs[col] = ['max', 'min', 'mean', pos_ratio]

    grouped = df.groupby(group_column_list).agg(agg_funcs)

    # 将多层列索引扁平化，并将 pos_ratio 列命名为 positive_ratio
    grouped.columns = [
        f"{col}_{stat}" if stat != "pos_ratio" else f"{col}_positive_ratio"
        for col, stat in grouped.columns
    ]

    # 添加每组数据的数量
    group_counts = df.groupby(group_column_list).size()
    grouped["group_count"] = group_counts
    return grouped.reset_index()


def get_good_file():
    """
    读取并筛选 'temp/all_result_df.parquet' 文件中的数据，
    保留满足条件的记录并对 feature 去重。
    """
    good_feature_df = pd.read_parquet('temp/all_result_df.parquet')
    sort_key = 'avg_net_profit_rate_new_min'

    good_feature_df = good_feature_df[
        good_feature_df['bin_seq'].isin([1, 1000]) &
        (good_feature_df[sort_key] > 0) &
        (good_feature_df['count_min'] > 10000)
    ]

    # 仅保留 spearman_pos_ratio_min 与 spearman_pos_ratio_max 同号的记录
    condition = (
        ((good_feature_df['spearman_pos_ratio_min'] > 0) & (good_feature_df['spearman_pos_ratio_max'] > 0)) |
        ((good_feature_df['spearman_pos_ratio_min'] < 0) & (good_feature_df['spearman_pos_ratio_max'] < 0))
    )
    good_feature_df = good_feature_df[condition]

    good_feature_df = good_feature_df.sort_values(by=sort_key, ascending=False)
    good_feature_df = good_feature_df.drop_duplicates(subset=['feature'], keep='first')
    print(f"good_feature_df的行数为：{len(good_feature_df)}")
    return good_feature_df


def merger_data():
    """
    合并 temp/corr 目录下所有包含 'good_corr_agg.parquet' 的文件，
    并对合并后的 DataFrame 按指定分组字段计算聚合统计信息。
    """
    # 如果需要根据 good_feature_df 进行筛选，可取消注释以下代码
    # good_feature_df = get_good_file()
    # good_feature_df['feature_bin_seq'] = good_feature_df['feature'].astype(str) + '_' + good_feature_df['bin_seq'].astype(str)
    # feature_bin_seq_list = good_feature_df['feature_bin_seq'].tolist()
    feature_bin_seq_list = []

    base_dir = 'temp/corr'
    if not os.path.exists(base_dir):
        print(f"目录不存在：{base_dir}")
        return

    file_list = [
        file for file in os.listdir(base_dir)
        if 'good_corr_agg.parquet' in file
    ]
    if feature_bin_seq_list:
        file_list = [file for file in file_list if any(fbs in file for fbs in feature_bin_seq_list)]

    print(f"找到 {len(file_list)} 个待合并文件。")
    df_list = []
    for file_name in file_list:
        file_path = os.path.join(base_dir, file_name)
        df = pd.read_parquet(file_path)
        df_list.append(df)
    merged_df = pd.concat(df_list, ignore_index=True)

    group_stats_df = group_statistics(
        merged_df,
        group_column_list=['sort_column', 'sort_side', 'corr_select_mode', 'threshold'],
        target_column_list=['net_profit_rate_new_mean', 'net_profit_rate_new_min', 'net_profit_rate_new_mean_diff', 'net_profit_rate_new_min_diff']
    )
    print("合并与分组统计完成。")
    # 可根据需求保存或进一步处理 group_stats_df


if __name__ == '__main__':
    debug1()