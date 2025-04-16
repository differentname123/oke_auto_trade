import os
import pandas as pd
import numpy as np
from multiprocessing import Pool


def build_corr_dict(corr_df):
    """
    将 corr_df 转换为策略对相关性的字典。
    键为 (min(strategy1, strategy2), max(strategy1, strategy2))
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
        if col not in ['timestamp', 'net_profit_rate_new']
           and 'new' not in col
           and 'index' not in col
    ]


def select_and_aggregate(corr_df, good_df, target_column_list):
    """
    说明:
      1. 将 corr_df 转换为策略对相关性的字典。
      2. 从 good_df 中筛选数值型候选列（排除 timestamp、net_profit_rate_new 及包含 'new' 或 'index' 的列）。
      3. 针对每个候选列及排序方向，
         对不同阈值和相关性筛选模式（low, middle, high）进行贪心式策略筛选，
         并计算 target_column_list 指标的均值、最大值和最小值。
         其中：
           - "low": 筛选时要求与所有已选策略的相关性均低于阈值。
           - "high": 筛选时要求与所有已选策略的相关性均不低于阈值（且相关性数据必须存在）。
           - "middle": 筛选时要求与所有已选策略的相关性均处于 [阈值, 阈值+10) 区间。
      4. 附加基于 good_df 全量数据的基础统计信息，并返回结果 DataFrame。
    """
    # 构造相关性字典
    corr_dict = build_corr_dict(corr_df)

    # 筛选候选的数值型列
    candidate_columns = get_candidate_columns(good_df)

    # 预构造每个候选列及排序方向对应的排序好的 DataFrame，避免重复排序
    sorted_df_dict = {
        (col, order): good_df.sort_values(by=col, ascending=(order == "asc"))
        for col in candidate_columns for order in ["asc", "desc"]
    }

    # 定义阈值列表：[-40, -30, ..., 60]；对于 middle 模式，将使用 [阈值, 阈值+10) 区间判断
    threshold_list = list(range(-40, 61, 10))
    result_frames = []

    def select_candidates(sorted_df, mode, threshold):
        """
        根据排序好的 DataFrame，从中按照相关性规则筛选策略候选项。

        参数:
          sorted_df: 按某候选列及排序方向排序好的 DataFrame。
          mode: 筛选模式，可以为以下三种：
                "low"    —— 要求与所有已选策略的相关性均低于阈值；
                "middle" —— 要求与所有已选策略的相关性均落在 [阈值, 阈值+10) 之间；
                "high"   —— 要求与所有已选策略的相关性均不低于阈值（且相关性数据必须存在）。
          threshold: 当前的相关性阈值。

        返回:
          经筛选后的策略行，列表中每个元素都是一个字典。
        """
        selected_ids = []
        selected_rows = []
        local_corr = corr_dict  # 局部引用，减少全局查找

        for row in sorted_df.itertuples(index=False):
            candidate = getattr(row, 'index')

            # 模式 "low": 如果任一已选策略与 candidate 的相关性 >= threshold，则跳过 candidate
            if mode == "low":
                if any(
                    local_corr.get((candidate, s) if candidate < s else (s, candidate), -np.inf) >= threshold
                    for s in selected_ids
                ):
                    continue

            # 模式 "high": 如果有任一已选策略与 candidate 的相关性不存在（默认为 -np.inf）或 < threshold，则跳过 candidate
            elif mode == "high":
                if any(
                    local_corr.get((candidate, s) if candidate < s else (s, candidate), -np.inf) < threshold
                    for s in selected_ids
                ):
                    continue

            # 模式 "middle": 要求与所有已选策略的相关性都处于 [threshold, threshold+10) 区间
            elif mode == "middle":
                if any(
                    not (threshold <= local_corr.get((candidate, s) if candidate < s else (s, candidate), -np.inf) < threshold+10)
                    for s in selected_ids
                ):
                    continue

            # 通过筛选，加入已选策略
            selected_ids.append(candidate)
            selected_rows.append(row._asdict())

        return selected_rows

    # 遍历每种候选排序、不同阈值以及相关性筛选模式，进行候选筛选与统计聚合
    for (col, order), sorted_df in sorted_df_dict.items():
        for threshold in threshold_list:
            for mode in ["low", "middle", "high"]:
                candidates = select_candidates(sorted_df, mode, threshold)
                if not candidates:
                    continue

                selected_df = pd.DataFrame(candidates)
                stats = {}

                # 计算 target_column_list 中每个指标的均值、最大值和最小值
                for metric in target_column_list:
                    stats[f"{metric}_mean"] = selected_df[metric].mean()
                    stats[f"{metric}_max"] = selected_df[metric].max()
                    stats[f"{metric}_min"] = selected_df[metric].min()

                # 记录筛选时使用的参数信息
                stats["sort_column"] = col
                stats["threshold"] = threshold
                stats["sort_side"] = order
                stats["n_selected"] = len(selected_df)
                stats["corr_select_mode"] = mode

                result_frames.append(stats)

    final_df = pd.DataFrame(result_frames)

    # 计算基于全量 good_df 数据的基础统计信息（不进行筛选）
    base_stats = {}
    for metric in target_column_list:
        base_stats[f"{metric}_mean"] = good_df[metric].mean()
        base_stats[f"{metric}_max"] = good_df[metric].max()
        base_stats[f"{metric}_min"] = good_df[metric].min()

        final_df[f"{metric}_mean_diff"] = final_df[f"{metric}_mean"] - base_stats[f"{metric}_mean"]
        final_df[f"{metric}_max_diff"] = final_df[f"{metric}_max"] - base_stats[f"{metric}_max"]
        final_df[f"{metric}_min_diff"] = final_df[f"{metric}_min"] - base_stats[f"{metric}_min"]

    # 如果需要，也可以将全量统计信息单独保留
    result_frames.append(base_stats)
    return final_df


def process_pair(file_pair):
    """处理单对文件，file_pair 为 (file_path, good_file_path)"""
    file_path, good_file_path = file_pair
    try:
        base_name = os.path.basename(file_path)
        inst_id = base_name.split('_')[0]
        feature = base_name.split('feature_')[1].split('.parquet')[0]
        # 使用文件所在目录作为保存目录
        base_dir = os.path.dirname(file_path)
        output_path = os.path.join(base_dir, f"{inst_id}_{feature}_good_corr_agg.parquet")
        if os.path.exists(output_path):
            print(f"文件已存在，跳过处理：{output_path}")
            return (file_path, "skipped")

        corr_df = pd.read_parquet(file_path)
        good_df = pd.read_parquet(good_file_path)

        # 仅保留 hold_time_mean 小于 5000 的数据
        good_df = good_df[good_df['hold_time_mean'] < 5000]

        # 仅保留 corr_df 中 Row1 和 Row2 均存在于 good_df 'index' 列中的数据
        index_list = good_df['index'].unique()
        corr_df = corr_df[
            corr_df['Row1'].isin(index_list) & corr_df['Row2'].isin(index_list)
        ]

        # 计算 weekly_net_profit_detail 列中数组的长度
        lengths = good_df['weekly_net_profit_detail'].apply(
            lambda x: len(x) if isinstance(x, np.ndarray) else np.nan
        )
        print(f"[{file_path}] 最长数组长度: {lengths.max()}")
        print(f"[{file_path}] 最短数组长度: {lengths.min()}")

        # 调用外部聚合函数进行处理
        result_df = select_and_aggregate(corr_df, good_df, ['net_profit_rate_new'])
        result_df['inst_id'] = inst_id
        result_df['feature'] = feature

        result_df.to_parquet(output_path, index=False)
        print(f"文件已保存：{output_path}")
        return (file_path, "success")
    except Exception as e:
        print(f"读取文件失败：{file_path}，错误信息：{e}")
        return (file_path, "failed")


def debug1():
    """
    遍历 temp/corr 目录中的文件，对每个符合条件的文件进行读取和处理，
    并将聚合结果保存到新的 parquet 文件中。使用 10 个进程进行加速。
    """
    base_dir = 'temp/corr'
    if not os.path.exists(base_dir):
        print(f"目录不存在：{base_dir}")
        return

    # 筛选出文件名中同时包含 '_feature_' 和 'corr' 的文件，排除包含 'good' 的文件
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

    # 使用 10 个进程并发处理文件对
    with Pool(processes=30) as pool:
        results = list(pool.imap_unordered(process_pair, file_path_pairs))

    # 可选：统计处理成功/失败的文件数
    success_count = sum(1 for _, status in results if status == "success")
    failed_count = sum(1 for _, status in results if status == "failed")
    print(f"处理完成，成功：{success_count} 个，失败：{failed_count} 个。")


def group_statistics(df, group_column_list, target_column_list):
    """
    根据group_column_list对df分组，并对target_column_list中的列依次计算：
    最大值、最小值、均值、为正的比例

    参数:
    - df: 输入的DataFrame
    - group_column_list: 用于分组的列名列表
    - target_column_list: 需要统计的目标列名列表

    返回:
    - 按分组统计后生成的DataFrame
    """

    # 定义一个计算“为正的比例”的函数
    def pos_ratio(s):
        count = s.count()  # 非NA的个数
        if count > 0:
            return (s > 0).sum() / count
        else:
            return np.nan

    # 针对每个目标列定义聚合操作
    agg_funcs = {}
    for col in target_column_list:
        # 每个目标列依次计算最大值、最小值、均值和为正的比例
        agg_funcs[col] = ['max', 'min', 'mean', pos_ratio]

    # 按指定的分组列进行分组，然后聚合
    grouped = df.groupby(group_column_list).agg(agg_funcs)

    # 聚合后的列是多层索引，此处将其转换为单层索引
    # pos_ratio函数生成的列名默认是'<lambda>'或函数名，这里统一命名为"positive_ratio"
    grouped.columns = [
        f"{col}_{stat}" if stat != "pos_ratio" else f"{col}_positive_ratio"
        for col, stat in grouped.columns
    ]
    # 计算每组的个数，并合并到结果中
    group_counts = df.groupby(group_column_list).size()
    grouped["group_count"] = group_counts

    # 重置索引使group_column_list成为DataFrame的普通列
    grouped = grouped.reset_index()
    return grouped

def merger_data():
    # 遍历temp/corr下面所有包含good_corr_agg.parquet的文件
    base_dir = 'temp/corr'
    if not os.path.exists(base_dir):
        print(f"目录不存在：{base_dir}")
        return
    file_list = [
        file for file in os.listdir(base_dir)
        if 'good_corr_agg.parquet' in file
    ]
    # file_list = file_list[:10]
    print(f"找到 {len(file_list)} 个待合并文件。")
    df_list = []
    for file_name in file_list:
        file_path = os.path.join(base_dir, file_name)
        print(f"文件路径：{file_path}")
        # 读取文件
        df = pd.read_parquet(file_path)
        df_list.append(df)
    # 合并所有DataFrame
    merged_df = pd.concat(df_list, ignore_index=True)
    group_statistics_df = group_statistics(
        merged_df,
        group_column_list=['sort_column', 'threshold', 'sort_side', 'corr_select_mode'],
        target_column_list=['net_profit_rate_new_mean', 'net_profit_rate_new_min','net_profit_rate_new_mean_diff', 'net_profit_rate_new_min_diff']
    )
    print()

if __name__ == '__main__':
    debug1()