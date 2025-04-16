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
         对不同阈值和相关性筛选模式（low/high）进行贪心式策略筛选，
         并计算 target_column_list 指标的均值、最大值和最小值。
      4. 附加基于 good_df 全量数据的基础统计信息，并返回结果 DataFrame。
    """
    # 构造相关性字典
    corr_dict = build_corr_dict(corr_df)

    # 筛选候选的数值型列
    candidate_columns = get_candidate_columns(good_df)

    # 预构造每个候选列和排序方向对应的排序 DataFrame，避免重复排序
    sorted_df_dict = {
        (col, order): good_df.sort_values(by=col, ascending=(order == "asc"))
        for col in candidate_columns for order in ["asc", "desc"]
    }

    # 定义阈值列表：[-40, -30, ..., 60]
    threshold_list = list(range(-40, 61, 10))
    result_frames = []

    def select_candidates(sorted_df, mode, threshold):
        """
        根据排序好的 DataFrame，从中按照相关性规则筛选策略候选项。

        参数:
          sorted_df: 按某候选列及排序方向排序好的 DataFrame
          mode: "low" 表示要求与现有所有已选策略相关性均低于阈值；
                "high" 表示要求与现有所有已选策略相关性均不低于阈值（且相关性必须存在）。
          threshold: 当前的相关性阈值。

        返回:
          经筛选后的策略行（以字典列表形式返回）。
        """
        selected_ids = []
        selected_rows = []
        local_corr = corr_dict  # 局部引用，减少全局查找

        for row in sorted_df.itertuples(index=False):
            candidate = getattr(row, 'index')

            if mode == "low":
                # 若存在任何已选策略与 candidate 的相关性 >= threshold，则跳过 candidate
                if any(
                        local_corr.get((candidate, s) if candidate < s else (s, candidate), -np.inf) >= threshold
                        for s in selected_ids
                ):
                    continue
            elif mode == "high":
                # 若存在任何已选策略与 candidate 的相关性不存在或 < threshold，则跳过 candidate
                if any(
                        local_corr.get((candidate, s) if candidate < s else (s, candidate), -np.inf) < threshold
                        for s in selected_ids
                ):
                    continue

            selected_ids.append(candidate)
            selected_rows.append(row._asdict())

        return selected_rows

    # 遍历每种候选排序、不同阈值以及相关性筛选模式，进行候选筛选与统计聚合
    for (col, order), sorted_df in sorted_df_dict.items():
        for threshold in threshold_list:
            for mode in ["low", "high"]:
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

    # 计算基于全量 good_df 数据的基础统计信息（不进行筛选）
    base_stats = {}
    for metric in target_column_list:
        base_stats[f"{metric}_mean"] = good_df[metric].mean()
        base_stats[f"{metric}_max"] = good_df[metric].max()
        base_stats[f"{metric}_min"] = good_df[metric].min()

    base_stats["sort_column"] = "base_stats"
    base_stats["threshold"] = None
    base_stats["sort_side"] = "base"
    base_stats["n_selected"] = len(good_df)
    base_stats["corr_select_mode"] = "base"

    result_frames.append(base_stats)
    final_df = pd.DataFrame(result_frames)
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
        results = pool.map(process_pair, file_path_pairs)

    # 可选：统计处理成功/失败的文件数
    success_count = sum(1 for _, status in results if status == "success")
    failed_count = sum(1 for _, status in results if status == "failed")
    print(f"处理完成，成功：{success_count} 个，失败：{failed_count} 个。")


if __name__ == '__main__':
    debug1()