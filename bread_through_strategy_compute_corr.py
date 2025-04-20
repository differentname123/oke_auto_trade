import os
import time
import itertools
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import spearmanr

# 全局变量，用于在多进程中保存解析后的行数据
GLOBAL_PARSED_ROWS = None


def compute_robust_correlation(array1, array2):
    """
    计算稳健相关性，使用 Pearson 和 Spearman 相关系数的均值。
    如果数据长度不足或标准差为 0，则返回 0。
    """
    n_common = min(len(array1), len(array2))
    if n_common < 3:
        return 0
    x = array1[:n_common]
    y = array2[:n_common]
    std_x = np.std(x)
    std_y = np.std(y)
    if std_x == 0 or std_y == 0:
        return 0
    # 计算 Pearson 相关系数
    pearson_corr = np.corrcoef(x, y)[0, 1]
    # 计算 Spearman 相关系数。如果结果为 nan 则置为 0。
    spearman_corr, _ = spearmanr(x, y)
    if np.isnan(spearman_corr):
        spearman_corr = 0
    return (pearson_corr + spearman_corr) / 2


def calculate_row_correlation(row1, row2):
    """
    根据字段 'monthly_net_profit_detail' 计算两行数据的相关性，
    最终返回相关系数乘以 100 后取整的结果。
    """
    target_field = "monthly_net_profit_detail"
    detail1 = row1.get(target_field, np.array([]))
    detail2 = row2.get(target_field, np.array([]))
    if not isinstance(detail1, np.ndarray):
        detail1 = np.array(detail1)
    if not isinstance(detail2, np.ndarray):
        detail2 = np.array(detail2)
    corr = compute_robust_correlation(detail1, detail2)
    return int(round(corr * 100))


def init_worker(rows):
    """
    每个 worker 进程初始化时调用，将解析后的行数据保存到全局变量中。
    """
    global GLOBAL_PARSED_ROWS
    GLOBAL_PARSED_ROWS = rows


def process_pair(pair):
    """
    处理单个行对任务，直接计算两行之间的相关性并返回关键信息字典。
    """
    i, j = pair
    row_a = GLOBAL_PARSED_ROWS[i]
    row_b = GLOBAL_PARSED_ROWS[j]
    corr_val = calculate_row_correlation(row_a, row_b)
    return {
        "Row1": row_a['index'],
        "Row2": row_b['index'],
        "Correlation": corr_val,
        "Row1_net_profit_rate": row_a.get("net_profit_rate"),
        "Row2_net_profit_rate": row_b.get("net_profit_rate"),
    }


def process_group(group_df, sort_key, group_threshold):
    """
    对一个分组的数据先按 sort_key 降序排序，然后遍历比较每一行与已保留行的相关性，
    若相关性大于 group_threshold 则舍弃当前行，最终返回过滤后的 DataFrame。
    """
    group_sorted = group_df.sort_values(by=sort_key, ascending=False)
    keep_rows = []
    for _, row in group_sorted.iterrows():
        drop_flag = False
        for kept_row in keep_rows:
            if calculate_row_correlation(row, kept_row) > group_threshold:
                drop_flag = True
                break
        if not drop_flag:
            keep_rows.append(row)
    if keep_rows:
        return pd.DataFrame(keep_rows)
    else:
        return pd.DataFrame(columns=group_df.columns)


def filtering(origin_good_df, grouping_column, sort_key, _unused_threshold):
    """
    对 DataFrame 进行预过滤：
      1. 按 grouping_column 升序排序后进行分组，每组最多 1000 行或者相邻行该字段差值大于 2 分为一组；
      2. 对每个分组调用 process_group 进行过滤，过滤掉组内高度相关的行；
      3. 最后合并各组过滤后的数据返回。
    """
    df_sorted = origin_good_df.sort_values(by=grouping_column, ascending=True).reset_index(drop=True)
    groups = []
    n = len(df_sorted)
    if n == 0:
        return pd.DataFrame(columns=origin_good_df.columns)
    start = 0
    ref_value = df_sorted.loc[0, grouping_column]
    for i in range(n):
        current_value = df_sorted.loc[i, grouping_column]
        if (i - start + 1) > 1000 or (current_value - ref_value > 2):
            groups.append(df_sorted.iloc[start:i])
            start = i
            ref_value = current_value
    if start < n:
        groups.append(df_sorted.iloc[start:n])
    # 根据组的数量动态计算组内相关性过滤阈值
    group_threshold = max(10, 100 - int(0.3 * len(groups)))
    print(f"总分组数量：{len(groups)} ，组内相关性阈值：{group_threshold}")

    filtered_dfs = []
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_group, group, sort_key, group_threshold) for group in groups]
        for future in futures:
            result = future.result()
            if not result.empty:
                filtered_dfs.append(result)
    if filtered_dfs:
        return pd.concat(filtered_dfs, ignore_index=True)
    return pd.DataFrame(columns=origin_good_df.columns)


def gen_statistic_data(origin_good_df, removal_threshold=99):
    """
    对原始 DataFrame 进行预处理：
      1. 重置索引（将原索引保存在 'index' 列中）；
      2. 利用 filtering 预过滤数据；
      3. 遍历所有行对，利用 ProcessPoolExecutor 计算行对相关性；
      4. 根据设定阈值删除相关性过高的行。
    返回：(redundant_pairs_df, filtered_origin_good_df)
    """
    start_time = time.time()
    try:
        origin_good_df = origin_good_df.reset_index(drop=True).reset_index()
    except Exception as e:
        print("重置索引时发生异常：", e)
    print(f'待计算的数据量：{len(origin_good_df)}')

    # 使用 'kai_count' 进行分组，'net_profit_rate' 作为组内排序键
    filtered_df = filtering(origin_good_df, grouping_column='kai_count', sort_key='net_profit_rate',
                            _unused_threshold=None)
    print(f'过滤后的数据量：{len(filtered_df)}')

    parsed_rows = filtered_df.to_dict("records")
    pair_indices = list(itertools.combinations(range(len(parsed_rows)), 2))
    results = []
    with ProcessPoolExecutor(max_workers=20, initializer=init_worker, initargs=(parsed_rows,)) as executor:
        # 每个 worker 根据全局变量计算行对的相关性
        for res in executor.map(process_pair, pair_indices, chunksize=1000):
            results.append(res)

    columns = ["Row1", "Row2", "Correlation", "Row1_net_profit_rate", "Row2_net_profit_rate"]
    redundant_pairs_df = pd.DataFrame(results, columns=columns)
    print(f'行对相关性计算耗时：{time.time() - start_time:.2f} 秒')

    # 根据相关性高于 removal_threshold 的行对，选择删除其中一行
    indices_to_remove = set()
    for _, row in redundant_pairs_df[redundant_pairs_df['Correlation'] > removal_threshold].iterrows():
        if row['Row1_net_profit_rate'] >= row['Row2_net_profit_rate']:
            indices_to_remove.add(row['Row2'])
        else:
            indices_to_remove.add(row['Row1'])
    print(f'需要删除的行数：{len(indices_to_remove)}')

    filtered_origin_good_df = filtered_df[~filtered_df['index'].isin(indices_to_remove)].reset_index(drop=True)
    redundant_pairs_df = redundant_pairs_df[~(redundant_pairs_df['Row1'].isin(indices_to_remove) |
                                              redundant_pairs_df['Row2'].isin(indices_to_remove))]
    return redundant_pairs_df, filtered_origin_good_df


def find_all_valid_groups(file_path):
    """
    从指定的 parquet 文件中读取数据，
    调用 gen_statistic_data 计算统计数据，
    并将结果保存到 temp/corr 目录下，同时打印处理日志。
    """
    correlation_field = 'monthly_net_profit_detail'
    base_name = os.path.basename(file_path)
    output_path = f'temp/corr/{base_name}_origin_good_{correlation_field}.parquet'
    if os.path.exists(output_path):
        print(f'文件已存在，跳过处理：{output_path}')
        return
    origin_good_df = pd.read_parquet(file_path)
    if len(origin_good_df) > 20000:
        print(f'数据量过大，跳过处理：{len(origin_good_df)}')
        return
    redundant_pairs_df, filtered_origin_good_df = gen_statistic_data(origin_good_df)
    os.makedirs('temp/corr', exist_ok=True)
    redundant_pairs_df.to_parquet(f'temp/corr/{base_name}_corr_{correlation_field}.parquet', index=False)
    filtered_origin_good_df.to_parquet(output_path, index=False)
    print(f'保存统计数据：{file_path} -> {output_path} 当前时间: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    return filtered_origin_good_df, redundant_pairs_df


def debug():
    """
    调试入口函数：
      遍历 temp/corr 目录下符合条件的文件，调用 find_all_valid_groups 进行处理。
    """
    base_dir = 'temp/corr'
    if not os.path.exists(base_dir):
        print(f"目录不存在：{base_dir}")
        return
    file_list = os.listdir(base_dir)
    file_list = [file for file in file_list if '_feature_' in file and 'good' not in file and 'corr' not in file]
    print(f'找到 {len(file_list)} 个文件')
    for file_name in file_list:
        file_path = os.path.join(base_dir, file_name)
        find_all_valid_groups(file_path)


def example():
    debug()


if __name__ == '__main__':
    example()