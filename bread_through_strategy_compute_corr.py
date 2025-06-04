"""
该代码主要用于对交易策略的性能数据进行去冗余和相似性过滤，其核心目标是筛选出具有差异化且优异表现的策略。主要功能包括：

稳健相关性计算：
对每条策略数据的盈利明细（字段为 weekly_net_profit_detail）计算稳健相关性，方法结合了 Pearson 与 Spearman 相关系数的均值，保证在数据量不足或标准差为零时返回 0。

行间相关性比较：
利用前述相关性计算函数，对任意两行数据（代表不同策略）的盈利明细进行比较，并以相关性乘以 100 后取整作为衡量指标。

分组内过滤：
根据指定分组字段（如 kai_count）对数据进行预分组，并在每个分组内按照指定排序键（如 capital_no_leverage）降序排列。然后，遍历组内数据，对任一行与已保留行计算相关性，若相关性超过动态设定的阈值，则舍弃当前行，从而去除组内高度相似的策略。

全局策略对比与冗余剔除：
将过滤后的策略数据转换为记录列表，并生成所有可能的策略对，利用多进程并行计算每对策略之间的相关性。对于相关系数超过设定阈值的策略对，根据各策略的净利润率决定删除其中表现较差的一方，进一步精炼策略集。

多阶段数据处理与结果保存：
代码提供了对单个及多个数据文件（parquet 格式）的处理入口，通过多进程加速的方式完成分组过滤和全局策略对比。处理结果（如策略相关性统计和过滤后的策略数据）最终保存至指定文件夹，便于后续策略选择和优化（借助外部工具函数 select_strategies_optimized）。

总体而言，该代码通过稳健的相关性计算、多层次的分组过滤与并行策略对比，有效剔除过于相似或冗余的策略数据，为进一步选出高效且差异化的交易策略提供了数据支撑。
"""

import os
import time
import itertools

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import spearmanr
import json

from common_utils import select_strategies_optimized

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
    target_field = "weekly_net_profit_detail"
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
        "Row1_net_profit_rate": row_a.get("capital_no_leverage"),
        "Row2_net_profit_rate": row_b.get("capital_no_leverage"),
    }


def process_group(group_df, sort_key, group_threshold):
    """
    对一个分组的数据先按 sort_key 降序排序，然后遍历比较每一行与已保留行的相关性，
    若相关性大于 group_threshold 则舍弃当前行，最终返回过滤后的 DataFrame。
    """
    start_time = time.time()
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
        print(f"组内过滤耗时：{time.time() - start_time:.2f} 秒，原始数量为{len(group_df)}保留行数：{len(keep_rows)}")
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
        if (i - start + 1) > 1000 or (current_value - ref_value > 10):
            groups.append(df_sorted.iloc[start:i])
            start = i
            ref_value = current_value
    if start < n:
        groups.append(df_sorted.iloc[start:n])
    # 根据组的数量动态计算组内相关性过滤阈值
    group_threshold = max(50, 98 - int(0.01 * len(groups)))
    print(f"总分组数量：{len(groups)} ，组内相关性阈值：{group_threshold}")

    filtered_dfs = []
    with ProcessPoolExecutor(max_workers=5) as executor:
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
        # 获取索引
        origin_good_df['index'] = origin_good_df.index
    except Exception as e:
        print("重置索引时发生异常：", e)
    print(f'待计算的数据量：{len(origin_good_df)}')

    # 使用 'kai_count' 进行分组，'net_profit_rate' 作为组内排序键
    filtered_df = filtering(origin_good_df, grouping_column='kai_count', sort_key='capital_no_leverage',
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
    correlation_field = 'weekly_net_profit_detail'
    base_name = os.path.basename(file_path)
    output_path = f'temp/corr/{base_name}_origin_good_{correlation_field}.parquet'
    # if os.path.exists(output_path):
    #     print(f'文件已存在，跳过处理：{output_path}')
    #     return
    origin_good_df = pd.read_parquet(file_path)

    origin_good_df = origin_good_df[origin_good_df['max_consecutive_loss'] > -30]

    # compute_rewarded_penalty_from_flat_df(origin_good_df)
    # if len(origin_good_df) > 20000:
    #     print(f'数据量过大，跳过处理：{len(origin_good_df)}')
    #     return
    redundant_pairs_df, filtered_origin_good_df = gen_statistic_data(origin_good_df)
    os.makedirs('temp/corr', exist_ok=True)
    redundant_pairs_df.to_parquet(f'temp/corr/{base_name}_corr_{correlation_field}.parquet', index=False)
    filtered_origin_good_df.to_parquet(output_path, index=False)
    print(f'保存统计数据：{file_path} -> {output_path} 当前时间: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    return filtered_origin_good_df, redundant_pairs_df


def compute_corr():
    """
    计算所有好分组的相关性
    调试入口函数：
      遍历 temp/corr 目录下符合条件的文件，调用 find_all_valid_groups 进行处理。
    """
    inst_id_list = ['BTC', 'ETH', 'SOL', 'TON', 'DOGE', 'XRP', 'OKB']
    is_reverse_list = [False, True]
    for inst_id in inst_id_list:
        df_list = []
        output_path = f'temp_back/{inst_id}_all_short_filter_similar_strategy.parquet'
        if os.path.exists(output_path):
            print(f'文件已存在，直接加载：{output_path}')
        else:
            for is_reverse in is_reverse_list:
                file_path = f'temp_back/{inst_id}_{is_reverse}_short_filter_similar_strategy.parquet'
                df = pd.read_parquet(file_path)
                df_list.append(df)
            df = pd.concat(df_list, ignore_index=True)
            # 将df按照capital_no_leverage降序排序
            df.sort_values(by='capital_no_leverage', ascending=False, inplace=True)
            df =df.head(10000)
            df.to_parquet(output_path, index=False)
            print(f'合并数据：{output_path} 当前时间: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
        find_all_valid_groups(output_path)


def final_compute_corr():
    inst_id_list = [ 'SOL', 'TON', 'DOGE', 'XRP', 'PEPE']
    is_reverse = 'all'

    for inst_id in inst_id_list:
        corr_path = f"temp/corr/{inst_id}_{is_reverse}_filter_similar_strategy.parquet_corr_weekly_net_profit_detail.parquet"
        origin_good_path = f"temp/corr/{inst_id}_{is_reverse}_filter_similar_strategy.parquet_origin_good_weekly_net_profit_detail.parquet"
        strategy_df = pd.read_parquet(origin_good_path)
        # 计算score，逻辑为对kai_count取对数，然后除以max_consecutive_loss
        # strategy_df['score666'] = strategy_df['kai_count'].apply(lambda x: np.log(x) if x > 0 else 0) / strategy_df['max_consecutive_loss']

        # strategy_df = filter_good_df(inst_id)

        correlation_df = pd.read_parquet(corr_path)
        selected_strategies, selected_correlation_df = select_strategies_optimized(strategy_df, correlation_df, k=40,
                                    penalty_scaler=0.1, use_absolute_correlation=True)


        print(f'策略数量：{len(selected_strategies)}')

def add_side(df):
    """
    添加 is_reverse 列，True 表示反向交易，False 表示正向交易。
    :param df: DataFrame
    :return: DataFrame
    """
    # 遍历df
    for index, row in df.iterrows():
        side = 'buy'
        kai_column = row['kai_column']
        if row['is_reverse']:
            if 'long' in kai_column:
                side = 'sell'
        else:
            if 'short' in kai_column:
                side = 'sell'
        df.at[index, 'side'] = side
    return df


def filter_similar_strategy_all():
    """
    过滤掉太过于相似的策略。
    :return:
    """
    inst_id_list = ['BTC', 'ETH', 'SOL', 'TON', 'DOGE', 'XRP', 'OKB']
    is_reverse_list = [False, True]
    for inst_id in inst_id_list:
        df_list = []
        for is_reverse in is_reverse_list:
            data_file = f'temp_back\statistic_results_final_{inst_id}_{is_reverse}_short.parquet'
            if not os.path.exists(data_file):
                print(f'文件不存在，跳过处理：{data_file}')
                continue

            output_path = f'temp_back/{inst_id}_{is_reverse}_all_short_filter_similar_strategy.parquet'
            # if os.path.exists(output_path):
            #     print(f'文件已存在，跳过处理：{output_path}')
            #     continue
            data_df = pd.read_parquet(data_file)
            # data_df = data_df[data_df['max_hold_time'] < 5000]
            data_df = data_df[data_df['kai_count'] > 50]
            data_df = data_df[data_df['max_consecutive_loss'] > -30]
            data_df = data_df[data_df['hold_time_mean'] < 3000]
            # capital_no_leverage
            data_df = data_df[data_df['capital_no_leverage'] > 1.1]
            df_list.append(data_df)
        if len(df_list) == 0:
            continue
        data_df = pd.concat(df_list, ignore_index=True)
        temp_path = f'temp_back/{inst_id}_all_short.parquet'
        # if os.path.exists(temp_path):
        #     print(f'文件已存在，跳过处理：{temp_path}')
        #     continue
        data_df = add_side(data_df)
        data_df = data_df[data_df['side'] == 'sell']
        data_df.to_parquet(temp_path, index=False)
        # if os.path.exists(output_path):
        #     print(f'文件已存在，跳过处理：{output_path}')
        #     continue
        print(f'处理 {inst_id} 的数据 {len(df_list)}，数据量：{len(data_df)}')
        while True:
            filtered_df = filtering(data_df, grouping_column='kai_count', sort_key='capital_no_leverage', _unused_threshold=None)
            print(f'{inst_id} 过滤后的数据量：{len(filtered_df)} 过滤前数据量：{len(data_df)}')
            if (abs(filtered_df.shape[0] - data_df.shape[0]) < 0.01 * data_df.shape[0]) or abs(filtered_df.shape[0] - data_df.shape[0]) == 0:
                break
            data_df = filtered_df
            print(f'继续过滤')


        filtered_df.to_parquet(output_path, index=False)
        print(f'保存过滤后的数据：{output_path} 长度：{len(filtered_df)}')

def filter_similar_strategy():
    """
    过滤掉太过于相似的策略。
    :return:
    """
    inst_id_list = ['BTC', 'ETH', 'SOL', 'TON', 'DOGE', 'XRP', 'OKB']
    is_reverse_list = [False, True]
    for is_reverse in is_reverse_list:
        for inst_id in inst_id_list:
            data_file = f'temp_back\statistic_results_final_{inst_id}_{is_reverse}.parquet'
            if not os.path.exists(data_file):
                print(f'文件不存在，跳过处理：{data_file}')
                continue

            output_path = f'temp_back/{inst_id}_{is_reverse}_all_filter_similar_strategy.parquet'
            # if os.path.exists(output_path):
            #     print(f'文件已存在，跳过处理：{output_path}')
            #     continue
            data_df = pd.read_parquet(data_file)
            # data_df = data_df[data_df['max_hold_time'] < 5000]
            data_df = data_df[data_df['kai_count'] > 50]
            data_df = data_df[data_df['max_consecutive_loss'] > -30]
            data_df = data_df[data_df['hold_time_mean'] < 3000]
            # capital_no_leverage
            data_df = data_df[data_df['capital_no_leverage'] > 1.1]
            print(f'处理 {inst_id} 的数据，数据量：{len(data_df)}')
            while True:
                filtered_df = filtering(data_df, grouping_column='kai_count', sort_key='capital_no_leverage', _unused_threshold=None)
                print(f'{inst_id} 过滤后的数据量：{len(filtered_df)} 过滤前数据量：{len(data_df)}')
                if filtered_df.shape[0] == data_df.shape[0]:
                    break
                data_df = filtered_df
                print(f'继续过滤')


            filtered_df.to_parquet(output_path, index=False)
            print(f'保存过滤后的数据：{output_path} 长度：{len(filtered_df)}')

def example():
    filter_similar_strategy_all()
    # final_compute_corr()
    # compute_corr()




if __name__ == '__main__':
    example()