import os
import time
import itertools
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import spearmanr
import json

# 全局变量，用于在多进程中保存解析后的行数据
GLOBAL_PARSED_ROWS = None

from typing import Tuple, List, Optional # Added Optional for clarity

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
        "Row1_net_profit_rate": row_a.get("net_profit_rate"),
        "Row2_net_profit_rate": row_b.get("net_profit_rate"),
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
        if (i - start + 1) > 1000 or (current_value - ref_value > 2):
            groups.append(df_sorted.iloc[start:i])
            start = i
            ref_value = current_value
    if start < n:
        groups.append(df_sorted.iloc[start:n])
    # 根据组的数量动态计算组内相关性过滤阈值
    group_threshold = max(50, 90 - int(0.1 * len(groups)))
    print(f"总分组数量：{len(groups)} ，组内相关性阈值：{group_threshold}")

    filtered_dfs = []
    with ProcessPoolExecutor(max_workers=3) as executor:
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
    correlation_field = 'weekly_net_profit_detail'
    base_name = os.path.basename(file_path)
    output_path = f'temp/corr/{base_name}_origin_good_{correlation_field}.parquet'
    # if os.path.exists(output_path):
    #     print(f'文件已存在，跳过处理：{output_path}')
    #     return
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
    计算所有好分组的相关性
    调试入口函数：
      遍历 temp/corr 目录下符合条件的文件，调用 find_all_valid_groups 进行处理。
    """
    inst_id_list = ['BTC', 'ETH']
    is_reverse = True
    for inst_id in inst_id_list:
        file_path = f'temp/final_good_{inst_id}_{is_reverse}_filter_all.parquet'
        find_all_valid_groups(file_path)


def select_strategies_optimized(
    strategy_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    k: int,
    strategy_id_col: str = 'index', # 新增参数：指定包含策略ID的列名
    count_col: str = 'score_score',       # 新增参数：指定包含计数的列名
    penalty_scaler: float = 1.0,
    use_absolute_correlation: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    使用贪婪算法选择一组策略，ID在指定列中，自动调整惩罚因子。

    目标是最大化总count，同时最小化策略间的相关性。

    Args:
        strategy_df (pd.DataFrame): 包含策略ID列和count列的DataFrame。
        correlation_df (pd.DataFrame): 包含策略对及其相关性的DataFrame。
                                        需要有 'Row1', 'Row2', 'Correlation' 列。
                                        'Row1', 'Row2'的值应能匹配 strategy_df 中 strategy_id_col 的值。
        k (int): 希望选出的策略数量。
        strategy_id_col (str): strategy_df 中包含策略ID的列名。默认为 'index'。
        count_col (str): strategy_df 中包含 count 的列名。默认为 'count'。
        penalty_scaler (float, optional): 自动计算的惩罚因子的缩放系数。
                                         默认为 1.0。大于1增加惩罚，小于1减少惩罚。
        use_absolute_correlation (bool, optional): 是否在计算惩罚时使用绝对相关性值。
                                                  默认为 True。

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - pd.DataFrame: 一个包含被选中策略行的DataFrame (来自原始 strategy_df)。
                            列和索引与原始 strategy_df 保持一致, 按选择顺序排序。
            - pd.DataFrame: 只包含选定策略之间相关性的新DataFrame。
                            列为 ['Row1', 'Row2', 'Correlation']。
    """

    # --- 1. 输入验证和数据准备 ---
    if strategy_id_col not in strategy_df.columns:
        raise ValueError(f"strategy_df 必须包含策略ID列: '{strategy_id_col}'")
    if count_col not in strategy_df.columns:
        raise ValueError(f"strategy_df 必须包含列: '{count_col}'")
    if not all(col in correlation_df.columns for col in ['Row1', 'Row2', 'Correlation']):
        raise ValueError("correlation_df 必须包含列: 'Row1', 'Row2', 'Correlation'")
    if k <= 0:
        empty_strategies = strategy_df.iloc[0:0] # 返回与输入结构相同的空DF
        empty_correlations = pd.DataFrame(columns=['Row1', 'Row2', 'Correlation'])
        return empty_strategies, empty_correlations

    # 检查策略ID列是否有重复值，这可能导致问题
    if strategy_df[strategy_id_col].duplicated().any():
        print(f"警告: 策略ID列 '{strategy_id_col}' 中存在重复值。这可能影响结果的准确性。")


    # 复制以防修改原始df
    original_strat_df = strategy_df.copy()
    # 创建内部使用的版本，将策略ID列设为索引，并确保是字符串
    strat_df_internal = strategy_df.copy()
    # --- 关键改动：使用指定列作为ID ---
    strat_df_internal['_internal_id_str'] = strat_df_internal[strategy_id_col].astype(str).str.strip()
    strat_df_internal = strat_df_internal.set_index('_internal_id_str', drop=True) # 使用临时字符串ID列作为索引

    corr_df = correlation_df.copy()
    # 确保相关性表中的ID也处理为字符串并去除空格
    # 假设 corr_df 中的 ID 类型与 strategy_df ID 列类型原本一致
    corr_df['Row1'] = corr_df['Row1'].astype(str).str.strip()
    corr_df['Row2'] = corr_df['Row2'].astype(str).str.strip()


    # --- 自动计算 Penalty Factor ---
    count_series = strat_df_internal[count_col] # 从内部DF获取count列
    if count_series.empty or count_series.isnull().all():
         print(f"警告: '{count_col}' 列为空或全是 NaN。使用默认 penalty_factor 1.0。")
         auto_penalty_factor = 1.0
    else:
         median_count = count_series.median()
         if pd.isna(median_count) or median_count == 0:
             mean_count = count_series.mean()
             if pd.isna(mean_count) or mean_count == 0:
                 print(f"警告: '{count_col}' 的中位数和均值都为 0 或 NaN。Penalty factor 可能无效。使用 1.0。")
                 median_count = 1.0
             else:
                 median_count = mean_count
         auto_penalty_factor = abs(median_count * penalty_scaler)
         print(f"自动计算 Penalty Factor 基准 (count 中位数/均值): {median_count:.2f}")
         print(f"使用的 Penalty Factor (基准 * scaler): {auto_penalty_factor:.2f}")

    # --- 构建相关性查找字典 ---
    print("正在构建相关性查找字典...")
    corr_dict = {}
    correlation_value_col = 'Correlation'
    row1_col = 'Row1'
    row2_col = 'Row2'
    # 使用处理过的字符串ID构建字典
    for row in corr_df.itertuples(index=False):
        s1_str = getattr(row, row1_col) # 已经是 string + stripped
        s2_str = getattr(row, row2_col) # 已经是 string + stripped
        corr = getattr(row, correlation_value_col)
        if use_absolute_correlation:
            corr = abs(corr)
        key = tuple(sorted((s1_str, s2_str)))
        # 如果同一个key存在，后面的会覆盖前面的，这通常没问题，除非你想聚合
        corr_dict[key] = corr
    print("相关性查找字典构建完成。")
    print(f"字典大小 (corr_dict): {len(corr_dict)}") # 打印大小以供检查


    def get_correlation(s1: str, s2: str, lookup_dict: dict) -> float:
        """辅助函数：从字典中查找相关性 (输入为字符串ID)"""
        if s1 == s2:
            return 1.0
        key = tuple(sorted((s1, s2)))
        value = lookup_dict.get(key, 0.0) # 缺失相关性默认为0
        # --- Debug print (可选) ---
        # if key not in lookup_dict:
        #      print(f"      DEBUG: Key {key} NOT FOUND in dict during lookup.")
        # --- End Debug ---
        return value

    # 获取所有有效策略的字符串ID (来自内部DF的索引)
    all_strategies_str = set(strat_df_internal.index)
    if not all_strategies_str:
         print("策略DataFrame内部处理后为空，无法选择。")
         return original_strat_df.iloc[0:0], corr_df.iloc[0:0]

    # 按 count 降序排序 (使用内部的 strat_df_internal)
    # 确保 count 列是数值类型
    strat_df_internal[count_col] = pd.to_numeric(strat_df_internal[count_col], errors='coerce')
    # 删除 count 为 NaN 的行，避免排序和选择出错
    strat_df_internal.dropna(subset=[count_col], inplace=True)
    all_strategies_str = set(strat_df_internal.index) # 更新有效策略集合

    if not all_strategies_str:
         print("在处理 count 列后，没有有效的策略，无法选择。")
         return original_strat_df.iloc[0:0], corr_df.iloc[0:0]


    sorted_strategies_str = strat_df_internal.sort_values(count_col, ascending=False).index.tolist()
    # sorted_strategies_str = [s for s in sorted_strategies_str if s in all_strategies_str] # 已通过dropna保证有效性

    if not sorted_strategies_str:
         print("排序后无有效策略，无法选择。")
         return original_strat_df.iloc[0:0], corr_df.iloc[0:0]

    # --- 2. 贪婪选择 ---
    selected_strategies_str = [] # 存储选中的策略的字符串ID
    candidate_pool_str = set(sorted_strategies_str)

    print(f"开始贪婪选择，目标数量 k={k}")

    # 选择第一个策略 (字符串ID)
    first_strategy_str = sorted_strategies_str[0]
    selected_strategies_str.append(first_strategy_str)
    candidate_pool_str.remove(first_strategy_str)
    # 使用 .loc 基于字符串索引查找 count
    print(f"  选择第 1 个策略: {first_strategy_str} (原始ID: {strat_df_internal.loc[first_strategy_str, strategy_id_col]}, Count: {strat_df_internal.loc[first_strategy_str, count_col]})")

    while len(selected_strategies_str) < k and candidate_pool_str:
        best_candidate_str = None
        best_score = -np.inf

        # 遍历所有候选策略 (字符串ID)
        for candidate_str in candidate_pool_str:
            # 计算与已选策略的最大相关性
            max_corr_with_selected = 0.0
            if selected_strategies_str:
                current_max_corr = 0.0
                for selected_strat_str in selected_strategies_str:
                    corr = get_correlation(candidate_str, selected_strat_str, corr_dict)
                    current_max_corr = max(current_max_corr, corr)
                max_corr_with_selected = current_max_corr

            # 获取候选策略的 count (使用 .loc 和字符串索引)
            candidate_count = strat_df_internal.loc[candidate_str, count_col]

            # 计算得分
            score = candidate_count - auto_penalty_factor * max_corr_with_selected

            # 更新最佳候选
            if score > best_score:
                best_score = score
                best_candidate_str = candidate_str

        if best_candidate_str is None:
            print(f"  在第 {len(selected_strategies_str) + 1} 步无法找到合适的候选策略，停止选择。")
            break

        # 添加最佳候选者 (字符串ID)
        selected_strategies_str.append(best_candidate_str)
        candidate_pool_str.remove(best_candidate_str)
        candidate_count = strat_df_internal.loc[best_candidate_str, count_col]
        original_id = strat_df_internal.loc[best_candidate_str, strategy_id_col] # 获取原始ID用于打印
        # 计算并打印相关信息
        final_max_corr = 0.0
        if len(selected_strategies_str) > 1:
             current_max_corr = 0.0
             for s_str in selected_strategies_str[:-1]:
                 corr = get_correlation(best_candidate_str, s_str, corr_dict)
                 current_max_corr = max(current_max_corr, corr)
             final_max_corr = current_max_corr
        print(f"  选择第 {len(selected_strategies_str)} 个策略: {best_candidate_str} (原始ID: {original_id}, Count: {candidate_count:.2f}, Score: {best_score:.2f}, MaxCorrWithSelected: {final_max_corr:.3f})")

    # --- 3. 从原始 DataFrame 中提取选定的策略 ---
    print(f"选择完成，共选出 {len(selected_strategies_str)} 个策略。")

    # 使用选定的字符串ID列表，在 *原始* DataFrame (original_strat_df) 中查找
    # 通过匹配处理过的 strategy_id_col 列来筛选
    selected_mask = original_strat_df[strategy_id_col].astype(str).str.strip().isin(selected_strategies_str)
    selected_strategies_df_unordered = original_strat_df[selected_mask].copy()

    # 保证输出的顺序与选择顺序一致
    if not selected_strategies_df_unordered.empty and selected_strategies_str:
         # 创建一个映射，从处理过的字符串ID到原始DataFrame中的行
         id_map = selected_strategies_df_unordered.set_index(selected_strategies_df_unordered[strategy_id_col].astype(str).str.strip())
         # 按照 selected_strategies_str 的顺序重新索引
         # 使用 reindex 并处理可能因重复ID或其他问题导致的缺失
         selected_strategies_df = id_map.loc[selected_strategies_str].copy()
         # 注意：如果原始 strategy_id_col 有重复值，loc[selected_strategies_str] 会包含重复行
         # 如果需要去除因原始重复ID导致的重复行，可以在这里处理
         # 例如：selected_strategies_df = selected_strategies_df[~selected_strategies_df.index.duplicated(keep='first')]
         # 或者基于原始ID列去重
         # selected_strategies_df = selected_strategies_df.drop_duplicates(subset=[strategy_id_col], keep='first')

         # 清理掉临时的索引
         selected_strategies_df.reset_index(drop=True, inplace=True)
    else:
         selected_strategies_df = selected_strategies_df_unordered # 如果没选出或为空，直接返回

    # --- 4. 生成选定策略间的相关性 DataFrame ---
    selected_strategies_set_str = set(selected_strategies_str) # 内部用字符串集合筛选

    # 筛选原始 correlation_df (其 Row1/Row2 已转为 string + stripped)
    corr_filter_mask = corr_df[row1_col].isin(selected_strategies_set_str) & \
                       corr_df[row2_col].isin(selected_strategies_set_str)
    selected_correlation_df = corr_df[corr_filter_mask].copy()

    # 可选：恢复相关性 DF 中 Row1/Row2 的原始类型（如果需要）
    # 这需要 strategy_id_col 的原始数据类型信息
    original_id_dtype = original_strat_df[strategy_id_col].dtype
    if not pd.api.types.is_string_dtype(original_id_dtype):
        # 创建从字符串ID映射回原始ID的字典
        # 需要处理重复ID：如果原始ID有重复，只保留第一个遇到的映射
        id_str_to_original_map = {}
        for _idx, row in original_strat_df.drop_duplicates(subset=[strategy_id_col], keep='first').iterrows():
            id_str = str(row[strategy_id_col]).strip()
            id_orig = row[strategy_id_col]
            id_str_to_original_map[id_str] = id_orig

        try:
            # 使用映射转换 Row1 和 Row2
            selected_correlation_df[row1_col] = selected_correlation_df[row1_col].map(id_str_to_original_map)
            selected_correlation_df[row2_col] = selected_correlation_df[row2_col].map(id_str_to_original_map)
            # 删除可能因映射失败产生的 NaN 行
            selected_correlation_df.dropna(subset=[row1_col, row2_col], inplace=True)
            # 尝试转换回原始数据类型
            selected_correlation_df[row1_col] = selected_correlation_df[row1_col].astype(original_id_dtype)
            selected_correlation_df[row2_col] = selected_correlation_df[row2_col].astype(original_id_dtype)
            print(f"相关性DataFrame中的ID已尝试恢复为原始类型: {original_id_dtype}")
        except Exception as e:
             print(f"警告：尝试将相关性DF中的ID转回原始类型时出错: {e}。将返回字符串形式的ID。")


    # --- 5. 返回结果 ---
    return selected_strategies_df, selected_correlation_df


def extract_nested_key(df, target_key):
    """
    从 DataFrame 中所有以 '_detail' 结尾的列（包含字典或JSON字符串）提取指定 target_key 的值，
    并将提取的值放入新的列中。

    Args:
        df (pd.DataFrame): 输入的 Pandas DataFrame。
        target_key (str): 需要从字典中提取的键名。

    Returns:
        pd.DataFrame: 包含新增列的 DataFrame 副本。
                      新列的命名规则是：原始列名去除 '_detail' 后缀，加上 '_<target_key>'。
    """
    df_result = df.copy() # 创建副本，避免修改原始 DataFrame
    detail_columns = [col for col in df.columns if col.endswith('_details')]

    for col_name in detail_columns:
        # 构建新列的名称
        base_name = col_name.rsplit('_details', 1)[0] # 去掉末尾的 _detail
        new_col_name = f"{base_name}_{target_key}"

        # 定义用于提取值的函数
        def get_value_from_cell(cell_data):
            if pd.isna(cell_data): # 处理 NaN 或 None 值
                return None

            data_dict = None
            # 检查是否已经是字典
            if isinstance(cell_data, dict):
                data_dict = cell_data
            # 检查是否是字符串，尝试解析为 JSON
            elif isinstance(cell_data, str):
                try:
                    parsed_data = json.loads(cell_data.replace("'", "\"")) # 尝试替换单引号以处理非标准JSON
                    if isinstance(parsed_data, dict):
                       data_dict = parsed_data
                    else:
                        # 解析结果不是字典（可能是列表、字符串、数字等）
                        return None
                except json.JSONDecodeError:
                    # 如果字符串不是有效的 JSON 格式，则认为无法提取
                    # 可以选择性地添加日志记录或警告
                    # print(f"Warning: Could not decode JSON in column {col_name}: {cell_data}")
                    return None
                except Exception as e:
                    # 处理其他可能的解析错误
                    # print(f"Warning: Error processing cell in column {col_name}: {e}")
                    return None

            # 如果成功获取了字典，则尝试提取 target_key 的值
            if data_dict is not None:
                return data_dict.get(target_key) # 使用 .get() 避免 KeyError，如果键不存在则返回 None

            # 如果不是字典也不是可解析的 JSON 字符串，则返回 None
            return None

        # 应用函数到列，并将结果存入新列
        df_result[new_col_name] = df_result[col_name].apply(get_value_from_cell)

    return df_result

def filter_good_df(inst_id):

    sort_key = 'avg_net_profit_rate_new_positive_ratio'

    good_feature_df = pd.read_parquet('temp/all_result_df_new_1000.parquet')
    new_df = extract_nested_key(good_feature_df, inst_id)
    new_df = new_df[(new_df[f'pos_ratio_{inst_id}'] > 0.5) &
                    (new_df[f'avg_net_profit_rate_new_{inst_id}'] > 5) &
                    (new_df['avg_net_profit_rate_new_min'] > -10) &
                    (new_df[sort_key] > 0.8) &
                    ((new_df['bin_seq'] > 990) | (new_df['bin_seq'] < 10)) &
                    (new_df[f'min_net_profit_{inst_id}'] > -20)
                    ]
    new_df['count'] = new_df.groupby(['feature'])['bin_seq'].transform('count')
    new_df = new_df[new_df['count'] > 1]
    new_df['feature_bin_seq'] = new_df['feature'].astype(str) + '_' + new_df['bin_seq'].astype(str)
    feature_bin_seq_list = new_df['feature_bin_seq'].tolist()
    final_df = pd.read_parquet('temp/corr/final_good_df.parquet')
    # 如果final_df包含value列，则删除
    if 'index' in final_df.columns:
        final_df = final_df.drop(columns=['index'])
    filter_df = final_df[final_df['inst_id'] == inst_id]
    filter_df = filter_df[filter_df['feature'].isin(feature_bin_seq_list)]
    # 将filter_df按照kai_column和pin_column分组然后统计相应的数量作为新的一列，叫做count
    filter_df['count'] = filter_df.groupby(['kai_column', 'pin_column'])['feature'].transform('count')
    filter_df = filter_df.drop_duplicates(subset=['kai_column', 'pin_column'])
    origin_good_path = f'temp/corr/{inst_id}_origin_good.parquet'
    strategy_df = pd.read_parquet(origin_good_path,columns=['kai_column', 'pin_column', 'index'])
    filter_df = pd.merge(filter_df, strategy_df, on=['kai_column', 'pin_column'], how='inner')
    return filter_df

def final_compute_corr():
    inst_id_list = [ 'SOL', 'TON', 'DOGE', 'XRP', 'PEPE']

    for inst_id in inst_id_list:
        corr_path = f'temp/corr/final_good_{inst_id}_True_filter_all.parquet_corr_weekly_net_profit_detail.parquet'
        origin_good_path = f'temp/corr/final_good_{inst_id}_True_filter_all.parquet_origin_good_weekly_net_profit_detail.parquet'
        strategy_df = pd.read_parquet(origin_good_path)
        # 计算score，逻辑为对kai_count取对数，然后除以max_consecutive_loss
        # strategy_df['score666'] = strategy_df['kai_count'].apply(lambda x: np.log(x) if x > 0 else 0) / strategy_df['max_consecutive_loss']

        # strategy_df = filter_good_df(inst_id)

        correlation_df = pd.read_parquet(corr_path)
        selected_strategies, selected_correlation_df = select_strategies_optimized(strategy_df, correlation_df, k=5,
                                    penalty_scaler=1.0, use_absolute_correlation=True)


        filter_df = final_df[final_df['inst_id'] == inst_id]
        # 按照kai_column和pin_column去重
        filter_df = filter_df.drop_duplicates(subset=['kai_column', 'pin_column'])
        print(f'处理 {inst_id} 的数据，数据量：{len(filter_df)}')
        redundant_pairs_df, filtered_origin_good_df = gen_statistic_data(filter_df)
        redundant_pairs_df.to_parquet(corr_path, index=False)
        filtered_origin_good_df.to_parquet(origin_good_path, index=False)

def filter_similar_strategy():
    """
    过滤掉太过于相似的策略。
    :return:
    """
    inst_id_list = ['BTC', 'ETH']
    required_columns = ['kai_count', 'net_profit_rate', 'weekly_net_profit_detail', 'max_hold_time', 'kai_column', 'pin_column', 'score_score']
    all_data_dfs = []  # 用于存储每个文件的 DataFrame
    is_reverse = True
    for inst_id in inst_id_list:
        data_file = f'temp_back/{inst_id}_{is_reverse}_pure_data_with_future.parquet'

        output_path = f'temp_back/{inst_id}_{is_reverse}_pure_data_with_future_filter_similar_strategy.parquet'
        # if os.path.exists(output_path):
        #     filtered_df = pd.read_parquet(output_path,columns=['kai_column', 'pin_column'])
        #     data_df = pd.read_parquet(data_file)
        #     merged_df = pd.merge(data_df, filtered_df, on=['kai_column', 'pin_column'], how='inner')
        #     merged_df.to_parquet(f'temp/final_good_{inst_id}_false_filter_all.parquet', index=False)
        #     print(f'文件已存在，跳过处理：{output_path}')
        #     continue
        data_df = pd.read_parquet(data_file, columns=required_columns)
        # data_df = data_df[data_df['max_hold_time'] < 5000]
        data_df = data_df[data_df['kai_count'] > 50]
        # data_df = data_df[data_df['kai_column'].str.contains('long', na=False)]
        # data_df = data_df.head(10000)
        print(f'处理 {inst_id} 的数据，数据量：{len(data_df)}')
        while True:
            filtered_df = filtering(data_df, grouping_column='kai_count', sort_key='score_score', _unused_threshold=None)
            print(f'{inst_id} 过滤后的数据量：{len(filtered_df)} 过滤前数据量：{len(data_df)}')
            if filtered_df.shape[0] == data_df.shape[0]:
                break
            data_df = filtered_df
            print(f'继续过滤')


        filtered_df.to_parquet(output_path, index=False)
        filtered_df = pd.read_parquet(output_path,columns=['kai_column', 'pin_column'])
        data_df = pd.read_parquet(data_file)
        merged_df = pd.merge(data_df, filtered_df, on=['kai_column', 'pin_column'], how='inner')
        merged_df.to_parquet(f'temp/final_good_{inst_id}_{is_reverse}_filter_all.parquet', index=False)
        print(f'保存过滤后的数据：{output_path} 长度：{len(filtered_df)}')


def example():
    # filter_similar_strategy()
    final_compute_corr()
    debug()




if __name__ == '__main__':
    example()