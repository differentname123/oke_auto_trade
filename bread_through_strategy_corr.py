import os
import time
import pandas as pd
import numpy as np
from multiprocessing import Pool
from scipy.cluster.hierarchy import linkage, fcluster

# 常量定义
THRESHOLD_LIST = list(range(-70, 91, 10))
VALID_MODES = ["low", "middle", "high"]
EXCLUDE_COLUMNS = ['timestamp', 'net_profit_rate_new']
MIDDLE_INTERVAL_LIST = [10, 20, 30, 40]
K_LIST = [1, 2, 3, 4, 5, 10, 20, 50]
CLUSTER_THRESHOLDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
SELECT_METHODS = ['first', 'middle', 'last']


def build_corr_matrix(corr_df, id_list):
    """
    构建 id->idx 映射，以及对应的 corr_mat 方阵。
    缺失位置用 np.nan 表示。
    """
    id2idx = {cid: i for i, cid in enumerate(id_list)}
    n = len(id_list)
    mat = np.full((n, n), np.nan, dtype=float)
    # 填对称矩阵
    rows = corr_df['Row1'].to_numpy()
    cols = corr_df['Row2'].to_numpy()
    vals = corr_df['Correlation'].to_numpy(dtype=float)
    for r, c, v in zip(rows, cols, vals):
        if r in id2idx and c in id2idx:
            i, j = id2idx[r], id2idx[c]
            mat[i, j] = v
            mat[j, i] = v
    # 主对角设为 100（自相关）
    np.fill_diagonal(mat, 100.0)
    return mat, id2idx


def get_candidate_columns(good_df):
    numeric_cols = good_df.select_dtypes(include=np.number).columns.tolist()
    return [
        col for col in numeric_cols
        if col not in EXCLUDE_COLUMNS
           and 'new' not in col
           and 'index' not in col
    ]


def create_sorted_df_dict(good_df, candidate_columns):
    return {
        (col, order): good_df.sort_values(by=col, ascending=(order == "asc"))
        for col in candidate_columns for order in ["asc", "desc"]
    }


def is_valid_candidate_idx(cand_idx, selected_idxs, mode, threshold, corr_mat, middle_interval):
    if not selected_idxs:
        return True
    corrs = corr_mat[cand_idx, selected_idxs]  # 一次性取所有
    if np.isnan(corrs).any():
        return False
    if mode == "low":
        return not (corrs >= threshold).any()
    if mode == "high":
        return not (corrs < threshold).any()
    # middle
    return ((corrs >= threshold) & (corrs < threshold + middle_interval)).all()


def select_valid_candidates_from_sorted(sorted_df, mode, threshold,
                                        corr_mat, id2idx, middle_interval=10):
    selected_idxs = []
    selected_rows = []
    for row in sorted_df.itertuples(index=False):
        cid = getattr(row, 'index')
        cand_idx = id2idx.get(cid, None)
        if cand_idx is None:
            continue
        if is_valid_candidate_idx(cand_idx, selected_idxs,
                                  mode, threshold, corr_mat, middle_interval):
            selected_idxs.append(cand_idx)
            selected_rows.append(row._asdict())
    return selected_rows


def max_min_distance_sampling(sorted_df, k, corr_mat, id2idx):
    if sorted_df.empty:
        return []
    candidates = list(sorted_df.itertuples(index=False))
    selected_idxs = []
    selected_rows = []
    # 选第一个
    first = candidates[0]
    cid0 = getattr(first, 'index')
    idx0 = id2idx[cid0]
    selected_idxs.append(idx0)
    selected_rows.append(first._asdict())
    remaining = candidates[1:]

    def cand_min_dist(idx_cand):
        # 取与所有已选的 corr
        ds = 1.0 - corr_mat[idx_cand, selected_idxs] / 100.0
        if np.isnan(ds).any():
            return -np.inf
        ds = np.clip(ds, 0.0, None)
        return ds.min() if ds.size > 0 else 0.0

    while len(selected_idxs) < k and remaining:
        best_d = -np.inf
        best_i = -1
        for i, cand in enumerate(remaining):
            idx_c = id2idx[getattr(cand, 'index')]
            d = cand_min_dist(idx_c)
            if d > best_d:
                best_d = d
                best_i = i
        if best_d == -np.inf:
            break
        chosen = remaining.pop(best_i)
        idx_ch = id2idx[getattr(chosen, 'index')]
        selected_idxs.append(idx_ch)
        selected_rows.append(chosen._asdict())
    return selected_rows


def cluster_based_selection(sorted_df, corr_mat, id2idx, precomputed):
    """
    precomputed: dict 包含 'Z'（linkage 矩阵）与 'candidates' 列表。
    """
    if not precomputed:
        # 初次构建
        candidates = list(sorted_df.itertuples(index=False))
        n = len(candidates)
        if n < 2:
            return [], None
        # 预筛：任何一对缺 corr->剔除二者
        valid = np.ones(n, dtype=bool)
        idxs = [id2idx[getattr(c, 'index')] for c in candidates]
        for i in range(n):
            for j in range(i+1, n):
                if np.isnan(corr_mat[idxs[i], idxs[j]]):
                    valid[i] = False
                    valid[j] = False
        filt = [candidates[i] for i in range(n) if valid[i]]
        if len(filt) < 2:
            # 少于2个可聚类
            return [filt[0]._asdict()] if filt else [], None
        # 构造 condensed distances
        m = len(filt)
        idxs2 = [id2idx[getattr(c, 'index')] for c in filt]
        dists = []
        for i in range(m-1):
            for j in range(i+1, m):
                v = corr_mat[idxs2[i], idxs2[j]]
                d = max(0.0, 1.0 - v/100.0)
                dists.append(d)
        Z = linkage(np.array(dists, dtype=float), method='average')
        return filt, Z

    else:
        filt, Z = precomputed['filt'], precomputed['Z']
        if not filt:
            return []
        if len(filt) < 2:
            return [filt[0]._asdict()]

    return None  # signal use cached


def select_cluster_variants(filt, Z, sim_threshold):
    """
    对预筛后候选 filt 与 linkage 矩阵 Z，针对一个 sim_threshold
    调用 fcluster 并返回每簇的 first/middle/last 三种代表行集。
    """
    cutoff = 2.0 - (sim_threshold / 50.0)
    labels = fcluster(Z, t=cutoff, criterion='distance')
    clusters = {}
    for c, lab in zip(filt, labels):
        clusters.setdefault(lab, []).append(c)
    rows = []
    for group in clusters.values():
        # first/middle/last 三种方法在外层循环分别调用
        rows.append(group)
    return rows


def select_and_aggregate(corr_df, good_df, target_columns):
    # 1. id 列表 & corr_mat
    id_list = good_df['index'].unique().tolist()
    corr_mat, id2idx = build_corr_matrix(corr_df, id_list)

    # 2. 列表、排序
    candidate_columns = get_candidate_columns(good_df)
    sorted_df_dict = create_sorted_df_dict(good_df, candidate_columns)

    results = []

    # 3. 贪心 corr
    for (col, order), sdf in sorted_df_dict.items():
        for thr in THRESHOLD_LIST:
            for mode in VALID_MODES:
                if mode == "middle":
                    for mid in MIDDLE_INTERVAL_LIST:
                        cand = select_valid_candidates_from_sorted(
                            sdf, mode, thr, corr_mat, id2idx, middle_interval=mid
                        )
                        if not cand:
                            continue
                        df = pd.DataFrame(cand)
                        stats = {f"{m}_mean": df[m].mean() for m in target_columns}
                        stats.update({f"{m}_max": df[m].max() for m in target_columns})
                        stats.update({f"{m}_min": df[m].min() for m in target_columns})
                        stats.update(dict(
                            sort_column=col, threshold=thr,
                            sort_side=order, n_selected=len(df),
                            corr_select_mode=mode,
                            middle_interval=mid,
                            selection_strategy="greedy_corr"
                        ))
                        results.append(stats)
                else:
                    cand = select_valid_candidates_from_sorted(
                        sdf, mode, thr, corr_mat, id2idx, middle_interval=0
                    )
                    if not cand:
                        continue
                    df = pd.DataFrame(cand)
                    stats = {f"{m}_mean": df[m].mean() for m in target_columns}
                    stats.update({f"{m}_max": df[m].max() for m in target_columns})
                    stats.update({f"{m}_min": df[m].min() for m in target_columns})
                    stats.update(dict(
                        sort_column=col, threshold=thr,
                        sort_side=order, n_selected=len(df),
                        corr_select_mode=mode,
                        middle_interval=None,
                        selection_strategy="greedy_corr"
                    ))
                    results.append(stats)

    # 4. max-min 抽样
    for (col, order), sdf in sorted_df_dict.items():
        for k in K_LIST:
            cand = max_min_distance_sampling(sdf, k, corr_mat, id2idx)
            if not cand:
                continue
            df = pd.DataFrame(cand)
            stats = {f"{m}_mean": df[m].mean() for m in target_columns}
            stats.update({f"{m}_max": df[m].max() for m in target_columns})
            stats.update({f"{m}_min": df[m].min() for m in target_columns})
            stats.update(dict(
                sort_column=col, k=k,
                sort_side=order, n_selected=len(df),
                selection_strategy="max_min_distance"
            ))
            results.append(stats)

    # 5. 基于聚类：预先对每个 (col,order) 做一次筛 & linkage
    cluster_cache = {}
    for key, sdf in sorted_df_dict.items():
        filt, Z = cluster_based_selection(sdf, corr_mat, id2idx, precomputed=None)
        cluster_cache[key] = {'filt': filt, 'Z': Z}
    # 再针对每个 sim_threshold/method 取结果
    for (col, order), sdf in sorted_df_dict.items():
        pc = cluster_cache[(col, order)]
        filt, Z = pc['filt'], pc['Z']
        if not filt:
            continue
        if len(filt) < 2:
            # 单点直接返回
            row = filt[0]._asdict()
            for method in SELECT_METHODS:
                stats = {f"{m}_mean": row[m] for m in target_columns}
                stats.update({f"{m}_max": row[m] for m in target_columns})
                stats.update({f"{m}_min": row[m] for m in target_columns})
                stats.update(dict(
                    sort_column=col,
                    sim_threshold=None,
                    sort_side=order,
                    n_selected=1,
                    cluster_select_method=method,
                    selection_strategy="cluster_based"
                ))
                results.append(stats)
            continue

        for sim in CLUSTER_THRESHOLDS:
            # fcluster 拆簇
            cutoff = 2.0 - (sim / 50.0)
            labels = fcluster(Z, t=cutoff, criterion='distance')
            clusters = {}
            for c, lab in zip(filt, labels):
                clusters.setdefault(lab, []).append(c)
            for method in SELECT_METHODS:
                selected = []
                for grp in clusters.values():
                    if method == 'first':
                        chosen = grp[0]
                    elif method == 'middle':
                        chosen = grp[len(grp)//2]
                    else:
                        chosen = grp[-1]
                    selected.append(chosen._asdict())
                df = pd.DataFrame(selected)
                stats = {f"{m}_mean": df[m].mean() for m in target_columns}
                stats.update({f"{m}_max": df[m].max() for m in target_columns})
                stats.update({f"{m}_min": df[m].min() for m in target_columns})
                stats.update(dict(
                    sort_column=col,
                    sim_threshold=sim,
                    sort_side=order,
                    n_selected=len(df),
                    cluster_select_method=method,
                    selection_strategy="cluster_based"
                ))
                results.append(stats)

    # 6. 合并结果，计算 diff
    final_df = pd.DataFrame(results)
    base = {f"{m}_mean": good_df[m].mean() for m in target_columns}
    base.update({f"{m}_max": good_df[m].max() for m in target_columns})
    base.update({f"{m}_min": good_df[m].min() for m in target_columns})
    for m in target_columns:
        final_df[f"{m}_mean_diff"] = final_df[f"{m}_mean"] - base[f"{m}_mean"]
        final_df[f"{m}_max_diff"] = final_df[f"{m}_max"] - base[f"{m}_max"]
        final_df[f"{m}_min_diff"] = final_df[f"{m}_min"] - base[f"{m}_min"]
    return final_df


def process_pair(file_pair):
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
        good_df = good_df[good_df['hold_time_mean'] < 5000]
        idxs = good_df['index'].unique()
        corr_df = corr_df[corr_df['Row1'].isin(idxs) & corr_df['Row2'].isin(idxs)]

        result_df = select_and_aggregate(corr_df, good_df, ['net_profit_rate_new'])
        result_df['inst_id'] = inst_id
        result_df['feature'] = feature
        result_df.to_parquet(output_path, index=False)

        print(f"文件已保存：{output_path} ，耗时：{time.time()-start_time:.2f}s")
        return (file_path, "success")
    except Exception as e:
        print(f"处理失败：{file_path}，错误：{e}")
        return (file_path, "failed")


def debug1():
    base_dir = 'temp/corr'
    if not os.path.exists(base_dir):
        print(f"目录不存在：{base_dir}")
        return
    files = [f for f in os.listdir(base_dir)
             if '_feature_' in f and 'corr' in f and 'good' not in f]
    pairs = []
    for fn in files:
        gp = fn.replace('corr', 'origin_good')
        fp, gp = os.path.join(base_dir, fn), os.path.join(base_dir, gp)
        if os.path.exists(gp):
            pairs.append((fp, gp))
        else:
            print(f"缺少 good 文件：{gp}")
    print(f"共 {len(pairs)} 对待处理。")
    if not pairs:
        return
    with Pool(30) as p:
        res = list(p.imap_unordered(process_pair, pairs))
    succ = sum(1 for _,st in res if st=="success")
    fail = sum(1 for _,st in res if st=="failed")
    print(f"完成，成功 {succ}，失败 {fail}。")


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

    grouped = df.groupby(group_column_list, dropna=False).agg(agg_funcs)

    # 将多层列索引扁平化，并将 pos_ratio 列命名为 positive_ratio
    grouped.columns = [
        f"{col}_{stat}" if stat != "pos_ratio" else f"{col}_positive_ratio"
        for col, stat in grouped.columns
    ]

    # 添加每组数据的数量
    group_counts = df.groupby(group_column_list, dropna=False).size()
    grouped["group_count"] = group_counts
    return grouped.reset_index()

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
    # file_list = file_list[:10]
    for file_name in file_list:
        file_path = os.path.join(base_dir, file_name)
        df = pd.read_parquet(file_path)
        df_list.append(df)
    merged_df = pd.concat(df_list, ignore_index=True)

    group_stats_df = group_statistics(
        merged_df,
        group_column_list=['sort_column', 'sort_side', 'corr_select_mode', 'threshold', 'middle_interval'],
        target_column_list=['net_profit_rate_new_mean', 'net_profit_rate_new_min',
                            'net_profit_rate_new_mean_diff', 'net_profit_rate_new_min_diff']
    )
    group_stats_df.to_parquet('temp/corr/merged_group_stats.parquet', index=False)
    print("合并与分组统计完成。")
    # 可根据需求保存或进一步处理 group_stats_df

if __name__ == '__main__':
    debug1()