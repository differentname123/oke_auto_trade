#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版本（支持通过 is_merged 控制是否先合并后处理），同时增加了月度得分：
  · beam_search_multi_k 取消节点大数组
  · 使用 shared_memory 零拷贝
  · 当 is_merged=True 时，合并同一 inst_id 的多个类型文件，然后再做后续处理；
    当 is_merged=False 时，对每种 type 单独处理。
  · 使用 weekly_net_profit_detail 和 weekly_kai_count_detail 得到 weekly_score，
    使用 monthly_net_profit_detail 和 monthly_kai_count_detail 得到 monthly_score，
    最终得分为 weekly_score 与 monthly_score 的和。
"""
import itertools
import os
import time
import gc
from pathlib import Path
import heapq
import numpy as np
import pandas as pd
import concurrent.futures
from scipy.stats import spearmanr
from multiprocessing import shared_memory, get_context


# -------------------- 稳健相关性计算函数 -------------------- #
def compute_robust_correlation(array1, array2):
    n_common = min(len(array1), len(array2))
    if n_common < 3:
        return 0.0
    x = array1[:n_common]
    y = array2[:n_common]
    std_x = np.std(x)
    std_y = np.std(y)
    if std_x == 0 or std_y == 0:
        return 0.0
    pearson_corr = np.corrcoef(x, y)[0, 1]
    spearman_corr, _ = spearmanr(x, y)
    if np.isnan(spearman_corr):
        spearman_corr = 0.0
    return float((pearson_corr + spearman_corr) / 2)


# -------------------- 评分函数 -------------------- #
def scoring_function(profit_detail: np.ndarray) -> float:
    p = np.asarray(profit_detail, dtype=float)
    n = p.size
    if n == 0:
        return 0.0
    win_rate = np.mean(p > 0)
    negative_profits = p[p < 0]
    avg_loss = np.abs(negative_profits).mean() if negative_profits.size > 0 else 0.0
    downside_std = negative_profits.std(ddof=0) if negative_profits.size > 0 else 0.0
    total_std = p.std(ddof=0)
    cumulative_profit = np.cumsum(p)
    peak = np.maximum.accumulate(cumulative_profit)
    drawdown = cumulative_profit - peak
    mdd_pct = 0.0 if peak.max() == 0 else -drawdown.min() / peak.max()
    mean_return = p.mean()
    sortino = np.inf if downside_std == 0 else mean_return / downside_std

    tau1, tau2, tau3, tau4, kappa = (
        max(avg_loss, 1e-9) * 2,
        max(downside_std, 1e-9) * 2,
        max(total_std, 1e-9) * 2,
        0.20,
        3.0,
    )
    score_W = 100 * win_rate
    score_L = 100 * np.exp(-avg_loss / tau1)
    score_ds = 100 * np.exp(-downside_std / tau2)
    score_std = 100 * np.exp(-total_std / tau3)
    score_MDD = 100 * np.exp(-mdd_pct / tau4)
    score_S = 100 * (1 - np.exp(-sortino / kappa)) if np.isfinite(sortino) else 100

    w = {"W": 0.45, "L": 0.25, "DS": 0.05, "STD": 0.05, "MDD": 0.10, "S": 0.10}
    total_score = (
            w["W"] * score_W
            + w["L"] * score_L
            + w["DS"] * score_ds
            + w["STD"] * score_std
            + w["MDD"] * score_MDD
            + w["S"] * score_S
    )
    return float(np.clip(total_score, 0, 100))


# -------------------- 根据 indices 计算指标（同时计算周度与月度数据） -------------------- #
def aggregate_and_metrics(indices: tuple,
                          weekly_profit_mat: np.ndarray,
                          weekly_kai_mat: np.ndarray,
                          n_weeks: int,
                          monthly_profit_mat: np.ndarray,
                          monthly_kai_mat: np.ndarray,
                          n_months: int) -> dict:
    cand_len = len(indices)
    # 周度聚合
    agg_profit_weekly = weekly_profit_mat[list(indices)].sum(axis=0)
    agg_kai_weekly = weekly_kai_mat[list(indices)].sum(axis=0)
    weekly_avg_profit = agg_profit_weekly / cand_len
    weekly_active_mask = agg_kai_weekly > 0
    weekly_active_count = np.count_nonzero(weekly_active_mask)

    if weekly_active_count == 0:
        weekly_metrics = dict(
            weekly_net_profit_sum=np.nan,
            active_week_ratio=np.nan,
            weekly_loss_rate=np.inf,
            weekly_net_profit_min=np.nan,
            weekly_net_profit_max=np.nan,
            weekly_net_profit_std=np.nan,
        )
    else:
        active_weekly_avg = weekly_avg_profit[weekly_active_mask]
        weekly_metrics = dict(
            weekly_net_profit_sum=active_weekly_avg.sum(),
            active_week_ratio=weekly_active_count / n_weeks,
            weekly_loss_rate=(active_weekly_avg < 0).mean(),
            weekly_net_profit_min=active_weekly_avg.min(),
            weekly_net_profit_max=active_weekly_avg.max(),
            weekly_net_profit_std=active_weekly_avg.std(),
        )
    weekly_score = scoring_function(weekly_avg_profit)

    # 月度聚合
    agg_profit_monthly = monthly_profit_mat[list(indices)].sum(axis=0)
    agg_kai_monthly = monthly_kai_mat[list(indices)].sum(axis=0)
    monthly_avg_profit = agg_profit_monthly / cand_len
    monthly_active_mask = monthly_kai_mat[list(indices)].sum(axis=0) > 0
    monthly_active_count = np.count_nonzero(monthly_active_mask)

    if monthly_active_count == 0:
        monthly_metrics = dict(
            monthly_net_profit_sum=np.nan,
            active_month_ratio=np.nan,
            monthly_loss_rate=np.inf,
            monthly_net_profit_min=np.nan,
            monthly_net_profit_max=np.nan,
            monthly_net_profit_std=np.nan,
        )
    else:
        active_monthly_avg = monthly_avg_profit[monthly_active_mask]
        monthly_metrics = dict(
            monthly_net_profit_sum=active_monthly_avg.sum(),
            active_month_ratio=monthly_active_count / n_months,
            monthly_loss_rate=(active_monthly_avg < 0).mean(),
            monthly_net_profit_min=active_monthly_avg.min(),
            monthly_net_profit_max=active_monthly_avg.max(),
            monthly_net_profit_std=active_monthly_avg.std(),
        )
    monthly_score = scoring_function(monthly_avg_profit)

    # 最终得分是周度得分与月度得分的和
    total_score = weekly_score + monthly_score

    metrics = {}
    metrics.update(weekly_metrics)
    metrics.update(monthly_metrics)
    metrics["weekly_score"] = weekly_score
    metrics["monthly_score"] = monthly_score
    metrics["score"] = total_score
    return metrics


# -------------------- 共享内存工具 -------------------- #
def _to_shared(arr: np.ndarray):
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr
    return shm, (arr.shape, arr.dtype.str)


def _attach_shared(name: str, shape, dtype_str):
    shm = shared_memory.SharedMemory(name=name, create=False)
    array = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    return shm, array


# -------------------- worker 初始化 -------------------- #
GLOBAL_PROFIT = None
GLOBAL_KAI = None
GLOBAL_PROFIT_MONTHLY = None
GLOBAL_KAI_MONTHLY = None
GLOBAL_SHM = []


def init_worker_shared(p_name, p_shape, p_dtype,
                       k_name, k_shape, k_dtype,
                       mp_name, mp_shape, mp_dtype,
                       mk_name, mk_shape, mk_dtype):
    global GLOBAL_PROFIT, GLOBAL_KAI, GLOBAL_PROFIT_MONTHLY, GLOBAL_KAI_MONTHLY, GLOBAL_SHM
    shm_p, arr_p = _attach_shared(p_name, p_shape, p_dtype)
    shm_k, arr_k = _attach_shared(k_name, k_shape, k_dtype)
    shm_mp, arr_mp = _attach_shared(mp_name, mp_shape, mp_dtype)
    shm_mk, arr_mk = _attach_shared(mk_name, mk_shape, mk_dtype)
    GLOBAL_PROFIT = arr_p
    GLOBAL_KAI = arr_k
    GLOBAL_PROFIT_MONTHLY = arr_mp
    GLOBAL_KAI_MONTHLY = arr_mk
    GLOBAL_SHM = [shm_p, shm_k, shm_mp, shm_mk]


# -------------------- 单个父节点扩展 -------------------- #
def expand_single_parent(args):
    (parent_tuple, last_idx, n_strategies, current_k, n_weeks, n_months) = args
    out = []
    for nxt in range(last_idx + 1, n_strategies):
        child = parent_tuple + (nxt,)
        metrics = aggregate_and_metrics(child,
                                        GLOBAL_PROFIT, GLOBAL_KAI, n_weeks,
                                        GLOBAL_PROFIT_MONTHLY, GLOBAL_KAI_MONTHLY, n_months)
        out.append((child, nxt, metrics))
    return out


# -------------------- Beam-Search -------------------- #
def beam_search_multi_k(weekly_profit_mat: np.ndarray,
                        weekly_kai_mat: np.ndarray,
                        monthly_profit_mat: np.ndarray,
                        monthly_kai_mat: np.ndarray,
                        max_k: int,
                        beam_width: int,
                        objective_func,
                        n_weeks: int,
                        n_months: int):
    n_strategies, _ = weekly_profit_mat.shape

    # 将周度数据放入共享内存
    shm_profit, profit_meta = _to_shared(weekly_profit_mat)
    shm_kai, kai_meta = _to_shared(weekly_kai_mat)
    # 将月度数据放入共享内存
    shm_monthly_profit, monthly_profit_meta = _to_shared(monthly_profit_mat)
    shm_monthly_kai, monthly_kai_meta = _to_shared(monthly_kai_mat)

    ctx = get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(os.cpu_count(), 25),
            mp_context=ctx,
            initializer=init_worker_shared,
            initargs=(shm_profit.name, *profit_meta,
                      shm_kai.name, *kai_meta,
                      shm_monthly_profit.name, *monthly_profit_meta,
                      shm_monthly_kai.name, *monthly_kai_meta),
    ) as pool:
        # 初始化 k=1 节点，每个节点仅包含 1 个策略
        beam = []
        for i in range(n_strategies):
            cand = (i,)
            metrics = aggregate_and_metrics(cand,
                                            weekly_profit_mat, weekly_kai_mat, n_weeks,
                                            monthly_profit_mat, monthly_kai_mat, n_months)
            beam.append((cand, i, metrics))
        beam = heapq.nsmallest(beam_width, beam, key=lambda x: objective_func(x[2]))
        layer_results = {1: beam}

        current_best_score = -objective_func(beam[0][2])
        global_best_score = current_best_score
        print(f"k=1: 最优得分: {current_best_score:.4f}, 全局最高得分: {global_best_score:.4f}")
        consecutive_no_improve = 0

        # 连续扩展 k=2 ... max_k
        for current_k in range(2, max_k + 1):
            if not beam:
                break
            t0 = time.time()
            print(
                f"Layer {current_k}: 扩展 {len(beam)} 个父节点 … 当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))}")

            tasks = [
                (parent[0], parent[1], n_strategies, current_k, n_weeks, n_months)
                for parent in beam
            ]
            futures = pool.map(expand_single_parent, tasks, chunksize=1)
            new_candidates = list(itertools.chain.from_iterable(futures))
            if not new_candidates:
                break

            beam = heapq.nsmallest(beam_width, new_candidates, key=lambda x: objective_func(x[2]))
            layer_results[current_k] = beam
            gc.collect()

            current_best_score = -objective_func(beam[0][2])
            if current_k > 20:
                if current_best_score > global_best_score:
                    global_best_score = current_best_score
                    consecutive_no_improve = 0
                else:
                    consecutive_no_improve += 1
            else:
                if current_best_score > global_best_score:
                    global_best_score = current_best_score
                    consecutive_no_improve = 0
                else:
                    consecutive_no_improve += 1

            print(f"k={current_k}: 最优得分: {current_best_score:.4f}, 全局最高得分: {global_best_score:.4f}")

            if current_k > 20 and consecutive_no_improve >= 5:
                print("连续5个 k 的最优得分没有提升，提前退出……")
                break

            print(f"完成 Layer {current_k}，用时 {time.time() - t0:.2f}s，保留 {len(beam)} 条")

    # 清理所有共享内存
    shm_profit.close();
    shm_profit.unlink()
    shm_kai.close();
    shm_kai.unlink()
    shm_monthly_profit.close();
    shm_monthly_profit.unlink()
    shm_monthly_kai.close();
    shm_monthly_kai.unlink()
    return layer_results


# -------------------- 主流程 -------------------- #
def choose_zuhe_beam_opt():
    # 开关：为 True 时先合并，再后续处理；否则按各类型单独处理。
    is_merged = False

    # 要处理的 inst_id 列表
    inst_id_list = ['BTC', 'ETH', 'SOL', 'TON', 'DOGE', 'XRP']
    # 待合并的类型（如果 is_merged=False，则对每种类型单独处理）
    type_list = ["all", "all_short"]
    max_k = 100
    beam_width = 100000
    objective = lambda m: -m["score"]
    out_dir = Path("temp_back")
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_merged:
        print("采用合并模式：对同一 inst_id 的所有类型先合并后处理")
        for inst in inst_id_list:
            print(f"\n==== 处理 {inst} 合并类型 ====")
            merged_file_path = out_dir / f"{inst}_merged_data.parquet"

            # 合并逻辑：如果合并文件不存在，则加载各类型数据后合并保存
            df_list = []
            for typ in type_list:
                df_path = out_dir / f"{inst}_True_{typ}_filter_similar_strategy.parquet"
                if not df_path.exists():
                    print(f"文件缺失 {df_path}，跳过该类型")
                    continue
                print(f"载入 {df_path}")
                df = pd.read_parquet(df_path)
                print(f"{df_path} 策略条数：{len(df)}")
                df_list.append(df)
            if len(df_list) == 0:
                print(f"{inst} 无任何类型数据，跳过")
                continue
            merged_df = pd.concat(df_list, ignore_index=True)
            merged_df.to_parquet(merged_file_path, index=False)
            beam_width = int(np.log1p(2000) * 100000 / np.log1p(len(merged_df)))

            print(f"写入合并文件 {merged_file_path}（{len(merged_df)} 行）  beam_width: {beam_width}")

            # 对合并后的数据进行处理
            elements_path = out_dir / f"result_elements_{inst}_adp_merged_op.parquet"
            if os.path.exists(elements_path):
                print(f"结果文件 {elements_path} 已存在，跳过处理")
                continue

            print(f"开始处理 {inst} 合并数据，策略条数: {len(merged_df)}")
            # 获取周度与月度的周期数
            weeks = len(merged_df.iloc[0]["weekly_net_profit_detail"])
            n_months = len(merged_df.iloc[0]["monthly_net_profit_detail"])
            weekly_profit_mat = np.stack(merged_df["weekly_net_profit_detail"].to_numpy()).astype(np.float32)
            weekly_kai_mat = np.stack(merged_df["weekly_kai_count_detail"].to_numpy()).astype(np.float32)
            monthly_profit_mat = np.stack(merged_df["monthly_net_profit_detail"].to_numpy()).astype(np.float32)
            monthly_kai_mat = np.stack(merged_df["monthly_kai_count_detail"].to_numpy()).astype(np.float32)

            t0 = time.time()
            layers = beam_search_multi_k(
                weekly_profit_mat, weekly_kai_mat,
                monthly_profit_mat, monthly_kai_mat,
                max_k=max_k, beam_width=beam_width,
                objective_func=objective,
                n_weeks=weeks, n_months=n_months
            )
            print(f"Beam-Search 完成, 用时 {time.time() - t0:.1f}s")

            # -------- 后续统计 ----------
            all_candidates = [item for k, layer in layers.items() if k >= 2 for item in layer]
            if not all_candidates:
                print("无组合结果")
                continue

            unique_pairs = {
                pair for item in all_candidates for pair in itertools.combinations(item[0], 2)
            }
            print(f"需计算相关性对数：{len(unique_pairs)}")

            from concurrent.futures import ThreadPoolExecutor
            corr_dict = {}
            with ThreadPoolExecutor(max_workers=32) as th_pool:
                for key, val in th_pool.map(
                        lambda pair: (pair, compute_robust_correlation(
                            weekly_profit_mat[pair[0]], weekly_profit_mat[pair[1]]
                        )),
                        unique_pairs,
                ):
                    corr_dict[key] = val

            results = []
            for cand, _, m in all_candidates:
                corr_lst = [corr_dict[pair] for pair in itertools.combinations(cand, 2)] or [0]
                res = {
                    "strategies": cand,
                    "k": len(cand),
                    **{f"{k}_merged": v for k, v in m.items()},
                    "max_correlation_merged": max(corr_lst),
                    "min_correlation_merged": min(corr_lst),
                    "avg_correlation_merged": float(np.mean(corr_lst)),
                }
                results.append(res)

            df_out = pd.DataFrame(results)
            df_out.to_parquet(elements_path, index=False)
            print(f"写入 {elements_path} OK （{len(df_out)} 行）")
    else:
        print("采用非合并模式：对每种 type 数据单独处理")
        for typ in type_list:
            for inst in inst_id_list:
                df_path = out_dir / f"{inst}_True_{typ}_filter_similar_strategy.parquet"
                if not df_path.exists():
                    print(f"文件缺失 {df_path}，跳过")
                    continue

                print(f"载入 {df_path}")
                df = pd.read_parquet(df_path)
                beam_width = int(np.power(1000, 0.6) * 100000 / np.power(len(df), 0.6))
                print(f"{df_path} 策略条数：{len(df)} beam_width: {beam_width}")
                print(f"\n==== 处理 {inst} ({typ}) ====")
                elements_path = out_dir / f"result_elements_{inst}_adp_{typ}_weekly_monthly.parquet"
                if os.path.exists(elements_path):
                    print(f"结果文件 {elements_path} 已存在，跳过处理")
                    continue

                weeks = len(df.iloc[0]["weekly_net_profit_detail"])
                n_months = len(df.iloc[0]["monthly_net_profit_detail"])
                weekly_profit_mat = np.stack(df["weekly_net_profit_detail"].to_numpy()).astype(np.float32)
                weekly_kai_mat = np.stack(df["weekly_kai_count_detail"].to_numpy()).astype(np.float32)
                monthly_profit_mat = np.stack(df["monthly_net_profit_detail"].to_numpy()).astype(np.float32)
                monthly_kai_mat = np.stack(df["monthly_kai_count_detail"].to_numpy()).astype(np.float32)

                t0 = time.time()
                layers = beam_search_multi_k(
                    weekly_profit_mat, weekly_kai_mat,
                    monthly_profit_mat, monthly_kai_mat,
                    max_k=max_k, beam_width=beam_width,
                    objective_func=objective,
                    n_weeks=weeks, n_months=n_months
                )
                print(f"Beam-Search 完成, 用时 {time.time() - t0:.1f}s")

                # -------- 后续统计 ----------
                all_candidates = [item for k, layer in layers.items() if k >= 2 for item in layer]
                if not all_candidates:
                    print("无组合结果")
                    continue

                unique_pairs = {
                    pair for item in all_candidates for pair in itertools.combinations(item[0], 2)
                }
                print(f"需计算相关性对数：{len(unique_pairs)}")

                from concurrent.futures import ThreadPoolExecutor
                corr_dict = {}
                with ThreadPoolExecutor(max_workers=25) as th_pool:
                    for key, val in th_pool.map(
                            lambda pair: (pair, compute_robust_correlation(
                                weekly_profit_mat[pair[0]], weekly_profit_mat[pair[1]]
                            )),
                            unique_pairs,
                    ):
                        corr_dict[key] = val

                results = []
                for cand, _, m in all_candidates:
                    corr_lst = [corr_dict[pair] for pair in itertools.combinations(cand, 2)] or [0]
                    res = {
                        "strategies": cand,
                        "k": len(cand),
                        **{f"{k}_merged": v for k, v in m.items()},
                        "max_correlation_merged": max(corr_lst),
                        "min_correlation_merged": min(corr_lst),
                        "avg_correlation_merged": float(np.mean(corr_lst)),
                    }
                    results.append(res)

                df_out = pd.DataFrame(results)
                df_out.to_parquet(elements_path, index=False)
                print(f"写入 {elements_path} OK （{len(df_out)} 行）")


if __name__ == "__main__":
    choose_zuhe_beam_opt()