import itertools
import os
import time
import gc
from pathlib import Path
import heapq
import numpy as np
import pandas as pd
import concurrent.futures


# -------------------- 旧评分函数 -------------------- #
def scoring_function_old(avg_profit: np.ndarray) -> float:
    """
    返回各周期平均收益中的最小值。
    """
    return float(avg_profit.min())


# -------------------- 新评分函数 -------------------- #
def scoring_function(weekly_net_profit_detail: np.ndarray) -> float:
    """
    给定一个策略的周度净利润数组，输出 0–100 的“稳健型”评分。

    评价原则：
      1. 周度胜率高          —— 奖励
      2. 亏损周亏得少        —— 奖励
      3. 下行波动小          —— 奖励
      4. 总体波动小          —— 奖励
      5. 最大回撤小          —— 奖励
      6. Sortino 比率高      —— 奖励（边际效用递减）

    参数:
        weekly_net_profit_detail (np.ndarray): 周度净利润数组

    返回:
        float: 0–100 的综合得分，越高代表越稳健
    """
    p = np.asarray(weekly_net_profit_detail, dtype=float)
    n = p.size
    if n == 0:
        return 0.0

    # --- 1. 计算各项指标 ---
    win_rate = np.mean(p > 0)
    negative_profits = p[p < 0]
    avg_loss = np.abs(negative_profits).mean() if negative_profits.size > 0 else 0.0
    downside_std = negative_profits.std(ddof=0) if negative_profits.size > 0 else 0.0
    total_std = p.std(ddof=0)

    cumulative_profit = np.cumsum(p)
    peak = np.maximum.accumulate(cumulative_profit)
    drawdown = cumulative_profit - peak
    max_drawdown = drawdown.min()
    mdd_pct = 0.0 if peak.max() == 0 else -max_drawdown / peak.max()

    mean_return = p.mean()
    sortino = np.inf if downside_std == 0 else mean_return / downside_std

    # --- 2. 阈值设定 ---
    tau1 = max(avg_loss, 1e-9) * 2
    tau2 = max(downside_std, 1e-9) * 2
    tau3 = max(total_std, 1e-9) * 2
    tau4 = 0.20
    kappa = 3.0

    # --- 3. 各项评分 ---
    score_W = 100 * win_rate
    score_L = 100 * np.exp(-avg_loss / tau1)
    score_ds = 100 * np.exp(-downside_std / tau2)
    score_std = 100 * np.exp(-total_std / tau3)
    score_MDD = 100 * np.exp(-mdd_pct / tau4)
    score_S = 100 * (1 - np.exp(-sortino / kappa)) if np.isfinite(sortino) else 100

    # --- 4. 加权汇总 ---
    weights = {
        "W": 0.45,
        "L": 0.25,
        "DS": 0.05,
        "STD": 0.05,
        "MDD": 0.10,
        "S": 0.10,
    }
    total_score = (weights["W"] * score_W +
                   weights["L"] * score_L +
                   weights["DS"] * score_ds +
                   weights["STD"] * score_std +
                   weights["MDD"] * score_MDD +
                   weights["S"] * score_S)
    return float(np.clip(total_score, 0, 100))


# -------------------- 计算指标函数（不含杠杆计算） -------------------- #
def calc_metrics_with_cache(agg_profit: np.ndarray,
                            agg_kai: np.ndarray,
                            cand_length: int,
                            n_weeks: int) -> dict:
    """
    针对活跃周（kai > 0），计算累计的收益、活跃周比例、负收益周比例、
    最低收益、最高收益和收益标准差。
    """
    avg_profit = agg_profit / cand_length
    active_mask = agg_kai > 0
    active_count = np.count_nonzero(active_mask)

    if active_count == 0:
        return {
            "weekly_net_profit_sum": np.nan,
            "active_week_ratio": np.nan,
            "weekly_loss_rate": np.inf,
            "weekly_net_profit_min": np.nan,
            "weekly_net_profit_max": np.nan,
            "weekly_net_profit_std": np.nan,
        }

    active_avg = avg_profit[active_mask]
    return {
        "weekly_net_profit_sum": active_avg.sum(),
        "active_week_ratio": active_count / n_weeks,
        "weekly_loss_rate": (active_avg < 0).mean(),
        "weekly_net_profit_min": active_avg.min(),
        "weekly_net_profit_max": active_avg.max(),
        "weekly_net_profit_std": active_avg.std(),
    }


# -------------------- 杠杆指标计算函数 -------------------- #
def calc_leverage_metrics(agg_profit: np.ndarray,
                          agg_kai: np.ndarray,
                          cand_length: int):
    """
    根据累计收益数据（仅针对活跃周：kai > 0）计算杠杆指标：
      - optimal_leverage: 最优整数杠杆
      - optimal_capital: 在最优杠杆下的累计收益率（初始本金为 1）
      - capital_no_leverage: 不加杠杆情况下的累计收益率
    """
    avg_profit = agg_profit / cand_length
    active_mask = agg_kai > 0
    active_count = np.count_nonzero(active_mask)

    if active_count == 0:
        return np.nan, np.nan, np.nan

    active_avg = avg_profit[active_mask]
    capital_no_leverage = np.prod(1 + active_avg / 100)

    min_profit = active_avg.min()
    if min_profit >= 0:
        max_possible_leverage = 30
    else:
        max_possible_leverage = int(1 / (abs(min_profit) / 100))

    L_values = np.arange(1, max_possible_leverage + 1)
    factors = 1 + np.outer(L_values, active_avg) / 100  # shape: (num_leverages, active_weeks)
    safe = np.all(factors > 0, axis=1)
    capitals = np.where(safe, np.prod(factors, axis=1), -np.inf)
    optimal_index = np.argmax(capitals)
    optimal_leverage = int(L_values[optimal_index])
    optimal_capital = capitals[optimal_index]

    return optimal_leverage, optimal_capital, capital_no_leverage


# -------------------- 基于累计数组计算指标（利用缓存） -------------------- #
def calc_metrics_from_agg(agg_profit: np.ndarray, agg_kai: np.ndarray, cand_length: int, n_weeks: int) -> dict:
    """
    根据已聚合的收益和开仓数据计算各项指标及评分。
    """
    metrics = calc_metrics_with_cache(agg_profit, agg_kai, cand_length, n_weeks)
    avg_profit = agg_profit / cand_length
    metrics["score"] = scoring_function(avg_profit)
    return metrics


# -------------------- 全局变量及并行初始化 -------------------- #
# 在子进程中使用全局变量来避免重复传递大数据
GLOBAL_PROFIT_MAT = None
GLOBAL_KAI_MAT = None


def init_worker(profit, kai):
    """
    ProcessPoolExecutor 的初始化函数，在子进程中设置全局变量。
    """
    global GLOBAL_PROFIT_MAT, GLOBAL_KAI_MAT
    GLOBAL_PROFIT_MAT = profit
    GLOBAL_KAI_MAT = kai


# -------------------- 候选项扩展函数 -------------------- #
def expand_candidate(parent, n_strategies, current_k, n_weeks):
    """
    对单个父候选项进行扩展。

    参数:
      parent: 元组 (candidate, last_idx, metrics, agg_profit, agg_kai)
      n_strategies: 总策略数
      current_k: 当前候选组合中策略数（父候选数+1）
      n_weeks: 周数

    返回:
      新候选项列表，每个候选项为 (candidate_tuple, last_idx, new_metrics, new_agg_profit, new_agg_kai)
    """
    candidate, last_idx, _, agg_profit, agg_kai = parent
    new_candidates = []
    for nxt in range(last_idx + 1, n_strategies):
        new_candidate = candidate + (nxt,)
        new_agg_profit = agg_profit + GLOBAL_PROFIT_MAT[nxt]
        new_agg_kai = agg_kai + GLOBAL_KAI_MAT[nxt]
        new_metrics = calc_metrics_from_agg(new_agg_profit, new_agg_kai, current_k, n_weeks)
        new_candidates.append((new_candidate, nxt, new_metrics, new_agg_profit, new_agg_kai))
    return new_candidates


# --------------------
# 全局工作函数，用于并行扩展候选项
# 参数以一个元组传入，避免局部 lambda 导致的 pickle 问题
# --------------------
def expand_candidate_worker(args):
    parent, n_strategies, current_k, n_weeks = args
    return expand_candidate(parent, n_strategies, current_k, n_weeks)


# -------------------- Beam-Search 多 k 并行扩展 -------------------- #
def beam_search_multi_k(profit_mat: np.ndarray,
                        kai_mat: np.ndarray,
                        max_k: int,
                        beam_width: int,
                        objective_func) -> dict:
    """
    使用 Beam-Search 搜索候选组合，每个候选项保存：
      (候选策略索引元组, 最后一个策略的索引, 指标字典, 累计收益数组, 累计开仓数组)

    利用并行化对候选项扩展进行加速。
    调整内容：
      1. 使用 max_workers=25 限制并行数。
      2. 在外层只创建一次进程池，而不是每一层都创建/销毁进程池。
    """
    n_strategies, n_weeks = profit_mat.shape

    # 初始化 k=1 候选项（无需并行化），直接缓存每个策略的累计数据
    beam = []
    for i in range(n_strategies):
        candidate = (i,)
        agg_profit = profit_mat[i].copy()
        agg_kai = kai_mat[i].copy()
        metrics = calc_metrics_from_agg(agg_profit, agg_kai, 1, n_weeks)
        beam.append((candidate, i, metrics, agg_profit, agg_kai))
    beam = heapq.nsmallest(beam_width, beam, key=lambda x: objective_func(x[2]))
    layer_results = {1: beam}

    # 创建一个进程池，在整个 Beam-Search 过程中重用
    with concurrent.futures.ProcessPoolExecutor(max_workers=25,
                                                initializer=init_worker,
                                                initargs=(profit_mat, kai_mat)) as pool:
        for current_k in range(2, max_k + 1):
            start_time = time.time()
            print(f"正在扩展 Layer {current_k} 当前时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            work_items = [(parent, n_strategies, current_k, n_weeks) for parent in beam]

            # 使用 submit 并利用 as_completed 来尽快收集各个任务的结果
            futures = [pool.submit(expand_candidate_worker, item) for item in work_items]
            candidate_lists = []
            for future in concurrent.futures.as_completed(futures):
                candidate_lists.append(future.result())

            new_beam = list(itertools.chain.from_iterable(candidate_lists))

            if not new_beam:
                break
            beam = heapq.nsmallest(beam_width, new_beam, key=lambda x: objective_func(x[2]))
            layer_results[current_k] = beam
            gc.collect()
            print(f"Layer {current_k} 扩展完成，耗时 {time.time() - start_time:.2f} 秒，候选数：{len(beam)} 当前时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    return layer_results


# -------------------- 主流程 -------------------- #
def choose_zuhe_beam_opt():
    """
    主流程：针对各实例进行策略组合选择。

    1. 根据实例名称读取策略数据；
    2. 针对每个实例采用并行 Beam-Search 寻找最优组合；
    3. 计算每个组合的各项指标及杠杆指标，
    4. 最终将聚合结果保存为 parquet 文件。
    """
    inst_id_list = ['BTC', 'ETH', 'SOL', 'TON', 'DOGE', 'XRP', 'OKB']
    max_k = 100
    beam_width = 10000  # 根据内存情况调整
    type_list = ['all', 'all_short']
    # 目标函数：使用 -score 使得得分越高的候选更优
    objective = lambda m: -m["score"]

    out_dir = Path("temp_back")
    out_dir.mkdir(parents=True, exist_ok=True)
    for type in type_list:
        for inst in inst_id_list:
            elements_path = out_dir / f"result_elements_{inst}_{beam_width}_{type}.parquet"

            # 如果历史文件存在，则加载提示
            if elements_path.exists():
                elements_df = pd.read_parquet(elements_path)
                if not elements_df.empty and "weekly_net_profit_std_merged" in elements_df.columns and "weekly_net_profit_sum_merged" in elements_df.columns:
                    elements_df['std_score'] = (elements_df['weekly_net_profit_std_merged'] /
                                                elements_df['weekly_net_profit_sum_merged'])
                print(f"已找到 {inst} 的历史结果文件：{elements_path}")

            print(f"\n==== 处理 {inst} ====")
            df_path = Path(f"temp_back/{inst}_True_{type}_filter_similar_strategy.parquet")
            if not df_path.exists():
                print(f"未找到文件 {df_path}，跳过该实例。")
                continue

            df = pd.read_parquet(df_path)
            print(f"策略条数：{len(df)}")
            weeks = len(df.iloc[0]['weekly_net_profit_detail'])
            print(f"每个策略的周数：{weeks}")

            # 提取收益和开仓数据，转换为矩阵（字段均为列表格式）
            profit_mat = np.stack(df['weekly_net_profit_detail'].to_numpy()).astype(np.float32)
            kai_mat = np.stack(df['weekly_kai_count_detail'].to_numpy()).astype(np.float32)

            t0 = time.time()
            layers = beam_search_multi_k(profit_mat, kai_mat,
                                         max_k=max_k, beam_width=beam_width,
                                         objective_func=objective)
            print(f"Beam-Search 用时 {time.time() - t0:.2f} 秒")

            results = []
            # 遍历所有层（跳过单一策略组合 k=1）
            for k, beam in layers.items():
                if k < 2:
                    continue
                for candidate_item in beam:
                    candidate = candidate_item[0]
                    metrics = candidate_item[2]
                    agg_profit = candidate_item[3]
                    agg_kai = candidate_item[4]
                    optimal_leverage, optimal_capital, no_leverage_capital = calc_leverage_metrics(
                        agg_profit, agg_kai, len(candidate))
                    score_val = scoring_function(agg_profit / len(candidate))
                    merged_info = {
                        "k": k,
                        "weekly_loss_rate": metrics["weekly_loss_rate"],
                        "active_week_ratio": metrics["active_week_ratio"],
                        "weekly_net_profit_sum": metrics["weekly_net_profit_sum"],
                        "weekly_net_profit_min": metrics["weekly_net_profit_min"],
                        "weekly_net_profit_max": metrics["weekly_net_profit_max"],
                        "weekly_net_profit_std": metrics["weekly_net_profit_std"],
                        "optimal_leverage": optimal_leverage,
                        "optimal_capital": optimal_capital,
                        "capital_no_leverage": no_leverage_capital,
                        "score": score_val,
                    }
                    result_item = {"strategies": candidate}
                    for key, value in merged_info.items():
                        result_item[f"{key}_merged"] = value
                    results.append(result_item)

            if not results:
                print(f"{inst} 未产生任何组合结果。")
                continue

            elements_df = pd.DataFrame(results)
            elements_df['cha_score'] = (elements_df['weekly_loss_rate_merged'] *
                                        elements_df['weekly_net_profit_min_merged'] * weeks / 2)
            elements_df['score_score'] = elements_df['cha_score'] / elements_df['weekly_net_profit_sum_merged']
            elements_df['score_score1'] = elements_df['weekly_net_profit_sum_merged'] / elements_df[
                'weekly_loss_rate_merged']

            elements_df.to_parquet(elements_path, index=False)
            print(f"统计结果已写入 {elements_path}（{len(elements_df)} 行）")


if __name__ == "__main__":
    choose_zuhe_beam_opt()