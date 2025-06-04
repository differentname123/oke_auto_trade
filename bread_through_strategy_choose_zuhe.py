import os
import time
import gc
from pathlib import Path
import heapq
import numpy as np
import pandas as pd


# -------------------- 评分函数 -------------------- #
def scoring_function_old(avg_profit: np.ndarray) -> float:
    """
    旧的评分函数：返回各周期平均收益中的最小值。

    参数:
        avg_profit (np.ndarray): 各周期的平均收益

    返回:
        float: 最小的平均收益作为分数
    """
    return float(avg_profit.min())


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

    # --- 1. 计算基本指标 ---
    win_rate = np.mean(p > 0)  # 胜率
    negative_profits = p[p < 0]
    avg_loss = np.abs(negative_profits).mean() if negative_profits.size > 0 else 0.0  # 平均亏损
    downside_std = negative_profits.std(ddof=0) if negative_profits.size > 0 else 0.0  # 下行波动率
    total_std = p.std(ddof=0)  # 总体波动率

    # 计算基于累计净值曲线的最大回撤
    cumulative_profit = np.cumsum(p)
    peak = np.maximum.accumulate(cumulative_profit)
    drawdown = cumulative_profit - peak
    max_drawdown = drawdown.min()  # 为负值
    mdd_pct = 0.0 if peak.max() == 0 else -max_drawdown / peak.max()  # 换算为正

    # Sortino 比率
    mean_return = p.mean()
    sortino = np.inf if downside_std == 0 else mean_return / downside_std

    # --- 2. 设置阈值 (τ / κ) ---
    tau1 = max(avg_loss, 1e-9) * 2
    tau2 = max(downside_std, 1e-9) * 2
    tau3 = max(total_std, 1e-9) * 2
    tau4 = 0.20  # 20% 回撤作为上限
    kappa = 3.0  # Sortino 比率达到 3 接近满分

    # --- 3. 计算各子分数 (0–100) ---
    score_W = 100 * win_rate
    score_L = 100 * np.exp(-avg_loss / tau1)
    score_ds = 100 * np.exp(-downside_std / tau2)
    score_std = 100 * np.exp(-total_std / tau3)
    score_MDD = 100 * np.exp(-mdd_pct / tau4)
    score_S = 100 * (1 - np.exp(-sortino / kappa)) if np.isfinite(sortino) else 100

    # --- 4. 加权汇总 ---
    weights = {
        "W": 0.35,
        "L": 0.20,
        "DS": 0.15,
        "STD": 0.10,
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


# -------------------- 指标计算函数（不含杠杆计算） -------------------- #
def calc_metrics_with_cache(agg_profit: np.ndarray,
                            agg_kai: np.ndarray,
                            cand_length: int,
                            n_weeks: int) -> dict:
    """
    针对活跃周（kai > 0），计算累计的收益、活跃周比例、负收益周比例、
    最低收益、最高收益和收益标准差。

    参数:
        agg_profit (np.ndarray): 累计利润数组
        agg_kai (np.ndarray): 累计开仓（kai）数组
        cand_length (int): 当前候选组合中的策略数量
        n_weeks (int): 周数

    返回:
        dict: 包含各项指标的字典
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

    参数:
        agg_profit (np.ndarray): 累计利润数组
        agg_kai (np.ndarray): 累计开仓数组
        cand_length (int): 当前候选组合中的策略数量

    返回:
        tuple: (optimal_leverage, optimal_capital, capital_no_leverage)
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
    factors = 1 + np.outer(L_values, active_avg) / 100  # (杠杆个数, 活跃周数)
    safe = np.all(factors > 0, axis=1)
    capitals = np.where(safe, np.prod(factors, axis=1), -np.inf)
    optimal_index = np.argmax(capitals)
    optimal_leverage = int(L_values[optimal_index])
    optimal_capital = capitals[optimal_index]

    return optimal_leverage, optimal_capital, capital_no_leverage


# -------------------- 基于候选索引计算聚合指标 -------------------- #
def calc_metrics_by_indices(candidate: tuple,
                            profit_mat: np.ndarray,
                            kai_mat: np.ndarray,
                            n_weeks: int) -> dict:
    """
    根据候选组合（策略索引）计算累计的利润和开仓数组，
    再调用 calc_metrics_with_cache 计算各项指标，并使用 scoring_function
    根据均值计算整体得分。

    参数:
        candidate (tuple): 候选策略索引组合
        profit_mat (np.ndarray): 策略收益矩阵
        kai_mat (np.ndarray): 策略开仓矩阵
        n_weeks (int): 周数

    返回:
        dict: 包含各项指标和得分的字典
    """
    agg_profit = np.sum(profit_mat[list(candidate)], axis=0)
    agg_kai = np.sum(kai_mat[list(candidate)], axis=0)
    metrics = calc_metrics_with_cache(agg_profit, agg_kai, len(candidate), n_weeks)
    avg_profit = agg_profit / len(candidate)
    metrics["score"] = scoring_function(avg_profit)
    return metrics


def get_aggregated_arrays(candidate: tuple,
                          profit_mat: np.ndarray,
                          kai_mat: np.ndarray):
    """
    根据候选组合索引，返回累计的 profit 与 kai 数组。

    参数:
         candidate (tuple): 候选策略索引
         profit_mat (np.ndarray): 策略收益矩阵
         kai_mat (np.ndarray): 策略开仓矩阵

    返回:
         tuple: (agg_profit, agg_kai)
    """
    agg_profit = np.sum(profit_mat[list(candidate)], axis=0)
    agg_kai = np.sum(kai_mat[list(candidate)], axis=0)
    return agg_profit, agg_kai


# -------------------- Beam-Search 多 k 同时返回（基于索引，延迟聚合） -------------------- #
def beam_search_multi_k(profit_mat: np.ndarray,
                        kai_mat: np.ndarray,
                        max_k: int,
                        beam_width: int,
                        objective_func) -> dict:
    """
    使用 Beam-Search 搜索候选组合，候选组合仅记录策略的索引元组，
    指标在后续聚合计算中延迟求值。依次扩展 k=1,2,...,max_k 的组合，
    每层使用 heapq.nsmallest 筛选 top beam_width 个候选（目标函数传入 -score）。

    参数:
        profit_mat (np.ndarray): 收益矩阵 (策略数 x 周数)
        kai_mat (np.ndarray): 开仓矩阵 (策略数 x 周数)
        max_k (int): 最大组合策略数
        beam_width (int): 每层保留的候选数量
        objective_func (callable): 对指标字典打分的函数

    返回:
        dict: 键为组合策略数 k，值为候选组合列表（包含索引元组、最后索引以及指标字典）
    """
    n_strategies, n_weeks = profit_mat.shape

    # k=1 的初始候选
    beam = []
    for i in range(n_strategies):
        candidate = (i,)
        metrics = calc_metrics_by_indices(candidate, profit_mat, kai_mat, n_weeks)
        beam.append((candidate, i, metrics))
    beam = heapq.nsmallest(beam_width, beam, key=lambda x: objective_func(x[2]))
    layer_results = {1: beam}

    # 逐层扩展，生成 k=2,3,... 的候选组合
    for _ in range(1, max_k):
        new_beam = []
        for candidate, last_idx, _ in beam:
            for nxt in range(last_idx + 1, n_strategies):
                new_candidate = candidate + (nxt,)
                new_metrics = calc_metrics_by_indices(new_candidate, profit_mat, kai_mat, n_weeks)
                new_beam.append((new_candidate, nxt, new_metrics))
        if not new_beam:
            break
        beam = heapq.nsmallest(beam_width, new_beam, key=lambda x: objective_func(x[2]))
        layer_results[len(beam[0][0])] = beam
        gc.collect()

    return layer_results


# -------------------- 主流程 -------------------- #
def choose_zuhe_beam_opt():
    """
    主流程：针对各实例进行策略组合选择。

    1. 根据实例名称读取策略数据；
    2. 对每个实例通过 beam-search 寻找最优组合；
    3. 计算每个组合的各项指标及杠杆指标，并保存聚合结果至 parquet 文件。
    """
    # 定义实例列表（可根据实际需要修改）
    inst_id_list = ['BTC', 'TON', 'DOGE', 'XRP', 'OKB']
    max_k = 50
    beam_width = 100  # 可根据内存情况调整
    objective = lambda m: -m["score"]

    out_dir = Path("temp_back")
    out_dir.mkdir(parents=True, exist_ok=True)

    for inst in inst_id_list:
        elements_path = out_dir / f"result_elements_{inst}_{beam_width}_opt.parquet"

        # 如果已存在之前的结果，可以选择加载（这里仅作提示）
        if elements_path.exists():
            elements_df = pd.read_parquet(elements_path)
            if not elements_df.empty and "weekly_net_profit_std_merged" in elements_df.columns and "weekly_net_profit_sum_merged" in elements_df.columns:
                elements_df['std_score'] = (elements_df['weekly_net_profit_std_merged'] /
                                            elements_df['weekly_net_profit_sum_merged'])
            print(f"已找到 {inst} 的历史结果文件：{elements_path}")

        print(f"\n==== 处理 {inst} ====")
        df_path = Path(f"temp_back/{inst}_True_all_filter_similar_strategy.parquet")
        if not df_path.exists():
            print(f"未找到文件 {df_path}，跳过该实例。")
            continue

        df = pd.read_parquet(df_path)
        print(f"策略条数：{len(df)}")
        # 假设每个策略的周数一致，以第一条记录为准
        weeks = len(df.iloc[0]['weekly_net_profit_detail'])
        print(f"每个策略的周数：{weeks}")

        # 提取收益和开仓数据（字段值均为列表格式），转换为矩阵
        profit_mat = np.stack(df['weekly_net_profit_detail'].to_numpy()).astype(np.float32)
        kai_mat = np.stack(df['weekly_kai_count_detail'].to_numpy()).astype(np.float32)

        t0 = time.time()
        layers = beam_search_multi_k(profit_mat, kai_mat,
                                     max_k=max_k, beam_width=beam_width,
                                     objective_func=objective)
        print(f"Beam-Search 用时 {time.time() - t0:.2f} 秒")

        # ---------------- 聚合结果：计算统计指标并添加评分 ---------------- #
        results = []
        # 遍历所有层，跳过 k=1 的单一策略组合
        for k, beam in layers.items():
            if k < 2:
                continue
            for candidate, _, metrics in beam:
                # 重新聚合候选组合的收益和开仓数据
                agg_profit, agg_kai = get_aggregated_arrays(candidate, profit_mat, kai_mat)
                # 计算杠杆指标
                optimal_leverage, optimal_capital, no_leverage_capital = calc_leverage_metrics(agg_profit, agg_kai,
                                                                                               len(candidate))
                # 重新计算评分，保证与 beam-search 中一致
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
                # 构造的字段名称增加 _merged 后缀，用于区分
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