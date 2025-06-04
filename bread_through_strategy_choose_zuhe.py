import os

import numpy as np
import pandas as pd
import time
from pathlib import Path
import heapq
import gc

# -------------------- 评分函数 -------------------- #
def scoring_function_old(avg_profit: np.ndarray) -> float:
    """
    输入：agg_profit / cand_length，也就是各周期的平均收益
    输出：分数（示例中取均值，你可以根据具体需求设计更复杂的评分逻辑）
    """
    score = avg_profit.min()
    return score


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

    参数
    ----
    weekly_net_profit_detail : np.ndarray
        长度 = 周数；元素 = 该周净利润（可为收益率 / 绝对金额，单位需一致）

    返回值
    ----
    float
        0–100 的综合得分，越高代表越稳健
    """

    p = np.asarray(weekly_net_profit_detail, dtype=float)
    n = p.size
    if n == 0:
        return 0.0

    # ---------- 1. 原始指标 ----------
    win_rate = np.mean(p > 0)                         # 胜率 W
    neg = p[p < 0]

    avg_loss = np.abs(neg).mean() if neg.size else 0  # 平均亏损 L̄
    downside_std = neg.std(ddof=0) if neg.size else 0 # Downside 波动 σ₋
    total_std = p.std(ddof=0)                         # 总波动 σ

    # 最大回撤（基于累计净值曲线）
    equity = np.cumsum(p)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    max_drawdown = drawdown.min()                     # 负数
    max_peak = peak.max()
    mdd_pct = 0.0 if max_peak == 0 else -max_drawdown / max_peak  # 转为正值

    # Sortino
    mean_return = p.mean()
    sortino = np.inf if downside_std == 0 else mean_return / downside_std

    # ---------- 2. 阈值（τ / κ） ----------
    # 以“自身 × 2” 作为衰减常数，使得指标等于自身时得分 ~37 分
    # 如需更一致的横向比较，可改为固定数或在外层标定
    tau1 = max(avg_loss, 1e-9) * 2
    tau2 = max(downside_std, 1e-9) * 2
    tau3 = max(total_std, 1e-9) * 2
    tau4 = 0.20                       # 20% 回撤作为“可接受”上限
    kappa = 3.0                       # Sortino≈3 时趋近满分

    # ---------- 3. 子分数 (0–100) ----------
    score_W   = 100 * win_rate
    score_L   = 100 * np.exp(-avg_loss      / tau1)
    score_ds  = 100 * np.exp(-downside_std  / tau2)
    score_std = 100 * np.exp(-total_std     / tau3)
    score_MDD = 100 * np.exp(-mdd_pct       / tau4)
    score_S   = 100 * (1 - np.exp(-sortino  / kappa)) if np.isfinite(sortino) else 100

    # ---------- 4. 加权汇总 ----------
    weights = {
        "W"  : 0.35,
        "L"  : 0.20,
        "DS" : 0.15,
        "STD": 0.10,
        "MDD": 0.10,
        "S"  : 0.10,
    }

    total_score = (
        weights["W"]  * score_W   +
        weights["L"]  * score_L   +
        weights["DS"] * score_ds  +
        weights["STD"]* score_std +
        weights["MDD"]* score_MDD +
        weights["S"]  * score_S
    )

    # 保证范围 0–100
    return float(np.clip(total_score, 0, 100))


# -------------------- 指标计算函数（不含杠杆计算） -------------------- #
def calc_metrics_with_cache(agg_profit: np.ndarray,
                            agg_kai: np.ndarray,
                            cand_length: int,
                            n_weeks: int):
    """
    计算如下指标（只针对活跃周，kai > 0）：
      - weekly_net_profit_sum: 活跃周累计平均收益之和
      - active_week_ratio: 活跃周比例
      - weekly_loss_rate: 活跃周中负收益周的比例
      - weekly_net_profit_min: 活跃周最低收益
      - weekly_net_profit_max: 活跃周最高收益
      - weekly_net_profit_std: 活跃周收益的标准差
    """
    avg = agg_profit / cand_length
    active_mask = agg_kai > 0
    active_cnt = np.count_nonzero(active_mask)

    if active_cnt == 0:
        return dict(
            weekly_net_profit_sum=np.nan,
            active_week_ratio=np.nan,
            weekly_loss_rate=np.inf,
            weekly_net_profit_min=np.nan,
            weekly_net_profit_max=np.nan,
            weekly_net_profit_std=np.nan
        )

    active_avg = avg[active_mask]
    sum_p = active_avg.sum()
    loss_rate = (active_avg < 0).mean()
    weekly_min = active_avg.min()
    weekly_max = active_avg.max()
    std_val = active_avg.std()  # 使用 numpy 内置的 std 计算标准差

    return dict(
        weekly_net_profit_sum=sum_p,
        active_week_ratio=active_cnt / n_weeks,
        weekly_loss_rate=loss_rate,
        weekly_net_profit_min=weekly_min,
        weekly_net_profit_max=weekly_max,
        weekly_net_profit_std=std_val
    )


# -------------------- 杠杆指标计算函数 -------------------- #
def calc_leverage_metrics(agg_profit: np.ndarray,
                          agg_kai: np.ndarray,
                          cand_length: int):
    """
    根据组合的收益数据（仅活跃周：kai > 0）计算杠杆指标：
      - optimal_leverage: 最优整数杠杆
      - optimal_capital: 在最优杠杆下累计收益率（初始本金为 1）
      - capital_no_leverage: 不加杠杆情况下的累计收益率
    """
    avg = agg_profit / cand_length
    active_mask = agg_kai > 0
    active_cnt = np.count_nonzero(active_mask)

    if active_cnt == 0:
        return np.nan, np.nan, np.nan

    active_avg = avg[active_mask]
    capital_no_leverage = np.prod(1 + active_avg / 100)

    min_profit = active_avg.min()
    if min_profit >= 0:
        max_possible_leverage = 30
    else:
        max_possible_leverage = int(1 / (abs(min_profit) / 100))

    L_values = np.arange(1, max_possible_leverage + 1)
    factors = 1 + np.outer(L_values, active_avg) / 100  # 形状 (max_possible_leverage, len(active_avg))
    safe = np.all(factors > 0, axis=1)
    capitals = np.where(safe, np.prod(factors, axis=1), -np.inf)
    optimal_index = np.argmax(capitals)
    optimal_leverage = int(L_values[optimal_index])
    optimal_capital = capitals[optimal_index]

    return optimal_leverage, optimal_capital, capital_no_leverage


# -------------------- 辅助函数：基于索引计算聚合指标 -------------------- #
def calc_metrics_by_indices(cand: tuple, profit_mat: np.ndarray, kai_mat: np.ndarray, n_weeks: int):
    """
    根据候选组合的索引计算累计的利润与开仓数组，然后调用 calc_metrics_with_cache 计算指标。
    同时利用 scoring_function 对平均后的每周利润计算得分，并添加到指标中。
    """
    agg_profit = np.sum(profit_mat[list(cand)], axis=0)
    agg_kai = np.sum(kai_mat[list(cand)], axis=0)
    metrics = calc_metrics_with_cache(agg_profit, agg_kai, len(cand), n_weeks)
    avg = agg_profit / len(cand)
    metrics["score"] = scoring_function(avg)
    return metrics


def get_aggregated_arrays(cand: tuple, profit_mat: np.ndarray, kai_mat: np.ndarray):
    """
    根据候选索引组合返回累计利润 agg_profit 和累计开仓 agg_kai 数组。
    """
    agg_profit = np.sum(profit_mat[list(cand)], axis=0)
    agg_kai = np.sum(kai_mat[list(cand)], axis=0)
    return agg_profit, agg_kai


# -------------------- Beam-Search，多 k 同时返回（基于索引，延迟聚合） -------------------- #
def beam_search_multi_k(profit_mat: np.ndarray,
                        kai_mat: np.ndarray,
                        max_k: int,
                        beam_width: int,
                        objective_func):
    """
    使用 Beam-Search 在不保存中间大数组的条件下搜索候选组合，
    仅记录候选组合中策略的索引元组。指标计算时基于原始矩阵聚合计算。

    参数 objective_func 用于对指标字典打分，本例中我们传入的是 -score（负评分）以实现“得分最大”的目标。
    """
    n_strategies, n_weeks = profit_mat.shape

    # k=1 时的初始候选，仅记录索引及对应指标（通过延迟聚合计算）
    beam = []
    for i in range(n_strategies):
        cand = (i,)
        metrics = calc_metrics_by_indices(cand, profit_mat, kai_mat, n_weeks)
        beam.append((cand, i, metrics))
    # 采用 heapq.nsmallest，由于目标函数传入 -score，因此相当于最大化实际得分
    beam = heapq.nsmallest(beam_width, beam, key=lambda x: objective_func(x[2]))
    layer_results = {1: beam}

    # 逐层扩展，生成 k=2,3,... 的组合
    for _ in range(1, max_k):
        new_beam = []
        for cand, last_idx, _ in beam:
            for nxt in range(last_idx + 1, n_strategies):
                new_cand = cand + (nxt,)
                new_metrics = calc_metrics_by_indices(new_cand, profit_mat, kai_mat, n_weeks)
                new_beam.append((new_cand, nxt, new_metrics))
        if not new_beam:
            break
        beam = heapq.nsmallest(beam_width, new_beam, key=lambda x: objective_func(x[2]))
        layer_results[len(beam[0][0])] = beam  # 当前候选组合的长度
        gc.collect()

    return layer_results


# -------------------- 主流程 -------------------- #
def choose_zuhe_beam_opt():
    # 定义实例列表，这里可以根据实际需要进行修改
    inst_id_list = ['SOL', 'TON', 'DOGE', 'XRP', 'OKB']
    max_k = 50
    beam_width = 10000  # 根据实际场景调整内存占用
    # 设置目标函数为 -score，使用 heapq.nsmallest 相当于选择得分最高的候选组合
    objective = lambda m: -m["score"]

    out_dir = Path("temp_back")
    out_dir.mkdir(parents=True, exist_ok=True)

    for inst in inst_id_list:
        elements_path = out_dir / f"result_elements_{inst}_{beam_width}.parquet"
        if os.path.exists(elements_path):
            elements_df = pd.read_parquet(elements_path) if elements_path.exists() else pd.DataFrame()
            elements_df['std_score'] = elements_df['weekly_net_profit_std_merged'] / elements_df['weekly_net_profit_sum_merged']
        print(f"\n==== 处理 {inst} ====")

        df_path = Path(f"temp_back/{inst}_True_all_filter_similar_strategy.parquet")
        if not df_path.exists():
            print(f"未找到文件 {df_path}，跳过该实例。")
            continue
        df = pd.read_parquet(df_path)
        print("策略条数：", len(df))
        weeks = len(df['weekly_net_profit_detail'].iloc[0])  # 获取每个策略的周数
        print(f"每个策略的周数：{weeks}")

        # 从 df 中提取 profit 和 kai 的详细数据（注意这些字段为列表形式）
        profit_mat = np.stack(df['weekly_net_profit_detail'].to_numpy()).astype(np.float32)
        kai_mat = np.stack(df['weekly_kai_count_detail'].to_numpy()).astype(np.float32)

        t0 = time.time()
        layers = beam_search_multi_k(profit_mat, kai_mat,
                                     max_k=max_k, beam_width=beam_width,
                                     objective_func=objective)
        print(f"Beam-Search 用时 {time.time() - t0:.2f}s")

        # -------------------- 聚合结果：保存统计指标并增加评分字段 -------------------- #
        results = []
        # 遍历所有层（k 从 2 开始，k=1 为单一策略不作组合）
        for k, beam in layers.items():
            if k < 2:
                continue
            for cand, last_idx, metrics in beam:
                # 根据候选组合索引重新计算累计数组
                agg_p, agg_k = get_aggregated_arrays(cand, profit_mat, kai_mat)
                # 计算杠杆指标
                optimal_leverage, optimal_capital, no_leverage_capital = calc_leverage_metrics(agg_p, agg_k, len(cand))
                # 重新计算评分（此处与 beam_search 中计算的评分保持一致）
                score_val = scoring_function(agg_p / len(cand))

                # 构造统计结果字典，字段名增加 _merged 后缀以便区分
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
                    "score": score_val  # 新增评分字段
                }
                result_item = {"strategies": cand}
                for col, val in merged_info.items():
                    result_item[col + "_merged"] = val

                results.append(result_item)

        if not results:
            print(f"{inst} 未产生任何组合结果。")
            continue

        # 将聚合结果转换为 DataFrame，并保留其它字段不变
        elements_df = pd.DataFrame(results)
        elements_df['cha_score'] = elements_df['weekly_loss_rate_merged'] * elements_df['weekly_net_profit_min_merged'] * weeks / 2
        elements_df['score_score'] = elements_df['cha_score'] / elements_df['weekly_net_profit_sum_merged']
        elements_df['score_score1'] = elements_df['weekly_net_profit_sum_merged'] / elements_df['weekly_loss_rate_merged']

        # 此外，新增的一列 "score_merged" 就代表该行的评分
        elements_df.to_parquet(elements_path, index=False)
        print(f"统计结果已写入 {elements_path}（{len(elements_df)} 行）")


if __name__ == "__main__":
    choose_zuhe_beam_opt()