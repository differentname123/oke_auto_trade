import numpy as np
import pandas as pd
import math
import time


def calc_metrics_with_cache(agg_profit: np.ndarray, agg_kai: np.ndarray, cand_length: int, n_weeks: int):
    """
    基于候选组合的累计净利润向量和累计 kai 向量，计算聚合指标。

    参数:
      agg_profit: 累加的每周净利润数组，shape = (n_weeks,)
      agg_kai: 累加的每周 kai 数组，shape = (n_weeks,)
      cand_length: 当前候选组合的策略数量
      n_weeks: 总周数

    返回:
      一个字典，包含:
        - weekly_net_profit_sum: 所有活跃周内的平均净利润累加和
        - active_week_ratio: 活跃周比例（活跃周数 / 总周数）
        - weekly_loss_rate: 活跃周中亏损周的比例
        - weekly_net_profit_min: 活跃周内最小的平均净利润
        - weekly_net_profit_max: 活跃周内最大的平均净利润
        - weekly_net_profit_std: 活跃周内平均净利润的标准差
      如果没有活跃周，则 weekly_loss_rate 设为 inf，其他值为 nan。
    """
    avg = agg_profit / cand_length  # 每周平均净利润
    active_mask = (agg_kai > 0)  # 哪些周是活跃周
    active_count = np.count_nonzero(active_mask)

    if active_count == 0:
        return {
            "weekly_net_profit_sum": np.nan,
            "active_week_ratio": np.nan,
            "weekly_loss_rate": np.inf,  # 无活跃周视为极差
            "weekly_net_profit_min": np.nan,
            "weekly_net_profit_max": np.nan,
            "weekly_net_profit_std": np.nan
        }

    active_avg = avg[active_mask]
    sum_p = active_avg.sum()
    loss_count = np.count_nonzero(active_avg < 0)
    loss_rate = loss_count / active_count
    weekly_min = active_avg.min()
    weekly_max = active_avg.max()
    mean_val = sum_p / active_count
    std_val = np.sqrt(max((active_avg ** 2).mean() - mean_val ** 2, 0.0))
    active_week_ratio = active_count / n_weeks

    return {
        "weekly_net_profit_sum": sum_p,
        "active_week_ratio": active_week_ratio,
        "weekly_loss_rate": loss_rate,
        "weekly_net_profit_min": weekly_min,
        "weekly_net_profit_max": weekly_max,
        "weekly_net_profit_std": std_val
    }


def beam_search_opt(profit_mat: np.ndarray, kai_mat: np.ndarray, k: int, beam_width: int):
    """
    使用 Beam Search 寻找组合大小为 k 的候选组合，从而降低遍历所有组合带来的计算耗时。

    利用缓存机制，每个候选组合保存：
      - candidate: 策略索引的元组
      - last_idx: 当前候选中最后一条策略的索引（用于确保后续扩展时候选不重复且有序）
      - agg_profit: 当前候选组合累计的每周净利润数组 (shape: (n_weeks,))
      - agg_kai: 当前候选组合累计的每周 kai 数组 (shape: (n_weeks,))
      - length: 当前候选组合的策略数量
      - metrics: 由累计数据计算出的聚合指标字典

    参数:
      profit_mat: shape=(n_strategies, n_weeks)，每行为一策略每周的净利润数据
      kai_mat: shape=(n_strategies, n_weeks)，每行为一策略每周的 kai 数组
      k: 目标组合大小
      beam_width: 每一层保留的候选数量

    返回:
      一个列表，列表中每个元素为 (candidate, last_idx, agg_profit, agg_kai, length, metrics)
      其中候选组合已经按照 metrics["weekly_loss_rate"] 从小到大排序（最优在前）。
    """
    n_strategies = profit_mat.shape[0]
    n_weeks = profit_mat.shape[1]
    beam_candidates = []

    # 初始候选：单一策略，每个策略构成一条候选
    for i in range(n_strategies):
        agg_profit = profit_mat[i, :].copy()
        agg_kai = kai_mat[i, :].copy()
        cand = (i,)
        metrics = calc_metrics_with_cache(agg_profit, agg_kai, 1, n_weeks)
        beam_candidates.append((cand, i, agg_profit, agg_kai, 1, metrics))

    # 按 weekly_loss_rate 从小到大排序，并保留 Top-B
    beam_candidates.sort(key=lambda x: x[5]["weekly_loss_rate"])
    beam_candidates = beam_candidates[:beam_width]

    # 逐步扩展组合大小，从 1 扩展到 k
    for current_length in range(1, k):
        new_beam = []
        for (cand, last_idx, agg_profit, agg_kai, length, metrics) in beam_candidates:
            # 后续策略索引必须大于当前组合最后一个策略的索引
            for next_idx in range(last_idx + 1, n_strategies):
                new_cand = cand + (next_idx,)
                # 更新累计数组：向量化加法
                new_agg_profit = agg_profit + profit_mat[next_idx, :]
                new_agg_kai = agg_kai + kai_mat[next_idx, :]
                new_length = length + 1
                new_metrics = calc_metrics_with_cache(new_agg_profit, new_agg_kai, new_length, n_weeks)
                new_beam.append((new_cand, next_idx, new_agg_profit, new_agg_kai, new_length, new_metrics))
        if not new_beam:
            break
        new_beam.sort(key=lambda x: x[5]["weekly_loss_rate"])
        beam_candidates = new_beam[:beam_width]

    return beam_candidates


def choose_zuhe_beam_opt():
    """
    主函数：
      1. 从 parquet 文件中加载数据并预处理（取排序后表现最佳的100条）
      2. 对于每个组合大小 k（例如 k=2,3,4,5）执行 Beam Search 得到候选组合
      3. 将各个 k 的搜索结果汇总到一个 DataFrame 中，并保存为单个文件
    """
    inst_id_list = ['BTC', 'XRP']
    for inst_id in inst_id_list:
        print(f"正在处理 {inst_id} 的组合搜索...")
        # 读取数据（请确保文件路径正确）
        df = pd.read_parquet(f"temp_back/{inst_id}_False_short_filter_similar_strategy.parquet")
        print(f"选取策略数：{len(df)}")

        # 将原始列表数据转换为二维 numpy 数组
        profit_mat = np.stack(df['weekly_net_profit_detail'].to_numpy(), axis=0).astype(np.float32)
        kai_mat = np.stack(df['weekly_kai_count_detail'].to_numpy(), axis=0).astype(np.float32)
        original_indices = df.index.to_numpy()

        all_rows = []
        # 对不同 k 值进行搜索（这里 k 的取值为2,3,4,5）
        for k in range(2, 30):
            beam_width = 1000  # 可根据需要调整 beam 宽度
            print(f"\n====== 开始 Beam Search：组合大小 k={k} ，beam_width={beam_width} ======")
            t0 = time.time()
            beam = beam_search_opt(profit_mat, kai_mat, k, beam_width)
            elapsed = time.time() - t0

            if not beam:
                print(f"k={k} 未找到合适的组合")
                continue

            # 将 beam 中每个候选转换为字典记录，便于后续汇总
            for candidate_entry in beam:
                cand, last_idx, agg_profit, agg_kai, length, metrics = candidate_entry
                # 转换候选组合索引为原始 DataFrame 中的标签
                orig_cand = tuple(original_indices[i] for i in cand)
                row = {
                    "k": k,
                    "index_list": orig_cand,
                    "weekly_loss_rate": metrics["weekly_loss_rate"],
                    "active_week_ratio": metrics["active_week_ratio"],
                    "weekly_net_profit_sum": metrics["weekly_net_profit_sum"],
                    "weekly_net_profit_min": metrics["weekly_net_profit_min"],
                    "weekly_net_profit_max": metrics["weekly_net_profit_max"],
                    "weekly_net_profit_std": metrics["weekly_net_profit_std"]
                }
                all_rows.append(row)

            print(f"k={k}: 得到候选数量 = {len(beam)}，耗时 {elapsed:.2f} 秒")

        # 将所有候选组合汇总到一个 DataFrame 中
        result_df = pd.DataFrame(all_rows)
        # 可按 k 和 weekly_loss_rate 排序（最优候选靠前）
        result_df = result_df.sort_values(["k", "weekly_loss_rate"])

        # 保存汇总结果为一个文件（例如，使用 parquet 格式）
        result_filename = f"temp_back/result_combined_{inst_id}.parquet"
        result_df.to_parquet(result_filename, index=False)

        print(f"\n所有 k 的组合结果已保存到文件：{result_filename}")


if __name__ == "__main__":
    choose_zuhe_beam_opt()