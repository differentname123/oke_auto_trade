import numpy as np
import pandas as pd
import math
import time
from pathlib import Path


def calc_metrics_with_cache(agg_profit: np.ndarray,
                            agg_kai: np.ndarray,
                            cand_length: int,
                            n_weeks: int):
    """计算相关指标（原函数逻辑无需太大改动）"""
    avg = agg_profit / cand_length
    active_mask = agg_kai > 0
    active_cnt = active_mask.sum()

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
    mean_val = sum_p / active_cnt
    std_val = np.sqrt(max((active_avg ** 2).mean() - mean_val ** 2, 0.0))

    return dict(
        weekly_net_profit_sum=sum_p,
        active_week_ratio=active_cnt / n_weeks,
        weekly_loss_rate=loss_rate,
        weekly_net_profit_min=weekly_min,
        weekly_net_profit_max=weekly_max,
        weekly_net_profit_std=std_val
    )


def beam_search_multi_k(profit_mat: np.ndarray,
                        kai_mat: np.ndarray,
                        max_k: int,
                        beam_width: int,
                        objective_func=lambda metrics: metrics["weekly_loss_rate"]):
    """
    一次 Beam-Search 同时得到 1 … max_k 所有层的 top-beam 组合。

    参数：
        profit_mat: 每个策略的 weekly_profit_detail (策略数行 × 周数列)
        kai_mat: 每个策略的 weekly_kai_count_detail (策略数行 × 周数列)
        max_k: 最大组合长度（层数）
        beam_width: 每一层保留的组合个数
        objective_func: 评价函数，接收一个指标字典，返回一个数值（默认用 weekly_loss_rate 作为评价指标）

    返回值：
        dict[int, list]
            键为 k，值为该 k 对应的 beam（已按 objective_func 返回的数值升序排序），
            每个元素结构：(cand, last_idx, agg_profit, agg_kai, length, metrics)
    """
    n_strategies, n_weeks = profit_mat.shape
    # 第 0 层：单策略初始化 Beam
    beam = []
    for i in range(n_strategies):
        agg_p = profit_mat[i].copy()
        agg_k = kai_mat[i].copy()
        cand = (i,)
        metrics = calc_metrics_with_cache(agg_p, agg_k, cand_length=1, n_weeks=n_weeks)
        beam.append((cand, i, agg_p, agg_k, 1, metrics))
    beam.sort(key=lambda x: objective_func(x[5]))
    beam = beam[:beam_width]

    # 存储各层结果
    layer_results = {1: beam}

    # 逐层扩展组合
    for current_len in range(1, max_k):
        new_beam = []
        for cand, last_idx, agg_p, agg_k, length, _ in beam:
            for nxt in range(last_idx + 1, n_strategies):
                ncand = cand + (nxt,)
                nagg_p = agg_p + profit_mat[nxt]
                nagg_k = agg_k + kai_mat[nxt]
                nmetrics = calc_metrics_with_cache(nagg_p, nagg_k, length + 1, n_weeks)
                new_beam.append((ncand, nxt, nagg_p, nagg_k, length + 1, nmetrics))
        if not new_beam:
            break
        new_beam.sort(key=lambda x: objective_func(x[5]))
        beam = new_beam[:beam_width]
        layer_results[current_len + 1] = beam

    return layer_results


def choose_zuhe_beam_opt():
    """
    主函数：一次 Beam-Search 直接拿到 k≤max_k 的全部最优组合。
    """
    inst_id_list = ['SOL']          # 根据需要配置所选实例
    max_k = 50                      # 组合的最大长度
    beam_width = 1_000              # Beam Search 的宽度

    evaluation_function = lambda metrics: metrics["weekly_loss_rate"]

    for inst in inst_id_list:
        print(f"\n==== 处理 {inst} ====")
        # ------------------------------------------------------------------ #
        # 1. 读取parquet文件
        df_path = Path(f"temp/corr/{inst}_all_filter_similar_strategy.parquet_origin_good_weekly_net_profit_detail.parquet")
        df = pd.read_parquet(df_path)
        print("策略条数：", len(df))

        # ------------------------------------------------------------------ #
        # 2. 将数据转为 numpy 数组
        profit_mat = np.stack(df['weekly_net_profit_detail'].to_numpy(),
                              axis=0).astype(np.float32)
        kai_mat = np.stack(df['weekly_kai_count_detail'].to_numpy(),
                           axis=0).astype(np.float32)
        original_indices = df.index.to_numpy()

        # ------------------------------------------------------------------ #
        # 3. Beam Search（一次实现所有组合层次）
        t0 = time.time()
        layers = beam_search_multi_k(profit_mat, kai_mat,
                                     max_k=max_k, beam_width=beam_width,
                                     objective_func=evaluation_function)
        print(f"Beam-Search 用时: {time.time() - t0:.2f}s")

        # ------------------------------------------------------------------ #
        # 4. 展开并保存结果
        rows = []
        for k, beam in layers.items():
            if k < 2:  # 题主只关注 k ≥ 2 的情况，可根据需要调整
                continue
            for cand, *_ , metrics in beam:
                rows.append(dict(
                    k=k,
                    index_list=tuple(original_indices[i] for i in cand),
                    weekly_loss_rate=metrics["weekly_loss_rate"],
                    active_week_ratio=metrics["active_week_ratio"],
                    weekly_net_profit_sum=metrics["weekly_net_profit_sum"],
                    weekly_net_profit_min=metrics["weekly_net_profit_min"],
                    weekly_net_profit_max=metrics["weekly_net_profit_max"],
                    weekly_net_profit_std=metrics["weekly_net_profit_std"]
                ))

        result_df = (pd.DataFrame(rows)
                       .sort_values(["k", "weekly_loss_rate"])
                       .reset_index(drop=True))

        out_path = Path("temp_back") / f"result_combined_{inst}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        print(f"已写入 {out_path}（{len(result_df)} 行）")


if __name__ == "__main__":
    choose_zuhe_beam_opt()