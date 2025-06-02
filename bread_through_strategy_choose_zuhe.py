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
    主函数：一次 Beam-Search 直接拿到 k≤max_k 的全部最优组合，
    并同时生成：
      1. 组合汇总表  result_df
      2. 元素明细表  elements_df
    """
    inst_id_list = ['BTC']          # 根据需要配置所选实例
    max_k = 10                      # 组合的最大长度
    beam_width = 1_000              # Beam Search 的宽度

    evaluation_function = lambda metrics: metrics["weekly_loss_rate"]

    for inst in inst_id_list:
        print(f"\n==== 处理 {inst} ====")
        # ------------------------------------------------------------------ #
        # 1. 读取 parquet
        df_path = Path(f"temp_back/{inst}_True_all_filter_similar_strategy.parquet")
        df = pd.read_parquet(df_path)
        print("策略条数：", len(df))

        # ------------------------------------------------------------------ #
        # 2. 转 numpy
        profit_mat = np.stack(df['weekly_net_profit_detail'].to_numpy(), axis=0).astype(np.float32)
        kai_mat    = np.stack(df['weekly_kai_count_detail'].to_numpy(),  axis=0).astype(np.float32)
        original_indices = df.index.to_numpy()

        # ------------------------------------------------------------------ #
        # 3. Beam Search
        t0 = time.time()
        layers = beam_search_multi_k(profit_mat, kai_mat,
                                     max_k=max_k, beam_width=beam_width,
                                     objective_func=evaluation_function)
        print(f"Beam-Search 用时: {time.time() - t0:.2f}s")

        # ------------------------------------------------------------------ #
        # 4-A. 组合汇总表
        rows_summary = []
        for k, beam in layers.items():
            if k < 2:        # 如需全部 k，请删除本行
                continue
            for cand, *_ , metrics in beam:
                rows_summary.append(dict(
                    k=k,
                    index_list=tuple(original_indices[i] for i in cand),
                    weekly_loss_rate=metrics["weekly_loss_rate"],
                    active_week_ratio=metrics["active_week_ratio"],
                    weekly_net_profit_sum=metrics["weekly_net_profit_sum"],
                    weekly_net_profit_min=metrics["weekly_net_profit_min"],
                    weekly_net_profit_max=metrics["weekly_net_profit_max"],
                    weekly_net_profit_std=metrics["weekly_net_profit_std"]
                ))

        result_df = (pd.DataFrame(rows_summary)
                       .sort_values(["k", "weekly_loss_rate"])
                       .reset_index(drop=True))

        # ------------------------------------------------------------------ #
        # 4-B. 元素明细表
        elem_dfs = []
        for k, beam in layers.items():
            if k < 2:
                continue
            for rank, (cand, *_ ) in enumerate(beam, start=1):
                sub = df.loc[original_indices[list(cand)]].copy()
                sub["k"] = k
                sub["comb_rank"] = rank     # 该 k 层内的名次（1=最佳）
                elem_dfs.append(sub)

        elements_df = (pd.concat(elem_dfs, ignore_index=True)
                         .reset_index(drop=True))

        # ------------------------------------------------------------------ #
        # 5. 保存
        out_dir = Path("temp_back")
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_path  = out_dir / f"result_combined_{inst}_all.parquet"
        elements_path = out_dir / f"result_elements_{inst}_all.parquet"

        result_df.to_parquet(summary_path,  index=False)
        elements_df.to_parquet(elements_path, index=False)

        print(f"组合汇总已写入 {summary_path}（{len(result_df)} 行）")
        print(f"元素明细已写入 {elements_path}（{len(elements_df)} 行）")


if __name__ == "__main__":
    choose_zuhe_beam_opt()