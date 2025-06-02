import numpy as np
import pandas as pd
import time
from pathlib import Path


# -------------------- 指标计算函数，无改动 -------------------- #
def calc_metrics_with_cache(agg_profit: np.ndarray,
                            agg_kai: np.ndarray,
                            cand_length: int,
                            n_weeks: int):
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


# -------------------- Beam-Search，多 k 同时返回 -------------------- #
def beam_search_multi_k(profit_mat: np.ndarray,
                        kai_mat: np.ndarray,
                        max_k: int,
                        beam_width: int,
                        objective_func=lambda m: m["weekly_loss_rate"]):
    n_strategies, n_weeks = profit_mat.shape

    # 初始化（k=1）
    beam = []
    for i in range(n_strategies):
        agg_p, agg_k = profit_mat[i].copy(), kai_mat[i].copy()
        metrics = calc_metrics_with_cache(agg_p, agg_k, 1, n_weeks)
        beam.append(((i,), i, agg_p, agg_k, 1, metrics))
    beam.sort(key=lambda x: objective_func(x[5]))
    beam = beam[:beam_width]

    layer_results = {1: beam}

    # 逐层扩展
    for cur_len in range(1, max_k):
        new_beam = []
        for cand, last_idx, agg_p, agg_k, length, _ in beam:
            for nxt in range(last_idx + 1, n_strategies):
                ncand = cand + (nxt,)
                nagg_p, nagg_k = agg_p + profit_mat[nxt], agg_k + kai_mat[nxt]
                nmetrics = calc_metrics_with_cache(nagg_p, nagg_k, length + 1, n_weeks)
                new_beam.append((ncand, nxt, nagg_p, nagg_k, length + 1, nmetrics))
        if not new_beam:
            break
        new_beam.sort(key=lambda x: objective_func(x[5]))
        beam = new_beam[:beam_width]
        layer_results[cur_len + 1] = beam

    return layer_results


# -------------------- 主流程 -------------------- #
def choose_zuhe_beam_opt():
    inst_id_list = ['BTC', 'ETH', 'SOL', 'TON', 'DOGE', 'XRP', 'OKB']     # 可自行修改
    max_k = 50
    beam_width = 1_000
    objective = lambda m: m["weekly_loss_rate"]
    # 5) 保存
    out_dir = Path("temp_back")
    out_dir.mkdir(parents=True, exist_ok=True)
    for inst in inst_id_list:
        elements_path = out_dir / f"result_elements_{inst}.parquet"
        if elements_path.exists():
            elements_df = pd.read_parquet(elements_path)
            elements_df['cha_score'] = elements_df['weekly_loss_rate_merged'] * elements_df['weekly_net_profit_min_merged']* elements_df['weekly_loss_rate_merged'] * elements_df['weekly_net_profit_min_merged']
            elements_df['score_score'] = elements_df['cha_score'] / elements_df['weekly_net_profit_sum_merged']
        print(f"\n==== 处理 {inst} ====")

        # 1) 读取 parquet
        df_path = Path(f"temp_back/{inst}_True_all_filter_similar_strategy.parquet")
        if not df_path.exists():
            print(f"未找到文件 {df_path}，跳过该实例。")
            continue
        df = pd.read_parquet(df_path)
        print("策略条数：", len(df))

        # 2) 转 numpy
        profit_mat = np.stack(df['weekly_net_profit_detail'].to_numpy()).astype(np.float32)
        kai_mat    = np.stack(df['weekly_kai_count_detail'].to_numpy()).astype(np.float32)
        orig_idx   = df.index.to_numpy()     # 保存原 dataframe 行索引

        # 3) Beam-Search
        t0 = time.time()
        layers = beam_search_multi_k(profit_mat, kai_mat,
                                     max_k=max_k, beam_width=beam_width,
                                     objective_func=objective)
        print(f"Beam-Search 用时 {time.time() - t0:.2f}s")

        # 4) 生成 elements_df
        elem_parts = []
        for k, beam in layers.items():
            if k < 2:                # 需要所有 k 删掉本行
                continue
            for cand, *_ , metrics in beam:
                # 取出该组合涉及的策略行
                part = df.loc[orig_idx[list(cand)]].copy()

                # 为每条记录追加 7 个 *_merged 列
                merged_info = dict(
                    k=k,
                    # index_list=tuple(orig_idx[i] for i in cand),
                    weekly_loss_rate=metrics["weekly_loss_rate"],
                    active_week_ratio=metrics["active_week_ratio"],
                    weekly_net_profit_sum=metrics["weekly_net_profit_sum"],
                    weekly_net_profit_min=metrics["weekly_net_profit_min"],
                    weekly_net_profit_max=metrics["weekly_net_profit_max"],
                    weekly_net_profit_std=metrics["weekly_net_profit_std"]
                )

                part['capital_no_leverage_min'] = part['capital_no_leverage'].min()
                part['max_capital_no_leverage_max'] = part['capital_no_leverage'].max()
                part['capital_no_leverage_mean'] = part['capital_no_leverage'].mean()
                part['net_profit_rate_min'] = part['net_profit_rate'].min()
                part['net_profit_rate_max'] = part['net_profit_rate'].max()
                part['net_profit_rate_mean'] = part['net_profit_rate'].mean()

                for col, val in merged_info.items():
                    part[col + "_merged"] = val

                elem_parts.append(part)

        elements_df = pd.concat(elem_parts, ignore_index=True)


        elements_df.to_parquet(elements_path, index=False)

        print(f"元素明细已写入 {elements_path}（{len(elements_df)} 行）")


if __name__ == "__main__":
    choose_zuhe_beam_opt()