import numpy as np
import pandas as pd
import time
from pathlib import Path


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

    注意：杠杆相关指标将在后续生成 elements_df 时再计算。
    """
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


# -------------------- 杠杆指标计算函数 -------------------- #
def calc_leverage_metrics(agg_profit: np.ndarray,
                          agg_kai: np.ndarray,
                          cand_length: int):
    """
    根据组合的收益数据（仅活跃周：kai > 0）计算杠杆指标：
      - optimal_leverage: 最优整数杠杆
      - optimal_capital: 在最优杠杆下累计收益率（初始本金为 1）
      - capital_no_leverage: 不加杠杆情况下的累计收益率

    计算步骤：
      1. 计算平均收益数组 active_avg = (agg_profit / cand_length)[agg_kai>0]
      2. 计算不加杠杆的累计收益率：逐笔累乘 (1 + profit/100)
      3. 根据最差的一笔确定允许的最大杠杆：
            若 min_profit >= 0，则默认上限为 10，
            否则需满足 1 + (min_profit/100)*L > 0  ==> L < 1 / (|min_profit|/100)
      4. 模拟遍历 1 到 max_possible_leverage，选择使累计资本最大的杠杆。
    """
    avg = agg_profit / cand_length
    active_mask = agg_kai > 0
    if active_mask.sum() == 0:
        return np.nan, np.nan, np.nan
    active_avg = avg[active_mask]

    # 不加杠杆的累计收益率
    capital_no_leverage = np.prod(1 + active_avg / 100)

    # 根据亏损交易确定最大安全杠杆
    min_profit = active_avg.min()
    if min_profit >= 0:
        max_possible_leverage = 30
    else:
        max_possible_leverage = int(1 / (abs(min_profit) / 100))

    optimal_leverage = None
    optimal_capital = -np.inf

    for L in range(1, max_possible_leverage + 1):
        current_capital = 1.0
        safe = True
        for profit in active_avg:
            factor = 1 + (profit / 100) * L
            if factor <= 0:
                safe = False
                break
            current_capital *= factor
        if safe and current_capital > optimal_capital:
            optimal_capital = current_capital
            optimal_leverage = L

    return optimal_leverage, optimal_capital, capital_no_leverage


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
    inst_id_list = ['BTC', 'ETH', 'SOL', 'TON', 'DOGE', 'XRP', 'OKB']  # 可自行修改
    max_k = 50
    beam_width = 1_000
    objective = lambda m: m["weekly_loss_rate"]

    out_dir = Path("temp_back")
    out_dir.mkdir(parents=True, exist_ok=True)
    for inst in inst_id_list:
        elements_path = out_dir / f"result_elements_{inst}.parquet"
        if elements_path.exists():
            elements_df = pd.read_parquet(elements_path)
            elements_df['cha_score'] = (
                elements_df['weekly_loss_rate_merged'] *
                elements_df['weekly_net_profit_min_merged'] *
                elements_df['weekly_loss_rate_merged'] *
                elements_df['weekly_net_profit_min_merged']
            )
            elements_df['score_score'] = elements_df['cha_score'] / elements_df['weekly_net_profit_sum_merged']
        print(f"\n==== 处理 {inst} ====")

        df_path = Path(f"temp_back/{inst}_True_all_filter_similar_strategy.parquet")
        if not df_path.exists():
            print(f"未找到文件 {df_path}，跳过该实例。")
            continue
        df = pd.read_parquet(df_path)
        print("策略条数：", len(df))

        # 将 DataFrame 中的列表转换为 numpy 数组
        profit_mat = np.stack(df['weekly_net_profit_detail'].to_numpy()).astype(np.float32)
        kai_mat = np.stack(df['weekly_kai_count_detail'].to_numpy()).astype(np.float32)
        orig_idx = df.index.to_numpy()  # 保存原 dataframe 行索引

        t0 = time.time()
        layers = beam_search_multi_k(profit_mat, kai_mat,
                                     max_k=max_k, beam_width=beam_width,
                                     objective_func=objective)
        print(f"Beam-Search 用时 {time.time() - t0:.2f}s")

        elem_parts = []
        # 遍历所有层（k 从 2 开始）
        for k, beam in layers.items():
            if k < 2:
                continue
            for cand, last_idx, agg_p, agg_k, length, metrics in beam:
                # 在生成元素明细时再计算杠杆指标
                optimal_leverage, optimal_capital, no_leverage_capital = calc_leverage_metrics(agg_p, agg_k, length)

                merged_info = dict(
                    k=k,
                    weekly_loss_rate=metrics["weekly_loss_rate"],
                    active_week_ratio=metrics["active_week_ratio"],
                    weekly_net_profit_sum=metrics["weekly_net_profit_sum"],
                    weekly_net_profit_min=metrics["weekly_net_profit_min"],
                    weekly_net_profit_max=metrics["weekly_net_profit_max"],
                    weekly_net_profit_std=metrics["weekly_net_profit_std"],
                    optimal_leverage=optimal_leverage,
                    optimal_capital=optimal_capital,
                    capital_no_leverage=no_leverage_capital
                )

                # 取出该组合涉及的策略记录
                part = df.loc[orig_idx[list(cand)]].copy()

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