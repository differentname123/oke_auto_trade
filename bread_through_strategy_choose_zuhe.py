import numpy as np
import pandas as pd
import time
from pathlib import Path


# -------------------- 指标计算函数，无改动部分 + 新增杠杆参数计算 -------------------- #
def calc_metrics_with_cache(agg_profit: np.ndarray,
                            agg_kai: np.ndarray,
                            cand_length: int,
                            n_weeks: int):
    """
    计算如下指标：
      - weekly_net_profit_sum: 活跃周（kai > 0）的利润求和（活跃周的平均收益再求和）
      - active_week_ratio: 活跃周比例
      - weekly_loss_rate: 活跃周中负收益周的比例
      - weekly_net_profit_min: 活跃周最低收益
      - weekly_net_profit_max: 活跃周最高收益
      - weekly_net_profit_std: 活跃周收益的标准差

    另外，新加入三个指标（基于 active_avg，即只对活跃周计算）：
      - optimal_leverage: 最优整数杠杆
      - optimal_capital: 在最优杠杆下累计收益率（初始本金为 1）
      - capital_no_leverage: 不加杠杆时的累计收益率
         （对于每笔收益，直接用 (1 + profit/100) 累乘）
    """
    # 平均收益 = 总收益 / 使用的交易周数（cand_length）
    avg = agg_profit / cand_length
    # 使用活跃周进行计算（kai > 0）
    active_mask = agg_kai > 0
    active_cnt = active_mask.sum()

    if active_cnt == 0:
        return dict(
            weekly_net_profit_sum=np.nan,
            active_week_ratio=np.nan,
            weekly_loss_rate=np.inf,
            weekly_net_profit_min=np.nan,
            weekly_net_profit_max=np.nan,
            weekly_net_profit_std=np.nan,
            optimal_leverage=np.nan,
            optimal_capital=np.nan,
            capital_no_leverage=np.nan
        )

    active_avg = avg[active_mask]

    # 计算基础指标
    sum_p = active_avg.sum()
    loss_rate = (active_avg < 0).mean()
    weekly_min = active_avg.min()
    weekly_max = active_avg.max()
    mean_val = sum_p / active_cnt
    std_val = np.sqrt(max((active_avg ** 2).mean() - mean_val ** 2, 0.0))

    # 计算不加杠杆的累计收益率（初始本金为 1）
    # 每笔收益直接累乘 1 + (profit/100)
    capital_no_leverage = np.prod(1 + active_avg / 100)

    # --------------------------------------------------------------------------
    # 计算最优杠杆
    # 模拟每笔交易时： capital *= 1 + (true_profit/100) * leverage
    # 为确保安全，任何交易后本金必须 > 0。
    # 先根据最差的一笔确定允许的最大杠杆：
    min_profit = active_avg.min()  # 最低收益（可能为负）
    if min_profit >= 0:
        max_possible_leverage = 20  # 无亏损时设默认上限
    else:
        # 必须满足 1 + (min_profit/100)*L > 0 ==> L < 1 / (abs(min_profit)/100)
        max_possible_leverage = int(1 / (abs(min_profit) / 100))

    optimal_leverage = None
    optimal_capital = -np.inf

    # 遍历 1 到最大安全杠杆，选择累计收益率最高的杠杆组合
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

    return dict(
        weekly_net_profit_sum=sum_p,
        active_week_ratio=active_cnt / n_weeks,
        weekly_loss_rate=loss_rate,
        weekly_net_profit_min=weekly_min,
        weekly_net_profit_max=weekly_max,
        weekly_net_profit_std=std_val,
        optimal_leverage=optimal_leverage,
        optimal_capital=optimal_capital,
        capital_no_leverage=capital_no_leverage
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
    inst_id_list = ['BTC', 'ETH', 'SOL', 'TON', 'DOGE', 'XRP', 'OKB']  # 可自行修改
    max_k = 50
    beam_width = 1_000
    objective = lambda m: m["weekly_loss_rate"]
    # 输出目录
    out_dir = Path("temp_back")
    out_dir.mkdir(parents=True, exist_ok=True)
    for inst in inst_id_list:
        elements_path = out_dir / f"result_elements_{inst}.parquet"
        if elements_path.exists():
            elements_df = pd.read_parquet(elements_path)
            elements_df['cha_score'] = elements_df['weekly_loss_rate_merged'] * elements_df['weekly_net_profit_min_merged'] * \
                                       elements_df['weekly_loss_rate_merged'] * elements_df['weekly_net_profit_min_merged']
            elements_df['score_score'] = elements_df['cha_score'] / elements_df['weekly_net_profit_sum_merged']
        print(f"\n==== 处理 {inst} ====")

        # 1) 读取 parquet 文件
        df_path = Path(f"temp_back/{inst}_True_all_filter_similar_strategy.parquet")
        if not df_path.exists():
            print(f"未找到文件 {df_path}，跳过该实例。")
            continue
        df = pd.read_parquet(df_path)
        print("策略条数：", len(df))

        # 2) 转 numpy 数组
        profit_mat = np.stack(df['weekly_net_profit_detail'].to_numpy()).astype(np.float32)
        kai_mat    = np.stack(df['weekly_kai_count_detail'].to_numpy()).astype(np.float32)
        orig_idx   = df.index.to_numpy()  # 保存原 dataframe 行索引

        # 3) Beam-Search
        t0 = time.time()
        layers = beam_search_multi_k(profit_mat, kai_mat,
                                     max_k=max_k, beam_width=beam_width,
                                     objective_func=objective)
        print(f"Beam-Search 用时 {time.time() - t0:.2f}s")

        # 4) 生成 elements_df
        elem_parts = []
        for k, beam in layers.items():
            if k < 2:  # k=1 的情况剔除掉
                continue
            for cand, *_ , metrics in beam:
                # 取出该组合涉及的策略行
                part = df.loc[orig_idx[list(cand)]].copy()

                # 为每条记录追加合并信息（merged 信息）
                merged_info = dict(
                    k=k,
                    weekly_loss_rate=metrics["weekly_loss_rate"],
                    active_week_ratio=metrics["active_week_ratio"],
                    weekly_net_profit_sum=metrics["weekly_net_profit_sum"],
                    weekly_net_profit_min=metrics["weekly_net_profit_min"],
                    weekly_net_profit_max=metrics["weekly_net_profit_max"],
                    weekly_net_profit_std=metrics["weekly_net_profit_std"],
                    optimal_leverage=metrics["optimal_leverage"],
                    optimal_capital=metrics["optimal_capital"],
                    capital_no_leverage=metrics["capital_no_leverage"]
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