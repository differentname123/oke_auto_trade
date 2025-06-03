import numpy as np
import pandas as pd
import time
from pathlib import Path
import heapq
import gc


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

    注意：杠杆相关指标将在后续生成元素统计结果时再计算。
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

    计算步骤：
      1. 计算平均收益数组 active_avg = (agg_profit / cand_length)[agg_kai>0]
      2. 计算不加杠杆的累计收益率：逐笔累乘 (1 + profit/100)
      3. 根据最差的一笔确定允许的最大杠杆：
            若 min_profit >= 0，则默认上限为 30，
            否则需满足 1 + (min_profit/100)*L > 0  ==> L < 1 / (|min_profit|/100)
      4. 模拟遍历 1 到 max_possible_leverage，选择使累计资本最大的杠杆。

    优化点：利用 numpy 的广播与矩阵运算，避免内层循环。
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
    # 利用 np.outer 构造一个二维数组，每行对应一个杠杆率下所有活跃周的因子
    factors = 1 + np.outer(L_values, active_avg) / 100  # 形状 (max_possible_leverage, len(active_avg))
    # 判断每个杠杆下是否所有因子均 > 0
    safe = np.all(factors > 0, axis=1)
    # 对不安全的杠杆设置为 -np.inf
    capitals = np.where(safe, np.prod(factors, axis=1), -np.inf)
    optimal_index = np.argmax(capitals)
    optimal_leverage = int(L_values[optimal_index])
    optimal_capital = capitals[optimal_index]

    return optimal_leverage, optimal_capital, capital_no_leverage


# -------------------- 辅助函数：基于索引计算聚合指标 -------------------- #
def calc_metrics_by_indices(cand: tuple, profit_mat: np.ndarray, kai_mat: np.ndarray, n_weeks: int):
    """
    根据候选组合的索引计算累计的利润与开仓数组，然后调用 calc_metrics_with_cache 计算指标。
    """
    # 将候选组合中的各行数据求和
    agg_profit = np.sum(profit_mat[list(cand)], axis=0)
    agg_kai = np.sum(kai_mat[list(cand)], axis=0)
    return calc_metrics_with_cache(agg_profit, agg_kai, len(cand), n_weeks)


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
                        objective_func=lambda m: m["weekly_loss_rate"]):
    """
    使用 Beam-Search 在不保存中间大数组的条件下搜索候选组合，
    仅记录候选组合中策略的索引元组。指标计算时基于原始矩阵聚合计算。
    """
    n_strategies, n_weeks = profit_mat.shape

    # k = 1 时的初始候选，仅记录索引及对应指标（通过延迟聚合计算）
    beam = []
    for i in range(n_strategies):
        cand = (i,)
        metrics = calc_metrics_by_indices(cand, profit_mat, kai_mat, n_weeks)
        beam.append((cand, i, metrics))
    beam = heapq.nsmallest(beam_width, beam, key=lambda x: objective_func(x[2]))
    layer_results = {1: beam}

    # 逐层扩展，生成 k=2,3,... 的组合
    for _ in range(1, max_k):
        new_beam = []
        for cand, last_idx, _ in beam:
            next_length = len(cand) + 1
            for nxt in range(last_idx + 1, n_strategies):
                new_cand = cand + (nxt,)
                new_metrics = calc_metrics_by_indices(new_cand, profit_mat, kai_mat, n_weeks)
                new_beam.append((new_cand, nxt, new_metrics))
        if not new_beam:
            break
        beam = heapq.nsmallest(beam_width, new_beam, key=lambda x: objective_func(x[2]))
        layer_results[len(beam[0][0])] = beam  # 当前候选长度
        # 主动释放内存，减少中间层占用
        gc.collect()

    return layer_results


# -------------------- 主流程 -------------------- #
def choose_zuhe_beam_opt():
    # 定义实例列表，这里可以根据实际需要进行修改
    inst_id_list = ['BTC', 'ETH', 'SOL', 'TON', 'DOGE', 'XRP', 'OKB']
    max_k = 100
    beam_width = 100000  # beam_width 数值较大时可能占用较多内存，根据实际场景调整
    objective = lambda m: m["weekly_loss_rate"]  # Beam-Search 目标函数

    out_dir = Path("temp_back")
    out_dir.mkdir(parents=True, exist_ok=True)

    for inst in inst_id_list:
        elements_path = out_dir / f"result_elements_{inst}.parquet"
        print(f"\n==== 处理 {inst} ====")

        df_path = Path(f"temp_back/{inst}_True_all_filter_similar_strategy.parquet")
        if not df_path.exists():
            print(f"未找到文件 {df_path}，跳过该实例。")
            continue
        df = pd.read_parquet(df_path)
        # 获取weekly_net_profit_detail的元素个数
        weeks = len(df['weekly_net_profit_detail'].iloc[0]) if not df.empty else 0
        print("每个策略的周数：", weeks)
        print("策略条数：", len(df))

        # 从 df 中提取 profit 和 kai 的详细数据（注意这些字段为列表形式）
        profit_mat = np.stack(df['weekly_net_profit_detail'].to_numpy()).astype(np.float32)
        kai_mat = np.stack(df['weekly_kai_count_detail'].to_numpy()).astype(np.float32)

        t0 = time.time()
        layers = beam_search_multi_k(profit_mat, kai_mat,
                                     max_k=max_k, beam_width=beam_width,
                                     objective_func=objective)
        print(f"Beam-Search 用时 {time.time() - t0:.2f}s")

        # -------------------- 聚合结果：仅保存统计指标 -------------------- #
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
                    "capital_no_leverage": no_leverage_capital
                }
                # 保存候选策略的索引（原始策略在 profit_mat 中的下标）
                result_item = {"strategies": cand}
                # 为了与原来版本保持一致，这里将所有统计字段名称均添加 _merged 后缀
                for col, val in merged_info.items():
                    result_item[col + "_merged"] = val

                results.append(result_item)

        if not results:
            print(f"{inst} 未产生任何组合结果。")
            continue

        # 将聚合结果转换为 DataFrame
        elements_df = pd.DataFrame(results)


        elements_df['cha_score'] = elements_df['weekly_loss_rate_merged'] * elements_df['weekly_net_profit_min_merged'] * weeks / 2
        elements_df['score_score'] = elements_df['cha_score'] / elements_df['weekly_net_profit_sum_merged']
        elements_df['score_score1'] = elements_df['weekly_net_profit_sum_merged'] / elements_df['weekly_loss_rate_merged']

        elements_df.to_parquet(elements_path, index=False)
        print(f"统计结果已写入 {elements_path}（{len(elements_df)} 行）")


if __name__ == "__main__":
    choose_zuhe_beam_opt()