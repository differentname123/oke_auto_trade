import os
import re

import pandas as pd
import numpy as np


def get_hard_filter_mask(df):
    """
    第一层：生死线硬过滤规则单独提取，方便后续在邻域平原检验中复用
    """
    return (
        (df['expectancy_ci_low'] > 0) &  # 95%置信下界期望必须为正
        (df['bootstrap_pnl_mean_p5'] > 0) &  # Bootstrap 最差5%情况必须盈利
        (df['cost_stress_20bps_annual'] > 0) &  # 增加双边40bps(单边20bps)摩擦后不能亏损
        (df['drop_top3_pnl_decay'] < 0.5) &  # 剔除最赚3笔后，利润缩水不能超过50%
        (df['drop_top5_pnl_decay'] < 0.6) &  # 【新增】剔除最赚5笔后缩水不能超过60%
        (df['asset_top1_share'] < 0.7) &  # 【新增】保留赢家通吃但拒绝单币赌注
        (df['negative_expectancy_assets'] <= 1) &  # 【新增】替代 HHI 的核心防御：基本盘不能流血
        (df['fee_to_pnl_ratio'] < 0.4) &  # 【新增】绝对生死线：拒绝向交易所打工
        (df['avg_recovery_time_days'] < 60) &  # 【新增】回撤修复速度
        (df['profitable_years_ratio'] >= 0.7) &  # 必须至少能在大部分年份存活
        # (df['max_time_under_water_days'] < 180) &  # 限制最长水下时间
        (df['max_drawdown'] > -0.70)  # 【放宽】避免误杀极端日恢复型策略
    )


def evaluate_and_print_top5(csv_path):
    print("正在加载回测数据并执行第一性原理过滤...\n")
    # 1. 读取数据
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ 找不到文件: {csv_path}，请检查路径。")
        return

    # 2. 第一层：生死线硬过滤 (直接淘汰脆弱、靠运气、扛不住滑点的拟合参数)
    mask = get_hard_filter_mask(df)
    filtered = df[mask].copy()

    if len(filtered) == 0:
        print("⚠️ 警告：没有任何参数组合通过基础的防过拟合硬性测试！")
        return

    # 3. 第二层：【体感 + 鲁棒】打分系统 (满分 100 分，采用分位数排名法)

    # --- 痛点/体感惩罚子项 (越小得分越高，所以 ascending=False) ---
    rank_ulcer = filtered['ulcer_index'].rank(pct=True, ascending=False)
    rank_underwater = filtered['max_time_under_water_days'].rank(pct=True, ascending=False)
    rank_loss_months = filtered['max_consecutive_losing_months'].rank(pct=True, ascending=False)

    # 【保留】25分：综合体感煎熬度惩罚
    filtered['score_pain'] = ((rank_ulcer + rank_underwater + rank_loss_months) / 3) * 25

    # 【降权与反向惩罚】8分：资金曲线平滑度
    base_stab_score = filtered['log_equity_r2'].rank(pct=True) * 8
    # 若 R² > 0.98，扣除 4 分作为反向惩罚，警惕过度拟合
    filtered['score_stab'] = np.where(filtered['log_equity_r2'] > 0.98, base_stab_score - 4, base_stab_score)

    # 【新增】18分：收益质量四联
    rank_sortino = filtered['sortino_ratio'].rank(pct=True)
    rank_tail = filtered['tail_ratio'].rank(pct=True)
    rank_pl = filtered['profit_loss_ratio'].rank(pct=True)
    rank_omega = filtered['omega_ratio'].rank(pct=True)
    filtered['score_quality'] = ((rank_sortino + rank_tail + rank_pl + rank_omega) / 4) * 18

    # 【新增】10分：最差重抽样期望兜底
    filtered['score_bootstrap'] = filtered['bootstrap_pnl_mean_p5'].rank(pct=True) * 10

    # 【新增】7分：回撤修复速度 (越小越好)
    filtered['score_recovery'] = filtered['avg_recovery_time_days'].rank(pct=True, ascending=False) * 7

    # 【新增】7分：滚动12个月最小收益
    filtered['score_roll_min'] = filtered['rolling_12m_return_min'].rank(pct=True) * 7

    # 【新增】5分：手续费占比 (越小越好)
    filtered['score_fee_effic'] = filtered['fee_to_pnl_ratio'].rank(pct=True, ascending=False) * 5

    # 【保留】15分：极限期望下界
    filtered['score_exp'] = filtered['expectancy_ci_low'].rank(pct=True) * 15

    # 【保留】15分：极限滑点抗性
    filtered['score_cost'] = filtered['cost_stress_20bps_annual'].rank(pct=True) * 15

    # 计算综合总分 (总权重大致110分制)
    filtered['robust_score'] = (
            filtered['score_pain'] +
            filtered['score_stab'] +
            filtered['score_quality'] +
            filtered['score_bootstrap'] +
            filtered['score_recovery'] +
            filtered['score_roll_min'] +
            filtered['score_fee_effic'] +
            filtered['score_exp'] +
            filtered['score_cost']
    )

    # 4. 第三层：邻域平原检验 (择时开关抖动测试 Gate Jitter Test)
    top50_candidates = filtered.sort_values('robust_score', ascending=False).head(50)
    valid_candidates = []

    # 🔴 [修改点 1] 提取多头和空头各自唯一的趋势窗口并排序
    long_btc_windows = sorted(df['long_BTC_TREND_WINDOW'].dropna().unique())
    short_btc_windows = sorted(df['short_BTC_TREND_WINDOW'].dropna().unique())

    for idx, row in top50_candidates.iterrows():
        current_long_btc = row['long_BTC_TREND_WINDOW']
        current_short_btc = row['short_BTC_TREND_WINDOW']

        long_idx = long_btc_windows.index(current_long_btc)
        short_idx = short_btc_windows.index(current_short_btc)

        # 🔴 [修改点 2] 分别提取多头和空头大盘择时开关的 ±1 步长邻域
        neighbor_long_vals = [current_long_btc]
        if long_idx > 0: neighbor_long_vals.append(long_btc_windows[long_idx - 1])
        if long_idx < len(long_btc_windows) - 1: neighbor_long_vals.append(long_btc_windows[long_idx + 1])

        neighbor_short_vals = [current_short_btc]
        if short_idx > 0: neighbor_short_vals.append(short_btc_windows[short_idx - 1])
        if short_idx < len(short_btc_windows) - 1: neighbor_short_vals.append(short_btc_windows[short_idx + 1])

        # 🔴 [修改点 3] 构建二维邻域查询：锁定其他参数基因，分别扰动多空 BTC 择时开关
        neighbor_mask = (
                df['long_BTC_TREND_WINDOW'].isin(neighbor_long_vals) &
                df['short_BTC_TREND_WINDOW'].isin(neighbor_short_vals) &
                (df['long_param_name'] == row['long_param_name']) &
                (df['short_param_name'] == row['short_param_name'])
        )
        neighbors = df[neighbor_mask]

        if len(neighbors) == 0:
            continue

        # 检验指标 1：硬过滤通过率 ≥ 60%
        hard_pass_rate = get_hard_filter_mask(neighbors).mean()

        # 检验指标 2：expectancy_ci_low 为正的比例 ≥ 70%
        exp_pos_rate = (neighbors['expectancy_ci_low'] > 0).mean()

        # 检验指标 3：ulcer_index 中位数相对中心的恶化幅度 ≤ 40%
        center_ulcer = row['ulcer_index']
        median_ulcer = neighbors['ulcer_index'].median()
        degradation = (median_ulcer - center_ulcer) / center_ulcer if center_ulcer > 0 else 0

        if hard_pass_rate >= 0.60 and exp_pos_rate >= 0.70 and degradation <= 0.40:
            valid_candidates.append(row)

    if not valid_candidates:
        print("⚠️ 警告：Top 50 参数全部未能通过【择时开关平原检验】，它们对历史拐点极度敏感！建议重新评估策略逻辑。")
        return

    valid_candidates_df = pd.DataFrame(valid_candidates)

    # 5. 第四层：提取最终入围者（取消复杂聚类，直接提取真金不怕火炼的 Top 5）
    top5 = valid_candidates_df.head(5)

    print(f"✅ 筛选完成！从 {len(df)} 组数据中提取出【实盘体感与抗压综合最优】的前 {len(top5)} 名：\n")
    print("=" * 85)

    # 6. 格式化输出 (保持原有格式不变)
    medals = ['🥇', '🥈', '🥉', '🏅', '🏅']

    for i, (_, row) in enumerate(top5.iterrows()):
        # 基础字段提取
        rank_medal = medals[i] if i < len(medals) else '🏅'
        score = row['robust_score']
        long_param = row.get('long_param_name', 'Unknown')
        short_param = row.get('short_param_name', 'Unknown')

        # 🔴 [修改点 4] 提取分别的长短趋势窗口
        long_trend_win = int(row.get('long_BTC_TREND_WINDOW', 0))
        short_trend_win = int(row.get('short_BTC_TREND_WINDOW', 0))

        # 核心指标提取
        exp_low = row.get('expectancy_ci_low', 0)
        total_ret = row.get('total_return', 0)
        r2 = row.get('log_equity_r2', 0)
        ret_2022 = row.get('year_2022_return', 0)
        ulcer = row.get('ulcer_index', 0)
        consecutive_losing = int(row.get('max_consecutive_losing_months', 0))
        stress_20 = row.get('cost_stress_20bps_annual', 0)
        fee_ratio = row.get('fee_to_pnl_ratio', 0)
        drop_top3 = row.get('drop_top3_pnl_decay', 0)
        neg_assets = int(row.get('negative_expectancy_assets', 0))

        # 打印输出卡片
        print(f" {rank_medal} 排名: 第 {i + 1} 名 | 综合体感分: {score:.2f} (侧重无痛执行与防拟合)")
        # 🔴 [修改点 5] 分开展示长短窗口
        print(f" 🧬 组合基因: 做多 [{long_param}](趋势:{long_trend_win}) + 做空 [{short_param}](趋势:{short_trend_win})")
        print(
            f"   ├─ 交易核心: {int(row['total_closed_trades'])}笔平仓 | 胜率: {row['win_rate'] * 100:.1f}% | 盈亏比: {row['profit_loss_ratio']:.2f} | 净期望下界: {exp_low:.4f}")
        print(
            f"   ├─ 收益状况: 总收益: +{total_ret * 100:.1f}% | 年化: +{row['annual_return'] * 100:.1f}% | 资金拟合度(R²): {r2:.2f} | 22年熊市: {ret_2022 * 100:+.1f}%")
        print(
            f"   ├─ 回撤体验: 最大回撤: {row['max_drawdown'] * 100:.1f}% | 最长水下: {row['max_time_under_water_days']:.0f}天 | 最长连亏: {consecutive_losing}个月 | 溃疡指数: {ulcer:.1f}")
        print(f"   ├─ 极限压力: 20bps滑点后年化: +{stress_20 * 100:.1f}% | 手续费占毛利比: {fee_ratio * 100:.1f}%")
        print(
            f"   └─ 运气剥离: 币种HHI: {row['asset_hhi']:.3f} | 去除最赚3笔后衰减: {drop_top3 * 100:.1f}% | 负期望币种数: {neg_assets} 个")
        print("-" * 85)
def print_advanced_report(metrics_dict):
    """
    高逼格、高信息密度的策略表现深度打印函数
    支持动态解析不固定数量的年份和币种
    """

    # 辅助函数：安全获取并处理潜在的 NaN 值
    def safe_get(key, default=0.0):
        val = metrics_dict.get(key, default)
        if pd.isna(val) or val is None:
            return default
        return val

    # ==========================================
    # 1. 提取策略参数
    # ==========================================
    mom = safe_get('param_MOM_WINDOW', 0)
    vol = safe_get('param_VOL_WINDOW', 0)
    btc_win = safe_get('param_BTC_TREND_WINDOW', 0)
    max_w = safe_get('param_MAX_WEIGHT', 0.0)
    top_k = safe_get('param_TOP_K', 0)

    print("\n [⚙️ 策略参数配置]")
    print(
        f"   ► MOM_WINDOW: {mom},  VOL_WINDOW: {vol},  BTC_TREND_WINDOW: {btc_win},  MAX_WEIGHT: {max_w},  TOP_K: {top_k}\n")

    # ==========================================
    # 2. 提取并计算核心绩效
    # ==========================================
    trades = int(safe_get('total_closed_trades', 0))
    win_rate = safe_get('win_rate', 0.0) * 100
    pl_ratio = safe_get('profit_loss_ratio', 0.0)
    exp_ci = safe_get('expectancy_ci_low', 0.0)

    tot_ret = safe_get('total_return', 0.0) * 100
    ann_ret = safe_get('annual_return', 0.0) * 100

    bull_ret = safe_get('bull_regime_total_return', 0.0) * 100
    bull_bars = int(safe_get('bull_regime_bars', 0))
    bear_ret = safe_get('bear_regime_total_return', 0.0) * 100
    bear_bars = int(safe_get('bear_regime_bars', 0))

    max_dd = safe_get('max_drawdown', 0.0) * 100
    max_uw = safe_get('max_time_under_water_days', 0.0)
    days_passed = safe_get('days_passed', 1)
    uw_ratio = (max_uw / days_passed) * 100 if days_passed > 0 else 0.0

    # 主目标得分 (如果有 composite_score 则用，否则用 sortino_ratio 代替)
    final_score = safe_get('composite_score', safe_get('sortino_ratio', 0.0))
    monthly_win = safe_get('monthly_positive_ratio', 0.0) * 100

    hhi = safe_get('asset_hhi', 0.0)
    top1_pnl = safe_get('top1_pnl_ratio', 0.0) * 100
    top3_decay = safe_get('drop_top3_pnl_decay', 0.0) * 100

    print(" [📊 核心基础绩效 (基于最佳时区表现)]")
    print(f"   ├─ 交易统计: {trades} 笔平仓 | 胜率: {win_rate:.1f}% | 盈亏比: {pl_ratio:.2f} | 期望下界: {exp_ci:.3f}")
    print(f"   ├─ 收益状况: 总收益: {tot_ret:+.1f}% | 年化收益: {ann_ret:+.1f}%")
    print(
        f"   ├─ 周期拆解: 牛市绝对收益: {bull_ret:+.1f}% (历经 {bull_bars} 根K线) | 熊市绝对收益: {bear_ret:+.1f}% (历经 {bear_bars} 根K线)")
    print(f"   ├─ 回撤体验: 最大回撤: {max_dd:.1f}% | 全局最长水下: {max_uw:.1f}天 (占总时长 {uw_ratio:.1f}%)")
    print(f"   ├─ 风险调整: FINAL_SCORE(主目标): {final_score:.3f} | 盈利月占比: {monthly_win:.1f}%")
    print(f"   └─ 集中度险: 币种HHI: {hhi:.3f} | 最赚1笔占比: {top1_pnl:.1f}% | Top3剔除衰减: {top3_decay:+.1f}%\n")

    # ==========================================
    # 3. 动态解析年度表现拆解
    # ==========================================
    print(" [📅 年度表现拆解]")
    # 正则提取所有含有 year_xxxx_return 的键，从而动态支持任意年份区间
    years = sorted(list(set([
        int(re.search(r'year_(\d{4})_return', k).group(1))
        for k in metrics_dict.keys() if re.match(r'year_\d{4}_return', k)
    ])))

    for y in years:
        y_ret = safe_get(f'year_{y}_return', 0.0) * 100
        y_dd = safe_get(f'year_{y}_max_dd', 0.0) * 100
        b_ret = safe_get(f'benchmark_year_{y}_return', 0.0) * 100
        # 如果 csv 中有全局超额就拿，没有就直接减出来
        exc = safe_get(f'year_{y}_excess_return', y_ret / 100 - b_ret / 100) * 100

        print(
            f"   ► {y}年: 策略收益: {y_ret:>7.1f}% (最大回撤 {y_dd:>6.1f}%) | 基准收益: {b_ret:>+6.1f}% | 🌟全局超额: {exc:>+6.1f}%")

    # ==========================================
    # 4. 动态解析标的盈亏贡献明细
    # ==========================================
    print("\n [🪙 标的盈亏贡献明细 (按净利润排序)]")
    assets_data = []
    # 正则提取所有的币种名称 asset_XXX_net_pnl
    for k in metrics_dict.keys():
        match = re.match(r'asset_([A-Z]+)_net_pnl', k)
        if match:
            coin = match.group(1)
            net_pnl = safe_get(f'asset_{coin}_net_pnl', 0.0)
            pnl_share = safe_get(f'asset_{coin}_pnl_share', 0.0) * 100
            trades_c = int(safe_get(f'asset_{coin}_trades', 0))
            win_rate_c = safe_get(f'asset_{coin}_win_rate', 0.0) * 100
            assets_data.append((coin, net_pnl, pnl_share, trades_c, win_rate_c))

    # 按净利润从高到低排序
    assets_data.sort(key=lambda x: x[1], reverse=True)

    for coin, pnl, share, trd, wr in assets_data:
        if trd > 0:  # 只有实际产生过交易的币种才展示
            print(
                f"   - {coin:<5} : 净利润 $ {pnl:>7.2f} (利润占比: {share:>4.1f}% ) | 交易: {trd:>3}笔 | 胜率: {wr:.1f}%")

    print("▲" * 78)

if __name__ == "__main__":
    result_csv =   r'W:\project\python_project\oke_auto_trade\param_search_results\grid_search_131274_LONG_ONLY_dynamic_pool_offset_0h_with_Benchmark.csv'
    df1 = pd.read_csv(r'W:\project\python_project\oke_auto_trade\param_search_results\event_streams\LONG_ONLY_Grid_No.1_2h_events.csv')


    if os.path.exists(result_csv):
        df_results = pd.read_csv(result_csv)
        if not df_results.empty:

            # 找到 param_name 为 Grid_No.31396_1h的行
            best_row = df_results[df_results['param_name'] == 'Grid_No.33097_0h'].iloc[0]

            # # 假设你想看 sortino_ratio 排名第一的策略结果
            # best_row = df_results.sort_values(by='sortino_ratio', ascending=False).iloc[0]

            # 把 Series 转为字典喂给高级打印函数
            best_metrics_dict = best_row.to_dict()

            print_advanced_report(best_metrics_dict)
    else:
        print("未找到结果文件，请确保已经执行完带有 Benchmark 的回测流程。")


    result_csv =   r'W:\project\python_project\oke_auto_trade\param_search_results\grid_search_131274_SHORT_ONLY_dynamic_pool_offset_0h_with_Benchmark.csv'
    df1 = pd.read_csv(r'W:\project\python_project\oke_auto_trade\param_search_results\event_streams\LONG_ONLY_Grid_No.1_2h_events.csv')


    if os.path.exists(result_csv):
        df_results = pd.read_csv(result_csv)
        if not df_results.empty:

            # 找到 param_name 为 Grid_No.31396_1h的行
            best_row = df_results[df_results['param_name'] == 'Grid_No.69393_0h'].iloc[0]

            # # 假设你想看 sortino_ratio 排名第一的策略结果
            # best_row = df_results.sort_values(by='sortino_ratio', ascending=False).iloc[0]

            # 把 Series 转为字典喂给高级打印函数
            best_metrics_dict = best_row.to_dict()

            print_advanced_report(best_metrics_dict)
    else:
        print("未找到结果文件，请确保已经执行完带有 Benchmark 的回测流程。")



    # # 将这里替换为你实际的 CSV 文件路径
    CSV_FILE_PATH = r"W:\project\python_project\oke_auto_trade\param_search_results\combined_metrics_results_op.csv"

    evaluate_and_print_top5(CSV_FILE_PATH)
