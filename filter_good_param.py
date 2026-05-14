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

    # 提取唯一的趋势窗口并排序，用于验证择时开关的敏感度
    btc_windows = sorted(df['param_BTC_TREND_WINDOW'].unique())

    for idx, row in top50_candidates.iterrows():
        current_btc = row['param_BTC_TREND_WINDOW']
        btc_idx = btc_windows.index(current_btc)

        # 提取当前大盘择时开关的 ±1 步长邻域
        neighbor_btc_vals = [current_btc]
        if btc_idx > 0: neighbor_btc_vals.append(btc_windows[btc_idx - 1])
        if btc_idx < len(btc_windows) - 1: neighbor_btc_vals.append(btc_windows[btc_idx + 1])

        # 构建邻域查询：严格锁定长短多空基因，只扰动 BTC 择时开关
        neighbor_mask = (
            df['param_BTC_TREND_WINDOW'].isin(neighbor_btc_vals) &
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
        trend_win = int(row.get('param_BTC_TREND_WINDOW', 0))

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
        print(f" 🧬 组合基因: 做多 [{long_param}] + 做空 [{short_param}] (趋势窗口: {trend_win})")
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


if __name__ == "__main__":
    # 将这里替换为你实际的 CSV 文件路径
    CSV_FILE_PATH = r"W:\project\python_project\oke_auto_trade\param_search_results\combined_metrics_results.csv"

    evaluate_and_print_top5(CSV_FILE_PATH)