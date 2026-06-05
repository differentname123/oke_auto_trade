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


def get_failure_statistics(df, dead_mask):
    """
    【新增】统计在硬过滤中被淘汰的样本，具体是因为哪些指标不达标被淘汰的。
    注意：一个样本可能同时触发多个淘汰条件，因此各项淘汰计数之和可能大于总淘汰人数。
    """
    dead_df = df[dead_mask]
    if dead_df.empty:
        return {}

    stats = {}

    # 精确镜像 get_hard_filter_mask 的条件，进行反向统计
    c1 = ~(dead_df['expectancy_ci_low'] > 0)
    if c1.sum() > 0: stats['期望为负'] = int(c1.sum())

    c2 = ~(dead_df['bootstrap_pnl_mean_p5'] > 0)
    if c2.sum() > 0: stats['最差5%亏损'] = int(c2.sum())

    c3 = ~(dead_df['cost_stress_20bps_annual'] > 0)
    if c3.sum() > 0: stats['滑点后亏损'] = int(c3.sum())

    c4 = ~(dead_df['drop_top3_pnl_decay'] < 0.5)
    if c4.sum() > 0: stats['Top3缩水>50%'] = int(c4.sum())

    c5 = ~(dead_df['drop_top5_pnl_decay'] < 0.6)
    if c5.sum() > 0: stats['Top5缩水>60%'] = int(c5.sum())

    c6 = ~(dead_df['asset_top1_share'] < 0.7)
    if c6.sum() > 0: stats['单币赌注过重'] = int(c6.sum())

    c7 = ~(dead_df['negative_expectancy_assets'] <= 1)
    if c7.sum() > 0: stats['负期望币>1个'] = int(c7.sum())

    c8 = ~(dead_df['fee_to_pnl_ratio'] < 0.4)
    if c8.sum() > 0: stats['手续费率>=40%'] = int(c8.sum())

    c9 = ~(dead_df['avg_recovery_time_days'] < 60)
    if c9.sum() > 0: stats['修复期>=60天'] = int(c9.sum())

    c10 = ~(dead_df['profitable_years_ratio'] >= 0.7)
    if c10.sum() > 0: stats['多数年份亏损'] = int(c10.sum())

    c11 = ~(dead_df['max_drawdown'] > -0.70)
    if c11.sum() > 0: stats['最大回撤超70%'] = int(c11.sum())

    # 按照淘汰数量从大到小排序，方便直观查看核心死因
    return dict(sorted(stats.items(), key=lambda item: item[1], reverse=True))


def count_inconsistent_rows(df: pd.DataFrame) -> int:
    # 修正后的正则表达式：匹配数字+单位，且后面紧跟下划线或处于字符串末尾
    pattern = re.compile(r"(\d+(?:min|h|d))(?=_|$)")

    def is_inconsistent(text):
        offsets = pattern.findall(str(text))
        # 如果提取到的 offset 种类大于 1，则说明不一致
        return len(set(offsets)) > 1

    return int(df["param_name"].apply(is_inconsistent).sum())


def filter_different_offsets(df: pd.DataFrame) -> pd.DataFrame:
    pattern = re.compile(r"(\d+(?:min|h|d))(?=_|$)")

    def is_inconsistent(text):
        offsets = pattern.findall(str(text))
        return len(set(offsets)) > 1

    total_rows = len(df)
    filtered_df = df[df["param_name"].apply(is_inconsistent)].copy()
    kept_rows = len(filtered_df)

    # 打印统计信息
    print(f"原始数据共 {total_rows} 行，过滤后保留了 {kept_rows} 行（offset不一致）。")

    return filtered_df


from typing import Tuple, List, Optional


def get_true_neighbor_cores(raw_df: pd.DataFrame, core_id: str, tolerance_pct=0.30) -> Tuple[
    List[str], Optional[float]]:
    """
    根据阈值找到所有参数均在合理范围内的邻居，并返回邻居列表以及最大复合权重乘积。

    返回:
        - 满足条件的 core_id 列表 (List[str])
        - 满足条件的邻居中，param_MAX_WEIGHT * param_TOP_K 的最大值 (float 或 None)
    """
    # 1. 异常处理：原表为空
    if raw_df is None or raw_df.empty:
        return [core_id], None

    # 2. 异常处理：找不到目标 core_id
    target_rows = raw_df[raw_df['param_name'] == core_id]
    if target_rows.empty:
        return [core_id], None

    row = target_rows.iloc[0]

    # 初始化掩码，默认全表符合
    mask = pd.Series(True, index=raw_df.index)

    # 动态遍历所有底层参数列 (所有以 param_ 开头且数值化的字段)
    for col in raw_df.columns:
        if col.startswith('param_') and col != 'param_name':
            if pd.api.types.is_numeric_dtype(raw_df[col]):
                val = row[col]
                if pd.notna(val):
                    margin = abs(val) * tolerance_pct
                    if margin == 0:
                        mask &= (raw_df[col] == val)
                    else:
                        mask &= (abs(raw_df[col] - val) <= margin + 1e-9)

    # 3. 提取筛选后的结果
    filtered_df = raw_df[mask]

    # 4. 计算最大乘积
    if filtered_df.empty:
        max_product = None
    else:
        # 🔴 核心改动：向量化计算两个字段的乘积，并取最大值
        # 即使这两个列中有 NaN，Pandas 的乘积和 max 也会自动跳过 NaN（除非全都是 NaN，此时 max 会返回 NaN）
        product_series = filtered_df['param_MAX_WEIGHT'] * filtered_df['param_TOP_K']
        max_product = product_series.max()

        # 如果计算结果是 NaN（比如对应的行数据缺失），将其转换为 None 更好处理
        if pd.isna(max_product):
            max_product = None

    # 返回：满足条件的列表，以及最大乘积值
    return filtered_df['param_name'].tolist(), max_product


def evaluate_and_print_top5(csv_path, raw_dfs_dict=None):
    print("正在加载回测数据并执行家族化第一性原理过滤...\n")
    # 1. 读取数据
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ 找不到文件: {csv_path}，请检查路径。")
        return

    result = count_inconsistent_rows(df)
    print(
        f"🔍 数据完整性检查：共有 {result} 行存在参数命名不一致问题（如同时包含 '1h' 和 '2h'）。建议修正这些行以确保数据质量。\n")

    # =========================================================================================
    # 🛠️ 第一阶段：基因提取与家族分组（Data Grouping）
    # =========================================================================================

    # 辅助正则函数：剥离时间后缀与提取时间后缀
    def strip_time_suffix(name):
        return re.sub(r'_\d+(min|h|d)$', '', str(name))

    def get_time_suffix(name):
        match = re.search(r'_(\d+(min|h|d))$', str(name))
        return match.group(1) if match else '0h'

    # 解析宏观核心参数与时间偏移
    df['core_long'] = df['long_param_name'].apply(strip_time_suffix)
    df['core_short'] = df['short_param_name'].apply(strip_time_suffix)
    df['offset_long'] = df['long_param_name'].apply(get_time_suffix)
    df['offset_short'] = df['short_param_name'].apply(get_time_suffix)

    # 生成唯一家族 ID：策略类型 + 核心长维参数 + 策略类型 + 核心短维参数
    def make_unified_family_id(row):
        part1 = f"{row['strat_A_type']}_{row['core_long']}"
        part2 = f"{row['strat_B_type']}_{row['core_short']}"

        # 🔴【修复】使用 frozenset，天然无序且可哈希，是最完美、可扩展的分组键值对
        return "_AND_".join(sorted(frozenset([part1, part2])))

    # 应用统一化 ID
    df['Family_ID'] = df.apply(make_unified_family_id, axis=1)

    family_sizes = df.groupby('Family_ID').size()

    # 2. 统计这些数量各自出现了多少次，并按数量从大到小排序
    size_distribution = family_sizes.value_counts().sort_index(ascending=False)

    print("\n" + "=" * 50)
    print(" 📊 家族成员数量分布概览")
    print("=" * 50)

    for size, count in size_distribution.items():
        if size == 25:
            print(f" ✅ [健康] 满编 25 个成员的家族 : {count:4d} 个")
        elif size == 15:
            print(f" ⚠️ [撕裂] 只有 15 个成员的家族 : {count:4d} 个 (A_B 顺向截断)")
        elif size == 10:
            print(f" ⚠️ [撕裂] 只有 10 个成员的家族 : {count:4d} 个 (B_A 逆向截断)")
        else:
            print(f" ❓ [异常] 包含 {size:2d} 个成员的家族 : {count:4d} 个")

    print("=" * 50 + "\n")

    # =========================================================================================
    # 🔬 第二阶段：单体独立质检与打分（奥卡姆剃刀降维版）
    # =========================================================================================

    # 1. 标记生存状态
    df['is_alive'] = get_hard_filter_mask(df)

    # 2. 剥离最好年份后的生存力 (提取为独立变量，修复单年 Bug)
    year_cols = [c for c in df.columns if c.startswith('year_') and c.endswith('_return')]
    if len(year_cols) > 1:
        yearly_returns = df[year_cols]
        rest_years_mean = (yearly_returns.sum(axis=1) - yearly_returns.max(axis=1)) / (len(year_cols) - 1)
        score_no_best_year_rank = rest_years_mean.rank(pct=True)
    elif len(year_cols) == 1:
        # 🔴【修复】只有一年数据时，直接用该年排名，但权重打对折作为时间不够的惩罚
        yearly_returns = df[year_cols]
        rest_years_mean = yearly_returns.iloc[:, 0]
        score_no_best_year_rank = rest_years_mean.rank(pct=True) * 0.5
    else:
        score_no_best_year_rank = pd.Series(0, index=df.index)
        rest_years_mean = pd.Series(0, index=df.index)

    df['rest_years_annual_mean'] = rest_years_mean

    # 🔴【重构】只用 3 个绝对正交的物理维度打分，干掉所有多重共线性的冗余

    # 维度 1：收益风险比效率 (35分) - 仅用 sortino_ratio 涵盖所有平滑度、回撤与水下痛点
    rank_efficiency = df['sortino_ratio'].rank(pct=True) * 35

    # 维度 2：周期跨越防线 (35分) - 仅用 滚动12月下限 + 剥离最强年，衡量抗周期能力
    rank_time_robust = (df['rolling_12m_return_min'].rank(pct=True) + score_no_best_year_rank) / 2 * 35

    # 维度 3：极限压力防线 (30分) - 仅用 滑点施压 + Bootstrap重抽样兜底，衡量抗微观物理摩擦能力
    rank_stress = (df['cost_stress_20bps_annual'].rank(pct=True) + df['bootstrap_pnl_mean_p5'].rank(pct=True)) / 2 * 30

    # 计算全局稳健总分 (纯粹的第一性原理)
    df['robust_score'] = rank_efficiency + rank_time_robust + rank_stress

    # =========================================================================================
    # 🛡️ 第三阶段：双层家族评估体系 (均值惩罚模式)
    # =========================================================================================

    family_results = []

    for family_id, group in df.groupby('Family_ID'):
        total_members = len(group)
        alive_count = group['is_alive'].sum()
        survival_rate = alive_count / total_members if total_members > 0 else 0

        # 【新增】获取本家族淘汰成员的具体原因统计
        family_dead_mask = ~group['is_alive']
        family_fail_stats = get_failure_statistics(group, family_dead_mask)

        # 第一层：外围抗扰动质检 (生存率必须 >= 80%)
        if survival_rate < 0.8:
            continue

        # 🔴【修复】第二层：提取实盘同源分身组合，且要求它本身必须是存活的！
        homo_group = group[(group['offset_long'] == group['offset_short']) & (group['is_alive'])]
        homo_group = group[(group['is_alive'])]

        if len(homo_group) == 0:
            continue

        # 同源单体核心打分 (抛弃双重暴击底线，改用夏普思维：族群均值 - 方差惩罚)
        homo_scores = homo_group['robust_score']

        mean_score = homo_scores.mean()
        std_score = homo_scores.std(ddof=0) if len(homo_scores) > 1 else 0

        penalty_factor = 1.0  # 均值模式下系数调整为 1.0
        family_final_score = mean_score - (penalty_factor * std_score)

        # 为报告准备输出字段：锁定最差同源分身作为实盘保底预期
        worst_homo_row = homo_group.loc[homo_scores.idxmax()]
        yield_spread = homo_group['annual_return'].max() - homo_group['annual_return'].min()

        family_results.append({
            'Family_ID': family_id,
            'survival_rate': survival_rate,
            'alive_count': alive_count,
            'total_members': total_members,
            'family_fail_stats': family_fail_stats,  # 【记录家族淘汰明细】
            'family_final_score': family_final_score,
            'worst_homo_row': worst_homo_row,
            'yield_spread': yield_spread,
            'annual_return_max': round(homo_group['annual_return'].max() * 100, 2),
            'annual_return_min': round(homo_group['annual_return'].min() * 100, 2)
        })

    if not family_results:
        print("⚠️ 警告：没有任何参数家族通过 80% 生存率的【外围抗扰动测试】！请检查数据或放宽初筛标准。")
        return

    # =========================================================================================
    # 🌍 第四阶段：家族群落参数平原检验 (Neighborhood Flatland Test) - 🔴 彻底重构
    # =========================================================================================

    valid_families = []
    family_df_temp = pd.DataFrame(family_results)

    for _, f_row in family_df_temp.iterrows():
        worst_row = f_row['worst_homo_row']

        strat_a = worst_row.get('strat_A_type')
        strat_b = worst_row.get('strat_B_type')
        core_long = worst_row.get('core_long')
        core_short = worst_row.get('core_short')

        if raw_dfs_dict is not None:
            raw_df_a = raw_dfs_dict.get(strat_a)
            raw_df_b = raw_dfs_dict.get(strat_b)

            # 🔴 1. 严格根据阈值找到各自的真实邻居 (所有参数均在合理浮动范围内)
            valid_cores_a, max_product_a = get_true_neighbor_cores(raw_df_a, core_long, tolerance_pct=0.30)

            valid_cores_b, max_product_b = get_true_neighbor_cores(raw_df_b, core_short, tolerance_pct=0.30)
            if max_product_a + max_product_b > 1:
                continue
            f_row['max_money_ratio'] = max_product_a + max_product_b
            # if len(valid_cores_a) > 0:
            #     print()
            # else:
            #     print()
        else:
            # 兼容未传入原始数据表的情况
            valid_cores_a = [core_long]
            valid_cores_b = [core_short]

        # 🔴 2. 完美匹配策略：两两组合产生出所有邻居候选者的统一 Family_ID
        # 彻底解决 A+B 和 B+A 的同源镜像匹配问题
        neighbor_family_ids = set()
        for ca in valid_cores_a:
            for cb in valid_cores_b:
                part1 = f"{strat_a}_{ca}"
                part2 = f"{strat_b}_{cb}"
                unified_id = "_AND_".join(sorted(frozenset([part1, part2])))
                neighbor_family_ids.add(unified_id)

        # 🔴 3. 根据统一 ID 到大池子里精确捕鱼，绝不越界串台
        neighborhood_mask = df['Family_ID'].isin(neighbor_family_ids)
        neighborhood_df = df[neighborhood_mask]

        if len(neighborhood_df) == 0:
            continue

        # 检验群落存活率 (拆解为总数和存活数)
        neighborhood_total = len(neighborhood_df)
        neighborhood_alive = neighborhood_df['is_alive'].sum()
        neighborhood_survival_rate = neighborhood_alive / neighborhood_total if neighborhood_total > 0 else 0

        # 【新增】获取群落邻居淘汰成员的具体原因统计
        neighbor_dead_mask = ~neighborhood_df['is_alive']
        neighbor_fail_stats = get_failure_statistics(neighborhood_df, neighbor_dead_mask)

        # 🔴 只保留群落生态也比较健康（>= 60%）的家族
        if neighborhood_survival_rate >= 0.60:
            f_row['neighborhood_survival_rate'] = neighborhood_survival_rate
            f_row['neighborhood_alive'] = neighborhood_alive  # 记录邻居存活数
            f_row['neighborhood_total'] = neighborhood_total  # 记录邻居总数
            f_row['neighbor_fail_stats'] = neighbor_fail_stats  # 【记录群落淘汰明细】
            valid_families.append(f_row)

    if not valid_families:
        print("⚠️ 警告：没有任何家族通过【家族群落参数平原检验】！参数全都在悬崖尖峰上。")
        return

    # =========================================================================================
    # 🏆 第五阶段：最终降维排序与全景输出
    # =========================================================================================

    family_df = pd.DataFrame(valid_families)
    top5_families = family_df.sort_values('family_final_score', ascending=False).head(50)
    # 固定随机顺序（数字 42 可以换成任意整数，只要不变，顺序就永远固定）
    # top5_families = top5_families.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"✅ 家族降维筛选完成！一共提取出【实盘底层抗击打与稳健分综合最优】的前 {len(top5_families)} 大参数家族：\n")
    print("=" * 85)

    medals = ['🥇', '🥈', '🥉', '🏅', '🏅']

    # 辅助函数：格式化死因统计字典输出
    def format_fail_stats(stats_dict):
        if not stats_dict:
            return "无淘汰"
        return " | ".join([f"{k}({v}个)" for k, v in stats_dict.items()])

    for i, (_, f_row) in enumerate(top5_families.iterrows()):
        rank_medal = medals[i] if i < len(medals) else '🏅'
        family_score = f_row['family_final_score']
        worst_row = f_row['worst_homo_row']

        # 家族宏观基因重组
        strat_A_type = worst_row.get('strat_A_type', 'Unknown')
        strat_B_type = worst_row.get('strat_B_type', 'Unknown')
        core_long = worst_row.get('core_long', 'Unknown')
        core_short = worst_row.get('core_short', 'Unknown')
        trend_long = int(worst_row.get('long_BTC_TREND_WINDOW', 0))
        trend_short = int(worst_row.get('short_BTC_TREND_WINDOW', 0))

        # 最差同源提取
        worst_offset = f"[{worst_row.get('offset_long', '0h')}+{worst_row.get('offset_short', '0h')}]"

        # 打印输出卡片 (升级至族群全景视角)
        print(f"第 {i + 1} 组家族 | 家族最终稳健分(Family_Final_Score): {family_score:.2f}")
        print(
            f" 🧬 家族宏观基因: {strat_A_type} [{core_long}](趋势:{trend_long}) + {strat_B_type} [{core_short}](趋势:{trend_short})")

        # # 🔴 [新增展现格式] 加入群落邻居平原存活率
        # # 🔴 [展现格式] 详细展示家族内部与外部群落邻居的存活比例
        # print(
        #     f" 🛡️ 抗扰动表现: 本家族存活 {f_row['alive_count']}/{f_row['total_members']} ({f_row['survival_rate'] * 100:.1f}%) | 群落邻居存活: {f_row.get('neighborhood_alive', 0)}/{f_row.get('neighborhood_total', 0)} ({f_row['neighborhood_survival_rate'] * 100:.1f}%)")
        #
        # # 🔴 [新增核心修改区] 打印家族成员及邻居由于哪些硬过滤规则被淘汰的详细统计
        # print(f"    ├─ 本家族淘汰主因: {format_fail_stats(f_row.get('family_fail_stats', {}))}")
        # print(f"    ├─ 群落邻居淘汰主因: {format_fail_stats(f_row.get('neighbor_fail_stats', {}))}")

        print(
            f"    └─ 同源组合收益极差: {f_row['yield_spread'] * 100:.1f}% | 收益下界: {f_row['annual_return_min']}% | 收益上界: {f_row['annual_return_max']}%")
        print(f" 📉 实盘保底表现 (源自本家族同源中表现最差的分身: {worst_offset})")

        # 🔴 [核心修改区] 将最丑陋一面的明细数据展示出来：精准暴露四大物理极限
        print(
            f"   ├─ 核心底线: 综合保底分: {worst_row['robust_score']:.2f} | 95%置信净期望: {worst_row.get('expectancy_ci_low', 0):.4f}")
        print(
            f"   ├─ 收益与敞口: 保底年化: +{worst_row['annual_return'] * 100:.1f}% (剥离最强年: {worst_row.get('rest_years_annual_mean', 0) * 100:+.1f}%) | 资金在市暴露度: {worst_row.get('time_in_market_pct', 0) * 100:.1f} | 资金最大占用: {f_row.get('max_money_ratio', 0) * 100:.1f}%")
        print(
            f"   ├─ 宏观心理折磨: 最大回撤: {worst_row['max_drawdown'] * 100:.1f}% | 最长水下: {worst_row.get('max_time_under_water_days', 0):.0f}天 | 历史最差滚动一年收益: {worst_row.get('rolling_12m_return_min', 0) * 100:.1f}%")
        print(
            f"   ├─ 微观执行极限: 单笔极限浮亏(MAE): {worst_row.get('mae_pct_worst', 0) * 100:.1f}% | 均持仓时长: {worst_row.get('avg_holding_hours', 0):.1f}h | 交易所打工率(Fee/PnL): {worst_row.get('fee_to_pnl_ratio', 0) * 100:.1f}%")
        print(
            f"   ├─ 摩擦与防线: 20bps滑点测试年化: +{worst_row.get('cost_stress_20bps_annual', 0) * 100:.1f}% |动量非对称性: 胜率: {worst_row.get('win_rate', 0) * 100:.1f}% | 盈亏比: {worst_row.get('profit_loss_ratio', 0):.2f}")
        print(
            f"   └─ 尾部动量捕获(肥尾依赖): 剔除最赚5笔后利润腰斩率: {worst_row.get('drop_top5_pnl_decay', 0) * 100:.1f}% | 负期望币种数: {int(worst_row.get('negative_expectancy_assets', 0))}个 | 币种HHI: {worst_row.get('asset_hhi', 0):.3f}")
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
    # # 将这里替换为你实际的 CSV 文件路径
    CSV_FILE_PATH = r"W:\project\python_project\oke_auto_trade\param_search_results\combined_metrics_results_op_op_op1.csv"

    # 🔴 [加载原始数据字典]：动态映射，传入任意包含全量基础属性参数的表
    raw_dfs_dict = {
        'LONG_ONLY': pd.read_csv(
            r"W:\project\python_project\oke_auto_trade\param_search_results\grid_search_131274_LONG_ONLY_dynamic_pool_GOLDEN_IMMUNE_with_Benchmark.csv"),
        'SHORT_ONLY': pd.read_csv(
            r"W:\project\python_project\oke_auto_trade\param_search_results\grid_search_131274_SHORT_ONLY_dynamic_pool_GOLDEN_IMMUNE_with_Benchmark.csv")
    }

    evaluate_and_print_top5(CSV_FILE_PATH, raw_dfs_dict=raw_dfs_dict)