"""
参数评价漏斗系统 - 独立评测脚本
依据"基础健康度 → Pareto 前沿 → 约束筛选 → 邻域稳定 → 时间稳定"五层流程，
从已保存的回测结果 CSV 中筛选最优参数。

所有阈值都是默认操作阈值，需根据策略特性调整。
"""

import os
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# 配置区
# ═══════════════════════════════════════════════════════════════════════════
RESULTS_PATH = r'W:\project\python_project\oke_auto_trade\param_search_results\grid_search_13608_both.csv'
OUTPUT_DIR   = r'W:\project\python_project\oke_auto_trade\param_search_results\evaluation'

# ---- Layer 1: 基础健康度过滤 (只淘汰反常，不筛选优秀) ----
HEALTH_FILTERS = {
    'min_total_trades':              30,      # 至少 30 笔平仓 (统计可解释)
    'min_active_assets':             2,       # 至少 2 个标的有交易
    'max_drawdown_threshold':        -0.45,   # MaxDD 不超过 45%
    'max_consecutive_losing_months': 8,       # 连续亏损不超过 8 个月
    'min_cost_stress_20bps_annual':  -0.1,   # 20bps 成本下年化不能<-5%
    'max_top1_pnl_ratio':            0.45,    # 单笔最高利润 ≤45% (避免极端单点)
    'max_top1_month_pnl_ratio':      0.55,    # 单月最高利润 ≤55%
    'max_negative_assets_ratio':     0.5,     # 负期望标的不超过 50%
    'max_time_under_water_days':     2500,     # 🌟 新增：最长水下阈值
}

# ---- Layer 2: Pareto 前沿目标 (4 个正交维度，避免重复) ----
PARETO_OBJECTIVES = {
    'sortino_ratio':              'maximize',  # 风险调整收益
    'max_time_under_water_days':  'minimize',  # 持仓体验
    'pnl_gini':                   'minimize',  # 利润分散性
    'monthly_positive_ratio':     'maximize',  # 时间稳定性
}

# ---- Layer 3: 约束式偏好筛选 (使用相对值/置信下界/衰减率) ----
CONSTRAINTS = {
    'monthly_positive_ratio':   ('>=', 0.4),  # 月度盈利占比 ≥50%
    'top1_pnl_ratio':           ('<=', 0.35),  # 单笔利润占比 ≤35%
    'drop_top3_pnl_decay':      ('<=', 0.55),  # 剔除 Top-3 后衰减 ≤55%
    'cost_stress_20bps_annual': ('>=', 0.0),   # 20bps 下年化非负
    'asset_hhi':                ('<=', 0.60),  # 标的集中度
    'expectancy_ci_low':        ('>=', 0.0),   # 单笔期望 95% 置信下界 ≥0
}

# ---- 主目标 (Pareto 内最终排序的单一指标) ----
PRIMARY_OBJECTIVE = 'sortino_ratio'

# ---- Layer 4: 邻域参数 ----

NEIGHBOR_CONFIG = {
    'radius':             1,      # 探测半径
    'cliff_threshold':    0.50,   # 邻居比当前低 >50% 即视为悬崖
    'min_neighbor_count': 3,      # 邻域至少有效点数
    'ignore_params':      ['param_MAX_WEIGHT']  # 🌟 核心修正：平原测试不考察权重维度
}

# ---- Layer 5: 时间稳定性参数 ----
TIME_STABILITY_CONFIG = {
    'min_profitable_years_ratio': 0.5,   # 盈利年份 ≥50%
    'require_both_halves_pos':    True,  # 上下半段都要正收益
}

# ═══════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════
def detect_param_cols(df):
    """识别参数列与变化的参数维度"""
    # 【修复】强制排除 param_name，只取真正的参数
    param_cols = [c for c in df.columns if c.startswith('param_') and c != 'param_name']

    varying = [c for c in param_cols if df[c].nunique() > 1]
    constant = [c for c in param_cols if df[c].nunique() <= 1]
    return param_cols, varying, constant


def safe_get(row, col, default=np.nan):
    return row[col] if col in row.index else default


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1: 基础健康度过滤
# ═══════════════════════════════════════════════════════════════════════════
def layer1_health_filter(df, filters):
    out = df.copy()
    out['L1_PASS']    = True
    out['L1_REASONS'] = ''

    def reject(mask, reason):
        if mask.any():
            out.loc[mask, 'L1_REASONS'] = out.loc[mask, 'L1_REASONS'] + reason + '; '
            out.loc[mask, 'L1_PASS']    = False

    if 'total_closed_trades' in out.columns:
        reject(out['total_closed_trades'].fillna(0) < filters['min_total_trades'],
               'too_few_trades')

    if 'active_assets' in out.columns:
        reject(out['active_assets'].fillna(0) < filters['min_active_assets'],
               'too_few_active_assets')

    if 'max_drawdown' in out.columns:
        reject(out['max_drawdown'].fillna(-1) < filters['max_drawdown_threshold'],
               'dd_too_deep')

    if 'max_consecutive_losing_months' in out.columns:
        reject(out['max_consecutive_losing_months'].fillna(99) >
               filters['max_consecutive_losing_months'],
               'losing_streak_too_long')

    if 'cost_stress_20bps_annual' in out.columns:
        reject(out['cost_stress_20bps_annual'].fillna(-1) <
               filters['min_cost_stress_20bps_annual'],
               'fragile_to_cost')

    if 'top1_pnl_ratio' in out.columns:
        reject(out['top1_pnl_ratio'].fillna(1) > filters['max_top1_pnl_ratio'],
               'top1_trade_concentrated')

    if 'top1_month_pnl_ratio' in out.columns:
        reject(out['top1_month_pnl_ratio'].fillna(1) >
               filters['max_top1_month_pnl_ratio'],
               'top1_month_concentrated')

    if 'active_assets' in out.columns and 'negative_expectancy_assets' in out.columns:
        denom = out['active_assets'].replace(0, np.nan)
        ratio = (out['negative_expectancy_assets'] / denom).fillna(0)
        reject(ratio > filters['max_negative_assets_ratio'],
               'too_many_negative_assets')
    # 🌟 新增：过滤最长水下期
    if 'max_time_under_water_days' in out.columns:
        # fillna(9999) 确保缺失值也被视为不合格直接淘汰
        reject(out['max_time_under_water_days'].fillna(9999) > filters['max_time_under_water_days'],
               'under_water_too_long')
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2: 多目标 Pareto 前沿
# ═══════════════════════════════════════════════════════════════════════════
def layer2_pareto_frontier(df, objectives, max_fronts=3):
    """
    升级版 Pareto 过滤：支持 Pareto Rank (前沿层级)
    max_fronts: 保留前 N 层 Pareto 前沿。设置为 1 则与原版严格 Pareto 完全一致。
                建议设置为 3~5，以保留“优秀的厚前沿带”，配合 L4 的平原分析。
    """
    out = df.copy()
    out['L2_PARETO'] = False
    out['PARETO_RANK'] = np.nan  # 记录它在第几层前沿

    cands = out[out['L1_PASS']].copy()
    if len(cands) == 0:
        print("⚠️ Layer 2: 没有通过 Layer 1 的候选")
        return out

    # 1. 整理矩阵 (统一转为 maximize)
    matrix, idx_list = [], cands.index.tolist()
    for col, direction in objectives.items():
        if col not in cands.columns:
            print(f"⚠️ Pareto 目标列缺失: {col}, 已跳过")
            continue
        v = cands[col].values.astype(float)
        worst_fill = -np.inf if direction == 'maximize' else np.inf
        v = np.where(np.isnan(v), worst_fill, v)
        if direction == 'minimize':
            v = -v
        matrix.append(v)

    if not matrix:
        return out

    M = np.array(matrix).T  # shape: (n, k)
    n = len(M)

    # 2. 分层寻找 Pareto 前沿 (非支配排序逻辑)
    remaining_indices = set(range(n))
    current_front_rank = 1
    pareto_pass_global_idx = []

    while remaining_indices and current_front_rank <= max_fronts:
        current_front = []
        rem_list = list(remaining_indices)

        # 在当前剩余点中找 Pareto 前沿
        for i in rem_list:
            is_dominated = False
            for j in rem_list:
                if i == j:
                    continue
                # j 支配 i
                if np.all(M[j] >= M[i]) and np.any(M[j] > M[i]):
                    is_dominated = True
                    break
            if not is_dominated:
                current_front.append(i)

        # 记录当前层的点
        for idx in current_front:
            real_idx = idx_list[idx]
            out.loc[real_idx, 'PARETO_RANK'] = current_front_rank
            pareto_pass_global_idx.append(real_idx)
            remaining_indices.remove(idx)

        current_front_rank += 1

    # 3. 标记最终通过 L2 的候选
    out.loc[pareto_pass_global_idx, 'L2_PARETO'] = True

    # [可选输出] 打印各层级数量，方便观察
    # print(f"  [Pareto 分层统计] 保留至前 {max_fronts} 层:")
    # for r in range(1, current_front_rank):
    #     print(f"    - 第 {r} 层前沿: {(out['PARETO_RANK'] == r).sum()} 组")

    return out

# ═══════════════════════════════════════════════════════════════════════════
# Layer 3: 约束式偏好筛选
# ═══════════════════════════════════════════════════════════════════════════
def layer3_constraint_filter(df, constraints):
    out = df.copy()
    out['L3_PASS']    = False
    out['L3_REASONS'] = ''

    cands = out[out['L2_PARETO']].copy()
    if len(cands) == 0:
        print("⚠️ Layer 3: Pareto 前沿为空")
        return out

    pass_mask = pd.Series(True, index=cands.index)

    for col, (op, threshold) in constraints.items():
        if col not in cands.columns:
            print(f"⚠️ 约束列缺失: {col}, 已跳过")
            continue

        # NaN 视为不通过
        v = cands[col]
        if op == '>=':
            fail = v.fillna(-np.inf) < threshold
        elif op == '<=':
            fail = v.fillna(np.inf) > threshold
        elif op == '>':
            fail = v.fillna(-np.inf) <= threshold
        elif op == '<':
            fail = v.fillna(np.inf) >= threshold
        else:
            continue

        rejected_idx = cands.index[fail]
        out.loc[rejected_idx, 'L3_REASONS'] = (
            out.loc[rejected_idx, 'L3_REASONS'] + f"{col}{op}{threshold}; "
        )
        pass_mask &= ~fail

    out.loc[cands.index[pass_mask], 'L3_PASS'] = True
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Layer 4: 参数平原 (动态半径邻域稳定性) 验证 [已剥离资金参数]
# ═══════════════════════════════════════════════════════════════════════════
def layer4_neighborhood(df, varying_params, primary_obj, config):
    out = df.copy()
    out['L4_TARGET_NEIGHBOR_COUNT'] = 0
    out['L4_NEIGHBOR_COUNT'] = 0
    out['L4_SURVIVING_NEIGHBOR_COUNT'] = 0
    out['L4_NEIGHBOR_HEALTH_RATE'] = np.nan
    out['L4_NEIGHBOR_OBJ_MEAN'] = np.nan
    out['L4_NEIGHBOR_OBJ_CV'] = np.nan
    out['L4_CLIFF'] = False
    out['L4_PASS'] = False

    # 🌟 核心修正：过滤掉不需要进行平原测试的资金管理参数
    ignored = config.get('ignore_params', [])
    search_dims = [p for p in varying_params if p not in ignored]

    if not search_dims:
        print("⚠️ Layer 4: 没有需要扫描的信号参数维度，跳过邻域分析")
        out['L4_PASS'] = out['L3_PASS']
        return out

    if primary_obj not in out.columns:
        return out

    radius = config.get('radius', 1)
    # 注意这里改成对 search_dims 进行遍历
    param_values = {col: sorted(out[col].dropna().unique()) for col in search_dims}

    for idx, row in out.iterrows():
        if not row.get('L3_PASS', False):
            continue

        mask = pd.Series(True, index=out.index)
        target_count = 1

        # 🌟 核心修正：仅在信号参数维度上寻找邻居，同时锁定当前的权重参数！
        # 也就是说，我们评估 MAX_WEIGHT=0.2 时的平原，只看周围同样是 0.2 权重的邻居
        for col in varying_params:
            if col in ignored:
                # 对于被忽略的参数（如权重），强制要求邻居必须和自己完全一样
                mask &= (out[col] == row[col])
            else:
                # 对于信号参数，寻找上下相邻的网格点
                cur_val = row[col]
                vals = param_values[col]
                try:
                    pos = vals.index(cur_val)
                except ValueError:
                    mask = mask & False
                    break

                start_idx = max(0, pos - radius)
                end_idx = min(len(vals), pos + radius + 1)
                allowed = vals[start_idx:end_idx]
                mask &= out[col].isin(allowed)
                target_count *= len(allowed)

        nbrs = out[mask]

        out.at[idx, 'L4_TARGET_NEIGHBOR_COUNT'] = target_count
        out.at[idx, 'L4_NEIGHBOR_COUNT'] = len(nbrs)

        if 'L1_PASS' in out.columns and len(nbrs) > 0:
            surviving_count = int(nbrs['L1_PASS'].fillna(False).sum())
            out.at[idx, 'L4_SURVIVING_NEIGHBOR_COUNT'] = surviving_count
            out.at[idx, 'L4_NEIGHBOR_HEALTH_RATE'] = nbrs['L1_PASS'].mean()

        obj_vals = nbrs[primary_obj].dropna()
        if len(obj_vals) >= 2:
            mu, sd = obj_vals.mean(), obj_vals.std()
            cv = abs(sd / mu) if abs(mu) > 1e-9 else np.inf
            out.at[idx, 'L4_NEIGHBOR_OBJ_MEAN'] = mu
            out.at[idx, 'L4_NEIGHBOR_OBJ_CV'] = cv

            cur = row[primary_obj]
            if pd.notna(cur) and cur > 1e-9:
                worst = obj_vals.min()
                if (cur - worst) / cur > config['cliff_threshold']:
                    out.at[idx, 'L4_CLIFF'] = True

        enough_nbrs = len(nbrs) >= config['min_neighbor_count']
        no_cliff = not out.at[idx, 'L4_CLIFF']
        out.at[idx, 'L4_PASS'] = bool(enough_nbrs and no_cliff)

    return out

# ═══════════════════════════════════════════════════════════════════════════
# Layer 5: 时间稳定性
# ═══════════════════════════════════════════════════════════════════════════
def layer5_time_stability(df, config):
    out = df.copy()
    out['L5_HALF_CONSISTENT']   = True
    out['L5_ANNUAL_OK']         = True
    out['L5_PASS']              = False

    # 上下半段一致性
    if config['require_both_halves_pos']:
        if 'first_half_return' in out.columns and 'second_half_return' in out.columns:
            out['L5_HALF_CONSISTENT'] = (
                (out['first_half_return'].fillna(-1) > 0) &
                (out['second_half_return'].fillna(-1) > 0)
            )

    # 盈利年份占比
    if 'profitable_years_ratio' in out.columns:
        out['L5_ANNUAL_OK'] = (
            out['profitable_years_ratio'].fillna(0) >=
            config['min_profitable_years_ratio']
        )

    out['L5_PASS'] = (
        out.get('L4_PASS', False) &
        out['L5_HALF_CONSISTENT'] &
        out['L5_ANNUAL_OK']
    )

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 综合得分 (邻域均值 × 健康率 / (1+CV))
# ═══════════════════════════════════════════════════════════════════════════
def compute_final_score(df, primary_obj):
    out = df.copy()
    out['FINAL_SCORE'] = np.nan

    valid_mask = out['L4_NEIGHBOR_OBJ_MEAN'].notna()
    if not valid_mask.any():
        return out

    nbrs_mean   = out.loc[valid_mask, 'L4_NEIGHBOR_OBJ_MEAN'].fillna(0)
    nbrs_health = out.loc[valid_mask, 'L4_NEIGHBOR_HEALTH_RATE'].fillna(0)
    nbrs_cv     = out.loc[valid_mask, 'L4_NEIGHBOR_OBJ_CV'].fillna(10).clip(upper=10)

    # 偏好：邻域均值高、健康率高、CV 低
    score = nbrs_mean * nbrs_health / (1 + nbrs_cv)
    out.loc[valid_mask, 'FINAL_SCORE'] = score

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 报告打印
# ═══════════════════════════════════════════════════════════════════════════
def _print_layer_report(df, pass_col, layer_name, total_initial):
    if pass_col not in df.columns:
        return
    n_pass = int(df[pass_col].fillna(False).sum())
    rate   = n_pass / total_initial * 100 if total_initial > 0 else 0
    print(f"  {layer_name}: 通过 {n_pass}/{total_initial} ({rate:.1f}%)")


def print_funnel_summary(df):
    total = len(df)
    print("\n" + "═" * 70)
    print("📊 漏斗筛选概览")
    print("═" * 70)
    _print_layer_report(df, 'L1_PASS',    "L1 健康度过滤    ", total)
    _print_layer_report(df, 'L2_PARETO',  "L2 Pareto 前沿   ", total)
    _print_layer_report(df, 'L3_PASS',    "L3 约束筛选      ", total)
    _print_layer_report(df, 'L4_PASS',    "L4 邻域稳定性    ", total)
    _print_layer_report(df, 'L5_PASS',    "L5 时间稳定性    ", total)
    print("═" * 70)


def print_rejection_summary(df, layer_col, reason_col, layer_name):
    rejected = df[~df[layer_col].fillna(False)] if layer_col in df.columns else pd.DataFrame()
    if len(rejected) == 0 or reason_col not in rejected.columns:
        return
    print(f"\n[{layer_name} 淘汰原因统计]")
    reason_counts = {}
    for r in rejected[reason_col].fillna(''):
        for tok in [t.strip() for t in r.split(';') if t.strip()]:
            reason_counts[tok] = reason_counts.get(tok, 0) + 1
    for k, v in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"   - {k:<32} : {v} 组")


# ═══════════════════════════════════════════════════════════════════════════
# 报告打印 (终极卡片版：含算分公式、水下占比、超额收益、标的贡献比)
# ═══════════════════════════════════════════════════════════════════════════
def print_top_candidates(df, primary_obj, varying_params, top_n=5):
    print("\n" + "═" * 80)
    print(f"🏆 最终推荐候选 (按 稳健综合分 排序，仅展示完美通过候选)")
    print("═" * 80)

    survivors = df[df['L5_PASS']].copy() if 'L5_PASS' in df.columns else pd.DataFrame()
    fallback_label = ''

    if len(survivors) == 0:
        for col, label in [('L4_PASS', 'L4 通过'), ('L3_PASS', 'L3 通过'), ('L2_PARETO', 'L2 Pareto 前沿')]:
            if col in df.columns:
                survivors = df[df[col].fillna(False)].copy()
                if len(survivors) > 0:
                    fallback_label = f' [回退到 {label}]'
                    break

    if len(survivors) == 0:
        print("⚠️ 所有层级均无候选，请放宽阈值")
        return

    if fallback_label:
        print(f"⚠️ 末层无候选{fallback_label}")

    survivors = survivors.sort_values(
        ['FINAL_SCORE', primary_obj], ascending=[False, False], na_position='last'
    ).head(top_n)

    # 辅助格式化函数
    def fmt_pct(val, dec=1):
        return f"{val * 100:+.{dec}f}%" if pd.notna(val) else "N/A"

    def fmt_pct_abs(val, dec=1):
        return f"{val * 100:.{dec}f}%" if pd.notna(val) else "N/A"

    def fmt_flt(val, dec=3):
        return f"{val:.{dec}f}" if pd.notna(val) else "N/A"

    def fmt_int(val):
        return f"{int(val)}" if pd.notna(val) else "0"

    for rank, (idx, row) in enumerate(survivors.iterrows(), 1):
        p_name = row.get('param_name', f'Candidate_{rank}')
        score = row.get('FINAL_SCORE', np.nan)

        print(f"\n" + "▼" * 80)
        print(f" 🎖️ 排位 No.{rank} | 参数代号: {p_name}")
        # ✨ 优化1: 解释稳健综合分的计算公式
        print(f"   ► 稳健综合分: {fmt_flt(score, 4)}  [计算逻辑: 邻域均值 × 邻域存活率 / (1 + 邻域CV)]")
        print("─" * 80)

        # 1. ⚙️ 参数配置
        print(" [⚙️ 策略参数配置]")
        param_strs = [f"{p.replace('param_', '')}: {row.get(p, 'N/A')}" for p in varying_params]
        print(f"   ► {',  '.join(param_strs)}")

        # 2. 📊 核心基础绩效
        print("\n [📊 核心基础绩效]")
        print(
            f"   ├─ 交易统计: {fmt_int(row.get('total_closed_trades'))} 笔平仓 | 胜率: {fmt_pct_abs(row.get('win_rate'))} | 盈亏比: {fmt_flt(row.get('profit_loss_ratio'), 2)}")
        print(
            f"   ├─ 收益状况: 总收益: {fmt_pct(row.get('total_return'))} | 年化收益: {fmt_pct(row.get('annual_return'))}")

        # ✨ 优化4: 水下时间与占比
        max_uw_days = row.get('max_time_under_water_days', np.nan)
        total_days = row.get('days_passed', np.nan)
        uw_str = f"{fmt_flt(max_uw_days, 1)}天"
        if pd.notna(max_uw_days) and pd.notna(total_days) and total_days > 0:
            uw_str += f" (占回测总时长 {fmt_pct_abs(max_uw_days / total_days, 1)})"

        print(
            f"   ├─ 回撤体验: 最大回撤: {fmt_pct(row.get('max_drawdown'))} | 最长水下期: {uw_str} | 平均恢复: {fmt_flt(row.get('avg_recovery_time_days'), 1)}天")
        print(
            f"   ├─ 风险调整: {primary_obj}(主目标): {fmt_flt(row.get(primary_obj))} | 盈利月占比: {fmt_pct_abs(row.get('monthly_positive_ratio'))}")
        print(
            f"   └─ 集中度险: 币种HHI: {fmt_flt(row.get('asset_hhi'))} | 最赚1笔占比: {fmt_pct_abs(row.get('top1_pnl_ratio'))}")

        # 3. 🛡️ 平原与鲁棒性
        print("\n [🛡️ 参数平原与鲁棒性验证]")
        target_nbrs = fmt_int(row.get('L4_TARGET_NEIGHBOR_COUNT', 0))
        actual_nbrs = fmt_int(row.get('L4_NEIGHBOR_COUNT', 0))
        survive_nbrs = fmt_int(row.get('L4_SURVIVING_NEIGHBOR_COUNT', 0))

        print(
            f"   ├─ 平原稳定性: 邻居容量 [目标: {target_nbrs} | 实际: {actual_nbrs} | 存活: {survive_nbrs}] | 存活率: {fmt_pct_abs(row.get('L4_NEIGHBOR_HEALTH_RATE'), 0)}")
        print(
            f"   ├─ 邻域绩效评估: 主目标均值: {fmt_flt(row.get('L4_NEIGHBOR_OBJ_MEAN'))} | 颠簸度CV: {fmt_flt(row.get('L4_NEIGHBOR_OBJ_CV'), 4)}")
        print(f"   └─ 成本压力测: 增加20bps双边滑点后，年化收益变为 -> {fmt_pct(row.get('cost_stress_20bps_annual'))}")

        # 4. 📅 年度表现拆解
        # ✨ 优化3: 引入基准涨跌幅与超额收益计算 (兼容无基准数据的CSV)
        print("\n [📅 年度表现拆解]")
        year_cols = [c for c in df.columns if c.startswith('year_') and c.endswith('_return')]
        if year_cols:
            years = sorted([int(c.split('_')[1]) for c in year_cols])
            for y in years:
                y_ret = row.get(f'year_{y}_return', np.nan)
                y_dd = row.get(f'year_{y}_max_dd', np.nan)
                y_bench_ret = row.get(f'year_{y}_benchmark_return', np.nan)
                y_bench_dd = row.get(f'year_{y}_benchmark_dd', np.nan)

                # 如果未来你的 CSV 生成了基准数据，这里会自动算超额
                if pd.notna(y_bench_ret):
                    excess_ret = y_ret - y_bench_ret
                    bench_str = f" | 基准收益: {fmt_pct(y_bench_ret):>7} (基准回撤 {fmt_pct(y_bench_dd):>7}) | 🌟超额收益: {fmt_pct(excess_ret):>7}"
                else:
                    bench_str = ""  # 当前CSV没基准数据时的优雅回退

                if pd.notna(y_ret):
                    print(f"   ► {y}年: 策略收益: {fmt_pct(y_ret):>7} (最大回撤 {fmt_pct(y_dd):>7}){bench_str}")
        else:
            print("   ► (无年度拆解数据)")

        # 5. 🪙 标的贡献明细
        print("\n [🪙 标的盈亏贡献明细 (按净利润排序)]")
        asset_cols = [c for c in df.columns if c.startswith('asset_') and c.endswith('_net_pnl')]
        if asset_cols:
            assets_info = []
            for c in asset_cols:
                coin = c.replace('asset_', '').replace('_net_pnl', '')
                net_pnl = row.get(c, 0.0)
                trades = row.get(f'asset_{coin}_trades', 0)
                win_r = row.get(f'asset_{coin}_win_rate', np.nan)
                # ✨ 优化2: 获取净利润占比
                share = row.get(f'asset_{coin}_pnl_share', np.nan)

                if trades > 0:
                    assets_info.append({'coin': coin, 'pnl': net_pnl, 'trades': trades, 'wr': win_r, 'share': share})

            assets_info = sorted(assets_info, key=lambda x: x['pnl'], reverse=True)
            for a in assets_info:
                print(
                    f"   - {a['coin']:<6}: 净利润 ${a['pnl']:>8.2f} (利润占比: {fmt_pct_abs(a['share'])} ) | 交易: {fmt_int(a['trades']):>3}笔 | 胜率: {fmt_pct_abs(a['wr'])}")
        else:
            print("   ► (无标的拆解数据)")

        print("▲" * 80 + "\n")

# ═══════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════
def evaluate(results_path):
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"找不到结果文件: {results_path}")

    df = pd.read_csv(results_path, encoding='utf-8-sig')

    print("═" * 70)
    print("🚀 参数评价漏斗系统")
    print("═" * 70)
    print(f"读取文件: {results_path}")
    print(f"参数组数: {len(df)}")

    param_cols, varying, constant = detect_param_cols(df)
    print(f"参数维度: {len(param_cols)} 维 (变化: {len(varying)}, 固定: {len(constant)})")
    print(f"  变化的参数: {varying}")
    if constant:
        print(f"  固定的参数: {constant}  ⚠️ 邻域分析仅基于变化维度")

    # 漏斗
    df = layer1_health_filter(df, HEALTH_FILTERS)
    df = layer2_pareto_frontier(df, PARETO_OBJECTIVES)
    df = layer3_constraint_filter(df, CONSTRAINTS)
    df = layer4_neighborhood(df, varying, PRIMARY_OBJECTIVE, NEIGHBOR_CONFIG)
    df = layer5_time_stability(df, TIME_STABILITY_CONFIG)
    df = compute_final_score(df, PRIMARY_OBJECTIVE)

    # 报告
    print_funnel_summary(df)
    print_rejection_summary(df, 'L1_PASS', 'L1_REASONS', 'L1 健康度')
    print_rejection_summary(df, 'L3_PASS', 'L3_REASONS', 'L3 约束')
    print_top_candidates(df, PRIMARY_OBJECTIVE, varying, top_n=5)

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(results_path))[0]
    save_path = os.path.join(OUTPUT_DIR, f'{base}_evaluated.csv')
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 完整评估结果已保存: {save_path}")

    return df


if __name__ == "__main__":
    result_df = evaluate(RESULTS_PATH)