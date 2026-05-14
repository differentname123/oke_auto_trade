"""
截面动量(CSM)专属参数评价漏斗系统 - 第一性原理终极重构版
包含完整的多空隔离逻辑、下置信界(LCB)评分，以及极其详尽的卡片式日志打印。
"""

import os
import numpy as np
import pandas as pd
import itertools
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════
# 全局配置区
# ═══════════════════════════════════════════════════════════════════════════
# 切换评估模式: 'LONG' (多头参数漏斗) 或 'SHORT' (空头参数漏斗)

# 文件路径配置
RESULTS_PATH = r'W:\project\python_project\oke_auto_trade\param_search_results\grid_search_131274_LONG_ONLY_dynamic_pool_offset_0h_with_Benchmark.csv'
EVAL_SIDE = 'LONG'
if 'SHORT' in RESULTS_PATH.upper():
    EVAL_SIDE = 'SHORT'


# ═══════════════════════════════════════════════════════════════════════════
# 📈 方案 A：LONG 专属评价体系
# ═══════════════════════════════════════════════════════════════════════════
LONG_CONFIG = {
    'L1_HEALTH': {
        'min_total_trades':              30,
        'min_active_assets':             3,
        'max_drawdown_threshold':        -0.35,
        'min_expectancy_ci_low':         0.0,
        'min_cost_stress_20bps_annual':  0.0,
        'max_drop_top3_pnl_decay':       0.70,
    },
    'L2_PARETO': {
        'sortino_ratio':              'maximize',
        'bull_regime_total_return':   'maximize', # 使用体制内绝对收益作为目标
        'expectancy_ci_low':          'maximize',
        'drop_top3_pnl_decay':        'minimize',
    },
    'L3_CONSTRAINTS': {
        'top1_pnl_ratio':           ('<=', 0.45), # ✅ 修复5.1: 统一列名映射
        'asset_hhi':                ('<=', 0.65),
    },
    'L5_TIME': {
        'min_profitable_years_ratio': 0.50,
    },
    'PRIMARY_OBJ': 'sortino_ratio'
}

# ═══════════════════════════════════════════════════════════════════════════
# 📉 方案 B：SHORT 专属评价体系 (宽进严评松绑版)
# ═══════════════════════════════════════════════════════════════════════════
SHORT_CONFIG = {
    'L1_HEALTH': {
        'min_total_trades':              30,
        'min_active_assets':             2,
        'max_drawdown_threshold':        -0.30,   # ✅ 放宽：从 -0.20 放宽到 -0.30，容忍熊市剧烈反弹
        'max_mae_pct_worst':             -0.35,   # ✅ 放宽：从 -0.25 放宽到 -0.35
        'min_bear_regime_total_return':  0.0,     # 🔒 底线：熊市必须赚钱，这个绝对不能动
        'min_cost_stress_20bps_annual':  0.0,     # ✅ 修复5.3: 收紧为 >= 0.0，与 L2 最大化目标自洽
        'min_expectancy_ci_low':         0.0,     # ✅ 修复5.4: 增加空头置信下界硬约束防小样本运气
        'max_avg_holding_hours':         120.0,   # ✅ 放宽：从 72 小时放宽到 120 小时(5天)
    },
    'L2_PARETO': {
        'calmar_ratio':               'maximize',
        'mae_pct_worst':              'maximize',
        'bear_regime_total_return':   'maximize',
        'cost_stress_20bps_annual':   'maximize', # 配合 L1 修改
    },
    'L3_CONSTRAINTS': {
        'top1_pnl_ratio':           ('<=', 0.45), # ✅ 修复5.1: 统一列名映射
        'profit_loss_ratio':        ('>=', 0.9),  # ✅ 放宽：只要期望下界好，盈亏比 0.9 也能接受
    },
    'L5_TIME': {
        'min_profitable_years_ratio': 0.30,       # ✅ 放宽：空头本来赚钱年份就少，30%即可
    },
    'PRIMARY_OBJ': 'calmar_ratio'
}

NEIGHBOR_CONFIG = {
    'radius':             1,
    'cliff_threshold':    0.40,
    'min_neighbor_count': 3,
    'ignore_params':      ['param_MAX_WEIGHT', 'param_TOP_K']
}

# ═══════════════════════════════════════════════════════════════════════════
# 核心逻辑区
# ═══════════════════════════════════════════════════════════════════════════
def detect_param_cols(df):
    param_cols = [c for c in df.columns if c.startswith('param_') and c != 'param_name']
    varying = [c for c in param_cols if df[c].nunique() > 1]
    constant = [c for c in param_cols if df[c].nunique() <= 1]
    return param_cols, varying, constant

def layer1_health_filter(df, filters):
    out = df.copy()
    out['L1_PASS']    = True
    out['L1_REASONS'] = ''

    def reject(mask, reason):
        if mask.any():
            out.loc[mask, 'L1_REASONS'] = out.loc[mask, 'L1_REASONS'] + reason + '; '
            out.loc[mask, 'L1_PASS']    = False

    if 'total_closed_trades' in out.columns:
        reject(out['total_closed_trades'].fillna(0) < filters.get('min_total_trades', 0), 'too_few_trades')
    if 'active_assets' in out.columns:
        # ✅ 修复5.2: 默认安全下限改为 2，避免配置遗失时退化为不筛选
        reject(out['active_assets'].fillna(0) < filters.get('min_active_assets', 2), 'too_few_active_assets')
    if 'max_drawdown' in out.columns:
        reject(out['max_drawdown'].fillna(-1) < filters.get('max_drawdown_threshold', -1), 'dd_too_deep')
    if 'expectancy_ci_low' in out.columns and 'min_expectancy_ci_low' in filters:
        reject(out['expectancy_ci_low'].fillna(-1) < filters['min_expectancy_ci_low'], 'stat_ci_low_negative')
    if 'cost_stress_20bps_annual' in out.columns and 'min_cost_stress_20bps_annual' in filters:
        reject(out['cost_stress_20bps_annual'].fillna(-1) < filters['min_cost_stress_20bps_annual'], 'fail_20bps_stress')
    if 'cost_stress_30bps_annual' in out.columns and 'min_cost_stress_30bps_annual' in filters:
        reject(out['cost_stress_30bps_annual'].fillna(-1) < filters['min_cost_stress_30bps_annual'], 'fail_30bps_stress')
    if 'drop_top3_pnl_decay' in out.columns and 'max_drop_top3_pnl_decay' in filters:
        reject(out['drop_top3_pnl_decay'].fillna(1) > filters['max_drop_top3_pnl_decay'], 'rely_too_much_on_top3')
    if 'mae_pct_worst' in out.columns and 'max_mae_pct_worst' in filters:
        reject(out['mae_pct_worst'].fillna(-1) < filters['max_mae_pct_worst'], 'mae_tail_risk_too_high')
    if 'bear_regime_total_return' in out.columns and 'min_bear_regime_total_return' in filters:
        reject(out['bear_regime_total_return'].fillna(-1) < filters['min_bear_regime_total_return'], 'loss_in_bear_regime')
    if 'avg_holding_hours' in out.columns and 'max_avg_holding_hours' in filters:
        reject(out['avg_holding_hours'].fillna(999) > filters['max_avg_holding_hours'], 'holding_too_long(funding_bleed)')
    return out

def layer2_pareto_frontier(df, objectives, max_fronts=5):
    out = df.copy()
    out['L2_PARETO'] = False
    out['PARETO_RANK'] = np.nan
    cands = out[out['L1_PASS']].copy()
    if len(cands) == 0: return out

    matrix, idx_list = [], cands.index.tolist()
    for col, direction in objectives.items():
        if col not in cands.columns: continue
        v = cands[col].values.astype(float)
        worst_fill = -np.inf if direction == 'maximize' else np.inf
        v = np.where(np.isnan(v), worst_fill, v)
        if direction == 'minimize': v = -v
        matrix.append(v)

    if not matrix: return out
    M = np.array(matrix).T
    remaining_indices = np.arange(len(M))
    current_front_rank = 1
    pareto_pass_global_idx = []

    while len(remaining_indices) > 0 and current_front_rank <= max_fronts:
        M_rem = M[remaining_indices]
        is_efficient = np.ones(len(M_rem), dtype=bool)
        for i in range(len(M_rem)):
            if is_efficient[i]:
                c = M_rem[i]
                eff_idx = np.where(is_efficient)[0]
                eff_idx = eff_idx[eff_idx != i]
                if len(eff_idx) == 0: break
                M_eff = M_rem[eff_idx]
                dom = np.all(c >= M_eff, axis=1) & np.any(c > M_eff, axis=1)
                if np.any(dom): is_efficient[eff_idx[dom]] = False

        current_front_local_idx = np.where(is_efficient)[0]
        current_front = remaining_indices[current_front_local_idx]
        for idx in current_front:
            real_idx = idx_list[idx]
            out.loc[real_idx, 'PARETO_RANK'] = current_front_rank
            pareto_pass_global_idx.append(real_idx)
        remaining_indices = remaining_indices[~is_efficient]
        current_front_rank += 1

    if pareto_pass_global_idx:
        out.loc[pareto_pass_global_idx, 'L2_PARETO'] = True
    return out

def layer3_constraint_filter(df, constraints):
    out = df.copy()
    out['L3_PASS']    = False
    out['L3_REASONS'] = ''
    cands = out[out['L2_PARETO']].copy()
    if len(cands) == 0: return out
    pass_mask = pd.Series(True, index=cands.index)

    for col, (op, threshold) in constraints.items():
        # ✅ 修复5.1: 删除了原有的硬编码 if col == 'max_top1_pnl_ratio' 分支，完全依赖标准映射逻辑
        if col not in cands.columns: continue
        v = cands[col]
        if op == '>=': fail = v.fillna(-np.inf) < threshold
        elif op == '<=': fail = v.fillna(np.inf) > threshold
        else: continue
        rejected_idx = cands.index[fail]
        out.loc[rejected_idx, 'L3_REASONS'] += f"{col}{op}{threshold}; "
        pass_mask &= ~fail

    out.loc[cands.index[pass_mask], 'L3_PASS'] = True
    return out


def layer4_neighborhood(df, varying_params, primary_obj, config):
    out = df.copy()
    n_rows = len(out)
    target_counts, neighbor_counts, surviving_counts = np.zeros(n_rows, dtype=int), np.zeros(n_rows,
                                                                                             dtype=int), np.zeros(
        n_rows, dtype=int)

    # ✅ 修复5.5: 新增 obj_stds 数组直接存储标准差
    health_rates, obj_means, obj_cvs, obj_stds = np.full(n_rows, np.nan), np.full(n_rows, np.nan), np.full(n_rows, np.nan), np.full(n_rows, np.nan)
    cliffs, passes = np.zeros(n_rows, dtype=bool), np.zeros(n_rows, dtype=bool)

    ignored = config.get('ignore_params', [])
    search_dims = [p for p in varying_params if p not in ignored]
    if not search_dims or primary_obj not in out.columns:
        out['L4_PASS'] = out.get('L3_PASS', False)
        return out

    radius = config.get('radius', 1)
    coords = np.full((n_rows, len(varying_params)), -1, dtype=int)
    param_vals_len = {}
    for i, col in enumerate(varying_params):
        vals = sorted(out[col].dropna().unique())
        param_vals_len[i] = len(vals)
        val_to_idx = {v: idx for idx, v in enumerate(vals)}
        coords[:, i] = out[col].map(val_to_idx).fillna(-1).astype(int)

    ignored_idx = set([i for i, col in enumerate(varying_params) if col in ignored])
    coord_map = defaultdict(list)
    for i, c in enumerate(coords):
        if not np.any(c < 0): coord_map[tuple(c)].append(i)

    l3_pass_mask = out.get('L3_PASS', pd.Series([False] * n_rows)).values
    l1_pass_arr = out.get('L1_PASS', pd.Series([False] * n_rows)).fillna(False).values
    primary_obj_arr = out[primary_obj].astype(float).values

    for idx in np.where(l3_pass_mask)[0]:
        target_coord = coords[idx]
        if np.any(target_coord < 0): continue
        target_count, ranges = 1, []
        for i in range(len(varying_params)):
            pos = int(target_coord[i])
            if i in ignored_idx:
                ranges.append([pos])
            else:
                length = param_vals_len[i]
                start, end = max(0, pos - radius), min(length - 1, pos + radius)
                ranges.append(list(range(start, end + 1)))
                target_count *= (end - start + 1)

        target_counts[idx] = target_count
        nbrs_idx = []
        for n_coord in itertools.product(*ranges):
            if n_coord in coord_map: nbrs_idx.extend(coord_map[n_coord])

        neighbor_counts[idx] = len(nbrs_idx)
        if len(nbrs_idx) > 0:
            surviving = l1_pass_arr[nbrs_idx].sum()
            surviving_counts[idx] = surviving
            health_rates[idx] = surviving / len(nbrs_idx)

        # 只提取 L1 存活的邻居数据计算均值和颠簸度
        surviving_nbrs_idx = [n_idx for n_idx in nbrs_idx if l1_pass_arr[n_idx]]
        obj_vals = primary_obj_arr[surviving_nbrs_idx]
        obj_vals = obj_vals[~np.isnan(obj_vals)]
        if len(obj_vals) >= 2:
            mu, sd = obj_vals.mean(), obj_vals.std(ddof=1)
            cv = abs(sd / mu) if abs(mu) > 1e-9 else np.inf
            # ✅ 修复5.5: 同步保存真实的标准差
            obj_means[idx], obj_cvs[idx], obj_stds[idx] = mu, cv, sd

        # 强制废除悬崖判断，将惩罚权移交给 LCB
        cliffs[idx] = False

        # 满足容量够、存活率达标（>=10%）、平原均值为正即可放行
        passes[idx] = bool((len(nbrs_idx) >= config.get('min_neighbor_count', 5)) and (health_rates[idx] >= 0.10) and (
                    obj_means[idx] > 0))

    out['L4_TARGET_NEIGHBOR_COUNT'], out['L4_NEIGHBOR_COUNT'], out[
        'L4_SURVIVING_NEIGHBOR_COUNT'] = target_counts, neighbor_counts, surviving_counts
    out['L4_NEIGHBOR_HEALTH_RATE'], out['L4_NEIGHBOR_OBJ_MEAN'], out[
        'L4_NEIGHBOR_OBJ_CV'] = health_rates, obj_means, obj_cvs
    # ✅ 修复5.5: 直接将算好的标准差附带到列中输出
    out['L4_NEIGHBOR_OBJ_STD'] = obj_stds
    out['L4_CLIFF'], out['L4_PASS'] = cliffs, passes
    return out



def layer5_time_stability(df, config):
    out = df.copy()
    if 'profitable_years_ratio' in out.columns:
        annual_ok = out['profitable_years_ratio'].fillna(0) >= config['min_profitable_years_ratio']
    else:
        annual_ok = pd.Series(True, index=out.index)
    out['L5_PASS'] = out.get('L4_PASS', False) & annual_ok
    return out

def compute_final_score(df):
    out = df.copy()
    out['FINAL_SCORE'] = np.nan
    # ✅ 修复5.5: 直接要求均值和标准差均有效（不为NaN），彻底拒绝孤岛参数
    valid_mask = out['L4_NEIGHBOR_OBJ_MEAN'].notna() & out['L4_NEIGHBOR_OBJ_STD'].notna()
    if not valid_mask.any(): return out

    nbrs_mean = out.loc[valid_mask, 'L4_NEIGHBOR_OBJ_MEAN']
    nbrs_health = out.loc[valid_mask, 'L4_NEIGHBOR_HEALTH_RATE']
    # ✅ 修复5.5: 直接采用真实的标准差，不再从CV反推，杜绝 fillna(0) 带来的数学畸变
    nbrs_std = out.loc[valid_mask, 'L4_NEIGHBOR_OBJ_STD']

    lcb = nbrs_mean - 1.5 * nbrs_std
    out.loc[valid_mask, 'FINAL_SCORE'] = lcb * nbrs_health
    return out

# ═══════════════════════════════════════════════════════════════════════════
# 完美复刻的丰富日志打印模块
# ═══════════════════════════════════════════════════════════════════════════
def print_funnel_summary(df):
    total = len(df)
    print("\n" + "═" * 70)
    print(f"📊 {EVAL_SIDE} 模式 漏斗筛选概览")
    print("═" * 70)
    for col, name in [('L1_PASS', "L1 硬生存底线"), ('L2_PARETO', "L2 Pareto前沿"),
                      ('L3_PASS', "L3 软约束筛选"), ('L4_PASS', "L4 邻域平原测试"), ('L5_PASS', "L5 时间跨度验证")]:
        if col in df.columns:
            n_pass = int(df[col].fillna(False).sum())
            rate = n_pass / total * 100 if total > 0 else 0
            print(f"  {name:<15}: 通过 {n_pass}/{total} ({rate:.1f}%)")

def print_rejection_summary(df, layer_col, reason_col, layer_name):
    rejected = df[~df[layer_col].fillna(False)] if layer_col in df.columns else pd.DataFrame()
    if len(rejected) == 0 or reason_col not in rejected.columns: return
    print(f"\n[{layer_name} 淘汰原因统计]")
    reason_counts = {}
    for r in rejected[reason_col].fillna(''):
        for tok in [t.strip() for t in r.split(';') if t.strip()]:
            reason_counts[tok] = reason_counts.get(tok, 0) + 1
    for k, v in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"   - {k:<40} : {v} 组")

def fmt_pct(val, dec=1):
    return f"{val * 100:+.{dec}f}%" if pd.notna(val) else "N/A"

def fmt_pct_abs(val, dec=1):
    return f"{val * 100:.{dec}f}%" if pd.notna(val) else "N/A"

def fmt_flt(val, dec=3):
    return f"{val:.{dec}f}" if pd.notna(val) else "N/A"

def fmt_int(val):
    return f"{int(val)}" if pd.notna(val) else "0"

def print_top_candidates(df, primary_obj, varying_params, top_n=5):
    print("\n" + "═" * 80)
    print(f"🏆 最终 {EVAL_SIDE} 生产环境候选 (按 平原下置信界 LCB 排序)")
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
        print(f"⚠️ 经过第一性原理的严苛筛选，没有完美的 {EVAL_SIDE} 参数。请复核底线阈值。")
        return

    if fallback_label:
        print(f"⚠️ 末层无候选{fallback_label}")

    survivors = survivors.sort_values(['FINAL_SCORE', primary_obj], ascending=[False, False]).head(top_n)

    for rank, (idx, row) in enumerate(survivors.iterrows(), 1):
        print(f"\n▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼")
        print(f" 🎖️ 排位 No.{rank} | 参数代号: {row.get('param_name', f'Candidate_{rank}')}")
        print(f"   ► 稳健综合分 LCB: {fmt_flt(row.get('FINAL_SCORE'))} [公式: (均值 - 1.5*标准差) × 存活率]")
        print("────────────────────────────────────────────────────────────────────────────────")

        # [⚙️ 策略参数配置]
        print(" [⚙️ 策略参数配置]")
        param_strs = [f"{p.replace('param_', '')}: {row.get(p, 'N/A')}" for p in varying_params]
        print(f"   ► {',  '.join(param_strs)}")

        # [📊 核心基础绩效]
        print("\n [📊 核心基础绩效]")
        print(f"   ├─ 交易统计: {fmt_int(row.get('total_closed_trades'))} 笔平仓 | 胜率: {fmt_pct_abs(row.get('win_rate'))} | 盈亏比: {fmt_flt(row.get('profit_loss_ratio'), 2)} | 期望下界: {fmt_flt(row.get('expectancy_ci_low'))}")
        print(f"   ├─ 收益状况: 总收益: {fmt_pct(row.get('total_return'))} | 年化收益: {fmt_pct(row.get('annual_return'))}")

        # 🌟 修复后的牛熊环境绝对收益与K线耗时
        bull_ret = row.get('bull_regime_total_return', np.nan)
        bull_bars = row.get('bull_regime_bars', np.nan)
        bear_ret = row.get('bear_regime_total_return', np.nan)
        bear_bars = row.get('bear_regime_bars', np.nan)
        print(f"   ├─ 周期拆解: 牛市绝对收益: {fmt_pct(bull_ret)} (历经 {fmt_int(bull_bars)} 根K线) | 熊市绝对收益: {fmt_pct(bear_ret)} (历经 {fmt_int(bear_bars)} 根K线)")

        max_uw_days = row.get('max_time_under_water_days', np.nan)
        total_days = row.get('days_passed', np.nan)
        uw_str = f"{fmt_flt(max_uw_days, 1)}天"
        if pd.notna(max_uw_days) and pd.notna(total_days) and total_days > 0:
            uw_str += f" (占总时长 {fmt_pct_abs(max_uw_days / total_days, 1)})"

        mae_str = f" | ☠️最差MAE: {fmt_pct(row.get('mae_pct_worst'))}" if EVAL_SIDE == 'SHORT' else ""
        print(f"   ├─ 回撤体验: 最大回撤: {fmt_pct(row.get('max_drawdown'))} | 全局最长水下: {uw_str}{mae_str}")
        print(f"   ├─ 风险调整: {primary_obj}(主目标): {fmt_flt(row.get(primary_obj))} | 盈利月占比: {fmt_pct_abs(row.get('monthly_positive_ratio'))}")

        drop_str = f" | Top3剔除衰减: {fmt_pct(row.get('drop_top3_pnl_decay'))}" if EVAL_SIDE == 'LONG' else ""
        print(f"   └─ 集中度险: 币种HHI: {fmt_flt(row.get('asset_hhi'))} | 最赚1笔占比: {fmt_pct_abs(row.get('top1_pnl_ratio'))}{drop_str}")

        # [🛡️ 参数平原与鲁棒性验证]
        print("\n [🛡️ 参数平原与鲁棒性验证]")
        target_nbrs = fmt_int(row.get('L4_TARGET_NEIGHBOR_COUNT', 0))
        actual_nbrs = fmt_int(row.get('L4_NEIGHBOR_COUNT', 0))
        survive_nbrs = fmt_int(row.get('L4_SURVIVING_NEIGHBOR_COUNT', 0))
        print(f"   ├─ 平原稳定性: 邻居容量 [目标: {target_nbrs} | 实际: {actual_nbrs} | 存活: {survive_nbrs}] | 存活率: {fmt_pct_abs(row.get('L4_NEIGHBOR_HEALTH_RATE'), 0)}")
        print(f"   ├─ 邻域绩效评估: 主目标均值: {fmt_flt(row.get('L4_NEIGHBOR_OBJ_MEAN'))} | 颠簸度CV: {fmt_flt(row.get('L4_NEIGHBOR_OBJ_CV'), 4)}")

        cost_col = 'cost_stress_20bps_annual' if EVAL_SIDE == 'LONG' else 'cost_stress_30bps_annual'
        print(f"   └─ 成本压力测: 增加极限滑点后，年化收益变为 -> {fmt_pct(row.get(cost_col))}")

        # [📅 年度表现拆解]
        print("\n [📅 年度表现拆解]")
        year_cols = [c for c in df.columns if c.startswith('year_') and c.endswith('_return') and 'excess' not in c]
        if year_cols:
            years = sorted(list(set([int(c.split('_')[1]) for c in year_cols])))
            for y in years:
                y_ret = row.get(f'year_{y}_return', np.nan)
                y_dd = row.get(f'year_{y}_max_dd', np.nan)
                y_bench_ret = row.get(f'benchmark_year_{y}_return', np.nan)
                excess_ret = row.get(f'year_{y}_excess_return', np.nan)

                if pd.notna(y_bench_ret):
                    bench_str = f" | 基准收益: {fmt_pct(y_bench_ret):>7} | 🌟全局超额: {fmt_pct(excess_ret):>7}"
                else:
                    bench_str = ""

                if pd.notna(y_ret):
                    print(f"   ► {y}年: 策略收益: {fmt_pct(y_ret):>7} (最大回撤 {fmt_pct(y_dd):>7}){bench_str}")
        else:
            print("   ► (无年度拆解数据)")

        # [🪙 标的盈亏贡献明细]
        print("\n [🪙 标的盈亏贡献明细 (按净利润排序)]")
        asset_cols = [c for c in df.columns if c.startswith('asset_') and c.endswith('_net_pnl')]
        if asset_cols:
            assets_info = []
            for c in asset_cols:
                coin = c.replace('asset_', '').replace('_net_pnl', '')
                net_pnl = row.get(c, 0.0)
                trades = row.get(f'asset_{coin}_trades', 0)
                win_r = row.get(f'asset_{coin}_win_rate', np.nan)
                share = row.get(f'asset_{coin}_pnl_share', np.nan)

                if trades > 0:
                    assets_info.append({'coin': coin, 'pnl': net_pnl, 'trades': trades, 'wr': win_r, 'share': share})

            assets_info = sorted(assets_info, key=lambda x: x['pnl'], reverse=True)
            for a in assets_info:
                print(f"   - {a['coin']:<6}: 净利润 ${a['pnl']:>8.2f} (利润占比: {fmt_pct_abs(a['share'])} ) | 交易: {fmt_int(a['trades']):>3}笔 | 胜率: {fmt_pct_abs(a['wr'])}")
        else:
            print("   ► (无标的拆解数据)")

        print("▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲\n")


# ═══════════════════════════════════════════════════════════════════════════
# 🎯 指定参数定点查询打印
# ═══════════════════════════════════════════════════════════════════════════
def print_specific_params(df, target_params, primary_obj, varying_params):
    print("\n" + "═" * 80)
    print(f"🎯 指定参数组合深度核查")
    print("═" * 80)

    mask = pd.Series(True, index=df.index)
    for k, v in target_params.items():
        col = f"param_{k}" if not k.startswith("param_") else k
        if col in df.columns:
            if isinstance(v, float):
                mask &= np.isclose(df[col].fillna(-999.9), v, atol=1e-5)
            else:
                mask &= (df[col] == v)
        else:
            print(f"⚠️ 找不到参数列: {col}")
            return

    match_df = df[mask]
    if len(match_df) == 0:
        print(f"❌ 未在回测结果中找到匹配的参数组合: \n{target_params}")
        return
    elif len(match_df) > 1:
        print(f"⚠️ 找到 {len(match_df)} 条匹配记录，仅展示第一条。")

    row = match_df.iloc[0]
    p_name = row.get('param_name', 'Target_Grid')
    score = row.get('FINAL_SCORE', np.nan)

    print(f"\n▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼")
    print(f" 🎯 目标追踪 | 参数代号: {p_name}")
    print(f"   ► 稳健综合分 LCB: {fmt_flt(score, 4)} [公式: (均值 - 1.5*标准差) × 存活率]")
    print("────────────────────────────────────────────────────────────────────────────────")

    print(" [⚙️ 策略参数配置]")
    param_strs = [f"{p.replace('param_', '')}: {row.get(p, 'N/A')}" for p in varying_params]
    print(f"   ► {',  '.join(param_strs)}")

    print("\n [🚦 漏斗诊断状态 (揭秘为何未上榜)]")
    l1 = "✅ 通过" if row.get('L1_PASS') else f"❌ 淘汰 ({row.get('L1_REASONS', '未知')})"
    l2 = "✅ 通过" if row.get('L2_PARETO') else f"❌ 淘汰 (被其他参数支配, Rank: {row.get('PARETO_RANK', 'N/A')})"
    l3 = "✅ 通过" if row.get('L3_PASS') else f"❌ 淘汰 ({row.get('L3_REASONS', '未知')})"
    l4 = "✅ 通过" if row.get('L4_PASS') else "❌ 淘汰 (平原悬崖或容量不足)"
    l5 = "✅ 通过" if row.get('L5_PASS') else "❌ 淘汰 (年度稳定比例不足)"
    print(f"   ├─ L1 基础健康: {l1}")
    print(f"   ├─ L2 Pareto  : {l2}")
    print(f"   ├─ L3 约束偏好: {l3}")
    print(f"   ├─ L4 邻域稳定: {l4}")
    print(f"   └─ L5 时间稳定: {l5}")

    print("\n [📊 核心基础绩效]")
    print(f"   ├─ 交易统计: {fmt_int(row.get('total_closed_trades'))} 笔平仓 | 胜率: {fmt_pct_abs(row.get('win_rate'))} | 盈亏比: {fmt_flt(row.get('profit_loss_ratio'), 2)} | 期望下界: {fmt_flt(row.get('expectancy_ci_low'))}")
    print(f"   ├─ 收益状况: 总收益: {fmt_pct(row.get('total_return'))} | 年化收益: {fmt_pct(row.get('annual_return'))}")

    bull_ret = row.get('bull_regime_total_return', np.nan)
    bull_bars = row.get('bull_regime_bars', np.nan)
    bear_ret = row.get('bear_regime_total_return', np.nan)
    bear_bars = row.get('bear_regime_bars', np.nan)
    print(f"   ├─ 周期拆解: 牛市绝对收益: {fmt_pct(bull_ret)} (历经 {fmt_int(bull_bars)} 根K线) | 熊市绝对收益: {fmt_pct(bear_ret)} (历经 {fmt_int(bear_bars)} 根K线)")

    max_uw_days = row.get('max_time_under_water_days', np.nan)
    total_days = row.get('days_passed', np.nan)
    uw_str = f"{fmt_flt(max_uw_days, 1)}天"
    if pd.notna(max_uw_days) and pd.notna(total_days) and total_days > 0:
        uw_str += f" (占总时长 {fmt_pct_abs(max_uw_days / total_days, 1)})"
    mae_str = f" | ☠️最差MAE: {fmt_pct(row.get('mae_pct_worst'))}" if EVAL_SIDE == 'SHORT' else ""
    print(f"   ├─ 回撤体验: 最大回撤: {fmt_pct(row.get('max_drawdown'))} | 全局最长水下: {uw_str}{mae_str}")
    print(f"   ├─ 风险调整: {primary_obj}(主目标): {fmt_flt(row.get(primary_obj))} | 盈利月占比: {fmt_pct_abs(row.get('monthly_positive_ratio'))}")
    drop_str = f" | Top3剔除衰减: {fmt_pct(row.get('drop_top3_pnl_decay'))}" if EVAL_SIDE == 'LONG' else ""
    print(f"   └─ 集中度险: 币种HHI: {fmt_flt(row.get('asset_hhi'))} | 最赚1笔占比: {fmt_pct_abs(row.get('top1_pnl_ratio'))}{drop_str}")

    print("\n [🛡️ 参数平原与鲁棒性验证]")
    target_nbrs = fmt_int(row.get('L4_TARGET_NEIGHBOR_COUNT', 0))
    actual_nbrs = fmt_int(row.get('L4_NEIGHBOR_COUNT', 0))
    survive_nbrs = fmt_int(row.get('L4_SURVIVING_NEIGHBOR_COUNT', 0))
    print(f"   ├─ 平原稳定性: 邻居容量 [目标: {target_nbrs} | 实际: {actual_nbrs} | 存活: {survive_nbrs}] | 存活率: {fmt_pct_abs(row.get('L4_NEIGHBOR_HEALTH_RATE'), 0)}")
    print(f"   ├─ 邻域绩效评估: 主目标均值: {fmt_flt(row.get('L4_NEIGHBOR_OBJ_MEAN'))} | 颠簸度CV: {fmt_flt(row.get('L4_NEIGHBOR_OBJ_CV'), 4)}")
    cost_col = 'cost_stress_20bps_annual' if EVAL_SIDE == 'LONG' else 'cost_stress_30bps_annual'
    print(f"   └─ 成本压力测: 增加极限滑点后，年化收益变为 -> {fmt_pct(row.get(cost_col))}")

    print("\n [📅 年度表现拆解]")
    year_cols = [c for c in df.columns if c.startswith('year_') and c.endswith('_return') and 'excess' not in c]
    if year_cols:
        years = sorted(list(set([int(c.split('_')[1]) for c in year_cols])))
        for y in years:
            y_ret = row.get(f'year_{y}_return', np.nan)
            y_dd = row.get(f'year_{y}_max_dd', np.nan)
            y_bench_ret = row.get(f'benchmark_year_{y}_return', np.nan)
            excess_ret = row.get(f'year_{y}_excess_return', np.nan)
            if pd.notna(y_bench_ret):
                bench_str = f" | 基准收益: {fmt_pct(y_bench_ret):>7} | 🌟全局超额: {fmt_pct(excess_ret):>7}"
            else:
                bench_str = ""
            if pd.notna(y_ret):
                print(f"   ► {y}年: 策略收益: {fmt_pct(y_ret):>7} (最大回撤 {fmt_pct(y_dd):>7}){bench_str}")
    else:
        print("   ► (无年度拆解数据)")

    print("\n [🪙 标的盈亏贡献明细 (按净利润排序)]")
    asset_cols = [c for c in df.columns if c.startswith('asset_') and c.endswith('_net_pnl')]
    if asset_cols:
        assets_info = []
        for c in asset_cols:
            coin = c.replace('asset_', '').replace('_net_pnl', '')
            net_pnl = row.get(c, 0.0)
            trades = row.get(f'asset_{coin}_trades', 0)
            win_r = row.get(f'asset_{coin}_win_rate', np.nan)
            share = row.get(f'asset_{coin}_pnl_share', np.nan)
            if trades > 0:
                assets_info.append({'coin': coin, 'pnl': net_pnl, 'trades': trades, 'wr': win_r, 'share': share})
        assets_info = sorted(assets_info, key=lambda x: x['pnl'], reverse=True)
        for a in assets_info:
            print(f"   - {a['coin']:<6}: 净利润 ${a['pnl']:>8.2f} (利润占比: {fmt_pct_abs(a['share'])} ) | 交易: {fmt_int(a['trades']):>3}笔 | 胜率: {fmt_pct_abs(a['wr'])}")
    else:
        print("   ► (无标的拆解数据)")

    print("▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲\n")

# ═══════════════════════════════════════════════════════════════════════════
# 执行主入口
# ═══════════════════════════════════════════════════════════════════════════
def evaluate(results_path, side='LONG'):
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"找不到结果文件: {results_path}")

    df = pd.read_csv(results_path, encoding='utf-8-sig')

    print("═" * 70)
    print(f"🚀 CSM 参数评价漏斗系统 [{side} 模式]")
    print("═" * 70)

    param_cols, varying, constant = detect_param_cols(df)

    cfg = LONG_CONFIG if side == 'LONG' else SHORT_CONFIG

    df = layer1_health_filter(df, cfg['L1_HEALTH'])
    df = layer2_pareto_frontier(df, cfg['L2_PARETO'])
    df = layer3_constraint_filter(df, cfg['L3_CONSTRAINTS'])
    df = layer4_neighborhood(df, varying, cfg['PRIMARY_OBJ'], NEIGHBOR_CONFIG)
    df = layer5_time_stability(df, cfg['L5_TIME'])
    df = compute_final_score(df)

    print_funnel_summary(df)
    print_rejection_summary(df, 'L1_PASS', 'L1_REASONS', 'L1 硬生存底线')
    print_top_candidates(df, cfg['PRIMARY_OBJ'], varying, top_n=5)

    return df

if __name__ == "__main__":
    result_df = evaluate(RESULTS_PATH, side=EVAL_SIDE)

    # ==========================================
    # 🎯 指定参数跟踪
    # ==========================================
    target_check = {
        'MOM_WINDOW': 36,
        'VOL_WINDOW': 90,
        'BTC_TREND_WINDOW': 180,
        'MAX_WEIGHT': 0.25,
        'TOP_K': 1
    }
    _, varying, _ = detect_param_cols(result_df)
    primary_obj = LONG_CONFIG['PRIMARY_OBJ'] if EVAL_SIDE == 'LONG' else SHORT_CONFIG['PRIMARY_OBJ']
    print_specific_params(result_df, target_check, primary_obj, varying)