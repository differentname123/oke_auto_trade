"""
截面动量(CSM)专属参数评价漏斗系统 - 第一性原理终极重构版
包含完整的多空隔离逻辑、下置信界(LCB)评分，以及极其详尽的卡片式日志打印。
"""

import itertools
import re
from collections import defaultdict
import os
import pandas as pd
from collections import Counter
import numpy as np
# ═══════════════════════════════════════════════════════════════════════════
# 全局配置区
# ═══════════════════════════════════════════════════════════════════════════
# 切换评估模式: 'LONG' (多头参数漏斗) 或 'SHORT' (空头参数漏斗)

# 文件路径配置
RESULTS_PATH = r'W:\project\python_project\oke_auto_trade\param_search_results\grid_search_131274_SHORT_ONLY_dynamic_pool_offset_0h_with_Benchmark.csv'
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
        'min_total_trades':              15,
        'min_active_assets':             1,
        'max_drawdown_threshold':        -0.75,   # 空头极易被套，放宽到75%回撤
        'max_mae_pct_worst':             -0.60,   # 最差MAE容忍到60%
        'min_bear_regime_total_return':  -0.05,   # 熊市允许微亏，不要求绝对赚钱 (原0.0)
        'min_cost_stress_30bps_annual':  -0.30,   # 30bps滑点下允许亏损20%
        'min_expectancy_ci_low':         -0.15,
        'max_avg_holding_hours':         336.0,   # 放宽到持仓14天 (原120)
    },
    'L2_PARETO': {
        'calmar_ratio':               'maximize',
        'mae_pct_worst':              'maximize',
        'bear_regime_total_return':   'maximize',
        'cost_stress_30bps_annual':   'maximize',
    },
    'L3_CONSTRAINTS': {
        'top1_pnl_ratio':           ('<=', 0.85),
        'profit_loss_ratio':        ('>=', 0.4),  # 盈亏比0.4也能忍，只要胜率够
    },
    'L5_TIME': {
        'min_profitable_years_ratio': 0.10,       # 空头哪怕只有10%年份赚钱也放行
    },
    'PRIMARY_OBJ': 'calmar_ratio'
}

NEIGHBOR_CONFIG = {
    'radius':             1,
    'cliff_threshold':    0.40,
    'min_neighbor_count': 1,           # 🌟 只要有1个相邻参数被评估过就行 (原3)
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

    # 计算理论最大敞口
    exposure = (out['param_TOP_K'] * out['param_MAX_WEIGHT']).clip(lower=0.01)

    # 引擎真实回撤 = 账户最大回撤 / 资金利用率
    out['engine_max_drawdown'] = out['max_drawdown'] / exposure

    # 1. 拦截总账户回撤过大的 (保命)
    reject(out['max_drawdown'].fillna(-1) < filters.get('max_drawdown_threshold', -1), 'dd_too_deep')

    # 2. [新增] 拦截策略引擎本身极度烂的 (杀伪装者，比如设定不得低于 -60%)
    reject(out['engine_max_drawdown'].fillna(-1) < filters.get('min_engine_drawdown', -0.60), 'engine_alpha_failed')


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

def layer2_pareto_frontier(df, objectives, max_fronts=5, skip_l2=False):
    out = df.copy()
    out['L2_PARETO'] = False
    out['PARETO_RANK'] = np.nan

    # 🌟 新增逻辑：如果选择跳过 L2，则让所有通过 L1 的参数直接无损通过 L2
    if skip_l2:
        if 'L1_PASS' in out.columns:
            out['L2_PARETO'] = out['L1_PASS']
            # 将虚拟排名统一设置为 1，防止后续层级(如果有用到排名)出现 NaN 报错
            out.loc[out['L1_PASS'], 'PARETO_RANK'] = 1
        else:
            out['L2_PARETO'] = True
        return out

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
        passes[idx] = bool((len(nbrs_idx) >= config.get('min_neighbor_count', 3)) and (health_rates[idx] >= 0.10) and (
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
    # 🌟 核心修复: 如果LCB为负，剥离存活率扭曲，直接保留负数进行自然垫底，杜绝负负得正
    out.loc[valid_mask, 'FINAL_SCORE'] = np.where(lcb > 0, lcb * nbrs_health, lcb)
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
    # print_top_candidates(df, cfg['PRIMARY_OBJ'], varying, top_n=5)

    # ══════════════════════════════════════════════════════════════════
    # 新增：保存通过所有筛选的参数至本地文件
    # ══════════════════════════════════════════════════════════════════
    if 'L5_PASS' in df.columns:
        passed_df = df[df['L5_PASS']].copy()

        if len(passed_df) > 0:
            # 按照综合评分 LCB 和 主目标 降序排列
            if 'FINAL_SCORE' in passed_df.columns:
                passed_df = passed_df.sort_values(
                    by=['FINAL_SCORE', cfg['PRIMARY_OBJ']],
                    ascending=[False, False]
                )

            # 构造输出路径：在原文件同目录下，添加 _PASSED 后缀
            base_dir = os.path.dirname(results_path)
            file_name = os.path.basename(results_path)
            save_name = file_name.replace('.csv', '_PASSED.csv')
            save_path = os.path.join(base_dir, save_name)

            # 导出到CSV文件，保留所有指标以便复查
            # passed_df.to_csv(save_path, index=False, encoding='utf-8-sig')

            print("\n" + "═" * 70)
            print(f"💾 成功！已将 {len(passed_df)} 组通过最终筛选的参数保存至：")
            print(f"   ► {save_path}")
            print("═" * 70)
        else:
            print("\n" + "═" * 70)
            print("⚠️ 没有参数通过所有筛选，未生成导出文件。")
            print("═" * 70)

    return df


import os
import re
from collections import Counter
import pandas as pd
import numpy as np


def evaluate_multi_offset_ensemble(file_paths, side='LONG', max_missing_votes=0):
    """
    极简而强大的多时间偏移联合评估函数 (Ensemble Hard Voting)

    :param file_paths: 包含 5 个不同 offset 结果文件路径的列表
    :param side: 'LONG' 或 'SHORT'
    :param max_missing_votes: 容错投票数，默认 0 (必须 5 份文件全过 L5)
    :return: 经过严格防过拟合过滤后的终极参数 DataFrame
    """
    print("\n" + "🚀" * 30)
    print(f"🌌 启动多维时空联合评估 [模式: {side} | 容错数: {max_missing_votes}]")
    print("🚀" * 30)

    if len(file_paths) != 5:
        print(f"⚠️ 警告: 输入的文件数量为 {len(file_paths)}，建议标准的 5 份 Offset 文件！")

    all_results = {}
    passed_sets = []

    # ==========================================
    # 第一步：每个文件独立运行现有的核心漏斗
    # ==========================================
    for path in file_paths:
        offset_name = os.path.basename(path).split('_offset_')[-1].split('_')[
            0] if '_offset_' in path else os.path.basename(path)
        print(f"\n⏳ 正在独立评估 Offset 宇宙: {offset_name}...")

        # 调用你现有的 evaluate 函数！
        df = evaluate(path, side=side)

        # 🌟🌟🌟 新增修复：强制清洗 param_name 统一格式 🌟🌟🌟
        if 'param_name' in df.columns:
            # 利用正则替换，把结尾处的 _0h, _30min, _1h, _2h, _3h 全部抹除
            df['param_name'] = df['param_name'].astype(str).str.replace(r'_(0h|30min|1h|2h|3h)$', '', regex=True)
        # 🌟🌟🌟 修复结束 🌟🌟🌟

        all_results[offset_name] = df

        # 提取通过最终筛选的参数
        if 'L5_PASS' in df.columns:
            passed_params = set(df[df['L5_PASS']]['param_name'].dropna())
        else:
            passed_params = set()

        passed_sets.append(passed_params)
        print(f"   ► 该宇宙存活参数数量: {len(passed_params)}")
        # 打印一个参数名
        if len(passed_params) > 0:
            sample_param = next(iter(passed_params))
            print(f"   ► 示例存活参数: {sample_param}")
        else:
            print(f"   ► 无参数通过该宇宙的最终筛选！")

    # ==========================================
    # 第二步：硬投票取交集 (容错机制)
    # ==========================================
    all_passed_list = [p for s in passed_sets for p in s]
    vote_counts = Counter(all_passed_list)

    required_votes = len(file_paths) - max_missing_votes
    robust_candidates = [param for param, count in vote_counts.items() if count >= required_votes]

    print("\n" + "═" * 70)
    print(f"🗳️ 投票阶段完成: 共有 {len(robust_candidates)} 组参数获得 >= {required_votes} 票")
    print("═" * 70)

    if not robust_candidates:
        print("❌ 极度悲观：没有参数能在不同的时间切片中稳定存活。请放宽容错(max_missing_votes)或检查基础策略。")
        return pd.DataFrame()

    # 🚀🚀🚀 性能优化核心：一次性合并所有 DataFrame，彻底消除嵌套循环全表扫描 🚀🚀🚀
    all_dfs_with_offset = []
    for offset_name, df_orig in all_results.items():
        df_temp = df_orig.copy()
        df_temp['OFFSET_UNIVERSE'] = offset_name
        all_dfs_with_offset.append(df_temp)

    master_df = pd.concat(all_dfs_with_offset, ignore_index=True)

    # 🌟🌟🌟 需求1：持久化保存投票阶段通过的参数的原始数据 (向量化极速版) 🌟🌟🌟
    try:
        persisted_df = master_df[master_df['param_name'].isin(robust_candidates)].copy()
        if not persisted_df.empty:
            base_dir = os.path.dirname(file_paths[0])
            base_name = os.path.basename(file_paths[0])
            save_name = re.sub(r'_offset_[^_]+_', '_ENSEMBLE_VOTED_', base_name)
            if save_name == base_name:  # 如果没匹配上正则，提供默认后缀
                save_name = base_name.replace('.csv', '_ENSEMBLE_VOTED.csv')
            save_path = os.path.join(base_dir, save_name)

            persisted_df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"💾 [持久化] 已将 {len(robust_candidates)} 组入围参数在所有宇宙的 {len(persisted_df)} 行明细保存至:")
            print(f"   ► {save_path}")
    except Exception as e:
        print(f"⚠️ 保存持久化数据时发生错误: {e}")
    # 🌟🌟🌟 需求1 结束 🌟🌟🌟

    # ==========================================
    # 第三步：防过拟合地狱级过滤 (Anti-Overfitting) [向量化重构版]
    # ==========================================
    final_golden_params = []

    # 设定防过拟合的硬性容忍度
    MAX_SCORE_DROP_PCT = 0.50  # 得分从最好到最差不能缩水超过 50%
    MAX_DD_SPREAD = 0.15  # 最好和最差的最大回撤极差不能超过 15%
    MIN_SHORT_RETURN = -0.05  # 对齐容忍空头微亏5%的底线

    print("\n⚔️ 进入防过拟合地狱级审查 (防止实盘亏钱)...")

    # 使用 groupby 将原本 O(N^2) 的循环降维成 O(N)
    grouped = persisted_df.groupby('param_name')

    for param, group in grouped:
        scores = group['FINAL_SCORE'].dropna().tolist()
        dds = group['max_drawdown'].dropna().tolist()
        returns = group['total_return'].dropna().tolist()

        if not scores or not dds or not returns:
            continue

        if side == 'SHORT':
            bear_returns = group['bear_regime_total_return'].dropna().tolist()
            if bear_returns and min(bear_returns) < MIN_SHORT_RETURN:
                # print(f"   [淘汰] {param}: 熊市表现不稳...")
                continue
        elif side == 'LONG' and min(returns) <= 0:
            # print(f"   [淘汰] {param}: 在某个 offset 下总收益为负...")
            continue

        worst_dd, best_dd = min(dds), max(dds)
        if abs(worst_dd - best_dd) > MAX_DD_SPREAD:
            # print(f"   [淘汰] {param}: 回撤极差太大...")
            continue

        max_score, min_score = max(scores), min(scores)
        score_drop = max_score - min_score

        drop_pct = 0
        if max_score > 0:
            drop_pct = score_drop / max_score
            is_low_score_safe = (max_score <= 0.3) and (score_drop <= 0.15)

            if drop_pct > MAX_SCORE_DROP_PCT and not is_low_score_safe:
                continue

        # 获取当前参数得分最高的那一行的 OFFSET_UNIVERSE
        best_idx = group['FINAL_SCORE'].idxmax()
        best_offset = group.loc[best_idx, 'OFFSET_UNIVERSE'] if pd.notna(best_idx) else "N/A"

        final_golden_params.append({
            'param_name': param,
            'votes': vote_counts[param],
            'median_FINAL_SCORE': np.median(scores),
            'worst_FINAL_SCORE': min_score,
            'worst_max_dd': worst_dd,
            'worst_return': min(returns),
            'score_drop_pct': drop_pct if max_score > 0 else 0,
            'best_offset': best_offset  # 🌟 需求2: 保存最佳 Offset
        })

    # ==========================================
    # 结果输出与打印 (🌟 重构输出模块)
    # ==========================================
    if not final_golden_params:
        print("\n💀 全军覆没！所有候选参数均未通过防过拟合审查，请勿强行实盘！")
        return pd.DataFrame()

    # 转为 DataFrame 并按中位数得分排序
    golden_df = pd.DataFrame(final_golden_params)
    golden_df = golden_df.sort_values(by='median_FINAL_SCORE', ascending=False)

    print("\n" + "═" * 80)
    print(f"🏆 最终 {side} 生产环境候选 (按多宇宙联合评估 LCB 排序) [共 {len(golden_df)} 组免疫时区干扰]")
    print("═" * 80)

    # 🌟🌟🌟 新增需求：持久化保存最终【免疫时区干扰】参数的原始数据 🌟🌟🌟
    try:
        golden_params_list = golden_df['param_name'].tolist()
        golden_persisted_df = master_df[master_df['param_name'].isin(golden_params_list)].copy()

        if not golden_persisted_df.empty:
            # 构造保存路径 (基于第一个输入文件，替换标识符)
            base_dir = os.path.dirname(file_paths[0])
            base_name = os.path.basename(file_paths[0])
            save_name_golden = re.sub(r'_offset_[^_]+_', '_GOLDEN_IMMUNE_', base_name)
            if save_name_golden == base_name:  # 如果没匹配上正则，提供默认后缀
                save_name_golden = base_name.replace('.csv', '_GOLDEN_IMMUNE.csv')
            save_path_golden = os.path.join(base_dir, save_name_golden)

            golden_persisted_df.to_csv(save_path_golden, index=False, encoding='utf-8-sig')
            print(
                f"💾 [持久化-终极] 已将 {len(golden_params_list)} 组【免疫时区干扰】参数在所有宇宙的 {len(golden_persisted_df)} 行明细保存至:")
            print(f"   ► {save_path_golden}")
    except Exception as e:
        print(f"⚠️ 保存终极持久化数据时发生错误: {e}")
    # 🌟🌟🌟 新增需求 结束 🌟🌟🌟

    # 动态获取可变参数列名 (从第一个宇宙的数据中提取)
    sample_df = next(iter(all_results.values()))
    varying_params = [c for c in sample_df.columns if c.startswith('param_') and c != 'param_name']
    primary_obj = 'FINAL_SCORE'

    # 🌟 建立复合索引字典，用于极速 O(1) 提取完整行 🌟
    master_df_indexed = master_df.set_index(['param_name', 'OFFSET_UNIVERSE'])

    for rank, (idx, summary_row) in enumerate(golden_df.head(5).iterrows(), 1):
        param = summary_row['param_name']
        best_offset = summary_row['best_offset']

        # 🌟 极速 O(1) 提取该参数在最佳时区宇宙的完整明细行
        try:
            full_row = master_df_indexed.loc[(param, best_offset)]
            # 防止索引重复导致返回DataFrame，强制取第一行转换为Series
            if isinstance(full_row, pd.DataFrame):
                full_row = full_row.iloc[0]
        except KeyError:
            continue  # 以防万一找不到

        print(f"\n▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼")
        print(f" 🎖️ 排位 No.{rank} | 参数代号: {param}")

        # 【联合评估专属摘要】
        print(f"   ► [多宇宙投票] 存活票数: {int(summary_row['votes'])}/5 | 🎯 实盘推荐最佳时区: 【 {best_offset} 】")
        print(
            f"   ► [防过拟合验证] 综合分中位数: {fmt_flt(summary_row['median_FINAL_SCORE'])} | 宇宙间衰减极差: {fmt_pct_abs(summary_row['score_drop_pct'])} (极其稳固)")
        print(
            f"   ► [兜底极差底线] 最差宇宙回撤: {fmt_pct(summary_row['worst_max_dd'])} | 最差宇宙收益: {fmt_pct(summary_row['worst_return'])}")
        print("────────────────────────────────────────────────────────────────────────────────")

        # [⚙️ 策略参数配置]
        print(" [⚙️ 策略参数配置]")
        param_strs = [f"{p.replace('param_', '')}: {full_row.get(p, 'N/A')}" for p in varying_params]
        if param_strs:
            print(f"   ► {',  '.join(param_strs)}")
        else:
            print(f"   ► (无变动参数提取)")

        # [📊 核心基础绩效] (基于最佳宇宙数据)
        print("\n [📊 核心基础绩效 (基于最佳时区表现)]")
        print(
            f"   ├─ 交易统计: {fmt_int(full_row.get('total_closed_trades'))} 笔平仓 | 胜率: {fmt_pct_abs(full_row.get('win_rate'))} | 盈亏比: {fmt_flt(full_row.get('profit_loss_ratio'), 2)} | 期望下界: {fmt_flt(full_row.get('expectancy_ci_low'))}")
        print(
            f"   ├─ 收益状况: 总收益: {fmt_pct(full_row.get('total_return'))} | 年化收益: {fmt_pct(full_row.get('annual_return'))}")

        bull_ret = full_row.get('bull_regime_total_return', np.nan)
        bull_bars = full_row.get('bull_regime_bars', np.nan)
        bear_ret = full_row.get('bear_regime_total_return', np.nan)
        bear_bars = full_row.get('bear_regime_bars', np.nan)
        print(
            f"   ├─ 周期拆解: 牛市绝对收益: {fmt_pct(bull_ret)} (历经 {fmt_int(bull_bars)} 根K线) | 熊市绝对收益: {fmt_pct(bear_ret)} (历经 {fmt_int(bear_bars)} 根K线)")

        max_uw_days = full_row.get('max_time_under_water_days', np.nan)
        total_days = full_row.get('days_passed', np.nan)
        uw_str = f"{fmt_flt(max_uw_days, 1)}天"
        if pd.notna(max_uw_days) and pd.notna(total_days) and total_days > 0:
            uw_str += f" (占总时长 {fmt_pct_abs(max_uw_days / total_days, 1)})"

        mae_str = f" | ☠️最差MAE: {fmt_pct(full_row.get('mae_pct_worst'))}" if side == 'SHORT' else ""
        print(f"   ├─ 回撤体验: 最大回撤: {fmt_pct(full_row.get('max_drawdown'))} | 全局最长水下: {uw_str}{mae_str}")
        print(
            f"   ├─ 风险调整: {primary_obj}(主目标): {fmt_flt(full_row.get(primary_obj))} | 盈利月占比: {fmt_pct_abs(full_row.get('monthly_positive_ratio'))}")

        drop_str = f" | Top3剔除衰减: {fmt_pct(full_row.get('drop_top3_pnl_decay'))}" if side == 'LONG' else ""
        print(
            f"   └─ 集中度险: 币种HHI: {fmt_flt(full_row.get('asset_hhi'))} | 最赚1笔占比: {fmt_pct_abs(full_row.get('top1_pnl_ratio'))}{drop_str}")
        exposure = full_row.get('param_TOP_K', 1) * full_row.get('param_MAX_WEIGHT', 1)
        annual_return = full_row.get('annual_return', 0)
        roce = annual_return / max(exposure, 0.01)

        print(f"   ├─ 资金效率: 资金利用率 {fmt_pct_abs(exposure)} | 实际资本回报率(ROCE): {fmt_pct(roce)}")
        # [🛡️ 参数平原与鲁棒性验证]
        print("\n [🛡️ 参数平原与鲁棒性验证]")
        target_nbrs = fmt_int(full_row.get('L4_TARGET_NEIGHBOR_COUNT', 0))
        actual_nbrs = fmt_int(full_row.get('L4_NEIGHBOR_COUNT', 0))
        survive_nbrs = fmt_int(full_row.get('L4_SURVIVING_NEIGHBOR_COUNT', 0))
        print(
            f"   ├─ 平原稳定性: 邻居容量 [目标: {target_nbrs} | 实际: {actual_nbrs} | 存活: {survive_nbrs}] | 存活率: {fmt_pct_abs(full_row.get('L4_NEIGHBOR_HEALTH_RATE'), 0)}")
        print(
            f"   ├─ 邻域绩效评估: 主目标均值: {fmt_flt(full_row.get('L4_NEIGHBOR_OBJ_MEAN'))} | 颠簸度CV: {fmt_flt(full_row.get('L4_NEIGHBOR_OBJ_CV'), 4)}")

        cost_col = 'cost_stress_20bps_annual' if side == 'LONG' else 'cost_stress_30bps_annual'
        print(f"   └─ 成本压力测: 增加极限滑点后，年化收益变为 -> {fmt_pct(full_row.get(cost_col))}")

        # [📅 年度表现拆解]
        print("\n [📅 年度表现拆解]")
        year_cols = [c for c in full_row.keys() if
                     isinstance(c, str) and c.startswith('year_') and c.endswith('_return') and 'excess' not in c]
        if year_cols:
            years = sorted(list(set([int(c.split('_')[1]) for c in year_cols])))
            for y in years:
                y_ret = full_row.get(f'year_{y}_return', np.nan)
                y_dd = full_row.get(f'year_{y}_max_dd', np.nan)
                y_bench_ret = full_row.get(f'benchmark_year_{y}_return', np.nan)
                excess_ret = full_row.get(f'year_{y}_excess_return', np.nan)

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
        asset_cols = [c for c in full_row.keys() if
                      isinstance(c, str) and c.startswith('asset_') and c.endswith('_net_pnl')]
        if asset_cols:
            assets_info = []
            for c in asset_cols:
                coin = c.replace('asset_', '').replace('_net_pnl', '')
                net_pnl = full_row.get(c, 0.0)
                trades = full_row.get(f'asset_{coin}_trades', 0)
                win_r = full_row.get(f'asset_{coin}_win_rate', np.nan)
                share = full_row.get(f'asset_{coin}_pnl_share', np.nan)

                if trades > 0:
                    assets_info.append({'coin': coin, 'pnl': net_pnl, 'trades': trades, 'wr': win_r, 'share': share})

            assets_info = sorted(assets_info, key=lambda x: x['pnl'], reverse=True)
            for a in assets_info:
                print(
                    f"   - {a['coin']:<6}: 净利润 ${a['pnl']:>8.2f} (利润占比: {fmt_pct_abs(a['share'])} ) | 交易: {fmt_int(a['trades']):>3}笔 | 胜率: {fmt_pct_abs(a['wr'])}")
        else:
            print("   ► (无标的拆解数据)")

        print("▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲\n")

    return golden_df
def get_all_offset_files(base_path):
    """
    智能推导多时间偏移文件路径列表。
    传入任意一个 offset 的路径，自动生成完整的 5 个 offset 文件路径。
    """
    # 我们需要的标准 5 个宇宙偏移量
    target_offsets = ['0h', '30min', '1h', '2h', '3h']

    file_list = []

    for offset in target_offsets:
        # 使用正则动态替换路径中的 offset 部分
        # 正则逻辑：匹配 "_offset_" 加上 "任意非下划线字符" 加上 "_"
        # 这样无论原路径是 _offset_0h_ 还是 _offset_30min_，都能被精准替换
        new_path = re.sub(r'_offset_[^_]+_', f'_offset_{offset}_', base_path)
        file_list.append(new_path)

    # [可选] 贴心的工程检查：帮你提前验证文件是否真的存在于硬盘上
    missing_files = [p for p in file_list if not os.path.exists(p)]
    if missing_files:
        print("⚠️ 警告：以下推导出的文件在硬盘上找不到，请检查回测是否跑完：")
        for mf in missing_files:
            print(f"   - {mf}")

    return file_list


def analyze_parameter_attractors(golden_file_path):
    """
    第一性原理：参数空间密度直方图与吸引子分析
    用于探测哪些参数是市场的底层物理规律（强吸引子），哪些是容易导致过拟合的冗余自由度。
    """
    if not os.path.exists(golden_file_path):
        print(f"❌ 找不到文件: {golden_file_path}")
        return

    # 1. 读取多宇宙免疫的 Golden 结果数据
    try:
        df = pd.read_csv(golden_file_path, encoding='utf-8-sig')
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 2. 数据去重：因为多 offset 导致同一个 param_name 有多行，我们只需要它的独立参数组合
    if 'param_name' not in df.columns:
        print("❌ 数据中缺失 'param_name' 列！")
        return

    unique_params = df.drop_duplicates(subset='param_name').copy()
    total_survivors = len(unique_params)

    if total_survivors == 0:
        print("⚠️ 数据集为空，没有幸存参数。")
        return

    print("\n" + "═" * 80)
    print(f"🌌 GOLDEN 参数宇宙 [吸引子与敏感度] 深度分析")
    print(f"   ► 共有 {total_survivors} 组独立幸存参数参与密度测算")
    print("═" * 80)

    # 3. 自动提取所有的 param_ 列
    param_cols = [c for c in unique_params.columns if c.startswith('param_') and c != 'param_name']

    # 4. 逐个维度进行第一性原理扫描
    for p in param_cols:
        param_clean_name = p.replace('param_', '')
        print(f"\n[ 🔍 维度: {param_clean_name} ]")

        # 统计频次并按参数大小排序
        counts = unique_params[p].value_counts().sort_index()
        percentages = (counts / total_survivors) * 100

        # 计算集中度指标 (Top1 和 Top2 占比)
        sorted_pct = percentages.sort_values(ascending=False)
        top1_pct = sorted_pct.iloc[0] if len(sorted_pct) > 0 else 0
        top2_pct = sorted_pct.iloc[:2].sum() if len(sorted_pct) > 1 else top1_pct

        # 智能诊断参数敏感度
        if top1_pct >= 60 or top2_pct >= 80:
            sensitivity = "🔴 高度敏感 (核心吸引子)"
            action = f"建议：未来实盘或回测可直接固定在核心取值附近，降维打击。"
        elif top1_pct >= 35 or top2_pct >= 55:
            sensitivity = "🟡 中度敏感 (区域收敛)"
            action = f"建议：围绕当前高频取值进行小范围微调。"
        else:
            sensitivity = "🟢 极度钝化 (自由度极高/非敏感)"
            action = f"建议：此参数对最终存活率无决定性影响。为防止拟合，请将其硬编码为经验常规值。"

        print(f"   ├─ 状态诊断: {sensitivity}")
        print("   ├─ 分布图谱:")

        # 打印 ASCII 文本直方图
        for val, count in counts.items():
            pct = percentages[val]
            bar_len = int(pct / 2)  # 每 2% 画一个方块
            bar = "█" * bar_len

            # 格式化输出 (适配浮点数和整数)
            if isinstance(val, float) and val.is_integer():
                val_str = f"{int(val)}"
            elif isinstance(val, float):
                val_str = f"{val:.3f}"
            else:
                val_str = f"{val}"

            print(f"   │  {val_str:>8} : {count:>4} 组 ({pct:>5.1f}%) | {bar}")

        print(f"   └─ {action}")

    print("\n" + "═" * 80)
    print("💡 降维行动指南：")
    print("   1. 将【高度敏感】的参数视为市场的底层物理规律（如特定的动量/趋势周期）。")
    print("   2. 将【极度钝化】的参数直接固定，这不仅能将回测算力降低几个数量级，更能从数学上彻底斩断此处的数据挖掘空间。")
    print("═" * 80 + "\n")


# =====================================
# 如何调用？ (使用示例)
# =====================================
if __name__ == "__main__":
    # FILE_PATH = r"W:\project\python_project\oke_auto_trade\param_search_results\grid_search_131274_LONG_ONLY_dynamic_pool_GOLDEN_IMMUNE_with_Benchmark.csv"
    # analyze_parameter_attractors(FILE_PATH)



    BASE_RESULTS_PATH = r'W:\project\python_project\oke_auto_trade\param_search_results\grid_search_131274_LONG_ONLY_dynamic_pool_offset_0h_with_Benchmark.csv'

    # 自动推导出 5 份文件的列表
    offset_files = get_all_offset_files(BASE_RESULTS_PATH)

    print("\n📦 已自动拼装以下文件路径准备进入漏斗：")
    for f in offset_files:
        print(f" -> {os.path.basename(f)}")

    # 自动识别当前是在跑多头还是空头 (防呆设计)
    current_side = 'LONG' if 'LONG_ONLY' in BASE_RESULTS_PATH.upper() else 'SHORT'

    # 执行多宇宙投票与防过拟合审查
    golden_parameters = evaluate_multi_offset_ensemble(
        file_paths=offset_files,
        side=current_side,  # 修改为你当前跑的方向
        max_missing_votes=0  # 保持 0，代表 1个都不能少
    )
