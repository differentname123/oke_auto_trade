# ============================================================
# ⚠️ 关键：必须在 import numpy/pandas 之前设置！
# 关闭 BLAS/OpenMP 内部多线程，避免与 multiprocessing 冲突
# ============================================================
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from itertools import product
import time as time_module
import pickle
import traceback
import multiprocessing as mp


def load_and_preprocess_data(file_list):
    print("⏳ 正在解析并合并数据...")
    dfs = []
    for file in file_list:
        basename = os.path.basename(file).split('_')
        coin_main, coin_sub = basename[0], basename[1]

        df = pd.read_csv(file)
        df['open_time'] = pd.to_datetime(df['open_time'])
        df.set_index('open_time', inplace=True)

        df_main = df[['close_main']].rename(columns={'close_main': coin_main})
        df_sub = df[['close_sub']].rename(columns={'close_sub': coin_sub})

        dfs.extend([df_main, df_sub])

    # 1. 基础合并
    price_df_1m_raw = pd.concat(dfs, axis=1).sort_index()
    price_df_1m_raw = price_df_1m_raw.loc[:, ~price_df_1m_raw.columns.duplicated()]

    # ==========================================
    # 🎯 核心修正：锁定全局共有区间 (Intersection)
    # ==========================================
    common_start = max([price_df_1m_raw[c].first_valid_index() for c in price_df_1m_raw.columns])
    common_end = min([price_df_1m_raw[c].last_valid_index() for c in price_df_1m_raw.columns])

    price_df_1m = price_df_1m_raw.loc[common_start:common_end].copy()

    print(f"✅ 成功锁定公共时间窗口: {common_start} 至 {common_end}")

    # ==========================================
    # 🛠️ 数据质量与填充统计 (共有区间内)
    # ==========================================
    total_1m_rows = len(price_df_1m)
    missing_counts = price_df_1m.isna().sum()

    print(f"\n🔍 【数据质量检测：共有区间内填充统计】 (总 1m K线数: {total_1m_rows})")
    for c in price_df_1m.columns:
        missing = missing_counts[c]
        fill_ratio = (missing / total_1m_rows) * 100
        alert_flag = " ⚠️[流动性差/频繁断档]" if fill_ratio > 5.0 else ""
        print(f"   - {c:8s}: 真实缺失/需填充 {missing:>8d} 条 | 填充率 {fill_ratio:>6.2f}%{alert_flag}")
    print("-" * 50)

    price_df_1m = price_df_1m.ffill()

    # ==========================================
    # ⚠️ 优化点 1：基于1m数据提取 4h 的 OHLC 数据
    # ==========================================
    df_4h_close = price_df_1m.resample('4h').last()
    df_4h_open = price_df_1m.resample('4h').first()
    df_4h_high = price_df_1m.resample('4h').max()
    df_4h_low = price_df_1m.resample('4h').min()

    # 组合为包含所有 OHLC 字段的大 DataFrame
    price_df_4h = df_4h_close.copy()
    for c in df_4h_close.columns:
        price_df_4h[f"{c}_open"] = df_4h_open[c]
        price_df_4h[f"{c}_high"] = df_4h_high[c]
        price_df_4h[f"{c}_low"] = df_4h_low[c]

    if not df_4h_close.empty:
        print(f"\n📈 【共有区间内各标的表现 (Buy & Hold)】:")
        roll_max = df_4h_close.cummax()
        drawdowns = (df_4h_close - roll_max) / roll_max
        max_drawdowns_pct = drawdowns.min() * 100

        total_pct_change = 0.0
        num_coins = len(df_4h_close.columns)

        for c in df_4h_close.columns:
            start_price = df_4h_close[c].iloc[0]
            end_price = df_4h_close[c].iloc[-1]
            pct_change = (end_price - start_price) / start_price * 100
            total_pct_change += pct_change
            mdd = max_drawdowns_pct[c]
            print(f"   - {c:8s}: 涨跌幅 {pct_change:>8.2f}%  |  最大回撤 {mdd:>8.2f}%")

        avg_pct_change = total_pct_change / num_coins if num_coins > 0 else 0.0
        avg_mdd = max_drawdowns_pct.mean() if num_coins > 0 else 0.0

        print("-" * 50)
        print(f"   >>> 📊 基准表现 (等权 Buy & Hold):")
        print(f"            平均涨跌幅: {avg_pct_change:+.2f}%")
        print(f"            平均最大回撤: {avg_mdd:.2f}%")

        print("-" * 50)
        print(f"   >>> 📅 【全局基准：各年度等权大盘表现】")
        for year, group in df_4h_close.groupby(df_4h_close.index.year):
            start_prices = group.iloc[0]
            end_prices = group.iloc[-1]
            pct_changes = (end_prices - start_prices) / start_prices * 100
            avg_beta = pct_changes.mean()
            coin_details = ", ".join([f"{c}: {pct:+.1f}%" for c, pct in pct_changes.items()])
            print(f"            ► {year}年: {avg_beta:>+7.2f}% | [{coin_details}]")
        print("=" * 50)

    return price_df_4h


# ==========================================
# 核心：策略引擎与回测逻辑 (二元做空测试版)
# ==========================================
def run_backtest(df, param_name="默认基准参数", custom_params=None, verbose=True):
    if custom_params is None:
        custom_params = {
            'MOM_WINDOW': 20 * 6,
            'VOL_WINDOW': 20 * 6,
            'BTC_TREND_WINDOW': 60 * 6,
            'MAX_WEIGHT': 0.30
        }

    MOM_WINDOW = custom_params['MOM_WINDOW']
    VOL_WINDOW = custom_params['VOL_WINDOW']
    BTC_TREND_WINDOW = custom_params['BTC_TREND_WINDOW']
    MAX_WEIGHT = custom_params['MAX_WEIGHT']

    if verbose:
        print(f"\n🚀 启动截面做空动量回测引擎 (验证：信号驱动二元进出)... [{param_name}]")
        print(
            f"   ⚙️ 参数配置: MOM_WIN={MOM_WINDOW}, VOL_WIN={VOL_WINDOW}, BTC_TREND={BTC_TREND_WINDOW}, MAX_WT={MAX_WEIGHT}")

    TOP_K = 2
    FEE_RATE = 0.0005
    INITIAL_CAPITAL = 10000.0

    # 过滤掉衍生出来的 _open, _high, _low 列，只保留基础币种作为交易对象 (即 close)
    coins = [c for c in df.columns if not any(suffix in c for suffix in ['_open', '_high', '_low'])]
    if 'BTC' not in coins:
        raise ValueError("数据中必须包含 BTC 作为宏观开关！")

    df_close = df[coins]
    returns = df_close.pct_change(MOM_WINDOW)

    # =========================================================
    # ⚠️ 优化点 2：计算 ATR_pct，并将 adj_mom 分母替换为 ATR_pct
    # =========================================================
    tr_df = pd.DataFrame(index=df.index, columns=coins)
    for c in coins:
        high = df[f"{c}_high"]
        low = df[f"{c}_low"]
        prev_close = df_close[c].shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr_df[c] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr_df.rolling(window=VOL_WINDOW).mean()
    atr_pct = atr / df_close  # 归一化为百分比形式

    # 【新增归因日志统计计算】：计算原版 std 波动率以对比
    log_returns = np.log(df_close / df_close.shift(1))
    old_volatility = log_returns.rolling(window=VOL_WINDOW).std() * np.sqrt(365 * 6)
    old_adj_mom = returns / (old_volatility + 1e-8)

    # 新版截面动量
    adj_mom = returns / (atr_pct + 1e-8)

    # 将 volatility 变量指向 atr_pct，保证后续仓位控制(inv_vol)同样基于 ATR 进行反向加权
    volatility = atr_pct

    # 打印归因验证统计日志
    if verbose:
        print(f"   [归因验证] 动量计算分母已从 原标准差(Std) 替换为 ATR_pct !")
        rank_corr = old_adj_mom.rank(axis=1).corrwith(adj_mom.rank(axis=1), axis=1).mean()
        print(f"             -> 截面排序平均相关系数 (Rank Corr): {rank_corr:.4f} (越低说明优化影响越大)")
        print(
            f"             -> 均值对比 | ATR_pct: {atr_pct.mean().mean():.4f}  vs  原波动率: {old_volatility.mean().mean():.4f}")

    btc_ma = df['BTC'].rolling(window=BTC_TREND_WINDOW).mean()
    # ⚠️ 做空逻辑：BTC跌破均线时开启做空趋势
    btc_trend_on = df['BTC'] < btc_ma

    cash = INITIAL_CAPITAL
    positions = {coin: 0.0 for coin in coins}
    trade_logs = []
    equity_curve = []

    start_idx = max(MOM_WINDOW, VOL_WINDOW, BTC_TREND_WINDOW)

    for i in range(start_idx, len(df)):
        current_time = df.index[i]
        prices = df.iloc[i]

        current_equity = cash + sum(positions[c] * prices[c] for c in coins)
        equity_curve.append({'time': current_time, 'equity': current_equity})

        top_coins = []

        # 1. 截面动量打分（计算当前信号）
        if btc_trend_on.iloc[i]:
            current_mom = adj_mom.iloc[i].dropna()
            # ⚠️ 做空逻辑：寻找动量为负的标的，并取跌幅最深的 TOP_K
            negative_mom = current_mom[current_mom < 0]
            if not negative_mom.empty:
                top_coins = negative_mom.nsmallest(TOP_K).index.tolist()

        # [平空] 只要持仓标的触发了平空信号（不再属于 top_coins，或宏观做空关闭） -> 直接全仓平空
        for c in coins:
            # ⚠️ 做空逻辑：空头仓位表现为负数，因此判断 < 0
            if positions[c] < 0:
                if c not in top_coins:
                    cover_amount = abs(positions[c])
                    actual_cover_val = cover_amount * prices[c]
                    fee = actual_cover_val * FEE_RATE

                    positions[c] += cover_amount  # 清零
                    cash -= (actual_cover_val + fee)

                    trade_logs.append({
                        "time": current_time, "action": "COVER", "coin": c,
                        "price": prices[c], "amount": cover_amount, "value": actual_cover_val,
                        "fee": fee, "reason": "Signal Exit (Not in Top Shorts / Trend Off)"
                    })

        # [开空] 如果目前有入场信号，并且当前空仓 -> 才分配资金开空。死拿不补仓、不减仓。
        if top_coins:
            # 仅为计算初始买入权重提供参考
            inv_vol = {}
            for c in top_coins:
                c_vol = volatility[c].iloc[i]
                inv_vol[c] = 1.0 / c_vol if c_vol > 0 else 0

            total_inv_vol = sum(inv_vol.values())

            for c in top_coins:
                # 🔴 关键核心：只有完全空仓时，才进行开空。一旦开空，无论涨跌不再微调
                if positions[c] == 0:
                    if total_inv_vol > 0:
                        raw_weight = inv_vol[c] / total_inv_vol
                        target_weight = min(raw_weight, MAX_WEIGHT)
                        # 基于当前净值分配空头仓位价值
                        target_val = current_equity * target_weight

                        # ⚠️ 做空逻辑：直接按目标价值开空，因为开空会增加现金，消耗的是保证金(净值)
                        short_val = target_val
                        fee = short_val * FEE_RATE
                        short_amount = short_val / prices[c]

                        positions[c] -= short_amount  # 仓位变负
                        cash += (short_val - fee)     # 获得卖空现金，扣除手续费

                        trade_logs.append({
                            "time": current_time, "action": "SHORT", "coin": c,
                            "price": prices[c], "amount": short_amount, "value": short_val,
                            "fee": fee, "reason": "Signal Entry (Top Shorts)"
                        })

    # ==========================================
    # 🔴 核心指标与高级统计计算 (保持完全不变)
    # ==========================================
    # 注意：这里的计算对负仓位天然完美兼容，因为负仓位 * 最终价格 就是负债的精确抵扣
    final_equity = cash + sum(positions[c] * df.iloc[-1][c] for c in coins)
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL

    curve_df = pd.DataFrame(equity_curve).set_index('time')
    curve_df['cum_max'] = curve_df['equity'].cummax()
    curve_df['drawdown'] = (curve_df['equity'] - curve_df['cum_max']) / curve_df['cum_max']
    max_drawdown = curve_df['drawdown'].min()

    days_passed = (curve_df.index[-1] - curve_df.index[0]).days
    annual_return = ((final_equity / INITIAL_CAPITAL) ** (365 / days_passed) - 1) if days_passed > 0 else 0.0

    curve_df['returns'] = curve_df['equity'].pct_change()
    mean_return = curve_df['returns'].mean()
    std_return = curve_df['returns'].std()
    sharpe_ratio = (mean_return / std_return * np.sqrt(365 * 6)) if std_return > 0 else 0
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')

    win_trades, loss_trades = 0, 0
    total_profit, total_loss = 0.0, 0.0
    holding_times = []
    coin_states = {c: {'qty': 0.0, 'cost': 0.0, 'entry_time': None} for c in coins}

    for log in trade_logs:
        c, action, amt, price, fee, time = log['coin'], log['action'], log['amount'], log['price'], log['fee'], log[
            'time']

        if action == 'SHORT':
            old_qty, old_cost = coin_states[c]['qty'], coin_states[c]['cost']
            new_qty = old_qty + amt
            if new_qty > 0:
                coin_states[c]['cost'] = (old_qty * old_cost + amt * price) / new_qty
            coin_states[c]['qty'] = new_qty
            if coin_states[c]['entry_time'] is None:
                coin_states[c]['entry_time'] = time

        elif action == 'COVER':
            cost_price = coin_states[c]['cost']
            if cost_price > 0:
                # ⚠️ 做空逻辑盈亏计算反转：开仓均价 - 当前平仓价
                pnl = amt * (cost_price - price) - fee
                log['pnl'] = pnl

                if pnl > 0:
                    win_trades += 1
                    total_profit += pnl
                else:
                    loss_trades += 1
                    total_loss += abs(pnl)

                if coin_states[c]['entry_time'] is not None:
                    holding_times.append(time - coin_states[c]['entry_time'])

            coin_states[c]['qty'] -= amt
            if coin_states[c]['qty'] < 1e-6:
                coin_states[c]['qty'], coin_states[c]['cost'], coin_states[c]['entry_time'] = 0.0, 0.0, None

    total_closed_trades = win_trades + loss_trades
    win_rate = win_trades / total_closed_trades if total_closed_trades > 0 else 0.0
    avg_profit = total_profit / win_trades if win_trades > 0 else 0.0
    avg_loss = total_loss / loss_trades if loss_trades > 0 else 0.0
    profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else float('inf')
    avg_holding_time = sum(holding_times, timedelta()) / len(holding_times) if holding_times else timedelta(0)

    if verbose:
        print("\n" + "=" * 45)
        print(f"📊 【测试结果面板】: {param_name}")
        print("-" * 45)
        print(f"💸[资金与收益]")
        print(f"  初始资金:     ${INITIAL_CAPITAL:.2f}")
        print(f"  最终资金:     ${final_equity:.2f}")
        print(f"  总收益率:     {total_return * 100:.2f}%")
        print(f"  年化收益率:   {annual_return * 100:.2f}%")
        print(f"🛡️ [风险与绩效指标]")
        print(f"  最大回撤:     {max_drawdown * 100:.2f}%")
        print(f"  夏普比率:     {sharpe_ratio:.2f}")
        print(f"  卡玛比率:     {calmar_ratio:.2f}")
        print(f"⚖️ [交易统计]")
        print(f"  总触发动作:   {len(trade_logs)} 次")
        print(f"  有效平仓笔数: {total_closed_trades} 笔")
        if total_closed_trades > 0:
            print(f"  胜率 (Win%):  {win_rate * 100:.2f}%")
            print(f"  盈亏比 (P/L): {profit_loss_ratio:.2f}")
            print(f"  单笔均盈:     ${avg_profit:.2f}")
            print(f"  单笔均亏:     ${avg_loss:.2f}")
            print(f"  平均持仓时间: {avg_holding_time}")
        else:
            print("  (无有效平仓记录)")
        print("=" * 45)

    # 构造返回的统计结果字典 (扫描时使用)
    stats = {
        'param_name': param_name,
        'MOM_WINDOW': MOM_WINDOW,
        'VOL_WINDOW': VOL_WINDOW,
        'BTC_TREND_WINDOW': BTC_TREND_WINDOW,
        'MAX_WEIGHT': MAX_WEIGHT,
        'final_equity': round(final_equity, 2),
        'total_return': round(total_return, 4),
        'annual_return': round(annual_return, 4),
        'max_drawdown': round(max_drawdown, 4),
        'sharpe_ratio': round(sharpe_ratio, 3),
        'calmar_ratio': round(calmar_ratio if not np.isinf(calmar_ratio) else 999.0, 3),
        'total_actions': len(trade_logs),
        'closed_trades': total_closed_trades,
        'win_rate': round(win_rate, 4),
        'profit_loss_ratio': round(profit_loss_ratio if not np.isinf(profit_loss_ratio) else 999.0, 3),
        'avg_profit': round(avg_profit, 2),
        'avg_loss': round(avg_loss, 2),
        'avg_holding_days': round(avg_holding_time.total_seconds() / 86400 if avg_holding_time else 0, 2),
    }

    return pd.DataFrame(trade_logs), curve_df, stats


# ==========================================
# 🔴 升级版：深度验证分析模块 (包含年度 Beta 对齐)
# ==========================================
def deep_robustness_check(logs_df, curve_df, price_df, param_name="", verbose=True):
    if verbose:
        print("\n" + "🔥" * 20)
        print(f"🕵️ 【深度鲁棒性检验报告】: {param_name}")
        print("🔥" * 20)

    yearly_results = {}
    stress_results = {}

    if curve_df.empty or logs_df.empty:
        if verbose:
            print("无交易数据，无法分析。")
        return yearly_results, stress_results

    # --- 1. 分年度/分季度绩效拆解 (Alpha vs Beta) ---
    if verbose:
        print("\n📅 [年度绩效拆解] (策略表现 vs 同期市场基准):")
    curve_df['year'] = curve_df.index.year

    if 'time' in logs_df.columns:
        logs_df['year'] = logs_df['time'].dt.year
    else:
        logs_df['year'] = pd.to_datetime(logs_df['time']).dt.year

    if 'pnl' in logs_df.columns:
        # ⚠️ 做空逻辑：原为 SELL 记录利润，现为 COVER 记录利润
        sell_logs = logs_df[(logs_df['action'] == 'COVER') & (logs_df['pnl'].notna())]
    else:
        sell_logs = pd.DataFrame()

    # 过滤衍生OHLC列，计算大盘时仅看Close价格
    coins = [c for c in price_df.columns if not any(sub in c for sub in ['_open', '_high', '_low'])]

    for year, group in curve_df.groupby('year'):
        start_eq = group['equity'].iloc[0]
        end_eq = group['equity'].iloc[-1]
        y_ret = (end_eq - start_eq) / start_eq * 100

        roll_max = group['equity'].cummax()
        y_mdd = ((group['equity'] - roll_max) / roll_max).min() * 100

        year_mask = price_df.index.year == year
        year_prices = price_df[year_mask][coins]

        if not year_prices.empty:
            start_prices = year_prices.iloc[0]
            end_prices = year_prices.iloc[-1]
            avg_beta = ((end_prices - start_prices) / start_prices * 100).mean()
        else:
            avg_beta = 0.0

        y_sells = sell_logs[sell_logs['year'] == year] if not sell_logs.empty else pd.DataFrame()
        trades_cnt = len(y_sells)
        if trades_cnt > 0 and 'pnl' in y_sells.columns:
            y_win = (y_sells['pnl'] > 0).sum()
            y_win_rate = y_win / trades_cnt * 100

            sum_win = y_sells[y_sells['pnl'] > 0]['pnl'].sum()
            avg_win = sum_win / y_win if y_win > 0 else 0.0

            y_loss_cnt = trades_cnt - y_win
            sum_loss = abs(y_sells[y_sells['pnl'] <= 0]['pnl'].sum())
            avg_loss_y = sum_loss / y_loss_cnt if y_loss_cnt > 0 else 0.0

            y_pl_ratio = avg_win / avg_loss_y if avg_loss_y > 0 else 0.0

            trade_stats = f"{trades_cnt:>3d} 笔平仓 | 胜率: {y_win_rate:>5.1f}% | 盈亏比: {y_pl_ratio:>4.2f}"
        else:
            y_win_rate = 0.0
            y_pl_ratio = 0.0
            trade_stats = "  0 笔平仓"

        excess_ret = y_ret - avg_beta

        yearly_results[year] = {
            'return': round(y_ret, 2),
            'drawdown': round(y_mdd, 2),
            'beta': round(avg_beta, 2),
            'excess': round(excess_ret, 2),
            'trades': trades_cnt,
            'win_rate': round(y_win_rate, 2),
            'pl_ratio': round(y_pl_ratio, 2)
        }

        if verbose:
            print(
                f"   ► 【{year}年】 策略收益: {y_ret:>+7.2f}% (最大回撤 {y_mdd:>7.2f}%) | 等权大盘: {avg_beta:>+7.2f}% | 超额: {excess_ret:>+7.2f}%")
            print(f"            交易统计: {trade_stats}")
            print("-" * 50)

    # --- 2. 摩擦成本极限压力测试 (Transaction Cost Stress Test) ---
    if verbose:
        print("\n🌪️[滑点与手续费压力测试] (检验低胜率策略的生存力):")
    stress_fees = [0.0005, 0.0010, 0.0020, 0.0030]  # 万5, 千1, 千2, 千3
    base_fee_rate = 0.0005

    for test_fee in stress_fees:
        # ⚠️ 做空逻辑：动作标签对应修改
        buy_volume = logs_df[logs_df['action'] == 'SHORT']['value'].sum()
        sell_volume = logs_df[logs_df['action'] == 'COVER']['value'].sum()
        total_trading_volume = buy_volume + sell_volume

        extra_fee_rate = max(0, test_fee - base_fee_rate)
        extra_friction_loss = total_trading_volume * extra_fee_rate

        original_final_equity = curve_df['equity'].iloc[-1]
        stressed_equity = original_final_equity - extra_friction_loss
        stressed_return = (stressed_equity - 10000) / 10000

        bps_key = int(test_fee * 10000)
        stress_results[bps_key] = round(stressed_return * 100, 2)

        if verbose:
            status = "✅ 存活" if stressed_return > 0 else "💀 破产"
            print(f"   - 单边综合成本 {test_fee * 10000:>2.0f} bps: 最终收益率 {stressed_return * 100:>8.2f}%[{status}]")

    if verbose:
        print("=" * 60)

    return yearly_results, stress_results


# ==========================================
# 🆕 多进程支持: 全局变量 + worker (必须在模块顶层，否则无法 pickle)
# ==========================================
_GLOBAL_DF = None


def _init_worker(df):
    """子进程初始化函数：将大 DataFrame 设为全局，避免每次任务重复序列化"""
    global _GLOBAL_DF
    _GLOBAL_DF = df


def _worker_run_backtest(task):
    """子进程任务函数"""
    idx, params = task
    df = _GLOBAL_DF

    if df is None:
        return {'idx': idx, 'success': False, 'result': None,
                'error': '_GLOBAL_DF is None (worker init failed)'}

    param_name = (f"P{idx + 1:05d}_M{params['MOM_WINDOW']}_V{params['VOL_WINDOW']}"
                  f"_T{params['BTC_TREND_WINDOW']}_W{int(params['MAX_WEIGHT'] * 100)}")

    try:
        logs_df, curve_df, stats = run_backtest(
            df, param_name=param_name, custom_params=params, verbose=False
        )
        yearly_results, stress_results = deep_robustness_check(
            logs_df, curve_df, df, param_name=param_name, verbose=False
        )

        full_result = dict(stats)
        for year, ydata in yearly_results.items():
            full_result[f'Y{year}_return'] = ydata['return']
            full_result[f'Y{year}_drawdown'] = ydata['drawdown']
            full_result[f'Y{year}_beta'] = ydata['beta']
            full_result[f'Y{year}_excess'] = ydata['excess']
            full_result[f'Y{year}_trades'] = ydata['trades']
            full_result[f'Y{year}_win_rate'] = ydata['win_rate']
            full_result[f'Y{year}_pl_ratio'] = ydata['pl_ratio']

        for fee_bps, ret in stress_results.items():
            full_result[f'stress_{fee_bps}_return'] = ret
            full_result[f'stress_{fee_bps}_survive'] = 1 if ret > 0 else 0

        return {'idx': idx, 'success': True, 'result': full_result, 'error': None}
    except Exception as e:
        return {'idx': idx, 'success': False, 'result': None,
                'error': f"{e}\n{traceback.format_exc()}"}


# ==========================================
# 🆕 综合评分函数 (多维度归一化加权)
# ==========================================
def compute_composite_scores(df):
    df = df.copy()

    # 1. 年度收益相关指标
    year_cols = [c for c in df.columns if c.startswith('Y') and c.endswith('_return')]
    if year_cols:
        df['worst_year_return'] = df[year_cols].min(axis=1, skipna=True)
        df['best_year_return'] = df[year_cols].max(axis=1, skipna=True)
        df['mean_year_return'] = df[year_cols].mean(axis=1, skipna=True)
        df['std_year_return'] = df[year_cols].std(axis=1, skipna=True).fillna(0)
        df['positive_years'] = (df[year_cols] > 0).sum(axis=1)
        df['annual_consistency'] = df['mean_year_return'] / (df['std_year_return'] + 1e-8)

    # 2. 滑点存活档位
    stress_pct_cols = [c for c in df.columns if c.startswith('stress_') and c.endswith('_return')]
    if stress_pct_cols:
        df['stress_survival_count'] = (df[stress_pct_cols] > 0).sum(axis=1)
    else:
        df['stress_survival_count'] = 0

    # 3. 期望值
    df['expectancy'] = df['win_rate'] * df['avg_profit'] - (1 - df['win_rate']) * df['avg_loss']

    # 4. 硬性筛选 (推荐参数必须全通过)
    hard_conditions = (
        (df['total_return'] > 0) &
        (df['max_drawdown'] > -0.40) &
        (df['closed_trades'] >= 50)
    )
    if 'worst_year_return' in df.columns:
        hard_conditions = hard_conditions & (df['worst_year_return'] > -25)
    if 'stress_5_return' in df.columns:
        hard_conditions = hard_conditions & (df['stress_5_return'] > 0)
    df['pass_hard_filter'] = hard_conditions.astype(int)

    # 5. 归一化函数 (min-max)
    def normalize(s, lo=None, hi=None):
        s = s.copy()
        if lo is not None:
            s = s.clip(lower=lo)
        if hi is not None:
            s = s.clip(upper=hi)
        s_min, s_max = s.min(), s.max()
        if s_max == s_min:
            return pd.Series([0.5] * len(s), index=s.index)
        return (s - s_min) / (s_max - s_min)

    # 6. 各维度评分 (0~1)
    df['score_calmar'] = normalize(df['calmar_ratio'], lo=-5, hi=10)
    df['score_annual_return'] = normalize(df['annual_return'], lo=-1, hi=5)
    df['score_max_dd'] = normalize(df['max_drawdown'])  # 已是负值，越大越好
    df['score_pl_ratio'] = normalize(df['profit_loss_ratio'], lo=0, hi=5)

    if 'worst_year_return' in df.columns:
        df['score_worst_year'] = normalize(df['worst_year_return'])
    else:
        df['score_worst_year'] = 0.5

    if 'annual_consistency' in df.columns:
        df['score_consistency'] = normalize(df['annual_consistency'], lo=-5, hi=5)
    else:
        df['score_consistency'] = 0.5

    df['score_stress'] = normalize(df['stress_survival_count'])

    # 7. 加权综合评分
    df['composite_score'] = (
        df['score_calmar'] * 0.25 +
        df['score_annual_return'] * 0.15 +
        df['score_max_dd'] * 0.15 +
        df['score_worst_year'] * 0.20 +
        df['score_consistency'] * 0.10 +
        df['score_stress'] * 0.10 +
        df['score_pl_ratio'] * 0.05
    )

    # 通过硬筛选的额外加分 0.1 (强烈推荐项)
    df['composite_score'] = (df['composite_score'] + df['pass_hard_filter'] * 0.1).round(4)

    return df


# ==========================================
# 🆕 参数邻域稳定性分析 (识别参数高原 vs 尖峰)
# ==========================================
def compute_parameter_stability(results_df_sorted, top_n=50, neighbor_radius=1):
    """
    对 Top N 参数计算其参数空间相邻邻居的平均得分
    高原参数(周围都好)优先于尖峰参数(独苗一个高分)
    """
    if 'composite_score' not in results_df_sorted.columns or results_df_sorted.empty:
        return pd.DataFrame()

    keys = ['MOM_WINDOW', 'VOL_WINDOW', 'BTC_TREND_WINDOW', 'MAX_WEIGHT']

    # 全部参数组合 -> 得分映射表
    param_to_score = {}
    for _, row in results_df_sorted.iterrows():
        key = tuple(row[k] for k in keys)
        param_to_score[key] = row['composite_score']

    # 各维度的去重排序值列表
    unique_values = {}
    for i, k in enumerate(keys):
        unique_values[k] = sorted(set(p[i] for p in param_to_score.keys()))

    stability_records = []
    for _, row in results_df_sorted.head(top_n).iterrows():
        key = tuple(row[k] for k in keys)
        own_score = row['composite_score']

        neighbor_scores = []
        for i, k in enumerate(keys):
            vals = unique_values[k]
            try:
                cur_idx = vals.index(key[i])
            except ValueError:
                continue

            for offset in range(-neighbor_radius, neighbor_radius + 1):
                if offset == 0:
                    continue
                nb_idx = cur_idx + offset
                if 0 <= nb_idx < len(vals):
                    nb_key = list(key)
                    nb_key[i] = vals[nb_idx]
                    nb_key = tuple(nb_key)
                    if nb_key in param_to_score:
                        neighbor_scores.append(param_to_score[nb_key])

        if neighbor_scores:
            avg_neighbor = float(np.mean(neighbor_scores))
            min_neighbor = float(np.min(neighbor_scores))
            # 稳定性得分: 自身分数 + 邻居均分 各占一半，再扣除一些"邻居最差跌幅"
            stability_score = (own_score + avg_neighbor) / 2 - max(0, own_score - min_neighbor) * 0.3
        else:
            avg_neighbor = own_score
            min_neighbor = own_score
            stability_score = own_score * 0.7  # 没邻居的扣分

        rec = dict(row)
        rec['neighbor_count'] = len(neighbor_scores)
        rec['neighbor_avg_score'] = round(avg_neighbor, 4)
        rec['neighbor_min_score'] = round(min_neighbor, 4)
        rec['stability_score'] = round(stability_score, 4)
        stability_records.append(rec)

    stability_df = pd.DataFrame(stability_records).sort_values(
        'stability_score', ascending=False).reset_index(drop=True)
    return stability_df


# ==========================================
# 🆕 参数网格搜索主函数 (多进程并行 + 断点续传)
# ==========================================
def parameter_grid_search(df, param_grid, output_dir="./param_search_results",
                          top_n=30, use_parallel=True, n_workers=None,
                          checkpoint_interval=50, resume_from=None):
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combinations = list(product(*values))
    total_combos = len(combinations)

    print("\n" + "=" * 70)
    print(f"🔬 启动参数网格搜索")
    print(f"📊 参数空间:")
    for k, v in param_grid.items():
        print(f"     - {k}: {v}")
    print(f"📊 总组合数: {total_combos}")

    os.makedirs(output_dir, exist_ok=True)

    # === 断点续传：尝试从 checkpoint 恢复 ===
    completed_results = []
    completed_idx_set = set()
    if resume_from and os.path.exists(resume_from):
        try:
            with open(resume_from, 'rb') as f:
                completed_results = pickle.load(f)
            completed_idx_set = {r['idx'] for r in completed_results if r.get('success')}
            print(f"♻️  从 checkpoint 恢复: 已完成 {len(completed_idx_set)} 组")
        except Exception as e:
            print(f"⚠️  加载 checkpoint 失败: {e}")
            completed_results = []
            completed_idx_set = set()

    # 准备待执行任务 (跳过已完成的 idx)
    tasks = []
    for idx, combo in enumerate(combinations):
        if idx in completed_idx_set:
            continue
        params = dict(zip(keys, combo))
        tasks.append((idx, params))

    print(f"📊 待执行任务: {len(tasks)} 组 (跳过已完成 {len(completed_idx_set)} 组)")

    # 并行配置
    cpu_count = mp.cpu_count()
    if use_parallel and len(tasks) > 0:
        if n_workers is None:
            n_workers = max(1, cpu_count - 1)
        n_workers = min(n_workers, len(tasks), cpu_count)
        print(f"🚀 模式: 多进程并行 ({n_workers}/{cpu_count} cores)")
    else:
        print(f"🐢 模式: 单进程顺序")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(output_dir, f"checkpoint_{timestamp}.pkl")

    all_results = list(completed_results)
    new_completed = 0
    start_time = time_module.time()

    def _save_checkpoint():
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(all_results, f)
        except Exception as e:
            print(f"⚠️  保存 checkpoint 失败: {e}")

    def _print_progress(completed_count, result):
        elapsed = time_module.time() - start_time
        avg_time = elapsed / max(1, new_completed)
        remaining_tasks = len(tasks) - new_completed
        eta_min = (avg_time * remaining_tasks) / 60

        if result['success']:
            r = result['result']
            print(f"  [{completed_count:>5d}/{total_combos}] "
                  f"M={r['MOM_WINDOW']:>3d} V={r['VOL_WINDOW']:>3d} "
                  f"T={r['BTC_TREND_WINDOW']:>4d} W={r['MAX_WEIGHT']:.2f} | "
                  f"Ret:{r['total_return'] * 100:>+7.1f}% "
                  f"DD:{r['max_drawdown'] * 100:>6.1f}% "
                  f"Cal:{r['calmar_ratio']:>5.2f} "
                  f"Trd:{r['closed_trades']:>3d} | "
                  f"ETA:{eta_min:>5.1f}m")
        else:
            print(f"  [{completed_count:>5d}/{total_combos}] ❌ idx={result['idx']} 失败")

    # === 执行任务 ===
    if len(tasks) > 0:
        try:
            if use_parallel:
                with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(df,)) as pool:
                    for result in pool.imap_unordered(_worker_run_backtest, tasks):
                        all_results.append(result)
                        new_completed += 1
                        completed_count = len(all_results)
                        _print_progress(completed_count, result)

                        if new_completed % checkpoint_interval == 0:
                            _save_checkpoint()
            else:
                global _GLOBAL_DF
                _GLOBAL_DF = df
                for task in tasks:
                    result = _worker_run_backtest(task)
                    all_results.append(result)
                    new_completed += 1
                    completed_count = len(all_results)
                    _print_progress(completed_count, result)

                    if new_completed % checkpoint_interval == 0:
                        _save_checkpoint()
        except KeyboardInterrupt:
            print("\n⚠️  用户中断，正在保存 checkpoint...")
            _save_checkpoint()
            print(f"💾 已保存到: {checkpoint_path}")
            print(f"   下次启动设置 resume_from='{checkpoint_path}' 即可续跑")
            raise

    # 最终保存 checkpoint
    _save_checkpoint()
    print(f"\n💾 Checkpoint: {checkpoint_path}")

    # === 后处理 ===
    success_results = [r['result'] for r in all_results if r.get('success')]
    failed_count = len(all_results) - len(success_results)
    print(f"✅ 成功: {len(success_results)} 组 | ❌ 失败: {failed_count} 组")

    if not success_results:
        print("❌ 无任何成功结果。")
        return None

    results_df = pd.DataFrame(success_results)
    results_df = compute_composite_scores(results_df)
    results_df_sorted = results_df.sort_values('composite_score', ascending=False).reset_index(drop=True)

    # 计算参数邻域稳定性
    stability_df = compute_parameter_stability(results_df_sorted, top_n=max(top_n * 2, 50), neighbor_radius=1)

    # === 保存 CSV (完整) ===
    full_csv = os.path.join(output_dir, f"full_results_{timestamp}.csv")
    results_df_sorted.to_csv(full_csv, index=False, encoding='utf-8-sig')
    print(f"💾 完整结果 CSV: {full_csv}")

    # === 保存稳定性分析 CSV ===
    if not stability_df.empty:
        stability_csv = os.path.join(output_dir, f"stability_{timestamp}.csv")
        stability_df.to_csv(stability_csv, index=False, encoding='utf-8-sig')
        print(f"💾 邻域稳定性 CSV: {stability_csv}")

    # === Excel 多维度排行榜 ===
    try:
        excel_path = os.path.join(output_dir, f"rankings_{timestamp}.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            results_df_sorted.head(top_n).to_excel(writer, sheet_name='综合评分Top', index=False)
            results_df.sort_values('calmar_ratio', ascending=False).head(top_n).to_excel(
                writer, sheet_name='卡玛比率Top', index=False)
            results_df.sort_values('total_return', ascending=False).head(top_n).to_excel(
                writer, sheet_name='总收益Top', index=False)
            results_df.sort_values('max_drawdown', ascending=False).head(top_n).to_excel(
                writer, sheet_name='最小回撤Top', index=False)
            results_df.sort_values('sharpe_ratio', ascending=False).head(top_n).to_excel(
                writer, sheet_name='夏普比率Top', index=False)

            if 'worst_year_return' in results_df.columns:
                results_df.sort_values('worst_year_return', ascending=False).head(top_n).to_excel(
                    writer, sheet_name='最差年度Top', index=False)
            if 'annual_consistency' in results_df.columns:
                results_df.sort_values('annual_consistency', ascending=False).head(top_n).to_excel(
                    writer, sheet_name='年度稳定性Top', index=False)
            if 'stress_survival_count' in results_df.columns:
                results_df.sort_values(['stress_survival_count', 'composite_score'],
                                       ascending=[False, False]).head(top_n).to_excel(
                    writer, sheet_name='抗滑点Top', index=False)

            passed = results_df[results_df['pass_hard_filter'] == 1].sort_values(
                'composite_score', ascending=False)
            if not passed.empty:
                passed.to_excel(writer, sheet_name='通过硬筛选', index=False)

            if not stability_df.empty:
                stability_df.head(top_n).to_excel(writer, sheet_name='参数邻域稳定性Top', index=False)

        print(f"💾 多维度 Excel:  {excel_path}")
    except ImportError:
        print("⚠️  未安装 openpyxl，跳过 Excel 导出。可执行: pip install openpyxl")
    except Exception as e:
        print(f"⚠️  Excel 导出失败: {e}")

    print_top_results(results_df_sorted, stability_df, top_n=10)

    return results_df_sorted


# ==========================================
# 🆕 终端友好打印 Top 结果
# ==========================================
def print_top_results(results_df, stability_df=None, top_n=10):
    print("\n" + "=" * 110)
    print(f"🏆 综合评分 Top {top_n} 参数组合")
    print("=" * 110)

    cols_to_show = [
        'MOM_WINDOW', 'VOL_WINDOW', 'BTC_TREND_WINDOW', 'MAX_WEIGHT',
        'total_return', 'annual_return', 'max_drawdown', 'calmar_ratio',
        'closed_trades', 'win_rate', 'profit_loss_ratio',
        'worst_year_return', 'stress_survival_count', 'pass_hard_filter', 'composite_score'
    ]
    cols_exist = [c for c in cols_to_show if c in results_df.columns]
    top_view = results_df.head(top_n)[cols_exist].copy()

    if 'total_return' in top_view.columns:
        top_view['total_return'] = (top_view['total_return'] * 100).round(2).astype(str) + '%'
    if 'annual_return' in top_view.columns:
        top_view['annual_return'] = (top_view['annual_return'] * 100).round(2).astype(str) + '%'
    if 'max_drawdown' in top_view.columns:
        top_view['max_drawdown'] = (top_view['max_drawdown'] * 100).round(2).astype(str) + '%'
    if 'win_rate' in top_view.columns:
        top_view['win_rate'] = (top_view['win_rate'] * 100).round(2).astype(str) + '%'

    print(top_view.to_string(index=False))

    if not results_df.empty:
        best = results_df.iloc[0]
        print(f"\n🥇 综合评分最优参数:")
        print(f"   MOM_WINDOW       = {int(best['MOM_WINDOW'])}")
        print(f"   VOL_WINDOW       = {int(best['VOL_WINDOW'])}")
        print(f"   BTC_TREND_WINDOW = {int(best['BTC_TREND_WINDOW'])}")
        print(f"   MAX_WEIGHT       = {best['MAX_WEIGHT']:.2f}")
        print(f"   ----")
        print(f"   总收益率         = {best['total_return'] * 100:+.2f}%")
        print(f"   年化收益率       = {best['annual_return'] * 100:+.2f}%")
        print(f"   最大回撤         = {best['max_drawdown'] * 100:.2f}%")
        print(f"   卡玛比率         = {best['calmar_ratio']:.2f}")
        print(f"   是否通过硬筛选   = {'✅ 是' if best['pass_hard_filter'] == 1 else '❌ 否'}")
        print(f"   综合评分         = {best['composite_score']:.4f}")

        passed = results_df[results_df['pass_hard_filter'] == 1]
        if not passed.empty and (best['pass_hard_filter'] != 1):
            best_passed = passed.iloc[0]
            print(f"\n🔒 [硬筛选过滤后] 推荐参数 (满足全部稳健性条件):")
            print(f"   MOM={int(best_passed['MOM_WINDOW'])} VOL={int(best_passed['VOL_WINDOW'])} "
                  f"T={int(best_passed['BTC_TREND_WINDOW'])} W={best_passed['MAX_WEIGHT']:.2f} | "
                  f"Ret:{best_passed['total_return'] * 100:+.2f}% "
                  f"Cal:{best_passed['calmar_ratio']:.2f} "
                  f"Score:{best_passed['composite_score']:.4f}")

    # === 邻域稳定性 Top 输出 ===
    if stability_df is not None and not stability_df.empty:
        print("\n" + "=" * 110)
        print(f"🌄 【参数邻域稳定性】Top 10 (高原参数：周围一圈都好，强烈推荐)")
        print("=" * 110)
        stab_cols = ['MOM_WINDOW', 'VOL_WINDOW', 'BTC_TREND_WINDOW', 'MAX_WEIGHT',
                     'total_return', 'max_drawdown', 'calmar_ratio',
                     'composite_score', 'neighbor_avg_score', 'neighbor_min_score',
                     'neighbor_count', 'stability_score']
        stab_cols_exist = [c for c in stab_cols if c in stability_df.columns]
        stab_view = stability_df.head(10)[stab_cols_exist].copy()
        if 'total_return' in stab_view.columns:
            stab_view['total_return'] = (stab_view['total_return'] * 100).round(2).astype(str) + '%'
        if 'max_drawdown' in stab_view.columns:
            stab_view['max_drawdown'] = (stab_view['max_drawdown'] * 100).round(2).astype(str) + '%'
        print(stab_view.to_string(index=False))

        best_stab = stability_df.iloc[0]
        print(f"\n🌟 [邻域稳定性最优] 推荐参数 (建议优先选这个！):")
        print(f"   MOM_WINDOW       = {int(best_stab['MOM_WINDOW'])}")
        print(f"   VOL_WINDOW       = {int(best_stab['VOL_WINDOW'])}")
        print(f"   BTC_TREND_WINDOW = {int(best_stab['BTC_TREND_WINDOW'])}")
        print(f"   MAX_WEIGHT       = {best_stab['MAX_WEIGHT']:.2f}")
        print(f"   总收益率: {best_stab['total_return'] * 100:+.2f}% | "
              f"卡玛: {best_stab['calmar_ratio']:.2f} | "
              f"自身评分: {best_stab['composite_score']:.4f} | "
              f"邻居均分: {best_stab['neighbor_avg_score']:.4f}")
    print("=" * 110)


# ==========================================
# 主程序执行入口
# ==========================================
if __name__ == "__main__":
    file_list = ["kline_data/BTC_ETH_1m.csv", "kline_data/DOGE_SOL_1m.csv", "kline_data/TON_XRP_1m.csv"]
    df_4h = load_and_preprocess_data(file_list)

    # ============================================================
    # 🎛️ 模式选择: "scan" = 参数扫描 / "test" = 原 4 组测试
    # ============================================================
    MODE = "scan"

    if MODE == "test":
        test_scenarios = [
            {"name": "基准参数",
             "params": {'MOM_WINDOW': 20 * 6, 'VOL_WINDOW': 20 * 6, 'BTC_TREND_WINDOW': 60 * 6, 'MAX_WEIGHT': 0.30}},
            {"name": "挑战组 1 (短期)",
             "params": {'MOM_WINDOW': 10 * 6, 'VOL_WINDOW': 10 * 6, 'BTC_TREND_WINDOW': 30 * 6, 'MAX_WEIGHT': 0.30}},
            {"name": "挑战组 2 (长期)",
             "params": {'MOM_WINDOW': 30 * 6, 'VOL_WINDOW': 30 * 6, 'BTC_TREND_WINDOW': 90 * 6, 'MAX_WEIGHT': 0.30}},
            {"name": "挑战组 3 (低仓)",
             "params": {'MOM_WINDOW': 20 * 6, 'VOL_WINDOW': 20 * 6, 'BTC_TREND_WINDOW': 60 * 6, 'MAX_WEIGHT': 0.15}}
        ]
        for scenario in test_scenarios:
            logs_df, curve_df, _ = run_backtest(df_4h, param_name=scenario["name"], custom_params=scenario["params"])
            deep_robustness_check(logs_df, curve_df, df_4h, param_name=scenario["name"])
        print("\n✅ 所有参数组敏感性及深度检验执行完毕。")

    elif MODE == "scan":
        # ============================================================
        # 📐 参数空间预设 (基于 4h K线，每天 6 根)
        # ============================================================

        # 🚀 MICRO (~12 组)：调试用
        PARAM_GRID_MICRO = {
            'MOM_WINDOW': [60, 120, 180],
            'VOL_WINDOW': [120],
            'BTC_TREND_WINDOW': [360],
            'MAX_WEIGHT': [0.15, 0.20, 0.25, 0.30]
        }

        # ⚡ LIGHT (~36 组)
        PARAM_GRID_LIGHT = {
            'MOM_WINDOW': [60, 120, 180],
            'VOL_WINDOW': [120],
            'BTC_TREND_WINDOW': [180, 360, 540],
            'MAX_WEIGHT': [0.15, 0.20, 0.25, 0.30]
        }

        # 🔥 STANDARD (~180 组)
        PARAM_GRID_STANDARD = {
            'MOM_WINDOW': [60, 90, 120, 150, 180],
            'VOL_WINDOW': [60, 120, 180],
            'BTC_TREND_WINDOW': [180, 360, 540],
            'MAX_WEIGHT': [0.15, 0.20, 0.25, 0.30]
        }

        # 💪 LARGE (~720 组，8核约 1.5h)
        # MOM 7.5~40天 / VOL 10~30天 / TREND 30~120天 / 仓位 15%~50%
        PARAM_GRID_LARGE = {
            'MOM_WINDOW': [60, 90, 120, 150, 180, 240],            # 10/15/20/25/30/40 天
            'VOL_WINDOW': [60, 90, 120, 180],                       # 10/15/20/30 天
            'BTC_TREND_WINDOW': [180, 270, 360, 540, 720],          # 30/45/60/90/120 天
            'MAX_WEIGHT': [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
        }
        # 6 × 4 × 5 × 6 = 720

        # 🌌 ULTRA (~1728 组，8核约 3.5h) ✅ 推荐
        # MOM 7.5~50天 / VOL 7.5~40天 / TREND 15~120天 / 仓位 10%~40%
        PARAM_GRID_ULTRA = {
            'MOM_WINDOW': [45, 60, 90, 120, 150, 180, 240, 300],    # 7.5/10/15/20/25/30/40/50 天
            'VOL_WINDOW': [45, 60, 90, 120, 180, 240],              # 7.5/10/15/20/30/40 天
            'BTC_TREND_WINDOW': [90, 180, 270, 360, 540, 720],      # 15/30/45/60/90/120 天
            'MAX_WEIGHT': [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
        }
        # 8 × 6 × 6 × 6 = 1728

        # 💀 INSANE (~3360 组，8核约 7h)
        # MOM 5~60天 / VOL 5~40天 / TREND 15~180天 / 仓位 10%~40%
        PARAM_GRID_INSANE = {
            'MOM_WINDOW': [30, 45, 60, 90, 120, 150, 180, 240, 300, 360],  # 5~60 天
            'VOL_WINDOW': [30, 45, 60, 90, 120, 150, 180, 240],            # 5~40 天
            'BTC_TREND_WINDOW': [90, 180, 270, 360, 540, 720, 1080],       # 15~180 天
            'MAX_WEIGHT': [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
        }
        # 10 × 8 × 7 × 6 = 3360

        # 🔥💀 EXTREME (~6370 组，8核约 13h)：彻底覆盖
        PARAM_GRID_EXTREME = {
            'MOM_WINDOW': [24, 36, 48, 60, 72, 90, 120, 150, 180, 210, 240, 300, 360],  # 4~60天 13个
            'VOL_WINDOW': [24, 36, 48, 60, 90, 120, 150, 180, 240, 360],                # 4~60天 10个
            'BTC_TREND_WINDOW': [90, 180, 270, 360, 540, 720, 1080],                    # 7个
            'MAX_WEIGHT': [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]                    # 7个
        }
        # 13 × 10 × 7 × 7 = 6370

        # ============================================================
        # 👇 选择参数空间 (推荐 ULTRA 起步)
        # ============================================================
        param_grid = PARAM_GRID_EXTREME
        # param_grid = PARAM_GRID_LARGE
        # param_grid = PARAM_GRID_INSANE
        # param_grid = PARAM_GRID_EXTREME

        # ============================================================
        # 并行 & 断点续传配置
        # ============================================================
        USE_PARALLEL = True       # True=多进程并行 (强烈推荐)
        N_WORKERS = None          # None = 自动 (CPU核数-1)；可手动设 4/8/16 等
        CHECKPOINT_INTERVAL = 50  # 每完成多少组保存一次中间结果
        RESUME_FROM = None        # 中断后续跑：填入之前的 checkpoint_xxx.pkl 完整路径

        results_df = parameter_grid_search(
            df_4h,
            param_grid,
            output_dir="./param_search_results_short",
            top_n=30,
            use_parallel=USE_PARALLEL,
            n_workers=N_WORKERS,
            checkpoint_interval=CHECKPOINT_INTERVAL,
            resume_from=RESUME_FROM
        )

        print("\n✅ 参数扫描全部完成。请查看 ./param_search_results/ 目录下的 CSV 和 Excel 文件。")