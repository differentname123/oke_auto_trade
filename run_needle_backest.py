import traceback
import pandas as pd
import numpy as np
from datetime import timedelta
import itertools
import os
import time
from multiprocessing import Pool


# ==========================================
# 0. 新增：数据重采样函数
# ==========================================
def resample_data(df, minutes):
    """
    将分钟数据合并为指定周期的K线
    """
    if minutes <= 1:
        return df.copy()

    # 确保时间列是 datetime 类型
    df = df.copy()
    # 假设此时 df 已经有 open_time 且为 datetime

    # 定义聚合规则
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    # 执行重采样 (on='open_time' 指定根据时间列聚合)
    # rule = f'{minutes}T' 代表分钟级别
    df_resampled = df.resample(f'{minutes}T', on='open_time').agg(agg_dict).dropna().reset_index()

    return df_resampled


# ==========================================
# 1. 核心计算函数
# ==========================================

def calculate_indicators(df, config):
    """
    计算技术指标，生成原始信号
    """
    df = df.copy()

    # 提取参数
    ma_period = config['ma_period']
    std_dev_mult = config['std_dev_mult']
    vol_mult = config['vol_mult']

    # 基础指标
    df['ma'] = df['close'].rolling(ma_period).mean()
    df['std'] = df['close'].rolling(ma_period).std()
    df['lower_band'] = df['ma'] - (std_dev_mult * df['std'])
    df['vol_ma'] = df['volume'].rolling(ma_period).mean()

    # --- 信号计算逻辑 ---
    cond_price = df['low'] < df['lower_band']
    cond_vol = df['volume'] > (df['vol_ma'] * vol_mult)
    cond_shape = df['close'] > df['lower_band']

    # 综合信号
    df['signal_long'] = cond_price & cond_vol & cond_shape

    return df


# ==========================================
# 2. 核心回测引擎 (已修改：支持多单持仓)
# ==========================================
def run_backtest(df, config):
    """
    执行回测 - 支持多单并行
    """
    # 预计算指标
    df = calculate_indicators(df, config)
    records = df.to_dict('records')

    trades = []

    # --- 修改点：使用列表存储所有当前持仓 ---
    # 每个元素是一个字典，包含该订单的独特信息
    positions = []

    # 用于统计持仓数量分布
    active_counts = []

    cum_realized_profit = 0.0

    # 提取配置参数
    ma_period = config['ma_period']
    capital = config['capital_per_trade']
    fee_rate = config['fee_rate']
    sl_pct = config['sl_pct']
    exit_mode = config['exit_mode']
    tp_ratio = config.get('tp_ratio', 2.0)

    # 遍历K线
    for i in range(ma_period, len(records)):
        curr_bar = records[i]
        prev_bar = records[i - 1]
        last_close = prev_bar['close']
        current_ma = curr_bar['ma']

        # --- A. 构建价格路径 ---
        if curr_bar['close'] >= curr_bar['open']:
            price_path = [last_close, curr_bar['open'], curr_bar['low'], curr_bar['high'], curr_bar['close']]
            path_types = ['last_close', 'open', 'low', 'high', 'close']
        else:
            price_path = [last_close, curr_bar['open'], curr_bar['high'], curr_bar['low'], curr_bar['close']]
            path_types = ['last_close', 'open', 'high', 'low', 'close']

        # 标记本根K线是否已经开过仓 (防止一根K线内多次触发开仓，通常一根K线只响应一次信号)
        entry_action_this_bar = False

        # --- B. 路径游走 ---
        for idx, price in enumerate(price_path):
            p_type = path_types[idx]

            # 1. 检查所有现有持仓的止盈/止损 (遍历副本以允许修改原列表)
            # 使用 positions[:] 创建副本进行遍历
            for pos in positions[:]:
                # 获取该订单的参数
                pos_entry_price = pos['entry_price']
                pos_sl = pos['stop_loss_price']
                pos_tp = pos['take_profit_price']  # 可能是 0 (MA模式) 或 固定值 (RR模式)

                # 确定当前止盈目标
                current_target = 0.0
                if exit_mode == 'ma':
                    current_target = current_ma
                elif exit_mode == 'rr':
                    current_target = pos_tp

                should_exit = False
                exit_price = 0.0
                exit_reason = ""

                # --- 检查止损 ---
                if price <= pos_sl:
                    exit_price = pos_sl
                    # 跳空处理
                    if p_type == 'open' and price < pos_sl:
                        exit_price = price
                    exit_reason = "止损"
                    should_exit = True

                # --- 检查止盈 ---
                elif price >= current_target:
                    exit_price = current_target
                    # 跳空处理
                    if p_type == 'open' and price > current_target:
                        exit_price = price
                    exit_reason = f"止盈({exit_mode})"
                    should_exit = True

                # --- 执行平仓 ---
                if should_exit:
                    pnl = (exit_price - pos_entry_price) * (capital / pos_entry_price)
                    pnl -= capital * fee_rate * 2
                    duration_seconds = (curr_bar['open_time'] - pos['entry_time']).total_seconds()

                    trades.append({
                        '开仓时间': pos['entry_time'],
                        '平仓时间': curr_bar['open_time'],
                        '持仓时间': duration_seconds,
                        '退出模式': exit_reason,
                        '开仓价': pos_entry_price,
                        '平仓价': exit_price,
                        '净盈亏': pnl,
                        '参数组合': str(config)
                    })
                    cum_realized_profit += pnl
                    # 从持仓列表中移除该订单
                    positions.remove(pos)

            # 2. 检查开仓 (修改点：不再检查 if not position)
            # 只要本K线还没开过仓，且满足条件，就开新仓
            if not entry_action_this_bar and p_type == 'open':
                if prev_bar['signal_long'] and price < curr_bar['ma']:

                    # 计算该笔订单的止损止盈
                    entry_p = price
                    sl_p = entry_p * (1 - sl_pct)
                    tp_p = 0.0
                    if exit_mode == 'rr':
                        tp_p = entry_p * (1 + (sl_pct * tp_ratio))

                    # 加入持仓列表
                    new_pos = {
                        'entry_time': curr_bar['open_time'],
                        'entry_price': entry_p,
                        'stop_loss_price': sl_p,
                        'take_profit_price': tp_p
                    }
                    positions.append(new_pos)

                    entry_action_this_bar = True  # 锁定本K线，避免重复开仓

        # --- C. 统计数据 ---
        # 记录当前时刻的持仓单数
        active_counts.append(len(positions))

    return trades, active_counts  # 返回 active_counts 以供外部统计


# ==========================================
# 独立的工作函数
# ==========================================
def execute_single_backtest(args):
    """
    单个进程执行逻辑
    """
    df_raw, config, output_dir = args

    start_time = time.time()
    process_name = f"[P-{os.getpid()}]"

    # 文件名增加 Resample 参数
    resample_m = config.get('resample_min', 1)

    filename = (f"Resample{resample_m}m_"
                f"MA{config['ma_period']}_"
                f"STD{config['std_dev_mult']}_"
                f"Vol{config['vol_mult']}_"
                f"SL{config['sl_pct']}_"
                f"Mode-{config['exit_mode']}_"
                f"Ratio{config['tp_ratio']}.parquet")

    save_path = os.path.join(output_dir, filename)
    if os.path.exists(save_path):
        print(f"{process_name} 文件已存在，跳过: {filename}")
        return None
    try:
        # 1. 新增：先进行数据合并 (Resample)
        df_processing = resample_data(df_raw, resample_m)

        # 检查数据量是否足够
        if len(df_processing) < config['ma_period'] + 10:
            print(f"{process_name} 合并后数据不足，跳过")
            return None

        # 2. 执行回测 (接收两个返回值)
        trades_list, active_counts_list = run_backtest(df_processing, config)

        if not trades_list:
            print(f"{process_name} 参数 {config} -> 无交易触发")
            return None

        # 3. 数据处理与统计
        df_trades = pd.DataFrame(trades_list)

        total_profit = df_trades['净盈亏'].sum()
        trade_count = len(df_trades)
        win_rate = len(df_trades[df_trades['净盈亏'] > 0]) / trade_count if trade_count > 0 else 0
        avg_pnl = df_trades['净盈亏'].mean()

        # 计算持仓时间
        avg_seconds = df_trades['持仓时间'].mean()
        max_seconds = df_trades['持仓时间'].max()
        max_hours = round(max_seconds / 3600, 4)
        avg_hours = round(avg_seconds / 3600, 4)

        # --- 新增：计算持仓数量统计 ---
        max_concurrent = np.max(active_counts_list) if active_counts_list else 0
        avg_concurrent = np.mean(active_counts_list) if active_counts_list else 0

        # 空间优化：只保留最后 100 条
        if len(df_trades) > 100:
            df_trades_saved = df_trades.iloc[-100:].copy()
        else:
            df_trades_saved = df_trades.copy()

        # 4. 写入统计信息
        df_trades_saved['统计_总利润'] = total_profit
        df_trades_saved['统计_总交易数'] = trade_count
        df_trades_saved['统计_胜率'] = win_rate
        df_trades_saved['统计_平均盈亏'] = avg_pnl
        df_trades_saved['统计_平均持仓时间'] = avg_hours
        df_trades_saved['统计_最大持仓时间'] = max_hours

        # 新增统计字段写入
        df_trades_saved['统计_最大同时持仓数'] = max_concurrent
        df_trades_saved['统计_平均同时持仓数'] = round(avg_concurrent, 2)

        # 5. 保存
        df_trades_saved.to_parquet(save_path, index=False)

        # 6. 打印
        elapsed = time.time() - start_time
        print(
            f"{process_name} Resample:{resample_m}m | 利润:{total_profit:.1f} | 胜率:{win_rate:.1%} | "
            f"最大持仓数:{max_concurrent} | 耗时:{elapsed:.2f}s")

        # 7. 返回汇总
        summary = config.copy()
        summary.update({
            'total_trades': trade_count,
            'total_profit': total_profit,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'max_concurrent_pos': max_concurrent,  # 添加到汇总表
            'avg_concurrent_pos': avg_concurrent,  # 添加到汇总表
            'filename': filename,
            'execution_time': elapsed
        })
        return summary

    except Exception as e:
        traceback.print_exc()
        print(f"{process_name} 错误: {e} | 参数: {config}")
        return None


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 配置
    FILE_PATH = r"W:\project\python_project\oke_auto_trade\kline_data\origin_data_1m_10000000_BTC-USDT-SWAP_2026-02-13.csv"
    LINE_COUNT = 10000000
    RESULTS_DIR = "backtest_needle_multi"  # 修改目录名区分
    PROCESS_NUM = 10

    print("正在加载数据...")
    df_raw = pd.read_csv(FILE_PATH, nrows=LINE_COUNT)
    df_raw['open_time'] = pd.to_datetime(df_raw['timestamp'])  # 请确保CSV里有timestamp列
    # 如果没有timestamp列，请根据实际情况修改

    df_raw = df_raw[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    # 排序以确保 Resample 正确
    df_raw = df_raw.sort_values('open_time')
    print(f"数据加载完毕: {len(df_raw)} 行")

    # 2. 参数设置 (新增 resample_min)
    param_grid = {
        'resample_min': [1, 5, 15],  # 新增：1分钟，5分钟，15分钟K线
        'ma_period': [20, 60, 120, 180],
        'std_dev_mult': [2.5, 3.0, 3.5, 4, 5],
        'vol_mult': [2.5, 3, 4.0, 5, 6],
        'sl_pct': [0.05, 0.01, 0.015, 0.02, 0.03],
        'exit_mode': ['rr'],
        'tp_ratio': [2.0, 3.0, 4.0, 5.0, 6.0],
        'fee_rate': [0.0005],
        'capital_per_trade': [10000]
    }

    # 生成参数组合
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"共生成 {len(combinations)} 组参数组合，准备启用 {PROCESS_NUM} 个进程并行回测...")

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    tasks = [(df_raw, config, RESULTS_DIR) for config in combinations]

    t_start_all = time.time()

    with Pool(processes=PROCESS_NUM) as pool:
        results = pool.map(execute_single_backtest, tasks)

    valid_summaries = [res for res in results if res is not None]

    if valid_summaries:
        df_summary = pd.DataFrame(valid_summaries)
        df_summary = df_summary.sort_values(by='total_profit', ascending=False)

        summary_path = os.path.join(RESULTS_DIR, "FINAL_SUMMARY_REPORT.csv")
        df_summary.to_csv(summary_path, index=False)

        print("\n" + "=" * 50)
        print(f"全部完成! 总耗时: {time.time() - t_start_all:.2f}s")
        print(f"最佳参数组合 Top 3:")
        # 打印时展示一下K线周期和最大持仓数
        cols_to_show = ['resample_min', 'ma_period', 'std_dev_mult', 'total_profit', 'max_concurrent_pos']
        print(df_summary[cols_to_show].head(3).to_string())
        print(f"汇总报表已保存: {summary_path}")
    else:
        print("\n警告: 所有组合均未产生有效交易或发生错误。")