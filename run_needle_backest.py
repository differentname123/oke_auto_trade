import traceback

import pandas as pd
import numpy as np
from datetime import timedelta
import itertools  # 新增：用于生成参数组合
import os


# ==========================================
# 1. 核心计算函数 (已修改为接收 config 参数)
# ==========================================

def calculate_indicators(df, config):
    """
    计算技术指标，生成原始信号
    Args:
        df: 原始数据
        config: 当前回测的参数字典
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

    # --- 信号计算逻辑 (基于上一根K线收盘确认) ---
    # 1. 价格极其便宜 (Low 击穿下轨)
    cond_price = df['low'] < df['lower_band']

    # 2. 恐慌性抛售 (成交量放大)
    cond_vol = df['volume'] > (df['vol_ma'] * vol_mult)

    # 3. 拒绝下跌形态 (收盘收回下轨上方 或 长下影线)
    cond_shape = df['close'] > df['lower_band']

    # 综合信号
    df['signal_long'] = cond_price & cond_vol & cond_shape

    return df


# ==========================================
# 2. 核心回测引擎 (已修改支持多种退出模式)
# ==========================================
def run_backtest(df, config):
    """
    执行回测
    Args:
        df: 包含OHLCV的原始数据
        config: 参数字典
    """
    # 预计算指标 (传入当前配置)
    df = calculate_indicators(df, config)

    # 转换为列表加速
    records = df.to_dict('records')

    trades = []
    stats = []

    # 账户状态变量
    position = False
    entry_price = 0.0
    entry_time = None
    stop_loss_price = 0.0

    # 新增：固定止盈价格 (仅在 fixed_rr 模式下使用)
    fixed_take_profit_price = 0.0

    cum_realized_profit = 0.0

    # 提取配置参数
    ma_period = config['ma_period']
    capital = config['capital_per_trade']
    fee_rate = config['fee_rate']
    sl_pct = config['sl_pct']

    # 策略核心参数
    exit_mode = config['exit_mode']  # 'ma' 或 'rr'
    tp_ratio = config.get('tp_ratio', 2.0)  # 盈亏比，默认2.0

    # 遍历K线
    for i in range(ma_period, len(records)):
        curr_bar = records[i]
        prev_bar = records[i - 1]
        last_close = prev_bar['close']

        # --- A. 构建价格路径 (保持不变) ---
        price_path = []
        path_types = []

        if curr_bar['close'] >= curr_bar['open']:
            price_path = [last_close, curr_bar['open'], curr_bar['low'], curr_bar['high'], curr_bar['close']]
            path_types = ['last_close', 'open', 'low', 'high', 'close']
        else:
            price_path = [last_close, curr_bar['open'], curr_bar['high'], curr_bar['low'], curr_bar['close']]
            path_types = ['last_close', 'open', 'high', 'low', 'close']

        # --- B. 路径游走 ---
        trade_action_this_min = False

        # 计算当前MA (如果 exit_mode == 'ma' 需要用到)
        current_ma = curr_bar['ma']

        for idx, price in enumerate(price_path):
            p_type = path_types[idx]

            # 1. 检查持仓 (止盈/止损)
            if position:
                # ---------------------------------------------------
                # 动态决定止盈目标
                # ---------------------------------------------------
                target_price = 0.0
                if exit_mode == 'ma':
                    target_price = current_ma  # 回归均线即止盈
                elif exit_mode == 'rr':
                    target_price = fixed_take_profit_price  # 等待达到盈亏比价格

                # --- 检查止损 ---
                if price <= stop_loss_price:
                    exit_price = stop_loss_price
                    if p_type == 'open' and price < stop_loss_price:
                        exit_price = price  # 跳空低开

                    pnl = (exit_price - entry_price) * (capital / entry_price)
                    pnl -= capital * fee_rate * 2
                    duration_seconds = (curr_bar['open_time'] - entry_time).total_seconds()

                    trades.append({
                        '开仓时间': entry_time,
                        '平仓时间': curr_bar['open_time'],
                        '持仓时间':duration_seconds,  # 转字符串方便保存
                        '退出模式': "止损",
                        '开仓价': entry_price,
                        '平仓价': exit_price,
                        '净盈亏': pnl,
                        '参数组合': str(config)  # 记录该笔交易属于哪个参数
                    })
                    cum_realized_profit += pnl
                    position = False
                    trade_action_this_min = True
                    break

                # --- 检查止盈 ---
                elif price >= target_price:
                    exit_price = target_price
                    # 如果是跳空高开 (Open > Target)
                    if p_type == 'open' and price > target_price:
                        exit_price = price

                    # 在 MA 模式下，如果 Bar 内 High 穿过 MA，我们假设在 MA 成交
                    # 但如果是 RR 模式，Target 是固定的

                    pnl = (exit_price - entry_price) * (capital / entry_price)
                    pnl -= capital * fee_rate * 2
                    duration_seconds = (curr_bar['open_time'] - entry_time).total_seconds()
                    trades.append({
                        '开仓时间': entry_time,
                        '平仓时间': curr_bar['open_time'],
                        '持仓时间': duration_seconds,
                        '退出模式': f"止盈({exit_mode})",
                        '开仓价': entry_price,
                        '平仓价': exit_price,
                        '净盈亏': pnl,
                        '参数组合': str(config)
                    })
                    cum_realized_profit += pnl
                    position = False
                    trade_action_this_min = True
                    break

            # 2. 检查开仓
            if not position and not trade_action_this_min and p_type == 'open':
                # 过滤逻辑：信号存在 且 开盘价在均线下方 (避免高开直接触碰止盈)
                if prev_bar['signal_long'] and price < curr_bar['ma']:
                    position = True
                    entry_price = price
                    entry_time = curr_bar['open_time']

                    # 设定止损
                    stop_loss_price = entry_price * (1 - sl_pct)

                    # 设定固定止盈价格 (仅用于 RR 模式)
                    # 利润 = 本金 * (涨幅) => 涨幅 = sl_pct * tp_ratio
                    # 举例: SL 1%, Ratio 3 => TP 涨幅 3%
                    fixed_take_profit_price = entry_price * (1 + (sl_pct * tp_ratio))

        # --- C. 统计数据 (略微简化，仅保留关键字段) ---
        worst_equity = capital + cum_realized_profit
        if position:
            worst_price = curr_bar['low']
            float_pnl = (worst_price - entry_price) * (capital / entry_price)
            worst_equity += float_pnl

        # 为了节省内存，Stats 可以根据需要决定是否每分钟都存
        # 这里保留逻辑，但建议大数据量时只存每日/每小时快照
        # stats.append({...})

    # 返回Trades列表 (转DataFrame在外部做，方便合并)
    return trades


import pandas as pd
import numpy as np
import itertools
import os
import time
from multiprocessing import Pool


# ==========================================
# 独立的工作函数 (必须定义在 if __name__ 之外)
# ==========================================
def execute_single_backtest(args):
    """
    单个进程执行的各种逻辑：回测、统计、截取数据、保存
    Args:
        args: 一个元组 (df_raw, config, output_dir)
              注意：为了在多进程中传递，这里打包成一个参数
    """
    df_raw, config, output_dir = args

    start_time = time.time()
    process_name = f"[P-{os.getpid()}]"  # 打印进程ID方便调试
    filename = (f"MA{config['ma_period']}_"
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
        # 1. 执行回测
        trades_list = run_backtest(df_raw, config)

        # 如果没有交易，直接返回 None
        if not trades_list:
            print(f"{process_name} 参数 {config} -> 无交易触发")
            return None

        # 2. 数据处理与统计
        df_trades = pd.DataFrame(trades_list)

        # 计算核心统计指标
        total_profit = df_trades['净盈亏'].sum()
        trade_count = len(df_trades)
        win_rate = len(df_trades[df_trades['净盈亏'] > 0]) / trade_count if trade_count > 0 else 0
        avg_pnl = df_trades['净盈亏'].mean()

        # 3. 空间优化：只保留最后 100 条
        # 注意：先统计完总体的指标，再截取，否则统计数据就不对了
        if len(df_trades) > 100:
            df_trades_saved = df_trades.iloc[-100:].copy()
        else:
            df_trades_saved = df_trades.copy()

        # 4. 将统计汇总信息写入数据列 (方便单文件查看上下文)
        df_trades_saved['统计_总利润'] = total_profit
        df_trades_saved['统计_总交易数'] = trade_count
        df_trades_saved['统计_胜率'] = win_rate
        df_trades_saved['统计_平均盈亏'] = avg_pnl
        # 把配置参数也转为字符串存进去，防丢失
        # df_trades_saved['策略_配置详情'] = str(config)
        avg_seconds = df_trades['持仓时间'].mean()
        max_seconds = df_trades['持仓时间'].max()
        max_hours = round(max_seconds / 3600, 4)
        avg_hours = round(avg_seconds / 3600, 4)
        df_trades_saved['统计_平均持仓时间'] = avg_hours
        df_trades_saved['统计_最大持仓时间'] = max_hours
        # 5. 生成文件名并保存

        df_trades_saved.to_parquet(save_path, index=False)

        # 6. 计算耗时并打印
        elapsed = time.time() - start_time
        print(
            f"{process_name} 完成 | 耗时:{elapsed:.2f}s | 利润:{total_profit:.1f} | 胜率:{win_rate:.1%} | 统计_平均持仓时间：{avg_duration_str}  统计_最大持仓时间：{max_duration_str}   耗时 ：{elapsed:.2f}s")

        # 7. 返回汇总字典 (给主进程做总表用)
        summary = config.copy()
        summary.update({
            'total_trades': trade_count,
            'total_profit': total_profit,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
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
    RESULTS_DIR = "backtest_needle"
    PROCESS_NUM = 5  # 进程数量

    print("正在加载数据...")
    df_raw = pd.read_csv(FILE_PATH, nrows=LINE_COUNT)
    df_raw['open_time'] = pd.to_datetime(df_raw['timestamp'])  # 根据实际列名调整
    df_raw = df_raw[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    print(f"数据加载完毕: {len(df_raw)} 行")

    # 2. 参数设置
    param_grid = {
        'ma_period': [20, 60, 120, 180, 300],
        'std_dev_mult': [2, 2.5, 3.0, 3.5, 4, 5],
        'vol_mult': [2.5, 3, 4.0, 5, 6],
        'sl_pct': [0.05, 0.01, 0.015, 0.02, 0.03],
        'exit_mode': ['ma', 'rr'],
        'tp_ratio': [2.0, 3.0, 4.0, 5.0, 6.0],
        'fee_rate': [0.0005],
        'capital_per_trade': [10000]
    }

    # 生成参数组合
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"共生成 {len(combinations)} 组参数组合，准备启用 {PROCESS_NUM} 个进程并行回测...")

    # 准备结果目录
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    tasks = [(df_raw, config, RESULTS_DIR) for config in combinations]

    # 4. 多进程执行
    t_start_all = time.time()

    # 使用 with 语句自动管理 Pool 的关闭
    with Pool(processes=PROCESS_NUM) as pool:
        results = pool.map(execute_single_backtest, tasks)

    # 5. 汇总结果
    # 过滤掉 None (即报错或无交易的结果)
    valid_summaries = [res for res in results if res is not None]

    if valid_summaries:
        df_summary = pd.DataFrame(valid_summaries)
        # 按利润倒序
        df_summary = df_summary.sort_values(by='total_profit', ascending=False)

        summary_path = os.path.join(RESULTS_DIR, "FINAL_SUMMARY_REPORT.csv")
        df_summary.to_csv(summary_path, index=False)

        print("\n" + "=" * 50)
        print(f"全部完成! 总耗时: {time.time() - t_start_all:.2f}s")
        print(f"最佳参数组合 Top 3:")
        print(df_summary[['ma_period', 'std_dev_mult', 'exit_mode', 'tp_ratio', 'total_profit']].head(3).to_string())
        print(f"汇总报表已保存: {summary_path}")
    else:
        print("\n警告: 所有组合均未产生有效交易或发生错误。")