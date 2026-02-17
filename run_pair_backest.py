import pandas as pd
import numpy as np
import os
import time
from multiprocessing import Pool
from datetime import datetime
# 保持原本的引用
from get_feature import read_last_n_lines


# -----------------------------------------------------------------------------
# 1. 核心策略函数 (深度优化版)
# -----------------------------------------------------------------------------
def generate_pair_trading_signals(df_btc, df_eth, merged_df=None, window=60, z_entry=3.0, z_exit=0.5):
    """
    输入:
        df_btc, df_eth: 包含 'open_time' 和 'close' 的 DataFrame
        window: 滚动窗口大小 (例如 60 分钟)
        z_entry: 入场阈值 (标准差倍数)
        z_exit: 出场阈值
    输出:
        处理后的完整 DataFrame，包含信号、方向描述、Z-score等详细数据
    """

    # --- 1. 数据对齐 ---
    if merged_df is not None:
        df = merged_df
    else:
        df_btc = df_btc.sort_values('open_time').set_index('open_time')
        df_eth = df_eth.sort_values('open_time').set_index('open_time')
        # 合并数据，只保留两者都有的时间点 (Inner Join)
        df = pd.merge(df_btc[['close']], df_eth[['close']], left_index=True, right_index=True,
                      suffixes=('_btc', '_eth'))
        df.to_csv('kline_data/btc_eth.csv')

    # --- 2. 计算对数价格 ---
    df['log_btc'] = np.log(df['close_btc'])
    df['log_eth'] = np.log(df['close_eth'])

    # --- 3. 滚动计算动态对冲比率 (Rolling Beta) [优化核心] ---
    # 替换掉了缓慢的 statsmodels RollingOLS
    # 使用公式: Beta = Cov(x, y) / Var(x)
    #           Alpha = Mean(y) - Beta * Mean(x)

    # 预计算滚动统计量 (Pandas C底层优化，速度极快)
    rolling_cov = df['log_btc'].rolling(window=window).cov(df['log_eth'])
    rolling_var = df['log_eth'].rolling(window=window).var()
    rolling_mean_y = df['log_btc'].rolling(window=window).mean()
    rolling_mean_x = df['log_eth'].rolling(window=window).mean()

    # 计算 Beta 和 Alpha
    df['beta'] = rolling_cov / rolling_var
    df['alpha'] = rolling_mean_y - (df['beta'] * rolling_mean_x)

    # --- 4. 构建价差 (Spread) 和 Z-Score ---
    # Spread = Residual (残差)
    df['spread'] = df['log_btc'] - (df['beta'] * df['log_eth'] + df['alpha'])

    # 计算价差的滚动均值和标准差
    df['spread_mean'] = df['spread'].rolling(window=window).mean()
    df['spread_std'] = df['spread'].rolling(window=window).std()

    # Z-Score
    df['z_score'] = (df['spread'] - df['spread_mean']) / df['spread_std']

    # --- 5. 生成交易信号 (State Machine) ---
    # 使用 numpy 数组进行循环，避免 DataFrame 索引访问开销

    signals = np.zeros(len(df))
    current_position = 0  # 初始空仓

    # 转换为 numpy 数组加速读取
    z_values = df['z_score'].values
    # 填充 NaN 为 0 避免判断错误 (前 window 行通常是 NaN)
    z_values = np.nan_to_num(z_values, nan=0.0)

    # 循环优化: 纯数值比较
    for i in range(window, len(df)):
        z = z_values[i]

        # 逻辑判断
        if current_position == 0:
            # 入场逻辑
            if z > z_entry:
                current_position = 1  # 卖BTC / 买ETH
            elif z < -z_entry:
                current_position = -1  # 买BTC / 卖ETH

        elif current_position == 1:
            # 多头平仓逻辑
            if z < z_exit:
                current_position = 0

        elif current_position == -1:
            # 空头平仓逻辑
            if z > -z_exit:
                current_position = 0

        signals[i] = current_position

    df['signal'] = signals

    # --- 6. 生成人类可读的交易方向描述 ---
    # 为了速度，只在最后输出时映射，或者如果后续不需要该列可以省略，这里保留以防万一
    # (如果这步耗时，可以放到 extract 之后只对交易行做，但目前 vector map 很快)
    direction_map = {
        1: '做空价差 (卖BTC/买ETH)',
        -1: '做多价差 (买BTC/卖ETH)',
        0: '空仓观望'
    }
    df['trade_direction'] = df['signal'].map(direction_map)

    return df


def print_backtest_stats(trade_df):
    """
    保持原样
    """
    if trade_df.empty:
        # print("【统计报告】无交易记录，无法计算指标。")
        return

    print("-" * 50)
    print("             策略回测绩效报告 (Performance Report)")
    print("-" * 50)

    total_trades = len(trade_df)
    total_return = trade_df['net_pnl'].sum()
    avg_return = trade_df['net_pnl'].mean()

    winning_trades = trade_df[trade_df['net_pnl'] > 0]
    losing_trades = trade_df[trade_df['net_pnl'] <= 0]

    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    win_rate = win_count / total_trades if total_trades > 0 else 0

    gross_profit = winning_trades['net_pnl'].sum()
    gross_loss = abs(losing_trades['net_pnl'].sum())

    if gross_loss == 0:
        profit_factor = float('inf')
    else:
        profit_factor = gross_profit / gross_loss

    trade_df['cum_pnl'] = trade_df['net_pnl'].cumsum()
    trade_df['peak'] = trade_df['cum_pnl'].cummax()
    trade_df['drawdown'] = trade_df['cum_pnl'] - trade_df['peak']
    max_drawdown = trade_df['drawdown'].min()

    dd_idx = trade_df['drawdown'].idxmin()
    max_dd_time = trade_df.loc[dd_idx, 'close_time'] if pd.notna(dd_idx) else "N/A"

    avg_duration = trade_df['duration_mins'].mean()
    max_duration = trade_df['duration_mins'].max()

    print(f"1. 交易概况")
    print(f"   - 总交易次数: {total_trades} 次")
    print(f"   - 盈利次数:   {win_count} 次")
    print(f"   - 胜率 (Win Rate): {win_rate:.2%}")
    print(f"")
    print(f"2. 收益表现")
    print(f"   - 累计总收益率: {total_return:.2%} (单利累加)")
    print(f"   - 盈亏比 (Profit Factor): {profit_factor:.2f}")
    print(f"")
    print(f"3. 风险评估")
    print(f"   - 最大回撤 (Max Drawdown): {max_drawdown:.2%}")
    print(f"")
    print("-" * 50)

    return {
        'Total Return': total_return,
        'total_trades': total_trades,
        'Win Rate': win_rate,
        'Max Drawdown': max_drawdown,
        'Profit Factor': profit_factor,
        'avg_duration': avg_duration,
        '最长持仓时间': max_duration
    }


def extract_trades_from_signals(full_df):
    """
    优化后的交易提取函数：
    移除循环中的 DataFrame 切片操作 (.loc[i:]), 改为线性扫描
    """
    # --- 1. 数据预处理 ---
    df = full_df.reset_index()

    if 'open_time' not in df.columns:
        if 'index' in df.columns:
            df.rename(columns={'index': 'open_time'}, inplace=True)
        elif 'timestamp' in df.columns:
            df.rename(columns={'timestamp': 'open_time'}, inplace=True)

    # 提取需要的 numpy 数组，避免循环中访问 DataFrame
    # 这样速度极快
    signals = df['signal'].values
    times = df['open_time'].values
    close_btc = df['close_btc'].values
    close_eth = df['close_eth'].values
    z_scores = df['z_score'].values
    spreads = df['spread'].values
    betas = df['beta'].values
    alphas = df['alpha'].values

    # 识别状态变化点
    # 0 -> 1/-1 (开仓)
    # 1/-1 -> 0 (平仓)
    # 我们只需要遍历那些状态发生变化的行，而不是所有行

    trades = []

    # 寻找开仓点索引 (当前不为0，前一个为0)
    # 加上 [0] 是因为 np.where 返回 tuple
    # shift logic: 比较 signal[i] 和 signal[i-1]
    # np.diff 可以快速找到变化点，但为了逻辑清晰，我们使用简单的线性扫描
    # 因为已经转为 numpy array，for 循环即使跑 100万次也只需 0.x 秒

    open_idx = -1
    in_trade = False

    # 遍历 numpy 数组
    for i in range(1, len(signals)):
        sig = signals[i]
        prev_sig = signals[i - 1]

        # 开仓时刻
        if sig != 0 and prev_sig == 0:
            open_idx = i
            in_trade = True

        # 平仓时刻 (在持仓中，且当前信号变为0)
        elif in_trade and sig == 0 and prev_sig != 0:
            # 执行平仓结算
            close_idx = i

            # --- 核心计算 ---
            signal_dir = signals[open_idx]  # 1 or -1

            # 价格
            op_btc, cl_btc = close_btc[open_idx], close_btc[close_idx]
            op_eth, cl_eth = close_eth[open_idx], close_eth[close_idx]

            # 收益率
            btc_roi = (cl_btc - op_btc) / op_btc
            eth_roi = (cl_eth - op_eth) / op_eth

            if signal_dir == -1:
                # Long Spread: Long BTC, Short ETH
                net_pnl = btc_roi - eth_roi
                direction_str = '做多价差 (Long BTC/Short ETH)'
            else:
                # Short Spread: Short BTC, Long ETH
                net_pnl = eth_roi - btc_roi
                direction_str = '做空价差 (Short BTC/Long ETH)'

            trade_record = {
                'open_time': times[open_idx],
                'close_time': times[close_idx],
                # 假设 times 是 datetime 对象，如果不确定需转换，这里假设已经是 pandas Timestamp
                'duration_mins': (pd.Timestamp(times[close_idx]) - pd.Timestamp(times[open_idx])).total_seconds() / 60,
                'direction': direction_str,
                'btc_roi': f"{btc_roi * 100:.2f}",
                'eth_roi': f"{eth_roi * 100:.2f}",
                'net_pnl': round(net_pnl * 100, 2),
                'open_btc': op_btc, 'close_btc': cl_btc,
                'open_eth': op_eth, 'close_eth': cl_eth,
                'entry_z': z_scores[open_idx],
                'exit_z': z_scores[close_idx],
                'entry_spread': spreads[open_idx],
                'exit_spread': spreads[close_idx],
                'entry_beta': betas[open_idx],
                'exit_beta': betas[close_idx],
                'entry_alpha': alphas[open_idx],
                'exit_alpha': alphas[close_idx],
                'signal': signal_dir,
            }
            trades.append(trade_record)
            in_trade = False
            open_idx = -1

    if not trades:
        return pd.DataFrame()

    return pd.DataFrame(trades)


# ==========================================
# 多进程部分保持不变
# ==========================================
shared_df = None


def init_worker(df_to_share):
    global shared_df
    shared_df = df_to_share


def process_strategy(args):
    window, z_entry, z_exit = args
    if z_exit >= z_entry:
        return
    global shared_df
    merged_df = shared_df

    back_df_file = f'backtest_pair/result_w{window}_entry{z_entry}_exit{z_exit}.csv'

    if os.path.exists(back_df_file):
        # print(f"Skipping existing file: {back_df_file}")
        return

    start_time = time.time()

    # 运行优化后的策略
    full_df = generate_pair_trading_signals(None, None, merged_df=merged_df, window=window, z_entry=z_entry,
                                            z_exit=z_exit)

    # 运行优化后的提取
    detailed_result_df = extract_trades_from_signals(full_df)

    os.makedirs(os.path.dirname(back_df_file), exist_ok=True)
    detailed_result_df.to_csv(back_df_file)

    end_time = time.time()
    duration = end_time - start_time

    # 只打印耗时较长的，减少控制台刷屏
    # if duration > 1.0:
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time_str}] Saved: {back_df_file} | Time Cost: {duration:.4f}s")


if __name__ == '__main__':
    # 读取数据（只在主进程读取一次）
    print("Loading data...")
    # 确保这里的路径和你本地文件一致
    if os.path.exists('kline_data/btc_eth.csv'):
        original_df = pd.read_csv('kline_data/btc_eth.csv')
        # 确保时间列格式正确，这步很重要
        if 'open_time' in original_df.columns:
            original_df['open_time'] = pd.to_datetime(original_df['open_time'])
    else:
        # 兼容逻辑：如果没有预处理好的文件，这里可以加你的读取逻辑
        # original_df = ...
        print("Error: kline_data/btc_eth.csv not found.")
        exit()

    print(f"Data loaded. Rows: {len(original_df)}. Starting multiprocessing...")

    window_list = [60, 600, 1000, 2000, 4000, 6000, 10000, 20000, 40000, 60000, 100000]
    z_entry_list = [1.5, 2, 2.5, 3, 4, 5, 6, 7, 10, 15, 20]
    z_exit_list = [0.5, 1, 1.25, 1.5, 2, 3, 4]

    tasks = []
    for window in window_list:
        for z_entry in z_entry_list:
            for z_exit in z_exit_list:
                tasks.append((window, z_entry, z_exit))

    print(f"Total tasks: {len(tasks)}")

    # 适当调整进程数，根据你 CPU 核心数决定
    with Pool(processes=10, initializer=init_worker, initargs=(original_df,)) as pool:
        pool.map(process_strategy, tasks)