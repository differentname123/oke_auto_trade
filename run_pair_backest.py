import pandas as pd
import numpy as np
import os
import time
from multiprocessing import Pool
from datetime import datetime
# 保持原本的引用
from get_feature import read_last_n_lines

import pandas as pd
import numpy as np


# -----------------------------------------------------------------------------
# 0. 辅助算法: 卡尔曼滤波计算动态对冲比率 (新增核心)
# -----------------------------------------------------------------------------
def calculate_kalman_hedge_ratio(x, y, delta=1e-5, ve=1e-3):
    """
    使用卡尔曼滤波计算动态的 Alpha (截距) 和 Beta (斜率)
    模型: y = alpha + beta * x

    参数:
        x: 解释变量 (如 ETH log price)
        y: 被解释变量 (如 BTC log price)
        delta: 过程噪声协方差系数 (Process Noise Covariance).
               值越小，Beta变化越慢/越平滑；值越大，Beta越敏感。
               建议范围 1e-5 到 1e-4。
        ve: 测量噪声方差 (Measurement Noise Variance).

    返回:
        beta_series, alpha_series, spread_series
    """
    n_obs = len(x)

    # 初始化状态向量 [alpha, beta]
    state_mean = np.zeros(2)
    # 初始化状态协方差矩阵 (初始不确定性设大一点)
    state_cov = np.ones((2, 2)) * 1.0

    # 过程噪声矩阵 Q (控制参数变化的灵活性)
    # 假设 alpha 和 beta 遵循随机游走
    Q = np.eye(2) * delta

    # 测量噪声方差 R
    R = ve

    # 存放结果
    betas = np.zeros(n_obs)
    alphas = np.zeros(n_obs)
    spreads = np.zeros(n_obs)  # 这里的 spread 即预测残差 (Innovation)

    # 转换为 numpy 数组以提高循环速度
    x_arr = x.values if isinstance(x, pd.Series) else x
    y_arr = y.values if isinstance(y, pd.Series) else y

    # --- Kalman Filter 递归更新 ---
    # 由于前后依赖，无法向量化，但 Numpy 循环对于 10万行数据通常只需 1-2秒
    for t in range(n_obs):
        # 1. 预测阶段 (Prediction)
        # 状态预测: x(t|t-1) = x(t-1|t-1) (随机游走假设，预测值等于上一时刻值)
        state_pred = state_mean
        # 协方差预测: P(t|t-1) = P(t-1|t-1) + Q
        cov_pred = state_cov + Q

        # 2. 构建观测矩阵 H
        # y_t = alpha + beta * x_t
        # H = [1, x_t]
        H = np.array([1.0, x_arr[t]])

        # 3. 计算预测误差 (Innovation) 和 误差方差
        # y_pred = H @ state_pred
        y_pred = state_pred[0] + state_pred[1] * x_arr[t]
        error = y_arr[t] - y_pred  # 这就是未经标准化的 Spread

        # S = H P H.T + R
        S = H @ cov_pred @ H.T + R

        # 4. 计算卡尔曼增益 (Kalman Gain)
        # K = P H.T * S^-1
        K = cov_pred @ H.T / S

        # 5. 更新阶段 (Update)
        # state_new = state_pred + K * error
        state_mean = state_pred + K * error

        # cov_new = (I - K H) P
        # 这种写法数值上更稳定: P = (I - KH)P(I - KH)' + KRK'，但简易版通常够用
        state_cov = cov_pred - np.outer(K, H) @ cov_pred

        # 记录结果
        alphas[t] = state_mean[0]
        betas[t] = state_mean[1]
        spreads[t] = error

    return betas, alphas, spreads


# -----------------------------------------------------------------------------
# 1. 核心策略函数 (卡尔曼滤波优化版)
# -----------------------------------------------------------------------------
def generate_pair_trading_signals(df_btc, df_eth, merged_df=None, window=60, z_entry=3.0, z_exit=0.5, delta=1e-5):
    """
    输入:
        window: 这里的 window 仅用于计算 Spread 的 Z-Score (均值回归的周期)，
                不再用于计算 Beta (Beta由卡尔曼滤波全量历史决定)。
    """

    # --- 1. 数据对齐 ---
    if merged_df is not None:
        df = merged_df
    else:
        df_btc = df_btc.sort_values('open_time').set_index('open_time')
        df_eth = df_eth.sort_values('open_time').set_index('open_time')
        df = pd.merge(df_btc[['close']], df_eth[['close']], left_index=True, right_index=True,
                      suffixes=('_btc', '_eth'))

    # --- 2. 计算对数价格 ---
    df['log_btc'] = np.log(df['close_btc'])
    df['log_eth'] = np.log(df['close_eth'])

    # --- 3. 动态计算 Hedge Ratio (Kalman Filter) [核心替换] ---
    # 使用卡尔曼滤波替代原本的 Rolling OLS
    # 这里的 delta 控制 Beta 变化的灵活性，1e-5 是经验值，对应较平滑的变化
    betas, alphas, kf_spreads = calculate_kalman_hedge_ratio(
        df['log_eth'],
        df['log_btc'],
        delta=delta,  # 过程噪声 (可调: 1e-4 更灵敏, 1e-5 更平滑)
        ve=1e-3  # 观测噪声
    )

    df['beta'] = betas
    df['alpha'] = alphas

    # 注意: 卡尔曼滤波直接产出的 spread (error) 是 "当前价格 - 预测价格"
    # 也就是纯粹的残差，这比手动计算 log_btc - (beta*log_eth + alpha) 更准确，
    # 因为它是基于 t-1 时刻的预测参数计算 t 时刻的残差，避免了未来函数引入。
    df['spread'] = kf_spreads

    # --- 4. Z-Score 计算 ---
    # 虽然 Beta 不需要窗口，但判断 Spread 是否偏离均值，仍然需要一个基准窗口
    # 来定义"什么是正常波动范围"。

    df['spread_mean'] = df['spread'].rolling(window=window).mean()
    df['spread_std'] = df['spread'].rolling(window=window).std()

    # 计算 Z-Score
    df['z_score'] = (df['spread'] - df['spread_mean']) / df['spread_std']

    # --- 5. 生成交易信号 (State Machine) ---
    signals = np.zeros(len(df))
    current_position = 0
    z_values = df['z_score'].values
    z_values = np.nan_to_num(z_values, nan=0.0)

    # 从 window 开始循环 (虽然 Beta 有值，但 Z-Score 前 window 个是 NaN)
    for i in range(window, len(df)):
        z = z_values[i]

        if current_position == 0:
            # 入场
            if z > z_entry:
                current_position = 1
            elif z < -z_entry:
                current_position = -1

        elif current_position == 1:
            # 多头平仓
            if z < z_exit:
                current_position = 0

        elif current_position == -1:
            # 空头平仓
            if z > -z_exit:
                current_position = 0

        signals[i] = current_position

    df['signal'] = signals

    # 映射方向描述 (可选)
    # direction_map = {1: '做空价差', -1: '做多价差', 0: '空仓'}
    # df['trade_direction'] = df['signal'].map(direction_map)

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
    修正了盈亏计算逻辑，严格按照卡尔曼滤波计算出的入场 Beta 进行资金分配对冲。
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
    signals = df['signal'].values
    times = df['open_time'].values
    close_btc = df['close_btc'].values
    close_eth = df['close_eth'].values
    z_scores = df['z_score'].values
    spreads = df['spread'].values
    betas = df['beta'].values
    alphas = df['alpha'].values

    trades = []

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

            # 单腿基础收益率
            btc_roi = (cl_btc - op_btc) / op_btc
            eth_roi = (cl_eth - op_eth) / op_eth

            # 获取入场当时的 Beta 值 (作为对冲比率)
            entry_beta = betas[open_idx]

            # 【核心修正】：利用动态 Beta 计算净盈亏 (Net PnL)
            if signal_dir == -1:
                # Long Spread: 做多 1 单位 BTC, 做空 Beta 单位 ETH
                net_pnl = btc_roi - (entry_beta * eth_roi)
                direction_str = f'做多价差 (Long BTC / Short {entry_beta:.3f} ETH)'
            else:
                # Short Spread: 做空 1 单位 BTC, 做多 Beta 单位 ETH
                net_pnl = (entry_beta * eth_roi) - btc_roi
                direction_str = f'做空价差 (Short BTC / Long {entry_beta:.3f} ETH)'

            trade_record = {
                'open_time': times[open_idx],
                'close_time': times[close_idx],
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
    window, z_entry, z_exit, delta = args
    if z_exit >= z_entry:
        return
    global shared_df
    merged_df = shared_df

    back_df_file = f'backtest_pair/result_w{window}_entry{z_entry}_exit{z_exit}_delta{delta}_kalman.csv'

    if os.path.exists(back_df_file):
        # print(f"Skipping existing file: {back_df_file}")
        return

    start_time = time.time()

    # 运行优化后的策略
    full_df = generate_pair_trading_signals(None, None, merged_df=merged_df, window=window, z_entry=z_entry,
                                            z_exit=z_exit, delta=delta)

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

    window_list = [60, 120, 240, 600, 1000, 2000, 4000, 6000, 10000, 20000, 40000, 60000, 100000]
    z_entry_list = [1.5, 2, 2.5, 3, 4, 5, 6, 7, 10, 15, 20]
    z_exit_list = [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5]
    delta_list = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    tasks = []
    for window in window_list:
        for z_entry in z_entry_list:
            for z_exit in z_exit_list:
                for delta in delta_list:  # 增加一层循环
                    tasks.append((window, z_entry, z_exit, delta))

    print(f"Total tasks: {len(tasks)}")

    # 适当调整进程数，根据你 CPU 核心数决定
    with Pool(processes=10, initializer=init_worker, initargs=(original_df,)) as pool:
        pool.map(process_strategy, tasks)