import re
import pandas as pd
import numpy as np
import os
import time
from multiprocessing import Pool
from datetime import datetime
import numba as nb  # 【新增】引入 Numba 用于核心加速

# 保持原本的引用
from get_feature import read_last_n_lines


# -----------------------------------------------------------------------------
# 0. 辅助算法: 卡尔曼滤波计算动态对冲比率 (Numba 标量展开极速版)
# -----------------------------------------------------------------------------
@nb.njit(cache=True)
def fast_kalman_filter(x_arr, y_arr, delta, ve):
    """
    使用 Numba 即时编译和矩阵代数展开，速度提升百倍，数学结果与原版矩阵运算100%等价
    """
    n_obs = len(x_arr)

    alphas = np.zeros(n_obs)
    betas = np.zeros(n_obs)
    spreads = np.zeros(n_obs)
    std_devs = np.zeros(n_obs)

    # 初始状态均值
    alpha_mean = 0.0
    beta_mean = 0.0

    # 初始化状态协方差矩阵 P (原版是 np.ones((2, 2)) * 1.0)
    P00 = 1.0
    P01 = 1.0
    P10 = 1.0
    P11 = 1.0

    for t in range(n_obs):
        x = x_arr[t]
        y = y_arr[t]

        # 1. 预测阶段 (cov_pred = state_cov + Q)
        # Q = np.eye(2) * delta
        P00 += delta
        P11 += delta

        # 2. 预测误差与方差
        y_pred = alpha_mean + beta_mean * x
        error = y - y_pred

        # S = H @ cov_pred @ H.T + R  (H = [1, x])
        S = P00 + P01 * x + P10 * x + P11 * x * x + ve
        std_devs[t] = np.sqrt(S)

        # 3. 卡尔曼增益 K = cov_pred @ H.T / S
        K0 = (P00 + P01 * x) / S
        K1 = (P10 + P11 * x) / S

        # 4. 状态更新 (state_mean = state_pred + K * error)
        alpha_mean += K0 * error
        beta_mean += K1 * error

        # 5. 协方差更新 (state_cov = cov_pred - np.outer(K, H) @ cov_pred)
        # 严格的矩阵代数展开，避免了小矩阵创建开销
        new_P00 = P00 - K0 * (P00 + P10 * x)
        new_P01 = P01 - K0 * (P01 + P11 * x)
        new_P10 = P10 - K1 * (P00 + P10 * x)
        new_P11 = P11 - K1 * (P01 + P11 * x)

        P00 = new_P00
        P01 = new_P01
        P10 = new_P10
        P11 = new_P11

        # 记录结果
        alphas[t] = alpha_mean
        betas[t] = beta_mean
        spreads[t] = error

    return betas, alphas, spreads, std_devs


def calculate_kalman_hedge_ratio(x, y, delta=1e-5, ve=1e-3):
    """
    保留原版函数签名作为兼容封装层
    """
    x_arr = x.values if isinstance(x, pd.Series) else x
    y_arr = y.values if isinstance(y, pd.Series) else y
    return fast_kalman_filter(x_arr, y_arr, delta, ve)


# -----------------------------------------------------------------------------
# 1. 核心策略函数 (无滚动窗口的纯卡尔曼 Z-Score 优化版)
# -----------------------------------------------------------------------------
@nb.njit(cache=True)
def fast_generate_signals(z_values, window, z_entry, z_exit):
    """
    将原版的状态机生成信号逻辑提取并用 Numba 加速，逻辑100%未变
    """
    n = len(z_values)
    signals = np.zeros(n)
    current_position = 0

    for i in range(window, n):
        z = z_values[i]

        if current_position == 0:
            if z > z_entry:
                current_position = 1
            elif z < -z_entry:
                current_position = -1
        elif current_position == 1:
            if z < z_exit:
                current_position = 0
        elif current_position == -1:
            if z > -z_exit:
                current_position = 0

        signals[i] = current_position

    return signals


def generate_pair_trading_signals(merged_df=None, main_col='close_btc', sub_col='close_eth', window=60, z_entry=3.0,
                                  z_exit=0.5, delta=1e-5, ve=1e-3):
    if merged_df is not None:
        df = merged_df

    df['log_main'] = np.log(df[main_col])
    df['log_sub'] = np.log(df[sub_col])

    betas, alphas, kf_spreads, kf_std_devs = calculate_kalman_hedge_ratio(
        df['log_sub'],
        df['log_main'],
        delta=delta,
        ve=ve
    )

    df['beta'] = betas
    df['alpha'] = alphas
    df['spread'] = kf_spreads
    df['spread_std'] = kf_std_devs

    # 【修改】使用原生向量化除法，因卡尔曼方差(含ve)理论不可能为0，省略where判断大幅提速
    df['z_score'] = df['spread'] / df['spread_std']

    z_values = df['z_score'].values
    z_values = np.nan_to_num(z_values, nan=0.0)

    # 【修改】调用 Numba 加速版本的状态机
    df['signal'] = fast_generate_signals(z_values, window, z_entry, z_exit)

    return df


def print_backtest_stats(trade_df):
    """
    保持原版逻辑完全不变
    """
    if trade_df.empty:
        print("【统计报告】无交易记录，无法计算指标。")
        return

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
    avg_profit_per_trade = trade_df['net_pnl'].mean()

    return {
        'Total Return': total_return,
        'total_trades': total_trades,
        'Win Rate': win_rate,
        'avg_profit_per_trade': avg_profit_per_trade,
        'Max Drawdown': max_drawdown,
        'Profit Factor': profit_factor,
        'avg_duration': avg_duration,
        '最长持仓时间': max_duration
    }


def extract_trades_from_signals(full_df):
    """
    保持原版逻辑完全不变
    """
    df = full_df.reset_index()

    if 'open_time' not in df.columns:
        if 'index' in df.columns:
            df.rename(columns={'index': 'open_time'}, inplace=True)
        elif 'timestamp' in df.columns:
            df.rename(columns={'timestamp': 'open_time'}, inplace=True)

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

    for i in range(1, len(signals)):
        sig = signals[i]
        prev_sig = signals[i - 1]

        if sig != 0 and prev_sig == 0:
            open_idx = i
            in_trade = True

        elif in_trade and sig == 0 and prev_sig != 0:
            close_idx = i
            signal_dir = signals[open_idx]

            op_btc, cl_btc = close_btc[open_idx], close_btc[close_idx]
            op_eth, cl_eth = close_eth[open_idx], close_eth[close_idx]

            btc_roi = (cl_btc - op_btc) / op_btc
            eth_roi = (cl_eth - op_eth) / op_eth

            entry_beta = betas[open_idx]

            if signal_dir == -1:
                net_pnl = btc_roi - (entry_beta * eth_roi)
                direction_str = f'做多价差 (Long BTC / Short {entry_beta:.3f} ETH)'
            else:
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
    z_entry, z_exit, delta, ve, granularity, key_word = args
    window = 60
    if z_exit >= z_entry:
        return
    global shared_df

    if granularity > 1:
        return
        df_to_use = shared_df.copy()
        df_to_use.set_index('open_time', inplace=True)
        df_to_use = df_to_use.resample(f'{granularity}min').last().dropna().reset_index()
        merged_df = df_to_use
    else:
        merged_df = shared_df.copy()

    back_df_file = f'backtest_pair/{key_word}_w{window}_entry{z_entry}_exit{z_exit}_delta{delta}_ve{ve}_g{granularity}_kalman.csv'

    if os.path.exists(back_df_file):
        return

    start_time = time.time()

    full_df = generate_pair_trading_signals(merged_df=merged_df, window=window, z_entry=z_entry,
                                            z_exit=z_exit, delta=delta, ve=ve)

    detailed_result_df = extract_trades_from_signals(full_df)

    os.makedirs(os.path.dirname(back_df_file), exist_ok=True)
    detailed_result_df.to_csv(back_df_file)

    end_time = time.time()
    duration = end_time - start_time

    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time_str}] Saved: {back_df_file} | Time Cost: {duration:.4f}s")


def parse_backtest_filename(filepath):
    filename = os.path.basename(filepath)
    pattern = (
        r"w(?P<window>.+?)_"
        r"entry(?P<z_entry>.+?)_"
        r"exit(?P<z_exit>.+?)_"
        r"delta(?P<delta>.+?)_"
        r"ve(?P<ve>.+?)_"
        r"g(?P<granularity>.+?)_"
        r"kalman\.csv"
    )

    match = re.search(pattern, filename)
    if match:
        raw_data = match.groupdict()
        parsed_data = {}
        for key, value in raw_data.items():
            try:
                parsed_data[key] = int(value)
            except ValueError:
                try:
                    parsed_data[key] = float(value)
                except ValueError:
                    parsed_data[key] = value
        return parsed_data
    else:
        print(f"解析失败: {filename}")
        return None


def run_good_params():
    original_df = pd.read_csv('kline_data/sol_xrp.csv')
    final_df_path = 'kline_data/result_df.csv'
    df = pd.read_csv(final_df_path)
    df['score'] = -(df['avg_profit_per_trade'] - 0.1) * df['total_trades'] / (df['Max Drawdown'] - 0.5)
    df_filtered = df[df['score'] > 10]
    tasks = []
    for index, row in df_filtered.iterrows():
        params = parse_backtest_filename(row['file_name'])
        if params:
            task_tuple = (
                params['z_entry'],
                params['z_exit'],
                params['delta'],
                params['ve'],
                params['granularity']
            )
            if params['granularity'] == 1:
                tasks.append(task_tuple)

    print(f"Total filtered tasks: {len(tasks)}")

    with Pool(processes=10, initializer=init_worker, initargs=(original_df,)) as pool:
        pool.map(process_strategy, tasks)


def get_good_tasks(base_tasks, key_word):
    import itertools
    tasks = []
    temp_tasks = []
    copy_base_tasks = base_tasks.copy()
    for task in copy_base_tasks:
        temp_tasks.append(task[:5])
    base_tasks_set = set(temp_tasks)
    final_df_path = f'kline_data/result_df_{key_word}.csv'
    df = pd.read_csv(final_df_path)
    df = df[df['avg_profit_per_trade'] > 0.2]
    df = df[df['Max Drawdown'] > -20]

    df['avg_profit_per_trade'] = df['avg_profit_per_trade'] - 0.1
    df['profit'] = (df['avg_profit_per_trade']) * df['total_trades']
    df = df[df['profit'] > 20]

    df = df[df['trades_this_year'] > 0]

    df['score'] = np.log(df['avg_profit_per_trade'] + 1) * df['profit'] * (df['avg_profit_per_trade']) * df[
        'total_trades'] / -(df['Max Drawdown'] - 0.5) / np.log(df['最长持仓时间'] + 1)
    count = 0
    no_params_count = 0
    keep_count = 0
    for index, row in df.iterrows():
        params = parse_backtest_filename(row['file_name'])
        if params:
            task_tuple = (
                params['z_entry'],
                params['z_exit'],
                params['delta'],
                params['ve'],
                params['granularity']
            )

            if task_tuple in base_tasks_set:
                count += 1
                variable_keys = ['z_entry', 'z_exit', 'delta', 've']
                variations = []
                for key in variable_keys:
                    base_val = params[key]
                    variations.append([base_val * 0.7, base_val * 0.8, base_val * 0.9, base_val, base_val * 1.1, base_val * 1.2, base_val * 1.3])

                variations.append([params['granularity']])

                for combo in itertools.product(*variations):
                    # 为每一个combo 最后增加 key_word
                    combo_with_keyword = combo + (key_word,)

                    tasks.append(combo_with_keyword)
            else:
                combo_with_keyword = task_tuple + (key_word,)
                tasks.append(combo_with_keyword)
                keep_count += 1
        else:
            no_params_count += 1
    print(
        f"Filtered {len(df)} tasks from Filtered {count} keep_count {keep_count} base tasks from DataFrame. Total tasks after expansion: {len(tasks)} (No params found for {no_params_count} rows).")
    return tasks


if __name__ == '__main__':
    key_word = 'btc_sol'

    print("Loading data...")
    if os.path.exists(f'kline_data/{key_word}.csv'):
        original_df = pd.read_csv(f'kline_data/{key_word}.csv')
        if 'open_time' in original_df.columns:
            original_df['open_time'] = pd.to_datetime(original_df['open_time'])
    else:
        print("Error: kline_data/btc_sol.csv not found.")
        exit()

    print(f"Data loaded. Rows: {len(original_df)}. Starting multiprocessing...")

    window_list = [60]
    z_entry_list = [1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10]
    z_exit_list = [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5]
    delta_list = [1e-3, 1e-4, 5e-5, 1e-5, 5e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
    ve_list = [1e-2, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    granularity_list = [1]

    tasks = []
    for z_entry in z_entry_list:
        for z_exit in z_exit_list:
            for delta in delta_list:
                for ve in ve_list:
                    for granularity in granularity_list:
                        tasks.append((z_entry, z_exit, delta, ve, granularity, key_word))
    tasks = get_good_tasks(tasks, key_word)

    print(f"Total tasks: {len(tasks)}")

    with Pool(processes=5, initializer=init_worker, initargs=(original_df,)) as pool:
        pool.map(process_strategy, tasks)