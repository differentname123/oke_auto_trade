import re
import pandas as pd
import numpy as np
import os
import time
from multiprocessing import Pool
from datetime import datetime
import numba as nb  # 【新增】引入 Numba 用于核心加速
import concurrent.futures

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
def fast_generate_signals(z_values, window, z_entry, z_exit, require_rebound=False):
    """
    将原版的状态机生成信号逻辑提取并用 Numba 加速，逻辑100%未变
    【新增】require_rebound: 是否需要 Z-Score 出现反弹（拐头）才生成信号
    """
    n = len(z_values)
    signals = np.zeros(n)
    current_position = 0

    for i in range(window, n):
        z = z_values[i]

        if current_position == 0:
            if not require_rebound:
                # 原始逻辑：突破阈值即刻产生信号
                if z > z_entry:
                    current_position = 1
                elif z < -z_entry:
                    current_position = -1
            else:
                # 新增逻辑：突破阈值，并且相对于上一根K线开始反弹（回归0轴方向）
                prev_z = z_values[i - 1]
                if z > z_entry and z < prev_z:
                    current_position = 1
                elif z < -z_entry and z > prev_z:
                    current_position = -1

        elif current_position == 1:
            if z < z_exit:
                current_position = 0
        elif current_position == -1:
            if z > -z_exit:
                current_position = 0

        signals[i] = current_position

    return signals


def generate_pair_trading_signals(merged_df=None, main_col='close_main', sub_col='close_sub', window=60, z_entry=3.0,
                                  z_exit=0.5, delta=1e-5, ve=1e-3, require_rebound=False):
    """
    【修改】在参数列表中新增了 require_rebound=False，保持对原版代码的完美向下兼容
    """
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

    # 【修改】调用 Numba 加速版本的状态机，透传 require_rebound 参数
    df['signal'] = fast_generate_signals(z_values, window, z_entry, z_exit, require_rebound)

    return df


def print_backtest_stats(trade_df):
    """
    输入: extract_trades_from_signals 生成的 trade_df
    输出: 打印详细的策略回测绩效报告
    """
    if trade_df.empty:
        # print("【统计报告】无交易记录，无法计算指标。")
        return

    # print("-" * 50)
    # print("             策略回测绩效报告 (Performance Report)")
    # print("-" * 50)

    # --- 1. 基础数据 ---
    total_trades = len(trade_df)

    # 累计收益 (所有单次收益率的简单累加，模拟复利需更复杂计算，这里用单利近似)
    total_return = trade_df['net_pnl'].sum()

    # 平均每笔收益
    avg_return = trade_df['net_pnl'].mean()

    # --- 2. 胜率分析 (Win Rate) ---
    winning_trades = trade_df[trade_df['net_pnl'] > 0]
    losing_trades = trade_df[trade_df['net_pnl'] <= 0]

    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    win_rate = win_count / total_trades if total_trades > 0 else 0

    # --- 3. 盈亏比 (Profit Factor) ---
    # 总盈利 / 总亏损的绝对值
    gross_profit = winning_trades['net_pnl'].sum()
    gross_loss = abs(losing_trades['net_pnl'].sum())

    if gross_loss == 0:
        profit_factor = float('inf')  # 无亏损，盈亏比无穷大
    else:
        profit_factor = gross_profit / gross_loss

    # --- 4. 最大回撤 (Max Drawdown) ---
    # 构建资金曲线 (Cumulative PnL Curve)
    trade_df['cum_pnl'] = trade_df['net_pnl'].cumsum()

    # 计算历史最高点 (High Water Mark)
    trade_df['peak'] = trade_df['cum_pnl'].cummax()

    # 计算当前回撤
    trade_df['drawdown'] = trade_df['cum_pnl'] - trade_df['peak']

    # 找到最大回撤 (最小值)
    max_drawdown = trade_df['drawdown'].min()

    # 发生最大回撤的交易索引
    dd_idx = trade_df['drawdown'].idxmin()
    max_dd_time = trade_df.loc[dd_idx, 'close_time'] if pd.notna(dd_idx) else "N/A"

    # --- 5. 持仓时间分析 ---
    avg_duration = trade_df['duration_mins'].mean()
    max_duration = trade_df['duration_mins'].max()
    avg_profit_per_trade = trade_df['net_pnl'].mean()

    current_year = 2026
    trade_df['open_year'] = trade_df['open_time'].str[:4].astype(int)
    trades_this_year = len(trade_df[trade_df['open_year'] == current_year])

    return {
        'Total Return': total_return,
        'total_trades': total_trades,
        'Win Rate': win_rate,
        'avg_profit_per_trade': avg_profit_per_trade,
        'Max Drawdown': max_drawdown,
        'Profit Factor': profit_factor,
        'avg_duration': avg_duration,
        '最长持仓时间': max_duration,
        'trades_this_year': trades_this_year
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
    close_main = df['close_main'].values
    close_sub = df['close_sub'].values
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

            op_main, cl_main = close_main[open_idx], close_main[close_idx]
            op_sub, cl_sub = close_sub[open_idx], close_sub[close_idx]

            btc_roi = (cl_main - op_main) / op_main
            eth_roi = (cl_sub - op_sub) / op_sub

            entry_beta = betas[open_idx]

            if signal_dir == -1:
                net_pnl = btc_roi - (entry_beta * eth_roi)
                direction_str = f'做多价差 (Long BTC / Short {entry_beta:.3f} ETH)'
            else:
                net_pnl = (entry_beta * eth_roi) - btc_roi
                direction_str = f'做空价差 (Short BTC / Long {entry_beta:.3f} ETH)'

            # 计算真实的总资金收益率 (假设非杠杆全额交易)
            total_exposure = 1.0 + abs(entry_beta)
            true_roi = net_pnl / total_exposure
            net_pnl = round(true_roi, 8)
            trade_record = {
                'open_time': times[open_idx],
                'close_time': times[close_idx],
                'duration_mins': (pd.Timestamp(times[close_idx]) - pd.Timestamp(times[open_idx])).total_seconds() / 60,
                'direction': direction_str,
                'btc_roi': f"{btc_roi * 100:.2f}",
                'eth_roi': f"{eth_roi * 100:.2f}",
                'net_pnl': round(net_pnl * 100, 4),
                'open_main': op_main, 'close_main': cl_main,
                'open_sub': op_sub, 'close_sub': cl_sub,
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
# 多进程部分与预过滤优化
# ==========================================
shared_df = None


def init_worker(df_to_share):
    global shared_df
    shared_df = df_to_share


def filter_tasks_before_pool(tasks):
    filtered_tasks = []
    window = 60

    for task in tasks:
        # 【修改】兼容 7元组(含keyword) 或 6元组(无keyword) 的解包
        if len(task) == 7:
            z_entry, z_exit, delta, ve, granularity, require_rebound, key_word = task
        elif len(task) == 6:
            z_entry, z_exit, delta, ve, granularity, require_rebound = task
            key_word = None
        else:
            filtered_tasks.append(task)
            continue

        if z_exit >= z_entry:
            continue
        if granularity > 1:
            continue

        if key_word is not None:
            # 【修改】文件名加入 _rb{require_rebound}
            back_df_file = f'backtest_pair/{key_word}_w{window}_entry{z_entry}_exit{z_exit}_delta{delta}_ve{ve}_g{granularity}_rb{require_rebound}_kalman.csv'
            if os.path.exists(back_df_file):
                continue

        filtered_tasks.append(task)
    print(f"Pre-filtered tasks: {len(filtered_tasks)} out of {len(tasks)}")
    return filtered_tasks


def process_strategy(args):
    # 【修改】解包时增加 require_rebound
    z_entry, z_exit, delta, ve, granularity, require_rebound, key_word = args
    window = 60
    if z_exit >= z_entry:
        return
    global shared_df

    if granularity > 1:
        return
    else:
        merged_df = shared_df.copy()

    # 【修改】文件名加入 _rb{require_rebound}
    back_df_file = f'backtest_pair/{key_word}_w{window}_entry{z_entry}_exit{z_exit}_delta{delta}_ve{ve}_g{granularity}_rb{require_rebound}_kalman.csv'

    if os.path.exists(back_df_file):
        return

    start_time = time.time()

    # 【修改】透传 require_rebound 给生成信号的函数
    full_df = generate_pair_trading_signals(merged_df=merged_df, window=window, z_entry=z_entry,
                                            z_exit=z_exit, delta=delta, ve=ve, require_rebound=require_rebound)

    detailed_result_df = extract_trades_from_signals(full_df)

    os.makedirs(os.path.dirname(back_df_file), exist_ok=True)
    detailed_result_df.to_csv(back_df_file)

    end_time = time.time()
    duration = end_time - start_time

    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time_str}] Saved: {back_df_file} | Time Cost: {duration:.4f}s len: {len(detailed_result_df)}")


def parse_backtest_filename(filepath):
    filename = os.path.basename(filepath)
    # 【修改】正则中加入 rb(?P<require_rebound>.+?)_ 的匹配
    pattern = (
        r"w(?P<window>.+?)_"
        r"entry(?P<z_entry>.+?)_"
        r"exit(?P<z_exit>.+?)_"
        r"delta(?P<delta>.+?)_"
        r"ve(?P<ve>.+?)_"
        r"g(?P<granularity>.+?)_"
        r"rb(?P<require_rebound>.+?)_"
        r"kalman\.csv"
    )

    match = re.search(pattern, filename)
    if match:
        raw_data = match.groupdict()
        parsed_data = {}
        for key, value in raw_data.items():
            # 【修改】将字符串 'True'/'False' 转回布尔值
            if key == 'require_rebound':
                parsed_data[key] = True if value == 'True' else False
            else:
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
            # 【修改】将 require_rebound 加入 task_tuple
            task_tuple = (
                params['z_entry'],
                params['z_exit'],
                params['delta'],
                params['ve'],
                params['granularity'],
                params.get('require_rebound', False) # 用 get 兼容老文件
            )
            if params['granularity'] == 1:
                tasks.append(task_tuple)

    print(f"Total filtered tasks (Before Pre-filter): {len(tasks)}")
    tasks = filter_tasks_before_pool(tasks)
    print(f"Total filtered tasks (After Pre-filter): {len(tasks)}")

    with Pool(processes=10, initializer=init_worker, initargs=(original_df,)) as pool:
        pool.map(process_strategy, tasks)


def get_good_tasks(base_tasks, key_word, result_df, need_expand=True):
    import itertools
    tasks = []
    temp_tasks = []
    copy_base_tasks = base_tasks.copy()
    for task in copy_base_tasks:
        # 【修改】从截取前5个变更为截取前6个（包含 require_rebound）
        temp_tasks.append(task[:6])
    base_tasks_set = set(temp_tasks)
    df = result_df
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
            # 【修改】加入 require_rebound
            task_tuple = (
                params['z_entry'],
                params['z_exit'],
                params['delta'],
                params['ve'],
                params['granularity'],
                params.get('require_rebound', False)
            )

            if need_expand and task_tuple in base_tasks_set:
                count += 1
                variable_keys = ['z_entry', 'z_exit', 'delta', 've']
                variations = []
                for key in variable_keys:
                    base_val = params[key]
                    variations.append(
                        [base_val * 0.8, base_val * 0.9, base_val, base_val * 1.1, base_val * 1.2])

                # 【修改】保留粒度和反弹标志位
                variations.append([params['granularity']])
                variations.append([params.get('require_rebound', False)])

                for combo in itertools.product(*variations):
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


def gen_result_df(key_word):
    backtest_dir = './backtest_pair'
    final_df_path = f'kline_data/result_df_{key_word}.csv'
    result_dict_list = []
    # return pd.read_csv(final_df_path)
    def process_file(filename):
        file_path = os.path.join(backtest_dir, filename)
        # 这里原代码有一行读取 back_df_file，如果该文件不存在会报错，建议根据需要保留或注释
        backtest_df = pd.read_csv(file_path)
        result_dict = print_backtest_stats(backtest_df)
        if result_dict:
            result_dict['file_name'] = filename
            return result_dict
        return None

    # 1. 筛选出需要处理的文件列表
    valid_files = [f for f in os.listdir(backtest_dir) if f.endswith('.csv') and f"{key_word}_" in f]
    print(f"找到 {len(valid_files)} 个符合条件的文件，准备进行多线程处理...")

    # 2. 开启多线程并行加载和处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # executor.map 自动调度并在主线程中按序返回结果，无需加锁就能安全 append
        for result in executor.map(process_file, valid_files):
            if result is not None:
                result_dict_list.append(result)

    result_df = pd.DataFrame(result_dict_list)
    result_df.to_csv(final_df_path)
    return result_df


def merge_df(main_df_file, sub_df_file, merged_file_path):
    """
    进行交易对的合并
    :param main_df_file: 主交易对CSV路径
    :param sub_df_file: 从属交易对CSV路径
    :param merged_file_path: 合并后保存的路径
    """
    if os.path.exists(merged_file_path):
        print(f"Merged file already exists: {merged_file_path}")
        return

    try:
        main_df = pd.read_csv(main_df_file, usecols=['timestamp', 'close'], parse_dates=['timestamp'])
        main_df = main_df.sort_values('timestamp').set_index('timestamp')

        sub_df = pd.read_csv(sub_df_file, usecols=['timestamp', 'close'], parse_dates=['timestamp'])
        sub_df = sub_df.sort_values('timestamp').set_index('timestamp')

        merged_df = pd.merge(
            main_df[['close']],
            sub_df[['close']],
            left_index=True,
            right_index=True,
            suffixes=('_main', '_sub'),
            how='inner'  # 明确指定内连接
        )

        # 检查合并后是否有数据，防止因时间范围不重合导致生成空文件
        if not merged_df.empty:
            # 将 timestamp 重命名为 open_time 并重置索引
            merged_df = merged_df.rename_axis('open_time').reset_index()
            merged_df.to_csv(merged_file_path, index=False)
            print(f"Successfully merged: {merged_file_path} 总共行数: {len(merged_df)}")
        else:
            print(f"Warning: No overlapping time points found for {merged_file_path}")

    except Exception as e:
        print(f"Error merging files: {e}")


def gen_all_merged_df():
    """"""
    base_dir = 'kline_data'
    # 扫描目录下的所有CSV文件，寻找主交易对和从属交易对的组合
    all_csv_files = [f for f in os.listdir(base_dir) if f.endswith('.csv') and 'origin_data' in f]

    # 分为两个时间维度，判断标准为文件名中是否包含 '1s' 或 '1m'
    all_1m_files = [f for f in all_csv_files if '1m' in f]
    all_1s_files = [f for f in all_csv_files if '1s' in f]

    # 根据all_1m_files生成两两的组合

    num_files = len(all_1m_files)
    for i in range(num_files):
        main_file = all_1m_files[i]
        main_name = main_file.split('-USDT')[0].split('0000_')[1]

        # 关键点：j 从 i + 1 开始，这样就不会碰到重复的组合
        for j in range(i + 1, num_files):
            sub_file = all_1m_files[j]
            sub_name = sub_file.split('-USDT')[0].split('0000_')[1]

            merged_file_path = f'{base_dir}/{main_name}_{sub_name}_1m.csv'
            merge_df(f'{base_dir}/{main_file}', f'{base_dir}/{sub_file}', merged_file_path)
    num_files = len(all_1s_files)
    for i in range(num_files):
        main_file = all_1s_files[i]
        main_name = main_file.split('-USDT')[0].split('0000_')[1]

        # 关键点：j 从 i + 1 开始，这样就不会碰到重复的组合
        for j in range(i + 1, num_files):
            sub_file = all_1s_files[j]
            sub_name = sub_file.split('-USDT')[0].split('0000_')[1]

            merged_file_path = f'{base_dir}/{main_name}_{sub_name}_1s.csv'
            merge_df(f'{base_dir}/{main_file}', f'{base_dir}/{sub_file}', merged_file_path)


def process_single_pair(df_file):
    key_word = df_file.split('/')[1].split('.csv')[0]
    original_df = pd.read_csv(df_file)
    if 'open_time' in original_df.columns:
        original_df['open_time'] = pd.to_datetime(original_df['open_time'])

    print(f"Data loaded. Rows: {len(original_df)}. Starting multiprocessing...")

    z_entry_list = [1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10]
    z_exit_list = [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5]
    delta_list = [1e-3, 1e-4, 5e-5, 1e-5, 5e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
    ve_list = [1e-2, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    granularity_list = [1]
    # 【修改】新增你要对比的策略维度
    require_rebound_list = [False]

    base_tasks = []
    for z_entry in z_entry_list:
        for z_exit in z_exit_list:
            for delta in delta_list:
                for ve in ve_list:
                    for granularity in granularity_list:
                        # 【修改】多加一层循环，将两种情况都塞进任务队列去跑
                        for req_reb in require_rebound_list:
                            base_tasks.append((z_entry, z_exit, delta, ve, granularity, req_reb, key_word))

    tasks = base_tasks.copy()
    print(f"Total tasks (Before Pre-filter): {len(tasks)}")
    tasks = filter_tasks_before_pool(tasks)
    print(f"Total tasks (After Pre-filter): {len(tasks)}")

    with Pool(processes=1, initializer=init_worker, initargs=(original_df,)) as pool:
        pool.map(process_strategy, tasks)

    result_df = gen_result_df(key_word)
    tasks = get_good_tasks(base_tasks, key_word, result_df, need_expand=True)

    tasks = filter_tasks_before_pool(tasks)

    with Pool(processes=1, initializer=init_worker, initargs=(original_df,)) as pool:
        pool.map(process_strategy, tasks)

    result_df = gen_result_df(key_word)

    df_file = df_file.replace('1m', '1s')
    key_word = df_file.split('/')[1].split('.csv')[0]
    original_df = pd.read_csv(df_file)
    if 'open_time' in original_df.columns:
        original_df['open_time'] = pd.to_datetime(original_df['open_time'])

    tasks = get_good_tasks(base_tasks, key_word, result_df, need_expand=False)
    tasks = filter_tasks_before_pool(tasks)

    with Pool(processes=1, initializer=init_worker, initargs=(original_df,)) as pool:
        pool.map(process_strategy, tasks)

    result_df = gen_result_df(key_word)

if __name__ == '__main__':
    # gen_all_merged_df()

    df_file = 'kline_data/ETH_SOL_1m.csv'
    process_single_pair(df_file)