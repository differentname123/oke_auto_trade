import re

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
# -----------------------------------------------------------------------------
# 0. 辅助算法: 卡尔曼滤波计算动态对冲比率 (纯血卡尔曼升级版)
# -----------------------------------------------------------------------------
def calculate_kalman_hedge_ratio(x, y, delta=1e-5, ve=1e-3):
    """
    使用卡尔曼滤波计算动态的 Alpha (截距)、Beta (斜率) 及其自带的测量方差
    模型: y = alpha + beta * x
    """
    n_obs = len(x)

    # 初始化状态向量 [alpha, beta]
    state_mean = np.zeros(2)
    # 初始化状态协方差矩阵 (初始不确定性设大一点)
    state_cov = np.ones((2, 2)) * 1.0

    # 过程噪声矩阵 Q (控制参数变化的灵活性)
    Q = np.eye(2) * delta

    # 测量噪声方差 R
    R = ve

    # 存放结果
    betas = np.zeros(n_obs)
    alphas = np.zeros(n_obs)
    spreads = np.zeros(n_obs)
    std_devs = np.zeros(n_obs)  # 【新增】存放卡尔曼原生预测误差的标准差

    # 转换为 numpy 数组以提高循环速度
    x_arr = x.values if isinstance(x, pd.Series) else x
    y_arr = y.values if isinstance(y, pd.Series) else y

    for t in range(n_obs):
        # 1. 预测阶段 (Prediction)
        state_pred = state_mean
        cov_pred = state_cov + Q

        # 2. 构建观测矩阵 H
        H = np.array([1.0, x_arr[t]])

        # 3. 计算预测误差 (Innovation) 和 误差方差
        y_pred = state_pred[0] + state_pred[1] * x_arr[t]
        error = y_arr[t] - y_pred  # 纯粹残差

        # 【核心提取】S = H P H.T + R，这就是当前时刻预测误差的理论方差
        S = H @ cov_pred @ H.T + R

        # 记录当前的标准差 (方差开根号)
        std_devs[t] = np.sqrt(S)

        # 4. 计算卡尔曼增益 (Kalman Gain)
        K = cov_pred @ H.T / S

        # 5. 更新阶段 (Update)
        state_mean = state_pred + K * error
        state_cov = cov_pred - np.outer(K, H) @ cov_pred

        # 记录结果
        alphas[t] = state_mean[0]
        betas[t] = state_mean[1]
        spreads[t] = error

    return betas, alphas, spreads, std_devs


# -----------------------------------------------------------------------------
# 1. 核心策略函数 (卡尔曼滤波优化版)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 1. 核心策略函数 (无滚动窗口的纯卡尔曼 Z-Score 优化版)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 1. 核心策略函数 (无滚动窗口的纯卡尔曼 Z-Score 优化版)
# -----------------------------------------------------------------------------
def generate_pair_trading_signals(merged_df=None, main_col='close_btc', sub_col='close_eth', window=60, z_entry=3.0, z_exit=0.5, delta=1e-5, ve=1e-3):
    """
    输入:
        merged_df: 包含价格数据的 DataFrame。
        main_col: 【新增】主资产列名（对应原代码中的 close_eth，作为 Y 变量）。
        sub_col:  【新增】从资产列名（对应原代码中的 close_btc，作为 X 变量）。
        window: 不再用于计算移动平均，仅作为卡尔曼滤波的“预热期(Burn-in Period)”，
                跳过最开始不稳定的一段时期，避免产生错误信号。
        ve: 卡尔曼滤波测量噪声方差。
    """

    # --- 1. 数据对齐 ---
    if merged_df is not None:
        df = merged_df

    # --- 2. 计算对数价格 (通用化处理) ---
    # 使用通用的列名存储对数价格，避免硬编码特定币种
    df['log_main'] = np.log(df[main_col])
    df['log_sub'] = np.log(df[sub_col])

    # --- 3. 动态计算 Hedge Ratio (Kalman Filter) ---
    # 【修改】使用通用的 log_main 和 log_sub 作为输入
    # 注意：根据原代码逻辑，main (原eth) 是第一个参数，sub (原btc) 是第二个参数
    betas, alphas, kf_spreads, kf_std_devs = calculate_kalman_hedge_ratio(
        df['log_sub'],
        df['log_main'],
        delta=delta,
        ve=ve
    )

    df['beta'] = betas
    df['alpha'] = alphas
    df['spread'] = kf_spreads

    # --- 4. Z-Score 计算 (抛弃 Rolling，使用卡尔曼原生方差) ---
    df['spread_std'] = kf_std_devs

    # 卡尔曼滤波预测误差的数学期望（均值）天然为 0
    # 所以 Z-Score 直接等于: 残差 / 卡尔曼标准差
    # 加上 np.divide 是为了防止极早期标准差为 0 导致计算报错
    df['z_score'] = np.divide(df['spread'], df['spread_std'],
                              out=np.zeros_like(df['spread']),
                              where=df['spread_std'] != 0)

    # --- 5. 生成交易信号 (State Machine) ---
    signals = np.zeros(len(df))
    current_position = 0
    z_values = df['z_score'].values
    z_values = np.nan_to_num(z_values, nan=0.0)

    # 循环从 window 开始，单纯为了跳过卡尔曼刚初始化的震荡期
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

    return df


def print_backtest_stats(trade_df):
    """
    输入: extract_trades_from_signals 生成的 trade_df
    输出: 打印详细的策略回测绩效报告
    """
    if trade_df.empty:
        print("【统计报告】无交易记录，无法计算指标。")
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

    # # --- 6. 打印结果 ---
    # print(f"1. 交易概况")
    # print(f"   - 总交易次数: {total_trades} 次")
    # print(f"   - 盈利次数:   {win_count} 次")
    # print(f"   - 亏损次数:   {loss_count} 次")
    # print(f"   - 胜率 (Win Rate): {win_rate:.2%}")
    # print(f"")
    #
    # print(f"2. 收益表现")
    # print(f"   - 累计总收益率: {total_return:.2%} (单利累加)")
    # print(f"   - 平均单笔收益: {avg_return:.4%}")
    # print(f"   - 盈亏比 (Profit Factor): {profit_factor:.2f} (建议 > 1.5)")
    # print(f"")
    #
    # print(f"3. 风险评估")
    # print(f"   - 最大回撤 (Max Drawdown): {max_drawdown:.2%} (资金曲线峰值回落幅度)")
    # print(f"   - 最糟糕的交易时刻: {max_dd_time}")
    # print(f"")
    #
    # print(f"4. 时间效率")
    # print(f"   - 平均持仓时间: {avg_duration:.1f} 分钟")
    # print(f"   - 最长持仓时间: {max_duration:.1f} 分钟")
    # print("-" * 50)

    # 可选：返回关键指标字典供后续使用
    return {
        'Total Return': total_return,
        'total_trades':total_trades,
        'Win Rate': win_rate,
        'avg_profit_per_trade':avg_profit_per_trade,
        'Max Drawdown': max_drawdown,
        'Profit Factor': profit_factor,
        'avg_duration':avg_duration,
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
    # 【修改】解包时增加 granularity 参数
    z_entry, z_exit, delta, ve, granularity = args
    window = 60
    if z_exit >= z_entry:
        return
    global shared_df

    # 【新增】K线数据合并粒度逻辑
    if granularity > 1:
        df_to_use = shared_df.copy()
        df_to_use.set_index('open_time', inplace=True)
        # 按分钟进行重采样，对于收盘价等状态数据，直接取周期的最后一条记录（last）
        df_to_use = df_to_use.resample(f'{granularity}min').last().dropna().reset_index()
        merged_df = df_to_use
    else:
        merged_df = shared_df.copy()

    # 【修改】保存的文件名中增加 g{granularity} 参数标识，防止不同粒度的结果相互覆盖
    back_df_file = f'backtest_pair/result_w{window}_entry{z_entry}_exit{z_exit}_delta{delta}_ve{ve}_g{granularity}_kalman.csv'

    if os.path.exists(back_df_file):
        # print(f"Skipping existing file: {back_df_file}")
        return

    start_time = time.time()

    # 【修改】运行策略时传入 ve，并使用重采样后的 merged_df
    full_df = generate_pair_trading_signals(merged_df=merged_df, window=window, z_entry=z_entry,
                                            z_exit=z_exit, delta=delta, ve=ve)

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


def parse_backtest_filename(filepath):
    filename = os.path.basename(filepath)

    # 正则表达式不变，依然完美匹配你的格式
    pattern = (
        r"result_"
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

        # --- 优化的类型转换逻辑 ---
        for key, value in raw_data.items():
            try:
                # 1. 优先尝试转换为整数 (例如 w60, g1)
                parsed_data[key] = int(value)
            except ValueError:
                try:
                    # 2. 如果不是整数，尝试转换为浮点数
                    # 这能自动处理 '2.5', '0.0001', 以及科学计数法 '1e-07'
                    parsed_data[key] = float(value)
                except ValueError:
                    # 3. 如果都不是，保留为字符串 (例如 '5m', 'True')
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
        # 从该行的 'file_name' 列解析参数
        params = parse_backtest_filename(row['file_name'])

        if params:
            # 严格按照你要求的顺序组装任务元组
            # tasks.append((z_entry, z_exit, delta, ve, granularity))
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

    # 适当调整进程数，根据你 CPU 核心数决定
    with Pool(processes=10, initializer=init_worker, initargs=(original_df,)) as pool:
        pool.map(process_strategy, tasks)


def get_good_tasks(base_tasks):
    import itertools
    tasks = []
    base_tasks_set = set(base_tasks)  # 转换为集合，极大提高查找效率
    final_df_path = 'kline_data/result_df.csv'
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
        # 从该行的 'file_name' 列解析参数
        params = parse_backtest_filename(row['file_name'])

        if params:
            # 严格按照你要求的顺序组装任务元组
            # tasks.append((z_entry, z_exit, delta, ve, granularity))
            task_tuple = (
                params['z_entry'],
                params['z_exit'],
                params['delta'],
                params['ve'],
                params['granularity']
            )

            # 判断解析出的任务是否在 base_tasks 中，只有存在才进行拓展搜寻
            if task_tuple in base_tasks_set:
                count += 1
                variable_keys = ['z_entry', 'z_exit', 'delta', 've']
                variations = []
                for key in variable_keys:
                    base_val = params[key]
                    # 生成：[原值-10%, 原值, 原值+10%]
                    variations.append([base_val * 0.9, base_val, base_val * 1.1])

                # 2. 添加不需要变动的参数 (granularity)
                # 注意：为了配合 product，必须放入列表中，哪怕只有一个元素
                variations.append([params['granularity']])

                # 3. 使用 itertools.product 进行全量组合
                # *variations 表示将列表解包传入，相当于传入了5个列表
                for combo in itertools.product(*variations):
                    # combo 此时已经是 (z_entry, z_exit, delta, ve, granularity) 的格式
                    tasks.append(combo)
            else:
                # 如果不在 base_tasks 中，则不扩展，直接将原组合加入任务集
                tasks.append(task_tuple)
                keep_count += 1
        else:
            no_params_count += 1
    print(f"Filtered {len(df)} tasks from Filtered {count} keep_count {keep_count} base tasks from DataFrame. Total tasks after expansion: {len(tasks)} (No params found for {no_params_count} rows).")
    return tasks

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

    window_list = [60]
    z_entry_list = [1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10]
    z_exit_list = [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5]
    delta_list = [1e-3, 1e-4, 5e-5, 1e-5, 5e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
    ve_list = [1e-2, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    granularity_list = [1, 2, 5, 10, 20, 60, 120, 240, 480, 1440]  # 【新增】合并的粒度列表，1代表1分钟。你可以随意修改添加如 [1, 5, 15, 60] 等粒度

    tasks = []
    for z_entry in z_entry_list:
        for z_exit in z_exit_list:
            for delta in delta_list:
                for ve in ve_list:
                    for granularity in granularity_list:  # 【新增】增加一层粒度的循环
                        tasks.append((z_entry, z_exit, delta, ve, granularity))
    tasks = get_good_tasks(tasks)


    print(f"Total tasks: {len(tasks)}")

    # 适当调整进程数，根据你 CPU 核心数决定
    with Pool(processes=5, initializer=init_worker, initargs=(original_df,)) as pool:
        pool.map(process_strategy, tasks)

    # run_good_params()