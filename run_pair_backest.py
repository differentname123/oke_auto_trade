import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

# 假设 get_feature 是你本地的模块，保持引用
from get_feature import read_last_n_lines


# -----------------------------------------------------------------------------
# 1. 核心策略函数
# -----------------------------------------------------------------------------
def generate_pair_trading_signals(df_btc, df_eth, window=60, z_entry=3.0, z_exit=0.5):
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
    # 确保时间戳是索引且格式正确
    df_btc = df_btc.sort_values('open_time').set_index('open_time')
    df_eth = df_eth.sort_values('open_time').set_index('open_time')

    # 合并数据，只保留两者都有的时间点 (Inner Join)
    df = pd.merge(df_btc[['close']], df_eth[['close']], left_index=True, right_index=True, suffixes=('_btc', '_eth'))

    df.to_csv('kline/btc_eth.csv')
    print(len(df))
    # --- 2. 计算对数价格 ---
    df['log_btc'] = np.log(df['close_btc'])
    df['log_eth'] = np.log(df['close_eth'])

    # --- 3. 滚动计算动态对冲比率 (Rolling Beta) ---
    # Y = BTC, X = ETH (添加常数项截距)
    Y = df['log_btc']
    X = sm.add_constant(df['log_eth'])

    # 建立滚动模型
    rolling_model = RollingOLS(Y, X, window=window)
    rolling_res = rolling_model.fit()

    # [修改点1] 同时获取 Beta (log_eth系数) 和 Alpha (const截距)
    # params 包含 'const' 和 'log_eth' 两列
    df['beta'] = rolling_res.params['log_eth']
    df['alpha'] = rolling_res.params['const']

    # --- 4. 构建价差 (Spread) 和 Z-Score ---
    # [修改点2] Spread 计算修改为纯残差 (Residual)
    # 原公式: spread = log_btc - beta * log_eth (包含了不稳定的截距项)
    # 新公式: spread = log_btc - (beta * log_eth + alpha)
    # 这样得到的 spread 是纯粹的噪音，理论均值为0，更符合正态分布，消除了趋势带来的偏差
    df['spread'] = df['log_btc'] - (df['beta'] * df['log_eth'] + df['alpha'])

    # 计算价差的滚动均值和标准差
    # 注意：理论上残差的均值为0，但为了适应局部波动，依然计算滚动均值
    df['spread_mean'] = df['spread'].rolling(window=window).mean()
    df['spread_std'] = df['spread'].rolling(window=window).std()

    # Z-Score = (当前价差 - 均值) / 标准差
    df['z_score'] = (df['spread'] - df['spread_mean']) / df['spread_std']

    # --- 5. 生成交易信号 (State Machine) ---
    # 1 = 做空 Spread (卖BTC, 买ETH)
    # -1 = 做多 Spread (买BTC, 卖ETH)
    # 0 = 空仓

    signals = np.zeros(len(df))
    current_position = 0  # 初始空仓

    z_values = df['z_score'].values

    # 为了性能，将索引转换为列表或不使用索引循环，这里保持原逻辑但优化读取
    for i in range(window, len(df)):
        z = z_values[i]

        if np.isnan(z):
            continue

        # 逻辑判断
        if current_position == 0:
            # 入场逻辑
            if z > z_entry:
                current_position = 1  # 价差过大 -> 卖 BTC / 买 ETH
            elif z < -z_entry:
                current_position = -1  # 价差过小 -> 买 BTC / 卖 ETH

        elif current_position == 1:
            # 多头平仓逻辑 (Z 回归到 z_exit 以下)
            if z < z_exit:
                current_position = 0

        elif current_position == -1:
            # 空头平仓逻辑 (Z 回归到 -z_exit 以上)
            if z > -z_exit:
                current_position = 0

        signals[i] = current_position

    df['signal'] = signals

    # --- 6. 生成人类可读的交易方向描述 ---
    # 定义映射字典
    direction_map = {
        1: '做空价差 (卖BTC/买ETH)',
        -1: '做多价差 (买BTC/卖ETH)',
        0: '空仓观望'
    }
    df['trade_direction'] = df['signal'].map(direction_map)

    return df


def print_backtest_stats(trade_df):
    """
    输入: extract_trades_from_signals 生成的 trade_df
    输出: 打印详细的策略回测绩效报告
    """
    if trade_df.empty:
        print("【统计报告】无交易记录，无法计算指标。")
        return

    print("-" * 50)
    print("             策略回测绩效报告 (Performance Report)")
    print("-" * 50)

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

    # --- 6. 打印结果 ---
    print(f"1. 交易概况")
    print(f"   - 总交易次数: {total_trades} 次")
    print(f"   - 盈利次数:   {win_count} 次")
    print(f"   - 亏损次数:   {loss_count} 次")
    print(f"   - 胜率 (Win Rate): {win_rate:.2%}")
    print(f"")

    print(f"2. 收益表现")
    print(f"   - 累计总收益率: {total_return:.2%} (单利累加)")
    print(f"   - 平均单笔收益: {avg_return:.4%}")
    print(f"   - 盈亏比 (Profit Factor): {profit_factor:.2f} (建议 > 1.5)")
    print(f"")

    print(f"3. 风险评估")
    print(f"   - 最大回撤 (Max Drawdown): {max_drawdown:.2%} (资金曲线峰值回落幅度)")
    print(f"   - 最糟糕的交易时刻: {max_dd_time}")
    print(f"")

    print(f"4. 时间效率")
    print(f"   - 平均持仓时间: {avg_duration:.1f} 分钟")
    print(f"   - 最长持仓时间: {max_duration:.1f} 分钟")
    print("-" * 50)

    # 可选：返回关键指标字典供后续使用
    return {
        'Total Return': total_return,
        'total_trades':total_trades,
        'Win Rate': win_rate,
        'Max Drawdown': max_drawdown,
        'Profit Factor': profit_factor,
        'avg_duration':avg_duration,
        '最长持仓时间': max_duration
    }

def extract_trades_from_signals(full_df):
    """
    将逐行的时间序列信号转换为每一笔交易的详细报表
    包含：开平仓的 Z-Score/Beta/Alpha/Spread，以及 BTC/ETH 分别的收益率
    """
    # --- 1. 数据预处理 (修复 KeyError) ---
    # 如果 open_time 是索引，reset_index 会将其变成一列
    df = full_df.reset_index()

    # 检查列名，确保时间列叫 'open_time'
    if 'open_time' not in df.columns:
        if 'index' in df.columns:
            df.rename(columns={'index': 'open_time'}, inplace=True)
        elif 'timestamp' in df.columns:
            df.rename(columns={'timestamp': 'open_time'}, inplace=True)

    # 确保有 alpha 列 (如果你之前的代码没生成 alpha，这里可能会报错，请确保 upstream 有 alpha)
    # 如果 upstream 没有 alpha，可以注释掉下面涉及 alpha 的行

    # 计算前一行的信号
    df['prev_signal'] = df['signal'].shift(1).fillna(0)

    trades = []

    # --- 2. 寻找开仓点 ---
    # 逻辑：当前信号不为0，且前一时刻信号为0
    open_indices = df[(df['signal'] != 0) & (df['prev_signal'] == 0)].index

    for open_idx in open_indices:
        # 获取【开仓】时刻的数据行
        open_row = df.loc[open_idx]
        signal_dir = open_row['signal']  # 1 (做空Spread) 或 -1 (做多Spread)

        # --- 3. 寻找对应的平仓点 ---
        # 逻辑：从开仓点往后找，找到第一个 signal 变回 0 的行
        future_data = df.loc[open_idx:]
        close_candidates = future_data[future_data['signal'] == 0]

        if close_candidates.empty:
            continue  # 未平仓，跳过

        # 获取【平仓】时刻的数据行
        close_row = close_candidates.iloc[0]

        # --- 4. 计算收益率细节 ---
        # 计算单腿收益率 (无论做多做空，先算价格变化百分比)
        btc_roi_raw = (close_row['close_btc'] - open_row['close_btc']) / open_row['close_btc']
        eth_roi_raw = (close_row['close_eth'] - open_row['close_eth']) / open_row['close_eth']

        # 计算策略净收益 (Net PnL)
        # 逻辑：
        # Signal -1 (做多Spread): 买BTC，卖ETH -> 赚 BTC涨幅，亏 ETH涨幅
        # Signal  1 (做空Spread): 卖BTC，买ETH -> 亏 BTC涨幅，赚 ETH涨幅
        if signal_dir == -1:
            # Long Spread: Long BTC, Short ETH
            net_pnl = btc_roi_raw - eth_roi_raw
            direction_str = '做多价差 (Long BTC/Short ETH)'
        else:
            # Short Spread: Short BTC, Long ETH
            net_pnl = eth_roi_raw - btc_roi_raw
            direction_str = '做空价差 (Short BTC/Long ETH)'

        # --- 5. 构建详细记录 ---
        trade_record = {
            # --- 时间与方向 ---
            'open_time': open_row['open_time'],
            'close_time': close_row['open_time'],
            'duration_mins': (close_row['open_time'] - open_row['open_time']).total_seconds() / 60,
            'direction': direction_str,

            # --- 收益率表现 ---
            'btc_roi': f"{btc_roi_raw * 100:.2f}",  # BTC 单腿涨跌幅
            'eth_roi': f"{eth_roi_raw * 100:.2f}",  # ETH 单腿涨跌幅
            'net_pnl': round(net_pnl * 100, 2),  # 策略净收益 (数值)

            # --- 价格快照 ---
            'open_btc': open_row['close_btc'],
            'close_btc': close_row['close_btc'],
            'open_eth': open_row['close_eth'],
            'close_eth': close_row['close_eth'],

            # --- 核心指标对比 (Entry vs Exit) ---
            # Z-Score
            'entry_z': open_row['z_score'],
            'exit_z': close_row['z_score'],

            # Spread (价差/残差)
            'entry_spread': open_row['spread'],
            'exit_spread': close_row['spread'],

            # Beta (对冲比率) - 观察持仓期间 Beta 是否剧烈漂移
            'entry_beta': open_row['beta'],
            'exit_beta': close_row['beta'],

            # Alpha (截距)
            'entry_alpha': open_row.get('alpha', 0),  # 使用 .get 防止列不存在报错
            'exit_alpha': close_row.get('alpha', 0),
            'signal': signal_dir,


        }

        trades.append(trade_record)

    if not trades:
        return pd.DataFrame()

    return pd.DataFrame(trades)


if __name__ == '__main__':
    # 配置你的文件路径
    btc_file = r"W:\project\python_project\oke_auto_trade\kline_data\origin_data_1m_10000000_BTC-USDT-SWAP_2026-02-13.csv"
    eth_file = r'W:\project\python_project\oke_auto_trade\kline_data\origin_data_1m_10000000_ETH-USDT-SWAP_2026-02-13.csv'
    back_df_file = 'backtest/pair_trading_results.csv'

    # 这里原代码有一行读取 back_df_file，如果该文件不存在会报错，建议根据需要保留或注释
    backtest_df = pd.read_csv(back_df_file)
    print_backtest_stats(backtest_df)

    # 读取数据
    df_btc = read_last_n_lines(btc_file, 10000000)
    df_btc['open_time'] = pd.to_datetime(df_btc['timestamp'])

    df_eth = read_last_n_lines(eth_file, 10000000)
    df_eth['open_time'] = pd.to_datetime(df_eth['timestamp'])

    # --- 运行策略 ---
    full_df = generate_pair_trading_signals(df_btc, df_eth, window=6000)


    trade_cols = [
        'close_btc',  # BTC价格
        'close_eth',  # ETH价格
        'signal',  # 信号数值
        'trade_direction',  # 交易方向 (中文描述)
        'z_score',  # 判断依据：Z-Score
        'beta',  # 动态对冲比率
        'alpha', # [新增] 截距
        'spread'  # 原始价差 (残差)
    ]

    # 过滤出有信号的行（如果想看全部数据，去掉这个过滤即可）
    detailed_result_df = extract_trades_from_signals(full_df)

    # 如果 detailed_result_df 为空，说明没有触发交易，打印后5行原始数据查看状态
    if detailed_result_df.empty:
        print("当前数据范围内未触发交易信号。以下是最近的市场状态：")
        print(full_df[trade_cols].tail())
    else:
        print(f"策略运行结束，共捕捉到 {len(detailed_result_df)} 个持仓周期数据点。")
        print("\n--- 详细交易数据预览 (前10行) ---")
        print(detailed_result_df.head(10))

        print("\n--- 最近的交易状态 (后10行) ---")
        print(detailed_result_df.tail(10))

    # 你可以将结果保存为CSV以便深度分析
    detailed_result_df.to_csv(back_df_file)