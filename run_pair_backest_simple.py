import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


def simple_pair_backtest(df_path, lookback=4320, z_entry=2.5, z_exit=0.0,
                         z_stop_loss=4.0, max_hold_bars=4320 * 2, fee=0.0006):
    """
    极简滚动 OLS 配对交易回测系统 (带详细日志与协整破裂保护)
    z_stop_loss: 当 Z-score 偏离达到此值时，强制止损出局
    max_hold_bars: 最大持仓时间(K线根数)，超时不回归则强制平仓认错 (默认设为lookback的两倍)
    """
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 开始加载并处理数据...")
    df = pd.read_csv(df_path)
    df['open_time'] = pd.to_datetime(df['open_time'])
    df = df.sort_values('open_time').reset_index(drop=True)

    df['log_main'] = np.log(df['close_main'])
    df['log_sub'] = np.log(df['close_sub'])

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 计算滚动参数 (Lookback={lookback})...")
    cov = df['log_main'].rolling(window=lookback).cov(df['log_sub'])
    var = df['log_sub'].rolling(window=lookback).var()
    df['beta'] = cov / var

    df['spread'] = df['log_main'] - df['beta'] * df['log_sub']
    df['spread_mean'] = df['spread'].rolling(window=lookback).mean()
    df['spread_std'] = df['spread'].rolling(window=lookback).std()
    df['z_score'] = (df['spread'] - df['spread_mean']) / df['spread_std']

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 开始生成交易信号并打印日志...")
    df['signal'] = 0
    signals = np.zeros(len(df))
    locked_betas = np.zeros(len(df))

    current_pos = 0
    hold_bars = 0  # 记录当前持仓K线数

    z_scores = df['z_score'].fillna(0).values
    log_mains = df['log_main'].values
    log_subs = df['log_sub'].values
    betas = df['beta'].fillna(0).values
    spread_means = df['spread_mean'].fillna(0).values
    spread_stds = df['spread_std'].fillna(1).values
    times = df['open_time'].values  # 用于日志打印时间

    # 获取收盘价数组，用于在循环中实时计算当前收益
    close_mains = df['close_main'].values
    close_subs = df['close_sub'].values

    entry_beta, entry_mean, entry_std = 0.0, 0.0, 0.0
    entry_price_main, entry_price_sub = 0.0, 0.0  # 记录入场价格
    period_max_z, period_min_z = 0.0, 0.0  # 记录持仓期间的最高和最低Z值
    period_max_ret, period_min_ret = 0.0, 0.0  # 记录持仓期间的最高和最低收益率

    for i in range(len(df)):
        z = z_scores[i]

        if current_pos == 0:
            if z > z_entry:
                current_pos = -1
                entry_beta, entry_mean, entry_std = betas[i], spread_means[i], spread_stds[i]
                entry_price_main, entry_price_sub = close_mains[i], close_subs[i]
                hold_bars = 1
                period_max_z, period_min_z = z, z
                period_max_ret, period_min_ret = -np.inf, np.inf  # 初始化期间收益极值

                print(f"[开仓-做空价差] 时间: {times[i]} | 触发 Z: {z:.2f} > {z_entry}")
                print(f"   -> 锁定参数: Beta={entry_beta:.4f}, Mean={entry_mean:.6f}, Std={entry_std:.6f}")

            elif z < -z_entry:
                current_pos = 1
                entry_beta, entry_mean, entry_std = betas[i], spread_means[i], spread_stds[i]
                entry_price_main, entry_price_sub = close_mains[i], close_subs[i]
                hold_bars = 1
                period_max_z, period_min_z = z, z
                period_max_ret, period_min_ret = -np.inf, np.inf  # 初始化期间收益极值

                print(f"[开仓-做多价差] 时间: {times[i]} | 触发 Z: {z:.2f} < {-z_entry}")
                print(f"   -> 锁定参数: Beta={entry_beta:.4f}, Mean={entry_mean:.6f}, Std={entry_std:.6f}")

        else:  # 持仓状态
            hold_bars += 1
            current_true_spread = log_mains[i] - entry_beta * log_subs[i]
            true_z_score = (current_true_spread - entry_mean) / entry_std

            # 实时收益率计算
            ret_m = (close_mains[i] / entry_price_main) - 1.0
            ret_s = (close_subs[i] / entry_price_sub) - 1.0

            # 计算开仓时的资金权重配比
            w_m = 1.0 / (1.0 + abs(entry_beta))
            w_s = abs(entry_beta) / (1.0 + abs(entry_beta))

            if current_pos == 1:
                current_ret = w_m * ret_m - w_s * ret_s
            else:
                current_ret = -w_m * ret_m + w_s * ret_s

            # 扣除双边手续费估算净收益
            current_net_ret = current_ret - (fee * 2)

            # 更新期间的最高和最低收益
            period_max_ret = max(period_max_ret, current_net_ret)
            period_min_ret = min(period_min_ret, current_net_ret)

            # 更新期间的最高和最低Z值
            period_max_z = max(period_max_z, true_z_score)
            period_min_z = min(period_min_z, true_z_score)

            # 1. 正常均值回归平仓 (带 ✅ 图标和换行分割)
            if (current_pos == -1 and true_z_score < z_exit) or (current_pos == 1 and true_z_score > -z_exit):
                print(
                    f"✅ [平仓-均值回归] 时间: {times[i]} | 真实 Z-score: {true_z_score:.2f} 触及退出线. 持仓时长: {hold_bars} K线.")
                print(f"   -> 📈 期间 Z 极值: 最高 {period_max_z:.2f} | 最低 {period_min_z:.2f}")
                print(
                    f"   -> 💰 收益情况: 本次收益 {current_net_ret * 100:.2f}% | 期间最高收益 {period_max_ret * 100:.2f}% | 期间最低收益 {period_min_ret * 100:.2f}%\n" + "-" * 70 + "\n")
                current_pos = 0
                hold_bars = 0

            # 2. 空间止损 (偏离极端) (带 🛑 图标和换行分割)
            elif (current_pos == -1 and true_z_score > z_stop_loss) or (
                    current_pos == 1 and true_z_score < -z_stop_loss):
                print(
                    f"🛑 [平仓-极端止损] 时间: {times[i]} | 真实 Z-score: {true_z_score:.2f} 突破止损线 {z_stop_loss}. 关系破裂，斩仓！")
                print(f"   -> 📈 期间 Z 极值: 最高 {period_max_z:.2f} | 最低 {period_min_z:.2f}")
                print(
                    f"   -> 💸 收益情况: 本次收益 {current_net_ret * 100:.2f}% | 期间最高收益 {period_max_ret * 100:.2f}% | 期间最低收益 {period_min_ret * 100:.2f}%\n" + "-" * 70 + "\n")
                current_pos = 0
                hold_bars = 0

            # 3. 时间止损 (死气沉沉) (带 ⏰ 图标和换行分割)
            elif hold_bars >= max_hold_bars:
                print(f"⏰ [平仓-超时离场] 时间: {times[i]} | 持仓达到最大上限 {max_hold_bars} K线，强制认错离场.")
                print(f"   -> 📈 期间 Z 极值: 最高 {period_max_z:.2f} | 最低 {period_min_z:.2f}")
                print(
                    f"   -> 📉 收益情况: 本次收益 {current_net_ret * 100:.2f}% | 期间最高收益 {period_max_ret * 100:.2f}% | 期间最低收益 {period_min_ret * 100:.2f}%\n" + "-" * 70 + "\n")
                current_pos = 0
                hold_bars = 0

        signals[i] = current_pos
        locked_betas[i] = entry_beta if current_pos != 0 else 0.0

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 信号生成完毕，计算盈亏...")
    df['position'] = signals
    df['locked_beta'] = locked_betas

    df['actual_pos'] = df['position'].shift(1).fillna(0)
    df['actual_locked_beta'] = df['locked_beta'].shift(1).fillna(0)
    df['pos_change'] = df['actual_pos'].diff().fillna(0).abs()

    df['ret_main'] = df['close_main'].pct_change()
    df['ret_sub'] = df['close_sub'].pct_change()

    df['gross_pnl'] = df['actual_pos'] * (df['ret_main'] - df['actual_locked_beta'] * df['ret_sub']) / (
            1 + df['actual_locked_beta'].abs())
    df['net_pnl'] = df['gross_pnl'] - (df['pos_change'] * fee)
    df['equity'] = (1 + df['net_pnl']).cumprod()

    total_trades = df['pos_change'].sum() / 2
    total_ret = df['equity'].iloc[-1] - 1

    print(f"\n=== 极简回归测试结果 ===")
    print(f"参数: Lookback={lookback}, Z-Entry={z_entry}, Z-StopLoss={z_stop_loss}, MaxHold={max_hold_bars}")
    print(f"总交易次数: {int(total_trades)} 笔")
    print(f"总净收益率: {total_ret * 100:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.plot(df['open_time'], df['equity'])
    plt.title(f'Equity Curve (Lookback={lookback}, Entry={z_entry}, StopLoss={z_stop_loss})')
    plt.grid()
    plt.show()

    return df

# 使用方法：
df_result = simple_pair_backtest('kline_data/BTC_ETH_1m.csv')