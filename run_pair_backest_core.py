import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta


# ==========================================
# 第一部分：因子计算模块 (The Alpha Math)
# ==========================================

def calc_kaufman_er(series, window=20):
    change = series.diff(window).abs()
    volatility = series.diff().abs().rolling(window=window).sum()
    er = change / volatility
    return er.fillna(0)


def calc_volatility_squeeze(series, short_win=60):
    price_mean = series.rolling(window=short_win).mean()
    price_std = series.rolling(window=short_win).std()
    price_z = (series - price_mean) / price_std
    return price_z.fillna(0), price_std.fillna(0)


# ==========================================
# 第二部分：策略回测引擎 (Backtest Engine)
# ==========================================

def run_backtest(df):
    print(">>> 终局架构: STARV 7.1 (均值回归 + 纯理论防暴毙机制) 启动...")
    df['log_spread'] = np.log(df['close_main'])

    # 1. 效率系数 (用于判断噪音 vs 趋势)
    df['ER_Fast'] = calc_kaufman_er(df['log_spread'], window=5)  # 微观动能
    df['ER_Slow'] = calc_kaufman_er(df['log_spread'], window=15)  # 局部结构相变监测

    # 2. Z-Score (用于测算偏离度)
    df['Price_Z'], df['Local_Std'] = calc_volatility_squeeze(df['log_spread'], short_win=60)

    # 3. Z-Score 的一阶导数 (速度)，用于监测是否发生“动量逃逸”
    df['Z_Velocity'] = df['Price_Z'].diff(3)

    df = df.dropna().reset_index(drop=True)
    print(f">>> 因子计算完毕，开始逐K线撮合，共 {len(df)} 根 1分钟 K线...")

    position = 0
    entry_price = 0.0
    entry_time = None

    trade_log = []
    current_equity = 1.0
    equity_curve = np.ones(len(df))

    FEE_RATE = 0.0004
    ROUND_TRIP_FEE = 2 * FEE_RATE

    # ================= 理论参数 (非拟合常数) =================
    Z_EXTREME_THRESH = 2.6  # 统计学极值：2.6 sigma (正态分布中 99% 的边界)
    ER_EXHAUST_THRESH = 0.35  # 物理极值：微观动能处于布朗运动态 (无序)

    # 【理论防线 1】相变切断：如果在持仓期间，局部结构突然变得极其有序 (ER > 0.65)，证明假设破产
    PHASE_TRANSITION_ER = 0.65
    # 【理论防线 2】时间引力失效：经验半衰期极限，假设为 20 分钟
    MAX_GRAVITY_MINS = 20
    # =========================================================

    for i in range(5, len(df)):
        current_time = df['open_time'].iloc[i]
        price = df['log_spread'].iloc[i]

        price_z = df['Price_Z'].iloc[i - 1]
        price_z_current = df['Price_Z'].iloc[i]
        er_fast = df['ER_Fast'].iloc[i]
        er_slow = df['ER_Slow'].iloc[i]
        z_velocity = df['Z_Velocity'].iloc[i]

        # -------------------
        # 入场逻辑：只在“无序”极值处捡硬币
        # -------------------
        if position == 0:
            if er_fast < ER_EXHAUST_THRESH:
                # 向上偏离极值且速度开始放缓，做空
                if price_z > Z_EXTREME_THRESH and z_velocity < 0:
                    position = -1
                    entry_price = price
                    entry_time = current_time
                    print(f"[入场] {current_time} | 逆势做空 | Z: {price_z:.2f}, ER: {er_fast:.2f}")

                # 向下偏离极值且速度开始放缓，做多
                elif price_z < -Z_EXTREME_THRESH and z_velocity > 0:
                    position = 1
                    entry_price = price
                    entry_time = current_time
                    print(f"[入场] {current_time} | 逆势做多 | Z: {price_z:.2f}, ER: {er_fast:.2f}")

        # -------------------
        # 出场逻辑：动态状态机证伪
        # -------------------
        elif position != 0:
            hold_time = (current_time - entry_time).total_seconds() / 60.0
            gross_return = (price - entry_price) * position
            net_return = gross_return - ROUND_TRIP_FEE
            exit_reason = None

            # 【圣杯时刻】完美回归均值
            if (position == 1 and price_z_current >= 0) or (position == -1 and price_z_current <= 0):
                exit_reason = "均值回归完成(Z=0)"

            # 【防线 1：相变切断】如果我们做回归，但市场打出了流畅的单边直线 (ER_Slow 飙升)
            # 意味着我们接飞刀接在了主升浪的起点，旧模型物理崩塌，立刻逃生！
            elif er_slow > PHASE_TRANSITION_ER and gross_return < 0:
                exit_reason = f"结构相变逃生(ER>{PHASE_TRANSITION_ER})"

            # 【防线 2：动量逃逸】Z-score 反向以极快的速度继续扩大 (列车加速)
            elif (position == 1 and z_velocity < -0.5) or (position == -1 and z_velocity > 0.5):
                if gross_return < -0.001:  # 仅在亏损且加速时切断
                    exit_reason = "动量逃逸(Z加速扩大)"

            # 【防线 3：引力失效】过了预定半衰期依然没有回到均线附近
            elif hold_time >= MAX_GRAVITY_MINS:
                exit_reason = f"引力失效({MAX_GRAVITY_MINS}m不归)"

            if exit_reason:
                current_equity += net_return
                gross_pnl_percent = gross_return * 100
                net_pnl_percent = net_return * 100

                trade_log.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'direction': 'Long' if position == 1 else 'Short',
                    'hold_time_mins': hold_time,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'gross_return': gross_return,
                    'net_return': net_return,
                    'gross_pnl_percent': gross_pnl_percent,
                    'net_pnl_percent': net_pnl_percent,
                    'reason': exit_reason
                })

                print(
                    f"  [出场] {current_time} | 原因: {exit_reason:<22} | 耗时: {hold_time}m | 毛利: {gross_pnl_percent:+.3f}% | 净利: {net_pnl_percent:+.3f}%")

                position = 0
                entry_price = 0.0

        if position == 0:
            equity_curve[i] = current_equity
        else:
            open_fee = 1 * FEE_RATE
            unrealized_gross = (price - entry_price) * position
            equity_curve[i] = current_equity + unrealized_gross - open_fee

    df['Equity'] = equity_curve
    return pd.DataFrame(trade_log), df


if __name__ == "__main__":
    # 请继续使用你的数据测试
    df = pd.read_csv("kline_data/BTC_ETH_1m.csv", parse_dates=['open_time'])

    trades_df, curve_df = run_backtest(df)

    if len(trades_df) > 0:
        print("\n" + "=" * 70)
        print("📊 STARV 7.1 Alpha 绩效统计 (全天候极值逆转 + 理论防御)")
        print("=" * 70)

        total_trades = len(trades_df)
        gross_winning = trades_df[trades_df['gross_return'] > 0]
        gross_losing = trades_df[trades_df['gross_return'] <= 0]
        gross_win_rate = len(gross_winning) / total_trades if total_trades > 0 else 0
        total_gross_pct = trades_df['gross_return'].sum() * 100
        avg_gross_win_pct = gross_winning['gross_return'].mean() * 100 if len(gross_winning) > 0 else 0
        avg_gross_loss_pct = gross_losing['gross_return'].mean() * 100 if len(gross_losing) > 0 else 0
        gross_pf = abs(gross_winning['gross_return'].sum() / gross_losing['gross_return'].sum()) if len(
            gross_losing) > 0 and gross_losing['gross_return'].sum() != 0 else float('inf')
        gross_expectancy = (gross_win_rate * avg_gross_win_pct) + ((1 - gross_win_rate) * avg_gross_loss_pct)

        net_winning = trades_df[trades_df['net_return'] > 0]
        net_losing = trades_df[trades_df['net_return'] <= 0]
        net_win_rate = len(net_winning) / total_trades if total_trades > 0 else 0
        total_net_pct = trades_df['net_return'].sum() * 100
        avg_net_win_pct = net_winning['net_return'].mean() * 100 if len(net_winning) > 0 else 0
        avg_net_loss_pct = net_losing['net_return'].mean() * 100 if len(net_losing) > 0 else 0
        net_pf = abs(net_winning['net_return'].sum() / net_losing['net_return'].sum()) if len(net_losing) > 0 and \
                                                                                          net_losing[
                                                                                              'net_return'].sum() != 0 else float(
            'inf')
        net_expectancy = (net_win_rate * avg_net_win_pct) + ((1 - net_win_rate) * avg_net_loss_pct)

        roll_max = curve_df['Equity'].cummax()
        drawdown_pct = (curve_df['Equity'] - roll_max) * 100
        max_drawdown_pct = drawdown_pct.min()

        print(f"总交易次数: {total_trades} 次")
        print("-" * 70)
        print(f"{'指标':<12} | {'无手续费 (Gross Alpha)':<22} | {'带手续费 (Net Alpha)':<22}")
        print("-" * 70)
        print(f"{'[胜率]':<12} | {gross_win_rate:>18.2%}     | {net_win_rate:>18.2%}")
        print(f"{'[总收益率]':<10} | {total_gross_pct:>18.2f}%    | {total_net_pct:>18.2f}%")
        print(f"{'[单笔期望]':<10} | {gross_expectancy:>18.4f}%    | {net_expectancy:>18.4f}%")
        print(f"{'[盈亏比]':<11} | {gross_pf:>18.2f}     | {net_pf:>18.2f}")
        print(f"{'[平均盈利]':<10} | {avg_gross_win_pct:>18.3f}%    | {avg_net_win_pct:>18.3f}%")
        print(f"{'[平均亏损]':<10} | {avg_gross_loss_pct:>18.3f}%    | {avg_net_loss_pct:>18.3f}%")
        print("-" * 70)
        print(f"最大回撤幅:  {max_drawdown_pct:.2f}% (含手续费)")

        print("\n🔍 出场原因分布 (这是检验理论防线是否起效的核心):")
        reason_counts = trades_df['reason'].value_counts()
        for reason, count in reason_counts.items():
            avg_gross_pnl = trades_df[trades_df['reason'] == reason]['gross_pnl_percent'].mean()
            print(f"   - {reason:<25}: {count:>3} 次 | 平均毛收益: {avg_gross_pnl:+.3f}%")

        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(curve_df['open_time'], curve_df['log_spread'], color='grey', alpha=0.5, label='Log Price')
        ax1.set_ylabel('Log Price', color='grey')
        ax2 = ax1.twinx()
        ax2.plot(curve_df['open_time'], curve_df['Equity'], color='red', linewidth=2, label='Strategy Equity')
        ax2.set_ylabel('Cumulative Equity', color='red')
        plt.title('STARV 7.1 Single Coin (Theoretical Mean Reversion + Phase Abort)')
        ax2.legend(loc='upper left')
        fig.tight_layout()
        plt.show()
    else:
        print("没有触发交易。")