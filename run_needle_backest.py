import pandas as pd
import numpy as np
from datetime import timedelta

# ==========================================
# 1. 策略配置与数据准备
# ==========================================
CONFIG = {
    'ma_period': 20,  # 均线周期
    'std_dev_mult': 3.2,  # 布林带宽度 (接针建议宽一点)
    'vol_mult': 2.5,  # 成交量放大倍数
    'sl_pct': 0.01,  # 止损百分比 (1%)
    'fee_rate': 0.0005,  # 单边手续费 (0.05%)
    'capital_per_trade': 10000  # 单笔投入资金 (USDT)
}


def calculate_indicators(df):
    """
    计算技术指标，生成原始信号
    """
    df = df.copy()

    # 基础指标
    df['ma'] = df['close'].rolling(CONFIG['ma_period']).mean()
    df['std'] = df['close'].rolling(CONFIG['ma_period']).std()
    df['lower_band'] = df['ma'] - (CONFIG['std_dev_mult'] * df['std'])
    df['vol_ma'] = df['volume'].rolling(CONFIG['ma_period']).mean()

    # --- 信号计算逻辑 (基于上一根K线收盘确认) ---
    # 1. 价格极其便宜 (Low 击穿下轨)
    cond_price = df['low'] < df['lower_band']

    # 2. 恐慌性抛售 (成交量放大)
    cond_vol = df['volume'] > (df['vol_ma'] * CONFIG['vol_mult'])

    # 3. 拒绝下跌形态 (收盘收回下轨上方 或 长下影线)
    # 这里使用简单的判定：收盘价必须 > 下轨 (代表虚破)
    cond_shape = df['close'] > df['lower_band']

    # 综合信号 (Signal 为 1 代表该分钟收盘后出现了买入机会)
    df['signal_long'] = cond_price & cond_vol & cond_shape

    return df


# ==========================================
# 2. 核心回测引擎 (路径模拟)
# ==========================================
def run_backtest(df):
    # 预计算指标
    df = calculate_indicators(df)

    # 转换为列表或字典以提高迭代速度 (Pandas iterrows 较慢)
    records = df.to_dict('records')

    trades = []  # 交易记录
    stats = []  # 分钟快照

    # 账户状态变量
    position = False  # 是否持仓
    entry_price = 0.0
    entry_time = None
    stop_loss_price = 0.0
    take_profit_price = 0.0  # 动态止盈目标

    cum_realized_profit = 0.0  # 累计已平仓利润

    # 从第20根K线开始遍历
    for i in range(CONFIG['ma_period'], len(records)):
        curr_bar = records[i]
        prev_bar = records[i - 1]  # 信号来源

        last_close = prev_bar['close']

        # --- A. 构建分钟内的价格路径 ---
        # 逻辑：阳线(Close>=Open) -> 先跌后涨 -> [Last_Close, Open, Low, High, Close]
        #       阴线(Close<Open)  -> 先涨后跌 -> [Last_Close, Open, High, Low, Close]
        # 注意：Entry通常发生在 Open 时刻，所以 Open 在路径中非常重要

        price_path = []
        path_types = []  # 标记路径点的类型，用于判断是哪个价格触发了事件

        if curr_bar['close'] >= curr_bar['open']:
            # 阳线路径
            price_path = [last_close, curr_bar['open'], curr_bar['low'], curr_bar['high'], curr_bar['close']]
            path_types = ['last_close', 'open', 'low', 'high', 'close']
        else:
            # 阴线路径
            price_path = [last_close, curr_bar['open'], curr_bar['high'], curr_bar['low'], curr_bar['close']]
            path_types = ['last_close', 'open', 'high', 'low', 'close']

        # --- B. 路径游走与交易逻辑 ---

        # 记录本分钟内是否发生了交易（防止一分钟内开平多次，简化逻辑）
        trade_action_this_min = False

        # 临时变量，用于计算本分钟最低资产
        min_equity_in_bar = cum_realized_profit + (
            CONFIG['capital_per_trade'] if position else CONFIG['capital_per_trade'])

        # 遍历路径点
        for idx, price in enumerate(price_path):
            p_type = path_types[idx]

            # 1. 检查持仓止盈止损 (如果我们有持仓)
            if position:
                # 动态止盈更新: 如果当前价格均线发生变化(虽然分钟内ma通常视为固定，但这里用Bar的ma值)
                # 策略设定：价格触及当根K线的MA即止盈
                current_ma = curr_bar['ma']

                # --- 检查止损 ---
                if price <= stop_loss_price:
                    # 触发止损
                    exit_price = stop_loss_price
                    # 如果是跳空低开直接穿过止损 (例如 Open 就低于 SL)
                    if p_type == 'open' and price < stop_loss_price:
                        exit_price = price

                    pnl = (exit_price - entry_price) * (CONFIG['capital_per_trade'] / entry_price)
                    pnl -= CONFIG['capital_per_trade'] * CONFIG['fee_rate'] * 2

                    trades.append({
                        '开仓时间': entry_time,
                        '平仓时间': curr_bar['open_time'],  # 简化为分钟级时间
                        '持仓时间': curr_bar['open_time'] - entry_time,
                        '目标价格': current_ma,
                        '止损价格': stop_loss_price,
                        '方向': "做多(止损)",
                        '开仓价': entry_price,
                        '平仓价': exit_price,
                        '净盈亏': pnl
                    })
                    cum_realized_profit += pnl
                    position = False
                    trade_action_this_min = True
                    break  # 本分钟路径结束 (已平仓)

                # --- 检查止盈 ---
                elif price >= current_ma:
                    # 触发止盈
                    exit_price = current_ma
                    # 如果跳空高开
                    if p_type == 'open' and price > current_ma:
                        exit_price = price

                    pnl = (exit_price - entry_price) * (CONFIG['capital_per_trade'] / entry_price)
                    pnl -= CONFIG['capital_per_trade'] * CONFIG['fee_rate'] * 2

                    trades.append({
                        '开仓时间': entry_time,
                        '平仓时间': curr_bar['open_time'],
                        '持仓时间': curr_bar['open_time'] - entry_time,
                        '目标价格': current_ma,
                        '止损价格': stop_loss_price,
                        '方向': "做多(止盈)",
                        '开仓价': entry_price,
                        '平仓价': exit_price,
                        '净盈亏': pnl
                    })
                    cum_realized_profit += pnl
                    position = False
                    trade_action_this_min = True
                    break  # 本分钟路径结束

            # 2. 检查开仓 (如果我们没持仓，且允许开仓)
            # 只有在 'open' 时刻才能执行开仓动作 (模拟实盘：看到上一根信号，这根开盘买入)
            if not position and not trade_action_this_min and p_type == 'open':
                # 【修改点】增加 filter: 只有当开盘价低于均线一定幅度时才开仓，
                # 防止跳空高开直接在止盈线上方买入，导致 "止盈但亏手续费" 的情况。
                # 简单修复：确保价格小于 MA
                if prev_bar['signal_long'] and price < curr_bar['ma']:
                    position = True
                    entry_price = price  # 即 Open Price
                    entry_time = curr_bar['open_time']
                    stop_loss_price = entry_price * (1 - CONFIG['sl_pct'])
                    # 止盈是动态的，不需要存固定值，但为了记录可以存一个初始目标
                    take_profit_price = prev_bar['ma']

                    # --- C. 计算分钟统计数据 (Stats) ---

        # 计算本分钟内账户权益的最低点 (压力测试)
        # 这里的逻辑：遍历刚才的 path，计算每一个点的浮动盈亏

        current_equity = CONFIG['capital_per_trade']  # 假设只操作这一笔本金
        worst_equity = current_equity  # 默认

        if position:
            # 如果这分钟结束时还持仓，我们要看这一分钟内最惨跌到哪
            # 或者是这分钟中间持仓，后来平掉了，也要看持仓期间最惨跌到哪
            # 简化算法：直接取本分钟 Low 计算最差浮盈 (如果持多单)

            worst_price = curr_bar['low']
            # 如果 Low 比开仓价还低，计算最大浮亏
            float_pnl = (worst_price - entry_price) * (CONFIG['capital_per_trade'] / entry_price)
            worst_equity = current_equity + float_pnl
        else:
            # 如果当前空仓，最低资产就是本金
            worst_equity = current_equity

        # 资产比例
        invested = CONFIG['capital_per_trade'] if position else 0
        ratio = 1.0
        if invested > 0:
            ratio = worst_equity / invested

        stats.append({
            '时间': curr_bar['open_time'],  # 使用收盘时间或开盘时间皆可，统一即可
            '收盘价': curr_bar['close'],
            '持仓方向': '多单' if position else '空仓',
            '持仓单数': 1 if position else 0,
            '投入资金': invested,
            '本分钟最低资产值': worst_equity + cum_realized_profit,  # 加上已落袋的钱
            '资产比例': ratio,
            '累计利润': cum_realized_profit
        })

    # 【修改点】返回 df (包含技术指标列), trades, stats
    return df, pd.DataFrame(trades), pd.DataFrame(stats)


# ==========================================
# 3. 执行部分
# ==========================================

# 假设你已经读取了数据到 df 变量中
# df 必须包含列: ['open_time', 'open', 'high', 'low', 'close', 'volume']
# open_time 必须是 datetime 对象

if __name__ == "__main__":

    file_path = r"W:\project\python_project\oke_auto_trade\kline_data\origin_data_1m_10000000_BTC-USDT-SWAP_2026-02-13.csv"
    line_count = 10000
    # 注意：这里读取时确保有 volume 列
    df = pd.read_csv(file_path, nrows=line_count)
    df['open_time'] = pd.to_datetime(df['timestamp'])

    # 只保留 df 的 ['open_time', 'open', 'high', 'low', 'close', 'volume']这几个字段
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    debug_df = pd.read_csv('debug_data_with_indicators.csv')  # 读取一份完整数据用于调试验证
    # 原始代码中读取了 parquet，这里保留逻辑，但实际回测依赖上面的 df
    # 如果是为了复用之前的 trades 结构，可以保留，但 run_backtest 会生成新的
    df_trades_df = pd.read_parquet('paired_trades_needle.parquet')
    df_stats_df = pd.read_parquet('minute_stats_needle.parquet')

    try:
        print("开始回测...")
        if 'df' in locals():
            # 【修改点】接收三个返回值
            df_with_indicators, df_trades, df_stats = run_backtest(df)

            # 输出文件
            df_trades.to_parquet('paired_trades_needle.parquet', index=False)
            df_stats.to_parquet('minute_stats_needle.parquet', index=False)

            # 你可以将带有指标的原始数据也保存，方便验证
            df_with_indicators.to_csv('debug_data_with_indicators.csv', index=False)

            print(f"回测完成。")
            print(f"交易笔数: {len(df_trades)}")
            print(f"最终累计利润: {df_stats.iloc[-1]['累计利润']:.2f}")

            # 简单的验证打印
            print("\n--- 指标数据验证 (前1行) ---")
            print(df_with_indicators[['open_time', 'close', 'ma', 'lower_band', 'signal_long']].tail(1))

        else:
            print("错误：未找到数据变量 df，请先加载数据。")

    except Exception as e:
        print(f"发生错误: {e}")