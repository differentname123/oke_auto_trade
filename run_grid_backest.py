import pandas as pd
import os
from datetime import datetime
from multiprocessing import Pool, freeze_support

# 全局变量，用于在子进程中存储数据，避免重复传递
worker_df = None
from get_feature import read_last_n_lines


# ==========================================
# 模块 1：独立的价格计算函数 (信号发生器)
# 【保持完全原样，严禁修改逻辑】
# ==========================================
def get_minute_triggers(row, last_close, initial_base, current_level, grid_pct):
    """
    计算单根 K 线内触发的所有网格交易价格点。
    返回: triggers (触发事件列表), current_level (更新后的网格层级)
    """
    triggers = []

    # 1. 拆解这根 K 线的物理运行轨迹
    if row.close >= row.open:
        path = [last_close, row.open, row.low, row.high, row.close]
    else:
        path = [last_close, row.open, row.high, row.low, row.close]

    curr_p = path[0]

    # 2. 沿着轨迹计算触发点
    for i in range(1, len(path)):
        next_p = path[i]
        is_gap = (i == 1 and last_close != row.open)  # 第一段为可能的跳空缺口

        while curr_p != next_p:
            # 通过固定初始基准价与整数网格层级进行计算
            up_line = initial_base * (1 + (current_level + 1) * grid_pct)
            down_line = initial_base * (1 + (current_level - 1) * grid_pct)

            # 向上突破
            if next_p > curr_p and curr_p <= up_line <= next_p:
                exec_price = next_p if is_gap else up_line
                current_level += 1
                triggers.append({'time': row.timestamp, 'price': exec_price, 'type': 'UP', 'next_p': next_p,
                                 'level': current_level})
                curr_p = up_line

            # 向下突破
            elif next_p < curr_p and curr_p >= down_line >= next_p:
                exec_price = next_p if is_gap else down_line
                current_level -= 1
                triggers.append({'time': row.timestamp, 'price': exec_price, 'type': 'DOWN', 'next_p': next_p,
                                 'level': current_level})
                curr_p = down_line
            else:
                curr_p = next_p

    return triggers, current_level


# ==========================================
# 主回测程序 (模块 2 & 模块 3 的整合)
# ==========================================
def backtest_dynamic_grid(df, grid_pct=0.005, margin=10000.0, leverage=10.0, fee_rate=0.001, stop_loss_pct=0.02):
    """
    修改说明：
    1. 新增 stop_loss_pct 参数，用于计算止损价格。
    2. 在主循环开始处新增“止损检查”逻辑。
    """
    # 【性能优化 1：向量化处理时间】
    if isinstance(df['timestamp'].iloc[0], str):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.sort_values('timestamp').reset_index(drop=True)

    initial_base = df['open'].iloc[0]
    current_level = 0
    last_close = initial_base

    # 订单列表
    longs_stack = []  # 待平多单
    shorts_stack = []  # 待平空单
    paired_trades = []  # 配对交易记录
    minute_stats = []  # 每分钟资产记录

    cumulative_profit = 0.0
    notional = margin * leverage

    # 全局汇总变量
    total_long_qty = 0.0
    total_long_cost = 0.0
    total_short_qty = 0.0
    total_short_cost = 0.0

    for row in df.itertuples():

        # =========================================================
        # --- 新增步骤：止损检查 (Stop Loss Check) ---
        # 必须在处理网格触发信号之前，检查这一分钟的极端价格是否触发了止损
        # =========================================================

        # 1. 检查多单止损 (依据 Low Price)
        if longs_stack:
            surviving_longs = []
            for order in longs_stack:
                # 如果最低价击穿止损价
                if row.low <= order['stop_loss_price']:
                    # 确定执行价格：如果是跳空低开(Open < SL)，则按 Open 止损，否则按 SL 价格止损
                    sl_exec_price = order['stop_loss_price']
                    if row.open < sl_exec_price:
                        sl_exec_price = row.open

                    qty = order['qty']
                    gross_pnl = (sl_exec_price - order['price']) * qty
                    close_fee = (qty * sl_exec_price) * fee_rate
                    net_pnl = gross_pnl - order['fee'] - close_fee
                    holding_time = row.timestamp - order['time']

                    cumulative_profit += net_pnl

                    # 记录止损单
                    paired_trades.append({
                        '开仓时间': order['time'],
                        '平仓时间': row.timestamp,
                        '持仓时间': holding_time,
                        '目标价格': order['target_price'],
                        '止损价格': order['stop_loss_price'],  # 记录计划止损价
                        '方向': '做多(止损)',
                        '开仓价': order['price'],
                        '平仓价': sl_exec_price,
                        '净盈亏': net_pnl
                    })

                    # 更新全局统计变量
                    total_long_qty -= qty
                    total_long_cost -= (order['price'] * qty)
                else:
                    surviving_longs.append(order)
            longs_stack = surviving_longs

        # 2. 检查空单止损 (依据 High Price)
        if shorts_stack:
            surviving_shorts = []
            for order in shorts_stack:
                # 如果最高价击穿止损价
                if row.high >= order['stop_loss_price']:
                    # 确定执行价格：如果是跳空高开(Open > SL)，则按 Open 止损
                    sl_exec_price = order['stop_loss_price']
                    if row.open > sl_exec_price:
                        sl_exec_price = row.open

                    qty = order['qty']
                    gross_pnl = (order['price'] - sl_exec_price) * qty
                    close_fee = (qty * sl_exec_price) * fee_rate
                    net_pnl = gross_pnl - order['fee'] - close_fee
                    holding_time = row.timestamp - order['time']

                    cumulative_profit += net_pnl

                    # 记录止损单
                    paired_trades.append({
                        '开仓时间': order['time'],
                        '平仓时间': row.timestamp,
                        '持仓时间': holding_time,
                        '方向': '做空(止损)',
                        '目标价格': order['target_price'],
                        '止损价格': order['stop_loss_price'],
                        '开仓价': order['price'],
                        '平仓价': sl_exec_price,
                        '净盈亏': net_pnl
                    })

                    # 更新全局统计变量
                    total_short_qty -= qty
                    total_short_cost -= (order['price'] * qty)
                else:
                    surviving_shorts.append(order)
            shorts_stack = surviving_shorts

        # --- 步骤 A：获取当前分钟的所有触发信号 (原逻辑) ---
        triggers, current_level = get_minute_triggers(row, last_close, initial_base, current_level, grid_pct)

        # --- 步骤 B：模块 2 - 持仓维护与交易配对 ---
        for t in triggers:
            exec_price = t['price']
            exec_time = t['time']
            trigger_level = t['level']

            if t['type'] == 'UP':
                # 涨了：优先平多。
                closed_orders = []
                i = 0
                while i < len(longs_stack):
                    if exec_price >= longs_stack[i]['target_price'] - 1e-8:
                        popped_order = longs_stack.pop(i)
                        closed_orders.append(popped_order)

                        # 同步维护全局变量
                        total_long_qty -= popped_order['qty']
                        total_long_cost -= (popped_order['price'] * popped_order['qty'])
                        break
                    else:
                        i += 1

                if closed_orders:
                    for closed_order in closed_orders:
                        order = closed_order
                        qty = order['qty']
                        actual_close_price = max(exec_price, order['target_price'])
                        gross_pnl = (actual_close_price - order['price']) * qty
                        close_fee = (qty * actual_close_price) * fee_rate
                        net_pnl = gross_pnl - order['fee'] - close_fee
                        holding_time = exec_time - order['time']

                        cumulative_profit += net_pnl

                        paired_trades.append({
                            '开仓时间': order['time'],
                            '平仓时间': exec_time,
                            '持仓时间': holding_time,
                            '目标价格': order['target_price'],
                            '止损价格': order['stop_loss_price'],
                            '方向': '做多(止盈)',
                            '开仓价': order['price'],
                            '平仓价': actual_close_price,
                            '净盈亏': net_pnl
                        })
                else:
                    # 开空单
                    qty = notional / exec_price
                    # 【新增】：计算止损价格
                    sl_price = exec_price * (1 + stop_loss_pct)

                    shorts_stack.append({
                        'time': exec_time,
                        'price': exec_price,
                        'qty': qty,
                        'fee': notional * fee_rate,
                        'target_price': initial_base * (1 + (trigger_level - 1) * grid_pct),
                        'stop_loss_price': sl_price  # 存储止损价
                    })

                    # 同步维护全局变量
                    total_short_qty += qty
                    total_short_cost += (exec_price * qty)

            elif t['type'] == 'DOWN':
                # 跌了：优先平空。
                closed_orders = []
                i = 0
                while i < len(shorts_stack):
                    if exec_price <= shorts_stack[i]['target_price'] + 1e-8:
                        popped_order = shorts_stack.pop(i)
                        closed_orders.append(popped_order)

                        # 同步维护全局变量
                        total_short_qty -= popped_order['qty']
                        total_short_cost -= (popped_order['price'] * popped_order['qty'])
                        break
                    else:
                        i += 1

                if closed_orders:
                    for closed_order in closed_orders:
                        order = closed_order
                        qty = order['qty']
                        actual_close_price = min(exec_price, order['target_price'])
                        gross_pnl = (order['price'] - actual_close_price) * qty
                        close_fee = (qty * actual_close_price) * fee_rate
                        net_pnl = gross_pnl - order['fee'] - close_fee
                        holding_time = exec_time - order['time']

                        cumulative_profit += net_pnl

                        paired_trades.append({
                            '开仓时间': order['time'],
                            '平仓时间': exec_time,
                            '持仓时间': holding_time,
                            '方向': '做空(止盈)',
                            '目标价格': order['target_price'],
                            '止损价格': order['stop_loss_price'],
                            '开仓价': order['price'],
                            '平仓价': actual_close_price,
                            '净盈亏': net_pnl
                        })
                else:
                    # 开多单
                    qty = notional / exec_price
                    # 【新增】：计算止损价格
                    sl_price = exec_price * (1 - stop_loss_pct)

                    longs_stack.append({
                        'time': exec_time,
                        'price': exec_price,
                        'qty': qty,
                        'fee': notional * fee_rate,
                        'target_price': initial_base * (1 + (trigger_level + 1) * grid_pct),
                        'stop_loss_price': sl_price  # 存储止损价
                    })

                    # 同步维护全局变量
                    total_long_qty += qty
                    total_long_cost += (exec_price * qty)

        # --- 步骤 C：模块 3 - 每分钟最低资产值核算 ---
        current_orders = len(longs_stack) + len(shorts_stack)
        invested_capital = current_orders * margin
        if len(longs_stack) > 0 and len(shorts_stack) > 0:
            # 这里可能会有短暂的同时持仓（刚开仓尚未处理完），但在网格策略中是允许锁仓的，或者逻辑检查
            pass

        if longs_stack:
            direction = '多单'
        elif shorts_stack:
            direction = '空单'
        else:
            direction = '空仓'

        min_asset_value = invested_capital

        # 浮盈计算
        if longs_stack:
            worst_price = row.low
            total_fee = len(longs_stack) * notional * fee_rate
            unrealized = (worst_price * total_long_qty) - total_long_cost - total_fee
            min_asset_value += unrealized

        elif shorts_stack:
            worst_price = row.high
            total_fee = len(shorts_stack) * notional * fee_rate
            unrealized = total_short_cost - (worst_price * total_short_qty) - total_fee
            min_asset_value += unrealized

        minute_stats.append({
            '时间': row.timestamp,
            '收盘价': row.close,
            '持仓方向': direction,
            '持仓单数': current_orders,
            '投入资金': invested_capital,
            '本分钟最低资产值': min_asset_value,
            '资产比例': min_asset_value / invested_capital if invested_capital > 0 else 1,
            '累计利润': cumulative_profit
        })

        last_close = row.close

    return pd.DataFrame(paired_trades), pd.DataFrame(minute_stats)


def init_worker(shared_df):
    """
    子进程初始化函数
    """
    global worker_df
    worker_df = shared_df


def run_single_backtest(params):
    """
    单个回测任务的包装函数
    params: (grid_pct, leverage, line_count)
    """
    grid_pct, leverage, line_count, stop_loss_pct = params

    # 获取当前进程内存中的 df
    global worker_df
    df = worker_df

    print(f"开始回测 [进程ID: {os.getpid()}]，网格={grid_pct}, 杠杆={leverage} 当前时间: {datetime.now()}")

    # 构造文件名
    result_df_file = f'backtest/paired_trades_grid_{int(grid_pct * 1000)}_lev_{leverage}_{line_count}_{stop_loss_pct}.csv'
    minute_df_file = f'backtest/minute_stats_grid_{int(grid_pct * 1000)}_lev_{leverage}_{line_count}_{stop_loss_pct}.csv'

    # 检查文件是否存在
    if os.path.exists(result_df_file) and os.path.exists(minute_df_file):
        print(f"结果已存在，跳过: 网格={grid_pct}, 杠杆={leverage}")
        return

    start_time = datetime.now()

    try:
        result_df, minute_df = backtest_dynamic_grid(df, grid_pct=grid_pct, leverage=leverage, stop_loss_pct=stop_loss_pct)

        # 确保目录存在
        os.makedirs('backtest', exist_ok=True)

        result_df.to_csv(result_df_file, index=False)
        minute_df.to_csv(minute_df_file, index=False)

        print(f"回测完成 [进程ID: {os.getpid()}]，网格={grid_pct}, 杠杆={leverage}，耗时: {datetime.now() - start_time}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"回测出错: 网格={grid_pct}, 杠杆={leverage}, 错误: {e}")


if __name__ == '__main__':
    freeze_support()

    # 请根据实际情况修改路径
    file_path = r"W:\project\python_project\oke_auto_trade\kline_data\origin_data_1m_10000000_BTC-USDT-SWAP_2026-02-13.csv"
    line_count = 10000000

    print("正在读取数据...")
    if os.path.exists(file_path):
        main_df = pd.read_csv(file_path, nrows=line_count)
        print(f"数据读取完成，行数: {len(main_df)}")
    else:
        print("警告：文件路径不存在，请检查路径。")
        main_df = pd.DataFrame()  # 空防止报错

    grid_pct_list = [0.003, 0.005, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3]
    stop_loss_pct_list = [0.003, 0.005, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3]
    leverage_list = [1, 3, 5, 10, 20, 50, 100]

    # 1. 准备任务列表
    tasks = []
    for grid_pct in grid_pct_list:
        for leverage in leverage_list:
            for stop_loss_pct in stop_loss_pct_list:
                tasks.append((grid_pct, leverage, line_count, stop_loss_pct))

    print(f"总任务数: {len(tasks)}，准备启动 3 个进程...")

    if not main_df.empty:
        with Pool(processes=3, initializer=init_worker, initargs=(main_df,)) as pool:
            pool.map(run_single_backtest, tasks)

    print("所有回测任务完成。")