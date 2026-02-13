from datetime import datetime

import pandas as pd


# ==========================================
# 模块 1：独立的价格计算函数 (信号发生器)
# ==========================================
def get_minute_triggers(row, last_close, initial_base, current_level, grid_pct):
    """
    计算单根 K 线内触发的所有网格交易价格点。
    【修改核心】：引入 initial_base 和 current_level 替代原有的动态 base_price。
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
            # 【核心修改】：通过固定初始基准价与整数网格层级进行计算，彻底消灭基准价连乘带来的网格偏移(Drift)
            up_line = initial_base * (1 + (current_level + 1) * grid_pct)
            down_line = initial_base * (1 + (current_level - 1) * grid_pct)

            # 向上突破
            if next_p > curr_p and curr_p <= up_line <= next_p:
                exec_price = next_p if is_gap else up_line
                current_level += 1
                # 【修复1】：将该段轨迹的物理极值点 next_p 传入 trigger，提供给主循环进行绝对界限判断
                # 【新增修改】：将当前网格层级 level 传入，确保主循环开单的目标价绝对锚定初始网格体系
                triggers.append({'time': row.timestamp, 'price': exec_price, 'type': 'UP', 'next_p': next_p, 'level': current_level})
                curr_p = up_line

            # 向下突破
            elif next_p < curr_p and curr_p >= down_line >= next_p:
                exec_price = next_p if is_gap else down_line
                current_level -= 1
                # 【修复1】：同上，传入物理极值点 next_p
                # 【新增修改】：同上，传入 level
                triggers.append({'time': row.timestamp, 'price': exec_price, 'type': 'DOWN', 'next_p': next_p, 'level': current_level})
                curr_p = down_line
            else:
                curr_p = next_p

    return triggers, current_level


# ==========================================
# 主回测程序 (模块 2 & 模块 3 的整合)
# ==========================================
def backtest_dynamic_grid(df, grid_pct=0.005, margin=10000.0, leverage=10.0, fee_rate=0.001):
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 【核心修改】：使用固定的初始基准价与网格层级跟踪，代替原先会产生漂移的游走 base_price
    initial_base = df['open'].iloc[0]
    current_level = 0
    last_close = initial_base

    # 订单列表与记录表 (取消栈的概念，退化为普通列表)
    longs_stack = []  # 待平多单
    shorts_stack = []  # 待平空单
    paired_trades = []  # 模块2：配对交易记录
    minute_stats = []  # 模块3：每分钟资产记录

    cumulative_profit = 0.0  # 新增：累计利润池

    notional = margin * leverage  # 单次交易名义价值

    for row in df.itertuples():
        # --- 步骤 A：获取当前分钟的所有触发信号 ---
        # 【修改】：传入并接收 initial_base 和 current_level
        triggers, current_level = get_minute_triggers(row, last_close, initial_base, current_level, grid_pct)

        # --- 步骤 B：模块 2 - 持仓维护与交易配对 ---
        for t in triggers:
            exec_price = t['price']
            exec_time = t['time']
            trigger_level = t['level']  # 【修改】：提取层级

            if t['type'] == 'UP':
                # 涨了：优先平多。【核心修改】：取消栈顶限制，遍历列表寻找任何达到目标价的独立多单
                closed_orders = []
                i = 0
                while i < len(longs_stack):
                    # 【修复1 & 修复2】：只要触发价或该段物理最高点(next_p)达到了 target_price，全部抽出平仓
                    if exec_price >= longs_stack[i]['target_price'] - 1e-8:
                        closed_orders.append(longs_stack.pop(i))
                        break
                    else:
                        i += 1

                if closed_orders:
                    # 将本次满足条件的所有单子全部平掉
                    for closed_order in closed_orders:
                        order = closed_order
                        qty = order['qty']

                        # 真实平仓价：如果是因为 next_p 摸到而平仓，按 target_price 算，确保不吃亏
                        actual_close_price = max(exec_price, order['target_price'])

                        gross_pnl = (actual_close_price - order['price']) * qty
                        close_fee = (qty * actual_close_price) * fee_rate
                        net_pnl = gross_pnl - order['fee'] - close_fee
                        holding_time = datetime.strptime(exec_time, '%Y-%m-%d %H:%M:%S') - datetime.strptime(order['time'], '%Y-%m-%d %H:%M:%S')

                        cumulative_profit += net_pnl

                        paired_trades.append({
                            '开仓时间': order['time'],
                            '平仓时间': exec_time,
                            '持仓时间': holding_time,
                            '目标价格': order['target_price'],
                            '方向': '做多',
                            '开仓价': order['price'],
                            '平仓价': actual_close_price,
                            '净盈亏': net_pnl
                        })
                else:
                    qty = notional / exec_price
                    shorts_stack.append({
                        'time': exec_time,
                        'price': exec_price,
                        'qty': qty,
                        'fee': notional * fee_rate,
                        # 【核心修改】：目标平仓价必须严格与固定的对应网格线对齐 (目标为平推到上一层网格)
                        'target_price': initial_base * (1 + (trigger_level - 1) * grid_pct)
                    })

            elif t['type'] == 'DOWN':
                # 跌了：优先平空。【核心修改】：取消栈顶限制，遍历列表寻找任何达到目标价的独立空单
                closed_orders = []
                i = 0
                while i < len(shorts_stack):
                    # 【修复1 & 修复2】：只要触发价或该段物理最低点(next_p)达到了 target_price，全部抽出平仓
                    if exec_price <= shorts_stack[i]['target_price'] + 1e-8:
                        closed_orders.append(shorts_stack.pop(i))
                        break
                    else:
                        i += 1

                if closed_orders:
                    # 将本次满足条件的所有单子全部平掉
                    for closed_order in closed_orders:
                        order = closed_order
                        qty = order['qty']

                        # 真实平仓价：取 target_price 和 exec_price 间的最小值，确保不吃亏
                        actual_close_price = min(exec_price, order['target_price'])

                        gross_pnl = (order['price'] - actual_close_price) * qty
                        close_fee = (qty * actual_close_price) * fee_rate
                        net_pnl = gross_pnl - order['fee'] - close_fee
                        holding_time = datetime.strptime(exec_time, '%Y-%m-%d %H:%M:%S') - datetime.strptime(order['time'], '%Y-%m-%d %H:%M:%S')

                        cumulative_profit += net_pnl

                        paired_trades.append({
                            '开仓时间': order['time'],
                            '平仓时间': exec_time,
                            '持仓时间': holding_time,
                            '方向': '做空',
                            '目标价格': order['target_price'],
                            '开仓价': order['price'],
                            '平仓价': actual_close_price,
                            '净盈亏': net_pnl
                        })
                else:
                    qty = notional / exec_price
                    longs_stack.append({
                        'time': exec_time,
                        'price': exec_price,
                        'qty': qty,
                        'fee': notional * fee_rate,
                        # 【核心修改】：目标平仓价必须严格与固定的对应网格线对齐 (目标为反弹到下一层网格)
                        'target_price': initial_base * (1 + (trigger_level + 1) * grid_pct)
                    })

        # --- 步骤 C：模块 3 - 每分钟最低资产值核算 ---
        # 1. 资金投入
        current_orders = len(longs_stack) + len(shorts_stack)
        invested_capital = current_orders * margin
        if len(longs_stack) > 0 and len(shorts_stack) > 0:
            print(f"警告：同一时间存在多单和空单，当前多单数={len(longs_stack)}，空单数={len(shorts_stack)}，请检查逻辑！")


        # 2. 持仓方向
        if longs_stack:
            direction = '多单'
        elif shorts_stack:
            direction = '空单'
        else:
            direction = '空仓'

        # 3. 计算本分钟的最低资产值 (极端压力测试)
        min_asset_value = invested_capital

        if longs_stack:
            worst_price = row.low
            for order in longs_stack:
                unrealized = (worst_price - order['price']) * order['qty'] - order['fee']
                min_asset_value += unrealized
        elif shorts_stack:
            worst_price = row.high
            for order in shorts_stack:
                unrealized = (order['price'] - worst_price) * order['qty'] - order['fee']
                min_asset_value += unrealized

        minute_stats.append({
            '时间': row.timestamp,
            '收盘价': row.close,
            '持仓方向': direction,
            '持仓单数': current_orders,
            '投入资金': invested_capital,
            '本分钟最低资产值': min_asset_value,
            '资产比例': min_asset_value / (invested_capital) if invested_capital > 0 else 1,
            '累计利润': cumulative_profit
        })

        last_close = row.close

    return pd.DataFrame(paired_trades), pd.DataFrame(minute_stats)


if __name__ == '__main__':
    file_path = r"W:\project\python_project\oke_auto_trade\kline_data\origin_data_1m_10000000_BTC-USDT-SWAP_2026-02-13.csv"
    df = pd.read_csv(file_path, nrows=1000000)
    # result_df = pd.read_csv('paired_trades.csv')
    # minute_df = pd.read_csv('minute_stats.csv')
    result_df, minute_df = backtest_dynamic_grid(df)
    result_df.to_csv('paired_trades.csv', index=False)
    minute_df.to_csv('minute_stats.csv', index=False)
    print()