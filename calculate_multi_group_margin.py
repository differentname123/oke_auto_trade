import time


def calculate_multi_group_margin(
        leverage: float,
        target_loss_percent: float,  # 全局最大忍受的下跌/上涨比例，例如 5.0 表示 5%
        max_grids_per_group: int,  # 每个大组允许的最大网格数量
        fixed_qty: float = 1.0,  # 每次固定开仓的数量
        add_step_percent: float = 0.01,  # 每次价格波动加仓的百分比
        initial_price: float = 1.0,  # 初始价格
        direction: str = 'long'  # 新增：'long' 表示做多，'short' 表示做空
) -> dict:
    """
    功能：在每次固定开仓数量的多组网格策略下，计算要扛住指定的跌幅/涨幅，
    各组网格的价格区间、单组所需保证金，以及全局总计需要的初始保证金。
    不考虑手续费、MMR 和 滑点。
    """
    if leverage <= 0 or target_loss_percent <= 0 or add_step_percent <= 0 or fixed_qty <= 0 or max_grids_per_group <= 0:
        raise ValueError("所有数值参数必须 > 0")
    if direction not in ['long', 'short']:
        raise ValueError("direction 参数必须是 'long' 或 'short'")

    r = add_step_percent / 100.0
    sign = 1 if direction == 'long' else -1  # 盈亏计算方向乘数

    # 计算全局目标止损价/爆仓价 (底线)
    if direction == 'long':
        target_price = initial_price * (1.0 - target_loss_percent / 100.0)
    else:
        target_price = initial_price * (1.0 + target_loss_percent / 100.0)

    current_price = initial_price

    total_margin_all_groups = 0.0
    groups_info = []
    group_id = 1

    # 定义价格是否尚未触及目标的检查函数
    def is_active(cp, tp):
        return cp >= tp if direction == 'long' else cp <= tp

    # 只要当前价格还没跌穿/涨穿全局目标价，就继续开启新的网格组
    while is_active(current_price, target_price):
        group_start_price = current_price
        group_qty = 0.0
        group_cost = 0.0
        group_max_margin = 0.0
        grids_in_group = 0
        last_executed_price = current_price

        # 模拟单个大组内部顺着网格一路下跌/上涨加仓的过程
        while grids_in_group < max_grids_per_group and is_active(current_price, target_price):
            # 1. 触发加仓：更新该组的持仓和成本
            group_qty += fixed_qty
            group_cost += fixed_qty * current_price
            grids_in_group += 1
            last_executed_price = current_price

            # 2. 检查加仓瞬间的“开仓保证金需求”
            # 利用 sign 自适应做多/做空的浮亏计算
            upnl_at_open = sign * (group_qty * current_price - group_cost)
            required_for_margin = (group_cost / leverage) - upnl_at_open
            if required_for_margin > group_max_margin:
                group_max_margin = required_for_margin

            # 计算下一个网格准备加仓的价格
            if direction == 'long':
                next_price = current_price * (1 - r)
                check_price = max(next_price, target_price)
            else:
                next_price = current_price * (1 + r)
                check_price = min(next_price, target_price)

            # 3. 检查在这个区间内波动时的生存底线需求
            upnl_at_bottom = sign * (group_qty * check_price - group_cost)
            required_for_survival = -upnl_at_bottom
            if required_for_survival > group_max_margin:
                group_max_margin = required_for_survival

            # 跌到/涨到下一个网格价，进入该组的下一次循环
            current_price = next_price

        # 4. 关键点：该组网格建仓完毕（或触及全局目标价）后，它需要一直扛单到“全局 target_price”
        # 该组在全局目标价时的极限浮亏
        upnl_at_global_target = sign * (group_qty * target_price - group_cost)

        # 维持该组仓位到全局目标价所需的极限保证金：(仓位价值 / 杠杆) - 极限浮亏
        required_at_global_target = (group_cost / leverage) - upnl_at_global_target
        if required_at_global_target > group_max_margin:
            group_max_margin = required_at_global_target

        # 记录该大组的信息
        groups_info.append({
            "group_id": group_id,
            "start_price": round(group_start_price, 6),
            "end_price": round(last_executed_price, 6),  # 该组最后一单的成交价
            "grid_count": grids_in_group,
            "group_qty": round(group_qty, 6),
            "required_margin": round(group_max_margin, 6)
        })

        # 累加到总保证金中
        total_margin_all_groups += group_max_margin
        group_id += 1

    return {
        "total_margin": round(total_margin_all_groups, 6),
        "groups_info": groups_info
    }


# ==========================================
# DCA 合约加仓计算函数（基于策略全额保证金核算，加入中途爆仓截断）
# ==========================================
def calculate_dca_info(
        price_deviation_percent: float,  # 价格偏差 (%)
        leverage: float,  # 杠杆
        initial_margin: float,  # 初始订单保证金
        dca_margin_base: float,  # 加仓单基础保证金
        max_dca_orders: int,  # 最大DCA订单数量
        tp_target_percent: float,  # 每轮止盈目标 (%)
        price_step_multiplier: float = 1.0,  # 加仓单价差乘数
        amount_multiplier: float = 1.1,  # 加仓金额乘数
        direction: str = 'short',  # 方向: 'long' 或 'short'
        initial_price: float = 1.0,  # 初始开仓价格（用于计算具体价位）
        extra_margin: float = 20.0  # 新增：额外保证金，默认为20
) -> dict:
    if direction not in ['long', 'short']:
        raise ValueError("direction 必须是 'long' 或 'short'")

    # 1. 预计算该策略锁定的总保证金 (护城河：包含初始、所有DCA预算，以及额外保证金)
    total_allocated_margin = initial_margin + extra_margin
    for i in range(1, max_dca_orders + 1):
        total_allocated_margin += dca_margin_base * (amount_multiplier ** (i - 1))

    total_notional = 0.0
    total_size = 0.0
    rounds_info = []

    cumulative_dev_percent = 0.0
    current_gap_percent = price_deviation_percent

    expected_rounds = max_dca_orders + 1
    actual_rounds = 0

    cumulative_margin_invested = 0.0  # 记录每一轮实际累计投入的订单保证金

    # 2. 依次计算每一轮的挂单与成交详情
    for i in range(expected_rounds):
        if i == 0:
            round_name = "初始订单"
            order_price = initial_price
            order_margin = initial_margin
        else:
            round_name = f"DCA #{i}"
            cumulative_dev_percent += current_gap_percent
            if direction == 'long':
                order_price = initial_price * (1 - cumulative_dev_percent / 100.0)
            else:
                order_price = initial_price * (1 + cumulative_dev_percent / 100.0)

            order_margin = dca_margin_base * (amount_multiplier ** (i - 1))

        # 检查在触发本轮挂单前，是否已经跌破/涨破了上一轮的爆仓价
        if i > 0:
            prev_liq_price = rounds_info[-1]["round_liq_price"]
            if direction == 'long' and order_price <= prev_liq_price:
                break  # 做多时，跌破了上一轮爆仓价，截断
            elif direction == 'short' and order_price >= prev_liq_price:
                break  # 做空时，涨破了上一轮爆仓价，截断

        # 只有存活下来，才更新下一次的加仓间距
        if i > 0:
            current_gap_percent *= price_step_multiplier

        # 更新仓位价值、币量、及当前累计投入的真实保证金
        cumulative_margin_invested += order_margin
        order_notional = order_margin * leverage
        order_size = order_notional / order_price

        total_notional += order_notional
        total_size += order_size
        avg_price = total_notional / total_size

        # 计算止盈目标
        if direction == 'long':
            tp_price = avg_price * (1 + tp_target_percent / 100.0)
        else:
            tp_price = avg_price * (1 - tp_target_percent / 100.0)

        tp_trigger_change = (tp_price - order_price) / order_price * 100

        # 计算本轮阶段的爆仓价（使用 策略总保证金 / 当前持仓币量）
        if direction == 'long':
            round_liq_price = avg_price - (total_allocated_margin / total_size)
        else:
            round_liq_price = avg_price + (total_allocated_margin / total_size)

        # 计算当前阶段爆仓价和均价的偏差比例绝对值
        liq_deviation_from_avg = abs(round_liq_price - avg_price) / avg_price * 100

        # 新增：计算当前价位下，至少需要的保证金（如果策略总资金少于这个值，在该价位就会爆仓）
        # 基于 0-MMR 模型，该值精确等于到达此执行价时产生的绝对浮亏
        min_survive_margin = total_size * abs(avg_price - order_price)

        rounds_info.append({
            "round_name": round_name,
            "order_price": order_price,
            "deviation_from_initial": cumulative_dev_percent if i > 0 else 0.0,
            "order_margin": order_margin,
            "cumulative_margin": cumulative_margin_invested,
            "avg_price": avg_price,
            "tp_price": tp_price,
            "tp_trigger_change": tp_trigger_change,
            "round_liq_price": round_liq_price,
            "liq_deviation_from_avg": liq_deviation_from_avg,
            "min_survive_margin": min_survive_margin  # 新增的输出字段
        })

        actual_rounds += 1

    # 3. 最终汇总（基于最后实际存活的轮次）
    final_avg_price = rounds_info[-1]["avg_price"]
    final_liq_price = rounds_info[-1]["round_liq_price"]
    max_tolerable_drop = abs(final_liq_price - initial_price) / initial_price * 100

    return {
        "expected_total_rounds": expected_rounds,
        "actual_rounds": actual_rounds,
        "total_margin_invested": total_allocated_margin,
        "final_avg_price": final_avg_price,
        "final_liquidation_price": final_liq_price,
        "max_tolerable_fluctuation_percent": max_tolerable_drop,
        "rounds_info": rounds_info
    }

def generate_sequence_from_sum(target, n, ratio, precision=6):
    if n <= 0:
        raise ValueError("次数 n 必须大于 0")

    # 求初始值
    if ratio == 1:
        a1 = target / n
    else:
        a1 = target * (1 - ratio) / (1 - ratio ** n)

    print(f"初始值 a1 = {round(a1, precision)}\n")

    sequence = []
    total = 0

    for i in range(n):
        value = a1 * (ratio ** i)
        total += value
        sequence.append(value)

        print(f"第{i+1}轮: {round(value, precision)}，累计: {round(total, precision)}")

    print(f"\n最终累计: {round(total, precision)} (目标: {target})")

    return sequence

# ==========================================
# 测试与使用示例
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("【合约DCA模式计算测试 (新增累计保证金与爆仓偏差测算)】")
    print("=" * 50)

    print(generate_sequence_from_sum(150, 3, 2))


    # result = calculate_multi_group_margin(
    #     leverage=125.0,
    #     target_loss_percent=20,  # 扛住 5% 的下跌
    #     max_grids_per_group=10000,  # 单个大组最多 150 个网格
    #     fixed_qty=0.017,  # 每次买 1 个
    #     add_step_percent=0.4,  # 每跌 0.01% 买一次
    #     initial_price=2300,  # 假设初始价格 100
    #     # direction='short',  # 做空
    # )
    #
    # print(f"【全局总需准备的保证金】: {result['total_margin']}\n")
    # print("【各网格大组详细信息】:")
    # for g in result['groups_info']:
    #     print(f"组别 {g['group_id']}:")
    #     print(f"  - 价格区间: {g['start_price']} -> {g['end_price']}")
    #     print(f"  - 网格数量: {g['grid_count']} 单")
    #     print(f"  - 该组累计币量: {g['group_qty']}")
    #     print(f"  - 该组需分配保证金: {g['required_margin']}")
    #     print("-" * 30)


    dca_result = calculate_dca_info(
        price_deviation_percent=22.0, # 价格偏差 (%)
        leverage=25.0,  # 杠杆
        initial_margin=0.8,  # 初始订单保证金
        dca_margin_base=0.72, # 加仓单基础保证金
        max_dca_orders=3,# 最大DCA订单数量
        tp_target_percent=15.0,  # 每轮止盈目标 (%)
        price_step_multiplier=2,# 加仓单价差乘数
        amount_multiplier=2,  # 加仓金额乘数
        direction='short',# 方向: 'long' 或 'short'
        initial_price=2.188, # 初始开仓价格（用于计算具体价位）
        extra_margin=1000      # 新增：额外保证金，默认为20
    )

    for r in dca_result['rounds_info']:
        print(f"[{r['round_name']}]")
        print(f"  订单价格 / 距初始单价格: {r['order_price']:.4f} / {r['deviation_from_initial']:.2f}%")
        print(f"  当轮委托保证金: {r['order_margin']:.2f}")
        print(f"  当前累计投入保证金: {r['cumulative_margin']:.2f}")
        print(f"  min_survive_margin: {r['min_survive_margin']:.2f}")

        print(f"  平均价格: {r['avg_price']:.4f}")
        print(f"  止盈价格: {r['tp_price']:.4f}")
        print(f"  触发止盈所需价格变动: {r['tp_trigger_change']:.2f}%")
        print(f"  当前阶段爆仓价(基于策略总资金池): {r['round_liq_price']:.4f}")
        print(f"  爆仓价距当前均价偏差: {r['liq_deviation_from_avg']:.2f}%")
        print("-" * 30)

    print("\n【最终风险与汇总评估】")
    print(f"预计执行轮次(含初始单): {dca_result['expected_total_rounds']} 轮")
    print(f"实际存活轮次(含初始单): {dca_result['actual_rounds']} 轮")

    if dca_result['actual_rounds'] < dca_result['expected_total_rounds']:
        print("⚠️ 警告：策略中途爆仓，后续加仓单无法被执行！")

    print(f"策略锁定总保证金池: {dca_result['total_margin_invested']:.4f} USDT")
    print(f"实际最终均价: {dca_result['final_avg_price']:.4f}")
    print(f"实际极限强平价格: {dca_result['final_liquidation_price']:.4f}")
    print(f"最大可忍受单边波动(距初始价): {dca_result['max_tolerable_fluctuation_percent']:.2f}%")
    print()