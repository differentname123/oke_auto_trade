import time


def calculate_multi_group_margin(
        leverage: float,
        target_loss_percent: float,  # 全局最大忍受的下跌比例，例如 5.0 表示 5%
        max_grids_per_group: int,  # 新增：每个大组允许的最大网格数量
        fixed_qty: float = 1.0,  # 每次固定开仓的数量
        add_step_percent: float = 0.01,  # 每次下跌加仓的波动
        initial_price: float = 1.0  # 初始价格
) -> dict:
    """
    功能：在每次固定开仓数量的多组网格策略下，计算要扛住指定的跌幅，
    各组网格的价格区间、单组所需保证金，以及全局总计需要的初始保证金。
    不考虑手续费、MMR 和 滑点。
    """
    if leverage <= 0 or target_loss_percent <= 0 or add_step_percent <= 0 or fixed_qty <= 0 or max_grids_per_group <= 0:
        raise ValueError("所有参数必须 > 0")

    r = add_step_percent / 100.0
    # 计算全局目标止损价/爆仓价 (底线)
    target_price = initial_price * (1.0 - target_loss_percent / 100.0)

    current_price = initial_price

    total_margin_all_groups = 0.0
    groups_info = []
    group_id = 1

    # 只要当前价格还没跌穿全局目标价，就继续开启新的网格组
    while current_price >= target_price:
        group_start_price = current_price
        group_qty = 0.0
        group_cost = 0.0
        group_max_margin = 0.0
        grids_in_group = 0
        last_executed_price = current_price

        # 模拟单个大组内部顺着网格一路下跌加仓的过程
        while grids_in_group < max_grids_per_group and current_price >= target_price:
            # 1. 触发加仓：更新该组的持仓和成本
            group_qty += fixed_qty
            group_cost += fixed_qty * current_price
            grids_in_group += 1
            last_executed_price = current_price

            # 2. 检查加仓瞬间的“开仓保证金需求”
            upnl_at_open = group_qty * current_price - group_cost
            required_for_margin = (group_cost / leverage) - upnl_at_open
            if required_for_margin > group_max_margin:
                group_max_margin = required_for_margin

            # 计算下一个网格准备加仓的价格
            next_price = current_price * (1 - r)
            check_price = max(next_price, target_price)

            # 3. 检查在这个区间内下跌时的生存底线需求
            upnl_at_bottom = group_qty * check_price - group_cost
            required_for_survival = -upnl_at_bottom
            if required_for_survival > group_max_margin:
                group_max_margin = required_for_survival

            # 跌到下一个网格价，进入该组的下一次循环
            current_price = next_price

        # 4. 关键点：该组网格建仓完毕（或触及全局目标价）后，它需要一直扛跌到“全局 target_price”
        # 该组在全局目标价时的极限浮亏
        upnl_at_global_target = group_qty * target_price - group_cost

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
# 测试与使用示例
# ==========================================
if __name__ == "__main__":
    result = calculate_multi_group_margin(
        leverage=125.0,
        target_loss_percent=5,  # 扛住 5% 的下跌
        max_grids_per_group=169,  # 单个大组最多 150 个网格
        fixed_qty=0.01,  # 每次买 1 个
        add_step_percent=0.015,  # 每跌 0.01% 买一次
        initial_price=2120.0  # 假设初始价格 100
    )

    print(f"【全局总需准备的保证金】: {result['total_margin']}\n")
    print("【各网格大组详细信息】:")
    for g in result['groups_info']:
        print(f"组别 {g['group_id']}:")
        print(f"  - 价格区间: {g['start_price']} -> {g['end_price']}")
        print(f"  - 网格数量: {g['grid_count']} 单")
        print(f"  - 该组累计币量: {g['group_qty']}")
        print(f"  - 该组需分配保证金: {g['required_margin']}")
        print("-" * 30)
    # time.sleep(100)