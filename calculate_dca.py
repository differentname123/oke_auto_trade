def calculate_dca_info(
        price_deviation_percent: float,  # 价格偏差 (%)
        leverage: float,  # 杠杆倍数
        initial_margin: float,  # 初始订单保证金
        dca_margin_base: float,  # 加仓单基础保证金
        max_dca_orders: int,  # 最大DCA订单数量
        tp_target_percent: float,  # 每轮止盈目标 (%)
        price_step_multiplier: float = 1.0,  # 加仓单价差乘数
        amount_multiplier: float = 1.1,  # 加仓金额乘数
        direction: str = 'short',  # 方向: 'long' 或 'short'
        initial_price: float = 1.0,  # 初始开仓价格
        extra_margin: float = 20.0  # 额外保证金
) -> dict:
    if direction not in ['long', 'short']:
        raise ValueError("direction 必须是 'long' 或 'short'")

    # 1. 预计算策略锁定的总保证金 (包含初始单、所有DCA预算及额外保证金)
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
    cumulative_margin_invested = 0.0

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

        # 检查触发本轮挂单前，是否已突破上一轮的爆仓价
        if i > 0:
            prev_liq_price = rounds_info[-1]["round_liq_price"]
            if direction == 'long' and order_price <= prev_liq_price:
                break
            elif direction == 'short' and order_price >= prev_liq_price:
                break

                # 只有未爆仓，才更新下一次的加仓间距
        if i > 0:
            current_gap_percent *= price_step_multiplier

        # 更新仓位价值、币量及当前累计投入的真实保证金
        cumulative_margin_invested += order_margin
        order_notional = order_margin * leverage
        order_size = order_notional / order_price

        total_notional += order_notional
        total_size += order_size
        avg_price = total_notional / total_size

        # 计算止盈目标价
        if direction == 'long':
            tp_price = avg_price * (1 + tp_target_percent / 100.0)
        else:
            tp_price = avg_price * (1 - tp_target_percent / 100.0)

        tp_trigger_change = (tp_price - order_price) / order_price * 100

        # 计算本轮阶段的爆仓价（基于策略总资金 / 当前持仓币量）
        if direction == 'long':
            round_liq_price = avg_price - (total_allocated_margin / total_size)
        else:
            round_liq_price = avg_price + (total_allocated_margin / total_size)

        liq_deviation_from_avg = abs(round_liq_price - avg_price) / avg_price * 100

        # 计算当前价位下生存所需的最低保证金
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
            "min_survive_margin": min_survive_margin
        })

        actual_rounds += 1

    # 3. 最终汇总评估（基于最后实际存活轮次）
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


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("【合约DCA模式计算测试】")
    print("=" * 50)

    dca_result = calculate_dca_info(
        price_deviation_percent=10.0,
        leverage=25.0,
        initial_margin=1,
        dca_margin_base=1,
        max_dca_orders=3,
        tp_target_percent=15.0,
        price_step_multiplier=2,
        amount_multiplier=2,
        direction='short',
        initial_price=4629,
        extra_margin=1000
    )

    for r in dca_result['rounds_info']:
        print(f"[{r['round_name']}]")
        print(f"  订单价格 / 距初始单价格: {r['order_price']:.4f} / {r['deviation_from_initial']:.2f}%")
        print(f"  当轮委托保证金: {r['order_margin']:.2f}")
        print(f"  当前累计投入保证金: {r['cumulative_margin']:.2f}")
        print(f"  最少生存保证金: {r['min_survive_margin']:.2f}")
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