import numpy as np
import matplotlib.pyplot as plt

def optimal_leverage(max_loss_rate, num_losses, max_profit_rate, num_profits, L_min=1, num_L=500):
    """
    计算最大化最终收益的最佳杠杆倍数，同时避免爆仓。

    参数：
    - max_loss_rate: 最大亏损率（负数，例如 -0.1 表示最大亏损 10%）
    - num_losses: 亏损的交易次数
    - max_profit_rate: 最大盈利率（正数，例如 0.3 表示最大盈利 30%）
    - num_profits: 盈利的交易次数
    - L_min: 杠杆的最小值（默认 1）
    - L_max: 杠杆的最大值（默认 50）
    - num_L: 计算杠杆的步数（默认 500）

    返回：
    - optimal_L: 最优杠杆倍数
    - max_balance: 该杠杆下的最大最终收益
    """
    # 计算单次交易的亏损率和盈利率
    r_loss = max_loss_rate / num_losses
    r_gain = max_profit_rate / num_profits

    L_max = abs(1 / max_loss_rate)  # 避免爆仓的最大杠杆

    # 计算最终收益函数
    def final_balance(L):
        after_loss = (1 + L * r_loss) ** num_losses
        after_loss = after_loss * (1 + L * max_loss_rate)
        if after_loss <= 0:  # 避免爆仓
            return 0
        after_gain = after_loss * (1 + L * r_gain) ** num_profits
        after_gain = after_gain * (1 + L * max_profit_rate)
        return after_gain

    # 计算不同杠杆下的最终收益
    L_values = np.linspace(L_min, L_max, num_L)
    balances = [final_balance(L) for L in L_values]

    # 找到最大收益对应的杠杆
    optimal_L = L_values[np.argmax(balances)]
    max_balance = max(balances)

    # 绘制杠杆 vs 最终收益的曲线
    plt.figure(figsize=(10, 5))
    plt.plot(L_values, balances, label="最终收益")
    plt.axvline(optimal_L, color='r', linestyle='--', label=f"最佳杠杆: {optimal_L:.2f}")
    plt.xlabel("杠杆倍数 L")
    plt.ylabel("最终账户余额")
    plt.title("杠杆 vs 最终收益")
    plt.legend()
    plt.grid()
    plt.show()

    return optimal_L, max_balance

if __name__ == "__main__":
    # 测试示例
    optimal_L, max_balance = optimal_leverage(-0.116, 48, 0.397, 243)
    print(f"最佳杠杆: {optimal_L:.2f}, 最大最终收益: {max_balance:.2f}")