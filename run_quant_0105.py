import numpy as np
import matplotlib.pyplot as plt

def optimal_leverage(max_loss_rate, num_losses, max_profit_rate, num_profits, max_single_loss, max_single_profit, other_rate, other_count, L_min=1, num_L=500):
    """
    """
    max_loss_rate = max_loss_rate / 100
    max_profit_rate = max_profit_rate / 100
    max_single_loss = max_single_loss / 100
    # 计算单次交易的亏损率和盈利率
    r_loss = max_loss_rate / num_losses

    L_max = abs(1 / max_single_loss)  # 避免爆仓的最大杠杆

    # 计算最终收益函数
    def final_balance(L):
        after_loss = (1 + L * r_loss) ** num_losses
        after_loss = after_loss * (1 + L * max_single_loss)
        if after_loss <= 0:  # 避免爆仓
            return 0
        after_gain = after_loss * (1 + L * max_profit_rate)
        after_gain = after_gain * (1 + L * max_single_profit)
        other_gain = after_gain * (1 + L * other_rate)
        return other_gain

    # 计算不同杠杆下的最终收益
    L_values = np.unique(np.linspace(L_min, L_max, num_L).astype(int))

    balances = [final_balance(L) for L in L_values]

    # 找到最大收益对应的杠杆
    optimal_L = L_values[np.argmax(balances)]
    max_balance = max(balances)
    return optimal_L, max_balance

if __name__ == "__main__":
    # 测试示例
    optimal_L, max_balance = optimal_leverage(-10.9, 12, 84, 382, -2.84,10.31, 81, 397)
    print(f"最佳杠杆: {optimal_L:.2f}, 最大最终收益: {max_balance:.2f}")