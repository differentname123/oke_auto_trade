import os
import math
from common_utils import get_config
import okx.Trade as Trade
import okx.Account as Account
import okx.MarketData as Market


# 设置代理（如果需要）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'



flag = "1"  # 实盘: 0, 模拟盘: 1

if flag == "1":
    # API 初始化
    apikey = get_config('api_key')
    secretkey = get_config('secret_key')
    passphrase = get_config('passphrase')
else:
    # API 初始化
    apikey = get_config('true_api_key')
    secretkey = get_config('true_secret_key')
    passphrase = get_config('true_passphrase')

# 初始化 OKX API 模块
tradeAPI = Trade.TradeAPI(apikey, secretkey, passphrase, False, flag)
accountAPI = Account.AccountAPI(apikey, secretkey, passphrase, False, flag)
marketAPI = Market.MarketAPI(apikey, secretkey, passphrase, False, flag)

# 全局配置
INST_ID = "BTC-USDT-SWAP"  # 交易对
LEVERAGE = 100             # 杠杆倍数
POSITION_RATIO = 0.5       # 仓位比例（如 50%）
TAKE_PROFIT_PERCENTAGE = 0.002  # 止盈百分比（1%）

def get_1h_high_low(inst_id):
    """获取过去 1 小时内的最高价和最低价"""
    try:
        candles = marketAPI.get_candlesticks(inst_id, bar="1H", limit=1)
        if candles and len(candles['data']) > 0:
            highest_price = float(candles['data'][0][2])  # 最高价
            lowest_price = float(candles['data'][0][3])   # 最低价
            return highest_price, lowest_price
    except Exception as e:
        print("获取 1 小时最高价和最低价失败：", e)
    return None, None

def get_account_equity():
    """获取账户 USDT 余额"""
    try:
        account_info = accountAPI.get_account_balance()
        for data in account_info['data']:
            for detail in data['details']:
                if detail['ccy'] == 'USDT':
                    return float(detail['eq'])  # 返回账户权益
    except Exception as e:
        print("获取账户权益失败：", e)
    return 0.0

def get_available_usdt_balance():
    """
    获取账户实际可用的 USDT 余额，包括冻结资金和持仓保证金的考虑。
    """
    try:
        # 1. 获取账户总权益
        account_info = accountAPI.get_account_balance()
        total_usdt_balance = 0.0

        # 遍历账户详情，找到 USDT 的总权益
        for data in account_info['data']:
            for detail in data['details']:
                if detail['ccy'] == 'USDT':
                    total_usdt_balance = float(detail['eq'])  # 总权益（包含冻结资金）

        # 2. 获取未完成的挂单并计算冻结的 USDT
        open_orders = tradeAPI.get_order_list(instType="SWAP", state="live")  # 查询未完成订单
        frozen_usdt_from_orders = 0.0

        if open_orders and 'data' in open_orders:
            for order in open_orders['data']:
                if order['instType'] == "SWAP" and order['state'] == "live":
                    px = float(order['px']) if order['px'] else 0.0  # 限价单价格
                    sz = float(order['sz']) if order['sz'] else 0.0  # 订单数量
                    frozen_usdt_from_orders += px * sz * 0.0001  # 冻结资金 = 价格 × 数量

        # 3. 获取当前持仓信息并计算持仓保证金
        positions = accountAPI.get_positions(instType="SWAP")  # 查询所有持仓
        frozen_usdt_from_positions = 0.0

        if positions and 'data' in positions:
            for position in positions['data']:
                if position['instType'] == "SWAP":
                    margin = float(position['imr']) if position['imr'] else 0.0  # 持仓所需保证金
                    frozen_usdt_from_positions += margin  # 累加所有持仓的保证金

        # 4. 计算实际可用余额
        available_usdt_balance = total_usdt_balance - frozen_usdt_from_orders - frozen_usdt_from_positions
        return max(available_usdt_balance, 0.0)  # 确保余额非负

    except Exception as e:
        print("获取账户可用余额失败：", e)
        return 0.0

def calculate_order_size(usdt_equity, entry_price, leverage, position_ratio):
    """计算下单量（合约数量），并限制单笔订单的最大价值为 5000000 USD"""
    # 计算总持仓金额
    position_amount = usdt_equity * position_ratio * leverage * 0.99  # 持仓金额
    # 每张合约的价值（假设每张合约价值为 0.01）
    contract_value_per_lot = entry_price * 0.01

    # 确保单笔订单总价值不超过 5000000 USD
    max_allowed_position = 5000000 * 0.99  # 单笔订单的最大价值
    capped_position_amount = min(position_amount, max_allowed_position)

    # 计算合约数量（向下取整）
    order_size = capped_position_amount / contract_value_per_lot
    return math.floor(order_size)

def place_order(inst_id, side, order_type, size, price=None, tp_price=None):
    """下单函数"""
    try:
        pos_side = "long" if side == "buy" else "short"

        # 构建下单参数
        order_params = {
            "instId": inst_id,
            "tdMode": "cross",  # 全仓模式
            "side": side,       # 买入或卖出
            "ordType": order_type,  # 订单类型：limit 或 market
            "sz": str(size),    # 下单量
            "posSide": pos_side  # 仓位方向：多头或空头
        }

        # 如果是限价单，添加价格参数
        if order_type == "limit" and price:
            order_params["px"] = str(price)

        # 如果需要止盈参数，添加止盈触发价格和订单价格
        if tp_price:
            order_params["tpTriggerPx"] = str(tp_price)
            order_params["tpOrdPx"] = str(tp_price)

        # 调用下单接口
        order = tradeAPI.place_order(**order_params)
        print(f"{side.upper()} 订单成功：", order)
        return order
    except Exception as e:
        print(f"{side.upper()} 订单失败，错误信息：", e)
        return None


if __name__ == "__main__":
    # 获取账户权益
    usdt_equity = get_available_usdt_balance()
    if usdt_equity <= 0:
        print("账户权益不足，无法下单。")
        exit()

    # 获取 1 小时内的最高价和最低价
    highest_price, lowest_price = get_1h_high_low(INST_ID)
    if not highest_price or not lowest_price:
        print("无法获取 1 小时内的最高价和最低价，退出程序。")
        exit()

    # 计算下单量
    short_size = calculate_order_size(usdt_equity, highest_price, LEVERAGE, POSITION_RATIO)
    long_size = calculate_order_size(usdt_equity, lowest_price, LEVERAGE, POSITION_RATIO)

    # # 测试：设置固定下单量
    # short_size = 0.1
    # long_size = 0.1
    avg_price = (highest_price + lowest_price) / 2
    # 计算止盈价格
    take_profit_price_short = min(highest_price * (1 - TAKE_PROFIT_PERCENTAGE), avg_price)  # 做空止盈价
    take_profit_price_long = max(lowest_price * (1 + TAKE_PROFIT_PERCENTAGE), avg_price)   # 做多止盈价

    print("最高价：", highest_price, "最低价：", lowest_price, "做空下单量：", short_size, "做多下单量：", long_size)

    # 做空限价单
    place_order(
        INST_ID,
        "sell",
        "limit",  # 限价单
        short_size,
        price=highest_price,  # 最高价作为下单价格
        tp_price=take_profit_price_short  # 止盈价格
    )

    # 做多限价单
    place_order(
        INST_ID,
        "buy",
        "limit",  # 限价单
        long_size,
        price=lowest_price,  # 最低价作为下单价格
        tp_price=take_profit_price_long  # 止盈价格
    )