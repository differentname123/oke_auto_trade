import os
import okx.Trade as Trade
import okx.MarketData as Market
import okx.Account as Account
from common_utils import get_config
import time

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
marketAPI = Market.MarketAPI(apikey, secretkey, passphrase, False, flag)
accountAPI = Account.AccountAPI(apikey, secretkey, passphrase, False, flag)

# 全局配置
INST_ID = "BTC-USDT-SWAP"  # 交易对
ORDER_SIZE = 1            # 每次固定下单量

def get_latest_price(inst_id):
    """获取最新价格"""
    try:
        ticker = marketAPI.get_ticker(inst_id)
        if ticker and 'data' in ticker and len(ticker['data']) > 0:
            latest_price = float(ticker['data'][0]['last'])  # 最新成交价
            return latest_price
    except Exception as e:
        print("获取最新价格失败：", e)
    return None

def get_position(inst_id):
    """获取当前持仓信息"""
    try:
        positions = accountAPI.get_positions(instType="SWAP")
        if positions and 'data' in positions:
            for position in positions['data']:
                if position['instId'] == inst_id:
                    pos_side = position['posSide']  # 仓位方向：long 或 short
                    avg_px = float(position['avgPx'])  # 持仓均价
                    pos_size = float(position['pos'])  # 持仓数量
                    return pos_side, avg_px, pos_size
    except Exception as e:
        print("获取持仓信息失败：", e)
    return None, None, 0

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
    offset = 25  # 下单价格的偏移量
    profit = 25
    count = 1000000
    while True:
        count -= 1
        if count < 0:
            break
        # 获取最新价格
        latest_price = get_latest_price(INST_ID)
        if not latest_price:
            print("无法获取最新价格，退出程序。")
            exit()

        # 计算止盈价格
        take_profit_price_long = latest_price + profit  # 多头止盈价格为买入价格 + 10
        take_profit_price_short = latest_price - profit  # 空头止盈价格为卖出价格 - 10

        # 做多限价单
        place_order(
            INST_ID,
            "buy",
            "limit",  # 限价单
            ORDER_SIZE,
            price=latest_price - offset,  # 买入价格
            tp_price=take_profit_price_long  # 止盈价格
        )

        # 做空限价单
        place_order(
            INST_ID,
            "sell",
            "limit",  # 限价单
            ORDER_SIZE,
            price=latest_price + offset,  # 卖出价格
            tp_price=take_profit_price_short  # 止盈价格
        )
        time.sleep(1)