import datetime
import os
import time
import logging

import numpy as np
import okx.Trade as Trade
import okx.MarketData as Market
import okx.Account as Account
from common_utils import get_config

"""
当前策略说明:
1.负偏差价格下单，基本上一下单就会秒成，增加下单的成功率（相应的盈利偏差就增加了 整体盈利差保持在90）
2.多空差异大时会减小多的那个方向的买入价格（认为下降空间很大或者不想继续增加这个方向的仓位了）。会增加小的那个方向的止盈利润（未改变买入价 表示还是希望能够买入增加持仓量。同时增加止盈是认为如果继续往反方向变化的话能够盈利多一点 减小损失）
3.价格相较于上一次的触发价格相差10才进行新的下单（避免单子都分布在一个价格区间）
"""

# 日志配置
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 设置代理（如果需要）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# 全局延时配置
DELAY_SHORT = 1  # 短延时（秒）

total_profit = 90
OFFSET = -45
PROFIT = total_profit - OFFSET
# 配置区域
CONFIG = {
    "INST_ID": "BTC-USDT-SWAP",  # 交易对
    "ORDER_SIZE": 10,  # 每次固定下单量
    "PRICE_THRESHOLD": 10,  # 最新价格变化的阈值
    "OFFSET": OFFSET,  # 下单价格偏移量（基础值）
    "PROFIT": PROFIT  # 止盈偏移量（基础值）
}

MAX_POSITION_RATIO = 1  # 最大持仓比例为0.9
flag = "1"  # 实盘: 0, 模拟盘: 1

# API 初始化
if flag == "1":
    apikey = get_config('api_key')
    secretkey = get_config('secret_key')
    passphrase = get_config('passphrase')
else:
    apikey = get_config('true_api_key')
    secretkey = get_config('true_secret_key')
    passphrase = get_config('true_passphrase')

tradeAPI = Trade.TradeAPI(apikey, secretkey, passphrase, False, flag)
marketAPI = Market.MarketAPI(apikey, secretkey, passphrase, False, flag)
accountAPI = Account.AccountAPI(apikey, secretkey, passphrase, False, flag)


# 通用工具函数
def safe_api_call(api_function, *args, **kwargs):
    """安全调用 API，捕获异常"""
    try:
        return api_function(*args, **kwargs)
    except Exception as e:
        logger.error(f"调用 {api_function.__name__} 失败：{e}")
        return None


def get_open_orders():
    """获取未完成的订单"""
    return safe_api_call(tradeAPI.get_order_list, instType="SWAP")


def cancel_order(inst_id, order_id):
    """撤销订单"""
    return safe_api_call(tradeAPI.cancel_order, instId=inst_id, ordId=order_id)


def get_latest_price(inst_id):
    """获取最新价格"""
    ticker = safe_api_call(marketAPI.get_ticker, inst_id)
    if ticker and 'data' in ticker and len(ticker['data']) > 0:
        return float(ticker['data'][0]['last'])  # 最新成交价
    return None


def get_positions(inst_id):
    """获取合约仓位信息"""
    pos_info = safe_api_call(accountAPI.get_positions, instType="SWAP", instId=inst_id)
    if pos_info and 'data' in pos_info:
        return pos_info['data']
    return []

def get_account_equity():
    """获取账户净资产（权益）"""
    account_info = safe_api_call(accountAPI.get_account_balance)
    if account_info and 'data' in account_info:
        # 示例：从data中找到总权益（各币种相加或使用USDT权益）
        # 这里假设所有权益都在 USDT 或统一计价下
        # 请根据实际返回结构自行修改解析逻辑
        total_equity = 0.0
        for item in account_info['data'][0]['details']:
            if item['ccy'] == 'USDT':
                total_equity += float(item['eq'])
        return total_equity
    return 0.0

def get_position_ratio(inst_id, latest_price):
    """计算多头和空头方向的仓位价值比例"""
    positions = get_positions(inst_id)
    total_equity = get_account_equity()
    total_equity *= 10000
    if total_equity == 0:
        return 0, 0

    long_value = 0.0
    short_value = 0.0

    for pos in positions:
        # posSide: long 或 short
        if pos['posSide'] == 'long':
            long_sz = float(pos['pos'])
            long_value += long_sz * latest_price  # 仓位价值
        elif pos['posSide'] == 'short':
            short_sz = float(pos['pos'])
            short_value += short_sz * latest_price

    long_ratio = long_value / total_equity if total_equity > 0 else 0
    short_ratio = short_value / total_equity if total_equity > 0 else 0
    return long_ratio, short_ratio


def release_funds(inst_id, latest_price, release_len):
    """根据当前价格和订单价格差距，取消指定数量的订单。"""
    open_orders = get_open_orders()
    if not open_orders:
        logger.warning("当前没有未完成的订单。")
        return

    sorted_orders = sorted(
        open_orders['data'],
        key=lambda order: abs(latest_price - float(order['px'])),
        reverse=True
    )

    for order in sorted_orders[:release_len]:
        order_id = order['ordId']
        order_price = float(order['px'])
        price_diff = abs(latest_price - order_price)
        last_updated = float(order['uTime']) / 1000  # 假设时间戳以毫秒为单位

        # 检查价格偏移和时间间隔
        if price_diff <= 200 and (time.time() - last_updated) <= 60:
            logger.warning(
                f"保留订单 {order_id}，订单价格：{order['px']}，最新价格：{latest_price}，差距：{price_diff}，时间间隔在1分钟内。"
            )
            continue

        logger.warning(
            f"取消订单 {order_id}，订单价格：{order['px']}，最新价格：{latest_price}，差距：{price_diff}"
        )
        cancel_order(inst_id, order_id)



def place_batch_orders(order_list):
    """批量下单"""
    result = safe_api_call(tradeAPI.place_multiple_orders, order_list)
    if result:
        with open("order_history.txt", "a") as f:
            f.write(str(result) + "\n")
        logger.warning("批量下单成功：%s", result)
    return result


def create_order(inst_id, side, price, size, pos_side, tp_px):
    """生成单个订单"""
    return {
        "instId": inst_id,
        "tdMode": "cross",
        "side": side,
        "ordType": "limit",
        "px": str(price),
        "sz": str(size),
        "posSide": pos_side,
        "tpTriggerPx": str(tp_px),
        "tpOrdPx": str(tp_px)
    }


def calc_adjustment(x, k=3):
    """
    计算曲线 y = 200 * (exp(k * x) - 1) / (exp(k) - 1)
    :param x: 输入值，可以是一个数或数组，范围为 [0, 1]
    :param k: 陡峭系数，默认为 5
    :return: 对应的 y 值
    """
    value = 200 * (np.exp(k * x) - 1) / (np.exp(k) - 1)
    # 将value向上取整
    return np.ceil(value)

def prepare_orders(latest_price, base_offset, base_profit, long_ratio, short_ratio):
    """
    根据多空比例和价格变化动态调整offset和profit:
    - 假设：当一方比例更大时，对该方仅调整offset，对另一方仅调整profit。
    - 当long_ratio > short_ratio时：多头为强势方
      -> 调整多头offset，调整空头profit
    - 当short_ratio > long_ratio时：空头为强势方
      -> 调整空头offset，调整多头profit

    price方向决定增减，一般根据需要可决定只增加:
    在此示例中，当价格上涨且空头强势：增大空头offset、增大多头profit。
    当价格下跌且多头强势：增大多头offset、增大空头profit。
    当价格上涨且多头强势：增大多头offset、增大空头profit。
    当价格下跌且空头强势：增大空头offset、增大多头profit。

    简化后逻辑：无论价格上涨或下跌，只要有明显强弱侧，就对强侧offset和弱侧profit根据差异放大。
    """
    order_list = []
    diff = abs(long_ratio - short_ratio)

    if diff <= 0.1:
        # 无持仓或持仓均衡，不调整
        adjusted_long_offset = base_offset
        adjusted_short_profit = base_profit
        adjusted_long_profit = base_profit
        adjusted_short_offset = base_offset
    else:
        adjustment = calc_adjustment(diff)
        # 判断哪一边强势
        if long_ratio > short_ratio:
            # 多头强势：只调整多头offset、空头profit
            # 不调整多头profit，也不调整空头offset
            adjusted_long_offset = base_offset + adjustment
            adjusted_short_profit = base_profit + adjustment
            adjusted_long_profit = base_profit
            adjusted_short_offset = base_offset
        else:
            # 空头强势：只调整空头offset、多头profit
            adjusted_short_offset = base_offset + adjustment
            adjusted_long_profit = base_profit + adjustment
            adjusted_long_offset = base_offset
            adjusted_short_profit = base_profit

    # 根据价格方向（可选逻辑，可简化）判断使用哪组调整值
    # 为了简化，这里不区分价格上涨或下跌的正负调整，只是统一增加
    # 如果需要根据价格方向调节加减，请自行在此添加逻辑

    # 最终决定下单:
    # 多单条件检查
    # 多单下单价格：latest_price - adjusted_long_offset
    # 多单止盈：latest_price + adjusted_long_profit
    # 如果多头超过比例不再开多
    if long_ratio < MAX_POSITION_RATIO:
        order_list.append(
            create_order(
                CONFIG["INST_ID"],
                "buy",
                latest_price - (adjusted_long_offset if long_ratio > short_ratio else base_offset),
                CONFIG["ORDER_SIZE"],
                "long",
                latest_price + (adjusted_long_profit if short_ratio > long_ratio else base_profit)
            )
        )
    else:
        logger.warning("多头仓位比例过高，暂停开多。")

    # 空单条件检查
    # 空单下单价格：latest_price + adjusted_short_offset
    # 空单止盈：latest_price - adjusted_short_profit
    if short_ratio < MAX_POSITION_RATIO:
        order_list.append(
            create_order(
                CONFIG["INST_ID"],
                "sell",
                latest_price + (adjusted_short_offset if short_ratio > long_ratio else base_offset),
                CONFIG["ORDER_SIZE"],
                "short",
                latest_price - (adjusted_short_profit if long_ratio > short_ratio else base_profit)
            )
        )
    else:
        logger.warning("空头仓位比例过高，暂停开空。")

    return order_list


# 主循环
if __name__ == "__main__":
    count = 1000000
    last_price = None

    while count > 0:
        try:
            count -= 1

            # 获取最新价格
            latest_price = get_latest_price(CONFIG["INST_ID"])
            if not latest_price:
                logger.warning("无法获取最新价格，跳过本次循环。")
                time.sleep(DELAY_SHORT)
                continue

            # 获取持仓比例
            long_ratio, short_ratio = get_position_ratio(CONFIG["INST_ID"], latest_price)

            # 检查价格变化是否超出阈值
            if last_price is not None and abs(latest_price - last_price) < CONFIG["PRICE_THRESHOLD"]:
                logger.warning("价格变化不足，跳过本次下单。上次价格：%s, 最新价格：%s", last_price, latest_price)
                time.sleep(DELAY_SHORT)
                continue



            # 准备订单（根据多空比例差异和价格走势动态调整）
            orders = prepare_orders(latest_price, CONFIG["OFFSET"], CONFIG["PROFIT"], long_ratio,
                                    short_ratio)

            # 如果有可下单的订单则批量下单
            # release_funds(CONFIG["INST_ID"], latest_price, 2)
            if orders:
                result = place_batch_orders(orders)
                result_str = str(result)
                if result_str and 'failed' in result_str:
                    logger.error("批量下单失败：%s", result_str)
                    release_funds(CONFIG["INST_ID"], latest_price, 2)
                else:
                    prev_price = last_price
                    last_price = latest_price

            time.sleep(DELAY_SHORT)

        except Exception as e:
            logger.error(f"主循环中发生异常：{e}")
            time.sleep(DELAY_SHORT)  # 暂停后继续运行
