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
    1.直接以当前价格比较大的偏差买入和卖出，最求高利润，可能交易次数很少。

历史策略记录:
    1.负偏差价格下单，基本上一下单就会秒成，增加下单的成功率（相应的盈利偏差就增加了 整体盈利差保持在90）
    2.多空差异大时会减小多的那个方向的买入价格（认为下降空间很大或者不想继续增加这个方向的仓位了）。会增加小的那个方向的止盈利润（未改变买入价 表示还是希望能够买入增加持仓量。同时增加止盈是认为如果继续往反方向变化的话能够盈利多一点 减小损失）
    3.价格相较于上一次的触发价格相差10才进行新的下单（避免单子都分布在一个价格区间）
    效果:
        还是无法应对单边价格变化的问题，如果价格一直上涨或者下跌，会导致持仓比例不断增加，最终导致爆仓。
"""

# 日志配置
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 设置代理（如果需要）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# 全局延时配置
DELAY_SHORT = 1  # 短延时（秒）

total_profit = 500
OFFSET = 300
PROFIT = total_profit - OFFSET
# 配置区域
CONFIG = {
    "INST_ID": "BTC-USDT-SWAP",  # 交易对
    "ORDER_SIZE": 0.1,  # 每次固定下单量
    "PRICE_THRESHOLD": 100,  # 最新价格变化的阈值
    "OFFSET": OFFSET,  # 下单价格偏移量（基础值）
    "PROFIT": PROFIT  # 止盈偏移量（基础值）
}

MAX_POSITION_RATIO = 1  # 最大持仓比例为1
flag = "0"  # 实盘: 0, 模拟盘: 1

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

def get_alg_open_orders():
    """获取未完成的订单"""
    return safe_api_call(tradeAPI.order_algos_list, ordType="move_order_stop")


def cancel_order(inst_id, order_id):
    """撤销订单"""
    return safe_api_call(tradeAPI.cancel_order, instId=inst_id, ordId=order_id)

def cancel_alg_order(inst_id, order_id):
    """撤销订单"""
    param = [{"instId": inst_id, "algoId": order_id}]
    return safe_api_call(tradeAPI.cancel_algo_order, param)


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

def create_take_profit_order(inst_id, pos_side, tp_price, quantity):
    """根据方向、价格和数量创建止盈单"""
    if pos_side == 'long':
        side = 'sell'
    else:
        side = 'buy'
    quantity = round(quantity, 1)
    result = tradeAPI.place_algo_order(
        instId=inst_id,
        tdMode='cross',
        side=side,
        posSide=pos_side,
        ordType='conditional',
        sz=str(quantity),  # 设置止盈单的数量
        tpTriggerPx=str(tp_price),
        tpOrdPx=str(tp_price)
    )
    print(f"创建止盈单：{result}")


def get_position_ratio(inst_id, latest_price):
    """计算多头和空头方向的仓位价值比例，并根据差距设置止盈单"""
    positions = get_positions(inst_id)
    total_equity = get_account_equity()
    total_equity *= 10000  # 账户总权益扩大10000倍
    if total_equity == 0:
        return 0, 0, 0, 0

    long_value = 0.0
    short_value = 0.0
    avg_long_price = 0.0
    avg_short_price = 0.0
    long_position_exists = False
    short_position_exists = False

    try:
        # 获取所有当前的止盈单
        take_profit_orders = tradeAPI.order_algos_list(ordType="conditional", instId=inst_id)
        if 'data' not in take_profit_orders:
            raise ValueError("返回的止盈单数据异常，未找到data字段。")

        # 获取止盈单列表
        take_profit_orders_data = take_profit_orders['data']

        # 初始化止盈单计数器
        long_tp_count = 0  # 多单止盈单个数
        short_tp_count = 0  # 空单止盈单个数
        existing_long_tp = 0  # 多单止盈单的数量总和
        existing_short_tp = 0  # 空单止盈单的数量总和
        long_sz = 0
        short_sz = 0

        # 当前时间（毫秒级）
        current_time = int(time.time() * 1000)

        # 统计已有的止盈单数量
        for order in take_profit_orders_data:
            if order['side'] == 'sell':  # 多单止盈
                long_tp_count += 1
                existing_long_tp += float(order['sz']) if order['sz'] != "" else 0
                # 检查是否超过2小时，超过则取消
                if current_time - int(order['cTime']) > 10 * 60 * 60 * 1000:  # 超过2小时
                    algo_orders = [{"instId": inst_id, "algoId": order['algoId']}]
                    tradeAPI.cancel_algo_order(algo_orders)
                    long_tp_count -= 1  # 更新多单止盈单计数器
                    existing_long_tp -= float(order['sz'])  # 更新多单止盈单数量总和
                    print(f"取消超过2小时的多单止盈单，algoId: {order['algoId']}")
            elif order['side'] == 'buy':  # 空单止盈
                short_tp_count += 1
                existing_short_tp += float(order['sz']) if order['sz'] != "" else 0
                # 检查是否超过2小时，超过则取消
                if current_time - int(order['cTime']) > 2 * 60 * 60 * 1000:  # 超过2小时
                    algo_orders = [{"instId": inst_id, "algoId": order['algoId']}]
                    tradeAPI.cancel_algo_order(algo_orders)
                    short_tp_count -= 1  # 更新空单止盈单计数器
                    existing_short_tp -= float(order['sz'])  # 更新空单止盈单数量总和
                    print(f"取消超过2小时的空单止盈单，algoId: {order['algoId']}")

        # 计算仓位价值和止盈单数量
        for pos in positions:
            if pos['posSide'] == 'long':
                long_sz = float(pos['pos']) if pos['pos'] != "" else 0
                long_value += long_sz * latest_price
                avg_long_price = float(pos['avgPx']) if pos['avgPx'] != "" else 0
                long_position_exists = True
            elif pos['posSide'] == 'short':
                short_sz = float(pos['pos']) if pos['pos'] != "" else 0
                short_value += short_sz * latest_price
                avg_short_price = float(pos['avgPx']) if pos['avgPx'] != "" else 0
                short_position_exists = True

        long_ratio = long_value / total_equity if total_equity > 0 else 0
        short_ratio = short_value / total_equity if total_equity > 0 else 0

        print(f"多头数量: {long_sz}, 空头数量: {short_sz}, 多头占比: {long_ratio}, 空头占比: {short_ratio} 多头止盈单数量: {long_tp_count}, 空头止盈单数量: {short_tp_count}")

        # 检查单个方向的止盈单数量是否超过50个
        def cancel_oldest_take_profit_order(inst_id, side):
            """取消最早的止盈单，并更新计数器，直到止盈单数量小于50"""
            nonlocal long_tp_count, short_tp_count, existing_long_tp, existing_short_tp

            # 根据止盈单数量判断是否需要继续取消
            if side == 'long' and long_tp_count >= 50:
                while long_tp_count >= 50:
                    # 取消最早的多单止盈单
                    for order in take_profit_orders_data:
                        if order['side'] == 'sell':
                            algo_orders = [{"instId": inst_id, "algoId": order['algoId']}]
                            tradeAPI.cancel_algo_order(algo_orders)
                            long_tp_count -= 1  # 更新多单止盈单计数器
                            existing_long_tp -= float(order['sz'])  # 更新多单止盈单数量总和
                            print(f"取消多单止盈单，algoId: {order['algoId']}")
                            break  # 取消一个后退出
            elif side == 'short' and short_tp_count >= 50:
                while short_tp_count >= 50:
                    # 取消最早的空单止盈单
                    for order in take_profit_orders_data:
                        if order['side'] == 'buy':
                            algo_orders = [{"instId": inst_id, "algoId": order['algoId']}]
                            tradeAPI.cancel_algo_order(algo_orders)
                            short_tp_count -= 1  # 更新空单止盈单计数器
                            existing_short_tp -= float(order['sz'])  # 更新空单止盈单数量总和
                            print(f"取消空单止盈单，algoId: {order['algoId']}")
                            break  # 取消一个后退出

        # 为多单设置止盈单（计算差距并创建止盈单）
        if long_position_exists:
            long_diff = long_sz - existing_long_tp  # 差距（应该使用数量总和）
            if long_diff > 0:
                cancel_oldest_take_profit_order(inst_id, 'long')  # 如果已有50个多单止盈单，取消一个
                long_diff = long_sz - existing_long_tp
                tp_price_long = min(latest_price, avg_long_price) + total_profit  # 多单止盈价格
                create_take_profit_order(inst_id, 'long', tp_price_long, long_diff)
                print(f"为多单设置止盈单，止盈价格: {tp_price_long}, 数量: {long_diff}")

        # 为空单设置止盈单（计算差距并创建止盈单）
        if short_position_exists:
            short_diff = short_sz - existing_short_tp  # 差距（应该使用数量总和）
            if short_diff > 0:
                cancel_oldest_take_profit_order(inst_id, 'short')  # 如果已有50个空单止盈单，取消一个
                short_diff = short_sz - existing_short_tp  # 差距（应该使用数量总和）
                tp_price_short = max(avg_short_price, latest_price) - total_profit  # 空单止盈价格
                create_take_profit_order(inst_id, 'short', tp_price_short, short_diff)
                print(f"为空单设置止盈单，止盈价格: {tp_price_short}, 数量: {short_diff}")

        return long_ratio, short_ratio, avg_long_price, avg_short_price

    except Exception as e:
        print(f"发生错误: {e}")
        return 0, 0, 0, 0



def release_alg_near_funds(inst_id):
    """根据当前价格和订单价格差距，取消指定数量的订单，强制释放同一方向中，委托价格差距小于100的订单。"""
    alg_open_orders = get_alg_open_orders()
    px_key = 'activePx'
    id_key = 'algoId'
    if not alg_open_orders:
        logger.warning("当前没有未完成的移动止盈止损订单。")
        return

    buy_orders = []
    sell_orders = []

    # 分类订单
    for order in alg_open_orders['data']:
        if order['side'] == 'buy':
            buy_orders.append(order)
        elif order['side'] == 'sell':
            sell_orders.append(order)

    # 按照价格排序：优先处理价格差距较小的订单
    buy_orders = sorted(buy_orders, key=lambda order: float(order[px_key]))
    sell_orders = sorted(sell_orders, key=lambda order: float(order[px_key]))

    max_price_diff = 2 * CONFIG["PRICE_THRESHOLD"]  # 最大价格差距

    # 取消买单方向的订单，检查相邻订单之间的价格差距
    for i in range(1, len(buy_orders)):

        order1 = buy_orders[i-1]
        order2 = buy_orders[i]

        price_diff = abs(float(order2[px_key]) - float(order1[px_key]))

        if price_diff < max_price_diff:
            order_id = order1[id_key]  # 取消前面一个订单（价格低）
            logger.warning(
                f"强制取消买单 {order_id}，价格差距：{price_diff}"
            )
            cancel_alg_order(inst_id, order_id)

    # 取消卖单方向的订单，检查相邻订单之间的价格差距
    for i in range(1, len(sell_orders)):

        order1 = sell_orders[i-1]
        order2 = sell_orders[i]

        price_diff = abs(float(order2[px_key]) - float(order1[px_key]))

        # 如果价格差距小于100，取消这两个订单中的一个
        if price_diff < max_price_diff:
            order_id = order2[id_key]  # 取消后面一个订单（价格高）
            logger.warning(
                f"强制取消卖单 {order_id}，价格差距：{price_diff}"
            )
            cancel_alg_order(inst_id, order_id)


def release_near_funds(inst_id):
    """根据当前价格和订单价格差距，取消指定数量的订单，强制释放同一方向中，委托价格差距小于100的订单。"""
    open_orders = get_open_orders()
    if not open_orders:
        logger.warning("当前没有未完成的订单。")
        return

    buy_orders = []
    sell_orders = []

    # 分类订单
    for order in open_orders['data']:
        if order['side'] == 'buy':
            buy_orders.append(order)
        elif order['side'] == 'sell':
            sell_orders.append(order)

    # 按照价格排序：优先处理价格差距较小的订单
    buy_orders = sorted(buy_orders, key=lambda order: float(order['px']))
    sell_orders = sorted(sell_orders, key=lambda order: float(order['px']))

    max_price_diff = 2 * CONFIG["PRICE_THRESHOLD"]  # 最大价格差距

    # 取消买单方向的订单，检查相邻订单之间的价格差距
    for i in range(1, len(buy_orders)):

        order1 = buy_orders[i-1]
        order2 = buy_orders[i]

        price_diff = abs(float(order2['px']) - float(order1['px']))

        if price_diff < max_price_diff:
            order_id = order1['ordId']  # 取消前面一个订单（价格低）
            logger.warning(
                f"强制取消买单 {order_id}，价格差距：{price_diff}"
            )
            cancel_order(inst_id, order_id)

    # 取消卖单方向的订单，检查相邻订单之间的价格差距
    for i in range(1, len(sell_orders)):

        order1 = sell_orders[i-1]
        order2 = sell_orders[i]

        price_diff = abs(float(order2['px']) - float(order1['px']))

        # 如果价格差距小于100，取消这两个订单中的一个
        if price_diff < max_price_diff:
            order_id = order2['ordId']  # 取消后面一个订单（价格高）
            logger.warning(
                f"强制取消卖单 {order_id}，价格差距：{price_diff}"
            )
            cancel_order(inst_id, order_id)

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
        if price_diff <= 1000 and (time.time() - last_updated) <= 60:
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

def create_trailing_stop_order(inst_id, side, trigger_price, size, pos_side, trail_value=10):
    """生成一个移动止盈止损订单"""
    return {
        "instId": inst_id,               # 交易对
        "tdMode": "cross",               # 交易模式（全仓）
        "side": side,                    # 平仓方向（'buy' 或 'sell'）
        "ordType": "move_order_stop",    # 移动止盈止损类型
        "sz": str(size),                 # 数量
        "posSide": pos_side,             # 仓位方向（'long' 或 'short'）
        "activePx": str(trigger_price),  # 触发止盈止损的价格
        "callbackSpread": str(10),   # 移动止盈止损的价格波动幅度
    }


def calc_adjustment(x, k=3):
    """
    计算曲线 y = 200 * (exp(k * x) - 1) / (exp(k) - 1)
    :param x: 输入值，可以是一个数或数组，范围为 [0, 1]
    :param k: 陡峭系数，默认为 5
    :return: 对应的 y 值
    """
    value = 1000 * (np.exp(k * x) - 1) / (np.exp(k) - 1)
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
    move_order_list = []
    diff = abs(long_ratio - short_ratio)

    if diff <= 0.1:
        # 无持仓或持仓均衡，不调整
        adjusted_long_offset = base_offset
        adjusted_short_profit = base_profit
        adjusted_long_profit = base_profit
        adjusted_short_offset = base_offset
    else:
        adjustment = calc_adjustment(diff)
        print(f"多空比例差异：{diff}，调整值：{adjustment}")
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
        move_order_list.append(
            create_trailing_stop_order(
                CONFIG["INST_ID"],
                "buy",
                latest_price - 2 * (adjusted_long_offset if long_ratio > short_ratio else base_offset),
                CONFIG["ORDER_SIZE"],
                "long"
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
        move_order_list.append(
            create_trailing_stop_order(
                CONFIG["INST_ID"],
                "sell",
                latest_price + 2 * (adjusted_short_offset if short_ratio > long_ratio else base_offset),
                CONFIG["ORDER_SIZE"],
                "short"
            )
        )
    else:
        logger.warning("空头仓位比例过高，暂停开空。")
    # 调整order_list的顺序，如果多头比例大于空头，将空头放在前面
    if long_ratio > short_ratio:
        order_list.reverse()
    return order_list, move_order_list

def get_take_profit_orders():
    """获取所有正在委托的止盈单"""
    tp_orders = safe_api_call(tradeAPI.order_algos_list, ordType="conditional")
    if tp_orders and 'data' in tp_orders:
        return tp_orders['data']  # 返回条件单数据
    return []

def modify_take_profit_orders():
    """查询并修改所有止盈单"""
    tp_orders = get_take_profit_orders()
    if tp_orders:
        for order in tp_orders:
            order_id = order['algoId']
            current_tp_trigger_price = float(order['tpTriggerPx'])
            current_tp_order_price = float(order['tpOrdPx'])

            # 增加50作为新的止盈触发价和止盈价格
            new_tp_trigger_price = current_tp_trigger_price + 50
            new_tp_order_price = current_tp_order_price + 50

            # 修改止盈单
            result = safe_api_call(
                tradeAPI.amend_algo_order,
                instId=CONFIG["INST_ID"],   # 传入交易对
                algoId=order_id,            # 止盈单的唯一标识
                newTpTriggerPx=str(new_tp_trigger_price),  # 新的触发价格
                newTpOrdPx=str(new_tp_order_price)         # 新的止盈价格
            )

            if result:
                logger.info(f"止盈单 {order_id} 修改成功，新止盈触发价格为 {new_tp_trigger_price}，新止盈价格为 {new_tp_order_price}")
            else:
                logger.error(f"止盈单 {order_id} 修改失败")
    else:
        logger.info("没有找到需要修改的止盈单")


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
            long_ratio, short_ratio, avg_long_price, avg_short_price = get_position_ratio(CONFIG["INST_ID"], latest_price)

            # 检查价格变化是否超出阈值
            if last_price is not None and abs(latest_price - last_price) < CONFIG["PRICE_THRESHOLD"]:
                logger.warning("价格变化不足，跳过本次下单。上次价格：%s, 最新价格：%s 差值：%s", last_price, latest_price, abs(latest_price - last_price))
                time.sleep(DELAY_SHORT)
                continue



            # 准备订单（根据多空比例差异和价格走势动态调整）
            orders, move_order_list = prepare_orders(latest_price, CONFIG["OFFSET"], CONFIG["PROFIT"], long_ratio,
                                    short_ratio)

            if orders:
                for order in move_order_list:
                    result = tradeAPI.place_algo_order(**order)
                    print(f"移动止盈止损订单：{result}")
                for order in orders:
                    temp_order = [order]
                    logger.info("下单信息：%s", order)
                    result = place_batch_orders(temp_order)
                # result = place_batch_orders(orders)
                result_str = str(result)
                if result_str and 'failed' in result_str:
                    logger.error("批量下单失败：%s", result_str)
                    release_funds(CONFIG["INST_ID"], latest_price, 2)
                else:
                    prev_price = last_price
                    last_price = latest_price
                    print(orders)
                release_near_funds(CONFIG["INST_ID"])
                release_alg_near_funds(CONFIG["INST_ID"])

            time.sleep(DELAY_SHORT)

        except Exception as e:
            logger.error(f"主循环中发生异常：{e}")
            time.sleep(DELAY_SHORT)  # 暂停后继续运行
