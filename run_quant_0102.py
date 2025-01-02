import datetime
import os
import time
import logging

import numpy as np
import okx.Trade as Trade
import okx.MarketData as Market
import okx.Account as Account
import pandas as pd

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

total_profit = 1000
OFFSET = 1000
PROFIT = total_profit - OFFSET
# 配置区域
CONFIG = {
    "INST_ID": "BTC-USDT-SWAP",  # 交易对
    "ORDER_SIZE": 100,  # 每次固定下单量
    "PRICE_THRESHOLD": 500,  # 最新价格变化的阈值
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


class LatestDataManager:
    def __init__(self, capacity, INST_ID):
        self.capacity = capacity
        self.inst_id = INST_ID
        self.max_single_size = 100 # 底层数据最大单次获取数量
        self.data_df = get_train_data(max_candles=self.capacity, inst_id=self.inst_id)

    def get_newest_data(self):
        recent_data_df = get_train_data(max_candles=self.max_single_size, is_newest=True, inst_id=self.inst_id)
        # 判断recent_data_df第一个时间戳（timestamp）是否在data_df中
        if recent_data_df['timestamp'].iloc[0] in self.data_df['timestamp'].values:
            print('历史数据在最新数据中，更新数据')
            # 合并两个df
            self.data_df = pd.concat([recent_data_df, self.data_df], ignore_index=True)
            # 去重
            self.data_df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
            # 排序
            self.data_df.sort_values(by='timestamp', ascending=True, inplace=True)
            # 保留最新的capacity条数据
            self.data_df = self.data_df.iloc[-self.capacity:]
            return self.data_df

        else:
            print('历史数据不在最新数据中，重新获取数据')
            self.data_df = get_train_data(max_candles=self.capacity)
            return self.data_df







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
    """计算多头和空头方向的仓位价值比例，并对没有止盈单的仓位设置相应的止盈单"""
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
    long_sz = 0
    short_sz = 0

    try:
        # 获取所有当前的止盈单
        take_profit_orders = tradeAPI.order_algos_list(ordType="conditional", instId=inst_id)
        if 'data' not in take_profit_orders:
            raise ValueError("返回的止盈单数据异常，未找到data字段。")

        # 获取止盈单列表
        take_profit_orders_data = take_profit_orders['data']

        # 存储已有止盈单的仓位信息，方便后续查找
        existing_long_tp_qty = 0
        existing_short_tp_qty = 0
        existing_long_tps = []  # 存储多单止盈单的信息 (sz, algoId, cTime)
        existing_short_tps = [] # 存储空单止盈单的信息 (sz, algoId, cTime)

        # 当前时间（毫秒级）
        current_time = int(time.time() * 1000)

        # 遍历止盈单，统计数量并取消超时的止盈单
        for order in take_profit_orders_data:
            if order['side'] == 'sell':  # 多单止盈
                existing_long_tp_qty += float(order['sz']) if order['sz'] != "" else 0
                existing_long_tps.append({
                    'sz': float(order['sz']) if order['sz'] != "" else 0,
                    'algoId': order['algoId'],
                    'cTime': int(order['cTime'])
                })
                # 检查是否超过10小时，超过则取消
                if current_time - int(order['cTime']) > 10 * 60 * 60 * 1000:
                    algo_orders = [{"instId": inst_id, "algoId": order['algoId']}]
                    tradeAPI.cancel_algo_order(algo_orders)
                    print(f"取消超过10小时的多单止盈单，algoId: {order['algoId']}")
            elif order['side'] == 'buy':  # 空单止盈
                existing_short_tp_qty += float(order['sz']) if order['sz'] != "" else 0
                existing_short_tps.append({
                    'sz': float(order['sz']) if order['sz'] != "" else 0,
                    'algoId': order['algoId'],
                    'cTime': int(order['cTime'])
                })
                # 检查是否超过2小时，超过则取消
                if current_time - int(order['cTime']) > 2 * 60 * 60 * 1000:
                    algo_orders = [{"instId": inst_id, "algoId": order['algoId']}]
                    tradeAPI.cancel_algo_order(algo_orders)
                    print(f"取消超过2小时的空单止盈单，algoId: {order['algoId']}")

        # 计算仓位价值
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

        print(f"多头数量: {long_sz}, 空头数量: {short_sz}, 多头占比: {long_ratio}, 空头占比: {short_ratio} 已有多单止盈数量: {existing_long_tp_qty}, 已有空单止盈数量: {existing_short_tp_qty}")
        diff_long_sz = long_sz - existing_long_tp_qty
        # 为多单设置止盈单
        if long_position_exists and diff_long_sz > 0:
            tp_price_long = avg_long_price + total_profit  # 多单止盈价格
            tp_price_long = max(tp_price_long, latest_price)  # 限制止盈价格
            create_take_profit_order(inst_id, 'long', tp_price_long, diff_long_sz)
            print(f"为多单设置止盈单，止盈价格: {tp_price_long}, 数量: {long_sz}")
        diff_short_size = short_sz - existing_short_tp_qty
        # 为空单设置止盈单
        if short_position_exists and diff_short_size > 0:
            tp_price_short = avg_short_price - total_profit  # 空单止盈价格
            tp_price_short = min(tp_price_short, latest_price)  # 限制止盈价格
            create_take_profit_order(inst_id, 'short', tp_price_short, diff_short_size)
            print(f"为空单设置止盈单，止盈价格: {tp_price_short}, 数量: {short_sz}")
        # 计算还可以开多少单
        return long_sz, short_sz, avg_long_price, avg_short_price

    except Exception as e:
        print(f"发生错误: {e}")
        return 0, 0, 0, 0



def release_alg_old_funds(inst_id, remain_count=1):
    """每个方向只保留最新的remain_count个订单。"""
    alg_open_orders = get_alg_open_orders()
    id_key = 'algoId'
    time_key = 'createTime'  # 假设你的订单数据中有创建时间

    if not alg_open_orders:
        logger.warning("当前没有未完成的移动止盈止损订单。")
        return

    buy_orders = [order for order in alg_open_orders['data'] if order['side'] == 'buy']
    sell_orders = [order for order in alg_open_orders['data'] if order['side'] == 'sell']

    # 按照创建时间倒序排序，最新的在前面
    buy_orders.sort(key=lambda order: order.get(time_key, 0), reverse=True)
    sell_orders.sort(key=lambda order: order.get(time_key, 0), reverse=True)

    # 取消多余的买单
    if len(buy_orders) > remain_count:
        orders_to_cancel = buy_orders[remain_count:]
        for order in orders_to_cancel:
            order_id = order[id_key]
            logger.warning(f"取消多余的买单 {order_id}，保留最新的 {remain_count} 个。")
            cancel_alg_order(inst_id, order_id)

    # 取消多余的卖单
    if len(sell_orders) > remain_count:
        orders_to_cancel = sell_orders[remain_count:]
        for order in orders_to_cancel:
            order_id = order[id_key]
            logger.warning(f"取消多余的卖单 {order_id}，保留最新的 {remain_count} 个。")
            cancel_alg_order(inst_id, order_id)


def release_near_funds(inst_id):
    """只保留创建时间在1分钟内的订单，取消其他所有订单。"""
    start_time = time.time()
    open_orders = get_open_orders()
    if not open_orders:
        logger.warning("当前没有未完成的订单。")
        return

    now_ts = int(time.time() * 1000)  # 当前时间戳，毫秒级
    one_minute_ago_ts = now_ts - 60 * 1000  # 1分钟前的时间戳

    for order in open_orders['data']:
        order_id = order['ordId']
        order_ctime = int(order['cTime'])

        if order_ctime < one_minute_ago_ts:
            logger.warning(
                f"取消订单 {order_id}，创建时间：{order_ctime}，早于1分钟前"
            )
            print(cancel_order(inst_id, order_id))
        else:
            logger.info(
                f"保留订单 {order_id}，创建时间：{order_ctime}，在1分钟内"
            )
    print(f"取消订单耗时：{time.time() - start_time:.2f}秒")
def release_funds(inst_id, latest_price, release_len):
    """根据当前价格和订单价格差距，取消指定数量的订单。"""
    start_time = time.time()
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
        if price_diff <= 1000 and (time.time() - last_updated) <= 6000:
            logger.warning(
                f"保留订单 {order_id}，订单价格：{order['px']}，最新价格：{latest_price}，差距：{price_diff}，时间间隔在1分钟内。"
            )
            continue

        logger.warning(
            f"取消订单 {order_id}，订单价格：{order['px']}，最新价格：{latest_price}，差距：{price_diff}"
        )
        print(cancel_order(inst_id, order_id))
        print(f"取消订单耗时：{time.time() - start_time:.2f}秒")



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


def get_kline_data_newest(inst_id, bar="1m", limit=100, max_candles=1000):
    """
    从OKX获取历史K线数据，并返回DataFrame。

    :param inst_id: 产品ID，例如 "BTC-USDT"
    :param bar: 时间粒度，例如 "1m", "5m", "1H" 等
    :param limit: 单次请求的最大数据量，默认100，最大100
    :param max_candles: 请求的最大K线数量，默认1000
    :return: pandas.DataFrame，包含K线数据
    """
    all_data = []
    after = ''  # 初始值为None，获取最新数据
    fetched_candles = 0  # 已获取的K线数量
    fail_count = 0  # 失败次数
    max_retries = 3  # 最大重试次数

    while fetched_candles < max_candles:
        try:
            # 调用OKX API获取历史K线数据
            response = marketAPI.get_candlesticks(instId=inst_id, bar=bar, after=after, limit=limit)

            if response.get("code") != "0":
                print(f"获取K线数据失败，错误代码：{response.get('code')}，错误消息：{response.get('msg')}")
                time.sleep(1)
                fail_count += 1
                if fail_count >= max_retries:
                    print(f"连续失败 {max_retries} 次，停止获取。")
                    break
            else:
                fail_count = 0
                # 提取返回数据
                data = response.get("data", [])
                if not data:
                    print("无更多数据，已全部获取。")
                    break

                # 解析数据并添加到总数据中
                all_data.extend(data)
                fetched_candles += len(data)

                # 更新 `after` 参数为当前返回数据的最早时间戳，用于获取更早的数据
                after = data[-1][0]

                # 如果获取的数据量小于limit，说明数据已获取完毕
                if len(data) < limit:
                    break

                # 短暂延迟，避免触发API限频
                time.sleep(0.2)

        except Exception as e:
            print(f"获取K线数据时出现异常：{e}")
            break

    # 将所有数据转换为DataFrame，即使all_data为空也能处理
    if all_data:
        df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", "volCcy", "volCcyQuote",
                                             "confirm"])

        # 数据类型转换
        # 将时间戳转换为 datetime 对象，并将其设置为 UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float) / 1000, unit="s", utc=True)

        # 将 UTC 时间转换为北京时间（Asia/Shanghai 时区，UTC+8）
        df["timestamp"] = df["timestamp"].dt.tz_convert('Asia/Shanghai')
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

        # 按时间排序
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    else:
        print("未能获取到任何K线数据。")
        return pd.DataFrame() # 返回一个空的 DataFrame

def get_kline_data(inst_id, bar="1m", limit=100, max_candles=1000):
    """
    从OKX获取历史K线数据，并返回DataFrame。

    :param inst_id: 产品ID，例如 "BTC-USDT"
    :param bar: 时间粒度，例如 "1m", "5m", "1H" 等
    :param limit: 单次请求的最大数据量，默认100，最大100
    :param max_candles: 请求的最大K线数量，默认1000
    :return: pandas.DataFrame，包含K线数据
    """
    all_data = []
    after = ''  # 初始值为None，获取最新数据
    fetched_candles = 0  # 已获取的K线数量
    fail_count = 0  # 失败次数
    max_retries = 3  # 最大重试次数

    while fetched_candles < max_candles:
        try:
            # 调用OKX API获取历史K线数据
            response = marketAPI.get_history_candlesticks(instId=inst_id, bar=bar, after=after, limit=limit)

            if response.get("code") != "0":
                print(f"获取K线数据失败，错误代码：{response.get('code')}，错误消息：{response.get('msg')}")
                time.sleep(1)
                fail_count += 1
                if fail_count >= max_retries:
                    print(f"连续失败 {max_retries} 次，停止获取。")
                    break
            else:
                fail_count = 0
                # 提取返回数据
                data = response.get("data", [])
                if not data:
                    print("无更多数据，已全部获取。")
                    break

                # 解析数据并添加到总数据中
                all_data.extend(data)
                fetched_candles += len(data)

                # 更新 `after` 参数为当前返回数据的最早时间戳，用于获取更早的数据
                after = data[-1][0]

                # 如果获取的数据量小于limit，说明数据已获取完毕
                if len(data) < limit:
                    break

                # 短暂延迟，避免触发API限频
                time.sleep(0.2)

        except Exception as e:
            print(f"获取K线数据时出现异常：{e}")
            break

    # 将所有数据转换为DataFrame，即使all_data为空也能处理
    if all_data:
        df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", "volCcy", "volCcyQuote",
                                             "confirm"])

        # 数据类型转换
        # 将时间戳转换为 datetime 对象，并将其设置为 UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float) / 1000, unit="s", utc=True)

        # 将 UTC 时间转换为北京时间（Asia/Shanghai 时区，UTC+8）
        df["timestamp"] = df["timestamp"].dt.tz_convert('Asia/Shanghai')
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

        # 按时间排序
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    else:
        print("未能获取到任何K线数据。")
        return pd.DataFrame() # 返回一个空的 DataFrame

def add_target_variables_op(df, max_decoder_length=30):
    """
    添加目标变量，包括未来多个时间步的涨幅和跌幅，针对close、high、low的多阈值计算。

    :param df: 数据集
    :param thresholds: 阈值列表，默认[0.1, 0.2, ..., 1.0]
    :param max_decoder_length: 预测未来的时间步长，默认5
    :return: 添加了目标变量的DataFrame
    """
    # 创建一个新的字典，用于存储所有新增列
    new_columns = {}


    for col in ["close"]:
        # 获取下一个时间点的最高价和最低价
        next_high = df['high'].shift(-1)
        next_low = df['low'].shift(-1)

        # 计算下一个时间点的最高价和最低价相对于当前时间点的涨跌幅
        new_columns[f"{col}_next_max_up"] = (next_high - df[col]) / df[col] * 100  # 最高价涨幅
        new_columns[f"{col}_next_max_down"] = (df[col] - next_low) / df[col] * 100  # 最低价跌幅

        for step in range(10, max_decoder_length + 1, 10):  # 未来 1 到 max_decoder_length 分钟
            # 获取未来 step 个时间窗口内的最高价和最低价
            future_max_high = df['high'].rolling(window=step, min_periods=1).max().shift(-step)
            future_min_low = df['low'].rolling(window=step, min_periods=1).min().shift(-step)

            # 计算未来 step 个时间窗口内的最大涨幅和跌幅 (修正部分)
            new_columns[f"{col}_max_up_t{step}"] = (future_max_high - df[col]) / df[col] * 100 #最大涨幅用最高价
            new_columns[f"{col}_max_down_t{step}"] = (df[col] - future_min_low) / df[col] * 100 #最大跌幅用最低价

            # 计算未来 step 个时间窗口的涨跌幅
            future_return = (df['close'].shift(-step) - df['close']) / df['close'] * 100
            new_columns[f"{col}_max_return_t{step}"] = future_return
    # 使用 pd.concat 一次性将所有新列添加到原数据框
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    return df

def get_train_data(inst_id="BTC-USDT-SWAP", bar="1m", limit=100, max_candles=1000, is_newest=False):
    # inst_id = "BTC-USDT-SWAP"
    # bar = "1m"
    # limit = 100
    # max_candles = 60 * 24

    # 获取数据
    if is_newest:
        kline_data = get_kline_data_newest(inst_id=inst_id, bar=bar, limit=limit, max_candles=max_candles)
    else:
        kline_data = get_kline_data(inst_id=inst_id, bar=bar, limit=limit, max_candles=max_candles)

    if not kline_data.empty:
        # print("成功获取K线数据，开始处理...")

        # 添加时间特征
        # kline_data = add_time_features(kline_data)

        # 添加目标变量
        kline_data = add_target_variables_op(kline_data)

        # 重置索引
        kline_data.reset_index(drop=True, inplace=True)
        return kline_data
    else:
        print("未能获取到任何K线数据。")
        return pd.DataFrame()


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
        # 将 order 增量写入到文件
        with open("order_history.txt", "a") as f:
            f.write(str(order) + "\n")

        print(f"{side.upper()} 订单成功：", order)
        return order
    except Exception as e:
        print(f"{side.upper()} 订单失败，错误信息：", e)
        return None

# 主循环
if __name__ == "__main__":
    count = 1000000
    last_price = None
    INST_ID = "TON-USDT-SWAP"
    ORDER_SIZE = 1
    offset = 0
    period_profit_map = {
        2010: 0.4 / 100,
        710: 0.2 / 100,
    }

    newest_data = LatestDataManager(2200, INST_ID)

    while count > 0:
        try:
            count -= 1
            current_time = time.time()
            current_time = pd.to_datetime(current_time, unit='s')
            start_time = time.time()

            # 控制每次循环的间隔，避免过于频繁地调用
            if start_time % 60 < 59:
                time.sleep(1)
                continue

            feature_df = newest_data.get_newest_data()
            latest_price = feature_df['close'].iloc[-1]

            for period, profit in period_profit_map.items():
                # 获取feature_df中最新数据close在period时间内的最大值和最小值
                max_price = feature_df['close'].iloc[-period:].max()
                min_price = feature_df['close'].iloc[-period:].min()
                # print(f"max_price: {max_price}, min_price: {min_price} latest_price: {latest_price} long_sz: {long_sz} short_sz: {short_sz}")
                # 做多限价单
                place_order(
                    INST_ID,
                    "buy",
                    "limit",  # 限价单
                    ORDER_SIZE,
                    price=min_price - offset,  # 买入价格
                    tp_price=min_price + profit * min_price   # 止盈价格
                )

                # 做空限价单
                place_order(
                    INST_ID,
                    "sell",
                    "limit",  # 限价单
                    ORDER_SIZE,
                    price=max_price + offset,  # 卖出价格
                    tp_price=max_price - profit * max_price  # 止盈价格
                )
                time.sleep(3)
                release_near_funds(INST_ID)

        except Exception as e:
            print(f"发生错误: {e}")
            continue
