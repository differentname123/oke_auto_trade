import os
import pandas as pd
import numpy as np


# ==========================================
# 1. 数据预处理适配器
# ==========================================
def preprocess_minute_data(df_list, time_offset='0h'):
    """
    将交易所拉取的【分钟级 K线列表】无损转换为信号引擎所需的【4H 截面 DataFrame】
    """
    dfs = []

    for df in df_list:
        if df is None or df.empty:
            continue

        df_copy = df.copy()

        # 1. 提取当前 df 对应的币种名称
        coin_name = df_copy['coin_name'].iloc[0]

        # 2. 统一时间索引处理
        if pd.api.types.is_numeric_dtype(df_copy['open_time']) and df_copy['open_time'].max() > 1e11:
            df_copy['open_time'] = pd.to_datetime(df_copy['open_time'], unit='ms')
        elif not pd.api.types.is_datetime64_any_dtype(df_copy['open_time']):
            df_copy['open_time'] = pd.to_datetime(df_copy['open_time'])

        df_copy.set_index('open_time', inplace=True)
        df_copy.sort_index(inplace=True)

        # 3. 核心对齐：使用与回测一致的重采样逻辑生成高低价
        df_4h_coin = df_copy['close'].resample('4h', offset=time_offset).agg(
            open='first',
            high='max',
            low='min',
            close='last'
        )

        # 4. 清理因时间错位产生的碎片空 K 线
        df_4h_coin.dropna(how='all', inplace=True)

        # 5. 统一重命名规范
        df_4h_coin.rename(columns={
            'open': f"{coin_name}_open",
            'high': f"{coin_name}_high",
            'low': f"{coin_name}_low",
            'close': coin_name
        }, inplace=True)

        dfs.append(df_4h_coin)

    if not dfs:
        raise ValueError("传入的 df_list 全为空或无法解析！")

    # 6. 横向合并与前向填充兜底
    df_raw = pd.concat(dfs, axis=1).sort_index()
    df_processed = df_raw.ffill()

    return df_processed


# ==========================================
# 2. 核心流式推演引擎 (账本级对齐，包含 Reason)
# ==========================================
def generate_historical_trade_logs(df: pd.DataFrame, params: dict, trade_mode: str, initial_capital=10000.0):
    """
    流式模拟引擎：生成与回测 100% 一致的 trade_logs DataFrame。
    """
    MOM_WINDOW = params['MOM_WINDOW']
    VOL_WINDOW = params['VOL_WINDOW']
    BTC_TREND_WINDOW = params['BTC_TREND_WINDOW']
    TOP_K = int(params.get('TOP_K', 2))
    MAX_WEIGHT = params['MAX_WEIGHT']
    FEE_RATE = 0.0005  # 费率保持一致

    coins = [c for c in df.columns if not any(suffix in c for suffix in ['_open', '_high', '_low'])]
    n_coins = len(coins)
    coin_to_idx = {c: idx for idx, c in enumerate(coins)}

    if 'BTC' not in coins:
        raise ValueError("数据中必须包含 BTC 作为宏观开关！")

    # 批量计算指标矩阵
    df_close = df[coins]
    returns = df_close.pct_change(MOM_WINDOW)

    high_df = df[[f"{c}_high" for c in coins]].copy()
    high_df.columns = coins
    low_df = df[[f"{c}_low" for c in coins]].copy()
    low_df.columns = coins
    prev_close = df_close.shift(1)

    tr_arr = np.fmax.reduce([
        (high_df - low_df).values,
        (high_df - prev_close).abs().values,
        (low_df - prev_close).abs().values
    ])

    atr = pd.DataFrame(tr_arr, index=df.index, columns=coins).rolling(window=VOL_WINDOW).mean()
    atr_pct = atr / df_close

    adj_mom = returns / (atr_pct + 1e-8)
    volatility = atr_pct

    btc_ma = df['BTC'].rolling(window=BTC_TREND_WINDOW).mean()
    btc_trend_on = df['BTC'] > btc_ma

    mom_arr = adj_mom[coins].values
    vol_arr = volatility[coins].values
    btc_trend_arr = btc_trend_on.values
    close_arr = df_close.values
    time_index = df.index

    # 状态机初始化
    cash = float(initial_capital)
    positions_arr = np.zeros(n_coins, dtype=float)
    coin_states = {c: {'qty': 0.0, 'cost': 0.0, 'side': None} for c in coins}
    trade_logs = []

    min_warmup = max(MOM_WINDOW, VOL_WINDOW, BTC_TREND_WINDOW)

    # 逐根 K 线流转推演
    for i in range(min_warmup, len(df)):
        current_time = time_index[i]
        prices_row = close_arr[i]

        current_equity = cash + np.dot(positions_arr, prices_row)

        current_mom = mom_arr[i]
        current_vol = vol_arr[i]
        is_btc_trend_on = btc_trend_arr[i]

        top_long_coins = []
        top_short_coins = []

        if is_btc_trend_on:
            if trade_mode in ['BOTH', 'LONG_ONLY']:
                mask = ~np.isnan(current_mom) & (current_mom > 0)
                if mask.any():
                    valid_idx = np.where(mask)[0]
                    valid_vals = current_mom[valid_idx]
                    order = np.argsort(-valid_vals, kind='stable')
                    top_long_coins = [coins[idx] for idx in valid_idx[order[:TOP_K]]]
        else:
            if trade_mode in ['BOTH', 'SHORT_ONLY']:
                mask = ~np.isnan(current_mom) & (current_mom < 0)
                if mask.any():
                    valid_idx = np.where(mask)[0]
                    valid_vals = current_mom[valid_idx]
                    order = np.argsort(valid_vals, kind='stable')
                    top_short_coins = [coins[idx] for idx in valid_idx[order[:TOP_K]]]

        # --- A. 平仓逻辑 ---
        for idx_c in range(n_coins):
            c = coins[idx_c]
            # 平多
            if positions_arr[idx_c] > 0 and c not in top_long_coins:
                sell_amount = positions_arr[idx_c]
                actual_sell_val = sell_amount * prices_row[idx_c]
                fee = actual_sell_val * FEE_RATE
                positions_arr[idx_c] = 0
                cash += (actual_sell_val - fee)

                cost = coin_states[c]['cost']
                pnl = sell_amount * (prices_row[idx_c] - cost) - fee

                trade_logs.append({
                    "time": current_time, "action": "SELL", "coin": c, "direction": "LONG", "event": "CLOSE",
                    "price": prices_row[idx_c], "amount": sell_amount, "value": actual_sell_val, "fee": fee,
                    "reason": "Signal Exit Long",
                    "target_weight": 0.0, "pnl": pnl
                })
                coin_states[c] = {'qty': 0.0, 'cost': 0.0, 'side': None}

            # 平空
            elif positions_arr[idx_c] < 0 and c not in top_short_coins:
                buy_amount = abs(positions_arr[idx_c])
                actual_buy_val = buy_amount * prices_row[idx_c]
                fee = actual_buy_val * FEE_RATE
                positions_arr[idx_c] = 0
                cash -= (actual_buy_val + fee)

                cost = coin_states[c]['cost']
                pnl = buy_amount * (cost - prices_row[idx_c]) - fee

                trade_logs.append({
                    "time": current_time, "action": "BUY", "coin": c, "direction": "SHORT", "event": "CLOSE",
                    "price": prices_row[idx_c], "amount": buy_amount, "value": actual_buy_val, "fee": fee,
                    "reason": "Signal Exit Short",
                    "target_weight": 0.0, "pnl": pnl
                })
                coin_states[c] = {'qty': 0.0, 'cost': 0.0, 'side': None}

        # --- B. 开仓逻辑 (多) ---
        if top_long_coins:
            inv_vols = [1.0 / current_vol[coin_to_idx[c]] if current_vol[coin_to_idx[c]] > 0 else 0 for c in
                        top_long_coins]
            total_inv_vol = sum(inv_vols)
            for k_, c in enumerate(top_long_coins):
                idx_c = coin_to_idx[c]
                if positions_arr[idx_c] == 0 and total_inv_vol > 0:
                    target_weight = min(inv_vols[k_] / total_inv_vol, MAX_WEIGHT)
                    target_val = current_equity * target_weight
                    buy_val = target_val / (1 + FEE_RATE) if cash >= target_val / (1 + FEE_RATE) else cash / (
                                1 + FEE_RATE)

                    if buy_val > 1.0:
                        fee = buy_val * FEE_RATE
                        buy_amount = buy_val / prices_row[idx_c]
                        positions_arr[idx_c] += buy_amount
                        cash -= (buy_val + fee)

                        coin_states[c] = {
                            'qty': buy_amount,
                            'cost': prices_row[idx_c] + (fee / buy_amount),
                            'side': 'LONG'
                        }

                        trade_logs.append({
                            "time": current_time, "action": "BUY", "coin": c, "direction": "LONG", "event": "OPEN",
                            "price": prices_row[idx_c], "amount": buy_amount, "value": buy_val, "fee": fee,
                            "reason": "Signal Entry Long",
                            "target_weight": target_weight, "pnl": np.nan
                        })

        # --- C. 开仓逻辑 (空) ---
        if top_short_coins:
            inv_vols = [1.0 / current_vol[coin_to_idx[c]] if current_vol[coin_to_idx[c]] > 0 else 0 for c in
                        top_short_coins]
            total_inv_vol = sum(inv_vols)
            for k_, c in enumerate(top_short_coins):
                idx_c = coin_to_idx[c]
                if positions_arr[idx_c] == 0 and total_inv_vol > 0:
                    target_weight = min(inv_vols[k_] / total_inv_vol, MAX_WEIGHT)
                    sell_val = current_equity * target_weight / (1 + FEE_RATE)

                    if sell_val > 1.0:
                        fee = sell_val * FEE_RATE
                        sell_amount = sell_val / prices_row[idx_c]
                        positions_arr[idx_c] -= sell_amount
                        cash += (sell_val - fee)

                        coin_states[c] = {
                            'qty': -sell_amount,
                            'cost': prices_row[idx_c] - (fee / sell_amount),
                            'side': 'SHORT'
                        }

                        trade_logs.append({
                            "time": current_time, "action": "SELL", "coin": c, "direction": "SHORT", "event": "OPEN",
                            "price": prices_row[idx_c], "amount": sell_amount, "value": sell_val, "fee": fee,
                            "reason": "Signal Entry Short",
                            "target_weight": target_weight, "pnl": np.nan
                        })

    return pd.DataFrame(trade_logs)


# ==========================================
# 3. 实盘自动化主流水线
# ==========================================
def run_live_pipeline(raw_minute_df_list):
    BEST_PARAMS = {
        'MOM_WINDOW': 24,
        'VOL_WINDOW': 24,
        'BTC_TREND_WINDOW': 120,
        'MAX_WEIGHT': 0.25,
        'TOP_K': 2
    }
    TIME_OFFSET = '0h'
    TRADE_MODE = 'LONG_ONLY'

    print("⏳ 1. 正在将分钟级数据组装为 4H 矩阵...")
    df_4h_ready = preprocess_minute_data(raw_minute_df_list, time_offset=TIME_OFFSET)

    if df_4h_ready is None or df_4h_ready.empty:
        return

    # 🔴 核心对齐：打印处理后的数据起始与截止时间日志
    start_time_str = df_4h_ready.index[0].strftime('%Y-%m-%d %H:%M:%S')
    end_time_str = df_4h_ready.index[-1].strftime('%Y-%m-%d %H:%M:%S')
    print(f"   起始: {start_time_str} | 截止: {end_time_str}")

    print("🧠 2. 正在运行状态推演机，生成全量理论 trade_logs...")
    logs_df = generate_historical_trade_logs(
        df=df_4h_ready,
        params=BEST_PARAMS,
        trade_mode=TRADE_MODE
    )

    if logs_df.empty:
        print("► 历史流转中尚未产生任何交易信号。")
        return

    # 3. 导出完整的流水日志 (这与你回测生成的 logs DataFrame 完全一致，可用于 Diff)
    output_path = "live_simulation_logs.csv"
    logs_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 全量交易流水(Ledger)已生成: {output_path}")

    # 4. 提取当下的实盘发单指令
    latest_kline_time = df_4h_ready.index[-1]
    current_actions = logs_df[logs_df['time'] == latest_kline_time]

    print(f"\n🎯 [当前截面时刻: {latest_kline_time} 实盘发单指令]")
    if current_actions.empty:
        print("   ► 当前无平仓或开仓信号，继续保持现有仓位。")
    else:
        for _, row in current_actions.iterrows():
            if row['event'] == 'CLOSE':
                print(
                    f"   🔴 平仓指令 | {row['action']:<4} {row['coin']:<4} | 方向: {row['direction']:<5} | 数量: {row['amount']:.4f} | 原因: {row['reason']}")
            elif row['event'] == 'OPEN':
                print(
                    f"   🟢 开仓指令 | {row['action']:<4} {row['coin']:<4} | 方向: {row['direction']:<5} | 目标权重: {row['target_weight'] * 100:.1f}% | 原因: {row['reason']}")


# ==========================================
# 4. 程序入口 (加载 CSV 并启动)
# ==========================================
if __name__ == "__main__":
    file_paths = [
        "kline_data/BTCUSDT_1m_2025-01-01_merged.csv",
        "kline_data/ETHUSDT_1m_2025-01-01_merged.csv",
        "kline_data/SOLUSDT_1m_2025-01-01_merged.csv",
        "kline_data/XRPUSDT_1m_2025-01-01_merged.csv",
        "kline_data/BNBUSDT_1m_2025-01-01_merged.csv",
        "kline_data/DOGEUSDT_1m_2025-01-01_merged.csv"
    ]

    raw_list = []
    print("📂 开始加载本地 CSV 数据...")

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"⚠️ 警告: 文件未找到，跳过该标的 -> {file_path}")
            continue

        try:
            base_name = os.path.basename(file_path)
            coin_name = base_name.split('_')[0].replace('USDT', '')

            df = pd.read_csv(file_path, usecols=['open_time', 'close'], engine='pyarrow')
            df['coin_name'] = coin_name

            raw_list.append(df)
            print(f"   ✅ 成功加载 {coin_name:<4} | 数据量: {len(df)} 行")
        except Exception as e:
            print(f"❌ 读取 {file_path} 失败: {e}")

    if not raw_list:
        print("❌ 错误：没有任何数据被成功加载，程序退出。请检查 kline_data 文件夹及其路径。")
    else:
        print(f"\n🚀 数据加载完毕，共 {len(raw_list)} 个标的。准备进入信号生成流水线...\n")
        print("═" * 70)
        run_live_pipeline(raw_list)