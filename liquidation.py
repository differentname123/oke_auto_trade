import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# 拉取 Coinalyze liquidation-history（小时粒度）并返回 Pandas DataFrame（timestamp, long, short）
import time
import requests
import datetime
import pandas as pd

BASE = "https://api.coinalyze.net/v1/liquidation-history"

def fetch_liquidations(api_key, symbol, from_ts, to_ts, interval="1hour", convert_to_usd=False, max_retries=5):
    """
    返回 DataFrame，列: timestamp(UTC, datetime), long (多头被爆金额), short (空头被爆金额)
    注意：API 返回字段示例为 t, l, s；这里按常见命名假定 l->long, s->short（文档示例仅给字段名）。
    """
    headers = {"api_key": api_key}  # 或者把 api_key 放到 params 中
    params = {
        "symbols": symbol,
        "interval": interval,
        "from": int(from_ts),
        "to": int(to_ts),
        "convert_to_usd": "true" if convert_to_usd else "false"
    }

    retries = 0
    while True:
        r = requests.get(BASE, params=params, headers=headers, timeout=20)
        if r.status_code == 200:
            data = r.json()
            break
        if r.status_code == 429 and retries < max_retries:
            # 遵循 Retry-After（秒）或简单指数退避
            wait = int(r.headers.get("Retry-After", min(60, 2 ** retries)))
            time.sleep(wait)
            retries += 1
            continue
        # 其他错误直接抛出（调用方可捕获）
        r.raise_for_status()

    if not data:
        return pd.DataFrame(columns=["timestamp", "long", "short"])

    # data 是一个 list，每个元素对应一个 symbol。取第一个匹配的或者全部合并（这里按传入 symbol 取第一个）
    entry = None
    if isinstance(data, list):
        # 找到 symbol 对应项（响应可能包含多个）
        for item in data:
            if item.get("symbol") == symbol:
                entry = item
                break
        if entry is None:
            entry = data[0]  # fallback
    else:
        entry = data

    history = entry.get("history", [])
    if not history:
        return pd.DataFrame(columns=["timestamp", "long", "short"])

    # history 项形如 {"t": 1610000000, "l": 123.4, "s": 56.7}
    df = pd.DataFrame(history)
    # 把时间戳转换为 UTC datetime，按秒单位
    if "t" in df.columns:
        df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
    else:
        df["timestamp"] = pd.NaT

    # 映射字段：l -> long（多头被爆），s -> short（空头被爆）
    df["long"] = df.get("l", 0).astype(float)
    df["short"] = df.get("s", 0).astype(float)

    # 最终只保留需要的列，并按时间排序（升序）
    res = df[["timestamp", "long", "short"]].sort_values("timestamp").reset_index(drop=True)
    return res

# ---------- 示例用法 ----------
if __name__ == "__main__":
    API_KEY = "0069dc05-141b-45e0-8a6f-9ac00f505117"  # 把这里换成你的 key（或在 params 中放 api_key）
    SYMBOL = "BTCUSDT_PERP.A"  # 文档示例形式；你可以换成你需要的交易对/合约代码。:contentReference[oaicite:3]{index=3}
    file_path = "kline_data/btc_liquidations_hourly.csv"
    df = pd.read_csv(file_path)


    # 拉取过去 7 天的小时数据举例
    to_ts = int(time.time())
    from_ts = to_ts - 7 * 24 * 3600

    df = fetch_liquidations(API_KEY, SYMBOL, from_ts, to_ts, interval="1hour", convert_to_usd=True)
    print(df.tail(20))    # 显示最近 20 条小时数据
    # 若需要保存为 CSV:
    df.to_csv(file_path, index=False)
