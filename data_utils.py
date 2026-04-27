import pandas as pd
import os


# ==========================================
# 数据解析、合成与基础统计模块
# ==========================================
def load_and_preprocess_data(file_list):
    print("⏳ 正在解析并合并数据...")
    dfs = []
    for file in file_list:
        basename = os.path.basename(file).split('_')
        coin_main, coin_sub = basename[0], basename[1]

        df = pd.read_csv(file)
        df['open_time'] = pd.to_datetime(df['open_time'])
        df.set_index('open_time', inplace=True)

        df_main = df[['close_main']].rename(columns={'close_main': coin_main})
        df_sub = df[['close_sub']].rename(columns={'close_sub': coin_sub})

        dfs.extend([df_main, df_sub])

    # 1. 基础合并
    price_df_1m_raw = pd.concat(dfs, axis=1).sort_index()
    price_df_1m_raw = price_df_1m_raw.loc[:, ~price_df_1m_raw.columns.duplicated()]

    # ==========================================
    # 🎯 核心修正：锁定全局共有区间 (Intersection)
    # ==========================================
    # 获取每个币种的第一个有效值(非NaN)时间和最后一个有效值时间
    # 大家的“最大”起步时间 -> 公共起点
    common_start = max([price_df_1m_raw[c].first_valid_index() for c in price_df_1m_raw.columns])
    # 大家的“最小”结束时间 -> 公共终点
    common_end = min([price_df_1m_raw[c].last_valid_index() for c in price_df_1m_raw.columns])

    # 截取纯净的共有区间
    price_df_1m = price_df_1m_raw.loc[common_start:common_end].copy()

    print(f"✅ 成功锁定公共时间窗口: {common_start} 至 {common_end}")

    # ==========================================
    # 🛠️ 数据质量与填充统计 (共有区间内)
    # ==========================================
    total_1m_rows = len(price_df_1m)
    missing_counts = price_df_1m.isna().sum()

    print(f"\n🔍 【数据质量检测：共有区间内填充统计】 (总 1m K线数: {total_1m_rows})")
    for c in price_df_1m.columns:
        missing = missing_counts[c]
        fill_ratio = (missing / total_1m_rows) * 100

        # 此时的 fill_ratio 才是真正反映流动性和交易断档的真实指标
        alert_flag = " ⚠️ [流动性差/频繁断档]" if fill_ratio > 5.0 else ""
        print(f"   - {c:8s}: 真实缺失/需填充 {missing:>8d} 条 | 填充率 {fill_ratio:>6.2f}%{alert_flag}")
    print("-" * 50)

    # 2. 填充与降频操作
    # 现在的填充只会发生在共有区间内部的断档，非常安全
    price_df_1m = price_df_1m.ffill()
    # 因为首尾非共有的 NaN 已经被切掉，这里 resample 后产生的 4H 也是完全对齐的
    price_df_4h = price_df_1m.resample('4h').last()

    # ==========================================
    # 🔴 涨跌幅与风险（最大回撤）统计
    # ==========================================
    if not price_df_4h.empty:
        print(f"\n📈 【共有区间内各标的表现 (Buy & Hold)】:")

        roll_max = price_df_4h.cummax()
        drawdowns = (price_df_4h - roll_max) / roll_max
        max_drawdowns_pct = drawdowns.min() * 100

        total_pct_change = 0.0
        num_coins = len(price_df_4h.columns)

        for c in price_df_4h.columns:
            start_price = price_df_4h[c].iloc[0]
            end_price = price_df_4h[c].iloc[-1]

            pct_change = (end_price - start_price) / start_price * 100
            total_pct_change += pct_change
            mdd = max_drawdowns_pct[c]

            print(f"   - {c:8s}: 涨跌幅 {pct_change:>8.2f}%  |  最大回撤 {mdd:>8.2f}%")

        avg_pct_change = total_pct_change / num_coins if num_coins > 0 else 0.0
        avg_mdd = max_drawdowns_pct.mean() if num_coins > 0 else 0.0

        print("-" * 50)
        print(f"   >>> 📊 基准表现 (等权 Buy & Hold):")
        print(f"            平均涨跌幅: {avg_pct_change:+.2f}%")
        print(f"            平均最大回撤: {avg_mdd:.2f}%")
        print("=" * 50)

    return price_df_4h