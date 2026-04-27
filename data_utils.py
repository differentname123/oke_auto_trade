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

    price_df_1m = pd.concat(dfs, axis=1).sort_index().ffill()
    price_df_4h = price_df_1m.resample('4h').last().dropna()
    print(f"✅ 数据合并完成！共有 {len(price_df_4h)} 根 4H K线。包含币种: {list(price_df_4h.columns)}")

    # ==========================================
    # 🔴 涨跌幅统计：计算各标的涨跌幅与等权平均值
    # ==========================================
    if not price_df_4h.empty:
        start_time = price_df_4h.index[0]
        end_time = price_df_4h.index[-1]
        print(f"\n📅 【全局数据绝对区间】: {start_time} 至 {end_time}")
        print(f"📈 【整体区间内各标的涨跌幅 (Buy & Hold)】:")

        total_pct_change = 0.0
        num_coins = len(price_df_4h.columns)

        for c in price_df_4h.columns:
            start_price = price_df_4h[c].iloc[0]
            end_price = price_df_4h[c].iloc[-1]
            pct_change = (end_price - start_price) / start_price * 100
            total_pct_change += pct_change
            print(f"   - {c}: {pct_change:+.2f}%")

        # 计算等权平均涨跌幅
        avg_pct_change = total_pct_change / num_coins if num_coins > 0 else 0.0
        print(f"   >>> 📊 基准表现 (等权 Buy & Hold 平均): {avg_pct_change:+.2f}%")
        print("=" * 50)

    return price_df_4h