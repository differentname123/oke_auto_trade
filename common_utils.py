import json
import os

import numpy as np
import pandas as pd

def select_strategies_optimized(
    strategy_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    k: int,
    strategy_id_col: str = 'index',  # 新增参数：指定包含策略ID的列名
    count_col: str = 'capital_no_leverage',  # 新增参数：指定包含计数的列名
    penalty_scaler: float = 1.0,
    use_absolute_correlation: bool = True,
):
    """
    使用贪婪算法选择一组策略，ID在指定列中，自动调整惩罚因子。

    目标是最大化总count，同时最小化策略间的相关性。

    Args:
        strategy_df (pd.DataFrame): 包含策略ID列和count列的DataFrame。
        correlation_df (pd.DataFrame): 包含策略对及其相关性的DataFrame。
                                        需要有 'Row1', 'Row2', 'Correlation' 列。
                                        'Row1', 'Row2'的值应能匹配 strategy_df 中 strategy_id_col 的值。
        k (int): 希望选出的策略数量。
        strategy_id_col (str): strategy_df 中包含策略ID的列名。默认为 'index'。
        count_col (str): strategy_df 中包含 count 的列名。默认为 'capital_no_leverage'。
        penalty_scaler (float, optional): 自动计算的惩罚因子的缩放系数。
                                         默认为 1.0。大于1增加惩罚，小于1减少惩罚。
        use_absolute_correlation (bool, optional): 是否在计算惩罚时使用绝对相关性值。
                                                  默认为 True。

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - pd.DataFrame: 一个包含被选中策略行的DataFrame (来自原始 strategy_df)。
                            列和索引与原始 strategy_df 保持一致, 按选择顺序排序。
            - pd.DataFrame: 只包含选定策略之间相关性的新DataFrame。
                            列为 ['Row1', 'Row2', 'Correlation']。
    """

    # --- 1. 输入验证和数据准备 ---
    if strategy_id_col not in strategy_df.columns:
        raise ValueError(f"strategy_df 必须包含策略ID列: '{strategy_id_col}'")
    if count_col not in strategy_df.columns:
        raise ValueError(f"strategy_df 必须包含列: '{count_col}'")
    if not all(col in correlation_df.columns for col in ['Row1', 'Row2', 'Correlation']):
        raise ValueError("correlation_df 必须包含列: 'Row1', 'Row2', 'Correlation'")
    if k <= 0:
        empty_strategies = strategy_df.iloc[0:0]  # 返回与输入结构相同的空DF
        empty_correlations = pd.DataFrame(columns=['Row1', 'Row2', 'Correlation'])
        return empty_strategies, empty_correlations

    # 检查策略ID列是否有重复值，这可能导致问题
    if strategy_df[strategy_id_col].duplicated().any():
        print(f"警告: 策略ID列 '{strategy_id_col}' 中存在重复值。这可能影响结果的准确性。")

    # 复制以防修改原始df
    original_strat_df = strategy_df.copy()
    strat_df_internal = strategy_df.copy()
    strat_df_internal['_internal_id_str'] = strat_df_internal[strategy_id_col].astype(str).str.strip()
    strat_df_internal = strat_df_internal.set_index('_internal_id_str', drop=True)  # 使用临时字符串ID列作为索引

    corr_df = correlation_df.copy()

    corr_df['Row1'] = corr_df['Row1'].astype(str).str.strip()
    corr_df['Row2'] = corr_df['Row2'].astype(str).str.strip()

    # --- 自动计算 Penalty Factor ---
    count_series = strat_df_internal[count_col]  # 从内部DF获取count列
    if count_series.empty or count_series.isnull().all():
         print(f"警告: '{count_col}' 列为空或全是 NaN。使用默认 penalty_factor 1.0。")
         auto_penalty_factor = 1.0
    else:
         median_count = count_series.median()
         if pd.isna(median_count) or median_count == 0:
             mean_count = count_series.mean()
             if pd.isna(mean_count) or mean_count == 0:
                 print(f"警告: '{count_col}' 的中位数和均值都为 0 或 NaN。Penalty factor 可能无效。使用 1.0。")
                 median_count = 1.0
             else:
                 median_count = mean_count
         auto_penalty_factor = abs(median_count * penalty_scaler)
         print(f"自动计算 Penalty Factor 基准 (count 中位数/均值): {median_count:.2f}")
         print(f"使用的 Penalty Factor (基准 * scaler): {auto_penalty_factor:.2f}")

    # --- 构建相关性查找字典 ---
    print("正在构建相关性查找字典...")
    corr_dict = {}
    correlation_value_col = 'Correlation'
    row1_col = 'Row1'
    row2_col = 'Row2'
    # 使用处理过的字符串ID构建字典
    for row in corr_df.itertuples(index=False):
        s1_str = getattr(row, row1_col)  # 已经是字符串且已去空格
        s2_str = getattr(row, row2_col)
        corr = getattr(row, correlation_value_col)
        if use_absolute_correlation:
            corr = abs(corr)
        key = tuple(sorted((s1_str, s2_str)))
        corr_dict[key] = corr
    print("相关性查找字典构建完成。")
    print(f"字典大小 (corr_dict): {len(corr_dict)}")  # 打印大小以供检查

    def get_correlation(s1: str, s2: str, lookup_dict: dict) -> float:
        """辅助函数：从字典中查找相关性 (输入为字符串ID)"""
        if s1 == s2:
            return 1.0
        key = tuple(sorted((s1, s2)))
        value = lookup_dict.get(key, 0.0)  # 缺失相关性默认为0
        return value

    # 获取所有有效策略的字符串ID (来自内部DF的索引)
    all_strategies_str = set(strat_df_internal.index)
    if not all_strategies_str:
         print("策略DataFrame内部处理后为空，无法选择。")
         return original_strat_df.iloc[0:0], corr_df.iloc[0:0]

    strat_df_internal[count_col] = pd.to_numeric(strat_df_internal[count_col], errors='coerce')
    strat_df_internal.dropna(subset=[count_col], inplace=True)
    all_strategies_str = set(strat_df_internal.index)  # 更新有效策略集合

    if not all_strategies_str:
         print("在处理 count 列后，没有有效的策略，无法选择。")
         return original_strat_df.iloc[0:0], corr_df.iloc[0:0]

    sorted_strategies_str = strat_df_internal.sort_values(count_col, ascending=False).index.tolist()

    if not sorted_strategies_str:
         print("排序后无有效策略，无法选择。")
         return original_strat_df.iloc[0:0], corr_df.iloc[0:0]

    # --- 2. 贪婪选择 ---
    selected_strategies_str = []  # 存储选中的策略的字符串ID
    candidate_pool_str = set(sorted_strategies_str)

    print(f"开始贪婪选择，目标数量 k={k}")

    # 选择第一个策略 (字符串ID)
    first_strategy_str = sorted_strategies_str[0]
    selected_strategies_str.append(first_strategy_str)
    candidate_pool_str.remove(first_strategy_str)

    # 设置相关性阈值：如果候选策略与任一已选策略的相关性超过该阈值，则不被考虑。
    correlation_threshold = 30

    while len(selected_strategies_str) < k and candidate_pool_str:
        best_candidate_str = None
        best_score = -np.inf

        # 遍历所有候选策略 (字符串ID)
        for candidate_str in candidate_pool_str:
            # 计算候选策略与已选策略之间的最大相关性
            max_corr_with_selected = 0.0
            if selected_strategies_str:
                current_max_corr = 0.0
                for selected_strat_str in selected_strategies_str:
                    corr = get_correlation(candidate_str, selected_strat_str, corr_dict)
                    current_max_corr = max(current_max_corr, corr)
                max_corr_with_selected = current_max_corr

            # 如果候选策略与任一已选策略的相关性超过阈值，则跳过该候选策略
            if max_corr_with_selected > correlation_threshold:
                continue

            candidate_count = strat_df_internal.loc[candidate_str, count_col]

            # 计算得分：在 count 的基础上扣除相关性惩罚
            score = candidate_count - auto_penalty_factor * max_corr_with_selected

            # 更新最佳候选
            if score > best_score:
                best_score = score
                best_candidate_str = candidate_str

        if best_candidate_str is None:
            print(f"  在第 {len(selected_strategies_str) + 1} 步无法找到合适的候选策略（满足相关性阈值要求），停止选择。")
            break

        # 添加最佳候选者 (字符串ID)
        selected_strategies_str.append(best_candidate_str)
        candidate_pool_str.remove(best_candidate_str)
        candidate_count = strat_df_internal.loc[best_candidate_str, count_col]
        original_id = strat_df_internal.loc[best_candidate_str, strategy_id_col]  # 获取原始ID用于打印
        # 计算并打印相关信息
        final_max_corr = 0.0
        if len(selected_strategies_str) > 1:
            current_max_corr = 0.0
            for s_str in selected_strategies_str[:-1]:
                corr = get_correlation(best_candidate_str, s_str, corr_dict)
                current_max_corr = max(current_max_corr, corr)
            final_max_corr = current_max_corr

    # --- 3. 从原始 DataFrame 中提取选定的策略 ---
    print(f"选择完成，共选出 {len(selected_strategies_str)} 个策略。")

    # 根据处理后的字符串ID筛选原始策略
    selected_mask = original_strat_df[strategy_id_col].astype(str).str.strip().isin(selected_strategies_str)
    selected_strategies_df_unordered = original_strat_df[selected_mask].copy()

    # 保证输出的顺序与选择顺序一致
    if not selected_strategies_df_unordered.empty and selected_strategies_str:
        id_map = selected_strategies_df_unordered.set_index(selected_strategies_df_unordered[strategy_id_col].astype(str).str.strip())
        selected_strategies_df = id_map.loc[selected_strategies_str].copy()
        selected_strategies_df.reset_index(drop=True, inplace=True)
    else:
        selected_strategies_df = selected_strategies_df_unordered

    selected_strategies_set_str = set(selected_strategies_str)

    corr_filter_mask = corr_df[row1_col].isin(selected_strategies_set_str) & \
                       corr_df[row2_col].isin(selected_strategies_set_str)
    selected_correlation_df = corr_df[corr_filter_mask].copy()

    # 尝试将相关性 DataFrame 中的 Row1/Row2 恢复为原始类型
    original_id_dtype = original_strat_df[strategy_id_col].dtype
    if not pd.api.types.is_string_dtype(original_id_dtype):
        id_str_to_original_map = {}
        for _idx, row in original_strat_df.drop_duplicates(subset=[strategy_id_col], keep='first').iterrows():
            id_str = str(row[strategy_id_col]).strip()
            id_orig = row[strategy_id_col]
            id_str_to_original_map[id_str] = id_orig

        try:
            selected_correlation_df[row1_col] = selected_correlation_df[row1_col].map(id_str_to_original_map)
            selected_correlation_df[row2_col] = selected_correlation_df[row2_col].map(id_str_to_original_map)
            selected_correlation_df.dropna(subset=[row1_col, row2_col], inplace=True)
            selected_correlation_df[row1_col] = selected_correlation_df[row1_col].astype(original_id_dtype)
            selected_correlation_df[row2_col] = selected_correlation_df[row2_col].astype(original_id_dtype)
            print(f"相关性DataFrame中的ID已尝试恢复为原始类型: {original_id_dtype}")
        except Exception as e:
            print(f"警告：尝试将相关性DF中的ID转回原始类型时出错: {e}。将返回字符串形式的ID。")

    # --- 5. 返回结果 ---
    return selected_strategies_df, selected_correlation_df

def compute_signal(df, col_name):
    """
    根据历史行情数据(df)和指定信号名称(col_name)，生成交易信号和对应目标价格。

    说明：
      - 信号的目标价格不再使用 clip() 调整，
        而是在判断目标价格是否落在当前bar的 low 和 high 区间内，
        若目标价格超出区间，则认为信号无效（不产生信号）。
      - 当前支持的信号类型包括：
          - abs: 绝对百分比突破信号
              示例："abs_20_2_long" (20周期内最低价向上2%多头突破)
          - relate: 相对区间百分比突破信号
              示例："relate_20_50_short" (20周期区间顶部向下50%空头突破)
          - donchian: 唐奇安通道突破信号（实时价格触发优化）
              示例："donchian_20_long" (20周期最高价向上突破多头信号)
          - boll: 布林带信号
              示例："boll_20_2_long" 或 "boll_20_2_short"
          - macross: MACROSS 信号 (双均线交叉信号)
              示例："macross_10_20_long"
          - rsi: RSI 超买超卖反转信号
              示例："rsi_14_70_30_long"
          - macd: MACD交叉信号
              示例："macd_12_26_9_long"
          - cci: 商品通道指数超买超卖反转信号
              示例："cci_20_short"
              （若传入参数不足，则采用默认常数0.015）
          - atr: ATR波动率突破信号
              示例："atr_14_long"

    参数:
      df: pandas.DataFrame，必须包含以下列：
          "close": 收盘价
          "high": 最高价
          "low": 最低价
      col_name: 信号名称，格式如 "signalType_param1_param2_..._direction"

    返回:
      tuple:
        - signal_series: pandas.Series(bool)，当满足信号条件时为 True，否则为 False。
        - trade_price_series: pandas.Series(float)，信号触发时建议的目标交易价格；
          若目标价格超出当前bar的 low 和 high，则不产生信号。
    """

    parts = col_name.split('_')
    signal_type = parts[0]
    direction = parts[-1]

    if signal_type == 'abs':
        period = int(parts[1])
        abs_value = float(parts[2]) / 100
        if direction == "long":
            min_low_series = df['low'].shift(1).rolling(period).min()
            target_price = (min_low_series * (1 + abs_value)).round(4)
            signal_series = df['high'] > target_price
        else:
            max_high_series = df['high'].shift(1).rolling(period).max()
            target_price = (max_high_series * (1 - abs_value)).round(4)
            signal_series = df['low'] < target_price

        # 检查目标价格是否落在当前bar的low与high之间
        valid_price = (target_price >= df['low']) & (target_price <= df['high'])
        signal_series = signal_series & valid_price
        trade_price_series = target_price  # 直接使用计算的目标价格

        # 可选调试记录
        df['target_price'] = target_price
        df['signal_series'] = signal_series
        df['trade_price_series'] = trade_price_series
        return signal_series, trade_price_series

    elif signal_type == 'relate':
        period = int(parts[1])
        percent = float(parts[2]) / 100
        min_low_series = df['low'].shift(1).rolling(period).min()
        max_high_series = df['high'].shift(1).rolling(period).max()
        if direction == "long":
            target_price = (min_low_series + percent * (max_high_series - min_low_series)).round(4)
            signal_series = df['high'] > target_price
        else:
            target_price = (max_high_series - percent * (max_high_series - min_low_series)).round(4)
            signal_series = df['low'] < target_price

        valid_price = (target_price >= df['low']) & (target_price <= df['high'])
        signal_series = signal_series & valid_price
        trade_price_series = target_price
        return signal_series, trade_price_series

    elif signal_type == 'donchian':
        period = int(parts[1])
        if direction == "long":
            highest_high = df['high'].shift(1).rolling(period).max()
            signal_series = df['high'] > highest_high
            target_price = highest_high
        else:
            lowest_low = df['low'].shift(1).rolling(period).min()
            signal_series = df['low'] < lowest_low
            target_price = lowest_low

        valid_price = (target_price >= df['low']) & (target_price <= df['high'])
        signal_series = signal_series & valid_price
        trade_price_series = target_price.round(4)
        return signal_series, trade_price_series

    elif signal_type == 'boll':
        period = int(parts[1])
        std_multiplier = float(parts[2])
        ma = df['close'].rolling(window=period, min_periods=period).mean()
        std_dev = df['close'].rolling(window=period, min_periods=period).std()
        upper_band = (ma + std_multiplier * std_dev).round(4)
        lower_band = (ma - std_multiplier * std_dev).round(4)
        if direction == "long":
            signal_series = (df['close'].shift(1) < lower_band.shift(1)) & (df['close'] >= lower_band)
        else:  # short
            signal_series = (df['close'].shift(1) > upper_band.shift(1)) & (df['close'] <= upper_band)
        # 此处直接返回收盘价作为交易价格
        return signal_series, df["close"]

    elif signal_type == 'macross':
        fast_period = int(parts[1])
        slow_period = int(parts[2])
        fast_ma = df["close"].rolling(window=fast_period, min_periods=fast_period).mean().round(4)
        slow_ma = df["close"].rolling(window=slow_period, min_periods=slow_period).mean().round(4)
        if direction == "long":
            signal_series = (fast_ma.shift(1) < slow_ma.shift(1)) & (fast_ma >= slow_ma)
        else:
            signal_series = (fast_ma.shift(1) > slow_ma.shift(1)) & (fast_ma <= slow_ma)
        trade_price = df["close"]
        return signal_series, trade_price

    elif signal_type == 'rsi':
        period = int(parts[1])
        overbought = int(parts[2])
        oversold = int(parts[3])
        delta = df['close'].diff(1).astype(np.float32)
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        # 防止除0错误
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        if direction == "long":
            signal_series = (rsi.shift(1) < oversold) & (rsi >= oversold)
        else:
            signal_series = (rsi.shift(1) > overbought) & (rsi <= overbought)
        return signal_series, df['close']

    elif signal_type == 'macd':
        fast_period, slow_period, signal_period = map(int, parts[1:4])
        fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        if direction == "long":
            signal_series = (macd_line.shift(1) < signal_line.shift(1)) & (macd_line >= signal_line)
        else:
            signal_series = (macd_line.shift(1) > signal_line.shift(1)) & (macd_line <= signal_line)
        return signal_series, df["close"]

    elif signal_type == 'cci':
        period = int(parts[1])
        # 若参数不足，采用默认常数0.015
        if len(parts) == 3:
            constant = 0.015
        else:
            constant = float(parts[2]) / 100
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma = tp.rolling(period).mean()
        md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        cci = (tp - ma) / (constant * md)
        if direction == "long":
            signal_series = (cci.shift(1) < -100) & (cci >= -100)
        else:
            signal_series = (cci.shift(1) > 100) & (cci <= 100)
        return signal_series, df['close']

    elif signal_type == 'atr':
        period = int(parts[1])
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        atr_ma = atr.rolling(period).mean()
        if direction == "long":
            signal_series = (atr.shift(1) < atr_ma.shift(1)) & (atr >= atr_ma)
        else:
            signal_series = (atr.shift(1) > atr_ma.shift(1)) & (atr <= atr_ma)
        return signal_series, df['close']

    else:
        raise ValueError(f"未知信号类型: {signal_type}")

def get_config(key):
    """
    从 config.json 文件中获取指定字段的值
    :param key: 配置字段名
    :return: 配置字段值
    """
    # 获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接 config.json 文件的绝对路径
    config_file = os.path.join(base_dir, 'config.json')

    # 检查 config.json 文件是否存在
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件 '{config_file}' 不存在，请检查文件路径。")

    # 读取配置文件
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"配置文件 '{config_file}' 格式错误: {e}")

    # 获取指定字段的值
    if key not in config_data:
        raise KeyError(f"配置文件中缺少字段: {key}")

    return config_data[key]