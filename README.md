# oke_auto_trade 项目文档

本项目旨在**自动化寻找并评估量化交易策略**，并将最佳策略部署到实盘。整体流程可拆解为 6 个阶段，每一步都对应独立、可复用的模块与脚本。

---

## 一、项目概述

项目通过以下主要步骤实现量化交易策略的全流程：
1. 数据获取
2. 策略回测
3. 策略组合优化
4. 实盘操作部署

各阶段均有专门的模块与脚本支持，并通过文件持久化保存中间结果，以便后续分析和策略优选。

---

## 二、模块说明

### 2.1 数据获取

- **模块名称**: `get_feature_op.download_data`
- **功能描述**:
  - 从 OKX 交易所下载加密货币的历史 K 线数据；
  - 根据预设的配置批量获取不同交易对、时间粒度与数据量的 K 线数据，并保存为 CSV 文件。
- **副作用**:
  - **文件创建**:
    - 在 `kline_data` 目录下，如子目录不存在则创建之，然后保存 CSV 文件。
  - **CSV 文件内容**:
    - 经过初步处理（如将时间戳转换为北京时间、添加未来涨跌幅目标变量等）。
  - **文件命名**:
    - 命名格式示例：  
      `origin_data_1m_100000_BTC-USDT-SWAP_2024-01-24.csv`

---

### 2.2 策略回测

#### 2.2.1 策略过滤

- **模块**: `bread_through_strategy_fast_check_op.py`
- **功能描述**:
  - 通过半年轮动筛选出满足基础条件的候选策略。
- **副作用**:
  - **文件创建**:
    - 在 `temp` 文件夹下生成多个批次的候选文件；
    - 每半年的结果汇总至文件，例如：
      ```
      temp_back/all_files_{year}_1m_5000000_{inst_id}_short_donchian_1_20_1_relate_400_1000_100_1_40_6_cci_1_2000_1000_1_2_1_atr_1_3000_3000_boll_1_3000_100_1_50_2_rsi_1_1000_500_abs_1_100_100_40_100_1_macd_300_1000_50_macross_1_3000_100_1_3000_100__{is_reverse}.parquet
      ```

#### 2.2.2 好策略全量回测

- **模块**: `bread_through_strategy_ga_target.py`
- **功能描述**:
  - 对筛选后的好策略进行全量历史数据回测，并计算 20+ 个维度的绩效指标。
- **副作用**:
  - **文件创建**:
    - 生成回测结果文件，路径为：
      ```
      os.path.join("temp_back", f"statistic_results_final_{inst_id}_{is_reverse}.parquet")
      ```

#### 2.2.3 注意事项

- **参数调整**:
  - 回测中需修改 `is_reverse` 参数以区分正向与反向回测；
- **代码修改位置**:
  - `bread_through_strategy_fast_check.py:50`
  - `bread_through_strategy_ga_target.py:755`
  - `bread_through_strategy_ga_target.py:659`

---

### 2.3 策略组合优化

#### 2.3.1 过滤相似子策略

- **模块**: `bread_through_strategy_compute_corr.filter_similar_strategy_all`
- **功能描述**:
  - 针对同一方向（做多、做空）的策略：
    - 进行合并；
    - 过滤掉表现较差的策略；
    - 计算策略间两两相关系数，过滤掉相关性过高的策略。
- **副作用**:
  - **文件生成**:
    - 输出文件路径为：  
      `temp_back/{inst_id}_{is_reverse}_all_filter_similar_strategy.parquet`

#### 2.3.2 好策略组合

- **模块**: `bread_through_strategy_choose_zuhe.choose_zuhe_beam_opt`
- **功能描述**:
  - 对全量策略进行组合优化；
  - 计算各组合的表现打分、每周胜率、每周平均亏损等指标，最终选出最优策略组合。
- **副作用**:
  - **文件生成**:
    - 输出文件路径格式为：  
      `out_dir / f"result_elements_{inst}_adp_{typ}_op.parquet"`
  - 最终不同组合的得分情况如下:https://drive.google.com/drive/folders/18V24Eajn1sP5-W845w5GLF4n7o9A4tDT?usp=drive_link

---

### 2.4 实盘操作

#### 2.4.1 程序启动

- **模块**: `watchdog.py`
- **功能描述**:
  - 实现每天定时重启，确保资金信息更新并解决部分系统问题。

#### 2.4.2 实际操作

- **模块**: `run.py`
- **功能描述**:
  - 加载每种币种中表现最佳的文件，选出最优策略组合，执行实盘交易。

---

## 三、策略实现流程

本节描述从数据原始因变量定义到策略实盘部署的全流程步骤。

### 3.1 定义因变量

- **核心任务**:
  - 从原始 K 线或 Tick 数据中提取多种技术指标（如 MACD、RSI 等），并标准化为可组合的交易信号。
- **关键代码**:
  - `common_utils.compute_signal`

### 3.2 构造策略空间

- **方法**:
  - 对所有交易信号进行穷举组合，每一种组合即为一条候选策略。
- **优化手段**:
  - 采用 6 个月滑动窗口回测，显著降低内存占用与计算成本。
- **关键代码**:
  - `bread_through_strategy_fast_check.example`

### 3.3 快速筛选

- **过滤规则**:
  1. 最大连续亏损小于 30；
  2. 总收益大于 0。
- **说明**:
  - 在约 10% 的参数空间内完成评估，大幅缩短搜索时长。
- **关键代码**:
  - `bread_through_strategy_fast_check.fast_check`

### 3.4 详细回测

- **对象**:
  - 对通过快速筛选的候选策略进行全量回测。
- **输出**:
  - 输出 20+ 个维度的绩效指标，如周度收益、无杠杆最终收益等。
- **关键代码**:
  - `bread_through_strategy_ga_target.validation`

### 3.5 策略组合优化

- **方法**:
  1. 计算各策略之间的相关系数；
  2. 采用贪心算法，选取高收益且低相关性的策略，构建组合。
- **目标**:
  - 构建风险分散且收益领先的策略组合。
- **关键代码**:
  - `bread_through_strategy_choose_zuhe.py`

### 3.6 实盘部署

- **接口描述**:
  - 对接 OKX API，实现以下功能：
    - 分钟级行情数据拉取；
    - WebSocket 实时订阅；
    - 订单执行。
- **关键代码**:
  - `run.run_instrument`

---

## 四、目录结构（节选）