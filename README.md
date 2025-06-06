# oke_auto_trade
# 项目实现概述

> 本项目旨在**自动化寻找并评估量化交易策略**，并将最佳策略部署到实盘。整体流程可拆解为 6 个阶段，每一步都对应独立、可复用的模块与脚本。

---

## 1. 定义因变量  
- **核心任务**：从原始 K 线或 Tick 数据提取多种技术指标（如 MACD、RSI 等），并将其标准化为可组合的交易信号。  
- **关键代码**：`common_utils.compute_signal`

---

## 2. 构造策略空间  
- **做法**：对所有信号做**穷举组合**，每一种组合即一条候选策略。  
- **加速手段**：采用 *6 个月滑动窗口* 回测，显著降低内存与计算成本。  
- **关键代码**：`bread_through_strategy_fast_check.example`

---

## 3. 快速筛选  
- **过滤规则**：  
  1. 最大连续亏损 < 30  
  2. 总收益 > 0  
- 该步骤可在实现没轮时间段 ≈10% 的参数空间内完成评估，大幅缩短搜索时间。  
- **关键代码**：`bread_through_strategy_fast_check.fast_check`

---

## 4. 详细回测  
- **对象**：仅对通过快速筛选的策略进行**全量历史数据**回测。  
- **输出**：20+ 维度的绩效指标（如周度收益、无杠杆最终收益等）。  
- **关键代码**：`bread_through_strategy_ga_target.validation`

---

## 5. 组合优化  
- **方法**：  
  1. 计算策略两两相关系数  
  2. 采用贪心算法，优先选取高收益且低相关的策略  
- **目标**：构建风险分散、收益领先的策略组合。  
- **关键代码**：`bread_through_strategy_compute_corr.compute_corr` 'common_utils.select_strategies_optimized'

---

## 6. 实盘部署  
- **接口**：对接 OKX API，完成  
  - 分钟级行情拉取  
  - WebSocket 实时订阅  
  - 订单执行  
- **关键代码**：`run.run_instrument`

---

## 目录结构（节选）