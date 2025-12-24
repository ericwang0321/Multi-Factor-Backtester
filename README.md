# 📈 Quantitative Multi-Factor Backtesting System

## 1. 项目愿景 (Project Vision)

本项目旨在构建一个**高性能、工程化、模块化**的量化回测框架。核心目标是支持多因子选股策略（Multi-Factor Selection）与 ETF 轮动策略的快速验证与迭代。

**核心架构特点：**

* 🚀 **离线预计算 (Pre-computation)**：彻底分离“因子计算”与“策略回测”。通过 `run_factor_computation.py` 实现因子的全量向量化计算与持久化存储，回测速度提升 **100x**。
* 🏭 **工厂模式与自动注册 (Factory & Registry)**：采用工业级设计模式。新增策略只需添加装饰器 `@register_strategy`，无需修改引擎代码，真正做到 **开闭原则 (Open/Closed Principle)**。
* ⚙️ **配置解耦 (Config Decoupling)**：采用层级配置系统（Base + Environment），支持回测与实盘使用完全独立的参数集，防止环境污染。
* 💾 **高性能数据层**：基于 **DuckDB** 和 **Parquet** 构建本地数据仓库，支持海量行情与因子数据的秒级查询。
* ⚡ **实盘无缝切换**：采用适配器模式，通过 `LiveDataBridge` 复用回测策略逻辑，实现从回测到实盘的零代码修改迁移。

---

## 2. 当前进度 (Current Status)

**目前处于：阶段 4.5 - 架构重构与深度扩展 (Refactoring & Extension)**

* ✅ **策略工厂重构**：移除了硬编码的 `if/else` 判断，实现策略类的自动注册与参数自动注入。
* ✅ **配置系统重构**：实现了 `base.yaml` (基础设施) 与 `backtest.yaml`/`live.yaml` (环境参数) 的分离与递归合并。
* ✅ **数据仓库**：DuckDB + Parquet 架构，支持增量同步 IBKR/外部数据。
* ✅ **因子工厂**：`run_factor_computation.py` 支持 Xarray 全向量化计算与增量更新。
* ✅ **策略体系**：实现了 `LinearWeightedStrategy`（多因子线性加权 + 自动 Z-Score）。
* ✅ **回测引擎**：纯粹的事件驱动撮合引擎，支持滑点、佣金、多标的组合。
* ✅ **实盘/模拟盘对接**：基于 `ib_insync` 实现 IBKR 对接。支持自动数据预热、实时因子计算与自动下单。

---

## 3. 系统架构与数据流 (Architecture & Workflow)

本框架采用**产线分离**与**双模式运行**的设计思想：

### 模式 A: 离线回测 (Backtest)

```mermaid
graph LR
    A[数据源/IBKR] -->|run_data_sync.py| B(原始行情 Parquet)
    B -->|run_factor_computation.py| C(因子数据 Parquet)
    Config[config/backtest.yaml] -->|Load Params| F{策略工厂 Factory}
    C -->|Load Offline| D[策略 Strategy]
    F -.->|Create| D
    B -->|Load Price| E[回测引擎 BacktestEngine]
    D -->|Signal| E
    E -->|Result| F_Res[绩效分析/Streamlit]

```

### 模式 B: 实盘/模拟盘 (Live Trading)

```mermaid
graph LR
    A[IBKR TWS/Gateway] <-->|ib_connector| B(实时数据流)
    B -->|data_bridge| C{LiveDataBridge}
    C -- 1. fetch history --> D[数据预热 Warm-up]
    C -- 2. calc on-the-fly --> E[实时因子计算]
    Config[config/live.yaml] -->|Load Params| Fact{策略工厂 Factory}
    Fact -.->|Create| F[策略 Strategy]
    E -->|Feed| F
    F -->|Target Weights| G[交易员 LiveTrader]
    G -- 1. Diff Calc --> H[计算仓位差额]
    H -- 2. Place Order --> A

```

---

## 4. 文件结构说明 (File Directory)

### 📂 根目录 (Root)

* **`run_backtest.py`**: **[回测入口]**
* **作用**：读取配置文件，通过工厂创建策略实例，加载离线因子并运行回测。**无需修改此文件即可运行新策略。**


* **`run_live_strategy.py`**: **[实盘指挥官]**
* **作用**：实盘/模拟盘的主入口。连接 TWS -> 调用 Bridge 获取数据 -> 计算信号 -> 执行下单。


* **`run_factor_computation.py`**: **[因子工厂]**
* **作用**：读取全量行情，批量计算因子，并保存为 Parquet 文件。


* **`test_live_connection.py`**: **[连接测试]**
* **作用**：验证 IBKR 端口连接、数据权限及下单功能的健康检查脚本。


* **`app.py`**: **[Web 前端]**
* **作用**：Streamlit 可视化界面，用于数据探索和简易回测。



### 📂 config (配置中心) **[New]**

* **`base.yaml`**: **[基础设施配置]**
* 存放不随环境变化的全局路径（如数据存储路径、Universe 文件路径）。


* **`backtest.yaml`**: **[回测专用配置]**
* 存放回测时间段、初始资金、以及**回测时的策略参数**（如因子权重）。


* **`live.yaml`**: **[实盘专用配置]**
* 存放实盘交易账户ID、实盘更严格的风控参数、以及实盘生效的策略模型路径。



### 📂 quant_core (核心逻辑包)

#### 🔹 `quant_core/strategies/` (策略库)

* **`base.py`**: 包含 `BaseStrategy` 基类以及 **核心工厂逻辑 (`create_strategy_instance`, `@register_strategy`)**。
* **`rules.py`**: 线性策略实现 (`LinearWeightedStrategy`)。
* **`__init__.py`**: 负责暴露工厂接口并导入策略模块以触发注册。

#### 🔹 `quant_core/live/` (实盘模块)

* **`ib_connector.py`**: 基于 `ib_insync` 的 TWS 连接器。
* **`data_bridge.py`**: 数据适配器，负责“回测-实盘”数据格式的统一。
* **`trader.py`**: 交易执行器，负责计算仓位差额并下单。

---

## 5. 开发者指南：如何新增策略 (Developer Guide)

本框架采用**全自动注册机制**。假设你想新增一个 **深度学习策略 (Deep Learning Strategy)**，你只需要关注策略本身的逻辑，**无需修改 `run_backtest.py**`。

### 第一步：创建策略文件

在 `quant_core/strategies/` 下新建 `dl_strategy.py`。
使用 `@register_strategy` 装饰器给它起个名字（例如 `'dl_model'`）。

```python
# quant_core/strategies/dl_strategy.py
import pandas as pd
from typing import List
from .base import BaseStrategy, register_strategy  # <--- 引入装饰器

# 1. 注册策略 (key: 'dl_model')
@register_strategy('dl_model')
class DeepLearningStrategy(BaseStrategy):
    
    # 2. 初始化 (注意：必须接收 **kwargs 并传给 super)
    def __init__(self, name, model_path, feature_cols, top_k=5, **kwargs):
        super().__init__(name, top_k=top_k, **kwargs) # 自动处理风控参数
        self.model_path = model_path
        self.feature_cols = feature_cols
        # load_model(self.model_path) ...
        print(f"[{name}] DL模型已加载: {model_path}")
    
    # 3. 声明所需因子 (系统会自动去加载数据)
    def get_required_factors(self) -> List[str]:
        return self.feature_cols
    
    # 4. 核心逻辑
    def calculate_scores(self, factor_df: pd.DataFrame) -> pd.Series:
        # data = factor_df[self.feature_cols]
        # scores = self.model.predict(data)
        return pd.Series() # 返回打分

```

### 第二步：确保模块被导入

打开 `quant_core/strategies/__init__.py`，添加一行 import。
*这一步是为了让 Python 解释器读到你的文件，从而触发装饰器注册。*

```python
# quant_core/strategies/__init__.py
from .base import create_strategy_instance, STRATEGY_REGISTRY
from . import rules
from . import dl_strategy  # <--- 新增这一行

```

### 第三步：修改配置文件

在 `config/backtest.yaml` (或 `live.yaml`) 中，修改 `type` 并添加对应的参数块。

```yaml
strategy:
  # 1. 对应 @register_strategy('dl_model')
  type: 'dl_model'  

  common:
    name: 'LSTM_Alpha_v1'
    top_k: 5
    risk:
      stop_loss_pct: 0.05

  # 2. 工厂会自动把这个块里的参数传给你的 __init__
  dl_model_params:
    model_path: 'models/lstm_v1.pth'
    feature_cols: ['alpha001', 'volatility_20d', 'rsi']

```

**完成！** 直接运行 `python run_backtest.py` 即可。工厂会自动识别并加载你的新策略。

---

## 6. 快速开始 (Quick Start)

### 场景一：离线回测 (Backtest)

1. **准备配置**：编辑 `config/backtest.yaml`，设置你想要的策略参数。
2. **数据准备**：确保 `data/processed` 下有 parquet 数据。
3. **运行**：
```bash
python run_backtest.py

```


*程序将自动读取配置、通过工厂创建策略、自动加载所需因子、跑完回测并保存结果图表。*

### 场景二：实盘/模拟盘交易 (Live Trading)

1. **连接**：打开 TWS/Gateway，开启 API 端口 (默认 7497)。
2. **配置**：编辑 `config/live.yaml`，确认实盘风控参数。
3. **测试**：
```bash
python test_live_connection.py

```


4. **启动**：
```bash
python run_live_strategy.py

```



---

## 7. 后续规划 (Roadmap)

### 🚀 短期目标 (Short-term)

1. **实盘定时任务**：引入 `APScheduler`，实现开盘自动连接、收盘自动断开。
2. **更多因子**：录入 WorldQuant Alpha 101 剩余因子。

### 🌟 中期目标 (Mid-term)

1. **机器学习集成**：完善 `DeepLearningStrategy` 模板，支持 PyTorch 模型的热加载。
2. **Web 看板升级**：将 Streamlit 升级为实盘监控台，实时显示 PnL 和 Log。