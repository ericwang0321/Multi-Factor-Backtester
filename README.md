# 📈 Quantitative Multi-Factor Backtesting System (my_llm_backtester)

## 1. 项目目标 (Project Goal)

本项目旨在构建一个**工程化、模块化、可扩展**的量化回测框架。核心目标是能够快速验证多因子选股策略（特别是 ETF 轮动或股票多因子组合），并提供可视化的分析工具。

**核心能力：**

* **多资产支持**：支持股票、ETF 等多种资产。
* **多因子架构**：内置大量传统因子及自定义因子（Eric 系列因子），支持因子组合与权重配置。
* **高性能数据**：基于 DuckDB 和 Parquet 进行本地化的高效数据存储与查询。
* **可视化交互**：提供 Streamlit 前端，用于策略参数调整、回测结果展示和因子 EDA 分析。

---

## 2. 当前进度 (Current Status)

**目前处于：阶段 2.5 - 策略与回测引擎联调优化期**

* ✅ **数据层**：已完成。从 CSV 迁移到了 Parquet + DuckDB，查询速度和存储效率大幅提升。
* ✅ **因子库**：已完成。`definitions.py` 中集成了大量研报复现因子（中信、国君、华泰）及自定义逻辑。
* ✅ **回测引擎**：已完成。支持按日循环、信号生成、持仓调整、交易成本计算。
* ✅ **策略接口**：已完成。标准化了 `on_bar` 接口，策略类 (`Strategy`) 与执行层 (`Engine`) 实现了逻辑解耦。
* ✅ **前端展示**：已完成。`app.py` 提供了策略配置和回测结果图表。
* 🔄 **待优化/进行中**：因子计算与策略执行的流水线整合（即如何更优雅地把算好的因子喂给策略）。

---

## 3. 文件结构说明 (File Directory & Usage)

### 📂 根目录 (Root)

* **`app.py`**: **[控制台/前端]**
* 基于 Streamlit 的 Web 界面。
* **作用**：你是用鼠标点点点的地方。负责接收用户输入的参数（回测时间、因子权重、资金量），调用回测引擎，并把结果画成漂亮的图表（净值曲线、最大回撤、月度收益热力图）。


* **`run_backtest.py`**: **[脚本入口]**
* 命令行版本的运行入口。
* **作用**：如果你不想打开网页，只想在后台快速跑一个策略并看 print 结果，或者为了调试代码逻辑，就运行这个文件。


* **`run_data_sync.py`**: **[数据同步]**
* **作用**：负责连接外部数据源（如 IBKR TWS API 或其他接口），下载最新数据，清洗后存入 Parquet 数据库。


* **`config.yaml`**: **[配置文件]**
* **作用**：存放全局配置，如数据存储路径、默认费率、API 端口等。避免硬编码。


* **`requirements.txt`**: **[依赖清单]**
* **作用**：列出了项目运行所需的所有 Python 库（pandas, numpy, duckdb, streamlit 等）。



### 📂 quant_core (核心逻辑包)

#### 🔹 `quant_core/data/` (数据层)

* **`query_helper.py`**: **[数据管家]**
* **作用**：全项目**唯一**的数据访问入口。它封装了 DuckDB 的 SQL 查询语句。
* **职责**：别人找它要 `get_history`（K线图），要 `get_all_symbols`（代码列表），或者要 `get_factor_values`。它负责去硬盘上的 Parquet 文件里把数据挖出来。



#### 🔹 `quant_core/factors/` (因子工厂)

* **`engine.py` (`FactorEngine`)**: **[因子计算引擎]**
* **作用**：负责调度因子的计算。
* **职责**：它知道应该去哪拿原始数据（调用 Helper），然后调用 `definitions.py` 里的公式，把原始价格变成因子值（比如把 Open/Close 变成 RSI）。


* **`definitions.py`**: **[因子公式库]**
* **作用**：存放具体的数学逻辑。
* **职责**：这里面全是具体的类（如 `RSI`, `Eric_Trend_Score`）。只管数学计算，不管数据从哪来。这里是你复现研报和存放“秘密武器”的地方。



#### 🔹 `quant_core/strategies/` (策略层)

* **`base.py`**: **[策略基类]**
* 定义了所有策略必须遵守的规则（比如必须有 `on_bar` 方法）。


* **`strategy.py` (`FactorTopNStrategy`)**: **[具体策略]**
* **作用**：你的主策略逻辑。
* **职责**：
1. **`get_required_factors()`**: 告诉引擎“我需要 Trend_Score 和 RSI”。
2. **`on_bar()`**: 每天收盘时，根据因子值给股票打分，排序，决定买前 5 名。它输出现金的目标权重。





#### 🔹 `quant_core/` (其他核心模块)

* **`backtest_engine.py`**: **[回测指挥官]**
* **作用**：整个系统的 CPU。
* **职责**：
1. 初始化资金和数据。
2. **Pre-computation**: 调用 `FactorEngine` 提前把因子算好（这是我们之前卡住的地方，现在简化了）。
3. **Event Loop**: 按天循环 (`for date in dates`)。
4. 每天问策略 (`strategy.on_bar`) 要交易信号。
5. 指挥 Portfolio 去买卖。




* **`portfolio.py`**: **[账户管家]**
* **作用**：模拟你的券商账户。
* **职责**：记录你有多少钱 (`cash`)，持有什么股票 (`positions`)，计算每天的市值 (`total_value`)，处理买卖时的佣金和滑点。


* **`performance.py`**: **[绩效分析师]**
* **作用**：回测结束后，计算成绩单。
* **职责**：计算 Sharpe Ratio, Alpha, Beta, 最大回撤等指标。



---

## 4. 后续方向与 Roadmap (Next Steps)

既然你不搞“动态因子注入”这个复杂功能了，我们可以走一条更稳健的路：

### 🛑 短期目标 (接下来 1-2 周)

1. **稳固因子计算流程**：
* 不要在回测运行时才去算因子（太慢且易出错）。
* **新建一个脚本 `run_factor_computation.py**`：每天收盘后运行一次，把所有因子算好，存入一个新的 Parquet 文件（比如 `factors_data.parquet`）。
* 这样 `BacktestEngine` 只需要“读取”因子，不需要“计算”因子。**这会让你的回测速度快 10 倍。**


2. **完善 `strategy.py**`：
* 确保你的 `FactorTopNStrategy` 能处理极端情况（比如某天因子全是 NaN，或者停牌）。


3. **丰富 Streamlit**：
* 增加一个“因子相关性分析”页面，看看你的 `Eric_Factor` 和 `Momentum` 是否相关性过高。



### 🚀 中期目标 (1-2 个月)

1. **机器学习整合 (Machine Learning)**：
* 既然你已经有了因子库，可以用 `XGBoost` 或 `LightGBM` 来训练模型，用因子预测下期收益率，替代简单的“加权打分”。


2. **实盘/模拟盘对接**：
* 利用 `ib_insync` 库，把 `Strategy` 生成的目标仓位，转换成真实的 IBKR 订单。
