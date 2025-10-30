# LLM 驱动的量化回测框架 (v2.1 - 动态日期调整)

本项目是一个基于大型语言模型（LLM，本项目使用 DeepSeek）进行投资决策的量化回测框架。框架采用面向对象（OOP）的设计，将数据处理、因子计算、投资组合管理、策略决策和回测引擎解耦，方便扩展和维护。

**核心特色 (v2.1):**

  * **强大的因子引擎**: 框架已升级，使用基于 `xarray` 和 `BaseAlpha` 类的专业因子库（源自 `prepare_factor_data.py`），支持复杂的横截面和时间序列因子计算。
  * **配置驱动的因子**: 您可以通过 `config.yaml` 动态指定任意已注册的因子及其解释，LLM 将自动接收并使用它们进行决策。
  * **LLM 自主决策**: LLM 不仅负责选择投资标的，还能自主决定持仓数量（在设定的最大值内）和每只标的的具体权重。
  * **智能日期调整**: 框架会自动检查您的数据历史长度和所需的因子计算缓冲期 (`factor_buffer_months`)，并在必要时**自动推迟回测开始日期**，确保长周期因子在回测开始时有有效值。

## 项目结构

```
your_project_directory/
├── llm_quant_lib/           # 核心量化库
│   ├── __init__.py
│   ├── data_handler.py      # 数据加载器
│   ├── factor_definitions.py # 【新】所有因子(BaseAlpha)的计算逻辑
│   ├── factor_engine.py     # 【新】因子引擎 (调用 factor_definitions.py)
│   ├── strategy.py          # 【新】策略 (从配置动态读取因子)
│   ├── portfolio.py         # 投资组合/交易执行
│   ├── backtest_engine.py   # 【新】回测引擎 (连接策略和因子引擎)
│   └── performance.py       # 性能分析
│
├── data/                    # 数据文件存储目录
│   ├── processed/
│   │   ├── price_with_simple_returns_2016_onwards.csv # (必需)
│   │   └── ... (基准文件)
│   └── reference/
│       └── sec_code_category_grouped.csv        # (必需)
│
├── plots/                   # 保存回测结果图表
├── logs/                    # 保存 AI 决策日志和持仓记录
├── requirements.txt         # Python 依赖库列表
├── config.yaml              # 【重要】项目配置文件
└── run_backtest.py          # 【新】主运行脚本 (含日期调整逻辑)
```

## 设置步骤

1.  **克隆或下载项目。**

2.  **创建虚拟环境 (推荐)。**

3.  **安装依赖库:**

    ```bash
    # 安装基础库
    pip install -r requirements.txt 
    ```

4.  **准备数据文件:** 确保价格数据、资产池定义、以及 `config.yaml` 中引用的所有基准文件都位于 `data` 目录下对应的子文件夹中。 **确保您的价格数据 CSV 包含足够长的历史** 以满足您最长因子的计算需求。

5.  **配置 `config.yaml` (关键步骤):**

      * 检查数据库连接（如果需要）、文件路径、基准映射等。
      * **【重要】配置 `backtest` 节点**:
          * `start_date`: 您**期望**的回测开始日期。
          * `factor_buffer_months`: **因子计算所需的缓冲期（月数）**。这个值应该**大于等于**您使用的所有因子中最长回看窗口所需的月数。例如，如果最长因子需要 168 天（约 8 个月），建议设置为 `9`。
      * **【重要】配置 `strategy.llm` 节点**:
          * `factors_to_use`: **必需**。指定要计算和发送给 LLM 的因子列表。
          * `factor_explanations`: **必需**。为 `factors_to_use` 中的每个因子提供解释，供 LLM 理解。

    **`config.yaml` 示例:**

    ```yaml
    backtest:
      start_date: "2016-10-01" # 期望的回测开始日期
      end_date: "2025-08-31"
      initial_capital: 1000000
      commission_rate: 0.001
      slippage: 0.0005
      rebalance_months: 1
      # --- 【新】因子缓冲期 ---
      factor_buffer_months: 9 # 例如，最长因子需要8个月，设为9

    strategy:
      llm:
        default_universe: "All"
        default_top_n: 5
        
        # --- 必需的因子配置 ---
        factors_to_use:
          - 'momentum' # 需要 168 天 (~8个月)
          - 'breakout_quality_score'
          # ... 其他因子 ...
        
        factor_explanations:
          'momentum': '168日价格动量。值越高越好。'
          'breakout_quality_score': '突破质量分。值越高越好。'
          # ... 其他解释 ...
    ```

6.  **配置环境变量 (必需):**

      * 设置 `DEEPSEEK_API_KEY` (必需)。
      * 设置 `DB_PASSWORD` (如果使用数据库，推荐)。

## 如何运行

在项目主目录下打开终端，运行 `run_backtest.py` 脚本。

**使用 `config.yaml` 中的配置运行:**

```bash
python run_backtest.py
```

脚本将加载 `config.yaml`。`FactorEngine` 会自动计算 `factors_to_use` 列表中的所有因子，并将它们连同 `factor_explanations` 中的解释一起发送给 LLM。

**通过命令行参数覆盖配置:**
(这只会覆盖 `universe`, `top_n` 以及**期望的** `start`/`end` 日期。`factor_buffer_months` 和因子列表始终从 `config.yaml` 读取)

```bash
# 运行 equity_global，期望从 2018 年开始，允许 AI 最多选择 3 支 ETF
python run_backtest.py --universe equity_global --start 2018-01-01 --topn 3
```

## 重要提示：自动日期调整

为了确保长周期因子在回测开始时有足够的数据进行计算，`run_backtest.py` 脚本包含了一个**自动调整机制**。

  * 脚本会检查您的数据文件 (`price_data_csv`) 中**实际存在的最早日期**。
  * 它会根据这个最早日期加上您在 `config.yaml` 中设置的 `factor_buffer_months`，计算出一个**最早可行的回测开始日期**。
  * 如果您请求的 `start_date`（来自 `config.yaml` 或命令行）**早于**这个最早可行日期，脚本会**自动将回测开始日期推迟**到该可行日期，并打印一条警告信息。

**这意味着您的实际回测区间可能比您请求的要短，请务必留意运行初期的提示信息！**

## 如何扩展 (添加新因子)

新的架构使得添加自定义因子变得非常容易：

1.  **定义因子**: 在 `llm_quant_lib/factor_definitions.py` 中，创建一个继承自 `BaseAlpha` 的新类，并实现 `predict()` 方法。
2.  **注册因子**: 打开 `llm_quant_lib/factor_engine.py`，在顶部的 `FACTOR_REGISTRY` 字典中，添加一个新条目：
    ```python
    'my_new_factor_name': (YourNewFactorClass, {'param1': 10}),
    ```
3.  **使用因子**: 打开 `config.yaml`，将 `'my_new_factor_name'` 添加到 `factors_to_use` 列表，并在 `factor_explanations` 中为 AI 添加一句解释。
4.  **检查缓冲期**: 确保 `config.yaml` 中的 `factor_buffer_months` 足够覆盖您新因子的回看窗口。

完成！下次运行 `run_backtest.py` 时，框架将自动计算并使用您的新因子。