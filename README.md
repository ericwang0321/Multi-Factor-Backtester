# LLM 驱动的量化回测框架 (v2.0 - Xarray 引擎版)

本项目是一个基于大型语言模型（LLM，本项目使用 DeepSeek）进行投资决策的量化回测框架。框架采用面向对象（OOP）的设计，将数据处理、因子计算、投资组合管理、策略决策和回测引擎解耦，方便扩展和维护。

**核心特色 (v2.0):**

  * **强大的因子引擎**: 框架已升级，使用基于 `xarray` 和 `BaseAlpha` 类的专业因子库（源自 `prepare_factor_data.py`），支持复杂的横截面和时间序列因子计算。
  * **配置驱动的因子**: 您现在可以通过 `config.yaml` 动态指定任意已注册的因子及其解释，LLM 将自动接收并使用它们进行决策。
  * **LLM 自主决策**: LLM 不仅负责选择投资标的，还能自主决定持仓数量（在设定的最大值内）和每只标的的具体权重。

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
└── run_backtest.py          # 主运行脚本
```

## 设置步骤

1.  **克隆或下载项目。**

2.  **创建虚拟环境 (推荐)。**

3.  **安装依赖库:**

    ```bash
    # 安装基础库
    pip install -r requirements.txt 
    ```

4.  **准备数据文件:** 确保价格数据、资产池定义、以及 `config.yaml` 中引用的所有基准文件都位于 `data` 目录下对应的子文件夹中。

5.  **配置 `config.yaml` (关键步骤):**

      * 检查数据库连接（如果需要）、文件路径、基准映射等。
      * **【新】配置 `strategy.llm` 节点**: 这是与 v1.0 最大的不同。您**必须**提供 `factors_to_use` 和 `factor_explanations`，否则 `LLMStrategy` 将报错。

    **`config.yaml` 示例:**

    ```yaml
    strategy:
      llm:
        default_universe: "All"
        default_top_n: 5
        
        # --- 【新】必需的配置 ---
        # 1. 告诉引擎要计算哪些因子
        # (名称必须与 factor_engine.py 中的 FACTOR_REGISTRY 键一致)
        factors_to_use:
          - 'momentum'
          - 'breakout_quality_score'
          - 'eric_adx_weighted_momentum'
          - 'amount_std'
          - 'smart_momentum_adx_r2'
          - 'eric_tail_skew'

        # 2. 告诉 LLM 如何理解这些因子
        factor_explanations:
          'momentum': '168日价格动量 (总收益率)。值越高越好。'
          'breakout_quality_score': '突破质量分。结合了动量、参与度和稳定性。值越高越好。'
          'eric_adx_weighted_momentum': 'ADX加权动量。使用ADX趋势强度调节信号。值越高越好。'
          'amount_std': '42日成交额标准差。一个衡量波动性的指标。'
          'smart_momentum_adx_r2': '智慧动量。基础动量 * (ADX趋势概率 * R-squared稳定性)。值越高越好。'
          'eric_tail_skew': '经验极端口概率偏度。测量近期极端收益的偏斜（反转信号）。值越高越好。'
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

脚本将加载 `config.yaml`，`FactorEngine` 会自动计算 `factors_to_use` 列表中的所有因子，并将它们连同 `factor_explanations` 中的解释一起发送给 LLM。

**通过命令行参数覆盖配置:**
(这只会覆盖 `default_universe`, `default_top_n` 以及回测日期，因子列表始终从 `config.yaml` 读取)

```bash
# 运行 equity_global，允许 AI 最多选择 3 支 ETF
python run_backtest.py --universe equity_global --start 2018-01-01 --topn 3
```

## 如何扩展 (添加新因子)

新的架构使得添加自定义因子变得非常容易：

1.  **定义因子**: 在 `llm_quant_lib/factor_definitions.py` 中，创建一个继承自 `BaseAlpha` 的新类，并实现 `predict()` 方法 (您可以参考文件中的60多个例子)。
2.  **注册因子**: 打开 `llm_quant_lib/factor_engine.py`，在顶部的 `FACTOR_REGISTRY` 字典中，添加一个新条目：
    ```python
    'my_new_factor_name': (YourNewFactorClass, {'param1': 10, 'param2': 20}),
    ```
3.  **使用因子**: 打开 `config.yaml`，将 `'my_new_factor_name'` 添加到 `factors_to_use` 列表，并在 `factor_explanations` 中为 AI 添加一句解释。

完成！下次运行 `run_backtest.py` 时，框架将自动计算并使用您的新因子。