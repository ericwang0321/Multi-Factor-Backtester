# LLM 驱动的量化回测框架

本项目是一个基于大型语言模型（LLM，本项目使用 DeepSeek）进行投资决策的量化回测框架。框架采用面向对象（OOP）的设计，将数据处理、因子计算、投资组合管理、策略决策和回测引擎解耦，方便扩展和维护。

## 项目结构

```

your\_project\_directory/
├── llm\_quant\_lib/          \# 核心量化库
│   ├── **init**.py         \# 包标识文件
│   ├── data\_handler.py     \# 数据加载与处理模块
│   ├── factor\_helpers.py   \# 时间序列计算辅助函数
│   ├── factor\_engine.py    \# 因子计算引擎 (EMA, MACD, RSI)
│   ├── portfolio.py        \# 投资组合状态管理与交易执行
│   ├── strategy.py         \# 策略定义 (包含 LLMStrategy)
│   ├── performance.py      \# 性能指标计算
│   └── backtest\_engine.py  \# 回测主引擎
├── data/                   \# 数据文件存储目录
│   ├── processed/          \# 处理后的数据
│   │   ├── price\_with\_simple\_returns\_2016\_onwards.csv \# 价格数据 (必需)
│   │   └── spxt\_index\_daily\_return.csv             \# 示例基准数据
│   └── reference/          \# 参考数据
│       └── sec\_code\_category\_grouped.csv        \# 资产池定义 (必需)
├── plots/                  \# 保存回测结果图表
│   └── equity\_us/          \# 按资产池分子目录
├── logs/                   \# 保存 AI 决策日志和持仓记录
│   └── equity\_us/          \# 按资产池分子目录
├── requirements.txt        \# Python 依赖库列表
└── run\_backtest.py         \# 主运行脚本

```

## 设置步骤

1. **克隆或下载项目:** 获取所有代码文件。

2. **创建虚拟环境 (推荐):**

```

python -m venv venv
source venv/bin/activate  \# Linux/macOS

# venv\\Scripts\\activate  \# Windows

```

3. **安装依赖库:** 确保你在项目主目录下 (`your_project_directory/`)，然后运行：

```

pip install -r requirements.txt

```

4. **准备数据文件:**

* 确保 `data/processed/price_with_simple_returns_2016_onwards.csv` 文件存在且包含所需的价格数据（至少需要 'datetime', 'sec_code', 'open', 'high', 'low', 'close', 'volume' 列）。你可以运行 `llm_quant_lib/data_handler.py` 中的 `_load_from_db` 相关逻辑（如果配置了数据库）来生成此文件，或者手动准备。

* 确保 `data/reference/sec_code_category_grouped.csv` 文件存在，并包含 `sec_code` 和 `universe` (或 `category_id`) 列，用于定义资产池。

* 确保基准数据文件（例如 `data/processed/spxt_index_daily_return.csv`）存在，包含日期列 (`report_date`) 和收益率列 (`default` 或其他，代码会自动识别）。

5. **配置 API 密钥:**

* **【必需】** 设置环境变量 `DEEPSEEK_API_KEY`，值为你的 DeepSeek API 密钥。这是最安全的方式。

* （不推荐）或者，直接在 `llm_quant_lib/strategy.py` 文件中的 `LLMStrategy` 类初始化函数里修改 `api_key` 参数。

6. **检查配置 (run_backtest.py):**

* 打开 `run_backtest.py` 文件。

* 根据需要修改 `START_DATE`, `END_DATE`。

* 检查 `DB_CONFIG` (如果需要从数据库加载数据) 是否正确配置。如果只使用 CSV，可以将其设为 `None`。

* 确认 `CSV_PRICE_PATH`, `UNIVERSE_DEFINITION_PATH`, `BENCHMARK_MAPPING` 中的文件路径和名称是否正确。

* 确认 `AI_STRATEGY_CONFIG` 中的 `universe_to_trade` 和 `top_n` 是否是你想要的设置。

## 如何运行

在项目主目录下 (`your_project_directory/`) 打开终端，运行 `run_backtest.py` 脚本。

**基础运行:**

```

python run\_backtest.py

```

这将使用 `run_backtest.py` 文件顶部定义的默认配置（资产池、日期范围、Top N）运行回测。

**使用命令行参数指定配置:**
你可以通过命令行参数覆盖默认配置：

```

python run\_backtest.py --universe equity\_global --start 2018-01-01 --end 2023-12-31 --topn 3

```

* `--universe`: 指定要回测的资产池名称 (必须与 `sec_code_category_grouped.csv` 中的名称匹配)。

* `--start`: 回测开始日期 (YYYY-MM-DD)。

* `--end`: 回测结束日期 (YYYY-MM-DD)。

* `--topn`: 让 AI 选择的 Top N 资产数量。

回测运行后，结果（性能指标）会打印在终端，净值曲线图会保存在 `plots/<universe_name>/` 目录下，AI 决策日志和持仓记录会保存在 `logs/<universe_name>/` 目录下。

## 主要假设

1. **DeepSeek API Key:** 需要有效的 DeepSeek API 密钥，并通过环境变量 `DEEPSEEK_API_KEY` 提供。

2. **数据文件:** 必需的价格数据、资产池定义和基准数据文件存在于指定的 `data` 目录结构中，且格式符合代码要求。

3. **数据库配置:** 如果 CSV 文件不存在，需要提供正确的数据库连接信息才能自动下载数据。

4. **因子依赖:** 当前 `FactorEngine` 需要 `open`, 'high', 'low', 'close', 'volume' 数据来计算 EMA, MACD, RSI。

5. **Python 环境:** 假设使用 Python 3.x，并已通过 `requirements.txt` 安装所有依赖。

6. **网络连接:** 运行时需要网络连接以调用 DeepSeek API。
