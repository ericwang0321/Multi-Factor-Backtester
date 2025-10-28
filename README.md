# LLM 驱动的量化回测框架

本项目是一个基于大型语言模型（LLM，本项目使用 DeepSeek）进行投资决策的量化回测框架。框架采用面向对象（OOP）的设计，将数据处理、因子计算、投资组合管理、策略决策和回测引擎解耦，方便扩展和维护。

**核心特色：** LLM 不仅负责选择投资标的，还能自主决定持仓数量（在设定的最大值内）和每只标的的具体权重。

## 项目结构

```

your\_project\_directory/
├── llm\_quant\_lib/           \# 核心量化库
│   ├── ... (省略内部文件)
├── data/                    \# 数据文件存储目录
│   ├── processed/
│   │   ├── price\_with\_simple\_returns\_2016\_onwards.csv \# (必需)
│   │   └── ... (基准文件)
│   └── reference/
│       └── sec\_code\_category\_grouped.csv        \# (必需)
├── plots/                   \# 保存回测结果图表
│   └── \<universe\_name\>/
├── logs/                    \# 保存 AI 决策日志和持仓记录
│   └── \<universe\_name\>/
├── requirements.txt         \# Python 依赖库列表
├── config.yaml              \# 项目配置文件
└── run\_backtest.py          \# 主运行脚本

````

## 设置步骤

1.  **克隆或下载项目。**
2.  **创建虚拟环境 (推荐)。**
3.  **安装依赖库:**
    ```bash
    pip install -r requirements.txt
    pip install pyyaml openai # 确保安装了 yaml 和 openai
    ```
4.  **准备数据文件:** 确保价格数据、资产池定义、以及 `config.yaml` 中引用的所有基准文件都位于 `data` 目录下对应的子文件夹中。
5.  **配置 `config.yaml`:**
    * 检查数据库连接（如果需要）、文件路径、基准映射、综合基准配置。
    * **`strategy.llm.default_top_n`**: 设置 AI **最多**可以选择的默认资产数量。
    * **不要**在 YAML 中写入 API 密钥或数据库密码。
6.  **配置环境变量 (必需):**
    * 设置 `DEEPSEEK_API_KEY` (必需)。
    * 设置 `DB_PASSWORD` (如果使用数据库，推荐)。

## 如何运行

在项目主目录下打开终端，运行 `run_backtest.py` 脚本。

**使用默认配置运行:**
```bash
python run_backtest.py
````

脚本将加载 `config.yaml`，使用默认设置运行回测。LLM 会决定持仓数量（不超过 `default_top_n`）和具体权重。

**通过命令行参数覆盖配置:**

```bash
# 运行 equity_global，允许 AI 最多选择 3 支 ETF 并决定权重
python run_backtest.py --universe equity_global --start 2018-01-01 --topn 3

# 运行 All (使用综合基准)，允许 AI 最多选择 10 支 ETF 并决定权重
python run_backtest.py --universe All --topn 10