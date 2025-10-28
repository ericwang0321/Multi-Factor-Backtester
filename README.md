# LLM 驱动的量化回测框架

本项目是一个基于大型语言模型（LLM，本项目使用 DeepSeek）进行投资决策的量化回测框架。框架采用面向对象（OOP）的设计，将数据处理、因子计算、投资组合管理、策略决策和回测引擎解耦，方便扩展和维护。

## 项目结构

```
your_project_directory/
├── llm_quant_lib/          # 核心量化库
│   ├── ... (省略内部文件)
├── data/                   # 数据文件存储目录
│   ├── processed/
│   │   ├── price_with_simple_returns_2016_onwards.csv # (必需)
│   │   └── spxt_index_daily_return.csv             # 示例基准
│   └── reference/
│       └── sec_code_category_grouped.csv        # (必需)
├── plots/                  # 保存回测结果图表
│   └── <universe_name>/
├── logs/                   # 保存 AI 决策日志和持仓记录
│   └── <universe_name>/
├── requirements.txt        # Python 依赖库列表
├── config.yaml             # 项目配置文件
└── run_backtest.py         # 主运行脚本
```

## 设置步骤

1.  **克隆或下载项目:** 获取所有代码文件。
2.  **创建虚拟环境 (推荐):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
3.  **安装依赖库:** 确保你在项目主目录下 (`your_project_directory/`)，然后运行：
    ```bash
    pip install -r requirements.txt
    # 添加 PyYAML 用于读取配置文件
    pip install pyyaml
    ```
4.  **准备数据文件:**
    * 确保 `data/processed/price_with_simple_returns_2016_onwards.csv` 文件存在且格式正确。
    * 确保 `data/reference/sec_code_category_grouped.csv` 文件存在且格式正确。
    * 确保对应的基准数据文件（例如 `spxt_index_daily_return.csv`）存在于 `data/processed/` 目录下。
5.  **配置 `config.yaml`:**
    * 打开项目根目录下的 `config.yaml` 文件。
    * **数据库:** 如果需要从数据库加载数据（例如 CSV 不存在时），请设置 `use_db: true` 并检查数据库连接信息。**强烈建议**将密码设置为环境变量 `DB_PASSWORD`，并在 YAML 中使用 `password: "${DB_PASSWORD}"` 来引用。
    * **文件路径:** 检查 `paths` 部分的文件路径是否正确。
    * **回测参数:** 在 `backtest` 部分设置默认的回测开始/结束日期、初始资金等。
    * **策略参数:** 在 `strategy.llm` 部分设置默认的资产池 (`default_universe`) 和选股数量 (`default_top_n`)。
    * **API 密钥:** **不要**在 YAML 文件中写入 API 密钥。代码会从环境变量 `DEEPSEEK_API_KEY` 读取。
6.  **配置环境变量 (必需):**
    * **【必需】** 设置环境变量 `DEEPSEEK_API_KEY`，值为你的 DeepSeek API 密钥。
    * **【推荐】** 如果使用数据库，设置环境变量 `DB_PASSWORD`，值为你的数据库密码。

## 如何运行

在项目主目录下 (`your_project_directory/`) 打开终端，运行 `run_backtest.py` 脚本。

**使用 `config.yaml` 中的默认配置运行:**
```bash
python run_backtest.py
```
脚本会加载 `config.yaml`，并使用其中定义的默认设置（如 `backtest.start_date`, `strategy.llm.default_universe` 等）进行回测。

**通过命令行参数覆盖 `config.yaml` 中的默认值:**
你可以通过命令行参数临时修改本次运行的设置，这会覆盖 `config.yaml` 中的对应项：
```bash
python run_backtest.py --universe equity_global --start 2018-01-01 --end 2023-12-31 --topn 3