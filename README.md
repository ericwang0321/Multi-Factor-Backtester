# LLM 驱动的量化回测框架 (v2.2 - 多模型支持)

本项目是一个基于大型语言模型（LLM）进行投资决策的量化回测框架。框架采用面向对象（OOP）的设计，将数据处理、因子计算、投资组合管理、策略决策和回测引擎解耦，方便扩展和维护。

**核心特色 (v2.2):**

  * **强大的因子引擎**: 框架已升级，使用基于 `xarray` 和 `BaseAlpha` 类的专业因子库（源自 `prepare_factor_data.py`），支持复杂的横截面和时间序列因子计算。
  * **配置驱动的因子**: 您可以通过 `config.yaml` 动态指定任意已注册的因子及其解释，LLM 将自动接收并使用它们进行决策。
  * **多模型支持**: 框架已重构，支持在多个 LLM 供应商（如 DeepSeek, 阿里云通义千问）及其特定模型（如 `qwen-plus`, `qwen-max`）之间通过配置文件灵活切换。
  * **LLM 自主决策**: LLM 不仅负责选择投资标的，还能自主决定持仓数量（在设定的最大值内）和每只标的的具体权重。
  * **智能日期调整**: 框架会自动检查您的数据历史长度和所需的因子计算缓冲期 (`factor_buffer_months`)，并在必要时**自动推迟回测开始日期**，确保长周期因子在回测开始时有有效值。

## 项目结构

```
your_project_directory/
├── llm_quant_lib/           # 核心量化库
│   ├── __init__.py
│   ├── data_handler.py      # 数据加载器
│   ├── factor_definitions.py # 所有因子(BaseAlpha)的计算逻辑
│   ├── factor_engine.py     # 因子引擎 (调用 factor_definitions.py)
│   ├── strategy.py          # 策略 (从配置动态读取因子和模型)
│   ├── portfolio.py         # 投资组合/交易执行
│   ├── backtest_engine.py   # 回测引擎 (连接策略和因子引擎)
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
└── run_backtest.py          # 主运行脚本 (含日期调整逻辑)
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

      * 检查数据库连接、文件路径、基准映射等。
      * **配置 `backtest` 节点**:
          * `start_date`: 您**期望**的回测开始日期。
          * `factor_buffer_months`: **因子计算所需的缓冲期（月数）**。这个值应该**大于等于**您使用的所有因子中最长回看窗口所需的月数（例如，`momentum` 需要 168 天，约 8 个月，建议设置为 `9`）。
      * **配置 `strategy.llm` 节点**:
          * `active_model`: **必需**。选择一个您想激活的模型（必须在 `models` 列表中定义）。
          * `api_providers`: **必需**。定义每个 LLM 供应商的 `base_url` 和 `api_key_env`（对应的环境变量名）。
          * `models`: **必需**。定义每个具体模型的 `provider` 和 `model_name`。
          * `factors_to_use`: **必需**。指定要计算和发送给 LLM 的因子列表。
          * `factor_explanations`: **必需**。为因子提供解释。

    **`config.yaml` 示例:**

    ```yaml
    backtest:
      start_date: "2017-08-01"
      end_date: "2025-09-30"
      # ...
      factor_buffer_months: 9 # 因子缓冲期

    strategy:
      llm:
        default_universe: "All"
        default_top_n: 5
        
        # --- 【新】多模型配置 ---
        active_model: "qwen_max_model" # 在这里切换模型

        api_providers:
          deepseek_provider:
            api_key_env: "DEEPSEEK_API_KEY"
            base_url: "https://api.deepseek.com"
          qwen_provider:
            api_key_env: "DASHSCOPE_API_KEY"
            base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"

        models:
          deepseek_chat:
            provider: "deepseek_provider"
            model_name: "deepseek-chat"
          qwen_plus_model:
            provider: "qwen_provider"
            model_name: "qwen-plus"
          qwen_max_model:
            provider: "qwen_provider"
            model_name: "qwen-max"

        # --- 必需的因子配置 ---
        factors_to_use:
          - 'momentum'
          - 'breakout_quality_score'
          # ... 其他因子 ...
        
        factor_explanations:
          'momentum': '168日价格动量。值越高越好。'
          'breakout_quality_score': '突破质量分。值越高越好。'
          # ... 其他解释 ...
    ```
    
6.  **配置环境变量 (关键步骤)**

    您**必须**为您在 `api_providers` 中定义的**所有** `api_key_env` 设置环境变量。例如，如果您在配置中启用了 `deepseek` 和 `dashscope`，您就必须设置 `DEEPSEEK_API_KEY` 和 `DASHSCOPE_API_KEY`。

    环境变量是**私密**的，不应提交到 Git 仓库。以下是不同操作系统的设置方法：

    ### 方式一：Windows 系统

    1.  **通过图形界面 (GUI) - 推荐**
        * 在 Windows 搜索栏搜索“环境变量”（或 “Edit the system environment variables”）。
        * 点击“环境变量...”按钮。
        * 在“XXX 的用户变量”或“系统变量”下点击“新建...”。
        * **变量名**填入 `DEEPSEEK_API_KEY`，**变量值**填入您的 API 密钥。
        * 一路点击“确定”保存。
        * **[图文教程]** 详细的步骤可以参考这篇指南：[Windows 10/11 设置环境变量详解 (知乎)](https://zhuanlan.zhihu.com/p/231668109)

    2.  **通过命令行 (PowerShell)**
        * **永久设置 (推荐):**
          ```powershell
          [Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "your_key_here", "User")
          ```
        * 通过此方法设置后，您需要**重启终端或 VS Code**才能使其生效。

    ### 方式二：macOS / Linux 系统

    1.  打开您的终端 (Terminal)。
    2.  确定您使用的 Shell 类型 (通常是 Zsh 或 Bash)。
        * 如果是 Zsh (macOS 默认)，编辑 `~/.zshrc` 文件：
          ```bash
          nano ~/.zshrc
          ```
        * 如果是 Bash，编辑 `~/.bash_profile` 或 `~/.bashrc` 文件：
          ```bash
          nano ~/.bash_profile
          ```
    3.  在文件的末尾添加以下内容 (以 DeepSeek 和 DashScope 为例)：
        ```bash
        export DEEPSEEK_API_KEY="your_key_here"
        export DASHSCOPE_API_KEY="your_other_key_here"
        ```
    4.  保存文件并退出 (在 `nano` 中是 `Ctrl+X`, `Y`, `Enter`)。
    5.  让配置立即生效（或直接重启终端）：
        ```bash
        source ~/.zshrc  # 或者 source ~/.bash_profile
        ```

    ### 方式三：使用 `.env` 文件 (推荐用于项目)

    如果您不希望设置全局环境变量，可以在本项目的根目录（与 `README.md` 同级）创建一个名为 `.env` 的文件。

    1.  创建 `.env` 文件。
    2.  在该文件中添加您的密钥：
        ```
        DEEPSEEK_API_KEY="your_key_here"
        DASHSCOPE_API_KEY="your_other_key_here"
        ```
    3.  **[安全]** 确保您的 `.gitignore` 文件中包含了 `.env` 这一行，防止密钥泄露。
    4.  *(假设)* 脚本主程序中已包含加载 `.env` 文件的代码 (例如使用 `python-dotenv` 库的 `load_dotenv()`)。此方法通常无需重启。

    -----

    ### **!! 重启生效提醒 !!**

    * **重要**: 如果您使用了**方法一 (Windows GUI)** 来设置变量，您**必须彻底重启 VS Code 和您的终端**，才能使新变量生效。仅仅重启终端窗口是不够的。
    * 如果您在 VS Code 内置的终端中运行，**重启 VS Code** 是最保险的生效方式。

## 如何运行

在项目主目录下打开终端，运行 `run_backtest.py` 脚本。

**使用 `config.yaml` 中的配置运行:**

```bash
python run_backtest.py
```

脚本将加载 `config.yaml`，`LLMStrategy` 会自动初始化 `active_model` 所指定的模型，`FactorEngine` 会自动计算 `factors_to_use` 列表中的所有因子。

**通过命令行参数覆盖配置:**
(这只会覆盖 `universe`, `top_n` 以及**期望的** `start`/`end` 日期。`active_model` 和 `factors_to_use` 始终从 `config.yaml` 读取)

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

## 如何扩展

### 扩展 1：添加新因子

1.  **定义因子**: 在 `llm_quant_lib/factor_definitions.py` 中，创建一个继承自 `BaseAlpha` 的新类，并实现 `predict()` 方法。
2.  **注册因子**: 打开 `llm_quant_lib/factor_engine.py`，在顶部的 `FACTOR_REGISTRY` 字典中，添加一个新条目：
    ```python
    'my_new_factor_name': (YourNewFactorClass, {'param1': 10}),
    ```
3.  **使用因子**: 打开 `config.yaml`，将 `'my_new_factor_name'` 添加到 `factors_to_use` 列表，并在 `factor_explanations` 中为 AI 添加一句解释。
4.  **检查缓冲期**: 确保 `config.yaml` 中的 `factor_buffer_months` 足够覆盖您新因子的回看窗口。

### 扩展 2：添加新 LLM 模型

1.  **添加供应商 (如果需要)**: 如果是新供应商（比如 Moonshot），打开 `config.yaml`，在 `api_providers` 下添加一个新条目：
    ```yaml
    moonshot_provider:
      api_key_env: "MOONSHOT_API_KEY"
      base_url: "https://api.moonshot.cn/v1"
    ```
    (并确保设置了 `MOONSHOT_API_KEY` 环境变量)。
2.  **添加模型**: 在 `models` 列表下添加新模型：
    ```yaml
    moonshot_v1_8k:
      provider: "moonshot_provider" # 引用您刚添加的供应商
      model_name: "moonshot-v1-8k"
    ```
3.  **激活模型**: 将 `active_model` 的值改为 `'moonshot_v1_8k'`。

完成！下次运行 `run_backtest.py` 时，框架将自动使用新模型。