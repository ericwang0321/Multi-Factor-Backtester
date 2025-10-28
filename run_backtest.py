# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse # 用于接收命令行参数

# ---【重要】确保 llm_quant_lib 在 Python 路径中 ---
# 将库文件所在的父目录添加到 sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# --- 导入我们创建的库 ---
try:
    from llm_quant_lib.data_handler import DataHandler
    from llm_quant_lib.strategy import LLMStrategy
    from llm_quant_lib.backtest_engine import BacktestEngine
    from llm_quant_lib.performance import calculate_extended_metrics, display_metrics
except ImportError as e:
    print("导入 llm_quant_lib 时出错，请确保：")
    print(f"1. llm_quant_lib 文件夹与此脚本 ({__file__}) 位于同一父目录下。")
    print("2. 或者 llm_quant_lib 的父目录已添加到 Python 环境变数 PYTHONPATH 中。")
    print(f"当前 sys.path: {sys.path}")
    print(f"原始错误: {e}")
    sys.exit(1)

# ======================================================================
# --- 1. 回测配置 ---
# ======================================================================

# ---【核心配置项】---
# 你可以在这里修改，或者通过命令行参数覆盖
UNIVERSE_TO_RUN = 'equity_us' # <<< 要回测的资產池 ('equity_us', 'bond', 'All', 等)
START_DATE = '2017-01-01'
END_DATE = '2024-07-31' # 根据你的数据调整

# ---【假设 2】数据库连接 (如果需要) ---
DB_CONFIG = {
    'username': 'intern52',
    'password': 'Gaoteng!wdy!0804', # 强烈建议使用环境变量或配置文件
    'host': '10.26.132.99',
    'port': '5432',
    'database': 'datalake'
}
# DB_CONFIG = None # 如果确定只用 CSV

# ---【假设 3】文件路径 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
REFERENCE_DATA_DIR = os.path.join(DATA_DIR, 'reference')

CSV_PRICE_PATH = os.path.join(PROCESSED_DATA_DIR, 'price_with_simple_returns_2016_onwards.csv')
UNIVERSE_DEFINITION_PATH = os.path.join(REFERENCE_DATA_DIR, 'sec_code_category_grouped.csv')
# ---【假设 4】基準文件路径 (根据资產池调整) ---
# 你可以为不同的 universe 指定不同的基準，或者使用一个通用的
BENCHMARK_MAPPING = {
    'equity_us': 'spxt_index_daily_return.csv',
    'equity_global': 'mxwd_index_daily_return.csv',
    'bond': 'global_bond_index_daily_return.csv',
    'commodity': 'bcom_index_daily_return.csv',
    'alternative': 'bcom_index_daily_return.csv', # 示例
    'All': 'mxwd_index_daily_return.csv' # 为 'All' 指定一个默认基準
}
BENCHMARK_FILENAME = BENCHMARK_MAPPING.get(UNIVERSE_TO_RUN, 'mxwd_index_daily_return.csv') # 默认使用 MSCI World
BENCHMARK_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, BENCHMARK_FILENAME)

# 回测核心参数
BACKTEST_CONFIG = {
    'INITIAL_CAPITAL': 1_000_000,
    'COMMISSION_RATE': 0.001,
    'SLIPPAGE': 0.0005,
    'REBALANCE_MONTHS': 1,
    'DB_CONFIG': DB_CONFIG,         # 传递给 DataHandler
    'PRICE_DATA_PATH': CSV_PRICE_PATH # 传递给 DataHandler
}

# AI 策略特定参数
AI_STRATEGY_CONFIG = {
    'universe_to_trade': UNIVERSE_TO_RUN, # <<< AI 策略也需要知道它在哪个池里决策
    'top_n': 5,
    # 'api_key': 'sk-your_deepseek_api_key' # 【假设 1 重申】
}

# 结果保存路径
PLOT_DIR = os.path.join(BASE_DIR, 'plots', UNIVERSE_TO_RUN) # 按资產池保存
LOG_DIR = os.path.join(BASE_DIR, 'logs', UNIVERSE_TO_RUN)  # 按资產池保存
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ======================================================================
# --- 命令行参数解析 (可选) ---
# ======================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="运行基于 LLM 的量化回测")
    parser.add_argument(
        "--universe", type=str, default=UNIVERSE_TO_RUN,
        help=f"要回测的资產池名称 (例如: equity_us, bond, All)。默认: {UNIVERSE_TO_RUN}"
    )
    parser.add_argument(
        "--start", type=str, default=START_DATE,
        help=f"回测开始日期 (YYYY-MM-DD)。默认: {START_DATE}"
    )
    parser.add_argument(
        "--end", type=str, default=END_DATE,
        help=f"回测结束日期 (YYYY-MM-DD)。默认: {END_DATE}"
    )
    parser.add_argument(
        "--topn", type=int, default=AI_STRATEGY_CONFIG['top_n'],
        help=f"让 AI 选择的 Top N 资產数量。默认: {AI_STRATEGY_CONFIG['top_n']}"
    )
    # 你可以添加更多参数，例如 API Key, benchmark 文件等
    return parser.parse_args()

# ======================================================================
# --- 主执行逻辑 ---
# ======================================================================
if __name__ == '__main__':
    # 解析命令行参数 (如果提供的话)
    args = parse_arguments()
    UNIVERSE_TO_RUN = args.universe
    START_DATE = args.start
    END_DATE = args.end
    AI_STRATEGY_CONFIG['top_n'] = args.topn
    AI_STRATEGY_CONFIG['universe_to_trade'] = UNIVERSE_TO_RUN # 确保 AI 策略知道当前资產池

    # 更新输出路径和基準路径
    PLOT_DIR = os.path.join(BASE_DIR, 'plots', UNIVERSE_TO_RUN)
    LOG_DIR = os.path.join(BASE_DIR, 'logs', UNIVERSE_TO_RUN)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    BENCHMARK_FILENAME = BENCHMARK_MAPPING.get(UNIVERSE_TO_RUN, 'mxwd_index_daily_return.csv')
    BENCHMARK_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, BENCHMARK_FILENAME)

    print(f"--- 开始回测 ---")
    print(f"资產池: {UNIVERSE_TO_RUN}")
    print(f"时间范围: {START_DATE} 到 {END_DATE}")
    print(f"AI选股数: {AI_STRATEGY_CONFIG['top_n']}")
    print(f"基準文件: {BENCHMARK_FILENAME}")

    # --- 1. 数据准备 ---
    print("\n--- 阶段 1: 数据准备 ---")
    try:
        data_handler = DataHandler(
            db_config=DB_CONFIG,
            csv_path=CSV_PRICE_PATH,
            start_date=START_DATE,
            end_date=END_DATE
        )
        raw_df = data_handler.load_data() # 加载价格数据
        universe_df = data_handler.load_universe_data(UNIVERSE_DEFINITION_PATH) # 加载资產池定义
        benchmark_df = data_handler.load_benchmark_data(BENCHMARK_DATA_PATH) # 加载基準数据
    except FileNotFoundError as e:
        print(f"错误: 数据文件未找到: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"数据准备阶段出错: {e}")
        sys.exit(1)

    # --- 2. 初始化策略 ---
    print("\n--- 阶段 2: 初始化策略 ---")
    try:
        strategy = LLMStrategy(
            universe_df=universe_df,
            **AI_STRATEGY_CONFIG
        )
        print(f"策略 '{strategy.__class__.__name__}' 初始化成功。")
    except ValueError as e:
        print(f"错误: 初始化 AI 策略失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"初始化 AI 策略时发生未知错误: {e}")
        sys.exit(1)

    # --- 3. 运行回测 ---
    print("\n--- 阶段 3: 执行回测 ---")
    engine = None # 初始化 engine 变量
    final_portfolio = None
    equity_curve = None
    try:
        engine = BacktestEngine(
            start_date=START_DATE,
            end_date=END_DATE,
            config=BACKTEST_CONFIG,
            strategy=strategy,
            data_handler=data_handler,
            universe_to_run=UNIVERSE_TO_RUN # <<< 传入要运行的资產池
        )
        equity_df, final_portfolio = engine.run() # 运行并接收结果

        if equity_df is None or final_portfolio is None:
            print("错误: 回测引擎未能成功运行或未返回有效结果。")
            sys.exit(1)

        equity_curve = equity_df['total_value'] # 提取净值 Series
        print("回测执行完毕。")

    except ValueError as e:
        print(f"错误: 回测引擎初始化或运行时失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"回测运行过程中发生未知错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- 4. 结果分析与展示 ---
    if equity_curve is not None and final_portfolio is not None:
        print("\n--- 阶段 4: 结果分析与展示 ---")

        # 准备基準净值曲线
        print("正在准备基準净值曲线...")
        initial_capital = BACKTEST_CONFIG['INITIAL_CAPITAL']
        benchmark_df.index = pd.to_datetime(benchmark_df.index) # 确保索引是 datetime
        benchmark_returns_aligned = benchmark_df['benchmark_return'].reindex(equity_curve.index, method='ffill').fillna(0.0)
        benchmark_equity = initial_capital * (1 + benchmark_returns_aligned).cumprod()
        benchmark_equity.iloc[0] = initial_capital # 确保起始值正确
        print("基準净值曲线准备完毕。")

        # 计算性能指标
        print("正在计算性能指标...")
        metrics = calculate_extended_metrics(
            portfolio_equity=equity_curve,
            benchmark_equity=benchmark_equity,
            portfolio_instance=final_portfolio
        )
        print("性能指标计算完毕。")

        # 打印指标
        display_metrics(metrics)

        # 绘制净值曲线图
        print("正在绘制净值曲线图...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 8))
        (equity_curve / equity_curve.iloc[0]).plot(ax=ax, label=f'LLM Strategy ({UNIVERSE_TO_RUN})', lw=2)
        (benchmark_equity / benchmark_equity.iloc[0]).plot(ax=ax, label=f'Benchmark ({BENCHMARK_FILENAME.split("_")[0]})', lw=2, linestyle=':')
        ax.set_title(f'LLM 量化策略回测 ({UNIVERSE_TO_RUN} | {START_DATE} to {END_DATE})', fontsize=16)
        ax.set_xlabel('日期')
        ax.set_ylabel('归一化净值 (初始值=1)')
        ax.legend()
        ax.grid(True)
        ax.set_yscale('log') # 使用对数坐标可能更清晰
        plt.tight_layout()
        plot_filename = f"equity_curve_{UNIVERSE_TO_RUN}_llm.png"
        plot_path = os.path.join(PLOT_DIR, plot_filename)
        try:
            plt.savefig(plot_path)
            print(f"净值曲线图已保存至: {plot_path}")
        except Exception as e:
            print(f"保存图表时出错: {e}")
        # plt.show() # 在服务器运行时可能不需要显示

        # 保存 AI 决策日誌
        print("正在保存 AI 决策日誌...")
        ai_log_df = strategy.get_trade_log()
        log_filename = f"ai_trade_log_{UNIVERSE_TO_RUN}.csv"
        log_path = os.path.join(LOG_DIR, log_filename)
        try:
            ai_log_df.to_csv(log_path, index=False, encoding='utf-8-sig')
            print(f"AI 决策日誌已保存至: {log_path}")
        except Exception as e:
            print(f"保存 AI 日誌时出错: {e}")

        # 保存持仓记录
        print("正在保存详细持仓记录...")
        holdings_df = final_portfolio.get_holdings_history()
        holdings_filename = f"holdings_history_{UNIVERSE_TO_RUN}_llm.csv"
        holdings_path = os.path.join(LOG_DIR, holdings_filename)
        try:
            holdings_df.to_csv(holdings_path, index=False, encoding='utf-8-sig')
            print(f"持仓记录已保存至: {holdings_path}")
        except Exception as e:
            print(f"保存持仓记录时出错: {e}")

    print("\n--- 回测脚本执行完毕 ---")

