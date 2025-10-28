# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse # 引入 argparse 用于处理命令行参数
import yaml     # 引入 yaml 用于读取配置文件
from datetime import datetime # 用于日期转换

# --- 导入我们创建的库 ---
try:
    from llm_quant_lib.data_handler import DataHandler
    from llm_quant_lib.strategy import LLMStrategy # 导入 AI 策略
    # from llm_quant_lib.strategy import StaticWeightStrategy # 如果需要静态策略对比
    from llm_quant_lib.backtest_engine import BacktestEngine
    from llm_quant_lib.performance import calculate_extended_metrics, display_metrics
except ImportError as e:
    print("导入 llm_quant_lib 时出错，请确保：")
    print("1. llm_quant_lib 文件夹与此脚本位于同一父目录下，或者")
    print("2. llm_quant_lib 的父目录已添加到 Python 环境变量 PYTHONPATH 中。")
    print(f"原始错误: {e}")
    sys.exit(1)

# ======================================================================
# --- 配置文件加载 ---
# ======================================================================
def load_config(config_path='config.yaml'):
    """加载 YAML 配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"配置文件 '{config_path}' 加载成功。")
        return config
    except FileNotFoundError:
        print(f"错误: 配置文件 '{config_path}' 未找到。请确保它位于项目根目录下。")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"错误: 解析配置文件 '{config_path}' 时出错: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"加载配置文件时发生未知错误: {e}")
        sys.exit(1)

# ======================================================================
# --- 命令行参数解析 ---
# ======================================================================
def parse_arguments(config):
    """解析命令行参数，使用配置文件中的值作为默认值"""
    parser = argparse.ArgumentParser(description="运行 LLM 量化回测框架 (配置来自 config.yaml)")

    # 从 config 文件获取默认值
    default_universe = config.get('strategy', {}).get('llm', {}).get('default_universe', 'equity_us')
    default_start = config.get('backtest', {}).get('start_date', '2017-01-01')
    default_end = config.get('backtest', {}).get('end_date', '2024-07-31')
    default_topn = config.get('strategy', {}).get('llm', {}).get('default_top_n', 5) # 现在代表 max_top_n

    parser.add_argument('--universe', type=str, default=default_universe,
                        help=f"要回测的资产池名称 (默认: {default_universe})")
    parser.add_argument('--start', type=str, default=default_start,
                        help=f"回测开始日期 YYYY-MM-DD (默认: {default_start})")
    parser.add_argument('--end', type=str, default=default_end,
                        help=f"回测结束日期 YYYY-MM-DD (默认: {default_end})")
    # --- 【修改点】更新 --topn 的帮助说明 ---
    parser.add_argument('--topn', type=int, default=default_topn,
                        help=f"AI 最多可以选择的 Top N 资产数量 (Max Top N) (默认: {default_topn})")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help="指定配置文件的路径 (默认: config.yaml)")

    return parser.parse_args()

# ======================================================================
# --- 主程序 ---
# ======================================================================
if __name__ == '__main__':

    # --- 1. 加载配置和解析参数 ---
    args = parse_arguments({}) # 先解析一次获取 config 文件路径
    config = load_config(args.config)
    args = parse_arguments(config) # 再次解析，使用 config 的值作为默认值

    START_DATE = args.start
    END_DATE = args.end
    SELECTED_UNIVERSE = args.universe
    MAX_TOP_N = args.topn # 重命名变量以明确含义

    # --- 2. 准备配置字典 (从加载的 config 中提取) ---
    db_conf = config.get('database', {})
    paths_conf = config.get('paths', {})
    backtest_conf = config.get('backtest', {})
    strategy_conf = config.get('strategy', {}).get('llm', {}) # 只取 llm 策略配置

    # 数据库配置 (处理环境变量引用)
    DB_CONFIG = None
    if db_conf.get('use_db', False):
        DB_CONFIG = {
            'username': os.path.expandvars(db_conf.get('username', 'intern52')),
            'password': os.path.expandvars(db_conf.get('password', '')), # 从环境变量读取
            'host': os.path.expandvars(db_conf.get('host', 'localhost')),
            'port': os.path.expandvars(db_conf.get('port', '5432')),
            'database': os.path.expandvars(db_conf.get('database', 'datalake'))
        }
        if not DB_CONFIG.get('password') and db_conf.get('password', '').startswith('${'):
             print(f"警告: 数据库密码配置为从环境变量 '{db_conf.get('password')}' 读取，但该环境变量未设置!")
        elif not DB_CONFIG.get('password'):
             print("警告: 数据库密码未在配置文件或环境变量中设置!")

    # 文件路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PRICE_PATH = os.path.join(BASE_DIR, paths_conf.get('price_data_csv', 'data/processed/default_price.csv'))
    UNIVERSE_DEFINITION_PATH = os.path.join(BASE_DIR, paths_conf.get('universe_definition', 'data/reference/default_universe.csv'))

    # 基准加载逻辑 (与之前版本相同)
    BENCHMARK_MAPPING = paths_conf.get('benchmark_mapping', {})
    COMPOSITE_BENCHMARK_CONFIG = paths_conf.get('composite_benchmark', {})
    BENCHMARK_DATA_PATH = None
    BENCHMARK_DISPLAY_NAME = "Benchmark"
    if SELECTED_UNIVERSE.lower() == 'all' and COMPOSITE_BENCHMARK_CONFIG:
        print(f"检测到 universe 为 'All'，将使用 config.yaml 中定义的综合基准。")
        BENCHMARK_DISPLAY_NAME = COMPOSITE_BENCHMARK_CONFIG.get('display_name', 'Composite Benchmark')
    else:
        BENCHMARK_FILENAME = BENCHMARK_MAPPING.get(SELECTED_UNIVERSE)
        if BENCHMARK_FILENAME:
            BENCHMARK_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', BENCHMARK_FILENAME)
            BENCHMARK_DISPLAY_NAME = BENCHMARK_FILENAME
            if not os.path.exists(BENCHMARK_DATA_PATH):
                print(f"警告: 在 config.yaml 中为 '{SELECTED_UNIVERSE}' 配置的基准文件 '{BENCHMARK_FILENAME}' 未找到。")
                BENCHMARK_DATA_PATH = None
        if not BENCHMARK_DATA_PATH:
            print(f"警告: 未找到或未配置资产池 '{SELECTED_UNIVERSE}' 的基准文件，将不计算相对指标。")
            BENCHMARK_DISPLAY_NAME = "N/A"

    # 回测核心参数
    BACKTEST_CONFIG = {
        'INITIAL_CAPITAL': backtest_conf.get('initial_capital', 1000000),
        'COMMISSION_RATE': backtest_conf.get('commission_rate', 0.001),
        'SLIPPAGE': backtest_conf.get('slippage', 0.0005),
        'REBALANCE_MONTHS': backtest_conf.get('rebalance_months', 1),
        'DB_CONFIG': DB_CONFIG,
        'PRICE_DATA_PATH': CSV_PRICE_PATH
    }

    # AI 策略特定参数 (传入 max_top_n)
    AI_STRATEGY_CONFIG = {
        'universe_to_trade': SELECTED_UNIVERSE,
        'top_n': MAX_TOP_N # 传递给 LLMStrategy 的是 max_top_n
    }

    # 结果保存路径
    PLOT_DIR_BASE = os.path.join(BASE_DIR, paths_conf.get('plot_dir_base', 'plots'))
    LOG_DIR_BASE = os.path.join(BASE_DIR, paths_conf.get('log_dir_base', 'logs'))
    PLOT_DIR = os.path.join(PLOT_DIR_BASE, SELECTED_UNIVERSE)
    LOG_DIR = os.path.join(LOG_DIR_BASE, SELECTED_UNIVERSE)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- 3. 数据准备 ---
    print(f"\n--- 阶段 1: 数据准备 (资产池: {SELECTED_UNIVERSE}) ---")
    benchmark_df = None
    try:
        data_handler = DataHandler(
            db_config=DB_CONFIG,
            csv_path=CSV_PRICE_PATH,
            start_date=START_DATE,
            end_date=END_DATE
        )
        raw_df = data_handler.load_data()
        universe_df = data_handler.load_universe_data(UNIVERSE_DEFINITION_PATH)

        if SELECTED_UNIVERSE.lower() == 'all' and COMPOSITE_BENCHMARK_CONFIG:
            benchmark_df = data_handler.load_composite_benchmark_data(COMPOSITE_BENCHMARK_CONFIG)
            if benchmark_df is None: print("警告: 综合基准加载或计算失败。")
        elif BENCHMARK_DATA_PATH:
            benchmark_df = data_handler.load_benchmark_data(BENCHMARK_DATA_PATH)
            if benchmark_df is None: print(f"警告: 单个基准文件 '{BENCHMARK_DATA_PATH}' 加载失败。")

        if raw_df is None or raw_df.empty:
            raise ValueError("DataHandler 未能加载任何价格数据。")
        print(f"数据加载完毕。回测范围: {raw_df['datetime'].min().date()} 到 {raw_df['datetime'].max().date()}")

    except FileNotFoundError as e: print(f"错误: 必需的数据文件未找到: {e}"); sys.exit(1)
    except ValueError as e: print(f"错误: 数据准备失败: {e}"); sys.exit(1)
    except Exception as e: print(f"数据准备阶段发生未知错误: {e}"); import traceback; traceback.print_exc(); sys.exit(1)

    # --- 4. 初始化策略 ---
    print("\n--- 阶段 2: 初始化策略 ---")
    try:
        strategy = LLMStrategy(
            universe_df=universe_df,
            **AI_STRATEGY_CONFIG
        )
        print(f"策略 '{strategy.__class__.__name__}' 初始化成功。")
        print(f"   - 资产池: {AI_STRATEGY_CONFIG['universe_to_trade']}")
        print(f"   - 最大选股数: {AI_STRATEGY_CONFIG['top_n']}") # 更新打印信息
        if os.getenv("DEEPSEEK_API_KEY"): print("   - API Key: 已从环境变量 DEEPSEEK_API_KEY 加载")

    except ValueError as e: print(f"错误: 初始化 AI 策略失败: {e}\n请确保已正确设置 DEEPSEEK_API_KEY 环境变量且 Key 有效。"); sys.exit(1)
    except Exception as e: print(f"初始化 AI 策略时发生未知错误: {e}"); import traceback; traceback.print_exc(); sys.exit(1)

    # --- 5. 运行回测 ---
    portfolio_history = None
    final_portfolio = None
    engine = None
    try:
        print("\n--- 阶段 3: 执行回测 ---")
        engine = BacktestEngine(
            start_date=START_DATE, end_date=END_DATE, config=BACKTEST_CONFIG,
            strategy=strategy, data_handler=data_handler, universe_to_run=SELECTED_UNIVERSE
        )
        portfolio_history, final_portfolio = engine.run()
        if portfolio_history is None or portfolio_history.empty: print("错误: 回测引擎未生成有效净值。"); sys.exit(1)
        print("回测执行完毕。")
    except ValueError as ve: print(f"!!! 回测引擎初始化或运行期间捕获到 ValueError: {ve}"); import traceback; traceback.print_exc(); sys.exit(1)
    except Exception as e: print(f"!!! 回测引擎初始化或运行期间发生未知错误: {e}"); import traceback; traceback.print_exc(); sys.exit(1)

    # --- 确保回测成功 ---
    if portfolio_history is None or final_portfolio is None: print("错误: 回测未能成功完成。"); sys.exit(1)

    # --- 6. 结果分析与展示 ---
    print("\n--- 阶段 4: 结果分析与展示 ---")
    try:
        # (结果分析和展示部分与上一个版本几乎相同，仅更新图表/文件名的变量名)
        equity_curve = portfolio_history['total_value']
        equity_curve.index = pd.to_datetime(equity_curve.index)

        benchmark_equity = pd.Series(dtype=float)
        benchmark_loaded_successfully = False
        if benchmark_df is not None and not benchmark_df.empty:
            print("正在准备基准净值曲线...")
            try:
                initial_capital = BACKTEST_CONFIG['INITIAL_CAPITAL']
                benchmark_df.index = pd.to_datetime(benchmark_df.index)
                benchmark_returns_aligned = benchmark_df['benchmark_return'].reindex(equity_curve.index, method='ffill').fillna(0.0)
                first_valid_idx = benchmark_returns_aligned.first_valid_index()
                if first_valid_idx is not None and first_valid_idx > equity_curve.index[0]: benchmark_returns_aligned.loc[:first_valid_idx] = 0.0
                benchmark_equity = initial_capital * (1 + benchmark_returns_aligned).cumprod()
                if not benchmark_equity.empty:
                     benchmark_equity.iloc[0] = initial_capital
                     benchmark_loaded_successfully = True
                     print("基准净值曲线准备完毕。")
                else:
                     print("警告: 准备基准净值曲线后结果为空。")
                     benchmark_equity = pd.Series(BACKTEST_CONFIG['INITIAL_CAPITAL'], index=equity_curve.index)
            except Exception as bench_err:
                 print(f"警告: 处理基准数据时出错: {bench_err}。将不进行对比分析。")
                 benchmark_equity = pd.Series(BACKTEST_CONFIG['INITIAL_CAPITAL'], index=equity_curve.index)
        else:
            print("未加载基准数据或数据为空，将不进行对比分析或绘图。")
            benchmark_equity = pd.Series(BACKTEST_CONFIG['INITIAL_CAPITAL'], index=equity_curve.index)

        print("正在计算性能指标...")
        metrics = calculate_extended_metrics(
            portfolio_equity=equity_curve, benchmark_equity=benchmark_equity, portfolio_instance=final_portfolio
        )
        print("性能指标计算完毕。")
        display_metrics(metrics, benchmark_loaded=benchmark_loaded_successfully)

        print("正在绘制净值曲线图...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 8))
        (equity_curve / equity_curve.iloc[0]).plot(ax=ax, label=f'LLM Strategy ({SELECTED_UNIVERSE})', lw=2)
        if benchmark_loaded_successfully:
            (benchmark_equity / benchmark_equity.iloc[0]).plot(ax=ax, label=f'Benchmark ({BENCHMARK_DISPLAY_NAME})', lw=2, linestyle=':')
        ax.set_title(f'LLM Quant Strategy Backtest ({START_DATE} to {END_DATE}) - {SELECTED_UNIVERSE}', fontsize=16)
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Value (Initial = 1)')
        ax.legend(); ax.grid(True); plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"equity_curve_{SELECTED_UNIVERSE}_llm_{START_DATE}_to_{END_DATE}_{timestamp}.png"
        plot_path = os.path.join(PLOT_DIR, plot_filename)
        try: plt.savefig(plot_path); print(f"Equity curve plot saved to: {plot_path}")
        except Exception as e: print(f"保存图表时出错: {e}")
        plt.show()

        print("正在保存 AI 决策日志...")
        ai_log_df = strategy.get_trade_log()
        log_filename = f"ai_trade_log_{SELECTED_UNIVERSE}_{START_DATE}_to_{END_DATE}_{timestamp}.csv"
        log_path = os.path.join(LOG_DIR, log_filename)
        try: ai_log_df.to_csv(log_path, index=False, encoding='utf-8-sig'); print(f"AI 决策日志已保存至: {log_path}")
        except Exception as e: print(f"保存 AI 日志时出错: {e}")

        print("正在保存详细持仓记录...")
        holdings_df = final_portfolio.get_holdings_history()
        holdings_filename = f"holdings_history_{SELECTED_UNIVERSE}_llm_{START_DATE}_to_{END_DATE}_{timestamp}.csv"
        holdings_path = os.path.join(LOG_DIR, holdings_filename)
        try:
            if not holdings_df.empty: holdings_df['datetime'] = pd.to_datetime(holdings_df['datetime']).dt.strftime('%Y-%m-%d')
            holdings_df.to_csv(holdings_path, index=False, encoding='utf-8-sig'); print(f"持仓记录已保存至: {holdings_path}")
        except Exception as e: print(f"保存持仓记录时出错: {e}")

    except ValueError as analysis_ve:
        if "The truth value of a DataFrame is ambiguous" in str(analysis_ve): print(f"!!! 结果分析阶段捕获到 DataFrame 布尔值模糊错误: {analysis_ve}\n!!! 请检查 performance.py 或结果分析代码。")
        else: print(f"!!! 结果分析阶段捕获到 ValueError: {analysis_ve}")
        import traceback; traceback.print_exc(); sys.exit(1)
    except Exception as analysis_e: print(f"!!! 结果分析阶段发生未知错误: {analysis_e}"); import traceback; traceback.print_exc(); sys.exit(1)

    print("\n--- 回测脚本执行完毕 ---")