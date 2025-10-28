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
    default_topn = config.get('strategy', {}).get('llm', {}).get('default_top_n', 5)

    parser.add_argument('--universe', type=str, default=default_universe,
                        help=f"要回测的资产池名称 (默认: {default_universe})")
    parser.add_argument('--start', type=str, default=default_start,
                        help=f"回测开始日期 YYYY-MM-DD (默认: {default_start})")
    parser.add_argument('--end', type=str, default=default_end,
                        help=f"回测结束日期 YYYY-MM-DD (默认: {default_end})")
    parser.add_argument('--topn', type=int, default=default_topn,
                        help=f"让 AI 选择的 Top N 资产数量 (默认: {default_topn})")
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
    TOP_N = args.topn

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
        if not DB_CONFIG['password']:
             print("警告: 数据库密码未在环境变量 DB_PASSWORD 中设置!")
             # 根据需要，可以选择退出或继续（如果数据库不需要密码）
             # sys.exit(1)

    # 文件路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 脚本所在目录通常是根目录
    CSV_PRICE_PATH = os.path.join(BASE_DIR, paths_conf.get('price_data_csv', 'data/processed/default_price.csv'))
    UNIVERSE_DEFINITION_PATH = os.path.join(BASE_DIR, paths_conf.get('universe_definition', 'data/reference/default_universe.csv'))

    # 动态设置基准文件路径
    BENCHMARK_MAPPING = paths_conf.get('benchmark_mapping', {})
    BENCHMARK_FILENAME = BENCHMARK_MAPPING.get(SELECTED_UNIVERSE)
    BENCHMARK_DATA_PATH = None
    if BENCHMARK_FILENAME:
        BENCHMARK_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', BENCHMARK_FILENAME) # 假设基准文件在 processed 目录下
        if not os.path.exists(BENCHMARK_DATA_PATH):
             print(f"警告: 在 config.yaml 中为 '{SELECTED_UNIVERSE}' 配置的基准文件 '{BENCHMARK_FILENAME}' 未找到。")
             BENCHMARK_DATA_PATH = None # 重置为 None
    if not BENCHMARK_DATA_PATH:
         print(f"警告: 未找到或未配置资产池 '{SELECTED_UNIVERSE}' 的基准文件，将不计算相对指标。")


    # 回测核心参数
    BACKTEST_CONFIG = {
        'INITIAL_CAPITAL': backtest_conf.get('initial_capital', 1000000),
        'COMMISSION_RATE': backtest_conf.get('commission_rate', 0.001),
        'SLIPPAGE': backtest_conf.get('slippage', 0.0005),
        'REBALANCE_MONTHS': backtest_conf.get('rebalance_months', 1),
        # 以下供 DataHandler 使用
        'DB_CONFIG': DB_CONFIG,
        'PRICE_DATA_PATH': CSV_PRICE_PATH # DataHandler 需要知道 CSV 路径
    }

    # AI 策略特定参数
    # API Key 必须通过环境变量 DEEPSEEK_API_KEY 提供
    AI_STRATEGY_CONFIG = {
        'universe_to_trade': SELECTED_UNIVERSE, # 使用命令行或默认值
        'top_n': TOP_N                        # 使用命令行或默认值
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
    try:
        data_handler = DataHandler(
            db_config=DB_CONFIG,
            csv_path=CSV_PRICE_PATH, # 传入 CSV 路径
            start_date=START_DATE,
            end_date=END_DATE
        )
        # load_data 会优先尝试 CSV，失败或需要DB时才连接数据库
        raw_df = data_handler.load_data()
        universe_df = data_handler.load_universe_data(UNIVERSE_DEFINITION_PATH)

        benchmark_df = None
        if BENCHMARK_DATA_PATH: # 只有路径有效才加载
            benchmark_df = data_handler.load_benchmark_data(BENCHMARK_DATA_PATH)

        print(f"数据加载完毕。回测范围: {raw_df['datetime'].min().date()} 到 {raw_df['datetime'].max().date()}")

    except FileNotFoundError as e:
        print(f"错误: 必需的数据文件未找到: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"数据准备阶段出错: {e}")
        sys.exit(1)

    # --- 4. 初始化策略 ---
    print("\n--- 阶段 2: 初始化策略 ---")
    try:
        # LLMStrategy 会自动查找环境变量 DEEPSEEK_API_KEY
        strategy = LLMStrategy(
            universe_df=universe_df,
            **AI_STRATEGY_CONFIG
        )
        print(f"策略 '{strategy.__class__.__name__}' 初始化成功。")
        print(f"  - 资产池: {AI_STRATEGY_CONFIG['universe_to_trade']}")
        print(f"  - 选股数量: {AI_STRATEGY_CONFIG['top_n']}")
        if os.getenv("DEEPSEEK_API_KEY"):
             print("  - API Key: 已从环境变量 DEEPSEEK_API_KEY 加载")
        else:
             # LLMStrategy 初始化时会报错，理论上不会执行到这里
             print("  - 警告: 环境变量 DEEPSEEK_API_KEY 未设置!")

    except ValueError as e: # 捕获 API Key 未找到的错误
        print(f"错误: 初始化 AI 策略失败: {e}")
        print("请确保已正确设置 DEEPSEEK_API_KEY 环境变量。")
        sys.exit(1)
    except Exception as e:
        print(f"初始化 AI 策略时发生未知错误: {e}")
        sys.exit(1)

    # --- 5. 运行回测 ---
    print("\n--- 阶段 3: 执行回测 ---")
    try:
        engine = BacktestEngine(
            start_date=START_DATE,
            end_date=END_DATE,
            config=BACKTEST_CONFIG,
            strategy=strategy,
            data_handler=data_handler # 传入 data_handler 实例
        )
        portfolio_history, final_portfolio = engine.run()

        if not portfolio_history:
            print("回测未生成任何结果，可能由于数据不足或配置错误。")
            sys.exit(1)
        print("回测执行完毕。")

    except ValueError as e:
        print(f"错误: 回测引擎初始化失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"回测运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- 6. 结果分析与展示 ---
    print("\n--- 阶段 4: 结果分析与展示 ---")

    # (结果分析和展示部分与上一个版本相同，无需修改)
    # ... (省略重复代码，请参考上一个回复中的结果分析部分) ...
    # 1. 处理回测结果
    equity_curve = pd.DataFrame(portfolio_history).set_index('datetime')['total_value']
    equity_curve.index = pd.to_datetime(equity_curve.index) # 确保索引是 DatetimeIndex

    # 2. 准备基准净值曲线
    benchmark_equity = pd.Series(dtype=float)
    if benchmark_df is not None and not benchmark_df.empty:
        print("正在准备基准净值曲线...")
        initial_capital = BACKTEST_CONFIG['INITIAL_CAPITAL']
        benchmark_df.index = pd.to_datetime(benchmark_df.index)
        # 确保基准日期覆盖回测日期，并向前填充
        benchmark_returns_aligned = benchmark_df['benchmark_return'].reindex(equity_curve.index, method='ffill').fillna(0.0)
        # 再次检查是否有前导 NaN (如果基准开始晚于策略)
        first_valid_idx = benchmark_returns_aligned.first_valid_index()
        if first_valid_idx is not None and first_valid_idx > equity_curve.index[0]:
             benchmark_returns_aligned.loc[:first_valid_idx] = 0.0 # 将开始前的收益设为0
        benchmark_equity = initial_capital * (1 + benchmark_returns_aligned).cumprod()
        # 确保起始值准确
        benchmark_equity.iloc[0] = initial_capital
        print("基准净值曲线准备完毕。")
    else:
        print("未加载基准数据或数据为空，将不进行对比分析或绘图。")
        benchmark_equity = pd.Series(BACKTEST_CONFIG['INITIAL_CAPITAL'], index=equity_curve.index)


    # 3. 计算性能指标
    print("正在计算性能指标...")
    metrics = calculate_extended_metrics(
        portfolio_equity=equity_curve,
        benchmark_equity=benchmark_equity,
        portfolio_instance=final_portfolio # 传入 Portfolio 实例以获取换手率
    )
    print("性能指标计算完毕。")

    # 4. 打印指标
    display_metrics(metrics, benchmark_loaded=(benchmark_df is not None and not benchmark_df.empty))

    # 5. 绘製淨值曲线图
    print("正在绘制净值曲线图...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # 绘制归一化净值
    (equity_curve / equity_curve.iloc[0]).plot(ax=ax, label=f'LLM Strategy ({SELECTED_UNIVERSE})', lw=2)
    # 只有在 benchmark 实际加载且非空时才绘制
    if benchmark_df is not None and not benchmark_df.empty:
        benchmark_display_name = BENCHMARK_FILENAME or "Benchmark"
        (benchmark_equity / benchmark_equity.iloc[0]).plot(ax=ax, label=f'Benchmark ({benchmark_display_name})', lw=2, linestyle=':')

    ax.set_title(f'LLM 量化策略回测 ({START_DATE} to {END_DATE}) - {SELECTED_UNIVERSE}', fontsize=16)
    ax.set_xlabel('日期')
    ax.set_ylabel('归一化净值 (初始值=1)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # 保存图表
    # 使用 datetime 生成更独特的文件名，避免覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"equity_curve_{SELECTED_UNIVERSE}_llm_{START_DATE}_to_{END_DATE}_{timestamp}.png"
    plot_path = os.path.join(PLOT_DIR, plot_filename)
    try:
        plt.savefig(plot_path)
        print(f"净值曲线图已保存至: {plot_path}")
    except Exception as e:
        print(f"保存图表时出错: {e}")
    plt.show()

    # 6. 保存 AI 决策日志
    print("正在保存 AI 决策日志...")
    ai_log_df = strategy.get_trade_log()
    log_filename = f"ai_trade_log_{SELECTED_UNIVERSE}_{START_DATE}_to_{END_DATE}_{timestamp}.csv"
    log_path = os.path.join(LOG_DIR, log_filename)
    try:
        ai_log_df.to_csv(log_path, index=False, encoding='utf-8-sig')
        print(f"AI 决策日志已保存至: {log_path}")
    except Exception as e:
        print(f"保存 AI 日志时出错: {e}")

    # 7. 保存持仓记录
    print("正在保存详细持仓记录...")
    holdings_df = pd.DataFrame(final_portfolio.holdings_history)
    holdings_filename = f"holdings_history_{SELECTED_UNIVERSE}_llm_{START_DATE}_to_{END_DATE}_{timestamp}.csv"
    holdings_path = os.path.join(LOG_DIR, holdings_filename)
    try:
        if not holdings_df.empty:
            holdings_df['datetime'] = pd.to_datetime(holdings_df['datetime']).dt.strftime('%Y-%m-%d')
        holdings_df.to_csv(holdings_path, index=False, encoding='utf-8-sig')
        print(f"持仓记录已保存至: {holdings_path}")
    except Exception as e:
        print(f"保存持仓记录时出错: {e}")

    print("\n--- 回测脚本执行完毕 ---")