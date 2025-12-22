# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse
import yaml
from datetime import datetime

# --- 导入模块 ---
try:
    from llm_quant_lib.data_handler import DataHandler
    from llm_quant_lib.strategy import FactorTopNStrategy  # 切换为因子策略
    from llm_quant_lib.backtest_engine import BacktestEngine
    from llm_quant_lib.performance import calculate_extended_metrics, display_metrics
except ImportError as e:
    print(f"导入库出错: {e}")
    sys.exit(1)

def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

if __name__ == '__main__':
    # 1. 加载配置
    config = load_config()
    
    # 提取基础配置
    START_DATE = config['backtest'].get('start_date', '2018-01-01')
    END_DATE = config['backtest'].get('end_date', '2024-07-31')
    SELECTED_UNIVERSE = config['strategy']['factor_strategy'].get('universe_to_trade', 'All')
    
    # 2. 数据准备
    print(f"\n--- 阶段 1: 数据准备 (资产池: {SELECTED_UNIVERSE}) ---")
    paths_conf = config.get('paths', {})
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PRICE_PATH = os.path.join(BASE_DIR, paths_conf.get('price_data_csv'))
    UNIVERSE_DEFINITION_PATH = os.path.join(BASE_DIR, paths_conf.get('universe_definition'))

    # 初始化 DataHandler
    data_handler = DataHandler(
        db_config=None,
        csv_path=CSV_PRICE_PATH,
        start_date=START_DATE,
        end_date=END_DATE
    )
    data_handler.load_data()
    universe_df = data_handler.load_universe_data(UNIVERSE_DEFINITION_PATH)

    # 3. 初始化因子策略
    print("\n--- 阶段 2: 初始化因子选股策略 ---")
    strategy_conf = config['strategy']['factor_strategy']
    strategy = FactorTopNStrategy(
        universe_df=universe_df,
        factor_name=strategy_conf['factor_name'],
        top_n=strategy_conf['top_n'],
        ascending=strategy_conf.get('ascending', False),
        universe_to_trade=SELECTED_UNIVERSE
    )
    print(f"策略初始化成功: 使用因子 '{strategy_conf['factor_name']}', Top {strategy_conf['top_n']}")

    # 4. 执行回测
    print("\n--- 阶段 3: 执行回测 ---")
    BACKTEST_CONFIG = {
        'INITIAL_CAPITAL': config['backtest'].get('initial_capital', 1000000),
        'COMMISSION_RATE': config['backtest'].get('commission_rate', 0.001),
        'SLIPPAGE': config['backtest'].get('slippage', 0.0005),
        'REBALANCE_DAYS': config['backtest'].get('rebalance_days', 20), # 新增天数配置
        'REBALANCE_MONTHS': config['backtest'].get('rebalance_months', 1)
    }

    engine = BacktestEngine(
        start_date=START_DATE,
        end_date=END_DATE,
        config=BACKTEST_CONFIG,
        strategy=strategy,
        data_handler=data_handler,
        universe_to_run=SELECTED_UNIVERSE
    )
    portfolio_history, final_portfolio = engine.run()

    # 5. 结果展示
    print("\n--- 阶段 4: 结果分析 ---")
    equity_curve = portfolio_history['total_value']
    
    # 这里简单处理基准曲线（假设用初始资金对齐）
    metrics = calculate_extended_metrics(
        portfolio_equity=equity_curve,
        benchmark_equity=equity_curve, # 临时用自身替代以展示格式
        portfolio_instance=final_portfolio
    )
    display_metrics(metrics, benchmark_loaded=False)

    # 绘图保存
    plt.figure(figsize=(12, 6))
    (equity_curve / equity_curve.iloc[0]).plot(title=f"Factor Strategy: {strategy_conf['factor_name']}")
    plt.grid(True)
    plt.show()