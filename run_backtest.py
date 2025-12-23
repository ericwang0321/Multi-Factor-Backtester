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
    from llm_quant_lib.strategy import FactorTopNStrategy
    from llm_quant_lib.backtest_engine import BacktestEngine
    from llm_quant_lib.performance import calculate_extended_metrics, display_metrics
    # [修改 1] 引入新的数据查询助手，替代旧的 DataHandler
    from llm_quant_lib.data.query_helper import DataQueryHelper
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
    
    # 2. 数据准备 (Parquet 模式)
    print(f"\n--- 阶段 1: 数据准备 (资产池: {SELECTED_UNIVERSE}) ---")
    
    # [修改 2] 初始化 DataQueryHelper
    # 确保这里的路径指向你真实存在的 parquet 文件
    helper = DataQueryHelper(storage_path='data/processed/all_price_data.parquet')
    
    # 获取资产列表 (Universe)
    # get_all_symbols 返回包含 sec_code 和 category_id 的 DataFrame
    universe_df = helper.get_all_symbols()
    print(f"数据加载完成。总标的数: {len(universe_df)}")

    # 3. 初始化因子策略
    print("\n--- 阶段 2: 初始化因子选股策略 ---")
    strategy_conf = config['strategy']['factor_strategy']
    
    # [修改 3] 适配新的多因子权重逻辑
    # 将配置文件中的单个 factor_name 转换为权重字典 {name: 1.0}
    factor_name = strategy_conf['factor_name']
    factor_weights = {factor_name: 1.0}
    
    strategy = FactorTopNStrategy(
        universe_df=universe_df,
        factor_weights=factor_weights, # 使用权重字典
        top_n=strategy_conf['top_n'],
        universe_to_trade=SELECTED_UNIVERSE
    )
    print(f"策略初始化成功: 使用因子 '{factor_name}', Top {strategy_conf['top_n']}")

    # 4. 执行回测
    print("\n--- 阶段 3: 执行回测 ---")
    BACKTEST_CONFIG = {
        'INITIAL_CAPITAL': config['backtest'].get('initial_capital', 1000000),
        'COMMISSION_RATE': config['backtest'].get('commission_rate', 0.001),
        'SLIPPAGE': config['backtest'].get('slippage', 0.0005),
        'REBALANCE_DAYS': config['backtest'].get('rebalance_days', 20),
        'REBALANCE_MONTHS': config['backtest'].get('rebalance_months', 1)
    }

    # [修改 4] 实例化引擎并传入 query_helper
    engine = BacktestEngine(
        start_date=START_DATE,
        end_date=END_DATE,
        config=BACKTEST_CONFIG,
        strategy=strategy,
        query_helper=helper, # 关键：传入 helper 而不是 data_handler
        universe_to_run=SELECTED_UNIVERSE
    )
    
    # [修改 5] 必须注入当前使用的因子权重，否则 FactorEngine 不知道算哪个因子
    engine.factor_engine.current_weights = factor_weights
    
    # 运行
    portfolio_history, final_portfolio = engine.run()

    # 5. 结果展示
    print("\n--- 阶段 4: 结果分析 ---")
    
    # 获取回测区间的基准数据 (为了计算超额收益)
    # 这里尝试读取 SPXT 作为默认基准，如果读不到则使用策略自身的起始资金做平线
    benchmark_equity = None
    try:
        bench_df = pd.read_csv('data/processed/spxt_index_daily_return.csv')
        bench_df['report_date'] = pd.to_datetime(bench_df['report_date'])
        bench_df = bench_df.set_index('report_date').sort_index()
        # 截取对应时间段
        b_rets = bench_df.loc[pd.to_datetime(START_DATE):pd.to_datetime(END_DATE), 'default']
        # 计算净值曲线
        benchmark_equity = (1 + b_rets).cumprod() * BACKTEST_CONFIG['INITIAL_CAPITAL']
        # 对齐索引
        benchmark_equity = benchmark_equity.reindex(portfolio_history.index, method='ffill')
    except Exception as e:
        print(f"⚠️ 无法加载基准数据 ({e})，将使用无风险基准。")
        benchmark_equity = pd.Series(BACKTEST_CONFIG['INITIAL_CAPITAL'], index=portfolio_history.index)

    equity_curve = portfolio_history['total_value']
    
    metrics = calculate_extended_metrics(
        portfolio_equity=equity_curve,
        benchmark_equity=benchmark_equity,
        portfolio_instance=final_portfolio
    )
    display_metrics(metrics, benchmark_loaded=True)

    # 绘图保存
    plt.figure(figsize=(12, 6))
    
    # 归一化净值曲线 (从1.0开始)
    strat_norm = equity_curve / equity_curve.iloc[0]
    bench_norm = benchmark_equity / benchmark_equity.iloc[0]
    
    strat_norm.plot(label='Strategy', linewidth=2)
    bench_norm.plot(label='Benchmark (SP500)', linestyle='--', alpha=0.7)
    
    plt.title(f"Factor Strategy Backtest: {factor_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    output_path = f"backtest_result_{factor_name}_{datetime.now().strftime('%Y%m%d')}.png"
    plt.savefig(output_path)
    print(f"\n📊 结果图表已保存至: {output_path}")
    plt.show() # 如果在服务器运行，可以注释掉这行