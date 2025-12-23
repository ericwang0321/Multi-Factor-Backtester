# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import yaml
from datetime import datetime

# --- 导入模块 ---
try:
    # [修改 1] 导入新策略和因子引擎
    from quant_core.strategies.rules import LinearWeightedStrategy
    from quant_core.factors.engine import FactorEngine
    
    from quant_core.backtest_engine import BacktestEngine
    from quant_core.performance import calculate_extended_metrics, display_metrics
    from quant_core.data.query_helper import DataQueryHelper
except ImportError as e:
    print(f"导入库出错: {e}")
    sys.exit(1)

def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# [新增] 临时数据准备函数 (与 App 逻辑一致)
def prepare_factor_data(factor_engine, codes, factors, start_date, end_date):
    print(f"正在内存中计算因子数据: {factors} ...")
    
    # 确保数据已初始化
    if factor_engine.xarray_data is None:
        factor_engine._get_xarray_data()
        
    data_dict = {}
    for f_name in factors:
        # 计算全量
        df = factor_engine._compute_and_cache_factor(f_name)
        if not df.empty:
            # 截取 + 堆叠
            # 转换为 (datetime, sec_code) MultiIndex
            df_slice = df.loc[str(start_date):str(end_date)]
            valid_cols = [c for c in df_slice.columns if c in codes]
            if valid_cols:
                stacked = df_slice[valid_cols].stack()
                stacked.name = f_name
                data_dict[f_name] = stacked
            print(f"  - {f_name} 计算完成")
            
    if not data_dict:
        return pd.DataFrame()
        
    # 合并为宽表
    full_df = pd.concat(data_dict.values(), axis=1)
    full_df.index.names = ['datetime', 'sec_code']
    return full_df

if __name__ == '__main__':
    # 1. 加载配置
    config = load_config()
    
    # 提取基础配置
    START_DATE = config['backtest'].get('start_date', '2018-01-01')
    END_DATE = config['backtest'].get('end_date', '2024-07-31')
    SELECTED_UNIVERSE = config['strategy']['factor_strategy'].get('universe_to_trade', 'All')
    
    # 2. 数据准备
    print(f"\n--- 阶段 1: 数据准备 (资产池: {SELECTED_UNIVERSE}) ---")
    helper = DataQueryHelper(storage_path='data/processed/all_price_data.parquet')
    
    # 获取资产列表
    universe_df = helper.get_all_symbols()
    universe_codes = universe_df['sec_code'].tolist()
    print(f"基础数据加载完成。总标的数: {len(universe_df)}")

    # 3. 初始化因子策略
    print("\n--- 阶段 2: 初始化策略与因子计算 ---")
    strategy_conf = config['strategy']['factor_strategy']
    
    # 解析配置中的因子
    # 兼容旧配置：如果 config 只有 factor_name，转为权重 1.0
    if 'weights' in strategy_conf:
        factor_weights = strategy_conf['weights']
    else:
        # 旧配置兼容
        f_name = strategy_conf.get('factor_name', 'rsi')
        factor_weights = {f_name: 1.0}
    
    factor_list = list(factor_weights.keys())
    
    # [关键步骤] 实例化因子引擎并准备数据
    # 这里是新架构的核心：策略运行前，数据必须就位
    f_engine = FactorEngine(query_helper=helper)
    factor_data = prepare_factor_data(
        f_engine, universe_codes, factor_list, START_DATE, END_DATE
    )
    
    if factor_data.empty:
        print("❌ 错误：未能计算出任何因子数据，请检查数据源或因子名称。")
        sys.exit(1)

    # 实例化新策略
    strategy = LinearWeightedStrategy(
        name="CLI_Linear_Strategy",
        weights=factor_weights,
        top_k=strategy_conf.get('top_n', 5)
    )
    
    # [关键步骤] 注入数据
    strategy.load_data(factor_data)
    print("✅ 策略初始化及数据注入完成。")

    # 4. 执行回测
    print("\n--- 阶段 3: 执行回测 ---")
    BACKTEST_CONFIG = {
        'INITIAL_CAPITAL': config['backtest'].get('initial_capital', 1000000),
        'COMMISSION_RATE': config['backtest'].get('commission_rate', 0.001),
        'SLIPPAGE': config['backtest'].get('slippage', 0.0005),
        'REBALANCE_DAYS': config['backtest'].get('rebalance_days', 20),
        'REBALANCE_MONTHS': config['backtest'].get('rebalance_months', 1)
    }

    # 实例化回测引擎
    engine = BacktestEngine(
        start_date=START_DATE,
        end_date=END_DATE,
        config=BACKTEST_CONFIG,
        strategy=strategy,
        query_helper=helper,
        universe_to_run=SELECTED_UNIVERSE
    )
    
    # 运行
    # 注意：engine.factor_engine.current_weights 不需要再设置了，策略自己全权负责
    portfolio_history, final_portfolio = engine.run()

    # 5. 结果展示
    print("\n--- 阶段 4: 结果分析 ---")
    
    # 尝试获取基准 (这里简化处理，尝试读 CSV，读不到就画平线)
    benchmark_equity = None
    try:
        # 尝试使用 Helper 获取基准 (如果你的 Helper 有这个功能)
        # 这里假设用 SPY 做基准
        bench_ret = helper.get_benchmark_returns('SPY')
        if not bench_ret.empty:
            bench_ret = bench_ret.loc[START_DATE:END_DATE]
            benchmark_equity = (1 + bench_ret).cumprod() * BACKTEST_CONFIG['INITIAL_CAPITAL']
            benchmark_equity = benchmark_equity.reindex(portfolio_history.index, method='ffill').fillna(BACKTEST_CONFIG['INITIAL_CAPITAL'])
    except Exception:
        pass
        
    if benchmark_equity is None:
        print("⚠️ 未找到基准数据，使用无风险基准。")
        benchmark_equity = pd.Series(BACKTEST_CONFIG['INITIAL_CAPITAL'], index=portfolio_history.index)

    equity_curve = portfolio_history['total_value']
    
    metrics = calculate_extended_metrics(
        portfolio_equity=equity_curve,
        benchmark_equity=benchmark_equity,
        portfolio_instance=final_portfolio
    )
    display_metrics(metrics, benchmark_loaded=True)

    # 简单绘图
    plt.figure(figsize=(12, 6))
    strat_norm = equity_curve / equity_curve.iloc[0]
    bench_norm = benchmark_equity / benchmark_equity.iloc[0]
    
    strat_norm.plot(label='Strategy', linewidth=2)
    bench_norm.plot(label='Benchmark', linestyle='--', alpha=0.7)
    
    plt.title(f"Backtest Result: {list(factor_weights.keys())}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = f"backtest_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_path)
    print(f"\n📊 结果图表已保存至: {output_path}")