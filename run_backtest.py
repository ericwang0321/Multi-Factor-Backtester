# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import yaml
from datetime import datetime

# --- 导入模块 ---
try:
    from quant_core.strategies.rules import LinearWeightedStrategy
    from quant_core.backtest_engine import BacktestEngine
    from quant_core.performance import calculate_extended_metrics, display_metrics
    from quant_core.data.query_helper import DataQueryHelper
except ImportError as e:
    print(f"导入库出错: {e}")
    sys.exit(1)

def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# [新增] 读取本地 parquet 因子的函数
def load_offline_factors(factor_names, start_date, end_date, universe_codes):
    """
    从 data/processed/factors/ 读取预计算好的因子文件
    """
    base_dir = 'data/processed/factors'
    loaded_data = {}
    
    print(f"正在加载离线因子数据: {factor_names} ...")
    
    for f_name in factor_names:
        file_path = os.path.join(base_dir, f"{f_name}.parquet")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"❌ 错误: 找不到因子文件 {file_path}")
            print(f"👉 请先运行 'python run_factor_computation.py' 生成因子数据！")
            sys.exit(1)
            
        # 1. 读取 Parquet (宽表: Index=Date, Cols=Stocks)
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"❌ 读取文件 {f_name} 失败: {e}")
            sys.exit(1)
        
        # 2. 时间切片
        df = df.loc[str(start_date):str(end_date)]
        
        # 3. 过滤 Universe (只保留当前资产池中的股票列)
        valid_cols = [c for c in df.columns if c in universe_codes]
        if not valid_cols:
            print(f"⚠️ 警告: 因子 {f_name} 在当前资产池({len(universe_codes)})中没有数据。")
            continue
            
        df = df[valid_cols]
        
        # 4. 堆叠 (Stack) 为 Series，方便后续合并
        stacked = df.stack()
        stacked.name = f_name
        loaded_data[f_name] = stacked
        
    if not loaded_data:
        return pd.DataFrame()
        
    # 5. 合并为大表 (Strategy 需要的格式)
    full_df = pd.concat(loaded_data.values(), axis=1)
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
    
    # 初始化 Helper
    helper = DataQueryHelper(storage_path='data/processed/all_price_data.parquet')
    
    # 获取资产列表
    universe_df = helper.get_all_symbols()
    universe_codes = universe_df['sec_code'].tolist()
    print(f"基础数据加载完成。总标的数: {len(universe_df)}")

    # 3. 初始化因子策略
    print("\n--- 阶段 2: 初始化策略与因子加载 ---")
    strategy_conf = config['strategy']['factor_strategy']
    
    # 解析因子权重
    if 'weights' in strategy_conf:
        factor_weights = strategy_conf['weights']
    else:
        # 兼容旧配置
        f_name = strategy_conf.get('factor_name', 'rsi')
        factor_weights = {f_name: 1.0}
    
    factor_list = list(factor_weights.keys())
    
    # 解析风控配置 (新增)
    risk_conf = strategy_conf.get('risk_management', {})
    
    # [关键步骤] 加载离线因子数据
    factor_data = load_offline_factors(
        factor_list, START_DATE, END_DATE, universe_codes
    )
    
    if factor_data.empty:
        print("❌ 错误：未能加载任何因子数据，无法启动回测。")
        sys.exit(1)

    # 实例化新策略 (注入风控参数)
    strategy = LinearWeightedStrategy(
        name="Offline_Linear_Strategy",
        weights=factor_weights,
        top_k=strategy_conf.get('top_n', 5),
        # --- 【修改】传入 Config 中的风控参数 ---
        stop_loss_pct=risk_conf.get('stop_loss_pct'),
        max_pos_weight=risk_conf.get('max_pos_weight'),
        max_drawdown_pct=risk_conf.get('max_drawdown_pct')
    )
    
    # 注入数据
    strategy.load_data(factor_data)
    print("✅ 策略初始化及离线数据注入完成。")

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
    portfolio_history, final_portfolio = engine.run()

    # 5. 结果展示
    print("\n--- 阶段 4: 结果分析 ---")
    
    # 尝试获取基准
    benchmark_equity = None
    try:
        bench_symbol = 'SPY' 
        bench_ret = helper.get_benchmark_returns(bench_symbol)
        
        if not bench_ret.empty:
            bench_ret = bench_ret.loc[START_DATE:END_DATE]
            benchmark_equity = (1 + bench_ret).cumprod() * BACKTEST_CONFIG['INITIAL_CAPITAL']
            benchmark_equity = benchmark_equity.reindex(portfolio_history.index, method='ffill').fillna(BACKTEST_CONFIG['INITIAL_CAPITAL'])
    except Exception as e:
        print(f"⚠️ 基准数据获取失败 ({e})，使用平线基准。")
        pass
        
    if benchmark_equity is None:
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
    
    plt.title(f"Backtest: {list(factor_weights.keys())} (StopLoss: {risk_conf.get('stop_loss_pct')})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = f"backtest_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_path)
    print(f"\n📊 结果图表已保存至: {output_path}")