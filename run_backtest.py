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
    print(f"❌ 导入库出错: {e}")
    sys.exit(1)

# [递归合并配置]
def recursive_update(base_dict, update_dict):
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            recursive_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def load_combined_configs(file_list):
    final_config = {}
    print(f"⚙️  正在加载配置序列: {file_list}")
    for config_path in file_list:
        if not os.path.exists(config_path):
            print(f"❌ 错误: 找不到配置文件: {config_path}")
            sys.exit(1)
        with open(config_path, 'r', encoding='utf-8') as f:
            current_conf = yaml.safe_load(f) or {}
            recursive_update(final_config, current_conf)
    return final_config

# [读取本地 parquet 因子]
def load_offline_factors(factor_names, start_date, end_date, universe_codes, data_dir):
    loaded_data = {}
    print(f"📂 正在加载因子数据: {factor_names} ...")
    
    for f_name in factor_names:
        file_path = os.path.join(data_dir, f"{f_name}.parquet")
        if not os.path.exists(file_path):
            print(f"❌ 找不到因子文件: {file_path}")
            sys.exit(1)
        try:
            df = pd.read_parquet(file_path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df.sort_index().loc[str(start_date):str(end_date)]
            valid_cols = df.columns.intersection(universe_codes)
            if valid_cols.empty:
                continue
            stacked = df[valid_cols].stack()
            stacked.name = f_name
            loaded_data[f_name] = stacked
        except Exception as e:
            print(f"❌ 读取 {f_name} 失败: {e}")
            sys.exit(1)
        
    if not loaded_data:
        return pd.DataFrame()
    full_df = pd.concat(loaded_data.values(), axis=1)
    full_df.index.names = ['datetime', 'sec_code']
    return full_df

if __name__ == '__main__':
    # 1. 加载配置
    CONFIG_FILES = ['config/base.yaml', 'config/backtest.yaml']
    config = load_combined_configs(CONFIG_FILES)
    
    # 提取顶层配置
    bt_conf = config.get('backtest', {})
    strat_conf = config.get('strategy', {})  # 这里拿到了整个 strategy 块
    
    START_DATE = bt_conf.get('start_date', '2018-01-01')
    END_DATE = bt_conf.get('end_date', '2024-07-31')
    DATA_HOME = config.get('data_home', 'data/processed')
    PRICE_PATH = os.path.join(DATA_HOME, 'all_price_data.parquet')
    FACTOR_DIR = os.path.join(DATA_HOME, 'factors')

    # 2. 数据准备
    print("\n--- 阶段 1: 数据准备 ---")
    helper = DataQueryHelper(storage_path=PRICE_PATH)
    universe_df = helper.get_all_symbols()
    universe_codes = universe_df['sec_code'].tolist()
    print(f"✅ 基础数据加载完成。总标的数: {len(universe_df)}")

    # 3. 策略参数解析 (核心修改处！！！)
    print("\n--- 阶段 2: 初始化策略与因子加载 ---")
    
    # A. 获取通用参数 (Common)
    common_conf = strat_conf.get('common', {})
    risk_conf = common_conf.get('risk', {}) # 你的 yaml 里叫 risk
    top_k = common_conf.get('top_k', 3)
    
    # B. 判断策略类型并提取参数
    strat_type = strat_conf.get('type', 'linear')
    print(f"ℹ️  当前策略模式: {strat_type}")

    weights = {}
    if strat_type == 'linear':
        # 进去 linear_params 里找 weights
        lin_params = strat_conf.get('linear_params', {})
        weights = lin_params.get('weights', {})
    else:
        print(f"⚠️ 暂不支持的策略类型: {strat_type}")
        sys.exit(1)

    if not weights:
        print("❌ 错误: 在 'linear_params' 中未找到 'weights'，请检查 config/backtest.yaml")
        sys.exit(1)
    
    # C. 加载因子
    factor_list = list(weights.keys())
    factor_data = load_offline_factors(
        factor_list, START_DATE, END_DATE, universe_codes, FACTOR_DIR
    )
    
    if factor_data.empty:
        print("❌ 无法加载因子数据，退出。")
        sys.exit(1)

    # 4. 实例化策略
    # 注意：这里把解析出来的 risk 参数传进去
    strategy = LinearWeightedStrategy(
        name=common_conf.get('name', "Backtest_Strategy"),
        weights=weights,
        top_k=top_k,
        stop_loss_pct=risk_conf.get('stop_loss_pct'),
        max_pos_weight=risk_conf.get('max_pos_weight'),
        max_drawdown_pct=risk_conf.get('max_drawdown_pct')
    )
    
    strategy.load_data(factor_data)
    print(f"✅ 策略就绪。权重: {weights}")
    print(f"🛡️  风控配置: {risk_conf}")

    # 5. 运行回测
    print("\n--- 阶段 3: 运行回测 ---")
    engine_config = {
        'INITIAL_CAPITAL': bt_conf.get('initial_capital', 1000000),
        'COMMISSION_RATE': bt_conf.get('commission_rate', 0.001),
        'SLIPPAGE': 0.0005,
        'BENCHMARK': bt_conf.get('benchmark', 'SPY'),
        'REBALANCE_DAYS': 20
    }

    engine = BacktestEngine(
        start_date=START_DATE,
        end_date=END_DATE,
        config=engine_config,
        strategy=strategy,
        query_helper=helper,
        universe_to_run='All'
    )
    
    portfolio_history, final_portfolio = engine.run()

    # 6. 结果展示
    print("\n--- 阶段 4: 结果分析 ---")
    
    benchmark_equity = None
    try:
        bench_ret = helper.get_benchmark_returns(engine_config['BENCHMARK'])
        if not bench_ret.empty:
            if not isinstance(bench_ret.index, pd.DatetimeIndex):
                bench_ret.index = pd.to_datetime(bench_ret.index)
            bench_ret = bench_ret.sort_index().loc[START_DATE:END_DATE]
            benchmark_equity = (1 + bench_ret).cumprod() * engine_config['INITIAL_CAPITAL']
            benchmark_equity = benchmark_equity.reindex(portfolio_history.index, method='ffill').fillna(engine_config['INITIAL_CAPITAL'])
    except:
        pass
    
    if benchmark_equity is None:
        benchmark_equity = pd.Series(engine_config['INITIAL_CAPITAL'], index=portfolio_history.index)

    metrics = calculate_extended_metrics(portfolio_history['total_value'], benchmark_equity, final_portfolio)
    display_metrics(metrics, benchmark_loaded=True)

    plt.figure(figsize=(12, 6))
    (portfolio_history['total_value'] / portfolio_history['total_value'].iloc[0]).plot(label='Strategy')
    (benchmark_equity / benchmark_equity.iloc[0]).plot(label='Benchmark', linestyle='--')
    plt.legend()
    plt.title(f"Backtest: {list(weights.keys())}")
    
    if not os.path.exists('results'): os.makedirs('results')
    plt.savefig(f"results/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    print("\n✅ 回测完成")