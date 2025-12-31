# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import yaml
from datetime import datetime
import traceback

# --- 导入模块 ---
try:
    # [修改] 引入工厂函数，不再引入具体的策略类
    from quant_core.strategies import create_strategy_instance
    
    from quant_core.backtest_engine import BacktestEngine
    from quant_core.performance import calculate_extended_metrics, display_metrics
    from quant_core.data.query_helper import DataQueryHelper
except ImportError as e:
    print(f"❌ 导入库出错: {e}")
    print("👉 请确保在项目根目录下运行，且 quant_core 包在 PYTHONPATH 中")
    sys.exit(1)

# ==========================================
# 🛠️ 辅助函数：配置加载与合并
# ==========================================

def recursive_update(base_dict, update_dict):
    """
    递归合并两个字典。
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            recursive_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def load_combined_configs(file_list):
    """
    按顺序加载并合并多个 YAML 配置文件。
    """
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

# ==========================================
# 📂 辅助函数：因子数据加载
# ==========================================

def load_offline_factors(factor_names, start_date, end_date, universe_codes, data_dir):
    """
    从指定目录读取 parquet 格式的因子文件，并对齐时间和标的。
    """
    loaded_data = {}
    print(f"📂 正在加载因子数据: {factor_names} ...")
    
    if not os.path.exists(data_dir):
        print(f"❌ 因子数据目录不存在: {data_dir}")
        sys.exit(1)
    
    for f_name in factor_names:
        file_path = os.path.join(data_dir, f"{f_name}.parquet")
        
        if not os.path.exists(file_path):
            print(f"❌ 错误: 找不到因子文件 {file_path}")
            print(f"👉 请先运行因子计算脚本生成该因子数据。")
            sys.exit(1)
            
        try:
            # 1. 读取 Parquet
            df = pd.read_parquet(file_path)
            
            # 2. 确保索引是时间格式
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # 3. 时间切片
            df = df.sort_index().loc[str(start_date):str(end_date)]
            
            # 4. 资产过滤 (只保留当前 Universe 的列)
            valid_cols = df.columns.intersection(universe_codes)
            if valid_cols.empty:
                print(f"⚠️ 警告: 因子 {f_name} 在当前资产池中无匹配数据，跳过。")
                continue
                
            df = df[valid_cols]
            
            # 5. 堆叠 (Stack) 为 Series 以便合并
            stacked = df.stack()
            stacked.name = f_name
            loaded_data[f_name] = stacked
            
        except Exception as e:
            print(f"❌ 处理因子 {f_name} 时发生异常: {e}")
            sys.exit(1)
        
    if not loaded_data:
        return pd.DataFrame()
        
    # 6. 合并所有因子为一张大表 (MultiIndex: datetime, sec_code)
    full_df = pd.concat(loaded_data.values(), axis=1)
    full_df.index.names = ['datetime', 'sec_code']
    
    return full_df

# ==========================================
# 🚀 主程序入口
# ==========================================

if __name__ == '__main__':
    # -----------------------------------------------------------
    # 1. 加载配置 (Base + Backtest)
    # -----------------------------------------------------------
    CONFIG_FILES = [
        'config/base.yaml',      
        'config/backtest.yaml'   
    ]
    
    config = load_combined_configs(CONFIG_FILES)
    
    # 提取各部分配置
    bt_conf = config.get('backtest', {})
    
    # 解析关键参数
    START_DATE = bt_conf.get('start_date', '2018-01-01')
    END_DATE = bt_conf.get('end_date', '2024-07-31')
    
    # [新增] 读取 Universe 设置 (默认为 'All')
    TARGET_UNIVERSE = bt_conf.get('universe', 'All')

    # 解析路径
    DATA_HOME = config.get('data_home', 'data/processed')
    PRICE_PATH = os.path.join(DATA_HOME, 'all_price_data.parquet')
    FACTOR_DIR = os.path.join(DATA_HOME, 'factors')

    # -----------------------------------------------------------
    # 2. 数据准备
    # -----------------------------------------------------------
    print(f"\n--- 阶段 1: 数据准备 ---")
    
    if not os.path.exists(PRICE_PATH):
        print(f"❌ 找不到价格数据文件: {PRICE_PATH}")
        sys.exit(1)

    helper = DataQueryHelper(storage_path=PRICE_PATH)
    
    # [新增] 根据 Universe 过滤资产池
    print(f"🎯 正在加载资产池: {TARGET_UNIVERSE}")
    universe_df = helper.get_all_symbols()
    
    if TARGET_UNIVERSE != 'All':
        if 'category_id' in universe_df.columns:
            # 筛选 category_id 匹配的资产
            universe_df = universe_df[universe_df['category_id'] == TARGET_UNIVERSE]
        else:
            print("⚠️ 警告: 数据中缺少 category_id 列，无法按 Universe 筛选，回退到 All。")

    universe_codes = universe_df['sec_code'].tolist()
    print(f"✅ 基础数据加载完成。本次回测标的数: {len(universe_codes)}")

    # -----------------------------------------------------------
    # 3. 初始化策略 (工厂模式)
    # -----------------------------------------------------------
    print("\n--- 阶段 2: 初始化策略与因子加载 ---")
    
    # 获取策略配置根节点
    strat_conf = config.get('strategy', {})
    
    try:
        # [核心重构点] 🏭 使用工厂自动创建策略实例
        # 无论 Linear 还是 ML，这里都不需要改代码
        strategy = create_strategy_instance(strat_conf)
        
        # [依赖反转] 🔗 让策略告诉我们需要加载哪些因子
        required_factors = strategy.get_required_factors()
        print(f"📋 策略 [{strategy.name}] 声明依赖因子: {required_factors}")
        
        # 加载离线因子数据
        if required_factors:
            factor_data = load_offline_factors(
                required_factors, START_DATE, END_DATE, universe_codes, data_dir=FACTOR_DIR
            )
            
            if factor_data.empty:
                print("❌ 错误：未能加载任何因子数据，无法启动回测。")
                sys.exit(1)
            
            # 注入数据
            strategy.load_data(factor_data)
        else:
            print("⚠️ 警告：策略未声明任何因子依赖。")

        print(f"✅ 策略初始化完成。")

    except Exception as e:
        print(f"❌ 策略初始化失败: {e}")
        traceback.print_exc() # 打印完整堆栈方便调试
        sys.exit(1)

    # -----------------------------------------------------------
    # 4. 执行回测
    # -----------------------------------------------------------
    print("\n--- 阶段 3: 执行回测 ---")
    
    # 构造回测引擎配置
    ENGINE_CONFIG = {
        'INITIAL_CAPITAL': bt_conf.get('initial_capital', 1000000),
        'COMMISSION_RATE': bt_conf.get('commission_rate', 0.001),
        'SLIPPAGE': bt_conf.get('slippage', 0.0005),
        'BENCHMARK': bt_conf.get('benchmark', 'SPY'),
        'REBALANCE_DAYS': bt_conf.get('rebalance_days', 20),
        'REBALANCE_MONTHS': bt_conf.get('rebalance_months', 1)
    }

    engine = BacktestEngine(
        start_date=START_DATE,
        end_date=END_DATE,
        config=ENGINE_CONFIG,
        strategy=strategy,
        query_helper=helper,
        universe_to_run=TARGET_UNIVERSE # [修改] 传入筛选后的 universe
    )
    
    # 运行回测
    portfolio_history, final_portfolio = engine.run()

    # -----------------------------------------------------------
    # 5. 结果分析与可视化
    # -----------------------------------------------------------
    print("\n--- 阶段 4: 结果分析 ---")
    
    # A. 获取基准数据
    benchmark_equity = None
    bench_symbol = ENGINE_CONFIG['BENCHMARK']
    
    try:
        print(f"📈 正在获取基准数据 ({bench_symbol})...")
        bench_ret = helper.get_benchmark_returns(bench_symbol)
        
        if not bench_ret.empty:
            if not isinstance(bench_ret.index, pd.DatetimeIndex):
                bench_ret.index = pd.to_datetime(bench_ret.index)
            
            bench_ret = bench_ret.sort_index().loc[START_DATE:END_DATE]
            initial_cap = ENGINE_CONFIG['INITIAL_CAPITAL']
            benchmark_equity = (1 + bench_ret).cumprod() * initial_cap
            benchmark_equity = benchmark_equity.reindex(portfolio_history.index, method='ffill').fillna(initial_cap)
            
    except Exception as e:
        print(f"⚠️ 基准数据获取失败 ({e})，使用平线代替。")
        pass
    
    if benchmark_equity is None:
        benchmark_equity = pd.Series(ENGINE_CONFIG['INITIAL_CAPITAL'], index=portfolio_history.index)

    # B. 计算核心指标
    metrics = calculate_extended_metrics(
        portfolio_equity=portfolio_history['total_value'],
        benchmark_equity=benchmark_equity,
        portfolio_instance=final_portfolio
    )
    
    display_metrics(metrics, benchmark_loaded=True)

    # C. 绘图
    plt.figure(figsize=(12, 6))
    
    strat_norm = portfolio_history['total_value'] / portfolio_history['total_value'].iloc[0]
    bench_norm = benchmark_equity / benchmark_equity.iloc[0]
    
    plt.plot(strat_norm, label='Strategy', linewidth=2, color='#1f77b4')
    plt.plot(bench_norm, label=f"Benchmark ({bench_symbol})", linestyle='--', alpha=0.7, color='#ff7f0e')
    
    plt.title(f"Backtest: {strategy.name} (Universe: {TARGET_UNIVERSE})") # [修改] 标题显示 Universe
    plt.xlabel("Date")
    plt.ylabel("Normalized Equity")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # D. 保存结果
    if not os.path.exists('results'):
        os.makedirs('results')
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"results/backtest_{timestamp}.png"
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n📊 结果图表已保存至: {output_path}")