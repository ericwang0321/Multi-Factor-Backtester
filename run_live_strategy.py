import pandas as pd
import numpy as np
import math
import time
from datetime import datetime

# 引入你的模块
from quant_core.live.trader import LiveTrader
from quant_core.live.data_bridge import LiveDataBridge
from quant_core.strategies.rules import LinearWeightedStrategy

# --- 1. 配置区域 ---
UNIVERSE_PATH = 'data/reference/sec_code_category_grouped.csv'

# 策略配置 (请确保这里的因子名在 LiveDataBridge 里已经写了计算公式)
STRATEGY_CONFIG = {
    'name': 'Live_MultiFactor_v1',
    'weights': {
        'alpha013': 0.6, 
        'rsi': 0.4
    },
    'top_k': 3,
    # 风控参数
    'stop_loss_pct': 0.05,       # 个股跌 5% 止损
    'max_pos_weight': 0.3,       # 单票最多买 30%
    'max_drawdown_pct': 0.15     # 账户回撤 15% 熔断
}

def weight_to_quantity(target_weights: dict, current_prices: pd.Series, total_equity: float) -> dict:
    """
    [核心逻辑] 将 目标权重(%) 转换为 目标股数(Share)
    """
    target_qtys = {}
    
    print(f"\n💰 资金分配 (总权益: ${total_equity:,.2f}):")
    
    for code, weight in target_weights.items():
        if weight == 0:
            target_qtys[code] = 0
            continue
            
        price = current_prices.get(code)
        if not price or pd.isna(price) or price <= 0:
            print(f"⚠️ 跳过 {code}: 无法获取有效价格 ({price})")
            continue
            
        # 1. 计算目标金额
        target_value = total_equity * weight
        
        # 2. 计算股数 (向下取整，保守处理)
        # 例如: 打算买 $1000，股价 $300 -> 买 3 股 ($900)，而不是 4 股 ($1200)
        qty = math.floor(target_value / price)
        
        target_qtys[code] = int(qty)
        print(f"  - {code}: 权重 {weight:.1%} | 价格 ${price:.2f} -> 目标金额 ${target_value:.0f} -> 股数 {qty}")
        
    return target_qtys

def build_portfolio_state(connector):
    """
    构建策略所需的 portfolio_state 字典
    包含: total_equity, positions, avg_costs
    """
    # 获取账户摘要
    summary = connector.ib.accountSummary()
    # 提取总权益 (NetLiquidation)
    total_equity = float(next((x.value for x in summary if x.tag == 'NetLiquidation'), 0))
    
    # 获取持仓详情 (包含均价)
    ib_positions = connector.ib.positions()
    
    positions = {}
    avg_costs = {}
    
    for p in ib_positions:
        # p.contract.localSymbol 通常是美股代码 'SPY'
        # 注意：如果你的策略用的是 'SPY.P'，这里可能需要反向映射。
        # 为了简单，这里假设策略产生的信号已经 strip 掉了后缀，或者 bridge 处理了一致性。
        symbol = p.contract.localSymbol 
        positions[symbol] = p.position
        avg_costs[symbol] = p.avgCost
        
    return {
        'total_equity': total_equity,
        'positions': positions,
        'avg_costs': avg_costs
    }

def main():
    print(f"🚀 [{datetime.now()}] 启动实盘策略执行脚本...")
    
    # 1. 初始化模块
    trader = LiveTrader()
    trader.start() # 连接 IB
    
    # 等待连接
    time.sleep(2)
    if not trader.connector.ib.isConnected():
        print("❌ 无法连接到 IB，脚本终止。")
        return

    bridge = LiveDataBridge(trader.connector, UNIVERSE_PATH)
    
    strategy = LinearWeightedStrategy(
        name=STRATEGY_CONFIG['name'],
        weights=STRATEGY_CONFIG['weights'],
        top_k=STRATEGY_CONFIG['top_k'],
        stop_loss_pct=STRATEGY_CONFIG['stop_loss_pct'],
        max_pos_weight=STRATEGY_CONFIG['max_pos_weight'],
        max_drawdown_pct=STRATEGY_CONFIG['max_drawdown_pct']
    )

    try:
        # --- Step 1: 准备数据 ---
        required_factors = list(STRATEGY_CONFIG['weights'].keys())
        today_str = datetime.now().strftime('%Y-%m-%d') # 获取今日日期字符串
        
        # 获取数据 (Index=Code, Columns=Factors)
        factor_df, current_prices = bridge.prepare_data_for_strategy(
            required_factors, 
            lookback_window=365, # 保持 365 以确保足够的预热
            bar_size='1 day'
        )
        
        if factor_df.empty:
            print("⚠️ 未获取到有效因子数据，跳过本次执行。")
            return

        # [🔍 调试打印] 看看因子到底算出来没？
        print(f"\n🔍 因子快照 (前3行): \n{factor_df.head(3)}")
        print(f"   包含 NaN? {factor_df.isnull().values.any()}")

        # ==============================================================================
        # [关键修复] 升维处理：构建 MultiIndex (Date, Code) 以适配 BaseStrategy
        # ==============================================================================
        # 1. 此时 factor_df 的 Index 是股票代码 (如 'SPY', 'AAPL')
        factor_df.index.name = 'sec_code' 
        factor_df = factor_df.reset_index() # 将 sec_code 变成一列
        
        # 2. 加上日期列
        factor_df['date'] = today_str 
        
        # 3. 重新设置为双重索引 (Date, sec_code)
        factor_df = factor_df.set_index(['date', 'sec_code'])
        # ==============================================================================

        # 注入数据到策略 (此时结构已符合策略预期)
        strategy.load_data(factor_df, price_df=None)

        # --- Step 2: 获取当前账户状态 ---
        portfolio_state = build_portfolio_state(trader.connector)
        total_equity = portfolio_state['total_equity']
        print(f"\n📊 当前账户净值: ${total_equity:,.2f}")

        # --- Step 3: 运行策略逻辑 (On Bar) ---
        # 注意：这里传入的 universe_codes 必须是纯代码列表
        # factor_df 现在是 MultiIndex，我们需要提取 Level 1 (sec_code)
        universe_codes = factor_df.index.get_level_values('sec_code').unique().tolist()
        
        # 调用策略
        target_weights = strategy.on_bar(
            date=today_str, # 必须和上面 factor_df['date'] 一致
            universe_codes=universe_codes,
            portfolio_state=portfolio_state,
            current_prices=current_prices
        )
        
        # [🔍 调试打印] 看看策略算出的权重
        print(f"🎯 策略输出目标权重: {target_weights}")

        if not target_weights and not portfolio_state['positions']:
            print("😴 策略无信号且空仓，无操作。")
        else:
            # --- Step 4: 执行交易 ---
            # [修复点 1] 清洗价格字典的 Key (从 'IAGG.B' -> 'IAGG')
            clean_prices = {}
            for k, v in current_prices.items():
                short_sym = k.split('.')[0]
                clean_prices[short_sym] = v
            
            # [修复点 2] 清洗目标权重的 Key (从 'IAGG.B' -> 'IAGG')
            clean_target_weights = {}
            for code, w in target_weights.items():
                symbol = code.split('.')[0] # 去掉后缀
                clean_target_weights[symbol] = w
            
            # 现在两个字典的 Key 都是 'IAGG', 'DBA' 了，可以匹配上了
            target_quantities = weight_to_quantity(clean_target_weights, clean_prices, total_equity)
            
            # 发送给 Trader 执行
            trader.execute_rebalance(target_quantities)

    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n👋 执行结束，断开连接。")
        trader.stop()

if __name__ == "__main__":
    main()