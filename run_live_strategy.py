import pandas as pd
import numpy as np
import math
import time
from datetime import datetime
import traceback

# --- 引入项目模块 ---
from quant_core.live.trader import LiveTrader
from quant_core.live.data_bridge import LiveDataBridge
from quant_core.strategies.rules import LinearWeightedStrategy

# --- [新增] 引入工具模块 ---
# (请确保 quant_core/utils/__init__.py 存在)
from quant_core.utils.logger import setup_logger
from quant_core.utils.notifier import Notifier

# ==============================================================================
# 1. 配置区域 (Configuration)
# ==============================================================================
UNIVERSE_PATH = 'data/reference/sec_code_category_grouped.csv'
CONFIG_PATH = 'config.yaml' # 包含邮件配置的yaml路径

# 策略配置
STRATEGY_CONFIG = {
    'name': 'Live_MultiFactor_v1',
    'weights': {
        'alpha013': 0.6, 
        'rsi': 0.4
    },
    'top_k': 3,
    'stop_loss_pct': 0.05,
    'max_pos_weight': 0.3,
    'max_drawdown_pct': 0.15
}

# 初始化全局工具 (Logger & Notifier)
# Logger 会自动写入 logs/live_trading_YYYY-MM-DD.log
logger = setup_logger(name='live_strategy')
notifier = Notifier(config_path=CONFIG_PATH)

# ==============================================================================
# 2. 辅助函数 (Helpers)
# ==============================================================================

def weight_to_quantity(target_weights: dict, current_prices: pd.Series, total_equity: float) -> dict:
    """
    [核心逻辑] 将 目标权重(%) 转换为 目标股数(Share)
    """
    target_qtys = {}
    logger.info(f"💰 资金分配计算 (总权益: ${total_equity:,.2f})...")
    
    log_details = [] # 用于邮件内容
    
    for code, weight in target_weights.items():
        if weight == 0:
            target_qtys[code] = 0
            continue
            
        price = current_prices.get(code)
        if not price or pd.isna(price) or price <= 0:
            logger.warning(f"⚠️ 跳过 {code}: 无法获取有效价格 ({price})")
            continue
            
        target_value = total_equity * weight
        qty = math.floor(target_value / price)
        target_qtys[code] = int(qty)
        
        info_str = f"  - {code}: 权重 {weight:.1%} | 价格 ${price:.2f} -> 目标 ${target_value:.0f} -> 股数 {qty}"
        logger.info(info_str)
        log_details.append(info_str)
        
    return target_qtys, "\n".join(log_details)

def build_portfolio_state(connector):
    """构建策略所需的 portfolio_state 字典"""
    summary = connector.ib.accountSummary()
    total_equity = float(next((x.value for x in summary if x.tag == 'NetLiquidation'), 0))
    
    ib_positions = connector.ib.positions()
    positions = {}
    avg_costs = {}
    
    for p in ib_positions:
        symbol = p.contract.localSymbol 
        positions[symbol] = p.position
        avg_costs[symbol] = p.avgCost
        
    return {
        'total_equity': total_equity,
        'positions': positions,
        'avg_costs': avg_costs
    }

# ==============================================================================
# 3. 主程序 (Main Execution Flow)
# ==============================================================================

def main():
    start_time = datetime.now()
    logger.info(f"🚀 启动实盘策略执行脚本...")
    
    trader = None
    try:
        # --- Step 0: 初始化与连接 ---
        trader = LiveTrader()
        trader.start()
        
        time.sleep(2)
        if not trader.connector.ib.isConnected():
            raise ConnectionError("无法连接到 IB TWS/Gateway，请检查软件是否开启 (Port 7497/7496)")

        bridge = LiveDataBridge(trader.connector, UNIVERSE_PATH)
        
        # --- Step 1: 准备数据 ---
        logger.info("⚡ [Data] 正在获取历史数据并计算因子 (Lookback: 365)...")
        
        required_factors = list(STRATEGY_CONFIG['weights'].keys())
        today_str = datetime.now().strftime('%Y-%m-%d')
        
        factor_df, current_prices = bridge.prepare_data_for_strategy(
            required_factors, 
            lookback_window=365,
            bar_size='1 day'
        )
        
        if factor_df.empty:
            logger.warning("⚠️ 未获取到有效因子数据，跳过本次执行。")
            return

        # 调试信息记录
        logger.info(f"🔍 因子快照 (前3行): \n{factor_df.head(3)}")
        
        # [关键修复] 构建 MultiIndex (Date, Code)
        factor_df.index.name = 'sec_code'
        factor_df = factor_df.reset_index()
        factor_df['date'] = today_str
        factor_df = factor_df.set_index(['date', 'sec_code'])

        # --- Step 2: 策略计算 ---
        strategy = LinearWeightedStrategy(
            name=STRATEGY_CONFIG['name'],
            weights=STRATEGY_CONFIG['weights'],
            top_k=STRATEGY_CONFIG['top_k'],
            stop_loss_pct=STRATEGY_CONFIG['stop_loss_pct'],
            max_pos_weight=STRATEGY_CONFIG['max_pos_weight'],
            max_drawdown_pct=STRATEGY_CONFIG['max_drawdown_pct']
        )
        strategy.load_data(factor_df, price_df=None)

        portfolio_state = build_portfolio_state(trader.connector)
        total_equity = portfolio_state['total_equity']
        logger.info(f"📊 当前账户净值: ${total_equity:,.2f}")

        universe_codes = factor_df.index.get_level_values('sec_code').unique().tolist()
        
        # 运行 On Bar
        target_weights = strategy.on_bar(
            date=today_str,
            universe_codes=universe_codes,
            portfolio_state=portfolio_state,
            current_prices=current_prices
        )
        logger.info(f"🎯 策略输出目标权重: {target_weights}")

        # --- Step 3: 交易执行与汇报 ---
        if not target_weights and not portfolio_state['positions']:
            logger.info("😴 策略无信号且空仓，无操作。")
            notifier.send(f"实盘报告 {today_str}", f"执行完毕。当前净值: ${total_equity:,.2f}\n无交易信号。")
        else:
            # 清洗 Key (去后缀)
            clean_prices = {k.split('.')[0]: v for k, v in current_prices.items()}
            clean_target_weights = {k.split('.')[0]: v for k, v in target_weights.items()}
            
            # 计算股数
            target_quantities, calc_details = weight_to_quantity(clean_target_weights, clean_prices, total_equity)
            
            # 发送订单
            logger.info("🔄 开始执行调仓...")
            trader.execute_rebalance(target_quantities)
            
            # [新增] 简易的订单确认 (等待 3 秒给 IB 处理)
            time.sleep(3)
            # [修复] 使用 openTrades()，因为它同时包含 Order 和 Contract 信息
            open_trades = trader.connector.ib.openTrades() 
            
            open_order_str = "\n".join([
                f"- {t.order.action} {t.order.totalQuantity} {t.contract.localSymbol} ({t.order.orderType}) | 状态: {t.orderStatus.status}" 
                for t in open_trades
            ])
            if not open_order_str:
                status_msg = "所有订单已成交 (或无挂单)。"
            else:
                status_msg = f"当前挂单 (Waiting):\n{open_order_str}"
            
            # 发送邮件通知
            email_body = (
                f"【实盘执行成功】\n"
                f"时间: {start_time}\n"
                f"账户净值: ${total_equity:,.2f}\n\n"
                f"--- 目标持仓计算 ---\n{calc_details}\n\n"
                f"--- 订单状态 ---\n{status_msg}"
            )
            notifier.send(f"实盘交易报告 {today_str}", email_body)
            logger.info("✅ 交易执行完毕，通知已发送。")

    except Exception as e:
        error_msg = f"❌ 实盘运行出错: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        # 发送报错通知
        notifier.send(f"【紧急】实盘报错 {datetime.now().strftime('%H:%M')}", f"{error_msg}\n\n{traceback.format_exc()}")
        
    finally:
        logger.info("👋 脚本退出，断开连接。")
        if trader:
            trader.stop()

if __name__ == "__main__":
    main()