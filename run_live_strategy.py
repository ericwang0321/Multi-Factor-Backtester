import pandas as pd
import numpy as np
import math
import time
from datetime import datetime
import traceback

# --- 1. 引入配置与策略工厂 ---
from config import load_config
from quant_core.strategies import get_strategy_instance

# --- 2. 引入业务模块 ---
from quant_core.live.trader import LiveTrader
from quant_core.live.data_bridge import LiveDataBridge
from quant_core.utils.logger import setup_logger
from quant_core.utils.notifier import Notifier

# ==============================================================================
# 全局配置初始化 (自动合并 base + live + secrets)
# ==============================================================================
CONF = load_config(mode='live')

# 初始化全局工具 (Logger 使用默认路径，Notifier 指向包含隐私密码的 secrets)
logger = setup_logger(name='live_strategy')
notifier = Notifier(config_path='config/secrets.yaml')

# ==============================================================================
# 辅助函数 (Helpers)
# ==============================================================================

def weight_to_quantity(target_weights: dict, current_prices: pd.Series, total_equity: float) -> tuple:
    """
    [核心逻辑] 将 目标权重(%) 转换为 目标股数(Share)
    """
    target_qtys = {}
    logger.info(f"💰 资金分配计算 (总权益: ${total_equity:,.2f})...")
    
    log_details = [] 
    
    for code, weight in target_weights.items():
        if weight == 0:
            target_qtys[code] = 0
            continue
            
        price = current_prices.get(code)
        if not price or pd.isna(price) or price <= 0:
            logger.warning(f"⚠️ 跳过 {code}: 无法获取有效价格 ({price})")
            continue
            
        # 计算目标股数 (向下取整)
        target_value = total_equity * weight
        qty = math.floor(target_value / price)
        target_qtys[code] = int(qty)
        
        info_str = f"  - {code}: 权重 {weight:.1%} | 价格 ${price:.2f} -> 目标 ${target_value:,.0f} -> 股数 {qty}"
        logger.info(info_str)
        log_details.append(info_str)
        
    return target_qtys, "\n".join(log_details)

def build_portfolio_state(connector):
    """
    对接 IB 获取当前账户实时净值与持仓
    """
    # 获取账户摘要 (NetLiquidation 代表总资产)
    summary = connector.ib.accountSummary()
    total_equity = float(next((x.value for x in summary if x.tag == 'NetLiquidation'), 0))
    
    # 获取持仓详情
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
# 主程序逻辑 (Main Execution Flow)
# ==============================================================================

def main():
    start_time = datetime.now()
    logger.info(f"🚀 启动实盘交易系统 [策略类型: {CONF['strategy']['type']}]")
    
    trader = None
    try:
        # --- Step 1: 策略实例化 (通过工厂模式) ---
        # 自动根据 CONF['strategy']['type'] 决定生成 Linear 还是 ML 策略
        strategy = get_strategy_instance(CONF['strategy'])
        
        # --- Step 2: 建立 IB 连接 ---
        trader = LiveTrader()
        # 使用 live.yaml 中的端口配置 (Paper: 7497, Live: 7496)
        trader.connector.port = CONF['ib_connection'].get('port', 7497)
        trader.start()
        
        # 等待连接稳定
        time.sleep(2)
        if not trader.connector.ib.isConnected():
            raise ConnectionError(f"无法连接到 IB (Port: {trader.connector.port})，请确保 TWS 已开启。")

        # 初始化数据桥接层
        bridge = LiveDataBridge(trader.connector, CONF['universe_path'])
        
        # --- Step 3: 数据准备 (依赖倒置) ---
        # 动态询问策略对象需要哪些因子，不再硬编码
        required_factors = strategy.get_required_factors()
        logger.info(f"📡 策略请求因子列表: {required_factors}")
        
        # 获取回看窗口数据 (默认 365 天)
        factor_df, current_prices = bridge.prepare_data_for_strategy(
            required_factors, 
            lookback_window=365,
            bar_size='1 day'
        )
        
        if factor_df.empty:
            logger.warning("⚠️ 数据获取为空，脚本终止。")
            return

        # 格式化数据以适配策略基类 (Date, Code MultiIndex)
        today_str = datetime.now().strftime('%Y-%m-%d')
        factor_df.index.name = 'sec_code'
        factor_df = factor_df.reset_index()
        factor_df['date'] = today_str
        factor_df = factor_df.set_index(['date', 'sec_code'])

        # --- Step 4: 运行策略逻辑 ---
        # 注入因子数据
        strategy.load_data(factor_df)

        # 获取当前实盘账户净值与仓位
        portfolio_state = build_portfolio_state(trader.connector)
        total_equity = portfolio_state['total_equity']
        logger.info(f"📊 当前账户权益: ${total_equity:,.2f}")

        # 计算目标权重
        universe_codes = factor_df.index.get_level_values('sec_code').unique().tolist()
        target_weights = strategy.on_bar(
            date=today_str,
            universe_codes=universe_codes,
            portfolio_state=portfolio_state,
            current_prices=pd.Series(current_prices)
        )
        logger.info(f"🎯 策略输出目标权重: {target_weights}")

        # --- Step 5: 交易执行与自动报告 ---
        if not target_weights and not portfolio_state['positions']:
            logger.info("😴 策略无信号且空仓，无操作。")
            notifier.send(f"实盘报告 {today_str}", f"执行完毕。账户净值: ${total_equity:,.2f}\n今日无交易信号。")
        else:
            # 清洗代码后缀 (如 'IAGG.B' -> 'IAGG') 确保匹配
            clean_prices = {k.split('.')[0]: v for k, v in current_prices.items()}
            clean_target_weights = {k.split('.')[0]: v for k, v in target_weights.items()}
            
            # 权重转股数
            target_quantities, calc_details = weight_to_quantity(clean_target_weights, clean_prices, total_equity)
            
            # 调用 Trader 执行调仓 (执行逻辑包含在 trader.py 中)
            logger.info("🔄 正在发送交易订单至 IB...")
            trader.execute_rebalance(target_quantities)
            
            # 等待 3 秒确保 IB 接收并返回订单状态
            time.sleep(3)
            
            # 查询挂单状态
            open_trades = trader.connector.ib.openTrades() 
            open_order_str = "\n".join([
                f"- {t.order.action} {t.order.totalQuantity} {t.contract.localSymbol} ({t.order.orderType}) | 状态: {t.orderStatus.status}" 
                for t in open_trades
            ])
            
            status_summary = open_order_str if open_order_str else "所有订单已成交或已进入队列。"
            
            # 发送全链路执行邮件报告
            email_body = (
                f"【实盘执行成功报告】\n"
                f"执行时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"账户总权益: ${total_equity:,.2f}\n"
                f"策略模式: {CONF['strategy']['type']}\n\n"
                f"--- 目标持仓计算细节 ---\n{calc_details}\n\n"
                f"--- 订单实时状态 ---\n{status_summary}"
            )
            notifier.send(f"实盘交易报告 {today_str}", email_body)
            logger.info("✅ 任务完成，汇报邮件已发送。")

    except Exception as e:
        error_info = traceback.format_exc()
        logger.error(f"❌ 系统运行异常: {e}\n{error_info}")
        notifier.send(f"🚨 实盘系统告警", f"异常时间: {datetime.now()}\n错误详情: {str(e)}\n\n堆栈信息:\n{error_info}")
        
    finally:
        if trader:
            logger.info("👋 正在断开连接并退出脚本。")
            trader.stop()

if __name__ == "__main__":
    main()