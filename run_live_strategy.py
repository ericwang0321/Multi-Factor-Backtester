# run_live_strategy.py
import pandas as pd
import numpy as np
import math
import time
import json
import os
import sys
from datetime import datetime
import traceback

# --- 1. 引入配置与策略工厂 ---
from config import load_config
# [修正] 这里原来写成了 get_strategy_instance，应该是 create_strategy_instance
from quant_core.strategies import create_strategy_instance

# --- 2. 引入业务模块 ---
from quant_core.live.trader import LiveTrader
from quant_core.live.data_bridge import LiveDataBridge
from quant_core.utils.logger import setup_logger
from quant_core.utils.notifier import Notifier

# ==============================================================================
# 全局配置与常量
# ==============================================================================
CONF = load_config(mode='live')
logger = setup_logger(name='live_strategy')
notifier = Notifier(config_path='config/secrets.yaml')

# [新增] 状态文件路径 (用于与 app.py 通信)
DATA_DIR = 'data/live'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
STATE_FILE = os.path.join(DATA_DIR, 'dashboard_state.json')
COMMAND_FILE = os.path.join(DATA_DIR, 'command.json')

# ==============================================================================
# 辅助函数 (Helpers)
# ==============================================================================

def save_dashboard_state(state_data):
    """
    [新增] 将当前运行状态写入 JSON 文件，供前端监控
    """
    try:
        # 补充时间戳
        state_data['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 写入临时文件再重命名，防止读写冲突 (Atomic Write)
        temp_file = STATE_FILE + '.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)
        os.replace(temp_file, STATE_FILE)
    except Exception as e:
        logger.error(f"无法写入状态文件: {e}")

def check_remote_commands(trader):
    """
    [新增] 检查是否有来自前端的控制指令
    """
    if not os.path.exists(COMMAND_FILE):
        return

    try:
        with open(COMMAND_FILE, 'r') as f:
            cmd = json.load(f)
        
        # 执行完立即删除指令文件，防止重复执行
        os.remove(COMMAND_FILE)
        
        action = cmd.get('action')
        logger.warning(f"⚠️ 收到远程指令: {action}")

        if action == 'STOP':
            logger.warning("🛑 执行紧急停止！")
            sys.exit(0) # 退出脚本
            
        elif action == 'FLAT_ALL':
            logger.warning("📉 执行一键清仓！")
            # 这里调用 trader 的清仓逻辑 (需在 trader.py 实现 close_all_positions)
            # 暂时示例：
            # trader.close_all_positions()
            notifier.send("实盘告警", "已执行远程一键清仓指令！")

        # [新增] 处理撤单指令
        elif action == 'CANCEL_ALL':
            logger.warning("🚫 执行全部撤单！")
            trader.cancel_all_orders() # 调用刚才加的方法
            notifier.send("实盘操作", "已执行全部撤单指令。")
            
    except Exception as e:
        logger.error(f"处理指令失败: {e}")

def weight_to_quantity(target_weights: dict, current_prices: pd.Series, total_equity: float) -> tuple:
    """
    [核心逻辑] 将 目标权重(%) 转换为 目标股数(Share)
    """
    target_qtys = {}
    log_details = [] 
    
    for code, weight in target_weights.items():
        if weight == 0:
            target_qtys[code] = 0
            continue
            
        price = current_prices.get(code)
        if not price or pd.isna(price) or price <= 0:
            continue
            
        target_value = total_equity * weight
        qty = math.floor(target_value / price)
        target_qtys[code] = int(qty)
        
        info_str = f"{code}: {weight:.1%} | ${price:.2f} -> {qty} shares"
        log_details.append(info_str)
        
    return target_qtys, "\n".join(log_details)

def build_portfolio_state(connector):
    """
    对接 IB 获取当前账户实时净值与持仓
    """
    if not connector.ib.isConnected():
        return {'total_equity': 0, 'positions': {}, 'avg_costs': {}, 'pnl': 0}

    # 获取账户摘要
    summary = connector.ib.accountSummary()
    # NetLiquidation: 总资产, UnrealizedPnL: 未实现盈亏
    total_equity = float(next((x.value for x in summary if x.tag == 'NetLiquidation'), 0))
    unrealized_pnl = float(next((x.value for x in summary if x.tag == 'UnrealizedPnL'), 0))
    
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
        'unrealized_pnl': unrealized_pnl,
        'positions': positions,
        'avg_costs': avg_costs
    }

# ==============================================================================
# 主程序逻辑
# ==============================================================================

def main():
    start_time = datetime.now()
    logger.info(f"🚀 启动实盘引擎 (Dashboard Mode) [策略: {CONF['strategy']['type']}]")
    
    # 初始化状态
    dashboard_data = {
        "status": "Starting",
        "strategy": CONF['strategy']['type'],
        "logs": [],
        "account": {}
    }
    save_dashboard_state(dashboard_data)

    trader = None
    try:
        # --- Step 1: 建立连接 ---
        trader = LiveTrader()
        trader.connector.port = CONF['ib_connection'].get('port', 7497)
        trader.start()
        
        time.sleep(2)
        if not trader.connector.ib.isConnected():
            raise ConnectionError("无法连接到 IB，请检查 TWS。")

        dashboard_data["status"] = "Connected"
        save_dashboard_state(dashboard_data)

        # --- Step 2: 策略执行 (Trading Phase) ---
        logger.info("🧠 开始执行策略逻辑...")
        
        # [修正] 实例化策略：使用 correct_strategy_instance
        strategy = create_strategy_instance(CONF['strategy'])
        bridge = LiveDataBridge(trader.connector, CONF['universe_path'])
        
        # 准备数据
        required_factors = strategy.get_required_factors()
        factor_df, current_prices = bridge.prepare_data_for_strategy(
            required_factors, lookback_window=365
        )

        if not factor_df.empty:
            # 格式化数据
            today_str = datetime.now().strftime('%Y-%m-%d')
            factor_df.index.name = 'sec_code'
            factor_df = factor_df.reset_index()
            factor_df['date'] = today_str
            factor_df = factor_df.set_index(['date', 'sec_code'])
            
            strategy.load_data(factor_df)
            
            # 获取状态
            portfolio_state = build_portfolio_state(trader.connector)
            dashboard_data["account"] = portfolio_state
            save_dashboard_state(dashboard_data)

            # 计算信号
            universe_codes = factor_df.index.get_level_values('sec_code').unique().tolist()
            target_weights = strategy.on_bar(
                date=today_str,
                universe_codes=universe_codes,
                portfolio_state=portfolio_state,
                current_prices=pd.Series(current_prices)
            )

            # 执行交易
            if target_weights or portfolio_state['positions']:
                clean_prices = {k.split('.')[0]: v for k, v in current_prices.items()}
                clean_target_weights = {k.split('.')[0]: v for k, v in target_weights.items()}
                
                target_qtys, details = weight_to_quantity(clean_target_weights, clean_prices, portfolio_state['total_equity'])
                
                logger.info(f"发送调仓指令...")
                trader.execute_rebalance(target_qtys)
                
                # 发送报告
                notifier.send(f"实盘执行报告 {today_str}", f"调仓已完成。\n{details}")
            else:
                logger.info("无信号或空仓，跳过交易。")

        # --- Step 3: 进入监控保活模式 (Monitoring Loop) ---
        # 这是一个死循环，保持脚本运行，以便 app.py 可以实时看到 PnL 变化
        logger.info("👁️ 交易逻辑结束，进入实时监控模式 (按 Ctrl+C 退出)...")
        dashboard_data["status"] = "Monitoring"
        
        # 记录最近的日志用于前端显示 (简单实现，实际可用 deque)
        recent_logs = ["System Initialized", "Trading Logic Completed", "Entering Monitor Mode"]

        while True:
            # 1. 检查前端指令 (Stop/Flat)
            check_remote_commands(trader)
            
            # 2. 更新账户状态 (心跳)
            if trader.connector.ib.isConnected():
                current_state = build_portfolio_state(trader.connector)
                dashboard_data["account"] = current_state
                dashboard_data["last_update"] = datetime.now().strftime('%H:%M:%S')
                
                # 更新日志 (模拟)
                dashboard_data["logs"] = recent_logs[-10:] 
                
                save_dashboard_state(dashboard_data)
            else:
                logger.warning("IB 连接断开，尝试重连...")
                dashboard_data["status"] = "Disconnected"
                save_dashboard_state(dashboard_data)
                try:
                    trader.start()
                except:
                    pass

            # 3. 频率控制 (每 3 秒刷新一次)
            time.sleep(3)

    except KeyboardInterrupt:
        logger.info("用户手动停止脚本。")
    except Exception as e:
        logger.error(f"❌ 异常退出: {e}")
        dashboard_data["status"] = "Error"
        dashboard_data["error"] = str(e)
        save_dashboard_state(dashboard_data)
        notifier.send("实盘崩溃", traceback.format_exc())
    finally:
        if trader:
            trader.stop()
        logger.info("脚本已结束。")

if __name__ == "__main__":
    main()