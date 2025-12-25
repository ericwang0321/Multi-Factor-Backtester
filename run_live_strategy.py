# run_live_strategy.py
import asyncio
import os
import sys
import json
import math
import traceback
import pandas as pd
from datetime import datetime
from pytz import timezone

# 引入调度器
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# 引入原有模块
from config import load_config
from quant_core.strategies import create_strategy_instance
from quant_core.live.trader import LiveTrader
from quant_core.live.data_bridge import LiveDataBridge
from quant_core.utils.logger import setup_logger
from quant_core.utils.notifier import Notifier

# ==============================================================================
# 全局配置与状态
# ==============================================================================
CONF = load_config(mode='live')
logger = setup_logger(name='live_daemon')
notifier = Notifier(config_path='config/secrets.yaml')

# 数据路径
DATA_DIR = 'data/live'
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
STATE_FILE = os.path.join(DATA_DIR, 'dashboard_state.json')
COMMAND_FILE = os.path.join(DATA_DIR, 'command.json')

# 全局变量
trader = None
scheduler = None

# ==============================================================================
# 1. 核心任务逻辑 (Tasks)
# ==============================================================================

async def job_trading_session():
    """
    【交易任务】每天美东时间 09:30 触发
    负责：连接检查 -> 数据拉取 -> 策略计算 -> 下单 -> 推送通知
    """
    logger.info("⏰ [Scheduler] 触发每日定时交易任务...")
    notifier.send("实盘启动", "正在执行每日定投策略逻辑...")
    
    try:
        # 1. 确保连接健康
        if not trader or not trader.connector.ib.isConnected():
            logger.warning("⚠️ IB 未连接，尝试重连...")
            # 这里的重连机制依赖于 IB 客户端自身的自动重连，或者可以在此添加显式重连逻辑
            return

        # 2. 策略实例化
        strategy = create_strategy_instance(CONF['strategy'])
        bridge = LiveDataBridge(trader.connector, CONF['universe_path'])
        
        # 3. 数据准备 (Data Pulling)
        logger.info("📡 正在拉取 IB 历史数据...")
        required_factors = strategy.get_required_factors()
        
        factor_df, current_prices = bridge.prepare_data_for_strategy(
            required_factors, lookback_window=365
        )

        if factor_df.empty:
            logger.error("❌ 数据获取为空，跳过本次交易")
            notifier.send("交易失败", "获取行情数据为空，策略未执行。")
            return

        # 4. 格式化数据并加载
        today_str = datetime.now().strftime('%Y-%m-%d')
        factor_df.index.name = 'sec_code'
        factor_df = factor_df.reset_index()
        factor_df['date'] = today_str
        factor_df = factor_df.set_index(['date', 'sec_code'])
        
        strategy.load_data(factor_df)
        
        # 5. 获取账户状态
        state = build_portfolio_state(trader.connector)
        
        # 6. 运行策略计算 (Core Logic)
        logger.info("🧠 正在计算策略信号...")
        universe_codes = factor_df.index.get_level_values('sec_code').unique().tolist()
        target_weights = strategy.on_bar(
            date=today_str,
            universe_codes=universe_codes,
            portfolio_state=state,
            current_prices=pd.Series(current_prices)
        )

        # 7. 执行交易 (Execution)
        if target_weights or state['positions']:
            clean_prices = {k.split('.')[0]: v for k, v in current_prices.items()}
            clean_weights = {k.split('.')[0]: v for k, v in target_weights.items()}
            
            target_qtys, details = weight_to_quantity(clean_weights, clean_prices, state['total_equity'])
            
            if target_qtys:
                logger.info(f"🔄 执行调仓: {target_qtys}")
                trader.execute_rebalance(target_qtys)
                notifier.send("交易完成", f"已发送订单至 TWS。\n{details}")
            else:
                logger.info("⚖️ 计算后持仓无变动。")
                notifier.send("交易跳过", "策略计算结果无持仓变动。")
        else:
            logger.info("💤 空仓且无信号。")

    except Exception as e:
        err_msg = traceback.format_exc()
        logger.error(f"❌ 交易任务异常: {e}\n{err_msg}")
        notifier.send("交易任务崩溃", f"请检查服务器日志。\n错误: {str(e)}")

async def job_heartbeat():
    """
    【心跳任务】每 5 秒运行一次
    负责：处理前端指令 -> 更新状态文件 -> 维持连接
    """
    # 1. 检查指令
    check_remote_commands(trader)
    
    # 2. 更新状态 (证明我还活着)
    if trader and trader.connector.ib.isConnected():
        state = build_portfolio_state(trader.connector)
        
        # 写入正在运行的状态
        state['status'] = "Running (Auto)"
        try:
            next_run = scheduler.get_job('daily_trading').next_run_time
            state['next_run'] = str(next_run)
        except:
            state['next_run'] = "Not Scheduled"
            
        save_dashboard_state(state)
    else:
        # 断连状态
        save_dashboard_state({'status': 'Disconnected', 'error': 'IB connection lost'})

# ==============================================================================
# 2. 辅助函数 (Helpers)
# ==============================================================================

def save_dashboard_state(state_data):
    """
    原子写入状态文件
    """
    try:
        # 统一添加最后更新时间 (这是 app.py 判断是否离线的依据)
        state_data['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        temp_file = STATE_FILE + '.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)
        os.replace(temp_file, STATE_FILE)
    except Exception: pass

def check_remote_commands(trader_instance):
    if not os.path.exists(COMMAND_FILE): return
    try:
        with open(COMMAND_FILE, 'r') as f: cmd = json.load(f)
        os.remove(COMMAND_FILE)
        
        action = cmd.get('action')
        logger.warning(f"⚠️ 收到远程指令: {action}")
        notifier.send("收到指令", f"正在执行: {action}")
        
        if action == 'STOP':
            # 这里抛出 SystemExit，会被 main_loop 的异常捕获处理，从而执行“遗言”逻辑
            logger.warning("🛑 停止指令已接收...")
            sys.exit(0)
        elif action == 'CANCEL_ALL':
            trader_instance.cancel_all_orders()
        elif action == 'FLAT_ALL':
            # [修改后] 真正的实装代码：
            logger.warning("📉 收到清仓指令，正在执行...")
            trader_instance.close_all_positions()
            notifier.send("⚠️ 紧急清仓", "已执行一键清仓 (FLAT ALL)，所有挂单已撤销，持仓正在市价卖出。")  
                      
    except SystemExit:
        raise # 重新抛出退出信号
    except Exception as e: 
        logger.error(f"指令处理失败: {e}")

def weight_to_quantity(weights, prices, equity):
    qtys = {}
    logs = []
    for code, w in weights.items():
        if w == 0: 
            qtys[code] = 0
            continue
        p = prices.get(code)
        if not p or p <= 0: continue
        qtys[code] = int(math.floor(equity * w / p))
        logs.append(f"{code}: {w:.1%} -> {qtys[code]} shares")
    return qtys, "\n".join(logs)

def build_portfolio_state(connector):
    if not connector.ib.isConnected(): return {'total_equity':0, 'positions':{}}
    summary = connector.ib.accountSummary()
    total_equity = float(next((x.value for x in summary if x.tag == 'NetLiquidation'), 0))
    pnl = float(next((x.value for x in summary if x.tag == 'UnrealizedPnL'), 0))
    positions = {p.contract.localSymbol: p.position for p in connector.ib.positions()}
    costs = {p.contract.localSymbol: p.avgCost for p in connector.ib.positions()}
    return {'total_equity': total_equity, 'unrealized_pnl': pnl, 'positions': positions, 'avg_costs': costs}

# ==============================================================================
# 3. 异步启动入口 (Main Entry) - 包含“遗言”逻辑
# ==============================================================================

async def main_loop():
    global trader, scheduler
    
    # --- 1. 初始化 ---
    trader = LiveTrader()
    port = CONF['ib_connection'].get('port', 7497) 
    trader.connector.port = port
    
    logger.info(f"🚀 正在连接 IB Gateway (Port: {port})...")
    trader.start() 
    
    for _ in range(5):
        if trader.connector.ib.isConnected(): break
        await asyncio.sleep(1)
    
    if not trader.connector.ib.isConnected():
        logger.error("❌ 无法连接 IB，请检查 TWS 是否开启。")
        return

    logger.info("✅ IB 连接成功，系统已就绪。")
    notifier.send("守护进程启动", f"实盘系统已上线 (PID: {os.getpid()})")

    # --- 2. 调度器 ---
    ny_tz = timezone('America/New_York')
    scheduler = AsyncIOScheduler(timezone=ny_tz)
    
    scheduler.add_job(
        job_trading_session, 
        CronTrigger(day_of_week='mon-fri', hour=9, minute=30, timezone=ny_tz),
        id='daily_trading'
    )
    scheduler.add_job(job_heartbeat, 'interval', seconds=5, id='heartbeat')
    scheduler.start()
    
    try:
        next_run = scheduler.get_job('daily_trading').next_run_time
        logger.info(f"📅 下次交易时间: {next_run} (Timezone: America/New_York)")
        logger.info("👁️ 进入后台监控模式 (按 Ctrl+C 退出)...")
    except: pass

    # --- 3. 守护循环与异常处理 (Robustness Layer) ---
    try:
        while True:
            await asyncio.sleep(1)
            
    except (KeyboardInterrupt, SystemExit):
        # [Case 1] 正常退出 (手动 Ctrl+C 或 网页点 STOP)
        logger.warning("👋 正在执行安全停机流程...")
        
        # 写遗言：把状态改成 Stopped
        save_dashboard_state({
            "status": "Stopped", 
            "info": "User manually stopped the service."
        })
        notifier.send("🔴 系统下线", "用户手动停止了守护进程。")
        
    except Exception as e:
        # [Case 2] 意外崩溃
        err_msg = traceback.format_exc()
        logger.error(f"☠️ 严重错误导致崩溃: {e}\n{err_msg}")
        
        # 写遗言：把状态改成 Crashed
        save_dashboard_state({
            "status": "Crashed", 
            "error": str(e)
        })
        notifier.send("☠️ 系统崩溃", f"守护进程意外退出！\n错误: {str(e)}")
        
    finally:
        # 无论如何都要关闭连接
        if trader:
            trader.stop()
        logger.info("✅ 进程已彻底结束。")

if __name__ == '__main__':
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        # 这里捕获是为了防止 asyncio.run 抛出的额外报错信息干扰视线
        pass