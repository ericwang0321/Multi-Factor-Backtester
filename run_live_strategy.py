# run_live_strategy.py
import asyncio
import os
import sys
import json
import math
import traceback
import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone

# [新增] 引入市场日历库 (pip install pandas_market_calendars)
import pandas_market_calendars as mcal

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
# 0. 市场日历检查工具 (Helpers)
# ==============================================================================

def check_is_market_open():
    """
    检查今天是否是美股交易日 (NYSE)
    返回: (bool, str) -> (是否开盘, 原因/描述)
    """
    # 获取纽约时间
    ny_tz = timezone('America/New_York')
    now_ny = datetime.now(ny_tz)
    today_str = now_ny.strftime('%Y-%m-%d')
    
    # 获取 NYSE 日历
    nyse = mcal.get_calendar('NYSE')
    
    # 检查今天是否有安排
    schedule = nyse.schedule(start_date=today_str, end_date=today_str)
    
    if schedule.empty:
        return False, f"Holiday/Weekend ({today_str})"
    
    # 额外检查：如果是提前休市 (Early Close)，也视为交易日，但可以记录一下
    return True, "Market Open"

# ==============================================================================
# 1. 核心任务逻辑 (Tasks)
# ==============================================================================

async def job_trading_session():
    """
    【交易任务】每天美东时间 09:15 触发 (盘前准备)
    逻辑：
    1. 检查是不是假期 -> 2. 拉取截至昨日的数据 -> 3. 算号 -> 4. 挂单 (TWS会自动等到09:30成交)
    """
    logger.info("⏰ [Scheduler] 触发每日定时任务...")
    
    # --- Step 1: 节假日检查 ---
    is_open, reason = check_is_market_open()
    if not is_open:
        logger.info(f"☕️ 今天美股休市: {reason}，任务跳过。")
        # 更新一下状态文件，告诉前端我醒过，但是没干活
        save_dashboard_state({
            "status": "Sleeping (Holiday)",
            "info": f"Market Closed: {reason}"
        })
        return

    notifier.send("实盘启动", f"正在执行每日策略逻辑 (盘前准备)...\n市场状态: {reason}")
    
    try:
        # 1. 确保连接健康
        if not trader or not trader.connector.ib.isConnected():
            logger.warning("⚠️ IB 未连接，尝试重连...")
            # 简单重连尝试
            try:
                if trader: trader.start()
            except: pass
            
            # 如果还连不上，报错退出
            await asyncio.sleep(5)
            if not trader or not trader.connector.ib.isConnected():
                notifier.send("连接失败", "IB TWS 未连接，无法交易。")
                return

        # 2. 策略实例化
        strategy = create_strategy_instance(CONF['strategy'])
        bridge = LiveDataBridge(trader.connector, CONF['universe_path'])
        
        # 3. 数据准备 (Data Pulling)
        # 注意：在 09:15 拉取数据时，IB 会返回截止到昨天收盘的日线数据
        logger.info("📡 正在拉取 IB 历史数据 (截至昨日收盘)...")
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
        
        # 这里计算出的 target_weights 是基于“昨天收盘价”算出的理想仓位
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
                
                # [关键] 此时是 09:15，发送的是普通 Market Order。
                # TWS 会将其状态置为 "PreSubmitted" (排队中)，直到 09:30 开盘瞬间触发。
                trader.execute_rebalance(target_qtys)
                
                notifier.send("挂单完成", f"已发送订单至 TWS (等待开盘成交)。\n{details}")
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
    """原子写入状态文件"""
    try:
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
            logger.warning("🛑 停止指令已接收...")
            sys.exit(0)
        elif action == 'CANCEL_ALL':
            trader_instance.cancel_all_orders()
        elif action == 'FLAT_ALL':
            # 调用 Trader 的一键清仓
            logger.warning("📉 收到清仓指令，正在执行...")
            trader_instance.close_all_positions()
            notifier.send("⚠️ 紧急清仓", "已执行一键清仓 (FLAT ALL)，所有挂单已撤销，持仓正在市价卖出。")
            
    except SystemExit:
        raise
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
# 3. 异步启动入口 (Main Entry)
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

    # --- 2. 调度器 (美东时间) ---
    ny_tz = timezone('America/New_York')
    scheduler = AsyncIOScheduler(timezone=ny_tz)
    
    # [修改点] 将时间改为 09:15，实现盘前算号
    scheduler.add_job(
        job_trading_session, 
        # 周一到周五触发，具体是否开盘由 job 内部的日历检查决定
        CronTrigger(day_of_week='mon-fri', hour=9, minute=15, timezone=ny_tz),
        id='daily_trading'
    )
    scheduler.add_job(job_heartbeat, 'interval', seconds=5, id='heartbeat')
    scheduler.start()
    
    try:
        next_run = scheduler.get_job('daily_trading').next_run_time
        logger.info(f"📅 下次任务检查时间: {next_run} (Timezone: America/New_York)")
        logger.info("👁️ 进入后台监控模式 (按 Ctrl+C 退出)...")
    except: pass

    # --- 3. 守护循环 ---
    try:
        while True:
            await asyncio.sleep(1)
            
    except (KeyboardInterrupt, SystemExit):
        logger.warning("👋 正在执行安全停机流程...")
        save_dashboard_state({"status": "Stopped", "info": "User manually stopped."})
        notifier.send("🔴 系统下线", "用户手动停止了守护进程。")
        
    except Exception as e:
        err_msg = traceback.format_exc()
        logger.error(f"☠️ 严重错误导致崩溃: {e}\n{err_msg}")
        save_dashboard_state({"status": "Crashed", "error": str(e)})
        notifier.send("☠️ 系统崩溃", f"守护进程意外退出！\n错误: {str(e)}")
        
    finally:
        if trader: trader.stop()
        logger.info("✅ 进程已彻底结束。")

if __name__ == '__main__':
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        pass