import os
import json
import time
import yaml
import logging
import sys
import requests  # [必须安装] pip install requests
from datetime import datetime
from telegram import Update, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ================= 配置与路径 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SECRETS_PATH = os.path.join(BASE_DIR, 'config', 'secrets.yaml')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'live')
STATE_FILE = os.path.join(DATA_DIR, 'dashboard_state.json')
COMMAND_FILE = os.path.join(DATA_DIR, 'command.json')

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ================= 核心辅助函数 =================

def load_secrets():
    """从 secrets.yaml 加载配置"""
    if not os.path.exists(SECRETS_PATH):
        logger.error(f"❌ 找不到配置文件: {SECRETS_PATH}")
        return None
    try:
        with open(SECRETS_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('telegram', {})
    except Exception as e:
        logger.error(f"❌ 读取配置文件失败: {e}")
        return None

def read_state():
    """读取主程序状态文件"""
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def send_command(action):
    """向主程序发送指令"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    cmd = {"action": action, "timestamp": time.time()}
    try:
        with open(COMMAND_FILE, 'w') as f:
            json.dump(cmd, f)
        return True
    except Exception as e:
        logger.error(f"指令写入失败: {e}")
        return False

# ================= [关键] 临终遗言 (同步发送) =================

def send_shutdown_alert():
    """
    程序退出前的最后一步操作。
    使用 requests 库同步发送，不依赖异步事件循环，确保消息一定能发出。
    """
    print("\n💀 正在发送离线通知...")
    
    conf = load_secrets()
    if not conf:
        return

    token = conf.get('token')
    chat_id = conf.get('chat_id')
    
    if token and chat_id:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": "🛑 **SYSTEM SHUTDOWN**\n\n量化系统已断开连接。\n(原因: 手动停止/Honcho关闭)",
            "parse_mode": "Markdown"
        }
        try:
            # 设置 2 秒超时，防止网络卡死导致程序退不出去
            requests.post(url, json=data, timeout=2)
            print("✅ 离线通知已发送给 Telegram 服务器。")
        except Exception as e:
            print(f"❌ 离线通知发送失败: {e}")

# ================= 权限与指令逻辑 =================

def restricted(func):
    """装饰器：只允许特定 Chat ID 操作"""
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        tg_conf = load_secrets()
        if not tg_conf: return
        
        allowed_id = str(tg_conf.get('chat_id'))
        # 部分更新可能没有 effective_user，做个保护
        if not update.effective_user: return
        
        user_id = str(update.effective_user.id)
        
        if user_id != allowed_id:
            await update.message.reply_text(f"⛔️ 未授权访问 (ID: {user_id})")
            logger.warning(f"Unauthorized access attempt from {user_id}")
            return
        return await func(update, context, *args, **kwargs)
    return wrapped

@restricted
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/start: 启动欢迎语"""
    await update.message.reply_text(
        "👋 量化管家已上线！\n"
        "点击左下角 **[Menu]** 按钮查看可用指令。"
    )
    # 强制刷新菜单
    await setup_commands(context.application)

@restricted
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/status: 查看系统状态"""
    state = read_state()
    if not state:
        await update.message.reply_text("⚠️ 无法读取状态文件 (dashboard_state.json)。")
        return

    # 心跳检测
    last_update = state.get('updated_at', 'Unknown')
    is_alive = "✅ 在线"
    color = "🟢"
    
    if last_update != 'Unknown':
        try:
            dt = datetime.strptime(last_update, '%Y-%m-%d %H:%M:%S')
            diff = (datetime.now() - dt).total_seconds()
            if diff > 60:
                is_alive = f"❌ 离线 ({int(diff)}s ago)"
                color = "🔴"
            elif diff > 15:
                is_alive = f"⚠️ 延迟 ({int(diff)}s ago)"
                color = "🟡"
        except: pass

    acct = state.get('account', {})
    equity = acct.get('total_equity', 0)
    pnl = acct.get('unrealized_pnl', 0)
    sys_status = state.get('status', 'Unknown')

    msg = (
        f"{color} **System Status**\n"
        f"------------------\n"
        f"State: `{sys_status}`\n"
        f"Heartbeat: {is_alive}\n"
        f"Updated: `{last_update}`\n\n"
        f"💰 **Net Liq**: `${equity:,.0f}`\n"
        f"📈 **Unrealized PnL**: `${pnl:,.0f}`"
    )
    await update.message.reply_text(msg, parse_mode='Markdown')

@restricted
async def positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/pos: 查看持仓"""
    state = read_state()
    if not state: return

    pos_dict = state.get('account', {}).get('positions', {})
    cost_dict = state.get('account', {}).get('avg_costs', {})
    
    # 过滤掉数量为 0 的
    active_pos = {k: v for k, v in pos_dict.items() if v != 0}
    
    if not active_pos:
        await update.message.reply_text("💤 当前账户为空仓 (Flat)。")
        return
        
    msg = "📋 **Positions**\n------------------\n"
    for sym, qty in active_pos.items():
        avg_cost = cost_dict.get(sym, 0)
        msg += f"🔹 **{sym}**: `{qty}` @ `${avg_cost:.2f}`\n"
        
    await update.message.reply_text(msg, parse_mode='Markdown')

@restricted
async def flat_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/flat: 一键清仓"""
    if send_command("FLAT_ALL"):
        await update.message.reply_text("📉 **已发送 [FLAT ALL] 指令！**\n正在以市价卖出所有持仓。")
    else:
        await update.message.reply_text("❌ 指令写入失败。")

@restricted
async def cancel_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/cancel: 撤单"""
    if send_command("CANCEL_ALL"):
        await update.message.reply_text("🚫 **已发送 [CANCEL ALL] 指令！**")
    else:
        await update.message.reply_text("❌ 指令写入失败。")

@restricted
async def stop_engine(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/stop: 停止主程序"""
    if send_command("STOP"):
        await update.message.reply_text("🛑 **已发送 [STOP] 指令！**")
    else:
        await update.message.reply_text("❌ 指令写入失败。")

@restricted
async def manual_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/check: 手动检查"""
    state = read_state()
    t = state.get('updated_at', 'Unknown') if state else "None"
    await update.message.reply_text(f"💓 状态文件时间戳: {t}")

# ================= 看门狗 (Watchdog) =================

async def watchdog_job(context: ContextTypes.DEFAULT_TYPE):
    """后台监控主程序心跳"""
    job_data = context.job.data
    if 'alert_sent' not in job_data: job_data['alert_sent'] = False
    
    tg_conf = load_secrets()
    if not tg_conf: return
    chat_id = tg_conf.get('chat_id')

    state = read_state()
    if not state: return 

    last_update = state.get('updated_at')
    if last_update:
        try:
            dt = datetime.strptime(last_update, '%Y-%m-%d %H:%M:%S')
            diff = (datetime.now() - dt).total_seconds()
            
            # 超过 120 秒没心跳 -> 报警
            if diff > 120 and not job_data['alert_sent']:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"🚨 **CRITICAL ALERT** 🚨\n\n主交易程序无响应！\n延迟: {int(diff)}秒\n请立即检查服务器！",
                    parse_mode='Markdown'
                )
                job_data['alert_sent'] = True
            
            # 恢复正常 -> 通知
            if diff < 60 and job_data['alert_sent']:
                 await context.bot.send_message(chat_id=chat_id, text="✅ 主程序心跳已恢复。")
                 job_data['alert_sent'] = False
        except: pass

# ================= 菜单设置 =================

async def setup_commands(application):
    """设置左下角 Menu 按钮"""
    commands = [
        BotCommand("status", "📊 查看状态"),
        BotCommand("pos", "📋 查看持仓"),
        BotCommand("check", "💓 检查连接"),
        BotCommand("cancel", "🚫 撤销挂单"),
        BotCommand("flat", "📉 [高危] 一键清仓"),
        BotCommand("stop", "🛑 停止程序"),
    ]
    await application.bot.set_my_commands(commands)

# ================= 主程序入口 =================

if __name__ == '__main__':
    # 1. 加载配置
    conf = load_secrets()
    if not conf or not conf.get('token'):
        print("❌ 错误：secrets.yaml 中找不到 Token")
        sys.exit(1)
        
    TOKEN = conf['token']
    print("🚀 Telegram Bot 启动中...")
    
    # 2. 构建应用
    application = ApplicationBuilder().token(TOKEN).build()

    # 3. 注册命令
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("pos", positions))
    application.add_handler(CommandHandler("flat", flat_all))
    application.add_handler(CommandHandler("cancel", cancel_all))
    application.add_handler(CommandHandler("stop", stop_engine))
    application.add_handler(CommandHandler("check", manual_check))

    # 4. 启动看门狗
    if application.job_queue:
        application.job_queue.run_repeating(watchdog_job, interval=60, first=10, data={})

    # 5. 启动后自动更新菜单
    async def post_init(app):
        await setup_commands(app)
    application.post_init = post_init

    print("✅ Bot 已上线！(按 Ctrl+C 关闭时会自动发送通知)")

    # 6. 【核心】带保护的运行循环
    try:
        # run_polling 会一直阻塞在这里，直到收到 Ctrl+C
        application.run_polling()
        
    except (KeyboardInterrupt, SystemExit):
        # 捕获退出信号，什么都不做，让它自然流转到 finally
        pass
        
    except Exception as e:
        print(f"❌ 发生运行时错误: {e}")
        
    finally:
        # =======================================================
        # 无论程序是因为报错挂了，还是被 Ctrl+C 关了，
        # 这里是它临死前必须经过的地方。
        # =======================================================
        send_shutdown_alert()
        print("👋 Bot 进程彻底结束")