import os
import json
import time
import yaml
import logging
from datetime import datetime
from telegram import Update, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ================= 配置与路径 =================
# 定义文件路径
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

# ================= 辅助函数 =================

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
    """读取主程序状态文件 (dashboard_state.json)"""
    if not os.path.exists(STATE_FILE):
        return None
    try:
        # 使用 try-except 防止读取时正好主程序在写入造成的冲突
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def send_command(action):
    """向主程序发送指令 (写入 command.json)"""
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

# ================= 权限检查装饰器 =================
# 确保只有你可以控制机器人，防止其他人搜索到你的机器人进行捣乱

def restricted(func):
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        # 重新加载配置以获取最新的 chat_id
        tg_conf = load_secrets()
        if not tg_conf: return
        
        allowed_id = str(tg_conf.get('chat_id'))
        user_id = str(update.effective_user.id)
        
        if user_id != allowed_id:
            await update.message.reply_text(f"⛔️ 未授权访问 (ID: {user_id})")
            logger.warning(f"Unauthorized access attempt from {user_id}")
            return
        return await func(update, context, *args, **kwargs)
    return wrapped

# ================= 机器人命令处理逻辑 =================

@restricted
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/start: 启动欢迎语并设置快捷菜单"""
    user = update.effective_user.first_name
    await update.message.reply_text(
        f"👋 你好, {user}!\n\n"
        "我是你的量化交易管家。\n"
        "点击左下角 **[Menu]** 按钮查看可用指令，或者直接输入 /help。"
    )
    # 强制刷新菜单
    await setup_commands(context.application)

@restricted
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/help: 显示帮助列表"""
    msg = (
        "🤖 **可用快捷指令清单**\n"
        "------------------------------\n"
        "📊 /status - 查看账户权益、PnL及系统状态\n"
        "📋 /pos - 查看当前持仓详情\n"
        "📉 /flat - 【⚠️高危】一键清仓 (市价全平)\n"
        "🚫 /cancel - 撤销所有未成交挂单\n"
        "🛑 /stop - 停止交易系统进程\n"
        "🔄 /check - 手动检查心跳连接"
    )
    await update.message.reply_text(msg, parse_mode='Markdown')

@restricted
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/status: 查看系统状态"""
    state = read_state()
    if not state:
        await update.message.reply_text("⚠️ 无法读取状态文件 (dashboard_state.json 不存在)。\n主程序可能未启动。")
        return

    # 计算延迟
    last_update = state.get('updated_at', 'Unknown')
    is_alive = "✅ 在线"
    color = "🟢"
    
    if last_update != 'Unknown':
        try:
            dt = datetime.strptime(last_update, '%Y-%m-%d %H:%M:%S')
            diff = (datetime.now() - dt).total_seconds()
            if diff > 60:
                is_alive = f"❌ 离线 (延迟 {int(diff)}秒)"
                color = "🔴"
            elif diff > 15:
                is_alive = f"⚠️ 延迟 (延迟 {int(diff)}秒)"
                color = "🟡"
        except: pass

    acct = state.get('account', {})
    equity = acct.get('total_equity', 0)
    pnl = acct.get('unrealized_pnl', 0)
    sys_status = state.get('status', 'Unknown')

    msg = (
        f"{color} **System Status**\n"
        f"------------------\n"
        f"Run State: `{sys_status}`\n"
        f"Heartbeat: {is_alive}\n"
        f"Update: `{last_update}`\n\n"
        f"💰 **Net Liq**: `${equity:,.2f}`\n"
        f"📈 **Unrealized PnL**: `${pnl:,.2f}`"
    )
    await update.message.reply_text(msg, parse_mode='Markdown')

@restricted
async def positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/pos: 查看持仓"""
    state = read_state()
    if not state:
        await update.message.reply_text("⚠️ 无法获取状态数据。")
        return

    pos_dict = state.get('account', {}).get('positions', {})
    cost_dict = state.get('account', {}).get('avg_costs', {})
    
    # 过滤掉数量为 0 的持仓
    active_pos = {k: v for k, v in pos_dict.items() if v != 0}
    
    if not active_pos:
        await update.message.reply_text("💤 当前账户为空仓 (Flat)。")
        return
        
    msg = "📋 **Current Positions**\n------------------\n"
    for sym, qty in active_pos.items():
        avg_cost = cost_dict.get(sym, 0)
        msg += f"🔹 **{sym}**: `{qty}` shares @ `${avg_cost:.2f}`\n"
        
    await update.message.reply_text(msg, parse_mode='Markdown')

@restricted
async def flat_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/flat: 一键清仓"""
    if send_command("FLAT_ALL"):
        await update.message.reply_text("📉 **已发送 [FLAT ALL] 指令！**\n正在以市价卖出所有持仓，请留意 TWS 成交。")
    else:
        await update.message.reply_text("❌ 指令发送失败，请检查服务器日志。")

@restricted
async def cancel_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/cancel: 撤单"""
    if send_command("CANCEL_ALL"):
        await update.message.reply_text("🚫 **已发送 [CANCEL ALL] 指令！**\n正在撤销所有挂单。")
    else:
        await update.message.reply_text("❌ 指令发送失败。")

@restricted
async def stop_engine(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/stop: 停止主程序"""
    if send_command("STOP"):
        await update.message.reply_text("🛑 **已发送 [STOP] 指令！**\n交易守护进程将终止。")
    else:
        await update.message.reply_text("❌ 指令发送失败。")

@restricted
async def manual_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/check: 手动心跳检查"""
    state = read_state()
    if state:
        t = state.get('updated_at', 'Unknown')
        await update.message.reply_text(f"💓 系统文件存在，最后更新: {t}")
    else:
        await update.message.reply_text("💔 找不到状态文件，系统可能未运行。")

# ================= 后台看门狗 (Watchdog) =================

async def watchdog_job(context: ContextTypes.DEFAULT_TYPE):
    """每分钟检查一次主程序是否还活着"""
    job_data = context.job.data
    # 防止重复报警的标志位
    if 'alert_sent' not in job_data: job_data['alert_sent'] = False
    
    tg_conf = load_secrets()
    if not tg_conf: return
    chat_id = tg_conf.get('chat_id')

    state = read_state()
    
    # 情况 1: 状态文件完全不存在
    if not state:
        return 

    # 情况 2: 检查时间戳
    last_update = state.get('updated_at')
    if last_update:
        try:
            dt = datetime.strptime(last_update, '%Y-%m-%d %H:%M:%S')
            diff = (datetime.now() - dt).total_seconds()
            
            # 阈值：120秒 (2分钟) 无心跳则报警
            if diff > 120 and not job_data['alert_sent']:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"🚨 **CRITICAL ALERT** 🚨\n\n主交易程序已失去响应！\n最后心跳: {last_update} ({int(diff)}秒前)\n\n请立即检查服务器！",
                    parse_mode='Markdown'
                )
                job_data['alert_sent'] = True
            
            # 恢复通知
            if diff < 60 and job_data['alert_sent']:
                 await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"✅ **SYSTEM RECOVERED**\n\n主程序心跳已恢复正常。",
                    parse_mode='Markdown'
                )
                 job_data['alert_sent'] = False
                 
        except Exception as e:
            logger.error(f"Watchdog error: {e}")

# ================= 菜单设置 =================

async def setup_commands(application):
    """自动设置 Telegram 左下角的 Menu 按钮"""
    commands = [
        BotCommand("status", "📊 查看系统状态 & PnL"),
        BotCommand("pos", "📋 查看当前持仓"),
        BotCommand("check", "💓 检查连接"),
        BotCommand("cancel", "🚫 撤销所有挂单"),
        BotCommand("flat", "📉 [高危] 一键市价清仓"),
        BotCommand("stop", "🛑 停止交易进程"),
        BotCommand("help", "❓ 显示帮助文档"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("✅ Telegram 快捷指令菜单已更新")

# ================= 主程序入口 =================

if __name__ == '__main__':
    # 1. 加载配置
    conf = load_secrets()
    if not conf or not conf.get('token'):
        print("❌ 错误：在 config/secrets.yaml 中找不到 telegram.token")
        exit(1)
        
    TOKEN = conf['token']
    
    print("🚀 正在启动 Telegram 交易管家...")
    
    # 2. 构建应用
    app = ApplicationBuilder().token(TOKEN).build()

    # 3. 注册命令处理
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("pos", positions))
    app.add_handler(CommandHandler("flat", flat_all))
    app.add_handler(CommandHandler("cancel", cancel_all))
    app.add_handler(CommandHandler("stop", stop_engine))
    app.add_handler(CommandHandler("check", manual_check))

    # 4. 启动启动时的钩子 (设置菜单)
    # 注意：post_init 只在 run_polling 启动后执行
    async def post_init(application):
        await setup_commands(application)
    app.post_init = post_init

    # 5. 启动看门狗 (每 60 秒检查一次)
    job_queue = app.job_queue
    if job_queue:
        job_queue.run_repeating(watchdog_job, interval=60, first=10, data={})
    else:
        print("⚠️ 警告: JobQueue 未启用，看门狗功能将不可用。请安装 python-telegram-bot[job-queue]")

    # 6. 开始轮询
    print("✅ Bot 已上线！请在 Telegram 中发送 /start")
    app.run_polling()