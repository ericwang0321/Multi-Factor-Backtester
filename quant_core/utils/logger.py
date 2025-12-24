import logging
import os
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

def setup_logger(name: str = 'live_trader', log_level: int = logging.INFO):
    """
    配置全局日志记录器 (Logger)
    
    功能:
    1. 自动创建 logs/ 目录
    2. 配置控制台输出 (StreamHandler)
    3. 配置每日文件轮转输出 (TimedRotatingFileHandler)
    
    Args:
        name: Logger 的名称
        log_level: 日志级别 (默认 INFO, 调试可用 DEBUG)
        
    Returns:
        logging.Logger: 配置好的 logger 对象
    """
    
    # 1. 确定日志保存路径
    # 获取当前项目根目录 (假设 utils 在 quant_core 下，向上两级)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    log_dir = os.path.join(project_root, 'logs')
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"📁 [System] 自动创建日志目录: {log_dir}")

    # 2. 获取 Logger 对象
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 3. 防止重复添加 Handler (避免日志重复打印)
    if logger.hasHandlers():
        return logger

    # 4. 定义日志格式
    # 格式示例: [2025-12-24 11:30:05] [INFO] 策略初始化完成...
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 5. 配置控制台输出 (Console Handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    # 6. 配置文件输出 (File Handler)
    # 文件名示例: logs/live_trading_2025-12-24.log
    today_str = datetime.now().strftime('%Y-%m-%d')
    log_filename = f"live_trading_{today_str}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # 使用 TimedRotatingFileHandler 实现按天分割日志
    # when='midnight': 每天午夜滚动
    # interval=1: 每1天
    # backupCount=30: 保留最近30天的日志
    file_handler = TimedRotatingFileHandler(
        filename=log_filepath,
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8' # 关键: 确保中文不乱码
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)

    logger.info(f"📝 日志系统初始化完成。日志文件: {log_filepath}")
    
    return logger

# 为了方便直接导入使用，也可以实例化一个默认 logger
# from quant_core.utils.logger import GLOBAL_LOGGER
# GLOBAL_LOGGER = setup_logger()