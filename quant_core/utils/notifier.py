import smtplib
import yaml
import os
import logging
import requests  # [新增] 引入 requests 用于发送 Telegram 消息
from email.mime.text import MIMEText
from email.header import Header

class Notifier:
    """
    消息通知模块
    支持: Email (SMTP) + Telegram Bot
    """
    
    # [修改] 默认路径改为 config/secrets.yaml，因为你的 Token 存在那里
    def __init__(self, config_path='config/secrets.yaml'):
        self.logger = logging.getLogger('live_trader') 
        self.config = self._load_config(config_path)
        
        # [修改] 适配 secrets.yaml 的结构 (直接读取 email 和 telegram 字段)
        self.email_config = self.config.get('email', {})
        self.tg_config = self.config.get('telegram', {})

    def _load_config(self, path):
        """加载 yaml 配置 (保留原有健壮的路径查找逻辑)"""
        # 获取项目根目录路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 这里假设 utils 在 quant_core 下，quant_core 在根目录下，所以往上跳两级
        project_root = os.path.dirname(os.path.dirname(current_dir))
        full_path = os.path.join(project_root, path)
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"❌ [Notifier] 无法读取配置文件: {full_path} | {e}")
            return {}

    def send_telegram(self, subject: str, message: str):
        """
        [新增] 发送 Telegram 通知
        """
        # 1. 检查开关和配置
        if not self.tg_config.get('enabled', False):
            return

        token = self.tg_config.get('token')
        chat_id = self.tg_config.get('chat_id')

        if not token or not chat_id:
            self.logger.warning("⚠️ [Notifier] Telegram 配置缺失 (Token 或 ChatID)，跳过发送。")
            return

        # 2. 发送请求
        try:
            # 改用 HTML 格式，比 Markdown 稳定得多
            full_text = f"<b>📢 {subject}</b>\n\n{message}"
            
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": full_text,
                "parse_mode": "HTML"  # [这里改为 HTML]
                }
            
            # 设置超时，防止网络卡顿影响交易线程
            resp = requests.post(url, json=payload, timeout=5)
            
            if resp.status_code == 200:
                self.logger.info("✅ [Notifier] Telegram 消息发送成功")
            else:
                self.logger.error(f"❌ [Notifier] Telegram 发送失败: {resp.text}")
                
        except Exception as e:
            self.logger.error(f"❌ [Notifier] Telegram 连接异常: {e}")

    def send_email(self, subject: str, message: str):
        """
        发送邮件通知 (保留原有逻辑，仅微调配置读取)
        """
        # [兼容] 如果 secrets.yaml 里没有 enabled 字段，默认尝试发送（或者你可以手动加 enabled: true）
        # 这里假设只要配置了 sender_email 就发送
        sender = self.email_config.get('sender_email')
        password = self.email_config.get('password') # 注意 secrets.yaml 里通常叫 password
        receiver = self.email_config.get('receiver_email')
        smtp_server = self.email_config.get('smtp_server')
        smtp_port = self.email_config.get('smtp_port', 587)

        # 如果关键信息缺失，则跳过
        if not all([sender, password, receiver, smtp_server]):
            # 只有当 email_config 有内容但缺字段时才警告，完全空则认为是未配置
            if self.email_config:
                self.logger.warning("⚠️ [Notifier] 邮件配置不完整，跳过发送。")
            return

        try:
            # 构造邮件内容
            msg = MIMEText(message, 'plain', 'utf-8')
            msg['From'] = sender
            msg['To'] = receiver
            msg['Subject'] = Header(subject, 'utf-8')

            # 连接 SMTP 服务器
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, [receiver], msg.as_string())
            server.quit()
            
            self.logger.info(f"📧 [Notifier] 邮件已发送至 {receiver}")
            
        except Exception as e:
            self.logger.error(f"❌ [Notifier] 邮件发送失败: {e}")

    def send(self, title, content):
        """
        通用发送接口
        """
        # 1. 优先发 Telegram (速度快)
        self.send_telegram(title, content)
        
        # 2. 发送邮件 (内容存档)
        self.send_email(f"[Quant] {title}", content)