import smtplib
import yaml
import os
import logging
from email.mime.text import MIMEText
from email.header import Header

class Notifier:
    """
    消息通知模块
    目前支持: Email (SMTP)
    """
    
    def __init__(self, config_path='config.yaml'):
        self.logger = logging.getLogger('live_trader') # 复用 logger
        self.config = self._load_config(config_path)
        self.email_config = self.config.get('notifications', {}).get('email', {})

    def _load_config(self, path):
        """加载 yaml 配置"""
        # 获取项目根目录路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        full_path = os.path.join(project_root, path)
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"❌ [Notifier] 无法读取配置文件: {e}")
            return {}

    def send_email(self, subject: str, message: str):
        """
        发送邮件通知
        """
        if not self.email_config.get('enabled', False):
            return

        sender = self.email_config.get('sender_email')
        password = self.email_config.get('sender_password')
        receiver = self.email_config.get('receiver_email')
        smtp_server = self.email_config.get('smtp_server')
        smtp_port = self.email_config.get('smtp_port', 587)

        if not all([sender, password, receiver, smtp_server]):
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
            server.starttls()  # 启用安全加密
            server.login(sender, password)
            server.sendmail(sender, [receiver], msg.as_string())
            server.quit()
            
            self.logger.info(f"📧 [Notifier] 邮件已发送至 {receiver}")
            
        except Exception as e:
            self.logger.error(f"❌ [Notifier] 邮件发送失败: {e}")

    def send(self, title, content):
        """通用发送接口，未来可以加钉钉/微信"""
        # 1. 发邮件
        self.send_email(f"[Quant] {title}", content)
        
        # 2. (预留) 发钉钉
        # self.send_dingtalk(...)