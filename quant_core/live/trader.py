import pandas as pd
import time
from .ib_connector import IBKRConnector

class LiveTrader:
    def __init__(self, account_id=None):
        # 初始化连接器
        self.connector = IBKRConnector()
        self.account_id = account_id # 可选，用于多账户时指定

    def start(self):
        """启动连接"""
        self.connector.connect()

    def stop(self):
        """停止连接"""
        self.connector.disconnect()

    def execute_rebalance(self, target_positions: dict):
        """
        核心交易逻辑：根据目标持仓进行调仓
        
        :param target_positions: 字典, 例如 {'SPY': 100, 'TLT': 50} 表示希望持有 100股 SPY 和 50股 TLT
        """
        print("\n--- 开始执行调仓逻辑 ---")
        
        # 1. 获取当前 IB 真实持仓
        current_positions = self.connector.get_current_positions()
        print(f"当前真实持仓: {current_positions}")
        print(f"目标持仓: {target_positions}")

        # 2. 计算差额并下单
        # 合并所有涉及的 symbol
        all_symbols = set(target_positions.keys()) | set(current_positions.keys())

        for symbol in all_symbols:
            target_qty = target_positions.get(symbol, 0) # 如果目标里没有，说明要清仓，target=0
            current_qty = current_positions.get(symbol, 0)
            
            diff = target_qty - current_qty
            
            if diff != 0:
                print(f"代码: {symbol} | 当前: {current_qty} -> 目标: {target_qty} | 需交易: {diff}")
                # 调用连接器下单
                self.connector.place_order(symbol, diff)
                # 为了防止 TWS 报 "Rate limit exceeded"，稍微停顿一下
                time.sleep(0.5)
            else:
                print(f"代码: {symbol} | 持仓已匹配 ({current_qty})，无需交易")

    def get_market_status(self):
        """简单的看盘状态"""
        nav = self.connector.get_account_summary()
        return {
            "Net Liquidation": nav,
            "Connected": self.connector.ib.isConnected()
        }

    def cancel_all_orders(self):
        """
        [新增] 撤销所有未成交的挂单
        """
        print("🚫 正在请求撤销所有订单...")
        # ib_insync 的全局撤单指令
        self.connector.ib.reqGlobalCancel()
        
    # =========== [新增以下方法] ===========
    def close_all_positions(self):
        """
        [新增] 一键清仓：市价平掉所有持仓
        """
        print("\n🚨 [FLAT ALL] 正在执行一键清仓程序...")
        
        # 1. 第一步：必须先撤销所有挂单，防止平仓后挂单成交又买回来了
        self.cancel_all_orders()
        time.sleep(1) # 给 TWS 一点时间处理撤单
        
        # 2. 第二步：获取当前持仓
        # 注意：这里我们不需要手动写循环卖出的逻辑，直接复用 execute_rebalance！
        current_positions = self.connector.get_current_positions()
        
        if not current_positions:
            print("✅ 当前账户为空仓，无需操作。")
            return

        print(f"当前持仓待清空: {list(current_positions.keys())}")

        # 3. 第三步：传入“空字典”作为目标
        # 逻辑：目标是 0，当前是 X，差额就是 -X (卖出所有)
        self.execute_rebalance(target_positions={})
        
        print("✅ 清仓指令已全部发送，账户即将归零。")