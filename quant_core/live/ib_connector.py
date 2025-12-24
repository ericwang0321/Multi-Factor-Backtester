import pandas as pd
from ib_insync import *
import nest_asyncio
import datetime

# 解决 Streamlit/Jupyter 中的事件循环冲突
nest_asyncio.apply()

class IBKRConnector:
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self._is_connected = False

    def connect(self):
        """连接到 IB TWS"""
        if not self.ib.isConnected():
            try:
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                self._is_connected = True
                print(f"✅ [IBKR] 成功连接到端口 {self.port} (Client ID: {self.client_id})")
            except Exception as e:
                print(f"❌ [IBKR] 连接失败: {e}")
                self._is_connected = False
        else:
            print("ℹ️ [IBKR] 已经连接")

    def disconnect(self):
        """断开连接"""
        if self.ib.isConnected():
            self.ib.disconnect()
            self._is_connected = False
            print("🔌 [IBKR] 已断开连接")

    def get_us_stock_contract(self, symbol: str):
        """
        创建美股/ETF合约对象
        IB 中 ETF (如 SPY) 和股票 (如 AAPL) 类型都是 'STK'
        """
        return Stock(symbol, 'SMART', 'USD')

    def get_historical_data(self, symbol: str, duration: str = '30 D', bar_size: str = '1 day') -> pd.DataFrame:
        """获取历史数据 (用于策略初始化)"""
        if not self.ib.isConnected():
            print("⚠️ 未连接，无法获取数据")
            return pd.DataFrame()

        contract = self.get_us_stock_contract(symbol)
        
        # 请求历史数据
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        # 转为 DataFrame
        df = util.df(bars)
        if df is not None and not df.empty:
            df.set_index('date', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            return df
        return pd.DataFrame()

    def get_current_positions(self) -> dict:
        """
        获取当前真实持仓
        返回格式: {'SPY': 100, 'AAPL': -50}
        """
        positions = self.ib.positions()
        pos_dict = {}
        for p in positions:
            # 这里的 contract.localSymbol 通常就是代码，如 'SPY'
            symbol = p.contract.localSymbol
            pos_dict[symbol] = p.position
        return pos_dict

    def get_account_summary(self):
        """获取账户净值等信息"""
        # tags: NetLiquidation (净值), AvailableFunds (可用资金)
        summary = self.ib.accountSummary()
        # 简单解析一下净值
        net_liq = next((x.value for x in summary if x.tag == 'NetLiquidation'), '0')
        return float(net_liq)

    def place_order(self, symbol: str, quantity: int, order_type: str = 'MKT'):
        """
        下单基础函数
        quantity: 正数为买，负数为卖
        """
        contract = self.get_us_stock_contract(symbol)
        action = 'BUY' if quantity > 0 else 'SELL'
        qty = abs(quantity)
        
        if order_type == 'MKT':
            order = MarketOrder(action, qty)
        else:
            # 扩展性：以后可以加限价单 LMT
            print(f"暂不支持的订单类型: {order_type}")
            return None

        trade = self.ib.placeOrder(contract, order)
        print(f"🚀 [Order] 已提交: {action} {qty} {symbol}")
        return trade