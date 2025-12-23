# -*- coding: utf-8 -*-
from ib_insync import Stock, util
import pandas as pd
import time
import xml.etree.ElementTree as ET
from datetime import datetime

class USEquityEngine:
    """
    专门负责下载美股（ETF与股票）数据的引擎
    支持：OHLCV, Turnover, MarketCap 计算
    """
    def __init__(self, ib_client):
        self.ib = ib_client

    def fetch_data(self, symbol: str, category: str, duration: str = '15 Y') -> pd.DataFrame:
        # 1. 剥离后缀进行查询 (SPY.P -> SPY)
        ib_symbol = symbol.split('.')[0]
        contract = Stock(ib_symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)

        print(f"📡 正在下载 {symbol} 的历史行情...")
        # 请求复权价格 (ADJUSTED_LAST)
        bars = self.ib.reqHistoricalData(
            contract, endDateTime='', durationStr=duration,
            barSizeSetting='1 day', whatToShow='ADJUSTED_LAST', useRTH=True
        )
        
        if not bars: return pd.DataFrame()
        
        df = util.df(bars)
        df = df.rename(columns={'date': 'datetime', 'average': 'avg_price'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['sec_code'] = symbol
        df['category_id'] = category

        # 2. 获取基本面数据以计算股本 (仅在需要计算 turnover 时执行)
        shares = self._get_shares_outstanding(contract)
        
        # 3. 计算你要求的字段
        df = df.sort_values('datetime')
        df['pre_close'] = df['close'].shift(1)
        df['simple_return'] = df['close'].pct_change().fillna(0)
        df['amount'] = df['volume'] * df['avg_price']
        df['shares_outstanding'] = shares
        # Turnover = 成交量 / 总股本
        df['turnover'] = df['volume'] / shares if shares > 0 else 0
        df['market_cap'] = df['close'] * shares
        df['create_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return df

    # --- 修改 us_equity_engine.py 中的这个私有方法 ---

    def _get_shares_outstanding(self, contract) -> float:
        """通过 IBKR 基本面接口获取发行股数，若失败则返回 0"""
        try:
            # 尝试获取基本面 XML
            # 如果是 ETF，经常会返回 Error 430，我们用 try-except 捕捉它
            raw_xml = self.ib.reqFundamentalData(contract, reportType='ReportsFinSummary')
            
            if not raw_xml:
                return 0.0
                
            tree = ET.fromstring(raw_xml)
            # 查找发行股数标签 (MSHOUT)
            for node in tree.iter('Ratio'):
                if node.get('FieldName') == 'mshout':
                    return float(node.text) * 1000000
        except Exception:
            # 静默处理 Error 430，不打印错误堆栈
            return 0.0
        return 0.0