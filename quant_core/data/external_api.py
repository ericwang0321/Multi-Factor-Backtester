import requests
import pandas as pd
from datetime import datetime, timedelta

class FinnhubService:
    """
    负责连接 Finnhub API 获取外部数据 (新闻, 情绪等)
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"

    def get_market_news(self, category="general"):
        """
        获取市场新闻 (Market News)
        :param category: general, forex, crypto, merger
        :return: List of news dicts
        """
        if not self.api_key: 
            return []
        
        try:
            # 请求 Finnhub News API
            url = f"{self.base_url}/news?category={category}&token={self.api_key}"
            res = requests.get(url, timeout=5) # 5秒超时防止卡顿
            
            if res.status_code == 200:
                data = res.json()
                # 只返回前 10 条最新的，避免页面过长
                return data[:10] if isinstance(data, list) else []
            else:
                print(f"⚠️ Finnhub API Error: {res.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ News fetch error: {e}")
            return []

    def get_company_sentiment(self, symbol):
        """
        获取个股内部人士情绪 (Insider Sentiment)
        :param symbol: 股票代码 (例如 AAPL)
        :return: DataFrame
        """
        if not self.api_key: 
            return pd.DataFrame()
        
        try:
            # 这里的 symbol 需要去掉 TradingView 的前缀 (例如 NASDAQ:AAPL -> AAPL)
            clean_symbol = symbol.split(':')[-1]
            
            # 获取过去 1 年的数据
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/stock/insider-sentiment?symbol={clean_symbol}&from={start_date}&token={self.api_key}"
            res = requests.get(url, timeout=5)
            
            if res.status_code == 200:
                json_data = res.json()
                data_list = json_data.get('data', [])
                
                if data_list:
                    df = pd.DataFrame(data_list)
                    # 简单清洗列名
                    if 'mspr' in df.columns:
                        df = df.rename(columns={'mspr': 'MSPR (Monthly)', 'change': 'Change'})
                    return df
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"❌ Sentiment fetch error: {e}")
            return pd.DataFrame()