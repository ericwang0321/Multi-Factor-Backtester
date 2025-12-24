import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple
from .ib_connector import IBKRConnector

class LiveDataBridge:
    """
    数据桥接层 (Data Bridge)
    
    职责:
    1. 管理实盘标的池 (Universe)。
    2. 从 IB 获取足够的历史窗口数据 (Warm-up Data)。
    3. 将 Raw OHLCV 数据实时转化为策略所需的 Factor Data。
    4. 兼容日频 (Daily) 和分钟频 (Intraday) 拓展。
    """

    def __init__(self, connector: IBKRConnector, universe_csv_path: str):
        self.connector = connector
        self.universe_map = self._load_universe(universe_csv_path)
        self.codes = list(self.universe_map.keys()) # ['PFF.O', 'SPY.P', ...]
        self.ib_symbols = list(self.universe_map.values()) # ['PFF', 'SPY', ...]
        
        # 缓存最近一次获取的数据，避免重复请求
        self.raw_data_cache: Dict[str, pd.DataFrame] = {} 

    def _load_universe(self, csv_path: str) -> Dict[str, str]:
        """
        读取标的文件，建立映射: 'SPY.P' -> 'SPY'
        IBKR 只识别 symbol (SPY)，但策略内部逻辑可能使用完整代码 (SPY.P)
        """
        if not os.path.exists(csv_path):
            print(f"⚠️ 警告: 找不到标的文件 {csv_path}")
            return {}
            
        df = pd.read_csv(csv_path)
        mapping = {}
        for code in df['sec_code']:
            # 简单的解析逻辑: 去掉后缀
            # PFF.O -> PFF, SPY.P -> SPY
            symbol = code.split('.')[0] 
            mapping[code] = symbol
            
        print(f"[{self.__class__.__name__}] 加载标的池: {len(mapping)} 个 ({list(mapping.values())[:3]}...)")
        return mapping

    def prepare_data_for_strategy(self, 
                                  required_factors: List[str], 
                                  lookback_window: int = 60,
                                  bar_size: str = '1 day') -> Tuple[pd.DataFrame, pd.Series]:
        """
        核心方法: 为策略准备 "当日" 的输入数据
        
        Args:
            required_factors: 策略需要的因子列表，如 ['RSI', 'Momentum']
            lookback_window: 回溯窗口。计算 MA20 至少需要 20 天，建议取 60+ 以防万一。
            bar_size: IB 数据频率 ('1 day', '1 min', '5 mins')
            
        Returns:
            factor_df: 因子数据 DataFrame (Index=Code, Columns=Factors) -> 用于算分
            current_prices: 最新价格 Series (Index=Code) -> 用于风控止损
        """
        print(f"\n⚡ 正在获取 {len(self.codes)} 只标的的数据 (Lookback: {lookback_window}, Freq: {bar_size})...")
        
        factor_records = []
        price_records = {}
        
        # 确定 IB 请求的 duration string
        # 如果是日频，请求 'X D'; 如果是分钟频，请求 'X S' (秒) 或 'X D'
        if 'day' in bar_size:
            duration_str = f"{lookback_window} D"
        else:
            # 简易处理分钟频逻辑: 假设请求最近 2 天的数据用于计算分钟级指标
            duration_str = "2 D" 

        for full_code, symbol in self.universe_map.items():
            # 1. 从 IB 获取历史 K 线 (包含直到最新一刻的数据)
            df = self.connector.get_historical_data(symbol, duration=duration_str, bar_size=bar_size)
            
            if df.empty:
                print(f"⚠️ 无数据: {symbol}")
                continue
                
            # 2. 提取最新价格 (用于 Stop Loss)
            # 假设 df 最后一行是最新数据
            current_price = df['Close'].iloc[-1]
            price_records[full_code] = current_price
            
            # 3. 实时计算因子 (Feature Engineering)
            # 这里是将 Raw Data -> Factor Data 的关键
            factors = self._calculate_factors_on_the_fly(df, required_factors)
            
            # 这里的 factors 是一个 Series (单只股票的因子值)
            # 我们将其放入列表，最后合并
            factors.name = full_code
            factor_records.append(factors)
            
        # 4. 组装结果
        if factor_records:
            factor_df = pd.DataFrame(factor_records) # Index 自动变为 code
        else:
            factor_df = pd.DataFrame()
            
        current_prices = pd.Series(price_records)
        
        return factor_df, current_prices

    def _calculate_factors_on_the_fly(self, df: pd.DataFrame, factor_names: List[str]) -> pd.Series:
        """
        即时因子计算引擎。
        注意: 这里必须复刻你在回测阶段 (Data Preparation) 的因子计算逻辑!
        如果回测用的是 talib，这里最好也用 talib。
        
        Args:
            df: 单只股票的 OHLCV 数据 (Index=Datetime)
            factor_names: 需要计算的因子名列表
        """
        # 确保按时间排序
        df = df.sort_index()
        
        results = {}
        close = df['Close']
        
        # --- 下面是简单的因子计算示例，需根据你的真实策略调整 ---
        
        # 1. Momentum (动量) - 假设是过去 20 天收益率
        if 'Momentum' in factor_names:
            # pct_change(20) 取最后一天作为当前值
            mom = close.pct_change(20).iloc[-1]
            results['Momentum'] = mom
            
        # 2. RSI (相对强弱) - 假设窗口 14
        if 'RSI' in factor_names:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            results['RSI'] = rsi.iloc[-1] # 取最新值

        # 3. Volatility (波动率) - 20天
        if 'Volatility' in factor_names:
            vol = close.pct_change().rolling(20).std().iloc[-1]
            results['Volatility'] = vol
            
        # --- ML 策略拓展预留 ---
        # 如果是 ML 策略，可以在这里加载预训练模型 (pkl)，
        # 将 df 的特征输入模型，预测出 'y_pred' 作为因子返回。
        
        # 填充缺失值 (防止计算初期 NaN 导致报错)
        res_series = pd.Series(results).fillna(0)
        
        # 确保返回所有请求的因子 (即使没算出来也要给个 0 或 NaN)
        for f in factor_names:
            if f not in res_series:
                res_series[f] = 0.0
                
        return res_series