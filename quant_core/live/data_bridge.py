# quant_core/live/data_bridge.py
import pandas as pd
import numpy as np
import xarray as xr
import os
from typing import List, Dict, Tuple
from .ib_connector import IBKRConnector

# [核心修改] 引入因子引擎配置和定义
from quant_core.factors.engine import FactorEngine

class LiveDataBridge:
    """
    数据桥接层 (Data Bridge) - 工程化重构版
    
    核心职责:
    1. 获取 IB 历史数据。
    2. 将 pandas 数据转换为 Xarray (模拟回测时的数据结构)。
    3. 调用 FactorEngine.FACTOR_REGISTRY 中的算子进行一致性计算。
    """

    def __init__(self, connector: IBKRConnector, universe_csv_path: str):
        self.connector = connector
        self.universe_map = self._load_universe(universe_csv_path)
        self.codes = list(self.universe_map.keys()) 
        
    def _load_universe(self, csv_path: str) -> Dict[str, str]:
        if not os.path.exists(csv_path):
            print(f"⚠️ 警告: 找不到标的文件 {csv_path}")
            return {}
        df = pd.read_csv(csv_path)
        mapping = {}
        for code in df['sec_code']:
            symbol = code.split('.')[0] 
            mapping[code] = symbol
        print(f"[{self.__class__.__name__}] 加载标的池: {len(mapping)} 个")
        return mapping

    def prepare_data_for_strategy(self, 
                                  required_factors: List[str], 
                                  lookback_window: int = 100, # 建议大一点，满足复杂因子的 warm-up
                                  bar_size: str = '1 day') -> Tuple[pd.DataFrame, pd.Series]:
        """
        核心: Fetch -> Convert to Xarray -> Compute using Factor Defs -> Return
        """
        print(f"\n⚡ [Live] 正在获取数据并计算因子 (Lookback: {lookback_window})...")
        
        # 1. 批量获取数据
        all_dfs = []
        current_prices = {}
        
        duration_str = f"{lookback_window} D" if 'day' in bar_size else "5 D"

        for full_code, symbol in self.universe_map.items():
            df = self.connector.get_historical_data(symbol, duration=duration_str, bar_size=bar_size)
            
            if df.empty:
                continue
                
            # 记录最新价格用于风控
            current_prices[full_code] = df['Close'].iloc[-1]
            
            # 为 DataFrame 增加 'sec_code' 列，方便后续合并转 Xarray
            df['sec_code'] = full_code
            
            # 确保索引名为 datetime
            df.index.name = 'datetime'
            df = df.reset_index() # 把 datetime 变成列
            
            all_dfs.append(df)
            
        if not all_dfs:
            return pd.DataFrame(), pd.Series()
            
        # 2. 合并并转换为 Xarray (模拟 FactorEngine._get_xarray_data 的逻辑)
        # 目标格式: (field, datetime, sec_code)
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # 标准化列名以匹配 definitions.py 的预期 (通常需要小写)
        combined_df.columns = [c.lower() for c in combined_df.columns]
        # 此时列应该有: datetime, open, high, low, close, volume, sec_code
        
        # 增加 amount, vwap 字段 (如果 IB 没给，先用近似值填充，防止因子报错)
        if 'amount' not in combined_df.columns:
            combined_df['amount'] = combined_df['close'] * combined_df['volume']
        if 'vwap' not in combined_df.columns:
            combined_df['vwap'] = combined_df['close'] # 降级处理
            
        # 转换为 Xarray DataArray
        try:
            # Pivot: Index=datetime, Columns=sec_code, Values=Fields
            # 这步稍微复杂，因为我们要转成 3D
            combined_df.set_index(['datetime', 'sec_code'], inplace=True)
            xr_ds = combined_df.to_xarray()
            xr_data = xr_ds.to_array(dim='field') # (field, datetime, sec_code)
        except Exception as e:
            print(f"❌ 数据转换 Xarray 失败: {e}")
            return pd.DataFrame(), pd.Series()

        # 3. 调用 FactorEngine 的逻辑进行计算
        factor_results = {}
        
        for factor_name in required_factors:
            # 从 Registry 获取类和参数
            config = FactorEngine.FACTOR_REGISTRY.get(factor_name)
            if not config:
                print(f"⚠️ 警告: 因子 {factor_name} 未在 Engine 注册，跳过。")
                continue
                
            factor_class, params = config
            
            try:
                # 实例化算子 (完全复用回测逻辑)
                factor_obj = factor_class(**params)
                
                # 计算 (传入 Xarray)
                # transform 返回的是一个 DataFrame (index=datetime, columns=sec_code)
                factor_values_df = factor_obj.transform(xr_data)
                
                # 4. 提取最新值 (Last Row)
                # 注意：回测时我们做 shift(1)，实盘时我们是在盘后或盘中运行，
                # 如果是盘中运行，transform 算出来的是截止当前时刻的值，直接取最后一行即可。
                # 如果你的策略是“基于昨天收盘价在今天开盘下单”，则需要 .iloc[-2]。
                # 这里假设是“基于最新数据产生的信号”，取 .iloc[-1]
                if not factor_values_df.empty:
                    latest_values = factor_values_df.iloc[-1]
                    factor_results[factor_name] = latest_values
                
            except Exception as e:
                print(f"❌ 计算因子 {factor_name} 出错: {e}")
                
        # 4. 组装结果 DataFrame
        # Index = sec_code, Columns = factor_names
        if factor_results:
            factor_df = pd.DataFrame(factor_results)
        else:
            factor_df = pd.DataFrame()
            
        return factor_df, pd.Series(current_prices)