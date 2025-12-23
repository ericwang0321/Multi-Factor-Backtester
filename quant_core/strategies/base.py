# quant_core/strategies/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

class BaseStrategy(ABC):
    """
    策略基类 (Abstract Base Class) - V2 (支持自定义权重)
    
    流水线:
    1. get_day_factors: 获取当日因子
    2. calculate_scores: 计算打分 (由子类实现，如 Linear 或 XGBoost)
    3. select_top_n: 根据分数筛选股票
    4. calculate_weights: 计算权重 (默认等权，子类可重写为 MinVar/RiskParity)
    """
    
    def __init__(self, name: str, top_k: int = 5):
        self.name = name
        self.top_k = top_k
        self.factor_data: Optional[pd.DataFrame] = None
        self.price_data: Optional[pd.DataFrame] = None # 预留给 MinVar 计算协方差用
        print(f"[{self.name}] 策略初始化完成。Target Top-K: {self.top_k}")

    def load_data(self, factor_df: pd.DataFrame, price_df: Optional[pd.DataFrame] = None):
        """注入数据 (因子 + 可选的价格数据)"""
        self.factor_data = factor_df
        if price_df is not None:
            self.price_data = price_df
        print(f"[{self.name}] 数据加载完成。")

    def get_day_factors(self, date, universe_codes: List[str]) -> pd.DataFrame:
        """获取当日因子切片"""
        if self.factor_data is None: return pd.DataFrame()
        if date not in self.factor_data.index.get_level_values(0): return pd.DataFrame()
        
        try:
            day_df = self.factor_data.loc[date]
            valid_codes = [c for c in universe_codes if c in day_df.index]
            return day_df.loc[valid_codes]
        except KeyError:
            return pd.DataFrame()

    @abstractmethod
    def calculate_scores(self, factor_df: pd.DataFrame) -> pd.Series:
        """【抽象方法】计算打分 (Step 1)"""
        pass

    def calculate_weights(self, selected_codes: List[str], date) -> Dict[str, float]:
        """
        【虚方法】计算权重 (Step 2)
        
        默认实现: 等权重 (1/N)
        扩展方向: 你可以在子类重写此方法，利用 self.price_data 计算协方差矩阵，
                 实现 MinVar, Mean-Variance 或 Risk Parity。
        """
        if not selected_codes:
            return {}
            
        # 默认：等权重
        w = 1.0 / len(selected_codes)
        return {code: w for code in selected_codes}

    def on_bar(self, date, universe_codes: List[str]) -> Dict[str, float]:
        """标准执行流水线"""
        # 1. 准备数据
        factors_df = self.get_day_factors(date, universe_codes)
        if factors_df.empty: return {}

        # 2. 算分 (Scoring)
        scores = self.calculate_scores(factors_df).dropna()
        if scores.empty: return {}

        # 3. 选股 (Selection)
        k = min(self.top_k, len(scores))
        selected_codes = scores.nlargest(k).index.tolist()

        # 4. 定权 (Weighting) - 这里调用新拆分出的方法
        target_positions = self.calculate_weights(selected_codes, date)
        
        return target_positions