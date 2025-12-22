# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class BaseStrategy(ABC):
    """策略基类 (接口定义)"""
    def __init__(self, universe_df: pd.DataFrame, **kwargs):
        if 'sec_code' not in universe_df.columns or 'universe' not in universe_df.columns:
            raise ValueError("universe_df 必须包含 'sec_code' 和 'universe' 列。")
        self.universe_df = universe_df
        self.params = kwargs

    @abstractmethod
    def get_target_weights(self, current_date: pd.Timestamp, factor_snapshot: pd.DataFrame, portfolio_state: dict) -> Dict[str, float]:
        """核心决策接口"""
        pass

    @abstractmethod
    def get_required_factors(self) -> List[str]:
        """返回策略需要的因子列表"""
        return []

class FactorTopNStrategy(BaseStrategy):
    """
    【经典因子选股策略】
    逻辑：在指定的资产池内，根据单一因子值进行截面排序，选择前 N 名等权重持仓。
    """
    def __init__(self, universe_df: pd.DataFrame, **kwargs):
        super().__init__(universe_df, **kwargs)
        # 从配置中读取参数
        self.factor_name = kwargs.get('factor_name')        # 使用哪个因子
        self.top_n = kwargs.get('top_n', 5)                # 选多少只
        self.ascending = kwargs.get('ascending', False)    # 默认 False 表示因子值越大越好
        self.universe_to_trade = kwargs.get('universe_to_trade', 'All')

        if not self.factor_name:
            raise ValueError("FactorTopNStrategy: 必须在配置中指定 'factor_name'。")
        
        self.trade_log: List[Dict] = []

    def get_required_factors(self) -> List[str]:
        """告诉引擎需要计算哪个因子"""
        return [self.factor_name]

class FactorTopNStrategy(BaseStrategy):
    """
    [Multi-Factor Top-N Strategy]
    Logic: Ranks assets based on a composite score derived from multiple factors.
    """
    def __init__(self, universe_df: pd.DataFrame, **kwargs):
        super().__init__(universe_df, **kwargs)
        # NEW: Accepts a list of factors instead of a single string
        self.factor_names = kwargs.get('factor_names', []) 
        self.top_n = kwargs.get('top_n', 5)
        self.ascending = kwargs.get('ascending', False)
        self.universe_to_trade = kwargs.get('universe_to_trade', 'All')

        if not self.factor_names:
            raise ValueError("FactorTopNStrategy: 'factor_names' list cannot be empty.")
        
        self.trade_log: List[Dict] = []

    def get_required_factors(self) -> List[str]:
        """Returns the list of all factors needed for the composite score."""
        return self.factor_names

    def get_target_weights(self, current_date: pd.Timestamp, factor_snapshot: pd.DataFrame, portfolio_state: dict) -> Dict[str, float]:
        """
        Ranks assets based on the 'composite_score' provided by the FactorEngine.
        """
        if self.universe_to_trade.lower() == 'all':
            assets_in_scope = factor_snapshot.index.tolist()
        else:
            assets_in_scope = self.universe_df[self.universe_df['universe'] == self.universe_to_trade]['sec_code'].tolist()
        
        # Use 'composite_score' if multiple factors exist, else use the single factor name
        target_col = 'composite_score' if len(self.factor_names) > 1 else self.factor_names[0]

        if target_col not in factor_snapshot.columns:
            return {}

        relevant_scores = factor_snapshot.loc[factor_snapshot.index.isin(assets_in_scope), target_col]
        relevant_scores = relevant_scores.dropna()

        if relevant_scores.empty:
            return {}

        # Sort and select Top N
        top_assets = relevant_scores.sort_values(ascending=self.ascending).head(self.top_n)
        
        if not top_assets.empty:
            weight = 1.0 / len(top_assets)
            target_weights = {code: weight for code in top_assets.index}
        else:
            target_weights = {}

        self.trade_log.append({
            "date": current_date,
            "justification": f"Selected Top {len(target_weights)} assets using {target_col}.",
            "selected_assets": list(target_weights.keys()),
            "final_weights": target_weights
        })

        return target_weights

    def get_trade_log(self) -> pd.DataFrame:
        """返回决策日志"""
        return pd.DataFrame(self.trade_log)
