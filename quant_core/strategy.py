# -*- coding: utf-8 -*-
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List

class BaseStrategy(ABC):
    def __init__(self, universe_df: pd.DataFrame, **kwargs):
        self.universe_df = universe_df
        self.params = kwargs

    @abstractmethod
    def get_target_weights(self, current_date, factor_snapshot, portfolio_state) -> Dict[str, float]:
        pass

    @abstractmethod
    def get_required_factors(self) -> List[str]:
        return []

class FactorTopNStrategy(BaseStrategy):
    """加权多因子策略"""
    def __init__(self, universe_df: pd.DataFrame, **kwargs):
        super().__init__(universe_df, **kwargs)
        # 核心修改：使用 factor_weights 接收权重字典
        self.factor_weights = kwargs.get('factor_weights', {}) 
        self.top_n = kwargs.get('top_n', 5)
        self.ascending = kwargs.get('ascending', False)
        self.universe_to_trade = kwargs.get('universe_to_trade', 'All')

        if not self.factor_weights:
            raise ValueError("FactorTopNStrategy: 'factor_weights' dictionary cannot be empty.")
        self.trade_log = []

    def get_required_factors(self) -> List[str]:
        return list(self.factor_weights.keys())

    def get_target_weights(self, current_date, factor_snapshot, portfolio_state) -> Dict[str, float]:
        target_col = 'composite_score' if len(self.factor_weights) > 1 else list(self.factor_weights.keys())[0]
        if target_col not in factor_snapshot.columns: return {}

        # 筛选资产池
        assets = factor_snapshot.index
        if self.universe_to_trade.lower() != 'all':
            assets = self.universe_df[self.universe_df['universe'] == self.universe_to_trade]['sec_code']
        
        scores = factor_snapshot.loc[factor_snapshot.index.isin(assets), target_col].dropna()
        top_assets = scores.sort_values(ascending=self.ascending).head(self.top_n)
        
        weights = {code: 1.0/len(top_assets) for code in top_assets.index} if not top_assets.empty else {}
        self.trade_log.append({"date": current_date, "final_weights": weights})
        return weights

    def get_trade_log(self):
        return pd.DataFrame(self.trade_log)