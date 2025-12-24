# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Dict, Optional, List

# 1. 引入基类和装饰器
from .base import BaseStrategy, register_strategy

# 2. 使用装饰器自动注册
#    'linear' 对应 yaml 配置文件里的 type: 'linear'
@register_strategy('linear')
class LinearWeightedStrategy(BaseStrategy):
    """
    传统多因子线性加权策略 (Linear Weighted Multi-Factor)
    """

    def __init__(self, name: str, weights: Dict[str, float], top_k: int = 5, **kwargs):
        """
        Args:
            name: 策略名
            weights: 因子权重字典 (特有参数)
            top_k: 持仓数量 (通用参数)
            **kwargs: 接收工厂传来的其他参数 (如 stop_loss_pct, max_pos_weight 等风控参数)
        """
        # 3. 必须将 kwargs 传给父类，因为风控参数都在里面
        super().__init__(name, top_k=top_k, **kwargs)
        
        self.weights = weights
        
        # 归一化权重
        total_w = sum(abs(v) for v in self.weights.values())
        if total_w != 0:
            self.weights = {k: v / total_w for k, v in self.weights.items()}
            
        print(f"[{self.name}] 因子权重配置: {self.weights}")

    def get_required_factors(self) -> List[str]:
        """
        返回权重字典的 keys，告诉系统我需要哪些因子
        """
        return list(self.weights.keys())

    def calculate_scores(self, factor_df: pd.DataFrame) -> pd.Series:
        """
        计算打分逻辑
        """
        # 初始化 0 分
        final_scores = pd.Series(0.0, index=factor_df.index)
        
        for factor_name, weight in self.weights.items():
            if factor_name not in factor_df.columns:
                continue
            
            raw_values = factor_df[factor_name]
            
            # Z-Score 标准化
            if raw_values.std() != 0:
                standardized_values = (raw_values - raw_values.mean()) / raw_values.std()
            else:
                standardized_values = raw_values - raw_values.mean()
            
            standardized_values = standardized_values.fillna(0)
            
            final_scores += standardized_values * weight
            
        return final_scores