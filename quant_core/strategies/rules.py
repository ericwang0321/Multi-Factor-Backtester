# quant_core/strategies/rules.py
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from .base import BaseStrategy

class LinearWeightedStrategy(BaseStrategy):
    """
    传统多因子线性加权策略 (Linear Weighted Multi-Factor)
    
    逻辑:
    1. 声明所需因子 (get_required_factors)。
    2. 获取当日因子值。
    3. Z-Score 标准化 (去量纲)。
    4. 按权重加权求和得到总分。
    5. 选 Top-K。
    """

    def __init__(self, name: str, weights: Dict[str, float], top_k: int = 5,
                 stop_loss_pct: Optional[float] = None,
                 max_pos_weight: Optional[float] = None,
                 max_drawdown_pct: Optional[float] = None):
        """
        Args:
            name: 策略名
            weights: 因子权重字典, e.g., {'RSI': 0.4, 'Momentum': 0.6}
            top_k: 持仓数量
            stop_loss_pct: 止损比例
            max_pos_weight: 单票限仓
            max_drawdown_pct: 熔断比例
        """
        # 将风控参数传递给父类 BaseStrategy
        super().__init__(name, top_k, stop_loss_pct, max_pos_weight, max_drawdown_pct)
        
        self.weights = weights
        
        # 归一化权重 (确保权重和为 1，虽然不强制，但为了规范)
        total_w = sum(abs(v) for v in self.weights.values()) # 使用绝对值求和
        if total_w != 0:
            # 保持原始符号，只缩放大小
            self.weights = {k: v / total_w for k, v in self.weights.items()}
            
        print(f"[{self.name}] 因子权重配置: {self.weights}")

    # =========================================================================
    # [新增] 实现基类的抽象方法：告诉主脚本我需要哪些因子
    # =========================================================================
    def get_required_factors(self) -> List[str]:
        """
        返回权重字典的 keys，即策略需要的因子列表。
        例如: ['alpha013', 'rsi']
        """
        return list(self.weights.keys())

    def calculate_scores(self, factor_df: pd.DataFrame) -> pd.Series:
        """
        实现基类的 calculate_scores 接口
        """
        # 1. 初始化总分 Series
        # 使用 factor_df 的 index (股票代码) 初始化 0 分
        final_scores = pd.Series(0.0, index=factor_df.index)
        
        # 2. 遍历每个因子进行加权
        for factor_name, weight in self.weights.items():
            if factor_name not in factor_df.columns:
                # 如果数据里没有这个因子，跳过（理论上 Bridge 已经准备好了，但为了稳健）
                continue
            
            # 获取原始因子值
            raw_values = factor_df[factor_name]
            
            # --- 关键步骤：Z-Score 标准化 ---
            # 目的：将 RSI(50) 和 Return(0.01) 拉到同一个水平线上比较
            # 公式：(x - mean) / std
            if raw_values.std() != 0:
                standardized_values = (raw_values - raw_values.mean()) / raw_values.std()
            else:
                standardized_values = raw_values - raw_values.mean()
            
            # 缺失值填充为 0 (代表平均水平，不加分也不减分)
            standardized_values = standardized_values.fillna(0)
            
            # 3. 累加得分
            final_scores += standardized_values * weight
            
        return final_scores