# quant_core/strategies/ml_strategy.py
from typing import List
import pandas as pd
from .base import BaseStrategy, register_strategy # 引入装饰器

# 1. 注册 (对应 yaml 里的 type: 'ml')
@register_strategy('ml')
class XGBoostStrategy(BaseStrategy):
    def __init__(self, name, model_path, feature_list, top_k=5, **kwargs):
        # 2. 接收 kwargs 并传给基类 (处理风控参数)
        super().__init__(name, top_k=top_k, **kwargs)
        
        self.model_path = model_path
        self.feature_list = feature_list
        # self.model = load_model(model_path) # 伪代码
        print(f"[{name}] ML模型已加载: {model_path}")

    # 3. 声明需要的因子
    def get_required_factors(self) -> List[str]:
        return self.feature_list

    def calculate_scores(self, factor_df: pd.DataFrame) -> pd.Series:
        # 这里写你的模型预测逻辑
        # ...
        return pd.Series()