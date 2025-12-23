# quant_core/strategies/ml_strategy.py
from .base import BaseStrategy
import pandas as pd
import joblib  # 用于加载模型

class XGBoostStrategy(BaseStrategy):
    def __init__(self, model_path: str, feature_names: list, top_k=5):
        super().__init__("XGBoost_V1", top_k)
        self.model = joblib.load(model_path) # 加载预训练好的模型
        self.feature_names = feature_names   # 确保预测时的特征顺序与训练时一致

    def calculate_scores(self, factor_df: pd.DataFrame) -> pd.Series:
        # 1. 检查特征列是否齐全
        missing_cols = set(self.feature_names) - set(factor_df.columns)
        if missing_cols:
            raise ValueError(f"Missing features: {missing_cols}")

        # 2. 准备 X (特征矩阵)
        X = factor_df[self.feature_names]
        
        # 3. 处理缺失值 (ML模型通常不喜欢NaN，这里简单填充，实际需更复杂处理)
        X = X.fillna(0) 

        # 4. 模型预测 (Predict Proba 或 Predict Score)
        # XGBoost 输出的是预测的收益率或上涨概率
        pred_scores = self.model.predict(X)

        return pd.Series(pred_scores, index=factor_df.index)