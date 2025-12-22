# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class FactorPreprocessor:
    """
    Modular engine for factor data cleaning, including winsorization and standardization.
    """
    
    @staticmethod
    def fill_missing(series, method='median'):
        """
        Fill NaN values in a factor series.
        :param series: pd.Series of factor values for a specific timestamp.
        :param method: 'median' or 'mean'.
        """
        if method == 'median':
            fill_val = series.median()
        else:
            fill_val = series.mean()
        return series.fillna(fill_val)

    @staticmethod
    def handle_outliers(series, method='mad', n=3):
        """
        De-extremum (Winsorization) to remove extreme values.
        :param method: 'mad' (Median Absolute Deviation) or 'sigma' (Standard Deviation).
        :param n: Threshold multiplier (usually 3).
        """
        if series.dropna().empty:
            return series
            
        if method == 'mad':
            # MAD 逻辑比标准差更稳健，不易受极端值干扰
            median = series.median()
            mad = (series - median).abs().median()
            # 1.4826 是正态分布下的缩放因子
            threshold = 1.4826 * mad * n
            return series.clip(lower=median - threshold, upper=median + threshold)
        elif method == 'sigma':
            mean = series.mean()
            std = series.std()
            return series.clip(lower=mean - n * std, upper=mean + n * std)
        return series

    @staticmethod
    def standardize(series):
        """
        Z-Score normalization to make factors comparable.
        Result: Mean = 0, Std = 1.
        """
        std = series.std()
        if std == 0 or np.isnan(std):
            return series - series.mean()
        return (series - series.mean()) / std

    @staticmethod
    def neutralize(df, factor_col, target_cols=['market_cap']):
        """
        Factor Neutralization: Remove unwanted exposure (e.g., Size bias).
        Returns the residuals of the linear regression.
        """
        # 准备回归数据，确保没有空值
        temp_df = df[[factor_col] + target_cols].dropna()
        if temp_df.empty:
            return df[factor_col]
            
        X = temp_df[target_cols]
        y = temp_df[factor_col]
        
        # 使用 OLS 回归提取残差，残差即为“纯净”的因子
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        
        residuals = y - y_pred
        # 重新对齐到原始索引
        return residuals.reindex(df.index)

    def process_cross_section(self, df, factor_col, winsorize=True, standardize=True):
        """
        Apply full cleaning pipeline to a cross-sectional DataFrame.
        """
        s = df[factor_col].copy()
        
        # 1. 填充缺失值
        s = self.fill_missing(s)
        
        # 2. 去极值
        if winsorize:
            s = self.handle_outliers(s)
            
        # 3. 标准化
        if standardize:
            s = self.standardize(s)
            
        return s