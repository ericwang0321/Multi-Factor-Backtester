# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

class FactorResearchEngine:
    """
    Core engine for calculating factor performance metrics: IC, Rank IC, IR, and Group Returns.
    """
    
    def __init__(self, factor_data):
        """
        Initialize with a combined dataset.
        :param factor_data: DataFrame with columns ['datetime', 'sec_code', 'factor_value', 'forward_return'].
        """
        self.df = factor_data.copy()

    def calculate_ic(self, method='rank'):
        """
        Calculate the time series of Information Coefficient (IC).
        :param method: 'rank' for Spearman Correlation (Rank IC), 'normal' for Pearson Correlation.
        """
        def calc_daily_ic(group):
            # 计算截面相关性：今天的因子值 vs 未来的收益率
            if method == 'rank':
                return group['factor_value'].corr(group['forward_return'], method='spearman')
            return group['factor_value'].corr(group['forward_return'], method='pearson')

        # 按天分组计算截面 IC
        ic_series = self.df.groupby('datetime').apply(calc_daily_ic)
        return ic_series

    def calculate_stats(self, ic_series):
        """
        Calculate summary statistics for the factor.
        Includes Mean IC, IC Volatility, and Information Ratio (IR).
        """
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        # IR 衡量因子提供超额收益的稳定性
        ir = ic_mean / ic_std if ic_std != 0 and not np.isnan(ic_std) else 0
        
        return {
            'IC Mean': ic_mean,
            'IC Std': ic_std,
            'IR': ir,
            'IC > 0 Rate': (ic_series > 0).mean(),
            'IC Absolute Mean': ic_series.abs().mean()
        }

    def calculate_group_returns(self, n_groups=5):
        """
        Perform Quantile Analysis by splitting assets into N groups based on factor values.
        Calculates the mean forward return for each group to check for monotonicity.
        """
        # 每天根据因子值将资产分为 N 组
        # 0 组为因子值最低，N-1 组为因子值最高
        self.df['group'] = self.df.groupby('datetime')['factor_value'].transform(
            lambda x: pd.qcut(x, n_groups, labels=False, duplicates='drop')
        )
        
        # 计算每组每日的平均未来收益率
        group_daily_ret = self.df.groupby(['datetime', 'group'])['forward_return'].mean().unstack()
        
        # 计算累积收益率曲线，用于可视化
        cum_group_ret = (1 + group_daily_ret.fillna(0)).cumprod()
        
        # 计算多空收益 (最高组 - 最低组)
        last_idx = group_daily_ret.columns[-1]
        first_idx = group_daily_ret.columns[0]
        ls_daily_ret = group_daily_ret[last_idx] - group_daily_ret[first_idx]
        cum_ls_ret = (1 + ls_daily_ret.fillna(0)).cumprod()
        
        return group_daily_ret, cum_group_ret, cum_ls_ret