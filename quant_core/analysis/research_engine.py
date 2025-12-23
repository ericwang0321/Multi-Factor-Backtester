# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

class FactorResearchEngine:
    """
    Engine for factor metrics with dynamic horizon support for wealth curves.
    """
    
    def __init__(self, factor_data):
        self.df = factor_data.copy()

    def calculate_ic(self, target_col='ret_nd', method='rank'):
        """
        Calculate IC based on the selected horizon return.
        """
        def calc_daily_ic(group):
            if method == 'rank':
                return group['factor_value'].corr(group[target_col], method='spearman')
            return group['factor_value'].corr(group[target_col], method='pearson')

        ic_series = self.df.groupby('datetime').apply(calc_daily_ic)
        return ic_series

    def calculate_stats(self, ic_series):
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ir = ic_mean / ic_std if ic_std != 0 and not np.isnan(ic_std) else 0
        return {
            'IC Mean': ic_mean, 'IC Std': ic_std, 'IR': ir,
            'IC > 0 Rate': (ic_series > 0).mean(),
            'IC Absolute Mean': ic_series.abs().mean()
        }

    def calculate_group_returns(self, target_col='ret_nd', n_groups=5, horizon=1):
        """
        Quantile analysis that adjusts the wealth curve based on horizon N.
        """
        # 1. 每日分组
        self.df['group'] = self.df.groupby('datetime')['factor_value'].transform(
            lambda x: pd.qcut(x, n_groups, labels=False, duplicates='drop')
        )
        
        # 2. 计算各组平均收益率 (这是 N 日收益率)
        group_ret_nd = self.df.groupby(['datetime', 'group'])[target_col].mean().unstack()
        
        # 3. [核心修正] 动态 Horizon 复利逻辑
        # 如果 horizon > 1，我们不能直接 compound(ret_nd)，否则会爆炸。
        # 正确做法：将 N 日收益转为等效 1 日收益 r_daily = (1 + ret_nd)^(1/N) - 1
        if horizon > 1:
            # 使用几何平均还原每日收益，确保 Y 轴尺度正确且反映 N 日预测力
            group_daily_equivalent = (1 + group_ret_nd.fillna(0)).pow(1/horizon) - 1
            cum_group_ret = (1 + group_daily_equivalent).cumprod()
        else:
            cum_group_ret = (1 + group_ret_nd.fillna(0)).cumprod()
        
        # 4. 计算多空净值 (Top - Bottom)
        ls_ret_nd = group_ret_nd[group_ret_nd.columns[-1]] - group_ret_nd[group_ret_nd.columns[0]]
        if horizon > 1:
            ls_daily_equivalent = (1 + ls_ret_nd.fillna(0)).pow(1/horizon) - 1
            cum_ls_ret = (1 + ls_daily_equivalent).cumprod()
        else:
            cum_ls_ret = (1 + ls_ret_nd.fillna(0)).cumprod()
            
        return group_ret_nd, cum_group_ret, cum_ls_ret