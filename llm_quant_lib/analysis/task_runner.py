# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from .preprocessor import FactorPreprocessor
from .research_engine import FactorResearchEngine
from ..factor_engine import FactorEngine

class FactorTaskRunner:
    """
    Orchestrator for Factor EDA passing horizon parameters to the engine.
    """
    
    def __init__(self, data_handler):
        self.factor_engine = FactorEngine(data_handler)
        self.preprocessor = FactorPreprocessor()

    def _get_cleaned_returns(self, horizon=1):
        """
        Prepare forward returns with cleaning and clipping.
        """
        df = self.factor_engine.data_handler.load_data()
        df = df.sort_values(['sec_code', 'datetime'])
        
        # 计算预测周期的收益率
        df['ret_nd'] = df.groupby('sec_code')['close'].shift(-horizon) / df['close'] - 1
        
        # 极值裁剪，防止极端坏点
        df['ret_nd'] = df['ret_nd'].clip(-0.5, 0.5)
        
        return df[['datetime', 'sec_code', 'ret_nd']]

    def run_analysis_pipeline(self, factor_name, horizon=1, n_groups=5):
        print(f"🚀 Running Factor EDA: {factor_name} (Horizon: {horizon}D)")

        # 1. 计算因子
        factor_df_wide = self.factor_engine._compute_and_cache_factor(factor_name)
        factor_df = factor_df_wide.stack(future_stack=True).reset_index()
        factor_df.columns = ['datetime', 'sec_code', 'factor_value']

        # 2. 获取收益率
        returns_df = self._get_cleaned_returns(horizon=horizon)

        # 3. 对齐
        merged_df = pd.merge(factor_df, returns_df, on=['datetime', 'sec_code'], how='inner')
        merged_df = merged_df.dropna(subset=['factor_value', 'ret_nd'])

        # 4. 预处理
        def clean_daily(group):
            group['factor_value'] = self.preprocessor.handle_outliers(group['factor_value'])
            group['factor_value'] = self.preprocessor.standardize(group['factor_value'])
            return group
        
        cleaned_df = merged_df.groupby('datetime', group_keys=False).apply(clean_daily)

        # 5. 指标计算：传入 horizon 以调整复利逻辑
        res_engine = FactorResearchEngine(cleaned_df)
        ic_series = res_engine.calculate_ic(target_col='ret_nd', method='rank')
        stats = res_engine.calculate_stats(ic_series)
        
        # 这里关键：传入 horizon，让引擎知道如何处理累积收益
        _, cum_group_ret, _ = res_engine.calculate_group_returns(target_col='ret_nd', n_groups=n_groups, horizon=horizon)

        return stats, ic_series, cum_group_ret