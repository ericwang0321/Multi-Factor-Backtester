# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xarray as xr
import bottleneck as bl
import statsmodels.api as sm
from typing import Dict, List, Optional, Any
from collections import defaultdict

# 从同级目录导入数据处理器和新的因子定义文件
from .data_handler import DataHandler
from .factor_definitions import * # 导入所有 BaseAlpha 因子类和辅助函数

class FactorEngine:
    """
    【升级版】因子引擎。
    使用 factor_definitions.py 中的 BaseAlpha 类和 xarray 进行计算。
    实现了全时间序列缓存，以提高回测速度。
    """
    
    def __init__(self, data_handler: DataHandler):
        """
        初始化因子引擎。

        Args:
            data_handler (DataHandler): 已初始化的数据处理器实例。
        """
        print("FactorEngine: 正在初始化 (使用 BaseAlpha/Xarray 引擎)...")
        self.data_handler = data_handler
        
        # 缓存
        self._factor_cache: Dict[str, pd.DataFrame] = {} # 缓存已计算的因子全序列 (DataFrame)
        self.xarray_data: Optional[xr.DataArray] = None # 缓存转换后的 Xarray 原始数据
        
        # --- 因子注册表 ---
        # 从 prepare_factor_data.py 复制而来
        # Key: 因子名称 (与 config.yaml 对应)
        # Value: (因子计算类, {参数字典})
        self.FACTOR_REGISTRY: Dict[str, Any] = {
            # --- 您新文件中的所有因子定义 ---
            'alpha006': (Alpha006, {'corr_period': 15}),
            'alpha007': (Alpha007, {'delta_period': 3, 'vol_mean_period': 10, 'rank_period': 50}),
            'alpha013': (Alpha013, {'cov_period': 5}),
            'alpha018': (Alpha018, {'std_period': 5, 'corr_period': 12}),
            'alpha023': (Alpha023, {'ts_period': 30, 'delta_period': 3}),
            'alpha033': (Alpha033, {}),
            'alpha040': (Alpha040, {'ts_period': 8}),
            'alpha041': (Alpha041, {}),
            'alpha042': (Alpha042, {}),
            'alpha053': (Alpha053, {'ts_period': 15}),
            'alpha060': (Alpha060, {'ts_period': 25}),
            'alpha191_014': (Alpha191_014, {'delay_period': 3}),
            'alpha191_018': (Alpha191_018, {'delay_period': 3}),
            'alpha191_020': (Alpha191_020, {'delay_period': 3}),
            'amihud_illiquidity': (AmihudIlliquidity, {'window': 126}),
            'amount_mean': (AmountMean, {'window': 252}),
            'amount_std': (AmountStd, {'window': 42}),
            'amount_volatility': (AmountVolatility, {'window': 63}),
            'cmo': (CMO, {'window': 40}),
            'dv_divergence': (DV_Divergence, {'price_window': 15, 'corr_window': 20}),
            'eric_adx_weighted_momentum': (Eric_ADX_Weighted_Momentum, {'momentum_window': 22, 'adx_window': 25}),
            'eric_asymmetric_vol_reversion': (Eric_Asymmetric_Volatility_Reversion, {'window': 60}),
            'eric_composite_mom_bbw': (Eric_Composite_Momentum_BBW, {'momentum_window': 11, 'adx_window': 14, 'bb_window': 80, 'mom_weight': 0.2}),
            'eric_cs_resid_vol': (Eric_CS_Residual_Volatility, {'window': 100}),
            'eric_enhanced_momentum': (EricEnhancedMomentum, {}),
            'eric_improved_alpha_022': (Eric_Improved_Alpha_022, {}),
            'eric_multi_dim_reversal': (EricMultiDimReversal, {'window': 120}),
            'eric_multi_dim_reversal_v2': (EricMultiDimReversalV2, {'short_window': 20, 'long_window': 120}),
            'eric_pv_corr': (EricFactor, {'window': 60}),
            'eric_pv_divergence': (Eric_PV_Divergence_Covariance, {'window': 10}),
            'eric_tail_skew': (Eric_Improved_Skewness, {'window': 84, 'std_multiple': 1.0}),
            'eric_u_shape_vol': (Eric_U_Shape_Volatility_Factor, {'window': 30}),
            'eric_vol_turnover_coupling': (Eric_Volatility_Turnover_Coupling, {'window': 10}),
            'falkenblog_disparity_mean': (Falkenblog_Disparity_Mean, {'window': 40}),
            'falkenblog_disparity_std': (Falkenblog_Disparity_Std, {'window': 40}),
            'gp_new_alpha_1': (GP_New_Alpha_1, {}),
            'gp_new_alpha_2': (GP_New_Alpha_2, {}),
            'kurtosis': (HT_Kurtosis, {'window': 60}),
            'logcap': (LogCap, {}),
            'momentum': (Momentum, {'window': 168}),
            'momentum_acceleration': (MomentumAcceleration, {'period1': 42, 'period2': 126}),
            'momentum_daily_mean': (HT_Momentum_DailyMean, {'window': 42}),
            'momentum_exp_weighted': (HT_Momentum_ExpWeighted, {'window': 189}),
            'momentum_turnover_weighted': (HT_Momentum_TurnoverWeighted, {'window': 33}),
            'monthly_avg_overnight_return': (Eric_Monthly_Avg_Overnight_Return, {'window': 55}),
            'monthly_std_overnight_return': (Eric_Monthly_Std_Overnight_Return, {'window': 30}),
            'rsi': (RSI, {'window': 12}),
            'skewness': (HT_Skewness, {'window': 42}),
            'stochastic_k': (StochasticK, {'window': 40}),
            'term_structure_proxy': (HT_TermStructure_RollYield_Proxy, {'window': 30}),
            'turnover_bias': (HT_Turnover_Bias, {'short_window': 66, 'long_window': 378}),
            'turnover_mean': (TurnoverMean, {'window': 10}),
            'turnover_std': (TurnoverStd, {'window': 63}),
            'turnover_std_bias': (HT_Turnover_Std_Bias, {'short_window': 66, 'long_window': 126}),
            'volatility': (HT_Volatility, {'window': 30}),
            'volume_mean': (VolumeMean, {'window': 10}),
            'volume_std': (VolumeStd, {'window': 30}),
            'trend_score': (TrendScore, {'window': 30}),
            'eric_asymmetric_vol_momentum': (HT_Asymmetric_Volatility_Momentum, {'mom_window': 90, 'vol_window': 90, 'delta_window': 20}),
            'ht_low_vol_breakout': (HT_Low_Volatility_Breakout, {'mom_window': 90, 'vol_window': 90}), 
            'ht_trend_exhaustion_reversal': (HT_Trend_Exhaustion_Reversal, {'window': 90}), # 修正了您新文件中的参数不匹配问题
            'ht_skewness_reversal': (HT_Skewness_Reversal, {'window': 120}),
            'ht_ts_vol_enhanced_mom': (HT_TS_Vol_Enhanced_Momentum, {'ts_window': 30, 'vol_window': 30, 'vol_rank_threshold': 0.5}),
            'probabilistic_trend_rsrs': (Probabilistic_Trend_RSRS, {
                'rsrs_window': 22, 'adx_window': 14, 'logistic_midpoint': 25.0, 'logistic_steepness': 0.2
            }),
            'quality_momentum_r2': (Quality_Momentum_R2, {
                'momentum_window': 126, 'regression_window': 126
            }),
            'smart_momentum_adx_r2': (Smart_Momentum_ADX_R2, {
                'window': 126, 'adx_window': 14
            }),
            'breakout_quality_score': (Breakout_Quality_Score, {
                'mom_window': 22, 'short_turn_window': 22, 'long_turn_window': 126, 'vol_window': 63
            }),
            'trend_certainty_score': (Trend_Certainty_Score, {
                'window': 63, 'adx_window': 14, 'adx_threshold': 25.0
            }),
        }
        print(f"FactorEngine: 初始化完成。已注册 {len(self.FACTOR_REGISTRY)} 个因子。")

    def _get_xarray_data(self) -> xr.DataArray:
        """
        (私有) 辅助函数：从 DataHandler 加载所有原始数据并转换为 Xarray 格式。
        只在第一次需要时执行一次。
        """
        if self.xarray_data is not None:
            return self.xarray_data

        print("FactorEngine: 首次计算，正在从 DataHandler 加载并转换所有数据到 Xarray...")
        
        # 1. 从 DataHandler 获取长格式的原始 DataFrame (包含 buffer)
        raw_df = self.data_handler.load_data()
        if raw_df is None or raw_df.empty:
            raise ValueError("FactorEngine: DataHandler 未能提供有效的原始数据。")

        # 2. 确保 vwap 存在 (您的因子库需要)
        if 'avg_price' in raw_df.columns: 
            raw_df = raw_df.rename(columns={'avg_price': 'vwap'})
        
        # 3. 筛选所需列并转换为宽格式
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turnover', 'market_cap', 'vwap']
        existing_cols = [col for col in required_cols if col in raw_df.columns]
        
        # 使用 pivot_table 处理重复项 (更健壮)
        wide_data_list = []
        for col in existing_cols:
            try:
                # 使用 pivot_table (默认 aggfunc='mean') 来处理 (datetime, sec_code) 重复
                wide_col = pd.pivot_table(raw_df, index='datetime', columns='sec_code', values=col)
                wide_data_list.append(wide_col)
            except Exception as e:
                print(f"FactorEngine: 转换列 {col} 为宽格式时出错: {e}")
                
        if not wide_data_list:
             raise ValueError("FactorEngine: 无法从原始数据创建任何宽格式列。")
             
        wide_data = pd.concat(wide_data_list, axis=1, keys=existing_cols)
        
        # 4. 填充缺失值 (因子库需要)
        wide_data = wide_data.interpolate(method='linear', limit_direction='both', axis=0)
        
        # 5. 转换为 Xarray (使用您新文件中的辅助函数)
        # 我们需要先定义这个辅助函数
        def convert_wide_df_to_xarray(wide_df: pd.DataFrame) -> xr.DataArray:
            """将多重索引的宽格式 DataFrame 转换为 xarray.DataArray"""
            if not isinstance(wide_df.columns, pd.MultiIndex):
                raise ValueError("输入必须是 MultiIndex DataFrame")
            stacked = wide_df.stack(level='sec_code', future_stack=True)
            stacked.index.names = ['datetime', 'sec_code']
            stacked.columns.name = 'field'
            return stacked.to_xarray().to_array('field')

        try:
            self.xarray_data = convert_wide_df_to_xarray(wide_data)
            print("FactorEngine: Xarray 数据转换并缓存成功。")
            return self.xarray_data
        except Exception as e:
            raise RuntimeError(f"FactorEngine: 转换宽 DataFrame 到 Xarray 时失败: {e}")

    def _compute_and_cache_factor(self, factor_name: str) -> pd.DataFrame:
        """
        (私有) 辅助函数：计算单个因子的全时间序列并将其存入缓存。
        """
        if factor_name not in self.FACTOR_REGISTRY:
            raise ValueError(f"FactorEngine: 因子 '{factor_name}' 未在 FACTOR_REGISTRY 中定义。")
            
        print(f"FactorEngine: 正在计算因子 '{factor_name}' 的全时间序列 (首次)...")
        
        try:
            # 1. 获取 xarray 格式的原始数据
            market_array = self._get_xarray_data()
            
            # 2. 从注册表获取因子类和参数
            factor_class, params = self.FACTOR_REGISTRY[factor_name]
            
            # 3. 实例化并运行计算
            factor_obj = factor_class(**params)
            # .transform() 方法返回一个 (datetime x sec_code) 的 DataFrame
            factor_df = factor_obj.transform(market_array) 
            
            # 4. 因子值滞后一期 (!!!!!!)
            # 这是为了确保在 decision_date (t-1) 使用的是 t-1 的可用数据
            # 您的新因子库在 `main` 函数中执行了 shift(1)，我们在这里也必须执行
            factor_df_shifted = factor_df.shift(1)
            
            # 5. 存入缓存
            self._factor_cache[factor_name] = factor_df_shifted
            print(f"FactorEngine: 因子 '{factor_name}' 已计算并缓存。")
            return factor_df_shifted
            
        except Exception as e:
            print(f"FactorEngine: 计算 Xarray 因子 {factor_name} 时发生严重错误: {e}")
            import traceback
            traceback.print_exc()
            # 存入一个空 DataFrame 以免重复计算
            self._factor_cache[factor_name] = pd.DataFrame() 
            return self._factor_cache[factor_name]

    def get_factor_snapshot(self, current_date, codes, factors, weights=None) -> pd.DataFrame:
            """获取截面快照并计算加权得分"""
            series_list = []
            for f in factors:
                if f not in self._factor_cache: self._compute_and_cache_factor(f)
                df = self._factor_cache[f]
                series = df.loc[current_date] if current_date in df.index else pd.Series(index=codes, dtype=float)
                series.name = f
                series_list.append(series.reindex(codes))

            snapshot_df = pd.concat(series_list, axis=1).fillna(0.0)

            # 加权组合逻辑
            if len(factors) > 1:
                # Z-Score 标准化
                z_df = snapshot_df[factors].apply(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x - x.mean())
                if weights is None: weights = {f: 1.0/len(factors) for f in factors}
                
                # 计算加权和
                snapshot_df['composite_score'] = sum(z_df[f] * weights.get(f, 0) for f in factors)
            
            return snapshot_df