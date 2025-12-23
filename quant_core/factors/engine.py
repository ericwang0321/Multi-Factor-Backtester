# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Any

# --- 引用调整 ---
from .definitions import * 
from ..data.query_helper import DataQueryHelper 

class FactorEngine:
    """
    【升级版】因子引擎 (IBKR Parquet 适配版)
    直接使用 DataQueryHelper 读取 Parquet。
    已根据数据缺失情况 (无 Turnover/MarketCap) 禁用相关因子。
    """
    
    def __init__(self, query_helper: Optional[DataQueryHelper] = None):
        """
        初始化因子引擎。
        Args:
            query_helper: 新的数据查询助手，替代旧的 data_handler
        """
        print("FactorEngine: 正在初始化 (Parquet Mode)...")
        
        if query_helper is None:
            self.query_helper = DataQueryHelper()
        else:
            self.query_helper = query_helper
        
        self._factor_cache: Dict[str, pd.DataFrame] = {} 
        self.xarray_data: Optional[xr.DataArray] = None
        
        # 存储当前策略使用的权重，供回测引擎调用
        self.current_weights = {} 
        
        # --- 因子注册表 (已根据您的数据情况清洗) ---
        self.FACTOR_REGISTRY: Dict[str, Any] = {
            # --- 纯量价 Alpha (保留) ---
            'alpha006': (Alpha006, {'corr_period': 15}),
            'alpha007': (Alpha007, {'delta_period': 3, 'vol_mean_period': 10, 'rank_period': 50}),
            'alpha013': (Alpha013, {'cov_period': 5}),
            'alpha018': (Alpha018, {'std_period': 5, 'corr_period': 12}),
            'alpha023': (Alpha023, {'ts_period': 30, 'delta_period': 3}),
            'alpha033': (Alpha033, {}),
            'alpha040': (Alpha040, {'ts_period': 8}),
            'alpha041': (Alpha041, {}),
            'alpha042': (Alpha042, {}), # 使用 vwap
            'alpha053': (Alpha053, {'ts_period': 15}),
            'alpha060': (Alpha060, {'ts_period': 25}),
            'alpha191_014': (Alpha191_014, {'delay_period': 3}),
            'alpha191_018': (Alpha191_018, {'delay_period': 3}),
            'alpha191_020': (Alpha191_020, {'delay_period': 3}),
            
            # --- 依赖 Amount (成交额) 的因子 (保留) ---
            'amihud_illiquidity': (AmihudIlliquidity, {'window': 126}),
            'amount_mean': (AmountMean, {'window': 252}),
            'amount_std': (AmountStd, {'window': 42}),
            'amount_volatility': (AmountVolatility, {'window': 63}),
            
            # --- 技术指标 (保留) ---
            'cmo': (CMO, {'window': 40}),
            'dv_divergence': (DV_Divergence, {'price_window': 15, 'corr_window': 20}),
            'rsi': (RSI, {'window': 12}),
            'stochastic_k': (StochasticK, {'window': 40}),
            'momentum': (Momentum, {'window': 168}),
            'momentum_acceleration': (MomentumAcceleration, {'period1': 42, 'period2': 126}),
            
            # --- Eric 系列 (部分保留) ---
            'eric_adx_weighted_momentum': (Eric_ADX_Weighted_Momentum, {'momentum_window': 22, 'adx_window': 25}),
            'eric_asymmetric_vol_reversion': (Eric_Asymmetric_Volatility_Reversion, {'window': 60}),
            'eric_asymmetric_vol_momentum': (HT_Asymmetric_Volatility_Momentum, {'mom_window': 90, 'vol_window': 90, 'delta_window': 20}),
            'eric_cs_resid_vol': (Eric_CS_Residual_Volatility, {'window': 100}),
            'eric_enhanced_momentum': (EricEnhancedMomentum, {}),
            'eric_improved_alpha_022': (Eric_Improved_Alpha_022, {}),
            'eric_pv_corr': (EricFactor, {'window': 60}),
            'eric_pv_divergence': (Eric_PV_Divergence_Covariance, {'window': 10}),
            'eric_tail_skew': (Eric_Improved_Skewness, {'window': 84, 'std_multiple': 1.0}),
            'eric_u_shape_vol': (Eric_U_Shape_Volatility_Factor, {'window': 30}),
            
            'falkenblog_disparity_mean': (Falkenblog_Disparity_Mean, {'window': 40}),
            'falkenblog_disparity_std': (Falkenblog_Disparity_Std, {'window': 40}),
            
            'kurtosis': (HT_Kurtosis, {'window': 60}),
            'momentum_daily_mean': (HT_Momentum_DailyMean, {'window': 42}),
            'monthly_avg_overnight_return': (Eric_Monthly_Avg_Overnight_Return, {'window': 55}),
            'monthly_std_overnight_return': (Eric_Monthly_Std_Overnight_Return, {'window': 30}),
            'skewness': (HT_Skewness, {'window': 42}),
            'term_structure_proxy': (HT_TermStructure_RollYield_Proxy, {'window': 30}),
            
            'volatility': (HT_Volatility, {'window': 30}),
            'volume_mean': (VolumeMean, {'window': 10}),
            'volume_std': (VolumeStd, {'window': 30}),
            'trend_score': (TrendScore, {'window': 30}),
            
            'ht_low_vol_breakout': (HT_Low_Volatility_Breakout, {'mom_window': 90, 'vol_window': 90}), 
            'ht_trend_exhaustion_reversal': (HT_Trend_Exhaustion_Reversal, {'window': 90}),
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
            'trend_certainty_score': (Trend_Certainty_Score, {
                'window': 63, 'adx_window': 14, 'adx_threshold': 25.0
            }),
        }
        print(f"FactorEngine: 初始化完成。已启用 {len(self.FACTOR_REGISTRY)} 个可用因子。")

    def _get_xarray_data(self) -> xr.DataArray:
        """
        从 DataQueryHelper 加载全量 Parquet 数据并转为 Xarray
        """
        if self.xarray_data is not None:
            return self.xarray_data

        print("FactorEngine: 正在从 QueryHelper 加载全量数据...")
        
        # 1. 使用 QueryHelper 获取清洗后的 DataFrame (含 vwap, amount)
        raw_df = self.query_helper.get_all_price_data()
        
        if raw_df.empty:
            raise ValueError("No data returned from QueryHelper!")

        # 2. 筛选所需列
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'vwap']
        existing_cols = [col for col in required_cols if col in raw_df.columns]
        
        # 3. 转换为宽格式 (Pivot)
        wide_data_list = []
        for col in existing_cols:
            try:
                wide_col = pd.pivot_table(raw_df, index='datetime', columns='sec_code', values=col)
                wide_data_list.append(wide_col)
            except Exception as e:
                print(f"FactorEngine: Pivot error on {col}: {e}")
                
        if not wide_data_list:
             raise ValueError("FactorEngine: Failed to pivot data.")
             
        wide_data = pd.concat(wide_data_list, axis=1, keys=existing_cols)
        
        # 4. 填充缺失值
        wide_data = wide_data.ffill().bfill() 
        
        # 5. 转换为 Xarray
        def convert_wide_df_to_xarray(wide_df: pd.DataFrame) -> xr.DataArray:
            if not isinstance(wide_df.columns, pd.MultiIndex):
                raise ValueError("Input must be MultiIndex DataFrame")
            stacked = wide_df.stack(level='sec_code', future_stack=True)
            stacked.index.names = ['datetime', 'sec_code']
            stacked.columns.name = 'field'
            return stacked.to_xarray().to_array('field')

        try:
            self.xarray_data = convert_wide_df_to_xarray(wide_data)
            print(f"FactorEngine: Xarray 构建完成。Shape: {self.xarray_data.shape}")
            return self.xarray_data
        except Exception as e:
            raise RuntimeError(f"Xarray conversion failed: {e}")

    def _compute_and_cache_factor(self, factor_name: str) -> pd.DataFrame:
        """
        计算单个因子并缓存
        """
        if factor_name not in self.FACTOR_REGISTRY:
            raise ValueError(f"Factor '{factor_name}' is not registered (or disabled).")
            
        if factor_name in self._factor_cache:
            return self._factor_cache[factor_name]

        print(f"FactorEngine: Computing {factor_name}...")
        try:
            market_array = self._get_xarray_data()
            factor_class, params = self.FACTOR_REGISTRY[factor_name]
            factor_obj = factor_class(**params)
            
            # 计算
            factor_df = factor_obj.transform(market_array)
            
            # 必须 Shift(1) 避免未来函数
            factor_df_shifted = factor_df.shift(1)
            
            self._factor_cache[factor_name] = factor_df_shifted
            return factor_df_shifted
            
        except Exception as e:
            print(f"Error computing {factor_name}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def get_factor_snapshot(self, current_date, codes, factors, weights=None) -> pd.DataFrame:
        """
        获取截面快照并计算加权得分 (BacktestEngine 调用此方法)
        """
        series_list = []
        
        # 确保数据已初始化
        if self.xarray_data is None:
            self._get_xarray_data()

        for f in factors:
            # 计算或获取缓存中的因子值
            df = self._compute_and_cache_factor(f)
            
            # 尝试获取当天的截面数据
            if df.empty:
                # 因子计算失败或被禁用
                series = pd.Series(index=codes, data=0.0)
            elif current_date in df.index:
                series = df.loc[current_date]
            else:
                # 当天没有数据 (比如非交易日或数据缺失)
                series = pd.Series(index=codes, dtype=float)
            
            series.name = f
            # 按照传入的 Universe 代码列表进行对齐
            series_list.append(series.reindex(codes))

        # 合并所有因子列
        snapshot_df = pd.concat(series_list, axis=1)
        
        # 填充缺失值 (防止后续加权计算出错)
        snapshot_df = snapshot_df.fillna(0.0)

        # 加权组合逻辑
        if weights is None:
            weights = {f: 1.0/len(factors) for f in factors} if factors else {}
            
        if len(factors) > 0:
            # Z-Score 标准化：确保不同量纲的因子可以公平加权
            z_df = snapshot_df[factors].apply(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x - x.mean())
            
            # 计算加权综合得分
            composite_scores = pd.Series(0.0, index=snapshot_df.index)
            for f in factors:
                w = weights.get(f, 0.0)
                if f in z_df.columns:
                    composite_scores += z_df[f] * w
            
            snapshot_df['composite_score'] = composite_scores
        else:
            snapshot_df['composite_score'] = 0.0
        
        return snapshot_df