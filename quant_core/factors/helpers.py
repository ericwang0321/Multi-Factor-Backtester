# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xarray as xr
import bottleneck as bl # 确保安装 pip install bottleneck
from numpy import abs, log, sign

# ==============================================================================
# === 核心时间序列辅助函数 (Core Time Series Helper Functions) ===
# ==============================================================================

def set_index_like(arr: np.ndarray, other: xr.DataArray):
    """将 numpy 数组转换为具有与另一个 xarray.DataArray 相同索引的 xarray.DataArray"""
    dims = other.coords.dims[: arr.ndim]
    coords = [(dim, other.coords[dim].values) for dim in dims]
    array = xr.DataArray(arr, coords=coords)
    for dim in dims:
        array.coords[dim] = other.coords[dim]
    return array

def where(cond, x, y=np.nan):
    return xr.where(cond, x, y)

def delay(arr, period=1):
    return arr.shift({'datetime': period})

def delta(arr, period=1):
    before = delay(arr, period)
    return arr - before

def pct_change(array, n=1):
    shifted_array = delay(array, n)
    shifted_array = where(shifted_array == 0, np.nan, shifted_array)
    return array / shifted_array - 1

def ts_rolling(array: xr.DataArray, n_period, **kwargs):
    """通用滚动窗口函数"""
    min_p = max(2, int(n_period * 0.8))
    return array.rolling({'datetime': n_period}, min_periods=min_p, **kwargs)

# --- 基础统计 ---
def ts_sum(arr, window=10):
    return ts_rolling(arr, window).sum()

def ts_mean(arr, window=10):
    return ts_rolling(arr, window).mean()

def ts_std(arr, window=10):
    return ts_rolling(arr, window).std()

def ts_min(arr, window=10):
    return ts_rolling(arr, window).min()

def ts_max(arr, window=10):
    return ts_rolling(arr, window).max()

def ts_argmax(arr, window=10):
    return ts_rolling(arr, window).argmax()

def ts_argmin(arr, window=10):
    return ts_rolling(arr, window).argmin()

# --- 高级统计 (Correlation, Covariance, Rank) ---
# [关键修复] 你的 Alpha006, Alpha018 等因子依赖这些函数

def ts_cov(x: xr.DataArray, y: xr.DataArray, window=10):
    """滚动协方差"""
    # 滚动计算需要对齐，xarray 的 rolling_exp 或 rolling 都支持自动对齐
    return ts_rolling(x, window).cov(y)

def ts_corr(x: xr.DataArray, y: xr.DataArray, window=10):
    """滚动相关系数"""
    return ts_rolling(x, window).corr(y)

def rank(arr: xr.DataArray, axis=-1, pct=True):
    """
    截面排名 (Cross-sectional Rank)
    通常用于 Alpha 因子，在 'sec_code' 维度上对股票进行排名
    """
    # 假设数据维度是 (datetime, sec_code)，rank 应该在 sec_code 维度进行
    # 如果 dim 参数传入错误，xarray 会报错，这里默认对最后一个维度(通常是股票)排名
    dim_name = arr.dims[-1] if axis == -1 else arr.dims[axis]
    
    # xarray 的 rank 方法
    ranked = arr.rank(dim=dim_name, pct=pct)
    return ranked

def scale(arr: xr.DataArray, scale=1):
    """
    标准化：类似 L1 norm，使绝对值之和为 scale
    (alpha101 中常见操作)
    """
    return arr.mul(scale).div(abs(arr).sum(dim='sec_code'))

# --- 指数移动平均 ---
def xr_ewm(data_array: xr.DataArray, alpha: float = None, span: int = None, adjust: bool = False, min_periods: int = 0) -> xr.DataArray:
    if alpha is None:
        if span is None:
            raise ValueError("必须提供 alpha 或 span")
        alpha = 2 / (span + 1)

    pandas_obj = data_array.to_pandas()
    
    # 兼容 DataFrame 和 Series(MultiIndex)
    if isinstance(pandas_obj, pd.Series) and isinstance(pandas_obj.index, pd.MultiIndex):
        ewm_pandas = pandas_obj.unstack().ewm(alpha=alpha, adjust=adjust, min_periods=min_periods).mean().stack()
        ewm_pandas = ewm_pandas.reindex(pandas_obj.index)
    else:
        ewm_pandas = pandas_obj.ewm(alpha=alpha, adjust=adjust, min_periods=min_periods).mean()

    return xr.DataArray(ewm_pandas, dims=data_array.dims, coords=data_array.coords)