# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xarray as xr
import bottleneck as bl
from numpy import abs, log, sign

# ==============================================================================
# === 核心时间序列辅助函数 (Core Time Series Helper Functions) ===
# === (从你的 prepare_factor_data.py 中提取和简化) ===
# ==============================================================================

# --- 省略了所有辅助函数定义，与上一版本完全相同 ---
def set_index_like(arr: np.ndarray, other: xr.DataArray):
    """将 numpy 数组转换为具有与另一个 xarray.DataArray 相同索引的 xarray.DataArray"""
    dims = other.coords.dims[: arr.ndim]
    coords = [(dim, other.coords[dim].values) for dim in dims]
    array = xr.DataArray(arr, coords=coords)
    for dim in dims:
        array.coords[dim] = other.coords[dim]
    return array

def where(cond, x, y=np.nan):
    """xarray.where 的一个简单封装，方便使用"""
    return xr.where(cond, x, y)

def delay(arr, period=1):
    """对时间序列数据进行延迟（滞后）"""
    return arr.shift({'datetime': period})

def delta(arr, period=1):
    """计算时间序列数据的时间差分"""
    before = delay(arr, period)
    return arr - before

def pct_change(array, n=1):
    """计算百分比变化率（收益率）"""
    shifted_array = delay(array, n)
    # 避免除以零
    shifted_array = where(shifted_array == 0, np.nan, shifted_array)
    return array / shifted_array - 1

def ts_rolling(array: xr.DataArray, n_period, **kwargs):
    """对时间序列数据进行滚动窗口计算 (xarray 版本)"""
    # 确保窗口期内至少有 80% 或至少 2 个有效数据点
    min_p = max(2, int(n_period * 0.8))
    return array.rolling({'datetime': n_period}, min_periods=min_p, **kwargs)

# --- 基础滚动的统计量 ---
def ts_sum(arr, window=10):
    """计算时间序列滚动求和"""
    return ts_rolling(arr, window).sum()

def ts_mean(arr, window=10):
    """计算时间序列滚动平均"""
    return ts_rolling(arr, window).mean()

def ts_std(arr, window=10):
    """计算时间序列滚动标准差"""
    return ts_rolling(arr, window).std()

# --- 指数移动平均 (Exponential Moving Average) for xarray ---
def xr_ewm(data_array: xr.DataArray, alpha: float = None, span: int = None, adjust: bool = False, min_periods: int = 0) -> xr.DataArray:
    """
    对 xarray.DataArray 沿 'datetime' 维度应用指数移动平均 (ewm)。
    优先使用 alpha，如果 alpha 为 None，则尝试使用 span 计算 alpha。
    """
    if alpha is None:
        if span is None:
            raise ValueError("必须提供 alpha 或 span 中的至少一个参数")
        # 根据 span 计算 alpha，这是常用的转换公式
        alpha = 2 / (span + 1)

    # 将 xarray DataArray 转换为 pandas DataFrame/Series 以便使用 pandas 的 .ewm() 方法
    # to_pandas() 可能会创建一个带有 MultiIndex 的 Series，需要后续处理
    pandas_obj = data_array.to_pandas()

    # 对 pandas 对象应用 ewm 计算
    # 需要处理单 Series 和 DataFrame 的情况，以及 MultiIndex 的情况
    if isinstance(pandas_obj, pd.Series) and isinstance(pandas_obj.index, pd.MultiIndex):
        # 假设索引是 (datetime, sec_code)，我们需要按 sec_code 分组计算
        # 先 unstack 成 DataFrame (datetime x sec_code)，计算 ewm，再 stack 回去
        ewm_pandas = pandas_obj.unstack().ewm(alpha=alpha, adjust=adjust, min_periods=min_periods).mean().stack()
        # 重新排序索引以匹配原始 xarray 的顺序，防止错位
        ewm_pandas = ewm_pandas.reindex(pandas_obj.index)
    elif isinstance(pandas_obj, pd.DataFrame): # 假设索引是 datetime, 列是 sec_code
         ewm_pandas = pandas_obj.ewm(alpha=alpha, adjust=adjust, min_periods=min_periods).mean()
    else: # 假设是单维时间序列 (只有 datetime 索引)
         ewm_pandas = pandas_obj.ewm(alpha=alpha, adjust=adjust, min_periods=min_periods).mean()

    # 将计算结果转换回 xarray DataArray，保持原始的维度和坐标
    return xr.DataArray(ewm_pandas, dims=data_array.dims, coords=data_array.coords)

# --- 如果需要，可以在这里添加 rank, ts_corr, ts_cov 等其他辅助函数 ---
# def rank(arr, pct=True, ascending=True): ...
# def ts_corr(x, y, window=10): ...
# def ts_cov(x, y, window=10): ...

