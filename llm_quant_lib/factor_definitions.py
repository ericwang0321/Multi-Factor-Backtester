# prepare_factor_data.py
# 整合了來自四個不同研報的量價因子計算腳本
# 包含: 中信期貨報告因子, 國泰君安191/101 Alpha因子, 華泰證券/期貨報告因子, 及 Falkenblog 隔夜/日內收益率因子

import pandas as pd
import numpy as np
import sys
import os
import bottleneck as bl
import xarray as xr
from numpy import abs, log, sign
import statsmodels.api as sm # <--- 新增這一行

# ==============================================================================
# === 1. 核心輔助函數 (Core Helper Functions) ===
# ==============================================================================

def set_index_like(arr: np.ndarray, other: xr.DataArray):
    """將 numpy 數組轉換為具有與另一個 xarray.DataArray 相同索引的 xarray.DataArray"""
    dims = other.coords.dims[: arr.ndim]
    coords = [(dim, other.coords[dim].values) for dim in dims]
    array = xr.DataArray(arr, coords=coords)
    for dim in dims:
        array.coords[dim] = other.coords[dim]
    return array

def where(cond, x, y=np.nan):
    """xarray.where 的一個簡單封裝，方便使用"""
    return xr.where(cond, x, y)

def get_field_value(arr: xr.DataArray, field):
    """從 xarray 中安全地獲取指定 'field' 的值"""
    if field in arr['field']:
        return arr.sel({'field': field})
    else:
        template = arr.isel(field=0)
        return xr.full_like(template, np.nan)

def delay(arr, period=1):
    """對時間序列數據進行延遲（滯後）"""
    return arr.shift({'datetime': period})

def delta(arr, period=1):
    """計算時間序列數據的差分"""
    before = delay(arr, period)
    return arr - before

def pct_change(array, n=1):
    """計算百分比變化率（收益率）"""
    shifted_array = delay(array, n)
    shifted_array = where(shifted_array == 0, np.nan, shifted_array)
    return array / shifted_array - 1

def rank(arr, pct=True, ascending=True):
    """對截面數據進行排序"""
    dim = 'sec_code'
    nan_mask = np.isnan(arr)
    arr_rank = arr.rank(dim=dim, pct=pct)
    if not ascending:
        valid_ranks = arr_rank.where(~nan_mask)
        max_rank = valid_ranks.max(dim=dim)
        arr_rank = max_rank - arr_rank
        if not pct:
            arr_rank += 1
    return arr_rank.where(~nan_mask, np.nan)

def ts_rolling(array: xr.DataArray, n_period, **kwargs):
    """對時間序列數據進行滾動窗口計算"""
    min_p = max(2, int(n_period * 0.8))
    return array.rolling({'datetime': n_period}, min_periods=min_p, **kwargs)

def ts_sum(arr, window=10): return ts_rolling(arr, window).sum()
def ts_mean(arr, window=10): return ts_rolling(arr, window).mean()
def ts_std(arr, window=10): return ts_rolling(arr, window).std()

def ts_skew(arr, window=10):
    """手動計算滾動偏度"""
    mu = ts_mean(arr, window)
    sigma = ts_std(arr, window)
    safe_sigma = where(sigma == 0, np.nan, sigma)
    centered_arr = arr - mu
    m3 = ts_mean(centered_arr**3, window)
    return m3 / (safe_sigma**3)

def ts_kurt(arr, window=10):
    """手動計算滾動超額峰度 (減去3)"""
    mu = ts_mean(arr, window)
    sigma = ts_std(arr, window)
    safe_sigma = where(sigma == 0, np.nan, sigma)
    centered_arr = arr - mu
    m4 = ts_mean(centered_arr**4, window)
    return m4 / (safe_sigma**4) - 3

def ts_cov(x, y, window=10):
    """計算滾動協方差"""
    na_mask = np.isnan(x) | np.isnan(y)
    x = where(na_mask, np.nan, x)
    y = where(na_mask, np.nan, y)
    xy = x * y
    e_xy = ts_rolling(xy, window).mean()
    e_x = ts_rolling(x, window).mean()
    e_y = ts_rolling(y, window).mean()
    count = ts_rolling(x, window).count()
    cov = e_xy - (e_x * e_y)
    return cov * (count / (count - 1))

def ts_corr(x, y, window=10):
    """計算滾動相關係數"""
    stddev_x = ts_std(x, window)
    stddev_y = ts_std(y, window)
    cov_array = ts_cov(x, y, window)
    denominator = stddev_y * stddev_x
    denominator = where(denominator == 0, np.nan, denominator)
    return cov_array / denominator

def ts_rank(x, window=10):
    """計算時間序列滾動排序"""
    axis = x.get_axis_num('datetime')
    return set_index_like(bl.move_rank(x.values, window=window, axis=axis), x)

def ts_min(arr, window=10): return ts_rolling(arr, window).min()
def ts_max(arr, window=10): return ts_rolling(arr, window).max()

def ts_argmax(x, window=10):
    """計算時間序列滾動最大值位置"""
    axis = x.get_axis_num('datetime')
    return set_index_like(window - 1 - bl.move_argmax(x.fillna(-np.inf).values, window=window, axis=axis), x)

def ts_argmin(x, window=10):
    """計算時間序列滾動最小值位置"""
    axis = x.get_axis_num('datetime')
    return set_index_like(window - 1 - bl.move_argmin(x.fillna(np.inf).values, window=window, axis=axis), x)
    
def ts_weighted_mean(x, weight):
    """計算時間序列滾動加權平均"""
    window = len(weight)
    weight_arr = xr.DataArray(weight, dims=['window']) / np.sum(weight)
    rx = ts_rolling(x, window)
    return rx.construct('window').dot(weight_arr)

def ts_decay_linear(x, window=10):
    """計算時間序列線性衰減加權平均"""
    weight = np.arange(1.0, 1.0 + window)
    return ts_weighted_mean(x, weight)

# [--- 在此處新增以下 xr_ewm 輔助函數 ---]
def xr_ewm(data_array: xr.DataArray, alpha: float, adjust: bool, min_periods: int) -> xr.DataArray:
    """
    對 xarray.DataArray 沿著 'datetime' 維度應用指數移動平均 (ewm)。
    
    Args:
        data_array: 輸入的 xarray DataArray。
        alpha: 平滑因子 alpha。
        adjust: ewm 的 adjust 參數。
        min_periods: 最小觀測期。
        
    Returns:
        計算完 ewm 的 xarray DataArray。
    """
    # 將 xarray DataArray 轉換為 DataFrame 以便使用 .ewm()
    df = data_array.to_pandas()
    # 應用 ewm 計算
    ewm_df = df.ewm(alpha=alpha, adjust=adjust, min_periods=min_periods).mean()
    # 將結果轉換回 xarray DataArray
    return xr.DataArray(ewm_df, dims=data_array.dims, coords=data_array.coords)

# [--- 請用以下版本替換你原來的 calc_adx 函數 ---]
def calc_adx(high: xr.DataArray, low: xr.DataArray, close: xr.DataArray, window: int = 14):
    """
    計算平均趨向指標 (ADX), +DI, 和 -DI
    使用 Wilder's Smoothing (等同於 alpha = 1/window 的 EWM)
    """
    # 1. 計算趨向變動 (+DM, -DM) 和真實波幅 (TR)
    move_up = delta(high, 1)
    move_down = -delta(low, 1)

    plus_dm = where((move_up > move_down) & (move_up > 0), move_up, 0)
    minus_dm = where((move_down > move_up) & (move_down > 0), move_down, 0)

    tr1 = abs(high - low)
    tr2 = abs(high - delay(close, 1))
    tr3 = abs(low - delay(close, 1))
    tr = xr.concat([tr1, tr2, tr3], dim='tr_calc').max(dim='tr_calc', skipna=True)

    # 2. 使用新的 xr_ewm 函數進行平滑
    alpha = 1 / window
    smooth_plus_dm = xr_ewm(plus_dm, alpha=alpha, adjust=False, min_periods=window)
    smooth_minus_dm = xr_ewm(minus_dm, alpha=alpha, adjust=False, min_periods=window)
    smooth_tr = xr_ewm(tr, alpha=alpha, adjust=False, min_periods=window)
    
    # 處理分母為零的情況
    safe_smooth_tr = where(smooth_tr == 0, np.nan, smooth_tr)

    # 3. 計算方向指標 (+DI, -DI)
    plus_di = 100 * (smooth_plus_dm / safe_smooth_tr)
    minus_di = 100 * (smooth_minus_dm / safe_smooth_tr)

    # 4. 計算趨向指標 (DX) 和平均趨向指標 (ADX)
    di_sum = plus_di + minus_di
    safe_di_sum = where(di_sum == 0, np.nan, di_sum)
    dx = 100 * (abs(plus_di - minus_di) / safe_di_sum)
    
    adx = xr_ewm(dx, alpha=alpha, adjust=False, min_periods=window)

    return adx, plus_di, minus_di

# [--- 在此處新增以下 scale 輔助函數 ---]
def scale(arr: xr.DataArray) -> xr.DataArray:
    """
    對截面數據進行標準化 (z-score scaling)。
    使得每個時間點上的因子截面均值為0，標準差為1。
    """
    dim = 'sec_code'
    mean = arr.mean(dim=dim)
    std = arr.std(dim=dim)
    safe_std = where(std == 0, np.nan, std)
    return (arr - mean) / safe_std

# [--- 在此處新增以下 cs_regression 輔助函數 ---]
def cs_regression(y: xr.DataArray, x: xr.DataArray) -> xr.DataArray:
    """
    對 xarray 數據進行每日橫截面回歸，並返回殘差。
    y = α + β*x + ε
    """
    y_no_field = y.drop_vars('field', errors='ignore')
    x_no_field = x.drop_vars('field', errors='ignore')
    data = xr.merge([y_no_field.rename('y'), x_no_field.rename('x')])
    
    def ols_resid(df_group, y_col='y', x_col='x'):
        """在單個截面上執行OLS回歸並返回殘差的函數"""
        df = df_group.to_dataframe()
        df_cleaned = df.dropna()
        
        if df_cleaned.shape[0] < 2:
            empty_series = pd.Series(np.nan, index=df.index)
            # 確保即使是空的 Series 也移除 datetime 層級
            return xr.DataArray(empty_series.droplevel('datetime'))

        Y = df_cleaned[y_col]
        X = sm.add_constant(df_cleaned[x_col])
        
        model = sm.OLS(Y, X).fit()
        predictions = model.predict(sm.add_constant(df.loc[df_cleaned.index, [x_col]]))
        residuals = df[y_col] - predictions
        final_residuals = residuals.reindex(df.index)
        
        # --- 核心修正點 ---
        # 我們返回的 Series/DataArray 不能包含 groupby 的維度('datetime')
        # 因此，我們從 MultiIndex 中移除 'datetime' 層級，只留下 'sec_code'
        final_residuals_single_index = final_residuals.droplevel('datetime')
        
        # 現在 Series 的 index 只有 'sec_code'，xarray 會將其作為唯一的維度
        return xr.DataArray(final_residuals_single_index)

    # apply 現在接收到的每日結果只包含 sec_code 維度，
    # 它可以順利地將這些結果沿著新的 'datetime' 維度進行拼接。
    residuals_xr = data.groupby('datetime').apply(ols_resid)
    
    return residuals_xr

def ts_regression_bivariate(y_data: xr.DataArray, x_data: xr.DataArray, window: int):
    """
    對兩個時間序列數據 (y, x) 進行滾動線性迴歸，一次性返回斜率和 R^2。
    專為 RSRS 這類因子設計。
    """
    # 構造滾動窗口視圖
    min_p = int(window * 0.8)
    y_win = y_data.rolling({'datetime': window}, min_periods=min_p).construct('window_dim')
    x_win = x_data.rolling({'datetime': window}, min_periods=min_p).construct('window_dim')
    
    # 計算各項統計量的和
    n = window
    sum_x = x_win.sum(dim='window_dim')
    sum_y = y_win.sum(dim='window_dim')
    sum_x2 = (x_win**2).sum(dim='window_dim')
    sum_y2 = (y_win**2).sum(dim='window_dim')
    sum_xy = (x_win * y_win).sum(dim='window_dim')
    
    # 計算協方差和方差的分子部分
    numerator_cov = n * sum_xy - sum_x * sum_y
    denominator_var_x = n * sum_x2 - sum_x**2
    denominator_var_y = n * sum_y2 - sum_y**2
    
    # 計算斜率 (beta)
    safe_denominator_var_x = where(denominator_var_x == 0, np.nan, denominator_var_x)
    slope = numerator_cov / safe_denominator_var_x

    # 計算 R-squared (決定係數)
    safe_denominator_r2 = where(denominator_var_x * denominator_var_y == 0, np.nan, denominator_var_x * denominator_var_y)
    r_squared = (numerator_cov**2) / safe_denominator_r2
    
    return slope, r_squared

def ts_regression_all_stats(y: xr.DataArray, window: int):
    """
    對時間序列數據進行滾動線性迴歸 (y ~ time)，一次性返回斜率, R^2, 和 y 的標準差。
    """
    x = xr.DataArray(np.arange(window), dims=['window_dim'])
    sum_x = x.sum()
    sum_x2 = (x**2).sum()
    n = window
    
    y_win = y.rolling({'datetime': window}, min_periods=int(window * 0.8)).construct('window_dim')
    
    sum_y = y_win.sum(dim='window_dim')
    sum_y2 = (y_win**2).sum(dim='window_dim')
    xy = y_win * x
    sum_xy = xy.sum(dim='window_dim')
    
    numerator_cov = n * sum_xy - sum_x * sum_y
    denominator_var_x = n * sum_x2 - sum_x**2
    denominator_var_y = n * sum_y2 - sum_y**2
    
    safe_denominator_var_x = where(denominator_var_x == 0, np.nan, denominator_var_x)
    slope = numerator_cov / safe_denominator_var_x

    safe_denominator_r2 = where(denominator_var_x * denominator_var_y == 0, np.nan, denominator_var_x * denominator_var_y)
    r_squared = (numerator_cov**2) / safe_denominator_r2
    
    # 計算 y 在窗口內的標準差
    # std_dev = sqrt( (n*Σy² - (Σy)²) / (n*(n-1)) )
    safe_denominator_std = where(n * (n - 1) == 0, np.nan, n * (n - 1))
    std_dev = np.sqrt(denominator_var_y / safe_denominator_std)

    return slope, r_squared, std_dev

def ts_max_drawdown(array: xr.DataArray, window: int):
    """
    計算滾動窗口內的最大回撤。
    公式: (當前值 - 區間最高值) / 區間最高值
    """
    # 這裡定義了變數 min_periods
    min_periods = int(window * 0.8)
    
    # 計算窗口內的滾動最高值
    # <--- 修正點: 將 min_p 改為 min_periods --->
    rolling_max = array.rolling({'datetime': window}, min_periods=min_periods).max()
    
    # 計算每個時間點的 drawdown
    drawdown = (array - rolling_max) / where(rolling_max == 0, np.nan, rolling_max)
    
    # 返回窗口期內 drawdown 的最小值 (即最大回撤)
    # <--- 修正點: 將 min_p 改為 min_periods --->
    return drawdown.rolling({'datetime': window}, min_periods=min_periods).min()

def calc_cci(high: xr.DataArray, low: xr.DataArray, close: xr.DataArray, window: int = 20):
    """
    計算順勢指標 (CCI)。
    """
    # 計算典型價格 (Typical Price)
    tp = (high + low + close) / 3
    
    # 計算典型價格的移動平均
    sma_tp = ts_mean(tp, window)
    
    # 計算平均絕對偏差 (Mean Deviation)
    mean_dev = ts_mean(abs(tp - sma_tp), window)
    
    # 避免除以零
    safe_mean_dev = where(mean_dev == 0, np.nan, mean_dev)
    
    # 計算 CCI
    cci = (tp - sma_tp) / (0.015 * safe_mean_dev)
    
    return cci

# [--- 在此處新增以下 pdf_scale 輔助函數 ---]
def pdf_scale(arr: xr.DataArray, a: float = 1.0) -> xr.DataArray:
    """
    根據研報定義的 scale 函數: a * X / sum(abs(X))
    對截面數據進行縮放處理。
    """
    dim = 'sec_code'
    # 處理分母為零的情況
    sum_abs = abs(arr).sum(dim=dim)
    safe_sum_abs = where(sum_abs == 0, np.nan, sum_abs)
    return a * arr / safe_sum_abs
# ==============================================================================
# === 1.5. 遺傳規劃輔助函數 (GP Helper Functions) ===
# ==============================================================================
def gp_add(x, y): return x + y
def gp_sub(x, y): return x - y
def gp_mul(x, y): return x * y
def gp_neg(x): return -x
def gp_abs(x): return abs(x)

def gp_pdiv(x, y):
    """受保護的除法 (Protected Division)，避免除以零"""
    return where(y == 0, np.nan, x / y)

def gp_sqrt(x):
    """受保護的平方根 (Protected Square Root)，忽略負值"""
    return np.sqrt(where(x < 0, np.nan, x))

def gp_log(x):
    """受保護的對數 (Protected Log)，忽略非正值"""
    return log(where(x <= 0, np.nan, x))

# GP 中的 max/min 通常是二元操作
def gp_max(x, y):
    # 使用 xr.ufuncs 來確保維度和坐標正確對齊
    return xr.ufuncs.maximum(x, y)

def gp_min(x, y):
    return xr.ufuncs.minimum(x, y)
#     
# ==============================================================================
# === 2. 因子基類 (Base Factor Class) ===
# ==============================================================================
class BaseAlpha:
    """所有因子計算類的統一基類"""
    def __init__(self):
        self.array_ = None

    def transform(self, X: xr.DataArray) -> pd.DataFrame:
        """主轉換函數，接收 xarray 數據，返回 pandas DataFrame 格式的因子值"""
        self.array_ = X
        result_xr = self.predict()
        if not isinstance(result_xr, xr.DataArray):
            raise TypeError(f"{self.__class__.__name__} did not return an xarray.DataArray")
        
        result_xr = xr.where(np.isinf(result_xr), np.nan, result_xr)
        result_series = result_xr.to_series()
        result_df = result_series.unstack(level='sec_code')
        
        for col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
        
        del self.array_
        return result_df

    def predict(self) -> xr.DataArray:
        """子類需要實現的核心計算邏輯"""
        raise NotImplementedError

    # --- 屬性方便地獲取數據 ---
    @property
    def close(self): return get_field_value(self.array_, 'close')
    @property
    def open(self): return get_field_value(self.array_, 'open')
    @property
    def low(self): return get_field_value(self.array_, 'low')
    @property
    def high(self): return get_field_value(self.array_, 'high')
    @property
    def volume(self): return where(get_field_value(self.array_, 'volume') == 0, 1e-9, get_field_value(self.array_, 'volume'))
    @property
    def amount(self): return get_field_value(self.array_, 'amount')
    @property
    def vwap(self): return get_field_value(self.array_, 'vwap')
    @property
    def turnover(self): return get_field_value(self.array_, 'turnover')
    @property
    def market_cap(self): return get_field_value(self.array_, 'market_cap')
    @property
    def returns(self): return pct_change(self.close, 1)

# ==============================================================================
# === 3. 因子實現 (Factor Implementations) ===
# ==============================================================================

# --- B.4 新增：為非類別因子創建的計算類 ---

class RSI(BaseAlpha):
    """相對強弱指數 (Relative Strength Index)"""
    def __init__(self, window=14):
        self.window = window
    def predict(self) -> xr.DataArray:
        delta_val = delta(self.close, 1)
        gain = where(delta_val > 0, delta_val, 0)
        loss = -where(delta_val < 0, delta_val, 0)
        avg_gain = ts_mean(gain, self.window)
        avg_loss = ts_mean(loss, self.window)
        safe_avg_loss = where(avg_loss == 0, 1e-9, avg_loss)
        rs = avg_gain / safe_avg_loss
        rsi = 100 - (100 / (1 + rs))
        return -rsi # 通常作為反轉指標，取負號

class StochasticK(BaseAlpha):
    """隨機指標K值 (Stochastic Oscillator %K)"""
    def __init__(self, window=14):
        self.window = window
    def predict(self) -> xr.DataArray:
        low_roll = ts_rolling(self.low, self.window).min()
        high_roll = ts_rolling(self.high, self.window).max()
        divisor = high_roll - low_roll
        safe_divisor = where(divisor == 0, 1e-9, divisor)
        k_percent = 100 * ((self.close - low_roll) / safe_divisor)
        return -k_percent # 通常作為反轉指標，取負號

class CMO(BaseAlpha):
    """錢德動量擺盪指標 (Chande Momentum Oscillator)"""
    def __init__(self, window=20):
        self.window = window
    def predict(self) -> xr.DataArray:
        momentum = delta(self.close, 1)
        up_sum = ts_sum(where(momentum > 0, momentum, 0), self.window)
        down_sum = ts_sum(abs(where(momentum < 0, momentum, 0)), self.window)
        divisor = up_sum + down_sum
        safe_divisor = where(divisor == 0, 1e-9, divisor)
        cmo = 100 * (up_sum - down_sum) / safe_divisor
        return -cmo

class DV_Divergence(BaseAlpha):
    """價量背離 (20日窗口)"""
    def __init__(self, price_window=20, corr_window=20):
        self.price_window = price_window
        self.corr_window = corr_window
    def predict(self) -> xr.DataArray:
        price_pct = pct_change(self.close, self.price_window)
        volume_pct = pct_change(self.volume, self.price_window)
        # 由於ts_corr是基於xarray的，這裡直接使用它
        # 假設 ts_corr 函數已在您的因子庫中定義
        from __main__ import ts_corr 
        return ts_corr(price_pct, volume_pct, self.corr_window)
    
# --- 3.1 中信期貨報告因子 (CITIC Futures Report Factors) ---
class LogCap(BaseAlpha):
    """因子: logcap (市值對數)"""
    def predict(self) -> xr.DataArray:
        return np.log(self.market_cap)

class TurnoverMean(BaseAlpha):
    """因子: turn_Nd (最近N日平均換手率)"""
    def __init__(self, window=21): self.window = window
    def predict(self) -> xr.DataArray: return ts_mean(self.turnover, self.window)

class TurnoverStd(BaseAlpha):
    """因子: std_turn_Nd (最近N日換手率標準差)"""
    def __init__(self, window=21): self.window = window
    def predict(self) -> xr.DataArray: return ts_std(self.turnover, self.window)

class VolumeMean(BaseAlpha):
    """因子: vol_Nm (最近N個月成交量均值)"""
    def __init__(self, window=21): self.window = window
    def predict(self) -> xr.DataArray: return ts_mean(self.volume, self.window)

class VolumeStd(BaseAlpha):
    """因子: std_vol_Nm (最近N個月成交量標準差)"""
    def __init__(self, window=21): self.window = window
    def predict(self) -> xr.DataArray: return ts_std(self.volume, self.window)

class VolumeChangeRatio(BaseAlpha):
    """因子: vol_change_ratio (成交量變動速率)"""
    def __init__(self, window=20): self.window = window
    def predict(self) -> xr.DataArray: return pct_change(self.volume, self.window)

class AmountMean(BaseAlpha):
    """因子: amount_Nm (最近N個月成交額均值)"""
    def __init__(self, window=21): self.window = window
    def predict(self) -> xr.DataArray: return ts_mean(self.amount, self.window)

class AmountMean_1(BaseAlpha):
    """因子: amount_Nm (最近N個月成交額均值)"""
    def __init__(self, window=21): self.window = window
    def predict(self) -> xr.DataArray: return -ts_mean(self.amount, self.window)

class AmountStd(BaseAlpha):
    """因子: std_amount_Nm (最近N個月成交額標準差)"""
    def __init__(self, window=21): self.window = window
    def predict(self) -> xr.DataArray: return ts_std(self.amount, self.window)

class AmountVolatility(BaseAlpha):
    """因子: a_volatility_Nm (成交額波動係數)"""
    def __init__(self, window=21): self.window = window
    def predict(self) -> xr.DataArray:
        mean_amount = ts_mean(self.amount, self.window)
        std_amount = ts_std(self.amount, self.window)
        return std_amount / where(mean_amount == 0, np.nan, mean_amount)

class AmihudIlliquidity(BaseAlpha):
    """因子: illiquidity (Amihud 非流動性因子)"""
    def __init__(self, window=21): self.window = window
    def predict(self) -> xr.DataArray:
        safe_amount = where(self.amount <= 0, np.nan, self.amount)
        illiq_ratio = abs(self.returns) / safe_amount
        return ts_mean(illiq_ratio, self.window)

class Momentum(BaseAlpha):
    """因子: mom_Nm (最近N天月收益率)"""
    def __init__(self, window=21): self.window = window
    def predict(self) -> xr.DataArray: return pct_change(self.close, self.window)

class MomentumAcceleration(BaseAlpha):
    """因子: mom_acc (動量加速度)"""
    def __init__(self, period1=126, period2=252): self.period1, self.period2 = period1, period2
    def predict(self) -> xr.DataArray:
        recent_return = pct_change(self.close, self.period1)
        prior_close = delay(self.close, self.period1)
        prior_prior_close = delay(self.close, self.period2)
        prior_return = prior_close / where(prior_prior_close == 0, np.nan, prior_prior_close) - 1
        return recent_return - prior_return

# --- 3.2 華泰證券/期貨報告因子 (HT Securities/Futures Report Factors) ---
class HT_Momentum_DailyMean(BaseAlpha):
    """計算滾動歷史日均收益率"""
    def __init__(self, window=60): self.window = window
    def predict(self) -> xr.DataArray: return ts_mean(self.returns, self.window)

class HT_Momentum_TotalReturn(BaseAlpha):
    """計算過去N個交易日的總體收益率"""
    def __init__(self, window=22): self.window = window
    def predict(self) -> xr.DataArray: return pct_change(self.close, self.window)

class HT_Momentum_TurnoverWeighted(BaseAlpha):
    """換手率加權動量"""
    def __init__(self, window=22): self.window = window
    def predict(self) -> xr.DataArray:
        weighted_returns_sum = ts_rolling((self.returns * self.turnover), self.window).sum()
        turnover_sum = ts_rolling(self.turnover, self.window).sum()
        return weighted_returns_sum / where(turnover_sum == 0, np.nan, turnover_sum)

class HT_Momentum_ExpWeighted(BaseAlpha):
    """指數衰減換手率加權動量"""
    def __init__(self, window=22): self.window = window
    def predict(self) -> xr.DataArray:
        days_ago = np.arange(self.window)
        decay_weights = np.exp(-days_ago / (self.window / 4))
        decay_weights = decay_weights[::-1]
        returns_win = self.returns.rolling(datetime=self.window, min_periods=int(self.window * 0.8)).construct("window_dim")
        turnover_win = self.turnover.rolling(datetime=self.window, min_periods=int(self.window * 0.8)).construct("window_dim")
        final_weights = turnover_win * decay_weights
        numerator = (returns_win * final_weights).sum(dim="window_dim", skipna=True)
        denominator = final_weights.sum(dim="window_dim", skipna=True)
        return numerator / where(denominator == 0, np.nan, denominator)

class HT_Turnover_Mean(BaseAlpha):
    """日均換手率"""
    def __init__(self, window=22): self.window = window
    def predict(self) -> xr.DataArray: return ts_mean(self.turnover, self.window)

class HT_Turnover_Std(BaseAlpha):
    """換手率波動率"""
    def __init__(self, window=22): self.window = window
    def predict(self) -> xr.DataArray: return ts_std(self.turnover, self.window)

class HT_Turnover_Bias(BaseAlpha):
    """換手率乖離率"""
    def __init__(self, short_window=22, long_window=500): self.short_window, self.long_window = short_window, long_window
    def predict(self) -> xr.DataArray:
        short_mean = ts_mean(self.turnover, self.short_window)
        long_mean = ts_mean(self.turnover, self.long_window)
        return short_mean / where(long_mean == 0, np.nan, long_mean) - 1

class HT_Turnover_Std_Bias(BaseAlpha):
    """換手率波動乖離率"""
    def __init__(self, short_window=22, long_window=500): self.short_window, self.long_window = short_window, long_window
    def predict(self) -> xr.DataArray:
        short_std = ts_std(self.turnover, self.short_window)
        long_std = ts_std(self.turnover, self.long_window)
        return short_std / where(long_std == 0, np.nan, long_std) - 1

class HT_Volatility(BaseAlpha):
    """日收益率的滾動標準差"""
    def __init__(self, window=60): self.window = window
    def predict(self) -> xr.DataArray: return ts_std(self.returns, self.window)

class HT_Skewness(BaseAlpha):
    """日收益率的滾動偏度"""
    def __init__(self, window=120): self.window = window
    def predict(self) -> xr.DataArray: return ts_skew(self.returns, self.window)

class HT_Kurtosis(BaseAlpha):
    """日收益率的滾動超額峰度"""
    def __init__(self, window=120): self.window = window
    def predict(self) -> xr.DataArray: return ts_kurt(self.returns, self.window)

class HT_TermStructure_RollYield_Proxy(BaseAlpha):
    """展期收益 (Roll Yield) 的代理指標"""
    def __init__(self, window=60): self.window = window
    def predict(self) -> xr.DataArray:
        roll_yield_proxy = self.close / self.open - 1
        return ts_mean(roll_yield_proxy, self.window)

# --- 3.3 國泰君安 101 & 191 Alpha 因子 (GTJA 101 & 191 Alpha Factors) ---
class Alpha001(BaseAlpha):
    def __init__(self, std_period=20, arg_period=5): self.std_period, self.arg_period = std_period, arg_period
    def predict(self):
        inner = where(self.returns < 0, ts_std(self.returns, self.std_period), self.close)
        signed_power_inner = sign(inner) * (abs(inner)**2)
        return rank(ts_argmax(signed_power_inner, self.arg_period)) - 0.5
# class Alpha002(BaseAlpha):
#     def __init__(self, delta_period=2, corr_period=6): self.delta_period, self.corr_period = delta_period, corr_period
#     def predict(self): return -1 * ts_corr(rank(delta(log(self.volume), self.delta_period)), rank((self.close - self.open) / self.open), self.corr_period)
# class Alpha003(BaseAlpha):
#     def __init__(self, corr_period=10): self.corr_period = corr_period
#     def predict(self): return -1 * ts_corr(rank(self.open), rank(self.volume), self.corr_period)
# class Alpha004(BaseAlpha):
#     def __init__(self, rank_period=9): self.rank_period = rank_period
#     def predict(self): return -1 * ts_rank(rank(self.low), self.rank_period)
# class Alpha005(BaseAlpha):
#     def __init__(self, rank_period=10): self.rank_period = rank_period
#     def predict(self): return rank(self.open - ts_mean(self.vwap, self.rank_period)) * (-1 * abs(rank(self.close - self.vwap)))
class Alpha006(BaseAlpha):
    def __init__(self, corr_period=10): self.corr_period = corr_period
    def predict(self): return -1 * ts_corr(self.open, self.volume, self.corr_period)
class Alpha007(BaseAlpha):
    def __init__(self, delta_period=7, vol_mean_period=20, rank_period=60): self.delta_period, self.vol_mean_period, self.rank_period = delta_period, vol_mean_period, rank_period
    def predict(self):
        adv20 = ts_mean(self.volume, self.vol_mean_period)
        close_delta = delta(self.close, self.delta_period)
        factor = -1 * ts_rank(abs(close_delta), self.rank_period) * sign(close_delta)
        return where(adv20 < self.volume, factor, -1.0)
class Alpha012(BaseAlpha):
    def __init__(self, vol_p=1, close_p=1): self.vol_p, self.close_p = vol_p, close_p
    def predict(self): return sign(delta(self.volume, self.vol_p)) * (-1 * delta(self.close, self.close_p))
class Alpha013(BaseAlpha):
    def __init__(self, cov_period=5): self.cov_period = cov_period
    def predict(self): return -1 * rank(ts_cov(rank(self.close), rank(self.volume), self.cov_period))
class Alpha018(BaseAlpha):
    def __init__(self, std_period=5, corr_period=10): self.std_period, self.corr_period = std_period, corr_period
    def predict(self):
        df = ts_corr(self.close, self.open, self.corr_period)
        return -1 * rank(ts_std(abs(self.close - self.open), self.std_period) + (self.close - self.open) + df)
class Alpha023(BaseAlpha):
    def __init__(self, ts_period=20, delta_period=2): self.ts_period, self.delta_period = ts_period, delta_period
    def predict(self):
        cond = ts_mean(self.high, self.ts_period) < self.high
        return where(cond, -1 * delta(self.high, self.delta_period), 0)
class Alpha033(BaseAlpha):
    def predict(self): return rank(-1 + (self.open / self.close))
class Alpha040(BaseAlpha):
    def __init__(self, ts_period=10): self.ts_period = ts_period
    def predict(self): return -1 * rank(ts_std(self.high, self.ts_period)) * ts_corr(self.high, self.volume, self.ts_period)
class Alpha041(BaseAlpha):
    def predict(self): return pow(self.high * self.low, 0.5) - self.vwap
class Alpha042(BaseAlpha):
    def predict(self): return rank(self.vwap - self.close) / rank(self.vwap + self.close)
class Alpha053(BaseAlpha):
    def __init__(self, ts_period=9): self.ts_period = ts_period
    def predict(self):
        inner = where(self.close - self.low == 0, 1e-9, self.close - self.low)
        return -1 * delta(((self.close - self.low) - (self.high - self.close)) / inner, self.ts_period)
class Alpha054(BaseAlpha):
    def __init__(self, power=5): self.power = power
    def predict(self):
        inner = where(self.low - self.high == 0, 1e-9, self.low - self.high)
        return -1 * (self.low - self.close) * (self.open ** self.power) / (inner * (self.close ** self.power))
class Alpha060(BaseAlpha):
    def __init__(self, ts_period=10): self.ts_period = ts_period
    def predict(self):
        divisor = where(self.high - self.low == 0, 1e-9, self.high - self.low)
        inner = ((self.close - self.low) - (self.high - self.close)) * self.volume / divisor
        return -((2 * rank(inner)) - rank(ts_argmax(self.close, self.ts_period)))
class Alpha101(BaseAlpha):
    def predict(self): return (self.close - self.open) / where(self.high - self.low == 0, 1e-9, self.high - self.low)
class Alpha191_014(BaseAlpha):
    def __init__(self, delay_period=5): self.delay_period = delay_period
    def predict(self): return self.close - delay(self.close, self.delay_period)
class Alpha191_018(BaseAlpha):
    def __init__(self, delay_period=5): self.delay_period = delay_period
    def predict(self): return self.close / delay(self.close, self.delay_period)
class Alpha191_020(BaseAlpha):
    def __init__(self, delay_period=6): self.delay_period = delay_period
    def predict(self): return (self.close - delay(self.close, self.delay_period)) / delay(self.close, self.delay_period) * 100

# --- 3.4 Falkenblog 因子 (Falkenblog Factors) ---

class Falkenblog_Disparity_Mean(BaseAlpha):
    """隔夜與日內收益率差異的滾動平均"""
    def __init__(self, window=20): self.window = window
    def predict(self) -> xr.DataArray:
        prev_close = delay(self.close, 1)
        safe_prev_close = where(prev_close == 0, np.nan, prev_close)
        overnight_ret = self.open / safe_prev_close - 1
        safe_open = where(self.open == 0, np.nan, self.open)
        intraday_ret = self.close / safe_open - 1
        disparity = overnight_ret - intraday_ret
        return ts_mean(disparity, self.window)

class Falkenblog_Disparity_Std(BaseAlpha):
    """隔夜與日內收益率差異的滾動標準差"""
    def __init__(self, window=20): self.window = window
    def predict(self) -> xr.DataArray:
        prev_close = delay(self.close, 1)
        safe_prev_close = where(prev_close == 0, np.nan, prev_close)
        overnight_ret = self.open / safe_prev_close - 1
        safe_open = where(self.open == 0, np.nan, self.open)
        intraday_ret = self.close / safe_open - 1
        disparity = overnight_ret - intraday_ret
        return ts_std(disparity, self.window)

# --- 3.5 Eric 因子 (Eric's Custom Factors) ---
class EricFactor(BaseAlpha):
    """
    因子: eric_Nd (量價雙低因子)
    邏輯: - (價格變異係數 * 對數成交量標準差)
    - 價格變異係數 (PVOL) = STD(Close) / SMA(Close)
    - 對數成交量標準差 (VVol) = STD(log(Volume + 1))
    """
    def __init__(self, window=20):
        self.window = window
        # 使用一個小的正常數 epsilon 來避免 log(0)
        self.epsilon = 1.0

    def predict(self) -> xr.DataArray:
        # 1. 計算價格波動率 (Price Volatility - PVOL)
        # 使用變異係數 (Coefficient of Variation) 來實現歸一化，使其尺度無關
        close_mean = ts_mean(self.close, self.window)
        close_std = ts_std(self.close, self.window)
        
        # 避免除以零
        safe_close_mean = where(close_mean == 0, np.nan, close_mean)
        pvol = close_std / safe_close_mean

        # 2. 計算成交量波動率 (Volume Volatility - VVol)
        # 先對成交量取對數，弱化極端值影響
        log_volume = log(self.volume + self.epsilon)
        
        # 計算對數成交量的滾動標準差
        vvol = ts_std(log_volume, self.window)

        # 3. 合成因子並取負號，使得量價雙低的值更優（更小的負數）
        factor = -1 * pvol * vvol
        
        return factor

# --- 3.6 Eric 多維度反轉因子 ---
class EricMultiDimReversal(BaseAlpha):
    """
    因子: eric_multi_reversal_20d (多維度反轉因子)
    邏輯: -1 * RANK(CLOSE / SMA(CLOSE,20)) * RANK(SMA(VOLUME,20)) * RANK(SMA(TURNOVER,20))
    """
    def __init__(self, window=20):
        self.window = window

    def predict(self) -> xr.DataArray:
        # 1. 計算各維度的基礎值
        # 價格相對強度
        close_ma = ts_mean(self.close, self.window)
        safe_close_ma = where(close_ma == 0, np.nan, close_ma)
        price_strength = self.close / safe_close_ma

        # 成交量強度 (20日均量)
        volume_strength = ts_mean(self.volume, self.window)

        # 換手率強度 (20日均換手率)
        turnover_strength = ts_mean(self.turnover, self.window)

        # 2. 對每個維度進行截面排名 (Rank)
        rank_price = rank(price_strength)
        rank_volume = rank(volume_strength)
        rank_turnover = rank(turnover_strength)

        # 3. 合成最終因子
        # 三個排名相乘，並乘以 -1 實現反轉邏輯
        factor = -1 * rank_price * rank_volume * rank_turnover
        
        return factor

class EricMultiDimReversalV2(BaseAlpha):
    """
    因子: eric_multi_reversal_v2 (多維度反轉因子 - 標準化版)
    邏輯: 通過計算各維度近期相對於其自身長期的偏離程度，
          使得不同規模的ETF可以在同一個基準上進行公平比較。
    公式: -1 * RANK(價_偏離度) * RANK(量_偏離度) * RANK(換手率_偏離度)
    """
    def __init__(self, short_window=20, long_window=120):
        self.short_window = short_window
        self.long_window = long_window

    def predict(self) -> xr.DataArray:
        # 1. 價格強度 (近期價格 vs. 近期均線) - 邏輯不變
        close_ma_short = ts_mean(self.close, self.short_window)
        safe_close_ma = where(close_ma_short == 0, np.nan, close_ma_short)
        price_strength = self.close / safe_close_ma

        # 2. 成交量強度 (近期均量 vs. 長期均量) - 核心修正
        volume_ma_short = ts_mean(self.volume, self.short_window)
        volume_ma_long = ts_mean(self.volume, self.long_window)
        safe_volume_ma_long = where(volume_ma_long == 0, np.nan, volume_ma_long)
        volume_strength_ratio = volume_ma_short / safe_volume_ma_long

        # 3. 換手率強度 (近期均換手率 vs. 長期均換手率) - 核心修正
        turnover_ma_short = ts_mean(self.turnover, self.short_window)
        turnover_ma_long = ts_mean(self.turnover, self.long_window)
        safe_turnover_ma_long = where(turnover_ma_long == 0, np.nan, turnover_ma_long)
        turnover_strength_ratio = turnover_ma_short / safe_turnover_ma_long

        # 4. 對每個標準化後的強度指標進行截面排名
        rank_price = rank(price_strength)
        rank_volume = rank(volume_strength_ratio)
        rank_turnover = rank(turnover_strength_ratio)

        # 5. 合成最終因子
        factor = -1 * rank_price * rank_volume * rank_turnover
        
        return factor
    
# [--- 在此處新增以下新的簡化版因子類 ---]
class Eric_ADX_Weighted_Momentum(BaseAlpha):
    """
    簡化版趨勢動量因子：僅使用 ADX 作為權重來調節動量信號。
    邏輯: Momentum * (ADX / 100)
    - Momentum: 基礎動量信號 (N日收益率)
    - ADX/100: 作為趨勢強度的連續權重。ADX低則削弱信號，ADX高則增強信號。
    """
    def __init__(self, momentum_window=22, adx_window=14):
        self.momentum_window = momentum_window
        self.adx_window = adx_window

    def predict(self) -> xr.DataArray:
        # 1. 計算基礎動量信號
        momentum = pct_change(self.close, self.momentum_window)

        # 2. 計算 ADX (我們只需要 ADX 值，所以忽略返回的 +DI 和 -DI)
        # _ (下劃線) 是一個常用的占位符，表示我們不關心這個返回值
        adx, _, _ = calc_adx(self.high, self.low, self.close, self.adx_window)

        # 3. 計算趨勢強度權重
        # 直接將 ADX / 100 作為權重。使用 fillna(0) 處理計算初期的 NaN 值。
        trend_strength = (adx / 100.0).fillna(0)

        # 4. 合成最終信號
        # 最終信號 = 基礎動量 * 趨勢強度權重
        signal = momentum * trend_strength
        
        return signal

# [--- 在此處新增以下新的 ADX 換手率加權動量因子類 ---]
class Eric_ADX_Turnover_Weighted_Momentum(BaseAlpha):
    """
    結合 ADX 趨勢強度的換手率加權動量因子。
    邏輯: TurnoverWeightedMomentum * (ADX / 100)
    - TurnoverWeightedMomentum: 換手率加權的滾動收益率，信號偏向於高成交活躍度的動量。
    - ADX/100: 作為趨勢強度的權重，用於放大趨勢行情中的信號，削弱震盪行情中的信號。
    """
    def __init__(self, momentum_window=22, adx_window=14):
        self.momentum_window = momentum_window
        self.adx_window = adx_window

    def predict(self) -> xr.DataArray:
        # 1. 計算基礎的「換手率加權動量」
        # 計算分子: sum(returns * turnover)
        weighted_returns_sum = ts_rolling((self.returns * self.turnover), self.momentum_window).sum()
        
        # 計算分母: sum(turnover)
        turnover_sum = ts_rolling(self.turnover, self.momentum_window).sum()
        
        # 得到基礎動量信號
        turnover_weighted_momentum = weighted_returns_sum / where(turnover_sum == 0, np.nan, turnover_sum)

        # 2. 計算 ADX 趨勢強度
        adx, _, _ = calc_adx(self.high, self.low, self.close, self.adx_window)

        # 3. 計算趨勢強度權重
        trend_strength = (adx / 100.0).fillna(0)

        # 4. 合成最終信號
        # 最終信號 = 換手率加權動量 * 趨勢強度權重
        signal = turnover_weighted_momentum * trend_strength
        
        return signal

class Eric_Monthly_Avg_Overnight_Return(BaseAlpha):
    """
    月度平均隔夜收益
    將每日的隔夜收益信號在一個月度窗口內進行滾動平均，
    將高頻信號轉化為能反映近期持續市場預期的低頻信號。
    """
    def __init__(self, window=22): # 預設窗口為一個月（約22個交易日）
        self.window = window

    def predict(self) -> xr.DataArray:
        # 1. 計算每日的隔夜收益率
        prev_close = delay(self.close, 1)
        safe_prev_close = where(prev_close == 0, np.nan, prev_close)
        daily_overnight_return = self.open / safe_prev_close - 1

        # 2. 對每日信號進行滾動平均，得到月度信號
        monthly_avg_signal = ts_mean(daily_overnight_return, self.window)
        
        return monthly_avg_signal

class Eric_Monthly_Std_Overnight_Return(BaseAlpha):
    """
    月度隔夜收益波動率
    計算每日隔夜收益在一個月度窗口內的滾動標準差，
    用於衡量近期隔夜消息的穩定性或市場情緒的不確定性。
    """
    def __init__(self, window=22):
        self.window = window

    def predict(self) -> xr.DataArray:
        # 1. 計算每日的隔夜收益率
        prev_close = delay(self.close, 1)
        safe_prev_close = where(prev_close == 0, np.nan, prev_close)
        daily_overnight_return = self.open / safe_prev_close - 1

        # 2. 對每日信號計算滾動標準差
        monthly_std_signal = ts_std(daily_overnight_return, self.window)
        
        return monthly_std_signal

# [--- 在此處新增布林帶寬度因子類，以便後續調用 ---]
class Eric_Bollinger_Bandwidth(BaseAlpha):
    """
    標準布林帶寬度 (Bollinger Bandwidth) 因子。
    邏輯: (上軌 - 下軌) / 中軌
    可以通過 sign 參數輕鬆切換為“帶寬收縮”因子。
    """
    def __init__(self, window=20, std_dev=2.0, use_contraction=False):
        self.window = window
        self.std_dev = std_dev
        self.use_contraction = use_contraction # 是否使用收縮信號

    def predict(self) -> xr.DataArray:
        mean = ts_mean(self.close, self.window)
        std = ts_std(self.close, self.window)
        
        upper_band = mean + self.std_dev * std
        lower_band = mean - self.std_dev * std
        
        bandwidth = (upper_band - lower_band) / where(mean == 0, np.nan, mean)
        
        if self.use_contraction:
            # 取負值，使得帶寬越窄，因子值越高
            return -bandwidth
        else:
            return bandwidth


# [--- 在此處新增「動量 x 布林帶收縮」複合因子類 ---]
class Eric_Composite_Momentum_BBW(BaseAlpha):
    """
    動量 x 布林帶收縮複合因子 (尋找趨勢延續點)
    結合了ADX換手率加權動量（趨勢強度）和布林帶寬度收縮（盤整信號）。
    旨在捕捉強勢趨勢中，波動率降低、即將再次啟動的機會。
    """
    def __init__(self, momentum_window=44, adx_window=14, bb_window=22, mom_weight=0.5, bbw_weight=0.5):
        self.momentum_window = momentum_window
        self.adx_window = adx_window
        self.bb_window = bb_window
        self.mom_weight = mom_weight
        self.bbw_weight = bbw_weight
        
        # 內部實例化子因子
        self.momentum_factor = Eric_ADX_Turnover_Weighted_Momentum(
            momentum_window=self.momentum_window,
            adx_window=self.adx_window
        )
        # 關鍵：設置 use_contraction=True 來獲取“帶寬收縮”信號
        self.bbw_factor = Eric_Bollinger_Bandwidth(
            window=self.bb_window,
            use_contraction=True 
        )

    def predict(self) -> xr.DataArray:
        # 1. 獲取兩個子因子的原始信號
        self.momentum_factor.array_ = self.array_
        momentum_signal = self.momentum_factor.predict()
        
        self.bbw_factor.array_ = self.array_
        bbw_contraction_signal = self.bbw_factor.predict()

        # 2. 對每個子信號進行橫截面排名，使其可比
        ranked_momentum = rank(momentum_signal, pct=True)
        ranked_bbw = rank(bbw_contraction_signal, pct=True)

        # 3. 將排名分數加權平均，得到最終複合因子分
        composite_signal = (self.mom_weight * ranked_momentum.fillna(0) + 
                            self.bbw_weight * ranked_bbw.fillna(0))
        
        return -composite_signal

# ... (你現有的所有因子類)

# [--- 在此處新增「非對稱波動率反轉」因子類 ---]
class Eric_Asymmetric_Volatility_Reversion(BaseAlpha):
    """
    非對稱波動率反轉因子
    核心表達式: (-1 * TS_MEAN(SIGNEDPOWER(CHANGE_PCT, 2), window))
    邏輯:
    1. SIGNEDPOWER: 對每日收益率進行帶符號的平方運算 (ret * abs(ret))，
       在保留方向的同時放大波動的幅度。
    2. TS_MEAN: 計算處理後序列的滾動平均值，衡量近期的平均非對稱波動水平。
    3. -1 *: 將結果取反，使得近期波動越劇烈的資產，因子值越低，從而捕捉反轉或低波動溢價。
    """
    def __init__(self, window=30):
        self.window = window

    def predict(self) -> xr.DataArray:
        # 1. 獲取每日收益率 (CHANGE_PCT)
        daily_returns = self.returns

        # 2. 計算帶符號平方轉換 (SIGNEDPOWER)
        # 這完全等價於 SIGNEDPOWER(daily_returns, 2)
        signed_power_returns = daily_returns * abs(daily_returns)

        # 3. 計算滾動均值 (TS_MEAN)
        mean_signed_volatility = ts_mean(signed_power_returns, self.window)
        
        # 4. 取反向 (-1 *)，得到最終因子信號
        final_signal = -1 * mean_signed_volatility
        
        return final_signal


# [--- 在此處新增 U 型波動率因子類 ---]
class Eric_U_Shape_Volatility_Factor(BaseAlpha):
    """
    U 型非對稱波動率因子 (獎勵兩端，懲罰中間)
    
    該因子基於一個觀察：非對稱波動率最高（強反轉預期）和最低（低波動溢價）
    的資產表現都很好，而中間水平的資產表現不佳。
    
    邏輯:
    1. 計算原始的“非對稱波動率反轉”因子。
    2. 對原始因子進行橫截面排名，得到 [0, 1] 區間的分數。
    3. 應用 U 型變換公式: (排名 - 0.5)^2，使得排名靠近兩端（0或1）的資產
       獲得高分，而排名在中間（0.5附近）的資產獲得低分。
    """
    def __init__(self, window=30):
        self.window = window
        # 內部實例化原始因子，用於計算基礎信號
        self.base_factor = Eric_Asymmetric_Volatility_Reversion(window=self.window)

    def predict(self) -> xr.DataArray:
        # 1. 獲取原始的“非對稱波動率反轉”因子信號
        self.base_factor.array_ = self.array_
        base_signal = self.base_factor.predict()

        # 2. 對原始信號進行橫截面排名
        ranked_signal = rank(base_signal, pct=True)

        # 3. 應用 U 型變換
        # 中心化 (減去0.5)，然後平方
        u_shape_signal = (ranked_signal - 0.5)**2
        
        return u_shape_signal

class Eric_Volatility_Turnover_Coupling(BaseAlpha):
    """
    波動率與換手率的負向耦合因子 (風險因子)
    核心表達式: SCALE(RANK(TS_STDDEV(CLOSE, window))) * -1 * TURN_RATE
    該因子旨在識別並懲罰市場中“高波動+高換手率”的過度投機股票。
    因子值越小（負得越多），代表該股票的投機風險越高。
    """
    def __init__(self, window=15):
        self.window = window

    def predict(self) -> xr.DataArray:
        # Step 1: 計算價格波動率 (TS_STDDEV)
        price_volatility = ts_std(self.close, self.window)

        # Step 2: 對波動率進行橫截面排序 (RANK)
        ranked_volatility = rank(price_volatility, pct=True)

        # Step 3: 對排序後的波動率進行標準化 (SCALE)
        scaled_ranked_volatility = scale(ranked_volatility)

        # Step 4: 與換手率進行負向耦合
        turnover_rate = self.turnover
        final_signal = scaled_ranked_volatility * -1 * turnover_rate
        
        return final_signal

# [--- 在此處新增「量價背離協動」因子類 ---]
class Eric_PV_Divergence_Covariance(BaseAlpha):
    """
    量價背離協動因子 (反轉型)
    核心表達式: TS_COVARIANCE(DELTA(VOLUME,1), DELTA(CLOSE, 1), window) * (-1)
    
    該因子通過計算價格日變動與成交量日變動的協方差，來捕捉量價背離信號。
    因子值越高，表明量價背離越嚴重，趨勢反轉的可能性越大。
    """
    def __init__(self, window=30):
        self.window = window

    def predict(self) -> xr.DataArray:
        # Step 1: 量價變動計算 (DELTA)
        delta_close = delta(self.close, 1)
        delta_volume = delta(self.volume, 1)

        # Step 2: 時序協方差計算 (TS_COVARIANCE)
        covariance = ts_cov(delta_close, delta_volume, self.window)
        
        # Step 3: 方向反轉 (* -1)
        final_signal = -1 * covariance
        
        return final_signal

# ... (你現有的所有因子類)

# [--- 在此處新增「截面量价残差波动率」因子類 ---]
class Eric_CS_Residual_Volatility(BaseAlpha):
    """
    截面量价残差波动率因子 (低風險/質量型)
    核心表達式: -TS_STDDEV(CS_REGRESSION(CLOSE, VOLUME), window)
    
    該因子衡量股價中無法被成交量解釋的“特異性”部分的波動率。
    因子值越高，代表該股票的量價關係越穩定，異質性風險越低。
    """
    def __init__(self, window=20):
        self.window = window

    def predict(self) -> xr.DataArray:
        # Step 1: 截面回歸提取殘差
        # 計算每日的收盤價對成交量的橫截面回歸殘差
        residuals = cs_regression(self.close, self.volume)

        # Step 2: 時序波動率計算
        # 計算殘差序列的滾動標準差
        residual_volatility = ts_std(residuals, self.window)
        
        # Step 3: 取負值，偏好低波動率
        final_signal = -1 * residual_volatility
        
        return final_signal

class Eric_Improved_Skewness(BaseAlpha):
    """
    因子: 基於偏度改進的經驗極端概率偏度因子
    理論核心: P(ret > mean + n*std) - P(ret < mean - n*std)
    實現方式: 在滾動窗口內，(收益超過上軌的天數 - 收益低於下軌的天數) / 窗口長度
    """
    def __init__(self, window=252, std_multiple=2.0):
        self.window = window
        self.std_multiple = std_multiple

    def predict(self) -> xr.DataArray:
        # 1. 獲取每日收益率 (self.returns 已經在 BaseAlpha 中定義好了)
        daily_returns = self.returns

        # 2. 計算滾動的均值和標準差 (使用您已有的輔助函數)
        rolling_mean = ts_mean(daily_returns, self.window)
        rolling_std = ts_std(daily_returns, self.window)

        # 3. 定義上下軌閾值
        upper_threshold = rolling_mean + self.std_multiple * rolling_std
        lower_threshold = rolling_mean - self.std_multiple * rolling_std

        # 4. 判斷每一天的收益是否超過了當天的動態閾值
        # 如果超過上軌，記為 1，否則為 0
        is_extreme_gain = xr.where(daily_returns > upper_threshold, 1, 0)
        # 如果低於下軌，記為 1，否則為 0
        is_extreme_loss = xr.where(daily_returns < lower_threshold, 1, 0)

        # 5. 在滾動窗口內，計算超過上軌和低於下軌的總天數
        gain_days = ts_sum(is_extreme_gain, self.window)
        loss_days = ts_sum(is_extreme_loss, self.window)
        
        # 6. 計算最終因子值
        factor_value = (gain_days - loss_days) / self.window

        # 7. 【核心修改】將結果取反，使其變為反轉信號
        #    現在，因子值越高，代表負偏度越嚴重，未來預期收益越高。
        return -1 * factor_value

# ... (Eric_Market_Neutral_Momentum class 的程式碼) ...

# [--- 在此處新增「改進版 Alpha 22 因子」類 ---]
class Eric_Improved_Alpha_022(BaseAlpha):
    """
    改進版 Alpha 22 因子 (Based on Improvement Idea 2)
    
    原始邏輯 (改進1): -1 * delta(covariance(high, volume, 5), 5) * rank(stddev(close, 20))
    最終邏輯 (改進2): -1 * abs( delta(covariance(high, volume, 5), 5) * rank(stddev(close, 20)) )
    
    這個因子旨在將一個具有對稱性收益的信號，通過取絕對值的方式，轉化為單調信號。
    """
    def __init__(self):
        # 這個因子的窗口期是固定的，所以不需要傳入參數
        super().__init__()

    def predict(self) -> xr.DataArray:
        # 步驟 1: 計算最高價與成交量在5日窗口內的協方差
        # covariance(self.high, self.volume, 5)
        cov_high_vol = ts_cov(self.high, self.volume, window=5)
        
        # 步驟 2: 計算該協方差的5日差分
        # delta(df, 5)
        delta_cov = delta(cov_high_vol, period=5)
        
        # 步驟 3: 計算收盤價20日標準差的截面排序
        # rank(stddev(self.close, 20))
        rank_std_close = rank(ts_std(self.close, window=20))
        
        # 步驟 4: 組合信號並應用最終的改進邏輯
        # 核心信號 = 差分 * 排序
        core_signal = delta_cov * rank_std_close
        
        # 應用 abs() 和 (-1) *
        final_signal = -1 * abs(core_signal)
        
        return final_signal

class EricRSRS(BaseAlpha):
    """
    阻力支撐相對強度 (RSRS) 因子
    源自光大證券金工研報，通過對過去N日 High-Low 序列進行線性迴歸，
    判斷買賣雙方的力量對比。
    
    此版本實現了研報中提到的兩種形式：
    1. 原始斜率 (slope)
    2. 斜率 * R-squared (slope_r2)
    """
    def __init__(self, window=18, use_r2_adjusted=False):
        """
        Args:
            window (int): 迴歸窗口期，研報建議值為 18。
            use_r2_adjusted (bool): 是否使用 R-squared 進行修正。
        """
        self.window = window
        self.use_r2_adjusted = use_r2_adjusted
    
    def predict(self) -> xr.DataArray:
        # 獲取 High 和 Low 序列
        high_prices = self.high
        low_prices = self.low
        
        # 調用新的雙變量迴歸函數
        slope, r_squared = ts_regression_bivariate(high_prices, low_prices, self.window)
        
        if self.use_r2_adjusted:
            # 返回 R-squared 修正後的版本
            return slope * r_squared
        else:
            # 返回原始的斜率版本
            return slope


# [--- 在此處新增以下基於華泰期貨研報的新因子類 ---]

class HT_Asymmetric_Volatility_Momentum(BaseAlpha):
    """
    因子: eric_asymmetric_vol_momentum (基於華泰期貨研報的非對稱波動動量)
    
    核心邏輯: 結合中期動量方向與波動率變化趨勢，捕捉“正/負波動率週期”的特徵。
    因子表達式: sign(pct_change(close, mom_window)) * delta(ts_std(returns, vol_window), delta_window)
    
    - 當價格上漲且波動率上升時，因子值為正，對應“正波動率週期”。
    - 當價格下跌且波動率上升時，因子值為負，對應“負波動率週期”。
    """
    def __init__(self, mom_window=90, vol_window=90, delta_window=20):
        """
        Args:
            mom_window (int): 計算動量的時間窗口。
            vol_window (int): 計算波動率的時間窗口。
            delta_window (int): 計算波動率變化的時間窗口。
        """
        self.mom_window = mom_window
        self.vol_window = vol_window
        self.delta_window = delta_window

    def predict(self) -> xr.DataArray:
        # 1. 計算中期價格動量方向
        momentum = pct_change(self.close, self.mom_window)
        momentum_direction = sign(momentum)
        
        # 2. 計算波動率
        volatility = ts_std(self.returns, self.vol_window)
        
        # 3. 計算波動率的近期變化
        volatility_change = delta(volatility, self.delta_window)
        
        # 4. 組合因子
        # 動量方向 * 波動率變化
        final_signal = momentum_direction * volatility_change
        
        return final_signal

# --- 3.8 遺傳規劃因子 (Genetic Programming Factors) ---
# 注意: 為了不修改原有邏輯，GP因子類會自行實例化其依賴的子因子類
# 對於非類別的因子，其計算邏輯會在此處被直接複製使用
class GP_New_Alpha_1(BaseAlpha):
    """
    GP因子: if(alpha_001, sqrt(amount_6m), proxy_large_order_inflow_1M)
    這是最新一輪GP運行收斂後的唯一因子。
    """
    def predict(self) -> xr.DataArray:
        # 實例化子因子計算器並獲取結果
        f_alpha_001_calc = Alpha001()
        f_alpha_001_calc.array_ = self.array_
        f1_cond = f_alpha_001_calc.predict()

        f_amount_6m_calc = AmountMean(window=126) # amount_6m 是 126 天的均值
        f_amount_6m_calc.array_ = self.array_
        f2_true_val = f_amount_6m_calc.predict()

        # 複製 'proxy_large_order_inflow_1M' 的計算邏輯
        f3_false_val = ts_mean(((self.close - (self.low + self.high) / 2) * self.volume), 20)

        # 組合表達式
        return where(f1_cond, gp_sqrt(f2_true_val), f3_false_val)

# --- 2025年8月27日新增的決策樹因子 ---
class GP_New_Alpha_2(BaseAlpha):
    """GP因子 Top-1: 複雜的決策樹結構"""
    def predict(self) -> xr.DataArray:
        def get_pred(factor_class, params={}):
            calc = factor_class(**params); calc.array_ = self.array_; return calc.predict()
        
        f_a001 = get_pred(Alpha001)
        f_vol_turn_15d = get_pred(Eric_Volatility_Turnover_Coupling, {'window': 15})
        f_cs_resid_40d = get_pred(Eric_CS_Residual_Volatility, {'window': 40})
        f_amt_vol_3m = get_pred(AmountVolatility, {'window': 63})
        
        # <--- 修正點: 直接複製計算邏輯，而不是調用不存在的類 --->
        f_proxy_1M = ts_mean(((self.close - (self.low + self.high) / 2) * self.volume), 20)

        term1 = where(f_vol_turn_15d, f_a001, f_proxy_1M)
        term2 = where(term1, f_cs_resid_40d, gp_max(f_proxy_1M, f_amt_vol_3m))
        term3 = where(f_a001, term2, gp_abs(f_cs_resid_40d))
        term4 = where(term3, f_cs_resid_40d, gp_max(f_proxy_1M, f_amt_vol_3m))
        term5 = where(f_a001, term4, gp_abs(f_cs_resid_40d))
        return where(term5, gp_abs(f_cs_resid_40d), f_proxy_1M)

class EricStableMomentumBurst(BaseAlpha):
    """
    穩健動量爆發因子 (Stable Momentum Burst)
    源自 WorldQuant Alpha 104 的改進思路。
    邏輯: Rank(短期動量) * Rank(近期穩定性) * 成交量爆發
    旨在尋找近期走勢穩健（低回撤）、且最近有放量加速上漲跡象的股票。
    """
    def __init__(self, short_window=3, long_window=30, vol_short=5, vol_long=20):
        self.short_window = short_window
        self.long_window = long_window
        self.vol_short = vol_short
        self.vol_long = vol_long

    def predict(self) -> xr.DataArray:
        # 1. 短期動量: 過去 short_window 日的收益率
        close_ret_short = pct_change(self.close, self.short_window)
        
        # 2. 近期穩定性: 過去 long_window 日的最大回撤
        # 我們希望回撤小，回撤值是一個負數（例如-0.2），越小（-0.1 > -0.2）越好，所以直接排名即可
        max_dd = ts_max_drawdown(self.close, self.long_window)
        
        # 3. 成交量爆發: 短期成交量 / 長期成交量
        vol_burst = ts_sum(self.volume, self.vol_short) / ts_sum(self.volume, self.vol_long)
        
        # 4. 組合三個信號
        # 對動量和穩定性信號進行排名
        rank_mom = rank(close_ret_short)
        rank_stability = rank(max_dd) # 回撤值本身是負的，-0.1 > -0.2，所以rank值越高代表回撤越小
        
        final_factor = rank_mom * rank_stability * vol_burst
        
        return final_factor

class EricEnhancedMomentum(BaseAlpha):
    """
    增強型動量因子 (Enhanced Momentum)
    
    這是一個複合因子，旨在尋找最優質的動量機會。
    
    組合邏輯:
    1. 核心動量: eric_stable_momentum_burst (近期穩定且放量上漲)。
    2. 質量過濾: eric_cs_resid_vol_40d (量價關係穩定，異質性風險低)。
    3. 風險規避: -mom_12m (規避已在長週期末端的極端動量股，防止動量崩盤)。
    """
    def __init__(self):
        # 內部實例化所有依賴的子因子
        self.factor_burst = EricStableMomentumBurst(short_window=3, long_window=30)
        self.factor_quality = Eric_CS_Residual_Volatility(window=40)
        self.factor_long_mom = Momentum(window=252) # 12個月約252個交易日

    def predict(self) -> xr.DataArray:
            # 1. 只计算我们需要的两个子因子
            self.factor_burst.array_ = self.array_
            signal_burst = self.factor_burst.predict()
            
            self.factor_quality.array_ = self.array_
            signal_quality = self.factor_quality.predict()
            
            # 2. 对每个子因子的原始值进行横截面排名
            rank_burst = rank(signal_burst, pct=True)
            rank_quality = rank(signal_quality, pct=True)
            
            # <--- 逻辑修改：只组合两个正向信号，移除长周期反转项 --->
            composite_signal = rank_burst.fillna(0) + rank_quality.fillna(0)
            
            return composite_signal
     
class EricCCI_Trend_Combo(BaseAlpha):
    """
    CCI 趨勢複合因子 (CCI Trend Combo)
    
    結合了 CCI 指標和移動均線趨勢過濾，旨在捕捉上升趨勢中的買點。
    
    邏輯: Rank(CCI 值) + Rank(收盤價 / 移動均線)
    """
    def __init__(self, cci_window=20, ma_window=60):
        self.cci_window = cci_window
        self.ma_window = ma_window

    def predict(self) -> xr.DataArray:
        # 1. 計算 CCI 信號
        cci_signal = calc_cci(self.high, self.low, self.close, self.cci_window)
        
        # 2. 計算趨勢過濾信號
        moving_avg = ts_mean(self.close, self.ma_window)
        safe_moving_avg = where(moving_avg == 0, np.nan, moving_avg)
        trend_signal = self.close / safe_moving_avg
        
        # 3. 對兩個信號分別進行橫截面排名
        rank_cci = rank(cci_signal, pct=True)
        rank_trend = rank(trend_signal, pct=True)
        
        # 4. 將排名分數相加，得到最終的複合因子分
        composite_signal = rank_cci.fillna(0) + rank_trend.fillna(0)
        
        return composite_signal


# [--- 在此處新增基於華泰期貨研報的“低波動突破”因子類 ---]

class HT_Low_Volatility_Breakout(BaseAlpha):
    """
    因子: ht_low_vol_breakout (基於華泰期貨研報的低波動突破)
    
    核心邏輯: 尋找處於上升趨勢，但相對波動率非常低的資產，
    旨在捕捉穩定趨勢的啟動點。
    因子表達式: pct_change(close, mom_window) * (-scale(ts_std(returns, vol_window)))
    """
    def __init__(self, mom_window=90, vol_window=90):
        """
        Args:
            mom_window (int): 計算動量的時間窗口。
            vol_window (int): 計算波動率的時間窗口。
        """
        self.mom_window = mom_window
        self.vol_window = vol_window

    def predict(self) -> xr.DataArray:
        # 1. 計算中期價格動量
        momentum = pct_change(self.close, self.mom_window)
        
        # 2. 計算波動率
        volatility = ts_std(self.returns, self.vol_window)
        
        # 3. 計算波動率的橫截面標準分 (z-score)
        # scale() 函數會將截面上波動率越大的值變得越大
        scaled_vol = scale(volatility)
        
        # 4. 創建低波動信號
        # 我們想要波動率越低，信號越強，所以乘以 -1
        low_vol_signal = -scaled_vol
        
        # 5. 組合因子
        # 動量 * 低波動信號
        final_signal = momentum * low_vol_signal
        
        return final_signal
    
# ==============================================================================
# === 3.X. 新增趨勢分數因子 (New Trend Score Factor) ===
# ==============================================================================
class TrendScore(BaseAlpha):
    """
    趨勢分數因子 (Trend Score)
    結合了趨勢強度 (年化收益率) 和趨勢穩定性 (R-squared)。
    核心邏輯: Trend Score = Annualized Return * R^2
    """
    def __init__(self, window=25):
        """
        Args:
            window (int): 滾動回歸的窗口期，根據您的流程圖，預設為25。
        """
        self.window = window
        self.annual_trading_days = 250 # 年化交易日數

    def predict(self) -> xr.DataArray:
        # 1. 數據預處理：對收盤價取對數
        log_close = log(self.close)

        # 2. 滾動回歸計算
        # 使用 ts_regression_all_stats 高效地一次性計算出斜率和 R^2
        # y 是 log_close, x 是隱含的時間序列 [0, 1, ..., window-1]
        slope, r_squared, _ = ts_regression_all_stats(log_close, self.window)

        # 3. 計算年化收益率
        # 斜率 slope 代表每日的對數收益率，我們將其年化
        # Annualized Return = exp(slope * 250) - 1
        annualized_return = np.exp(slope * self.annual_trading_days) - 1
        
        # 4. 計算最終的趨勢分數
        # Trend Score = Annualized Return * R^2
        trend_score = annualized_return * r_squared
        
        return trend_score

# [--- 請使用下面这个升级版，替换原有的 HT_Trend_Exhaustion_Reversal 因子类 ---]

class HT_Trend_Exhaustion_Reversal(BaseAlpha):
    """
    因子: ht_trend_exhaustion_reversal (升級版-連續分數)
    
    核心邏輯: 捕捉高波動後，動量反轉且波動率從峰值回落的信號。
    輸出一個連續的“衰竭分數”，分數越低（負得越多），反轉信號越強。
    對於多頭策略，因子值越大越好。
    """
    def __init__(self, window=90, vol_delta_window=5):
        """
        Args:
            window (int): 計算動量和波動率的時間窗口。
            vol_delta_window (int): 計算波動率變化的短期窗口。
        """
        self.window = window
        self.vol_delta_window = vol_delta_window

    def predict(self) -> xr.DataArray:
        # 1. 計算動量和波動率
        momentum = pct_change(self.close, self.window)
        volatility = ts_std(self.returns, self.window)
        vol_delta = delta(volatility, self.vol_delta_window)

        # 2. 將三個條件轉換為連續的百分比排名分數 (0到1)
        # 分數越高，代表條件的“強度”越大
        score_high_vol = rank(volatility, pct=True)
        score_mom_negative = rank(-momentum, pct=True) # 下跌越厉害，分數越高
        score_vol_decreasing = rank(-vol_delta, pct=True) # 波動率下降越快，分數越高
        
        # 3. 將三個分數相加，得到綜合的“衰竭總分”
        exhaustion_score = score_high_vol + score_mom_negative + score_vol_decreasing
        
        # 4. 應用核心前提過濾
        # 只對那些動量為負且波動率下降的資產計算分數，其他安全資產的分數為0
        # 這樣做可以讓因子的關注點更集中
        filtered_score = xr.where((momentum < 0) & (vol_delta < 0), exhaustion_score, 0)
        
        # 5. 將最終分數乘以-1
        # 使得衰竭信號越強的資產，最終因子值越小（負得越多）
        # 這就與我們“值越大越好”的選股策略保持了一致
        final_signal = -1 * filtered_score
        
        return final_signal

# [--- 在此處新增基於華泰期貨研報的“偏度反轉”因子類 ---]

class HT_Skewness_Reversal(BaseAlpha):
    """
    因子: ht_skewness_reversal (基於華泰期貨研報的偏度反轉因子)

    核心邏輯: 根據研報，偏度具有均值回歸特性，低（负）偏度的資產未來收益更高。
    因此，因子值直接取為偏度值的負數。
    因子表達式: -1 * ts_skew(returns, window)
    """
    def __init__(self, window=120):
        """
        Args:
            window (int): 計算偏度的時間窗口，研報中使用了120天。
        """
        self.window = window

    def predict(self) -> xr.DataArray:
        # 計算滾動偏度
        skew = ts_skew(self.returns, self.window)
        
        # 乘以-1，使得負偏度的資產獲得更高的因子值
        final_signal = -1 * skew
        
        return final_signal

# [--- 在此處新增“波動率增強的期限結構動量”因子類 ---]

class HT_TS_Vol_Enhanced_Momentum(BaseAlpha):
    """
    因子: ht_ts_vol_enhanced_mom (基於華泰期貨研報的波動率增強的期限結構動量)

    核心邏輯: 這是一個複合因子，它使用期限結構作為主要信號源，
    但只在波動率較高的資產上應用該信號。
    因子表達式: where(rank(volatility) > threshold, term_structure_proxy, 0)
    """
    def __init__(self, ts_window=30, vol_window=30, vol_rank_threshold=0.5):
        """
        Args:
            ts_window (int): 計算期限結構代理指標的時間窗口。
            vol_window (int): 計算波動率的時間窗口。
            vol_rank_threshold (float): 波動率的百分比排名篩選閾值。
                                       例如0.5代表只考慮波動率排在後半段的資產。
        """
        self.ts_window = ts_window
        self.vol_window = vol_window
        self.vol_rank_threshold = vol_rank_threshold
        
        # 内部实例化子因子，方便调用
        self.ts_factor = HT_TermStructure_RollYield_Proxy(window=self.ts_window)
        self.vol_factor = HT_Volatility(window=self.vol_window)

    def predict(self) -> xr.DataArray:
        # 1. 計算基礎的期限結構信號
        self.ts_factor.array_ = self.array_
        term_structure_signal = self.ts_factor.predict()
        
        # 2. 計算波動率，並判斷是否為高波動率資產
        self.vol_factor.array_ = self.array_
        volatility = self.vol_factor.predict()
        is_high_vol = rank(volatility, pct=True) > self.vol_rank_threshold
        
        # 3. 組合因子
        # 只有在高波動率的資產上，才保留其期限結構信號，否則信號為0
        final_signal = xr.where(is_high_vol, term_structure_signal, 0)
        
        return final_signal

# [--- 在此處新增「概率趨勢 RSRS」因子類 ---]
class Probabilistic_Trend_RSRS(BaseAlpha):
    """
    概率趨勢 RSRS 因子 (信號 x 確認)
    
    核心邏輯: RSRS_Slope * Trend_Probability(ADX)
    旨在尋找那些買方力量強勁 (高RSRS斜率)，且這種強勢發生在
    一個明確的趨勢環境 (高ADX -> 高趨勢概率) 中的資產。
    
    因子值越高，代表一個越高質量的趨勢跟隨信號。
    """
    def __init__(self, rsrs_window=18, adx_window=14, logistic_midpoint=25.0, logistic_steepness=0.2):
        """
        Args:
            rsrs_window (int): RSRS 回歸窗口。
            adx_window (int): ADX 計算窗口。
            logistic_midpoint (float): ADX 概率轉換的S型曲線中點，通常設為趨勢閾值25。
            logistic_steepness (float): S型曲線的陡峭度，控制轉換的靈敏度。
        """
        self.rsrs_window = rsrs_window
        self.adx_window = adx_window
        self.x0 = logistic_midpoint
        self.k = logistic_steepness
        
        # 內部實例化 RSRS 因子
        self.rsrs_factor = EricRSRS(window=self.rsrs_window, use_r2_adjusted=False)

    def predict(self) -> xr.DataArray:
        # 1. 計算核心方向信號：RSRS 斜率
        self.rsrs_factor.array_ = self.array_
        rsrs_slope = self.rsrs_factor.predict()

        # 2. 計算趨勢確認信號：ADX 趨勢概率
        adx, _, _ = calc_adx(self.high, self.low, self.close, self.adx_window)
        
        # 應用 Logistic 映射將 ADX 轉換為 (0, 1) 之間的概率
        # Prob = 1 / (1 + exp(-k * (ADX - x0)))
        logit_input = self.k * (adx - self.x0)
        trend_probability = (1 / (1 + np.exp(-logit_input))).fillna(0)

        # 3. 合成最終因子
        # RSRS斜率 * 趨勢概率
        final_signal = rsrs_slope * trend_probability
        
        return final_signal

# [--- 在此處新增「R²加權的質量動量」因子類 ---]
class Quality_Momentum_R2(BaseAlpha):
    """
    R²加權的質量動量因子 (動量 x 穩定性)
    
    核心邏輯: Rank(Momentum) + Rank(R_squared)
    旨在尋找那些不僅中期回報高，且其趨勢路徑穩定、可預測性強
    (對數價格對時間的迴歸 R² 高) 的資產。
    
    因子值越高，代表一個越高質量的動量信號。
    """
    def __init__(self, momentum_window=126, regression_window=126):
        self.momentum_window = momentum_window
        self.regression_window = regression_window
        
        # 內部實例化動量因子
        self.momentum_factor = Momentum(window=self.momentum_window)

    def predict(self) -> xr.DataArray:
        # 1. 計算核心動量信號
        self.momentum_factor.array_ = self.array_
        momentum_signal = self.momentum_factor.predict()

        # 2. 計算趨勢質量信號：R-squared
        # 對對數價格進行迴歸，結果更穩健
        log_close = log(self.close)
        # 使用你已有的高效迴歸函數
        _, r_squared, _ = ts_regression_all_stats(log_close, self.regression_window)

        # 3. 對兩個信號分別進行橫截面排名
        ranked_momentum = rank(momentum_signal, pct=True)
        ranked_r2 = rank(r_squared, pct=True)

        # 4. 合成最終因子 (排名相加)
        final_signal = ranked_momentum.fillna(0) + ranked_r2.fillna(0)
        
        return final_signal
    
# [--- 在此處新增「ADX-R² 雙重過濾智慧動量」因子類 ---]
class Smart_Momentum_ADX_R2(BaseAlpha):
    """
    ADX-R² 雙重過濾智慧動量因子 (動態交互)

    核心邏輯: Momentum * (Trend_Probability(ADX) * R_squared)
    旨在尋找「高品質趨勢」：一個同時被市場共識（高ADX）和穩定路徑（高R²）
    所驗證的動量信號。信心權重會動態調節基礎動量信號的強度。

    因子值越高，代表一個越高質量的趨勢跟隨信號。
    """
    def __init__(self, window=126, adx_window=14, logistic_midpoint=25.0, logistic_steepness=0.2):
        self.window = window
        self.adx_window = adx_window
        self.x0 = logistic_midpoint
        self.k = logistic_steepness
        self.momentum_factor = Momentum(window=self.window)

    def predict(self) -> xr.DataArray:
        # 1. 計算基礎動量 (油門)
        self.momentum_factor.array_ = self.array_
        momentum_signal = self.momentum_factor.predict()

        # 2. 計算信心權重 (智慧導航系統)
        # 2a. 趨勢強度: ADX 趨勢概率
        adx, _, _ = calc_adx(self.high, self.low, self.close, self.adx_window)
        logit_input = self.k * (adx - self.x0)
        trend_probability = (1 / (1 + np.exp(-logit_input))).fillna(0)
        
        # 2b. 趨勢穩定性: R-squared
        log_close = log(self.close)
        _, r_squared, _ = ts_regression_all_stats(log_close, self.window)

        # 2c. 合成信心權重
        confidence_weight = trend_probability * r_squared.fillna(0)

        # 3. 合成最終因子：動態加權
        # 只對上漲動量進行加權，避免懲罰處於穩定下跌趨勢的資產
        final_signal = where(momentum_signal > 0, momentum_signal * confidence_weight, momentum_signal)
        
        return final_signal

# [--- 在此處新增「突破質量分」因子類 ---]
class Breakout_Quality_Score(BaseAlpha):
    """
    突破質量分因子 (動態交互，自洽型)
    
    核心邏輯: Momentum * Participation * Stability
    旨在尋找那些從一個相對穩定的前期環境中，以高市場參與度
    （成交活躍）形式向上突破的資產。
    
    所有計算邏輯均內置，不依賴其他因子類。
    因子值越高，代表一個越高質量的突破信號。
    """
    def __init__(self, mom_window=22, short_turn_window=22, long_turn_window=126, vol_window=63):
        self.mom_window = mom_window
        self.short_turn_window = short_turn_window
        self.long_turn_window = long_turn_window
        self.vol_window = vol_window

    def predict(self) -> xr.DataArray:
        # 1. 計算突破力度 (Force)
        momentum = pct_change(self.close, self.mom_window)

        # 2. 計算市場參與度 (Participation)
        short_turnover = ts_mean(self.turnover, self.short_turn_window)
        long_turnover = ts_mean(self.turnover, self.long_turn_window)
        safe_long_turnover = where(long_turnover == 0, np.nan, long_turnover)
        participation_score = short_turnover / safe_long_turnover

        # 3. 計算前期穩定性 (Stability)
        volatility = ts_std(self.returns, self.vol_window)
        safe_volatility = where(volatility == 0, np.nan, volatility)
        stability_score = 1 / safe_volatility

        # 4. 合成最終因子
        # 只考慮上漲動量，過濾掉下跌中的信號
        final_signal = where(momentum > 0, 
                             momentum * participation_score.fillna(1) * stability_score.fillna(0),
                             0)
        
        return final_signal

# [--- 在此處新增「趨勢確定性分數」因子類 ---]
class Trend_Certainty_Score(BaseAlpha):
    """
    趨勢確定性分數 (動態交互，自洽型)
    
    核心邏輯: Avg_Return * (1 + Clarity_of_Direction) * (1 + Clarity_of_Path)
    旨在尋找那些基礎回報為正，且其趨勢在“方向明確性”（高ADX）和
    “路徑清晰度”（高R²）兩個維度上都得到驗證的資產。
    
    所有計算邏輯均內置，不依賴其他因子類。
    因子值越高，代表趨勢的“確定性”越強，信號越可靠。
    """
    def __init__(self, window=63, adx_window=14, adx_threshold=25.0):
        self.window = window
        self.adx_window = adx_window
        self.adx_threshold = adx_threshold

    def predict(self) -> xr.DataArray:
        # 1. 計算基礎回報
        base_return = ts_mean(self.returns, self.window)

        # 2. 計算方向明確性 (Clarity of Direction)
        adx, _, _ = calc_adx(self.high, self.low, self.close, self.adx_window)
        # ADX 超過閾值的部分作為強度溢價，並進行縮放
        direction_clarity = where(adx > self.adx_threshold, (adx - self.adx_threshold) / 50, 0).fillna(0)

        # 3. 計算路徑清晰度 (Clarity of Path)
        log_close = log(self.close)
        _, r_squared, _ = ts_regression_all_stats(log_close, self.window)

        # 4. 合成最終因子
        # 基礎回報 * (1 + 增強項1) * (1 + 增強項2)
        # 只在基礎回報為正時應用增強，避免放大負收益信號
        final_signal = where(base_return > 0,
                             base_return * (1 + direction_clarity) * (1 + r_squared.fillna(0)),
                             base_return)
        
        return final_signal