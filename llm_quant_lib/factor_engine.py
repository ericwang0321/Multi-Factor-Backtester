# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xarray as xr
import pandas_ta as ta # 使用 pandas_ta 库简化技术指标计算
from typing import Dict, List, Optional # <--- 【修正】添加这行导入语句
from collections import defaultdict # <--- 【修正】添加这行导入语句

# 导入必要的辅助函数
from .factor_helpers import xr_ewm, where, delta, pct_change, ts_mean, ts_rolling
# 假设 DataHandler 类定义在 .data_handler 模块中
from .data_handler import DataHandler

class FactorEngine:
    """
    计算 AI 策略所需的核心技术指标 (EMA, MACD, RSI)。
    设计为可扩展，方便后续添加更多因子。
    使用 pandas_ta 库简化计算。
    """
    REQUIRED_COLS = ['open', 'high', 'low', 'close', 'volume'] # 计算这些因子所需列

    def __init__(self, data_handler: DataHandler): # 【修正】添加类型提示
        """
        初始化因子引擎。

        Args:
            data_handler (DataHandler): 已初始化的数据处理器实例。
        """
        print("FactorEngine: 正在初始化...")
        self.data_handler = data_handler
        # 确保 DataHandler 已加载数据
        self.raw_df = self.data_handler.load_data()

        # 准备因子计算所需的宽格式价格数据
        self.prices: Dict[str, pd.DataFrame] = {} # 【修正】使用导入的 Dict
        missing_cols = []
        for col in self.REQUIRED_COLS:
            try:
                # 注意：这里我们获取所有代码的价格，后续在 get_snapshot 中过滤
                self.prices[col] = self.data_handler.get_pivot_prices(col, codes=None)
                if self.prices[col].empty:
                    missing_cols.append(col)
            except ValueError as e:
                missing_cols.append(f"{col} ({e})")

        if missing_cols:
            raise ValueError(f"FactorEngine: 无法获取计算因子所需的列: {', '.join(missing_cols)}。请确保数据完整。")

        # 将多个价格 DataFrame 合并为一个，MultiIndex 列格式为 ('close', 'SPY.P')
        # pandas_ta 需要这种格式
        print("FactorEngine: 正在合并价格数据以便计算因子...")
        self.multi_col_df = pd.concat(self.prices, axis=1)

        self._factor_cache: Dict[str, pd.DataFrame] = {} # 缓存因子计算结果 (宽格式 DataFrame) # 【修正】使用导入的 Dict
        print("FactorEngine: 初始化完成。")

    # 【修正】返回值类型提示 Optional[Dict[str, pd.DataFrame]]
    def _compute_ta_indicator(self, indicator_name: str, **kwargs) -> Optional[Dict[str, pd.DataFrame]]:
        """
        使用 pandas_ta 计算技术指标。

        Args:
            indicator_name (str): pandas_ta 支持的指标名称 (小写)。
            **kwargs: 传递给 pandas_ta 指标函数的参数 (例如 length=14)。

        Returns:
            Optional[Dict[str, pd.DataFrame]]: 包含一个或多个因子 DataFrame 的字典，如果计算失败则返回 None。
                                            键是因子名称 (例如 'ema_12', 'macd_line_12_26')。
        """
        # ---【内部逻辑保持不变】---
        # 生成缓存键 (确保 kwargs 顺序一致)
        kwargs_sorted = sorted(kwargs.items())
        cache_key = f"{indicator_name}_{'_'.join(f'{k}{v}' for k, v in kwargs_sorted)}"

        # 检查缓存 (直接返回缓存的整个字典或单个 DataFrame)
        # 注意: 缓存现在存储的是计算结果字典或单个 DataFrame
        cached_result = self._factor_cache.get(cache_key)
        if cached_result is not None:
             # 如果缓存的是一个字典 (如 MACD)，直接返回
             # 如果缓存的是单个 DataFrame (如 EMA)，为了统一返回格式，也包在字典里返回
             if isinstance(cached_result, pd.DataFrame):
                 return {cache_key: cached_result} # 包装成字典返回
             elif isinstance(cached_result, dict):
                 return cached_result # 直接返回字典
             else:
                 # 处理可能的异常情况
                 print(f"FactorEngine: 警告 - 缓存中存在未知类型的数据: {cache_key}")
                 # 尝试重新计算
                 pass # 继续执行计算逻辑

        print(f"FactorEngine: 正在计算指标 '{indicator_name}' (参数: {kwargs})...")
        try:
            # 获取 pandas_ta 中的指标函数
            # 修正查找逻辑，更清晰
            indicator_func = None
            if hasattr(ta, indicator_name):
                 indicator_func = getattr(ta, indicator_name)
            elif hasattr(ta.momentum, indicator_name):
                 indicator_func = getattr(ta.momentum, indicator_name)
            elif hasattr(ta.overlap, indicator_name):
                 indicator_func = getattr(ta.overlap, indicator_name)
            elif hasattr(ta.trend, indicator_name):
                 indicator_func = getattr(ta.trend, indicator_name)
            # 可以根据需要添加其他 ta 子模块

            if not indicator_func:
                print(f"FactorEngine: 错误 - pandas_ta 中不支持指标 '{indicator_name}'。")
                return None

            # 准备输入数据 (pandas_ta 通常需要小写的列名)
            # 我们需要为每个 sec_code 单独计算
            all_results_dict: Dict[str, List[pd.Series]] = defaultdict(list) # 用于收集每个因子的 Series 列表
            sec_codes = self.multi_col_df.columns.get_level_values(1).unique()

            for code in sec_codes:
                asset_df = self.multi_col_df.loc[:, pd.IndexSlice[:, code]].copy()
                if asset_df.empty: continue # 跳过没有数据的资產

                asset_df.columns = asset_df.columns.droplevel(1) # 移除顶层 ('open', 'close'...)
                asset_df.rename(columns=str.lower, inplace=True) # 转为小写

                # 确保计算所需列存在且有效
                required_cols_for_indicator = ['close'] # 默认需要 close
                if indicator_name in ['adx']: required_cols_for_indicator = ['high', 'low', 'close']
                if indicator_name in ['macd']: required_cols_for_indicator = ['close']
                if indicator_name in ['ema']: required_cols_for_indicator = ['close']
                # ...可以为其他指标添加所需列...

                # 检查数据是否足够长且有效
                if len(asset_df) < kwargs.get('length', kwargs.get('span', 1)) or asset_df[required_cols_for_indicator].isnull().all().any():
                     # print(f"FactorEngine: 资產 {code} 数据不足或无效，跳过 {indicator_name} 计算。")
                     continue # 跳过这个资產

                # ---【核心计算调用】---
                # 确定输入列
                input_series = asset_df['close'] # 默认
                if indicator_name in ['adx']:
                     # ADX 需要 DataFrame 输入 high, low, close
                     input_data = asset_df[['high', 'low', 'close']]
                # elif ... 其他需要特定输入的指标
                else:
                     input_data = input_series # 大部分指标只需要 close

                result = indicator_func(input_data, **kwargs)
                # ---【计算结束】---

                if result is None: continue # 计算失败，跳过

                # ---【处理返回结果】---
                if isinstance(result, pd.DataFrame):
                    # MACD 返回多列: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
                    if indicator_name == 'macd':
                        fast = kwargs.get("fast",12); slow = kwargs.get("slow",26); signal = kwargs.get("signal",9)
                        line_col = f'MACD_{fast}_{slow}_{signal}'
                        signal_col = f'MACDs_{fast}_{slow}_{signal}'
                        hist_col = f'MACDh_{fast}_{slow}_{signal}'
                        if line_col in result.columns:
                            all_results_dict[f'macd_line_{fast}_{slow}'].append(result[line_col].rename(code))
                        if signal_col in result.columns:
                            all_results_dict[f'macd_signal_{signal}'].append(result[signal_col].rename(code))
                        if hist_col in result.columns:
                            all_results_dict[f'macd_hist_{fast}_{slow}_{signal}'].append(result[hist_col].rename(code))
                    # ADX 返回多列: ADX_14, DMP_14, DMN_14
                    elif indicator_name == 'adx':
                        length = kwargs.get("length", 14)
                        adx_col = f'ADX_{length}'
                        if adx_col in result.columns:
                             all_results_dict[f'adx_{length}'].append(result[adx_col].rename(code))
                         # 如果需要 +DI (DMP), -DI (DMN) 可以在这里提取
                    else:
                        print(f"FactorEngine: 警告 - 指标 {indicator_name} 返回未处理的 DataFrame 格式。")

                elif isinstance(result, pd.Series):
                    # EMA 返回 EMA_span
                    if indicator_name == 'ema':
                         factor_col_name = f"ema_{kwargs.get('span', '')}"
                         all_results_dict[factor_col_name].append(result.rename(code))
                    # RSI 返回 RSI_length
                    elif indicator_name == 'rsi':
                         factor_col_name = f"rsi_{kwargs.get('length', '')}"
                         all_results_dict[factor_col_name].append(result.rename(code))
                    else:
                         # 其他返回 Series 的指标
                         factor_col_name = f"{indicator_name}_{kwargs.get('length', '')}"
                         all_results_dict[factor_col_name].append(result.rename(code))

            # --- 合并结果并缓存 ---
            final_factors_dict: Dict[str, pd.DataFrame] = {}
            for factor_col, series_list in all_results_dict.items():
                if series_list:
                    try:
                        factor_df = pd.concat(series_list, axis=1)
                        # 因子值滞后一期
                        factor_df_shifted = factor_df.shift(1)
                        # 将单个计算出的因子存入缓存
                        self._factor_cache[factor_col] = factor_df_shifted
                        final_factors_dict[factor_col] = factor_df_shifted
                    except Exception as concat_err:
                         print(f"FactorEngine: 合并因子 '{factor_col}' 时出错: {concat_err}")

            if not final_factors_dict:
                 print(f"FactorEngine: 指标 '{indicator_name}' 未能为任何资產生成有效结果。")
                 return None # 没有成功计算出任何因子

            print(f"FactorEngine: 指标 '{indicator_name}' 计算完成，生成因子: {list(final_factors_dict.keys())}。")
            # 将这个指标调用产生的所有因子（可能多个）一起存入一个以 cache_key 命名的缓存条目
            # 这与之前的逻辑不同，之前是按单个因子名存
            self._factor_cache[cache_key] = final_factors_dict # 缓存整个结果字典
            return final_factors_dict

        except Exception as e:
            print(f"FactorEngine: 计算指标 '{indicator_name}' 时发生严重错误: {e}")
            import traceback
            traceback.print_exc()
            return None # 返回 None 表示计算失败

    def get_factor_snapshot(self, current_date: pd.Timestamp, codes: List[str]) -> pd.DataFrame: # 【修正】使用导入的 List
        """
        获取指定日期、指定证券代码列表的所有已实现因子的截面快照。

        Args:
            current_date (pd.Timestamp): 需要获取快照的日期。
            codes (List[str]): 需要包含在快照中的证券代码列表。

        Returns:
            pd.DataFrame: 包含指定日期、指定代码的因子值的 DataFrame (索引: sec_code, 列: 因子名称)。
        """
        # print(f"FactorEngine: 正在为 {current_date.date()} (代码: {len(codes)}个) 生成因子快照...")
        all_factor_data: Dict[str, pd.DataFrame] = {} # 用于存储所有因子的宽格式 DataFrame

        # --- 按需计算并合并所有因子 ---
        # 1. 计算/获取 EMA(12) 和 EMA(26)
        # 注意：_compute_ta_indicator 返回的是字典
        ema12_dict = self._compute_ta_indicator('ema', span=12)
        if ema12_dict: all_factor_data.update(ema12_dict)
        ema26_dict = self._compute_ta_indicator('ema', span=26)
        if ema26_dict: all_factor_data.update(ema26_dict)

        # 2. 计算/获取 MACD (会产生多个因子)
        macd_dict = self._compute_ta_indicator('macd', fast=12, slow=26, signal=9)
        if macd_dict: all_factor_data.update(macd_dict)

        # 3. 计算/获取 RSI(14)
        rsi_dict = self._compute_ta_indicator('rsi', length=14)
        if rsi_dict: all_factor_data.update(rsi_dict)

        # --- 可扩展点 ---
        # adx_dict = self._compute_ta_indicator('adx', length=14)
        # if adx_dict: all_factor_data.update(adx_dict)
        # -----------------

        # --- 从所有已计算的因子中提取当天的截面数据 ---
        snapshot_series_list: List[pd.Series] = []
        valid_codes_in_snapshot = set()

        # 遍历所有计算出的因子 DataFrame (例如 'ema_12', 'macd_line_12_26', 'rsi_14' 等)
        for factor_name, factor_df in all_factor_data.items():
            if current_date in factor_df.index:
                # 提取当天的 Series (列是 sec_code)
                daily_series = factor_df.loc[current_date]
                # 只保留需要的 codes
                snapshot_series = daily_series[daily_series.index.intersection(codes)]
                # 设置 Series 的 name 为因子名
                snapshot_series.name = factor_name
                snapshot_series_list.append(snapshot_series)
                valid_codes_in_snapshot.update(snapshot_series.index)
            # else:
            #     print(f"FactorEngine: 警告 - 在 {current_date.date()} 因子 '{factor_name}' 无数据。")

        if not snapshot_series_list:
             print(f"FactorEngine: 警告 - 在 {current_date.date()} 未找到任何因子数据。")
             return pd.DataFrame(index=pd.Index(codes, name='sec_code')) # 返回空 DataFrame 但保留索引

        # 将所有因子的 Series 合并成一个 DataFrame (索引是 sec_code, 列是 factor_name)
        snapshot_df = pd.concat(snapshot_series_list, axis=1)
        snapshot_df.index.name = 'sec_code'

        # 确保所有请求的 codes 都在索引中，即使它们没有任何因子数据 (用 NaN 填充)
        snapshot_df = snapshot_df.reindex(codes)

        # 因子值预处理：填充缺失值 (AI 不喜欢 NaN)
        # 简单的填充策略：用 0 填充。更复杂的可以考虑中位数等。
        snapshot_df_filled = snapshot_df.fillna(0.0)

        # print(f"FactorEngine: {current_date.date()} 因子快照生成完毕。")
        return snapshot_df_filled

    def add_custom_factor(self, factor_name: str, factor_logic_func):
        """
        (未来扩展功能) 允许用户添加自定义因子计算逻辑。
        factor_logic_func 应该接收 self.multi_col_df 并返回一个宽格式的因子 DataFrame。
        """
        pass

