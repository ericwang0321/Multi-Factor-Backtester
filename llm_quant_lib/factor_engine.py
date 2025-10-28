# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xarray as xr
import pandas_ta as ta # 使用 pandas_ta 库简化技术指标计算

# 导入必要的辅助函数
from .factor_helpers import xr_ewm, where, delta, pct_change, ts_mean, ts_rolling

class FactorEngine:
    """
    计算 AI 策略所需的核心技术指标 (EMA, MACD, RSI)。
    设计为可扩展，方便后续添加更多因子。
    使用 pandas_ta 库简化计算。
    """
    REQUIRED_COLS = ['open', 'high', 'low', 'close', 'volume'] # 计算这些因子所需列

    def __init__(self, data_handler):
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
        self.prices: Dict[str, pd.DataFrame] = {}
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

        self._factor_cache: Dict[str, pd.DataFrame] = {} # 缓存因子计算结果 (宽格式 DataFrame)
        print("FactorEngine: 初始化完成。")

    def _compute_ta_indicator(self, indicator_name: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        使用 pandas_ta 计算技术指标。

        Args:
            indicator_name (str): pandas_ta 支持的指标名称 (小写)。
            **kwargs: 传递给 pandas_ta 指标函数的参数 (例如 length=14)。

        Returns:
            pd.DataFrame or None: 计算得到的因子值 (宽格式)，如果计算失败则返回 None。
        """
        cache_key = f"{indicator_name}_{'_'.join(f'{k}{v}' for k,v in sorted(kwargs.items()))}"
        if cache_key in self._factor_cache:
            return self._factor_cache[cache_key]

        print(f"FactorEngine: 正在计算指标 '{indicator_name}' (参数: {kwargs})...")
        try:
            # 获取 pandas_ta 中的指标函数
            indicator_func = getattr(ta.momentum if indicator_name in ['rsi', 'macd'] else ta.overlap if indicator_name == 'ema' else ta, indicator_name, None)
            if not indicator_func:
                 # 尝试在 ta.trend 中查找 (例如 adx)
                 indicator_func = getattr(ta.trend, indicator_name, None)

            if not indicator_func:
                print(f"FactorEngine: 错误 - pandas_ta 中不支持指标 '{indicator_name}'。")
                return None

            # 准备输入数据 (pandas_ta 通常需要小写的列名)
            # 我们需要为每个 sec_code 单独计算
            all_results = {}
            # 遍历所有证券代码
            sec_codes = self.multi_col_df.columns.get_level_values(1).unique()
            for code in sec_codes:
                # 提取该代码的 OHLCV 数据
                asset_df = self.multi_col_df.loc[:, pd.IndexSlice[:, code]].copy()
                # 重命名列为小写 (open, high, low, close, volume)
                asset_df.columns = asset_df.columns.droplevel(1)
                asset_df.rename(columns=str.lower, inplace=True)

                # 调用 pandas_ta 函数
                result = indicator_func(asset_df['close'] if indicator_name in ['ema','rsi'] else asset_df['high'] if indicator_name=='adx' else asset_df, **kwargs) # 根据指标需要传入不同列

                if isinstance(result, pd.DataFrame):
                    # 对于 MACD 或 ADX 这类返回多列的指标，我们需要选择需要的列
                    if indicator_name == 'macd':
                        # MACD 函数返回 MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
                        all_results[f'macd_line_{kwargs.get("fast",12)}_{kwargs.get("slow",26)}'] = result[f'MACD_{kwargs.get("fast",12)}_{kwargs.get("slow",26)}_{kwargs.get("signal",9)}'].rename(code)
                        all_results[f'macd_signal_{kwargs.get("signal",9)}'] = result[f'MACDs_{kwargs.get("fast",12)}_{kwargs.get("slow",26)}_{kwargs.get("signal",9)}'].rename(code)
                        all_results[f'macd_hist_{kwargs.get("fast",12)}_{kwargs.get("slow",26)}_{kwargs.get("signal",9)}'] = result[f'MACDh_{kwargs.get("fast",12)}_{kwargs.get("slow",26)}_{kwargs.get("signal",9)}'].rename(code)
                    elif indicator_name == 'adx':
                         all_results[f'adx_{kwargs.get("length",14)}'] = result[f'ADX_{kwargs.get("length",14)}'].rename(code)
                         # 如果需要 +DI, -DI 也可以在这里提取
                    else:
                         # 其他返回 DataFrame 的指标，取第一列？或者需要特别处理
                         print(f"FactorEngine: 警告 - 指标 {indicator_name} 返回 DataFrame，暂未处理。")
                elif isinstance(result, pd.Series):
                    # 对于返回 Series 的指标 (如 EMA, RSI)
                    factor_col_name = f"{indicator_name}_{kwargs.get('length', kwargs.get('span', ''))}"
                    if factor_col_name not in all_results:
                         all_results[factor_col_name] = []
                    all_results[factor_col_name].append(result.rename(code)) # 重命名 Series 的 name 为 sec_code

            # 合并结果
            final_factors = {}
            for factor_col, series_list in all_results.items():
                if series_list:
                    factor_df = pd.concat(series_list, axis=1)
                    # 因子值滞后一期，匹配你的原始回测逻辑（因子计算基于 t-1 数据，在 t 交易）
                    factor_df_shifted = factor_df.shift(1)
                    self._factor_cache[factor_col] = factor_df_shifted # 缓存结果
                    final_factors[factor_col] = factor_df_shifted

            print(f"FactorEngine: 指标 '{indicator_name}' 计算完成。")
            # 返回字典可能包含多个因子 (例如 MACD 有三个)
            return final_factors

        except Exception as e:
            print(f"FactorEngine: 计算指标 '{indicator_name}' 时出错: {e}")
            import traceback
            traceback.print_exc()
            return None # 返回 None 表示计算失败

    def get_factor_snapshot(self, current_date: pd.Timestamp, codes: List[str]) -> pd.DataFrame:
        """
        获取指定日期、指定证券代码列表的所有已实现因子的截面快照。

        Args:
            current_date (pd.Timestamp): 需要获取快照的日期。
            codes (List[str]): 需要包含在快照中的证券代码列表。

        Returns:
            pd.DataFrame: 包含指定日期、指定代码的因子值的 DataFrame (索引: sec_code, 列: 因子名称)。
        """
        # print(f"FactorEngine: 正在为 {current_date.date()} (代码: {len(codes)}个) 生成因子快照...")
        all_factor_data = {}

        # 1. 计算/获取 EMA
        ema_factors = self._compute_ta_indicator('ema', span=12)
        if ema_factors: all_factor_data.update(ema_factors)
        ema_factors = self._compute_ta_indicator('ema', span=26)
        if ema_factors: all_factor_data.update(ema_factors)

        # 2. 计算/获取 MACD
        macd_factors = self._compute_ta_indicator('macd', fast=12, slow=26, signal=9)
        if macd_factors: all_factor_data.update(macd_factors)

        # 3. 计算/获取 RSI
        rsi_factors = self._compute_ta_indicator('rsi', length=14)
        if rsi_factors: all_factor_data.update(rsi_factors)

        # --- 可扩展点 ---
        # adx_factors = self._compute_ta_indicator('adx', length=14)
        # if adx_factors: all_factor_data.update(adx_factors)
        # -----------------

        # 从缓存的 DataFrame 中提取当天的截面数据
        snapshot_dict = {}
        valid_codes_in_snapshot = set()

        for factor_name, factor_df in all_factor_data.items():
            if current_date in factor_df.index:
                # 提取当天的 Series，并只保留需要的 codes
                snapshot_series = factor_df.loc[current_date, factor_df.columns.intersection(codes)]
                snapshot_dict[factor_name] = snapshot_series
                valid_codes_in_snapshot.update(snapshot_series.index)
            # else:
            #     print(f"FactorEngine: 警告 - 在 {current_date.date()} 因子 '{factor_name}' 无数据。")

        if not snapshot_dict:
             print(f"FactorEngine: 警告 - 在 {current_date.date()} 未找到任何因子数据。")
             return pd.DataFrame(index=pd.Index(codes, name='sec_code')) # 返回空 DataFrame 但保留索引

        # 合并所有 Series 成为一个 DataFrame
        snapshot_df = pd.DataFrame(snapshot_dict)
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

