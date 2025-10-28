# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
from typing import List, Dict, Optional

class DataHandler:
    """
    负责从数据库或 CSV 文件加载、处理和提供原始价格及资产池数据。
    """
    def __init__(self, db_config: Optional[Dict] = None, csv_path: Optional[str] = None, start_date: str = '2016-01-01', end_date: str = '2025-12-31'):
        """
        初始化数据处理器。

        Args:
            db_config (dict, optional): 数据库连接信息。
            csv_path (str, optional): 预处理数据的 CSV 文件路径。
            start_date (str): 数据加载的起始日期。
            end_date (str): 数据加载的结束日期。
        """
        self.db_config = db_config
        self.csv_path = csv_path
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.raw_df: Optional[pd.DataFrame] = None # 长格式的原始数据
        # ---【修改】universe_df 内部存储格式改为以 sec_code 为索引 ---
        self.universe_df: Optional[pd.DataFrame] = None # 资产池定义 (index: sec_code, column: universe)
        self.all_codes: List[str] = [] # 所有加载的证券代码
        self.pivot_cache: Dict[str, pd.DataFrame] = {} # 缓存宽格式数据

        # --- 美股ETF代码列表 (如果需要从 DB 加载时使用) ---
        # (us_codes 列表保持不变, 省略...)
        self.us_codes = [
             'KSA.P', 'CBON.P', 'CQQQ.P', 'KWEB.P', 'MCHI.O', 'PFF.O', 'ICLN.O', 'ACWI.O', 'DBA.P', 'SMH.O', 'INDA.B', 'USO.P','EWT.P', 'VXUS.O', 'EWZ.P', 'BWX.P', 'EWG.P', 'EMB.O','IEMG.P', 'EMXC.O', 'EWJ.P', 'SPY.P', 'IUSG.O', 'IUSB.O','IGOV.O', 'VGK.P', 'IAGG.B', 'EWQ.P', 'EWA.P', 'IBB.O','DVY.O', 'QQQ.O', 'VONG.O', 'VTWO.O', 'IEF.O', 'SHY.O','MBB.O', 'TIP.P', 'IGSB.O', 'FLOT.B', 'VTIP.O', 'SHV.O','BIL.P', 'TLT.O', 'HYG.P', 'BNDX.O', 'EWU.P', 'BITO.P','CPER.P', 'SLV.P', 'VCLT.O', 'GLD.P', 'EWH.P', 'EFA.P','EWY.P', 'VEGI.P', 'XLE.P', 'VNQ.P', 'VTI.P', 'VEU.P','DBC.P', 'MDY.P', 'SPAB.P', 'VUG.P', 'VTV.P', 'VOT.P','VOE.P', 'VBK.P', 'VBR.P', 'SCZ.O', 'VNQI.O', 'PSP.P','BND.O', 'BSV.P', 'GSG.P', 'IEI.O', 'SPYG.P', 'SPYV.P','XRT.P', 'XLF.P', 'XHB.P', 'PPH.O', 'IJJ.P', 'IJK.P','IJS.P', 'IJT.O', 'JNK.P', 'CWB.P', 'ICVT.B','MINT.P', 'ANGL.O', 'XLV.P', 'XLB.P', 'XLC.P', 'XLI.P','XLK.P', 'XLP.P', 'XLU.P', 'XMMO.P', 'XLRE.P', 'XLY.P'
        ]

    def load_data(self) -> pd.DataFrame:
        """
        加载价格数据，优先从 CSV 加载，失败则从 DB 加载。
        返回长格式 DataFrame 并缓存。
        """
        if self.raw_df is not None:
            return self.raw_df

        loaded_from_csv = False
        if self.csv_path and os.path.exists(self.csv_path):
            print(f"DataHandler: 正在从 CSV 文件 '{self.csv_path}' 加载数据...")
            try:
                self.raw_df = pd.read_csv(self.csv_path, parse_dates=['datetime'])
                # ---【健壮性】确保 datetime 列是正确的类型 ---
                if 'datetime' not in self.raw_df.columns:
                    raise ValueError("CSV 文件缺少 'datetime' 列。")
                self.raw_df['datetime'] = pd.to_datetime(self.raw_df['datetime'], errors='coerce')
                self.raw_df = self.raw_df.dropna(subset=['datetime']) # 移除无法解析的日期行

                print("DataHandler: CSV 数据加载成功。")
                loaded_from_csv = True
            except Exception as e:
                print(f"DataHandler: 从 CSV 加载数据失败: {e}。将尝试从数据库加载。")
                self.raw_df = None

        if not loaded_from_csv:
            if self.db_config:
                print("DataHandler: 正在从数据库加载数据...")
                self.raw_df = self._load_from_db()
                if self.raw_df is not None and not self.raw_df.empty and self.csv_path:
                    print(f"DataHandler: 将数据保存到 '{self.csv_path}'...")
                    try:
                        output_dir = os.path.dirname(self.csv_path)
                        os.makedirs(output_dir, exist_ok=True)
                        self.raw_df.to_csv(self.csv_path, index=False)
                        print("DataHandler: 数据库数据已加载并保存到 CSV。")
                    except Exception as save_err:
                         print(f"DataHandler: 警告 - 保存数据到 CSV 时出错: {save_err}")
            else:
                raise FileNotFoundError(f"CSV 文件 '{self.csv_path}' 未找到，且未提供数据库配置。无法加载数据。")

        if self.raw_df is None or self.raw_df.empty:
             raise RuntimeError("DataHandler: 无法从任何来源加载有效的价格数据。")

        # --- 统一进行日期过滤和代码列表更新 ---
        # 在过滤前确保 datetime 列类型正确
        if not pd.api.types.is_datetime64_any_dtype(self.raw_df['datetime']):
             self.raw_df['datetime'] = pd.to_datetime(self.raw_df['datetime'], errors='coerce')
             self.raw_df = self.raw_df.dropna(subset=['datetime'])

        self.raw_df = self.raw_df[
            (self.raw_df['datetime'] >= self.start_date) &
            (self.raw_df['datetime'] <= self.end_date)
        ].copy()

        if self.raw_df.empty:
             raise ValueError(f"DataHandler: 在日期范围 {self.start_date.date()} 到 {self.end_date.date()} 内没有找到任何价格数据。")

        self.all_codes = sorted(self.raw_df['sec_code'].unique().tolist())
        print(f"DataHandler: 数据加载完成，包含 {len(self.all_codes)} 个证券代码，日期范围: {self.raw_df['datetime'].min().date()} 到 {self.raw_df['datetime'].max().date()}")
        return self.raw_df

    def _load_from_db(self) -> Optional[pd.DataFrame]:
        """从数据库下载并处理数据 (核心逻辑来自你的原始脚本)"""
        if not self.db_config:
            print("DataHandler: 未提供数据库配置。")
            return None

        # --- 检查密码是否存在 ---
        if not self.db_config.get('password'):
            print("DataHandler: 错误 - 数据库密码未提供 (请检查 DB_PASSWORD 环境变量)。无法连接数据库。")
            return None

        # ---【健壮性】添加数据库连接的 try-except ---
        try:
            engine = create_engine(
                f"postgresql+psycopg2://{self.db_config['username']}:{self.db_config['password']}@"
                f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            )
            # 尝试连接
            with engine.connect() as connection:
                print("DataHandler: 数据库连接成功。")
        except Exception as conn_err:
             print(f"DataHandler: 错误 - 无法连接到数据库: {conn_err}")
             print("DataHandler: 请检查数据库配置和网络连接。")
             return None

        df_merged: Optional[pd.DataFrame] = None
        try:
            # --- 下载价格数据 ---
            us_table = 'ods_mkt_ths_etf_us_daily_price_adj'
            # 确保代码列表非空
            if not self.us_codes:
                 print("DataHandler: 错误 - 未指定要从数据库下载的证券代码列表 (self.us_codes)。")
                 return None
            us_codes_str = ', '.join([f"'{code}'" for code in self.us_codes])
            us_query = f"SELECT * FROM {us_table} WHERE sec_code IN ({us_codes_str});"
            print("DataHandler: 正在下载价格数据...")
            df_price = pd.read_sql(us_query, con=engine)
            if df_price.empty:
                 print("DataHandler: 警告 - 未从数据库下载到任何价格数据。")
                 return pd.DataFrame() # 返回空 DataFrame
            print(f"DataHandler: 价格数据下载完成，共 {len(df_price)} 条记录。")

            df_price['datetime'] = pd.to_datetime(df_price['datetime'], errors='coerce') # 强制转换
            df_price = df_price.dropna(subset=['datetime']) # 移除无效日期
            df_price = df_price.sort_values(by=['sec_code', 'datetime'])

            # --- 下载基金信息 ---
            fund_info_table = 'ods_mkt_ths_etf_us_daily_fund_info'
            fund_info_query = f"SELECT datetime, sec_code, fund_net_value, fund_total_asset FROM {fund_info_table} WHERE sec_code IN ({us_codes_str});"
            print("DataHandler: 正在下载基金信息...")
            df_fund_info = pd.read_sql(fund_info_query, con=engine)
            print(f"DataHandler: 基金信息下载完成，共 {len(df_fund_info)} 条记录。")

            # --- 合并与处理 ---
            df_fund_info['datetime'] = pd.to_datetime(df_fund_info['datetime'], errors='coerce')
            df_fund_info = df_fund_info.dropna(subset=['datetime'])

            df_merged = pd.merge(df_price, df_fund_info, on=['datetime', 'sec_code'], how='left')
            df_merged = df_merged.sort_values(by=['sec_code', 'datetime'])

            # ---【健壮性】检查合并后是否为空 ---
            if df_merged.empty:
                print("DataHandler: 警告 - 价格与基金信息合并后数据为空。")
                return pd.DataFrame()

            # 重命名和计算 amount, market_cap (如果列存在)
            if 'volume' in df_merged.columns and 'avg_price' in df_merged.columns:
                 df_merged['amount'] = df_merged['volume'] * df_merged['avg_price']
            if 'fund_total_asset' in df_merged.columns:
                 df_merged.rename(columns={'fund_total_asset': 'market_cap'}, inplace=True)

            print("DataHandler: 正在填充缺失值...")
            columns_to_fill = ['open', 'high', 'low', 'close', 'volume', 'amount', 'avg_price', 'fund_net_value', 'market_cap']
            existing_columns_to_fill = [col for col in columns_to_fill if col in df_merged.columns]

            if not existing_columns_to_fill:
                 print("DataHandler: 警告 - 未找到任何需要填充的关键数值列。")
            else:
                 # ---【健壮性】确保 apply 的函数能处理空 group ---
                 def robust_fill(group):
                     if group.empty: return group
                     # 尝试转换为数值类型，无法转换的变为 NaN
                     for col in existing_columns_to_fill:
                          group[col] = pd.to_numeric(group[col], errors='coerce')
                     # 填充
                     group[existing_columns_to_fill] = group[existing_columns_to_fill].interpolate(method='linear', limit_direction='both', axis=0) # 按时间插值
                     group[existing_columns_to_fill] = group[existing_columns_to_fill].ffill()
                     group[existing_columns_to_fill] = group[existing_columns_to_fill].bfill()
                     return group

                 df_merged = df_merged.groupby('sec_code', group_keys=False).apply(robust_fill)
                 df_merged[existing_columns_to_fill] = df_merged[existing_columns_to_fill].fillna(0) # 最后用0填充剩余 NaN
                 print("DataHandler: 缺失值填充完成。")

            print("DataHandler: 正在计算衍生指标...")
            # ---【健壮性】计算衍生指标前检查所需列是否存在 ---
            if 'close' in df_merged.columns:
                 df_merged['simple_return'] = df_merged.groupby('sec_code')['close'].pct_change().fillna(0)
            else:
                 df_merged['simple_return'] = 0.0 # 或者 np.nan

            if 'market_cap' in df_merged.columns and 'fund_net_value' in df_merged.columns:
                 df_merged['fund_net_value_safe'] = df_merged['fund_net_value'].replace(0, np.nan) # 使用新列避免修改原始列
                 df_merged['shares_outstanding'] = df_merged['market_cap'] / df_merged['fund_net_value_safe']
                 df_merged['shares_outstanding'].replace([np.inf, -np.inf], np.nan, inplace=True)
                 # 按组填充 shares_outstanding
                 df_merged['shares_outstanding'] = df_merged.groupby('sec_code')['shares_outstanding'].ffill().bfill().fillna(0)
                 df_merged.drop(columns=['fund_net_value_safe'], inplace=True) # 移除临时列
            else:
                 df_merged['shares_outstanding'] = 0.0

            if 'volume' in df_merged.columns and 'shares_outstanding' in df_merged.columns:
                 df_merged['shares_outstanding_safe'] = df_merged['shares_outstanding'].replace(0, np.nan)
                 df_merged['turnover'] = df_merged['volume'] / df_merged['shares_outstanding_safe']
                 df_merged['turnover'].replace([np.inf, -np.inf], np.nan, inplace=True)
                 # 按组填充 turnover
                 df_merged['turnover'] = df_merged.groupby('sec_code')['turnover'].ffill().bfill().fillna(0)
                 df_merged.drop(columns=['shares_outstanding_safe'], inplace=True)
            else:
                 df_merged['turnover'] = 0.0
            print("DataHandler: 衍生指标计算完成。")

            # ---【健壮性】选择最终列时确保它们存在 ---
            final_columns = ['datetime', 'sec_code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'avg_price', 'simple_return', 'shares_outstanding', 'turnover', 'market_cap']
            final_columns_exist = [col for col in final_columns if col in df_merged.columns]

            # 处理 avg_price -> vwap (如果 avg_price 存在)
            if 'avg_price' in final_columns_exist:
                df_merged.rename(columns={'avg_price': 'vwap'}, inplace=True)
                final_columns_exist.remove('avg_price')
                if 'vwap' not in final_columns_exist:
                     final_columns_exist.append('vwap')

            # 确保 datetime 和 sec_code 始终包含
            if 'datetime' not in final_columns_exist: final_columns_exist.insert(0, 'datetime')
            if 'sec_code' not in final_columns_exist: final_columns_exist.insert(1, 'sec_code')
            final_columns_exist = list(dict.fromkeys(final_columns_exist)) # 去重并保持顺序

            df_merged = df_merged[final_columns_exist].copy() # 确保只保留需要的列

        except Exception as e:
            print(f"DataHandler: 数据库操作或数据处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None # 返回 None 表示失败
        finally:
            if 'engine' in locals() and engine: # 确保 engine 已定义
                engine.dispose()
                print("DataHandler: 数据库连接已关闭。")

        return df_merged


    def get_pivot_prices(self, price_type: str = 'close', codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取指定价格类型的宽格式 DataFrame (datetime x sec_code)，可选择过滤证券代码。

        Args:
            price_type (str): 'open', 'high', 'low', 'close', 'vwap' 等。
            codes (list, optional): 只包含指定的证券代码。默认为 None (包含所有已加载的代码)。

        Returns:
            pd.DataFrame: 宽格式的价格数据。
        """
        # ---【修改】确保 raw_df 已加载 ---
        if self.raw_df is None:
            print("DataHandler: 价格数据尚未加载，正在尝试加载...")
            self.load_data()
            if self.raw_df is None: # 如果加载失败
                 raise RuntimeError("DataHandler: 无法加载价格数据，无法生成宽格式数据。")

        cache_key = price_type
        if cache_key in self.pivot_cache:
            pivot_df = self.pivot_cache[cache_key]
        else:
            if price_type not in self.raw_df.columns:
                raise ValueError(f"DataHandler: 无法在原始数据中找到价格类型 '{price_type}'")

            print(f"DataHandler: 正在生成 {price_type} 的宽格式数据...")
            try:
                pivot_df = self.raw_df.pivot(index='datetime', columns='sec_code', values=price_type)
                pivot_df.sort_index(inplace=True)
                # 基础填充 (向前填充后向填充)
                pivot_df = pivot_df.ffill().bfill()
                self.pivot_cache[cache_key] = pivot_df
                print(f"DataHandler: {price_type} 宽格式数据已准备就绪。")
            except Exception as e:
                 raise RuntimeError(f"DataHandler: 生成 {price_type} 宽格式数据时出错: {e}")


        # 根据传入的 codes 进行过滤
        if codes:
            # 确保 codes 是列表或类似结构
            if not isinstance(codes, (list, set, pd.Index)):
                 codes = list(codes)
            codes_in_df = [c for c in codes if c in pivot_df.columns]
            if not codes_in_df:
                 print(f"DataHandler: 警告 - 在 {price_type} 数据中未找到任何指定的证券代码。")
                 return pd.DataFrame(index=pivot_df.index) # 返回空 DataFrame 但保留日期索引
            # ---【健壮性】确保返回的是副本，防止修改缓存 ---
            return pivot_df[codes_in_df].copy()
        else:
            # ---【健壮性】确保返回的是副本 ---
            return pivot_df.copy()

    def load_universe_data(self, universe_path: str) -> pd.DataFrame:
        """
        加载资产池定义文件 (sec_code_category_grouped.csv)。
        内部存储为以 sec_code 为索引的 DataFrame，但返回带列的 DataFrame。
        """
        # ---【修改】如果已加载，直接使用内部的 DataFrame 重置索引返回 ---
        if self.universe_df is not None:
            return self.universe_df.reset_index()

        print(f"DataHandler: 正在加载资产池定义文件: {universe_path}...")
        try:
            u_df = pd.read_csv(universe_path)
            # 重命名列
            if 'category_id' in u_df.columns:
                u_df = u_df.rename(columns={'category_id': 'universe'})
            if 'sec_code' not in u_df.columns or 'universe' not in u_df.columns:
                 raise ValueError("资产池文件必须包含 'sec_code' 和 'category_id'/'universe' 列。")

            # ---【修改】确保 raw_df 已加载以获取 all_codes ---
            if self.raw_df is None:
                 print("DataHandler: 价格数据尚未加载，正在尝试加载以获取证券代码列表...")
                 self.load_data()
                 if self.raw_df is None:
                      raise RuntimeError("无法加载价格数据，无法验证资产池代码。")

            # 确保所有已加载的代码都有资产池归属
            missing_codes = set(self.all_codes) - set(u_df['sec_code'].unique())
            if missing_codes:
                 print(f"DataHandler: 警告 - 以下证券代码未在资产池文件中定义，将归类为 'Unknown': {sorted(list(missing_codes))}")
                 unknown_df = pd.DataFrame({'sec_code': list(missing_codes), 'universe': 'Unknown'})
                 u_df = pd.concat([u_df, unknown_df], ignore_index=True)

            # ---【修改】内部存储格式：索引为 sec_code ---
            self.universe_df = u_df[['sec_code', 'universe']].drop_duplicates().set_index('sec_code')
            print("DataHandler: 资产池定义加载成功。")
            # ---【修改】返回带列的 DataFrame ---
            return self.universe_df.reset_index()
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到资产池定义文件: {universe_path}")
        except Exception as e:
            raise RuntimeError(f"加载资产池定义文件时出错: {e}")

    def get_codes_in_universe(self, universe_name: str) -> List[str]:
        """
        获取指定资产池包含的证券代码列表。支持 'All'。
        从内部以 sec_code 为索引的 DataFrame 获取。
        """
        # ---【修改】确保 raw_df 已加载以获取 all_codes ---
        if self.raw_df is None:
             print("DataHandler: 价格数据尚未加载，正在尝试加载以获取证券代码列表...")
             self.load_data()
             if self.raw_df is None:
                  raise RuntimeError("无法加载价格数据，无法获取代码列表。")

        if universe_name.lower() == 'all':
            return self.all_codes # 返回所有已加载的证券代码

        # ---【修改】确保 universe_df 已加载 ---
        if self.universe_df is None:
             raise RuntimeError("请先调用 load_universe_data 加载资产池定义。")

        # ---【核心修正】从以 sec_code 为索引的 DataFrame 中筛选 ---
        try:
            # 筛选出 universe 列等于 universe_name 的行的索引 (即 sec_code)
            codes = self.universe_df[self.universe_df['universe'] == universe_name].index.tolist()
        except KeyError: # 如果 'universe' 列不存在 (理论上 load 时已检查)
             print(f"DataHandler: 错误 - 内部 universe_df 格式不正确，缺少 'universe' 列。")
             return []
        except Exception as e:
             print(f"DataHandler: 筛选资产池 '{universe_name}' 时出错: {e}")
             return []


        # 过滤掉那些在价格数据中不存在的代码 (使用 self.all_codes)
        valid_codes = [code for code in codes if code in self.all_codes]

        if not valid_codes:
             print(f"DataHandler: 警告 - 资产池 '{universe_name}' 中没有找到任何有效的证券代码（可能未在价格数据中）。")

        return valid_codes

    def load_benchmark_data(self, benchmark_path: str) -> Optional[pd.DataFrame]:
        """
        加载基准收益率数据。
        返回以 datetime 为索引，包含 'benchmark_return' 列的 DataFrame。
        如果失败则返回 None。
        """
        print(f"DataHandler: 正在加载基准数据: {benchmark_path}...")
        try:
            benchmark_df = pd.read_csv(benchmark_path)
            date_col = None
            return_col = None

            # 自动查找日期列
            potential_date_cols = [c for c in benchmark_df.columns if 'date' in c.lower()]
            if 'report_date' in benchmark_df.columns: date_col = 'report_date'
            elif potential_date_cols: date_col = potential_date_cols[0]
            else: raise ValueError("基准文件必须包含日期列。")

            # 自动查找收益率列
            potential_return_cols = [c for c in benchmark_df.columns if 'return' in c.lower() or 'yield' in c.lower() or 'change' in c.lower()]
            if 'default' in benchmark_df.columns: return_col = 'default' # 优先使用 'default'
            elif potential_return_cols: return_col = potential_return_cols[0]
            else: raise ValueError("基准文件中找不到收益率列。")

            # 打印使用的列名
            print(f"DataHandler: 使用 '{date_col}' 作为日期列，'{return_col}' 作为收益率列。")

            benchmark_df['datetime'] = pd.to_datetime(benchmark_df[date_col], errors='coerce')
            benchmark_df = benchmark_df.dropna(subset=['datetime']) # 移除无效日期
            benchmark_df = benchmark_df.set_index('datetime')
            benchmark_df = benchmark_df[~benchmark_df.index.duplicated(keep='first')] # 去重
            benchmark_df.sort_index(inplace=True)

            benchmark_df = benchmark_df.rename(columns={return_col: 'benchmark_return'})
            benchmark_df['benchmark_return'] = pd.to_numeric(benchmark_df['benchmark_return'], errors='coerce')
            benchmark_df = benchmark_df.dropna(subset=['benchmark_return']) # 移除无效收益率

            # 过滤日期范围 (基于 DataHandler 的全局日期范围)
            benchmark_df = benchmark_df.loc[self.start_date:self.end_date]

            if benchmark_df.empty:
                 print("DataHandler: 警告 - 加载的基准数据在指定日期范围内为空。")
                 return None # 返回 None

            print("DataHandler: 基准数据加载成功。")
            return benchmark_df[['benchmark_return']]

        except FileNotFoundError:
            # 返回 None 而不是抛异常，让调用者决定如何处理
            print(f"DataHandler: 警告 - 找不到基准数据文件: {benchmark_path}")
            return None
        except ValueError as e: # 捕获列名查找错误
             print(f"DataHandler: 警告 - 加载基准数据失败: {e}")
             return None
        except Exception as e:
            print(f"DataHandler: 加载基准数据时发生未知错误: {e}")
            import traceback
            traceback.print_exc()
            return None

