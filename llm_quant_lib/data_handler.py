# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
from typing import List, Dict, Optional

class DataHandler:
    """
    负责从数据库或 CSV 文件加载、处理和提供原始价格及资產池数据。
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
        self.universe_df: Optional[pd.DataFrame] = None # 资產池定义
        self.all_codes: List[str] = [] # 所有加载的证券代码
        self.pivot_cache: Dict[str, pd.DataFrame] = {} # 缓存宽格式数据

        # --- 美股ETF代码列表 (如果需要从 DB 加载时使用) ---
        self.us_codes = [
             'KSA.P', 'CBON.P', 'CQQQ.P', 'KWEB.P', 'MCHI.O', 'PFF.O', 'ICLN.O', 'ACWI.O', 'DBA.P', 'SMH.O', 'INDA.B', 'USO.P','EWT.P', 'VXUS.O', 'EWZ.P', 'BWX.P', 'EWG.P', 'EMB.O','IEMG.P', 'EMXC.O', 'EWJ.P', 'SPY.P', 'IUSG.O', 'IUSB.O','IGOV.O', 'VGK.P', 'IAGG.B', 'EWQ.P', 'EWA.P', 'IBB.O','DVY.O', 'QQQ.O', 'VONG.O', 'VTWO.O', 'IEF.O', 'SHY.O','MBB.O', 'TIP.P', 'IGSB.O', 'FLOT.B', 'VTIP.O', 'SHV.O','BIL.P', 'TLT.O', 'HYG.P', 'BNDX.O', 'EWU.P', 'BITO.P','CPER.P', 'SLV.P', 'VCLT.O', 'GLD.P', 'EWH.P', 'EFA.P','EWY.P', 'VEGI.P', 'XLE.P', 'VNQ.P', 'VTI.P', 'VEU.P','DBC.P', 'MDY.P', 'SPAB.P', 'VUG.P', 'VTV.P', 'VOT.P','VOE.P', 'VBK.P', 'VBR.P', 'SCZ.O', 'VNQI.O', 'PSP.P','BND.O', 'BSV.P', 'GSG.P', 'IEI.O', 'SPYG.P', 'SPYV.P','XRT.P', 'XLF.P', 'XHB.P', 'PPH.O', 'IJJ.P', 'IJK.P','IJS.P', 'IJT.O', 'JNK.P', 'CWB.P', 'ICVT.B','MINT.P', 'ANGL.O', 'XLV.P', 'XLB.P', 'XLC.P', 'XLI.P','XLK.P', 'XLP.P', 'XLU.P', 'XMMO.P', 'XLRE.P', 'XLY.P'
        ] # 省略部分代码

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
                print("DataHandler: CSV 数据加载成功。")
                loaded_from_csv = True
            except Exception as e:
                print(f"DataHandler: 从 CSV 加载数据失败: {e}。将尝试从数据库加载。")
                self.raw_df = None

        if not loaded_from_csv:
            if self.db_config:
                print("DataHandler: 正在从数据库加载数据...")
                self.raw_df = self._load_from_db()
                if self.raw_df is not None and self.csv_path:
                    print(f"DataHandler: 将数据保存到 '{self.csv_path}'...")
                    output_dir = os.path.dirname(self.csv_path)
                    os.makedirs(output_dir, exist_ok=True)
                    self.raw_df.to_csv(self.csv_path, index=False)
                    print("DataHandler: 数据库数据已加载并保存到 CSV。")
            else:
                raise FileNotFoundError(f"CSV 文件 '{self.csv_path}' 未找到，且未提供数据库配置。无法加载数据。")

        if self.raw_df is None:
             raise RuntimeError("DataHandler: 无法从任何来源加载价格数据。")

        # 统一进行日期过滤和代码列表更新
        self.raw_df['datetime'] = pd.to_datetime(self.raw_df['datetime'])
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
        # --- 省略了数据库连接和查询的详细代码，与之前版本相同 ---
        # --- 假设查询成功，返回了 df_merged ---
        if not self.db_config:
            print("DataHandler: 未提供数据库配置。")
            return None

        engine = create_engine(
            f"postgresql+psycopg2://{self.db_config['username']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
        df_merged: Optional[pd.DataFrame] = None
        try:
            # --- 下载价格数据 ---
            us_table = 'ods_mkt_ths_etf_us_daily_price_adj'
            us_codes_str = ', '.join([f"'{code}'" for code in self.us_codes]) # 修正 SQL IN 子句
            us_query = f"SELECT * FROM {us_table} WHERE sec_code IN ({us_codes_str});"
            print("DataHandler: 正在下载价格数据...")
            df_price = pd.read_sql(us_query, con=engine)
            print(f"DataHandler: 价格数据下载完成，共 {len(df_price)} 条记录。")

            df_price['datetime'] = pd.to_datetime(df_price['datetime'])
            df_price = df_price.sort_values(by=['sec_code', 'datetime'])

            # --- 下载基金信息 ---
            fund_info_table = 'ods_mkt_ths_etf_us_daily_fund_info'
            fund_info_query = f"SELECT datetime, sec_code, fund_net_value, fund_total_asset FROM {fund_info_table} WHERE sec_code IN ({us_codes_str});"
            print("DataHandler: 正在下载基金信息...")
            df_fund_info = pd.read_sql(fund_info_query, con=engine)
            print(f"DataHandler: 基金信息下载完成，共 {len(df_fund_info)} 条记录。")

            # --- 合并与处理 ---
            df_fund_info['datetime'] = pd.to_datetime(df_fund_info['datetime'])
            df_merged = pd.merge(df_price, df_fund_info, on=['datetime', 'sec_code'], how='left')
            df_merged = df_merged.sort_values(by=['sec_code', 'datetime'])

            df_merged['amount'] = df_merged['volume'] * df_merged['avg_price']
            df_merged.rename(columns={'fund_total_asset': 'market_cap'}, inplace=True)

            print("DataHandler: 正在填充缺失值...")
            columns_to_fill = ['open', 'high', 'low', 'close', 'volume', 'amount', 'avg_price', 'fund_net_value', 'market_cap']
            existing_columns_to_fill = [col for col in columns_to_fill if col in df_merged.columns]

            def robust_fill(group):
                group[existing_columns_to_fill] = group[existing_columns_to_fill].interpolate(method='linear', limit_direction='both')
                group[existing_columns_to_fill] = group[existing_columns_to_fill].ffill()
                group[existing_columns_to_fill] = group[existing_columns_to_fill].bfill()
                return group

            df_merged = df_merged.groupby('sec_code', group_keys=False).apply(robust_fill)
            df_merged[existing_columns_to_fill] = df_merged[existing_columns_to_fill].fillna(0)
            print("DataHandler: 缺失值填充完成。")

            print("DataHandler: 正在计算衍生指标...")
            df_merged['simple_return'] = df_merged.groupby('sec_code')['close'].pct_change().fillna(0)
            df_merged['fund_net_value'] = df_merged['fund_net_value'].replace(0, np.nan)
            df_merged['shares_outstanding'] = df_merged['market_cap'] / df_merged['fund_net_value']
            df_merged['shares_outstanding'].replace([np.inf, -np.inf], np.nan, inplace=True)
            df_merged['shares_outstanding'] = df_merged.groupby('sec_code')['shares_outstanding'].ffill().bfill().fillna(0)
            df_merged['turnover'] = df_merged['volume'] / df_merged['shares_outstanding'].replace(0, np.nan)
            df_merged['turnover'].replace([np.inf, -np.inf], np.nan, inplace=True)
            df_merged['turnover'] = df_merged.groupby('sec_code')['turnover'].ffill().bfill().fillna(0)
            print("DataHandler: 衍生指标计算完成。")

            final_columns = ['datetime', 'sec_code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'avg_price', 'simple_return', 'shares_outstanding', 'turnover', 'market_cap']
            final_columns_exist = [col for col in final_columns if col in df_merged.columns]
            if 'avg_price' in df_merged.columns:
                df_merged.rename(columns={'avg_price': 'vwap'}, inplace=True)
                if 'vwap' not in final_columns_exist:
                     final_columns_exist.append('vwap')

            df_merged = df_merged[final_columns_exist].copy() # 确保只保留需要的列

        except Exception as e:
            print(f"DataHandler: 数据库操作失败: {e}")
            return None
        finally:
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
            pd.DataFrame: 宽格式的價格數據。
        """
        cache_key = price_type
        if cache_key in self.pivot_cache:
            pivot_df = self.pivot_cache[cache_key]
        else:
            if self.raw_df is None:
                self.load_data()
            if self.raw_df is None or price_type not in self.raw_df.columns:
                raise ValueError(f"无法加载数据或找不到價格类型 '{price_type}'")

            print(f"DataHandler: 正在生成 {price_type} 的宽格式数据...")
            pivot_df = self.raw_df.pivot(index='datetime', columns='sec_code', values=price_type)
            pivot_df.sort_index(inplace=True)
            # 基础填充
            pivot_df = pivot_df.ffill().bfill()
            self.pivot_cache[cache_key] = pivot_df
            print(f"DataHandler: {price_type} 宽格式数据已准备就绪。")

        # 根据传入的 codes 进行过滤
        if codes:
            codes_in_df = [c for c in codes if c in pivot_df.columns]
            if not codes_in_df:
                 print(f"DataHandler: 警告 - 在 {price_type} 数据中未找到指定的证券代码: {codes}")
                 return pd.DataFrame(index=pivot_df.index) # 返回空 DataFrame 但保留日期索引
            return pivot_df[codes_in_df]
        else:
            return pivot_df

    def load_universe_data(self, universe_path: str) -> pd.DataFrame:
        """
        加载资產池定义文件 (sec_code_category_grouped.csv) 并缓存。
        """
        if self.universe_df is not None:
            return self.universe_df

        print(f"DataHandler: 正在加载资產池定义文件: {universe_path}...")
        try:
            u_df = pd.read_csv(universe_path)
            # 重命名列
            if 'category_id' in u_df.columns:
                u_df = u_df.rename(columns={'category_id': 'universe'})
            if 'sec_code' not in u_df.columns or 'universe' not in u_df.columns:
                 raise ValueError("资產池文件必须包含 'sec_code' 和 'category_id'/'universe' 列。")

            # 确保所有已加载的代码都有资產池归属，如果没有，赋予 'Unknown'
            if self.raw_df is None: self.load_data() # 确保 self.all_codes 已填充
            missing_codes = set(self.all_codes) - set(u_df['sec_code'].unique())
            if missing_codes:
                 print(f"DataHandler: 警告 - 以下证券代码未在资產池文件中定义，将归类为 'Unknown': {missing_codes}")
                 unknown_df = pd.DataFrame({'sec_code': list(missing_codes), 'universe': 'Unknown'})
                 u_df = pd.concat([u_df, unknown_df], ignore_index=True)

            self.universe_df = u_df[['sec_code', 'universe']].drop_duplicates().set_index('sec_code')
            print("DataHandler: 资產池定义加载成功。")
            return self.universe_df.reset_index() # 返回 DataFrame 格式以便后续使用
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到资產池定义文件: {universe_path}")
        except Exception as e:
            raise RuntimeError(f"加载资產池定义文件时出错: {e}")

    def get_codes_in_universe(self, universe_name: str) -> List[str]:
        """
        获取指定资產池包含的证券代码列表。支持 'All'。
        """
        if self.raw_df is None: self.load_data() # 确保 all_codes 已填充

        if universe_name.lower() == 'all':
            return self.all_codes # 返回所有已加载的证券代码

        if self.universe_df is None:
            raise RuntimeError("请先调用 load_universe_data 加载资產池定义。")

        # 从 DataFrame 中查找
        codes = self.universe_df[self.universe_df['universe'] == universe_name]['sec_code'].tolist()

        # 过滤掉那些在价格数据中不存在的代码
        valid_codes = [code for code in codes if code in self.all_codes]

        if not valid_codes:
             print(f"DataHandler: 警告 - 资產池 '{universe_name}' 中没有找到任何有效的证券代码（可能未在价格数据中）。")

        return valid_codes

    def load_benchmark_data(self, benchmark_path: str) -> pd.DataFrame:
        """
        加载基准收益率数据并缓存。
        """
        # --- 省略了加载逻辑，与之前版本相同 ---
        # --- 确保返回的 DataFrame 索引为 datetime，列名为 'benchmark_return' ---
        print(f"DataHandler: 正在加载基准数据: {benchmark_path}...")
        try:
            benchmark_df = pd.read_csv(benchmark_path)
            date_col = 'report_date' # 假设日期列名
            return_col = 'default'    # 假设收益率列名

            if date_col not in benchmark_df.columns:
                # 尝试查找其他可能的日期列名
                potential_date_cols = [c for c in benchmark_df.columns if 'date' in c.lower()]
                if not potential_date_cols:
                    raise ValueError(f"基准文件必须包含日期列 (如 '{date_col}')。")
                date_col = potential_date_cols[0]
                print(f"DataHandler: 警告 - 未找到 '{date_col}' 列，将使用 '{date_col}' 作为日期列。")

            benchmark_df['datetime'] = pd.to_datetime(benchmark_df[date_col])
            benchmark_df = benchmark_df.set_index('datetime')
            benchmark_df = benchmark_df[~benchmark_df.index.duplicated(keep='first')] # 去重
            benchmark_df.sort_index(inplace=True)

            if return_col not in benchmark_df.columns:
                 potential_return_cols = [c for c in benchmark_df.columns if 'return' in c.lower() or 'yield' in c.lower() or 'change' in c.lower()]
                 if not potential_return_cols:
                     raise ValueError(f"基准文件中找不到收益率列 (如 '{return_col}')。")
                 return_col = potential_return_cols[0]
                 print(f"DataHandler: 警告 - 未找到 '{return_col}' 列，将使用 '{return_col}' 作为基准收益率列。")

            benchmark_df = benchmark_df.rename(columns={return_col: 'benchmark_return'})
            benchmark_df['benchmark_return'] = pd.to_numeric(benchmark_df['benchmark_return'], errors='coerce')

            # 过滤日期范围
            benchmark_df = benchmark_df.loc[self.start_date:self.end_date]
            print("DataHandler: 基准数据加载成功。")
            return benchmark_df[['benchmark_return']]

        except FileNotFoundError:
            raise FileNotFoundError(f"找不到基准数据文件: {benchmark_path}")
        except Exception as e:
            raise RuntimeError(f"加载基准数据时出错: {e}")

