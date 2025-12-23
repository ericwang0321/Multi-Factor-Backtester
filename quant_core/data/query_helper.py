# -*- coding: utf-8 -*-
import duckdb
import pandas as pd
import numpy as np

class DataQueryHelper:
    def __init__(self, storage_path='data/processed/all_price_data.parquet'):
        self.storage_path = storage_path

    def get_all_symbols(self):
        """获取数据库中所有的标的代码 (包含股票和基准)"""
        query = f"SELECT DISTINCT sec_code, category_id FROM '{self.storage_path}'"
        return duckdb.query(query).to_df()

    def get_history(self, symbol, start_date=None, end_date=None):
        """获取特定标的历史数据 (用于可视化展示)"""
        # 这个方法通用，查股票或查基准都可以
        sql = f"SELECT * FROM '{self.storage_path}' WHERE sec_code = '{symbol}'"
        if start_date: sql += f" AND datetime >= '{start_date}'"
        if end_date: sql += f" AND datetime <= '{end_date}'"
        sql += " ORDER BY datetime"
        return duckdb.query(sql).to_df()

    def get_market_summary(self):
        """获取市场概览统计"""
        sql = f"""
            SELECT category_id, 
                   count(distinct sec_code) as count, 
                   min(datetime) as start, 
                   max(datetime) as end
            FROM '{self.storage_path}'
            GROUP BY category_id
        """
        return duckdb.query(sql).to_df()

    def get_all_price_data(self):
        """
        [修改] 供 FactorEngine 使用：一次性加载全量数据并进行清洗
        *** 关键修改：增加了 WHERE category_id != 'benchmark' ***
        """
        # 1. 读取所有数据 (排除 Benchmark，防止因子计算混入 SPY 等 ETF)
        query = f"SELECT * FROM '{self.storage_path}' WHERE category_id != 'benchmark' ORDER BY sec_code, datetime"
        df = duckdb.query(query).to_df()
        
        # 2. 强制转换时间格式
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 3. 列名适配 (Mapping)
        if 'avg_price' in df.columns:
            df = df.rename(columns={'avg_price': 'vwap'})
            
        # 4. 数据完整性处理
        if 'amount' in df.columns:
            df['amount'] = df['amount'].fillna(0.0)
        else:
            df['amount'] = df['close'] * df['volume']

        # 5. 清理无效列 (Turnover/Cap 设为 NaN)
        if 'turnover' in df.columns:
            df['turnover'] = np.nan 
        if 'market_cap' in df.columns:
            df['market_cap'] = np.nan
        if 'shares_outstanding' in df.columns:
            df['shares_outstanding'] = np.nan

        return df

    def get_benchmark_returns(self, symbol: str) -> pd.Series:
        """
        [新增] 专门获取基准的收益率序列
        替代读取 CSV 的功能。
        返回: pd.Series (index=datetime, value=simple_return)
        """
        # 只查询时间和收益率，且必须是 benchmark 类型
        query = f"""
            SELECT datetime, simple_return 
            FROM '{self.storage_path}' 
            WHERE sec_code = '{symbol}' 
            AND category_id = 'benchmark'
            ORDER BY datetime
        """
        df = duckdb.query(query).to_df()
        
        if df.empty:
            # 如果没查到，返回空 Series
            return pd.Series(dtype=float)
            
        # 格式化
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        
        # 返回 Series
        return df['simple_return']