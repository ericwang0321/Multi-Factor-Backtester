# -*- coding: utf-8 -*-
import duckdb
import pandas as pd
import numpy as np

class DataQueryHelper:
    def __init__(self, storage_path='data/processed/all_price_data.parquet'):
        self.storage_path = storage_path

    def get_all_symbols(self):
        """获取数据库中所有的标的代码"""
        query = f"SELECT DISTINCT sec_code, category_id FROM '{self.storage_path}'"
        return duckdb.query(query).to_df()

    def get_history(self, symbol, start_date=None, end_date=None):
        """获取特定标的历史数据 (用于可视化展示)"""
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
        [新增] 供 FactorEngine 使用：一次性加载全量数据并进行清洗
        """
        # 1. 读取所有数据 (按时间和标的排序)
        query = f"SELECT * FROM '{self.storage_path}' ORDER BY sec_code, datetime"
        df = duckdb.query(query).to_df()
        
        # 2. 强制转换时间格式
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 3. 列名适配 (Mapping)
        # 因子库通常使用 'vwap'，而您的数据中可能是 'avg_price'
        if 'avg_price' in df.columns:
            df = df.rename(columns={'avg_price': 'vwap'})
            
        # 4. 数据完整性处理
        # 既然您确认 amount 是有效的，我们直接使用。
        # 如果 amount 某些行缺失，可以用 0 填充以防报错
        if 'amount' in df.columns:
            df['amount'] = df['amount'].fillna(0.0)
        else:
            # 万一完全没有 amount 列，才进行估算（兜底逻辑）
            df['amount'] = df['close'] * df['volume']

        # 5. [关键] 清理无效列
        # 您的 turnover 和 market_cap 都是 0，直接设为 NaN。
        # 这样依赖这些列的因子计算时会产出 NaN，而不是错误的 0 值。
        if 'turnover' in df.columns:
            df['turnover'] = np.nan 
        if 'market_cap' in df.columns:
            df['market_cap'] = np.nan
        if 'shares_outstanding' in df.columns:
            df['shares_outstanding'] = np.nan

        return df