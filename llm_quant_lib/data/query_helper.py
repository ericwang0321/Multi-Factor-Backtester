# -*- coding: utf-8 -*-
import duckdb
import pandas as pd

class DataQueryHelper:
    def __init__(self, storage_path='data/processed/all_price_data.parquet'):
        self.storage_path = storage_path

    def get_all_symbols(self):
        """获取数据库中所有的标的代码"""
        query = f"SELECT DISTINCT sec_code, category_id FROM '{self.storage_path}'"
        return duckdb.query(query).to_df()

    def get_history(self, symbol, start_date=None, end_date=None):
        """获取特定标的历史数据"""
        sql = f"SELECT * FROM '{self.storage_path}' WHERE sec_code = '{symbol}'"
        if start_date: sql += f" AND datetime >= '{start_date}'"
        if end_date: sql += f" AND datetime <= '{end_date}'"
        sql += " ORDER BY datetime"
        return duckdb.query(sql).to_df()

    def get_market_summary(self):
        """获取市场概览统计 (用于可视化展示)"""
        sql = f"""
            SELECT category_id, 
                   count(distinct sec_code) as count, 
                   min(datetime) as start, 
                   max(datetime) as end
            FROM '{self.storage_path}'
            GROUP BY category_id
        """
        return duckdb.query(sql).to_df()