# -*- coding: utf-8 -*-
import pandas as pd
import os
import time
from .engine.us_equity_engine import USEquityEngine

class DataManager:
    def __init__(self, ib_client):
        self.ib = ib_client
        # 路径对齐你的项目结构
        self.ref_path = 'data/reference/sec_code_category_grouped.csv'
        self.storage_path = 'data/processed/all_price_data.parquet'
        self.us_engine = USEquityEngine(self.ib)

    def sync_all_markets(self):
        """
        根据原始 CSV 中的类别，分发给不同的 Engine 执行下载
        """
        if not os.path.exists(self.ref_path):
            raise FileNotFoundError(f"找不到资产定义文件: {self.ref_path}")

        # 加载资产池定义 
        universe_df = pd.read_csv(self.ref_path)
        
        # --- 修复 KeyError: 'universe' 的核心逻辑 ---
        # 自动检测列名：优先找 'universe'，找不到就找 'category_id'
        if 'universe' in universe_df.columns:
            cat_col = 'universe'
        elif 'category_id' in universe_df.columns:
            cat_col = 'category_id'
        else:
            raise KeyError(f"资产定义文件 {self.ref_path} 必须包含 'universe' 或 'category_id' 列。")
        # ------------------------------------------

        all_market_data = []

        for _, row in universe_df.iterrows():
            symbol = row['sec_code']
            category = row[cat_col] # 使用检测到的列名
            
            # 模块化分发逻辑
            if any(key in category for key in ['equity', 'bond', 'commodity', 'alternative']):
                # 只要属于这些类别，都走 US 引擎（因为都在 IBKR 美股市场）
                try:
                    data = self.us_engine.fetch_data(symbol, category)
                    if not data.empty:
                        all_market_data.append(data)
                except Exception as e:
                    print(f"⚠️ 下载 {symbol} 时出错: {e}")
            
            # 频率控制，防止被 IBKR 断开
            time.sleep(1.2)

        if all_market_data:
            final_df = pd.concat(all_market_data)
            
            # 按照你要求的格式生成 ID 并排列列顺序
            final_df.insert(0, 'id', range(8000000, 8000000 + len(final_df)))
            
            # 确保保存目录存在
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # 保存为 Parquet
            final_df.to_parquet(self.storage_path, index=False, compression='snappy')
            print(f"✅ 同步完成！数据已保存至 {self.storage_path}")
        else:
            print("❌ 未下载到任何有效数据。")