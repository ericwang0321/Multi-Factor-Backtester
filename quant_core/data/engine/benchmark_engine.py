# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from ib_insync import *
import duckdb
from datetime import datetime
import os
import traceback

# --- 配置 ---
IB_PORT = 7497  # 模拟盘端口
CLIENT_ID = 99 
# 路径修正：确保指向正确的 Parquet 文件
PARQUET_PATH = '../../../data/processed/all_price_data.parquet'

BENCHMARKS = {
    'SPY': 'SPY',   # S&P 500
    'ACWI': 'ACWI', # Global Equity
    'AGG': 'AGG',   # Global Bond
    'GSG': 'GSG'    # Commodity
}

def fetch_benchmark_data(ib: IB, symbol: str):
    """下载单个 ETF 全量数据"""
    print(f"📥 Downloading {symbol} (Adjusted - ALL HISTORY)...")
    
    contract = Stock(symbol, 'SMART', 'USD')
    details = ib.reqContractDetails(contract)
    if not details:
        print(f"⚠️ Contract not found: {symbol}")
        return pd.DataFrame()
    contract = details[0].contract

    # 下载 50 年数据
    bars = ib.reqHistoricalData(
        contract, endDateTime='', durationStr='50 Y', barSizeSetting='1 day',
        whatToShow='ADJUSTED_LAST', useRTH=True, formatDate=1
    )
    if not bars: return pd.DataFrame()

    df = util.df(bars)
    
    # --- 数据清洗 ---
    df['datetime'] = pd.to_datetime(df['date'])
    df['sec_code'] = symbol
    df['category_id'] = 'benchmark'
    
    # [核心修复 1] 强制把 create_time 转为字符串，与旧数据兼容
    df['create_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 关键字段补全
    if 'average' in df.columns:
        df = df.rename(columns={'average': 'avg_price'})
    else:
        df['avg_price'] = df['close']
    
    df['amount'] = df['volume'] * df['close']
    df['pre_close'] = df['close'].shift(1)
    df['simple_return'] = df['close'].pct_change()
    
    # 填充缺失列
    df['id'] = 0 
    df['barCount'] = df['barCount'] if 'barCount' in df.columns else 0
    df['shares_outstanding'] = 0.0
    df['turnover'] = 0.0
    df['market_cap'] = 0.0
    
    # 补全 'id' 到列表
    required_cols = [
        'id', 'datetime', 'sec_code', 'open', 'high', 'low', 'close', 
        'volume', 'amount', 'avg_price', 'category_id', 
        'pre_close', 'simple_return', 'shares_outstanding', 
        'turnover', 'market_cap', 'create_time', 'barCount'
    ]
    
    available_cols = [c for c in required_cols if c in df.columns]
    return df[available_cols].dropna(subset=['close'])

def update_parquet_storage(new_df):
    """合并数据并自动处理 ID"""
    if new_df.empty: return

    abs_parquet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), PARQUET_PATH))
    print(f"💾 Updating Storage: {abs_parquet_path}")

    con = duckdb.connect()
    try:
        if os.path.exists(abs_parquet_path):
            # 1. 读取旧数据
            existing_df = con.execute(f"SELECT * FROM '{abs_parquet_path}'").df()
            
            # 剔除旧 Benchmark
            df_clean = existing_df[existing_df['category_id'] != 'benchmark'].copy()
            
            # [核心修复 2] 双重保险：确保两边都是字符串类型
            if 'create_time' in df_clean.columns:
                df_clean['create_time'] = df_clean['create_time'].astype(str)
            if 'create_time' in new_df.columns:
                new_df['create_time'] = new_df['create_time'].astype(str)

            # ID 自增逻辑
            max_id = df_clean['id'].max() if not df_clean.empty else 0
            new_df['id'] = range(max_id + 1, max_id + 1 + len(new_df))
            
            # 合并
            combined_df = pd.concat([df_clean, new_df], ignore_index=True)
        else:
            new_df['id'] = range(1, len(new_df) + 1)
            # 确保类型是字符串
            new_df['create_time'] = new_df['create_time'].astype(str)
            combined_df = new_df

        # 2. 写入
        combined_df.sort_values(['sec_code', 'datetime'], inplace=True)
        combined_df.to_parquet(abs_parquet_path, index=False)
        print(f"✅ Success! Database updated. Total rows: {len(combined_df)}")
        
    except Exception as e:
        print(f"❌ Storage Error: {e}")
        traceback.print_exc()
    finally:
        con.close()

def main():
    ib = IB()
    try:
        print(f"Connecting to IBKR on port {IB_PORT}...")
        ib.connect('127.0.0.1', IB_PORT, clientId=CLIENT_ID)
        print("✅ Connected.")
        
        all_data = []
        for name, symbol in BENCHMARKS.items():
            df = fetch_benchmark_data(ib, symbol)
            if not df.empty:
                print(f"   -> Got {len(df)} rows for {symbol}")
                all_data.append(df)
        
        if all_data:
            full_df = pd.concat(all_data)
            update_parquet_storage(full_df)
        else:
            print("⚠️ No data downloaded.")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        ib.disconnect()

if __name__ == '__main__':
    main()