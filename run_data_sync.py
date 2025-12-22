# -*- coding: utf-8 -*-
from ib_insync import IB
from llm_quant_lib.data.data_manager import DataManager
import yaml
import os

def main():
    # 1. 初始化 IB 客户端
    ib = IB()
    
    try:
        # 2. 连接到 TWS (请确保 TWS 已打开)
        print("正在连接 IBKR TWS (7497)...")
        ib.connect('127.0.0.1', 7497, clientId=10) # 使用独立的 clientId
        
        # 3. 初始化数据管理器
        # 它会自动去读取 data/reference/sec_code_category_grouped.csv
        dm = DataManager(ib)
        
        # 4. 执行全市场同步
        # 此处会根据 CSV 里的分类自动调用不同的 Engine (如 USEquityEngine)
        print("🚀 开始执行全市场数据同步任务...")
        dm.sync_all_markets()
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        
    finally:
        # 5. 断开连接
        if ib.isConnected():
            ib.disconnect()
            print("断开 IBKR 连接。")

if __name__ == "__main__":
    main()