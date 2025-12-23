# run_data_sync.py
from ib_insync import IB
from quant_core.data.data_manager import DataManager

ib = IB()
try:
    ib.connect('127.0.0.1', 7497, clientId=10)
    
    # 只需要这一行，就完成了：
    # 1. 自动识别 CSV 列名
    # 2. 调用 Engine 下载 15 年数据
    # 3. 计算 derivative 字段
    # 4. 保存为 Parquet
    # 5. 自动运行 DuckDB 质检
    dm = DataManager(ib)
    dm.run_pipeline(sync=True, check=True)
    
finally:
    ib.disconnect()