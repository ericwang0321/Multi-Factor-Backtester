# -*- coding: utf-8 -*-
import os
import sys
import argparse
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm

# --- 引入项目模块 ---
try:
    from quant_core.data.query_helper import DataQueryHelper
    from quant_core.factors.engine import FactorEngine
except ImportError as e:
    print(f"❌ 导入出错: {e}")
    print("请确保你在项目根目录下运行此脚本，例如: python run_factor_computation.py")
    sys.exit(1)

# --- 配置 ---
DATA_PATH = 'data/processed/all_price_data.parquet'
OUTPUT_DIR = 'data/processed/factors'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"📁 创建目录: {path}")

def main(args):
    start_time = time.time()
    print(f"\n🚀 [{datetime.now().strftime('%H:%M:%S')}] 启动因子计算任务...")
    
    ensure_dir(OUTPUT_DIR)

    # 1. 初始化数据 (只加载一次)
    print(f"📥 正在加载基础行情数据: {DATA_PATH} ...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误: 找不到行情文件 {DATA_PATH}。请先运行数据同步脚本。")
        sys.exit(1)
        
    helper = DataQueryHelper(storage_path=DATA_PATH)
    
    # 初始化引擎 (这一步会将 DataFrame 转为 Xarray，是内存消耗最大的一步)
    # 但由于我们是批量计算，只用初始化一次，这非常划算
    try:
        engine = FactorEngine(query_helper=helper)
        # 预加载数据到内存 (xarray_data)
        engine._get_xarray_data()
    except Exception as e:
        print(f"❌ 引擎初始化失败: {e}")
        sys.exit(1)

    # 2. 确定要计算哪些因子
    all_registered = sorted(list(engine.FACTOR_REGISTRY.keys()))
    
    if args.factors:
        # 用户指定了特定因子
        target_factors = [f for f in args.factors if f in all_registered]
        invalid = [f for f in args.factors if f not in all_registered]
        if invalid:
            print(f"⚠️ 警告: 以下因子未注册，将被忽略: {invalid}")
    else:
        # 默认计算所有
        target_factors = all_registered

    print(f"📋 计划处理 {len(target_factors)} 个因子。")

    # 3. 循环计算
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    # 使用 tqdm 显示进度条
    pbar = tqdm(target_factors, desc="Computing", unit="factor")
    
    for factor_name in pbar:
        output_path = os.path.join(OUTPUT_DIR, f"{factor_name}.parquet")
        
        # --- 增量逻辑检查 ---
        file_exists = os.path.exists(output_path)
        
        # 如果文件存在，且没有开启强制刷新，则跳过 (增量模式: 新增Factor)
        if file_exists and not args.force:
            pbar.set_postfix_str(f"Skipped {factor_name}")
            skip_count += 1
            continue
            
        # 开始计算
        try:
            pbar.set_postfix_str(f"Calc {factor_name}")
            
            # 调用引擎的核心计算方法
            # 这里的 _compute_and_cache_factor 会利用 xarray 进行全向量化计算
            # 速度非常快
            factor_df = engine._compute_and_cache_factor(factor_name)
            
            if factor_df.empty:
                print(f"\n⚠️ {factor_name} 计算结果为空，跳过保存。")
                fail_count += 1
                continue
                
            # 存储为 Parquet
            # 使用宽表格式存储 (Index=datetime, Columns=sec_code)
            # 这种格式读取最快，且文件体积最小
            factor_df.to_parquet(output_path, compression='snappy')
            
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ {factor_name} 计算失败: {str(e)}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    # 4. 总结
    elapsed = time.time() - start_time
    print(f"\n{'='*40}")
    print(f"🎉 任务完成! 耗时: {elapsed:.2f} 秒")
    print(f"✅ 成功计算/更新: {success_count}")
    print(f"⏭️ 跳过 (已存在): {skip_count}")
    print(f"❌ 失败: {fail_count}")
    print(f"📂 存储位置: {OUTPUT_DIR}")
    print(f"{'='*40}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="因子批量预计算脚本")
    
    # 参数: 强制重新计算所有 (适用于: 增加了新的一天数据，或修改了因子公式)
    parser.add_argument('--force', '-f', action='store_true', 
                        help="强制重新计算并覆盖现有的因子文件 (用于数据更新或公式修改后)")
    
    # 参数: 指定计算哪些因子 (适用于: 调试特定因子)
    parser.add_argument('--factors', nargs='+', type=str, 
                        help="指定要计算的因子名称列表 (例如: --factors rsi momentum)")
    
    args = parser.parse_args()
    
    main(args)