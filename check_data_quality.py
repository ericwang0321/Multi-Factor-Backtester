# -*- coding: utf-8 -*-
import duckdb
import pandas as pd

# 指向你生成的 Parquet 文件路径
parquet_path = 'data/processed/all_price_data.parquet'

def check_quality():
    # 1. 建立连接（DuckDB 可以直接查询文件）
    con = duckdb.connect()
    
    print("="*50)
    print("🚀 开始数据质量检查...")
    print("="*50)

    # 2. 统计总行数和总文件大小
    # 直接在 SQL 里引用文件路径即可
    summary = con.execute(f"""
        SELECT 
            count(*) as total_rows,
            count(distinct sec_code) as ticker_count,
            min(datetime) as start_date,
            max(datetime) as end_date
        FROM '{parquet_path}'
    """).df()
    
    print(f"📊 概览信息:")
    print(f"- 总行数: {summary['total_rows'][0]:,}")
    print(f"- 覆盖标的数量: {summary['ticker_count'][0]}")
    print(f"- 时间跨度: {summary['start_date'][0]} 至 {summary['end_date'][0]}")
    print("-" * 30)

    # 3. 检查各分类的数据分布
    print("📂 各分类资产分布:")
    category_dist = con.execute(f"""
        SELECT category_id, count(*) as rows, count(distinct sec_code) as tickers
        FROM '{parquet_path}'
        GROUP BY category_id
        ORDER BY rows DESC
    """).df()
    print(category_dist)
    print("-" * 30)

    # 4. 检查缺失值 (空值)
    # 检查最关键的价格和返回率字段
    print("🔍 关键字段空值检查:")
    null_checks = con.execute(f"""
        SELECT 
            sum(CASE WHEN close IS NULL THEN 1 ELSE 0 END) as close_nulls,
            sum(CASE WHEN simple_return IS NULL THEN 1 ELSE 0 END) as return_nulls,
            sum(CASE WHEN avg_price IS NULL OR avg_price = 0 THEN 1 ELSE 0 END) as zero_avg_price
        FROM '{parquet_path}'
    """).df()
    print(null_checks)
    print("-" * 30)

    # 5. 异常值检查 (比如涨跌幅超过 50% 的极端情况)
    print("🚩 极端涨跌幅预警 (可能存在除权除息未处理的情况):")
    outliers = con.execute(f"""
        SELECT datetime, sec_code, simple_return, close
        FROM '{parquet_path}'
        WHERE abs(simple_return) > 0.5
        ORDER BY abs(simple_return) DESC
        LIMIT 5
    """).df()
    if outliers.empty:
        print("✅ 未发现异常波动数据。")
    else:
        print(outliers)
    print("-" * 30)

    # 6. 数据完整性：统计每个标平均有多少天的历史
    print("📈 数据覆盖率最差的 5 个标的:")
    coverage = con.execute(f"""
        SELECT sec_code, count(*) as day_count
        FROM '{parquet_path}'
        GROUP BY sec_code
        ORDER BY day_count ASC
        LIMIT 5
    """).df()
    print(coverage)
    
    print("="*50)
    print("✅ 检查完毕！")

if __name__ == "__main__":
    check_quality()