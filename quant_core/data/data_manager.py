# -*- coding: utf-8 -*-
import pandas as pd
import os
import time
import duckdb
from datetime import datetime
from .engine.us_equity_engine import USEquityEngine
# [新增] 导入 benchmark 同步函数
from .engine.benchmark_engine import sync_benchmarks

class DataManager:
    """
    工业级数据管理器：负责资产调度、增量同步、数据持久化及质量审计。
    """
    def __init__(self, ib_client=None):
        """
        初始化数据管理器。
        :param ib_client: 已连接的 IB 客户端实例。
        """
        self.ib = ib_client
        # 路径配置：对齐项目标准目录结构
        self.ref_path = 'data/reference/sec_code_category_grouped.csv'
        self.storage_path = 'data/processed/all_price_data.parquet'
        
        # 引擎配置：仅在需要同步时初始化美股引擎
        self.us_engine = USEquityEngine(self.ib) if self.ib else None
        
        # 字段布局：严格执行要求的列顺序
        self.columns_layout = [
            'id', 'datetime', 'sec_code', 'category_id', 'pre_close', 'open', 'high', 'low', 'close', 
            'volume', 'amount', 'create_time', 'avg_price', 'simple_return', 
            'shares_outstanding', 'turnover', 'market_cap'
        ]

    def run_pipeline(self, sync=True, check=True, duration='15 Y'):
        """
        数据流水线唯一入口。
        """
        print("="*60)
        print(f"🚀 数据流水线启动 | 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

        if sync:
            # 1. 执行原有的美股同步 (Equities)
            self._execute_sync(default_duration=duration)
            
            # [新增] 2. 执行基准同步 (Benchmarks)
            # 放在股票同步之后，确保逻辑解耦
            print("\n📡 步骤 1.5: 同步基准数据 (Benchmarks)...")
            try:
                sync_benchmarks(self.ib)
            except Exception as e:
                print(f"⚠️ 基准数据同步失败: {e}")
        
        if check:
            self._execute_quality_check()
            
        print("="*60)
        print("✨ 流水线所有任务处理完毕。")
        print("="*60)

    def _get_last_sync_info(self):
        """
        利用 DuckDB 极速获取本地文件的同步进度。
        """
        if not os.path.exists(self.storage_path):
            return {}
        
        con = duckdb.connect()
        try:
            # 扫描 Parquet 文件获取每个标的的最新日期
            df_last = con.execute(f"""
                SELECT sec_code, max(datetime) as last_date 
                FROM '{self.storage_path}' 
                GROUP BY sec_code
            """).df()
            return dict(zip(df_last['sec_code'], df_last['last_date']))
        except Exception as e:
            print(f"⚠️ 无法读取本地同步进度: {e}")
            return {}
        finally:
            con.close()

    def _execute_sync(self, default_duration):
        """
        核心同步逻辑：支持智能跳过与增量补全。
        """
        print("📡 步骤 1: 检查本地进度并执行智能同步...")
        if not self.ib or not self.ib.isConnected():
            raise RuntimeError("❌ 错误: 执行同步需要有效的 IBKR 连接。")

        # 1. 获取本地每个标的的最后日期及全局最晚日期
        last_dates = self._get_last_sync_info()
        global_max_date = max(last_dates.values()) if last_dates else None
        
        if global_max_date:
            print(f"📊 数据库全局最新日期: {global_max_date.date()}")

        # 2. 加载资产池清单
        if not os.path.exists(self.ref_path):
            raise FileNotFoundError(f"❌ 找不到资产清单文件: {self.ref_path}")
            
        universe_df = pd.read_csv(self.ref_path)
        cat_col = 'universe' if 'universe' in universe_df.columns else 'category_id'
        
        new_data_list = []
        today = datetime.now()
        total_tickers = len(universe_df)

        # 3. 遍历资产池执行增量拉取
        for i, row in universe_df.iterrows():
            symbol = row['sec_code']
            category = row[cat_col]
            last_date = last_dates.get(symbol)
            
            # --- 智能跳过逻辑：如果已追平全局进度且当前非交易时段，则跳过 ---
            if global_max_date and last_date == global_max_date:
                print(f"[{i+1}/{total_tickers}] ⏩ {symbol} 已是全局最新 ({last_date.date()})，跳过下载。")
                continue
            
            # --- 动态下载时长计算 ---
            if last_date:
                # 至少取 2 天以确保数据衔接
                days_diff = (today - last_date).days
                fetch_duration = f"{min(days_diff + 2, 365)} D" 
                print(f"[{i+1}/{total_tickers}] 📥 {symbol} 增量拉取: 自 {last_date.date()} 起的 {fetch_duration} 数据...")
            else:
                fetch_duration = default_duration
                print(f"[{i+1}/{total_tickers}] 🆕 {symbol} 首次全量拉取: {fetch_duration}...")
            
            try:
                # 调用专用引擎执行下载与初级计算
                data = self.us_engine.fetch_data(symbol, category, duration=fetch_duration)
                if not data.empty:
                    new_data_list.append(data)
            except Exception as e:
                print(f"⚠️ 下载 {symbol} 时发生异常: {e}")
            
            # 频率控制：防止触发 IBKR Pacing Violation
            time.sleep(1.2)

        # 4. 执行数据合并与持久化
        if new_data_list:
            self._merge_and_save(new_data_list)
        else:
            print("✅ 检查完毕：所有资产数据已是最新，无需更新。")

    def _merge_and_save(self, new_data_list):
        """
        合并新旧数据，执行去重并固化为 Parquet。
        """
        print("💾 正在合并数据并更新本地 Parquet 仓库...")
        new_df = pd.concat(new_data_list)
        
        if os.path.exists(self.storage_path):
            old_df = pd.read_parquet(self.storage_path)
            combined_df = pd.concat([old_df, new_df])
        else:
            combined_df = new_df

        # 数据去重：基于时间和代码确保数据唯一性
        combined_df = combined_df.drop_duplicates(subset=['datetime', 'sec_code'], keep='last')
        combined_df = combined_df.sort_values(['datetime', 'sec_code'])
        
        # 重新生成全局唯一自增 ID
        if 'id' in combined_df.columns:
            combined_df = combined_df.drop(columns=['id'])
        combined_df.insert(0, 'id', range(8000000, 8000000 + len(combined_df)))

        # 写入 Parquet
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        combined_df[self.columns_layout].to_parquet(self.storage_path, index=False, compression='snappy')
        print(f"✅ 更新成功：数据库当前总记录数: {len(combined_df):,}")

    def _execute_quality_check(self):
        """
        利用 DuckDB 审计数据质量，确保无空值与数据断层。
        """
        print("\n🔍 步骤 2: 开始数据质量自动审计...")
        if not os.path.exists(self.storage_path):
            print(f"❌ 审计终止：未找到文件 {self.storage_path}")
            return

        con = duckdb.connect()
        try:
            # 基础分布审计
            res = con.execute(f"""
                SELECT count(*) as rows, count(distinct sec_code) as tickers,
                       min(datetime) as start_v, max(datetime) as end_v
                FROM '{self.storage_path}'
            """).df()
            
            # 关键字段空值审计
            nulls = con.execute(f"SELECT count(*) FROM '{self.storage_path}' WHERE close IS NULL").fetchone()[0]
            
            print(f"- 数据行数: {res['rows'][0]:,}")
            print(f"- 标的个数: {res['tickers'][0]}")
            print(f"- 日期覆盖: {res['start_v'][0]} 至 {res['end_v'][0]}")
            
            if nulls == 0:
                print("💎 质量结论: 数据完整，无缺失字段。")
            else:
                print(f"⚠️ 质量预警: 发现 {nulls} 条缺失记录，请核查数据源！")
                
        except Exception as e:
            print(f"❌ 审计过程出错: {e}")
        finally:
            con.close()