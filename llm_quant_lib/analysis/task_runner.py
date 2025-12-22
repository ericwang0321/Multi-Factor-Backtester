# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
# 导入同一目录下的预处理器和研究引擎
from .preprocessor import FactorPreprocessor
from .research_engine import FactorResearchEngine
# 从上级目录导入你原有的因子引擎
from ..factor_engine import FactorEngine

class FactorTaskRunner:
    """
    因子 EDA 全流程调度器：连接计算引擎、清洗引擎与研究引擎。
    """
    
    def __init__(self, data_handler):
        """
        初始化时必须注册 factor_engine。
        :param data_handler: 用于提供原始价格数据的 DataHandler 实例。
        """
        # 核心：实例化你原来的 Xarray 因子计算引擎
        # 这就是 app.py 中 runner.factor_engine 访问的对象
        self.factor_engine = FactorEngine(data_handler)
        self.preprocessor = FactorPreprocessor()

    def _get_forward_returns(self, horizon=1):
        """
        从数据处理器提取价格并计算未来 N 日收益率 (预测目标)。
        """
        # 加载全量价格数据
        df = self.factor_engine.data_handler.load_data()
        
        # 确保按代码和日期排序以正确执行 shift
        df = df.sort_values(['sec_code', 'datetime'])
        
        # 计算 T 到 T+horizon 的未来收益率
        # shift(-horizon) 将未来的价格拉回到今天，以便与今天的因子匹配
        df['forward_return'] = df.groupby('sec_code')['close'].shift(-horizon) / df['close'] - 1
        
        return df[['datetime', 'sec_code', 'forward_return']]

    def run_analysis_pipeline(self, factor_name, horizon=1, n_groups=5):
        """
        执行完整的分析流水线：计算 -> 清洗 -> 评价 -> 存储。
        """
        print(f"🚀 启动因子分析: {factor_name} (预测周期: {horizon}天)")

        # 1. 计算因子全序列 (调用你原有的 BaseAlpha 逻辑)
        # 返回的是 (datetime x sec_code) 的宽表
        factor_df_wide = self.factor_engine._compute_and_cache_factor(factor_name)
        
        # 将宽表转为长表，以便与收益率对齐
        factor_df = factor_df_wide.stack(future_stack=True).reset_index()
        factor_df.columns = ['datetime', 'sec_code', 'factor_value']

        # 2. 获取目标收益率数据
        returns_df = self._get_forward_returns(horizon=horizon)

        # 3. 因子值与未来收益率对齐
        merged_df = pd.merge(factor_df, returns_df, on=['datetime', 'sec_code'], how='inner')
        merged_df = merged_df.dropna(subset=['factor_value', 'forward_return'])

        # 4. 横截面清洗 (Preprocessor)
        print("🧼 正在执行横截面清洗 (去极值与标准化)...")
        # 确保每一天的处理是独立的，避免时序偏见
        def clean_daily(group):
            # 处理离群值 (Winsorization)
            group['factor_value'] = self.preprocessor.handle_outliers(group['factor_value'])
            # 标准化 (Z-Score)
            group['factor_value'] = self.preprocessor.standardize(group['factor_value'])
            return group
        
        cleaned_df = merged_df.groupby('datetime', group_keys=False).apply(clean_daily)

        # 5. 性能指标计算 (ResearchEngine)
        print("📊 正在计算 IC 指标与分层收益...")
        res_engine = FactorResearchEngine(cleaned_df)
        
        # 计算 Rank IC 时间序列
        ic_series = res_engine.calculate_ic(method='rank')
        # 计算核心统计量 (Mean IC, IR)
        stats = res_engine.calculate_stats(ic_series)
        # 计算 5 组分层累积收益
        _, cum_group_ret, _ = res_engine.calculate_group_returns(n_groups=n_groups)

        # 6. 分析结果持久化
        output_dir = 'data/processed/factor_analysis'
        os.makedirs(output_dir, exist_ok=True)
        
        # 将 IC 序列存入 Parquet 供 App 绘图使用
        output_file = f"{output_dir}/{factor_name}_analysis.parquet"
        ic_df = ic_series.to_frame(name='rank_ic').reset_index()
        ic_df.to_parquet(output_file, index=False)

        print(f"✅ 因子分析完成。存入: {output_file}")
        return stats, ic_series, cum_group_ret