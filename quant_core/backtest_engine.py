# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional

# [修改 1] 从新的策略包路径导入基类
from .portfolio import Portfolio 
from .strategies.base import BaseStrategy

# [修改 2] 导入 QueryHelper
from .data.query_helper import DataQueryHelper

# [修改 3] 移除 FactorEngine 的显式依赖
# 在新架构中，因子数据由外部脚本(run_backtest.py/app.py)准备好并注入给 Strategy
# Engine 不再直接指挥 FactorEngine 算数，只负责撮合交易

class BacktestEngine:
    """
    负责协调数据、投资组合和策略执行回测循环。
    适配 Parquet 数据源和新策略架构。
    """
    
    def __init__(self, start_date: str, end_date: str, config: Dict, strategy: BaseStrategy, query_helper: DataQueryHelper, universe_to_run: str = 'All'):
        self.effective_start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        self.config = config
        self.strategy = strategy
        self.query_helper = query_helper
        self.universe_to_run = universe_to_run

        # 1. 数据加载与 Pivot (保持不变)
        print("BacktestEngine: Loading and pivoting price data...")
        raw_df = self.query_helper.get_all_price_data()
        
        # 过滤 Universe
        if self.universe_to_run != 'All':
            # 假设 universe_to_run 是 category_id (如 'SP500', 'ETF')
            all_symbols = self.query_helper.get_all_symbols()
            # 注意：需确保 category_id 列存在
            if 'category_id' in all_symbols.columns:
                valid_codes = all_symbols[all_symbols['category_id'] == self.universe_to_run]['sec_code'].unique()
                raw_df = raw_df[raw_df['sec_code'].isin(valid_codes)]
                self.codes_in_universe = valid_codes.tolist()
            else:
                # 如果没有 category 概念，默认全选
                self.codes_in_universe = raw_df['sec_code'].unique().tolist()
        else:
            self.codes_in_universe = raw_df['sec_code'].unique().tolist()

        # Pivot 生成宽表
        try:
            self.all_open_prices = raw_df.pivot(index='datetime', columns='sec_code', values='open').ffill()
            self.all_close_prices = raw_df.pivot(index='datetime', columns='sec_code', values='close').ffill()
        except Exception as e:
            raise ValueError(f"Failed to pivot price data: {e}")

        # 截取回测时间段
        self.open_prices = self.all_open_prices.loc[self.effective_start_date:self.end_date].copy()
        self.close_prices = self.all_close_prices.loc[self.effective_start_date:self.end_date].copy()

        # 2. 初始化投资组合
        self.portfolio = Portfolio(
            self.open_prices,
            self.close_prices,
            config.get('INITIAL_CAPITAL', 1_000_000),
            config.get('COMMISSION_RATE', 0.001),
            config.get('SLIPPAGE', 0.0005)
        )
        self.portfolio.config = config

        # 3. 准备日期序列
        self.trade_dates = self.open_prices.index
        self.trade_execution_dates = self._get_trade_execution_dates(self.trade_dates, config)
        
        print(f"BacktestEngine: 调仓模式 - {'固定天数' if config.get('REBALANCE_DAYS') else '按月'}调仓")
        print(f"BacktestEngine: 预计执行调仓次数: {len(self.trade_execution_dates)}")

    def _get_trade_execution_dates(self, price_index: pd.DatetimeIndex, config: Dict) -> pd.DatetimeIndex:
        """计算实际调仓执行日期"""
        if price_index.empty:
            return pd.DatetimeIndex([])

        rebalance_days = config.get('REBALANCE_DAYS')
        if rebalance_days is not None and rebalance_days > 0:
            exec_dates = price_index[::rebalance_days]
            return pd.DatetimeIndex(exec_dates)
        
        rebalance_months = config.get('REBALANCE_MONTHS', 1)
        month_starts = price_index.to_period('M').start_time.unique()
        
        if rebalance_months > 1:
            month_starts = month_starts[::rebalance_months]
            
        execution_dates = []
        for month_start in month_starts:
             found_days = price_index[price_index >= month_start]
             if not found_days.empty:
                  actual_exec_date = found_days[0]
                  if actual_exec_date.month == month_start.month:
                       execution_dates.append(actual_exec_date)
        return pd.DatetimeIndex(sorted(list(set(execution_dates))))

    def run(self) -> tuple[Optional[pd.DataFrame], Optional[Portfolio]]:
        """执行回测循环"""
        desc_text = f"回测进度 ({self.universe_to_run})"
        progress_bar = tqdm(self.trade_dates, desc=desc_text)

        for current_date in progress_bar:
            # 每日更新净值
            self.portfolio.update_portfolio_value(current_date)

            # 触发调仓
            if current_date in self.trade_execution_dates:
                # [修改 4] 核心逻辑变更：直接调用 Strategy.on_bar
                # 在新架构中，Strategy 已经持有数据了，不需要我们从这里传 raw factor snapshot
                
                # 注意：current_date 是 datetime 类型
                # 调用策略获取目标仓位 {code: weight}
                target_weights = self.strategy.on_bar(current_date, self.codes_in_universe)
                
                # 执行调仓
                # 即使 target_weights 为空（空仓），rebalance 也会处理（卖出所有持仓）
                self.portfolio.rebalance(current_date, current_date, target_weights)

        final_equity = self.portfolio.get_portfolio_history()
        return final_equity.loc[self.effective_start_date:], self.portfolio