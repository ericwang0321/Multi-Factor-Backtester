# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional

# 从同级目录导入
# 假设 Portfolio 和 BaseStrategy 没变，我们继续使用它们
from .portfolio import Portfolio 
from .strategy import BaseStrategy

# --- 修改 1: 导入 QueryHelper 而不是 DataHandler ---
# from .data_handler import DataHandler # ❌ 删除
from .data.query_helper import DataQueryHelper # ✅ 新增

# 从当前包的 factors 子包导入 engine
from .factors.engine import FactorEngine

class BacktestEngine:
    """
    负责协调数据、因子、投资组合和策略执行回测循环。
    适配 Parquet 数据源 (DataQueryHelper)。
    """
    
    # --- 修改 2: 初始化参数变更为 query_helper ---
    def __init__(self, start_date: str, end_date: str, config: Dict, strategy: BaseStrategy, query_helper: DataQueryHelper, universe_to_run: str = 'All'):
        self.effective_start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        self.config = config
        self.strategy = strategy
        self.query_helper = query_helper # 使用新的 helper
        self.universe_to_run = universe_to_run

        # --- 修改 3: 数据加载逻辑重写 (Pivot 逻辑移到这里) ---
        print("BacktestEngine: Loading and pivoting price data...")
        
        # 1. 获取全量长格式数据 (Long Format)
        raw_df = self.query_helper.get_all_price_data()
        
        # 2. 过滤 Universe (如果需要)
        # 你的 QueryHelper.get_all_symbols() 返回的是 DataFrame
        if self.universe_to_run != 'All':
            # 这里假设 universe_to_run 是一个 category_id
            all_symbols = self.query_helper.get_all_symbols()
            valid_codes = all_symbols[all_symbols['category_id'] == self.universe_to_run]['sec_code'].unique()
            raw_df = raw_df[raw_df['sec_code'].isin(valid_codes)]
            self.codes_in_universe = valid_codes.tolist()
        else:
            self.codes_in_universe = raw_df['sec_code'].unique().tolist()

        # 3. 执行 Pivot 生成宽表 (Wide Format)
        # 你的 Portfolio 类需要宽格式的 Open 和 Close
        try:
            self.all_open_prices = raw_df.pivot(index='datetime', columns='sec_code', values='open')
            self.all_close_prices = raw_df.pivot(index='datetime', columns='sec_code', values='close')
            
            # 填充缺失值 (ffill 避免未来函数)
            self.all_open_prices = self.all_open_prices.ffill()
            self.all_close_prices = self.all_close_prices.ffill()
        except Exception as e:
            raise ValueError(f"Failed to pivot price data: {e}")

        # 4. 截取回测时间段
        self.open_prices = self.all_open_prices.loc[self.effective_start_date:self.end_date].copy()
        self.close_prices = self.all_close_prices.loc[self.effective_start_date:self.end_date].copy()

        # --- 修改 4: 初始化 FactorEngine (传入 query_helper) ---
        self.factor_engine = FactorEngine(self.query_helper)
        
        # 5. 初始化投资组合 (保留你原有的 Portfolio 类)
        self.portfolio = Portfolio(
            self.open_prices,
            self.close_prices,
            config.get('INITIAL_CAPITAL', 1_000_000),
            config.get('COMMISSION_RATE', 0.001),
            config.get('SLIPPAGE', 0.0005)
        )
        self.portfolio.config = config

        # 6. 准备日期序列
        self.trade_dates = self.open_prices.index
        self.trade_execution_dates = self._get_trade_execution_dates(self.trade_dates, config)
        
        print(f"BacktestEngine: 调仓模式 - {'固定天数' if config.get('REBALANCE_DAYS') else '按月'}调仓")
        print(f"BacktestEngine: 预计执行调仓次数: {len(self.trade_execution_dates)}")

    def _get_trade_execution_dates(self, price_index: pd.DatetimeIndex, config: Dict) -> pd.DatetimeIndex:
        """计算实际调仓执行日期 (逻辑保持不变)"""
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
        """执行回测循环 (逻辑保持不变)"""
        all_available_dates = self.all_close_prices.index
        # 确保使用正确的列名进行进度条显示
        desc_text = f"回测进度 ({self.universe_to_run})"
        progress_bar = tqdm(self.trade_dates, desc=desc_text)

        for current_date in progress_bar:
            # 每日更新净值
            self.portfolio.update_portfolio_value(current_date)

            # 触发调仓
            if current_date in self.trade_execution_dates:
                # 确定决策日 (T-1)
                try:
                    loc = all_available_dates.get_loc(current_date)
                    if loc == 0: continue
                    decision_date = all_available_dates[loc - 1]
                except (KeyError, IndexError): continue

                # 获取因子数据
                factors_needed = self.strategy.get_required_factors()
                
                # --- 修改 5: FactorEngine 调用调整 ---
                # 你的 FactorEngine.get_factor_snapshot 逻辑没变，这里参数兼容
                factor_snapshot = self.factor_engine.get_factor_snapshot(
                    decision_date,
                    codes=self.codes_in_universe,
                    factors=factors_needed,
                    weights=getattr(self.strategy, 'factor_weights', None)
                )
                if factor_snapshot.empty: continue

                # 获取目标权重并执行
                portfolio_state = {
                    'cash': self.portfolio.cash,
                    'current_positions': self.portfolio.current_positions.copy()
                }
                target_weights = self.strategy.get_target_weights(
                    decision_date, factor_snapshot, portfolio_state
                )
                
                self.portfolio.rebalance(decision_date, current_date, target_weights)

        final_equity = self.portfolio.get_portfolio_history()
        return final_equity.loc[self.effective_start_date:], self.portfolio # 返回 Portfolio 对象本身用于后续分析