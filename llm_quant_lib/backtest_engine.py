# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional

# 从同级目录导入
from .portfolio import Portfolio
from .strategy import BaseStrategy
from .data_handler import DataHandler
from .factor_engine import FactorEngine

class BacktestEngine:
    """
    负责协调数据、因子、投资组合和策略执行回测循环。
    已增强：支持按天 (REBALANCE_DAYS) 或按月 (REBALANCE_MONTHS) 调仓。
    """
    def __init__(self, start_date: str, end_date: str, config: Dict, strategy: BaseStrategy, data_handler: DataHandler, universe_to_run: str = 'All'):
        self.effective_start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        self.config = config
        self.strategy = strategy
        self.data_handler = data_handler
        self.universe_to_run = universe_to_run

        # 1. 获取资产列表
        self.codes_in_universe = self.data_handler.get_codes_in_universe(self.universe_to_run)

        # 2. 获取并过滤价格数据
        self.all_open_prices = self.data_handler.get_pivot_prices('open', codes=self.codes_in_universe)
        self.all_close_prices = self.data_handler.get_pivot_prices('close', codes=self.codes_in_universe)
        
        self.open_prices = self.all_open_prices.loc[self.effective_start_date:self.end_date].copy()
        self.close_prices = self.all_close_prices.loc[self.effective_start_date:self.end_date].copy()

        # 3. 初始化因子引擎与投资组合
        self.factor_engine = FactorEngine(self.data_handler)
        self.portfolio = Portfolio(
            self.open_prices,
            self.close_prices,
            config.get('INITIAL_CAPITAL', 1_000_000),
            config.get('COMMISSION_RATE', 0.001),
            config.get('SLIPPAGE', 0.0005)
        )
        self.portfolio.config = config

        # 4. 准备回测日期序列与调仓执行日
        self.trade_dates = self.open_prices.index
        # --- 【核心修改】传入整个 config 以判断调仓频率 ---
        self.trade_execution_dates = self._get_trade_execution_dates(self.trade_dates, config)
        
        print(f"BacktestEngine: 调仓模式 - {'固定天数' if config.get('REBALANCE_DAYS') else '按月'}调仓")
        print(f"BacktestEngine: 预计执行调仓次数: {len(self.trade_execution_dates)}")

    def _get_trade_execution_dates(self, price_index: pd.DatetimeIndex, config: Dict) -> pd.DatetimeIndex:
        """计算实际调仓执行日期"""
        if price_index.empty:
            return pd.DatetimeIndex([])

        # 优先检查是否设置了固定天数调仓
        rebalance_days = config.get('REBALANCE_DAYS')
        if rebalance_days is not None and rebalance_days > 0:
            # 使用切片轻松获取每隔 N 天的日期
            exec_dates = price_index[::rebalance_days]
            return pd.DatetimeIndex(exec_dates)
        
        # 否则执行按月调仓逻辑
        rebalance_months = config.get('REBALANCE_MONTHS', 1)
        month_starts = price_index.to_period('M').start_time.unique()
        
        if rebalance_months > 1:
            month_starts = month_starts[::rebalance_months]
            
        execution_dates = []
        for month_start in month_starts:
             # 寻找每月第一个可交易日
             found_days = price_index[price_index >= month_start]
             if not found_days.empty:
                  actual_exec_date = found_days[0]
                  if actual_exec_date.month == month_start.month:
                       execution_dates.append(actual_exec_date)
        return pd.DatetimeIndex(sorted(list(set(execution_dates))))

    def run(self) -> tuple[Optional[pd.DataFrame], Optional[Portfolio]]:
        """执行回测循环"""
        all_available_dates = self.all_close_prices.index
        progress_bar = tqdm(self.trade_dates, desc=f"回测进度 ({self.universe_to_run})")

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
                factor_snapshot = self.factor_engine.get_factor_snapshot(
                    decision_date,
                    codes=self.codes_in_universe,
                    factors=factors_needed
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
        return final_equity.loc[self.effective_start_date:], self.portfolio