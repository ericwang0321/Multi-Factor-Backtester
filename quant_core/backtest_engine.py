# quant_core/backtest_engine.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from .portfolio import Portfolio 
from .strategies.base import BaseStrategy
from .data.query_helper import DataQueryHelper

class BacktestEngine:
    """
    回测引擎 (BacktestEngine) - 工业级执行版 V5
    
    职责升级:
    1. 模拟真实交易时序: T-1 收盘决策 -> T 开盘执行。
    2. 提供双重价格流: 喂给 Portfolio 'Signal Price' 和 'Execution Price'。
    3. 严格处理 Gap Risk: 配合 Portfolio 的资金硬约束逻辑。
    """
    
    def __init__(self, 
                 start_date: str, 
                 end_date: str, 
                 config: Dict, 
                 strategy: BaseStrategy, 
                 query_helper: DataQueryHelper, 
                 universe_to_run: str = 'All'):
        
        self.effective_start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        self.config = config
        self.strategy = strategy
        self.query_helper = query_helper
        self.universe_to_run = universe_to_run

        # ==============================================================================
        # 1. 数据加载 (Data Loading)
        # ==============================================================================
        print(f"BacktestEngine: 正在加载 [{self.universe_to_run}] 数据矩阵...")
        self.all_open_prices, self.all_close_prices = self.query_helper.get_price_matrix(self.universe_to_run)
        
        if self.all_open_prices.empty:
            raise ValueError(
                f"❌ 错误: 未能加载到任何价格数据! Universe: '{self.universe_to_run}'"
            )

        # ==============================================================================
        # 2. 时间切片 (Time Slicing with Buffer)
        #    关键点：为了获取 T-1 收盘价，我们需要比 effective_start_date 多取几天数据作为 Buffer
        # ==============================================================================
        buffer_days = 20
        buffer_start = self.effective_start_date - pd.Timedelta(days=buffer_days)
        
        # 截取带 Buffer 的数据段，用于内部索引
        self.open_prices = self.all_open_prices.loc[buffer_start:self.end_date].copy()
        self.close_prices = self.all_close_prices.loc[buffer_start:self.end_date].copy()
        
        self.codes_in_universe = self.open_prices.columns.tolist()
        print(f"✅ 数据加载完成。覆盖标的数: {len(self.codes_in_universe)}")

        # ==============================================================================
        # 3. 初始化账户 (Portfolio Initialization)
        #    不再传入全量价格表，解耦，由 Engine 逐日喂入
        # ==============================================================================
        self.portfolio = Portfolio(
            initial_capital=config.get('INITIAL_CAPITAL', 1_000_000),
            commission_rate=config.get('COMMISSION_RATE', 0.001),
            slippage=config.get('SLIPPAGE', 0.0005)
        )

        # ==============================================================================
        # 4. 准备交易日历
        #    只遍历 effective_start_date 之后的日期 (不回测 Buffer 期)
        # ==============================================================================
        full_dates = self.open_prices.index
        self.trade_dates = full_dates[full_dates >= self.effective_start_date]
        self.trade_execution_dates = self._get_trade_execution_dates(self.trade_dates, config)
        
        print(f"BacktestEngine: 调仓模式 - {'固定天数' if config.get('REBALANCE_DAYS') else '按月'}调仓")
        print(f"BacktestEngine: 预计执行调仓次数: {len(self.trade_execution_dates)}")

    def _get_trade_execution_dates(self, price_index: pd.DatetimeIndex, config: Dict) -> pd.DatetimeIndex:
        """计算实际调仓执行日期"""
        if price_index.empty: return pd.DatetimeIndex([])
        
        rebalance_days = config.get('REBALANCE_DAYS')
        if rebalance_days and rebalance_days > 0:
            return pd.DatetimeIndex(price_index[::rebalance_days])
        
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

    def run(self) -> Tuple[Optional[pd.DataFrame], Optional[Portfolio]]:
        """
        执行回测主循环 (Main Loop)
        """
        desc_text = f"回测进度 ({self.universe_to_run})"
        progress_bar = tqdm(self.trade_dates, desc=desc_text)

        for current_date in progress_bar:
            # -----------------------------------------------------------
            # 1. 获取价格快照 (Price Snapshot)
            # -----------------------------------------------------------
            try:
                # Execution Price: 今天的开盘价 (我们假设在开盘时刻撮合)
                today_open = self.open_prices.loc[current_date]
                # Close Price: 今天的收盘价 (用于盘后结算净值)
                today_close = self.close_prices.loc[current_date]
            except KeyError:
                continue 

            # Signal Price: 昨天的收盘价 (用于定股数)
            # 使用 get_loc 找到当前日期的整数索引，然后 -1
            try:
                curr_idx = self.close_prices.index.get_loc(current_date)
                if curr_idx > 0:
                    yesterday_close = self.close_prices.iloc[curr_idx - 1]
                else:
                    # 极端情况：如果是数据的第一天，没有昨天，用今天开盘价代替估算
                    yesterday_close = today_open
            except Exception:
                yesterday_close = today_open

            # -----------------------------------------------------------
            # 2. 每日盯市 (Mark to Market)
            # -----------------------------------------------------------
            self.portfolio.update_portfolio_value(current_date, today_close)

            # -----------------------------------------------------------
            # 3. 交易执行 (Execution)
            # -----------------------------------------------------------
            if current_date in self.trade_execution_dates:
                # 调用策略生成信号
                target_weights = self.strategy.on_bar(current_date, self.codes_in_universe)
                
                # [核心逻辑升级] 传入双重价格
                # signal_prices=yesterday_close -> 确定买多少股
                # execution_prices=today_open -> 确定花多少钱 (含 Gap Risk)
                self.portfolio.rebalance(
                    date=current_date,
                    target_weights=target_weights,
                    signal_prices=yesterday_close,
                    execution_prices=today_open
                )

        # 回测结束，生成结果
        final_equity = self.portfolio.get_portfolio_history()
        
        # 截取有效时间段返回
        # [修复] 2018-01-01 是假期，直接 loc 会报错
        # 使用切片逻辑：找 >= start_date 的所有行
        valid_equity = final_equity[final_equity.index >= self.effective_start_date]
        
        return valid_equity, self.portfolio