# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

# 导入核心组件
from .portfolio import Portfolio 
from .strategies.base import BaseStrategy
from .data.query_helper import DataQueryHelper

class BacktestEngine:
    """
    回测引擎 (BacktestEngine) - 工业级重构版
    
    职责:
    1. 协调者 (Coordinator): 连接 Data, Strategy, Portfolio。
    2. 事件驱动 (Event-Driven): 按日推进时间轴。
    3. 极简主义: 数据清洗与变形逻辑已下放至 DataQueryHelper。
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
        # 1. 数据加载 (Data Loading) - [核心修改区域]
        #    不再在此处进行 Pivot 和 Filtering，直接向 QueryHelper 请求成品宽表。
        # ==============================================================================
        print(f"BacktestEngine: 正在加载 [{self.universe_to_run}] 的价格矩阵...")
        
        # 调用 Helper 的新方法，直接获取 Open/Close 宽表 (Index=Date, Col=Code)
        self.all_open_prices, self.all_close_prices = self.query_helper.get_price_matrix(self.universe_to_run)
        
        # 安全检查
        if self.all_open_prices.empty or self.all_close_prices.empty:
            raise ValueError(
                f"❌ 错误: 未能加载到任何价格数据! \n"
                f"   请检查: 1. Universe '{self.universe_to_run}' 是否存在; 2. 本地数据文件是否损坏。"
            )

        # ==============================================================================
        # 2. 时间切片 (Time Slicing)
        #    只保留回测窗口内的数据
        # ==============================================================================
        try:
            self.open_prices = self.all_open_prices.loc[self.effective_start_date:self.end_date].copy()
            self.close_prices = self.all_close_prices.loc[self.effective_start_date:self.end_date].copy()
        except KeyError:
            # 防止 start_date 早于数据最早日期导致切片报错
            self.open_prices = self.all_open_prices.loc[self.all_open_prices.index >= self.effective_start_date]
            self.open_prices = self.open_prices.loc[self.open_prices.index <= self.end_date]
            
            self.close_prices = self.all_close_prices.loc[self.all_close_prices.index >= self.effective_start_date]
            self.close_prices = self.close_prices.loc[self.close_prices.index <= self.end_date]

        if self.open_prices.empty:
            raise ValueError(f"❌ 错误: 在指定时间段 {start_date} ~ {end_date} 内无有效数据。")

        # 确定最终参与回测的标的列表 (以列名为准)
        self.codes_in_universe = self.open_prices.columns.tolist()
        print(f"✅ 数据加载完成。覆盖标的数: {len(self.codes_in_universe)}")

        # ==============================================================================
        # 3. 初始化账户 (Portfolio Initialization)
        # ==============================================================================
        self.portfolio = Portfolio(
            self.open_prices,
            self.close_prices,
            config.get('INITIAL_CAPITAL', 1_000_000),
            config.get('COMMISSION_RATE', 0.001),
            config.get('SLIPPAGE', 0.0005)
        )
        self.portfolio.config = config

        # ==============================================================================
        # 4. 准备交易日历 (Calendar Setup)
        # ==============================================================================
        self.trade_dates = self.open_prices.index
        self.trade_execution_dates = self._get_trade_execution_dates(self.trade_dates, config)
        
        print(f"BacktestEngine: 调仓模式 - {'固定天数' if config.get('REBALANCE_DAYS') else '按月'}调仓")
        print(f"BacktestEngine: 预计执行调仓次数: {len(self.trade_execution_dates)}")

    def _get_trade_execution_dates(self, price_index: pd.DatetimeIndex, config: Dict) -> pd.DatetimeIndex:
        """
        计算实际调仓执行日期 (Rebalance Schedule)
        支持：每 N 天调仓 / 每 N 月首个交易日调仓
        """
        if price_index.empty:
            return pd.DatetimeIndex([])

        # 模式 A: 固定天数间隔 (e.g., 每 20 个交易日)
        rebalance_days = config.get('REBALANCE_DAYS')
        if rebalance_days is not None and rebalance_days > 0:
            exec_dates = price_index[::rebalance_days]
            return pd.DatetimeIndex(exec_dates)
        
        # 模式 B: 按月调仓 (e.g., 每月第 1 个交易日)
        rebalance_months = config.get('REBALANCE_MONTHS', 1)
        
        # 获取所有月份的开始时间 (Period Index)
        month_starts = price_index.to_period('M').start_time.unique()
        
        if rebalance_months > 1:
            month_starts = month_starts[::rebalance_months]
            
        execution_dates = []
        for month_start in month_starts:
             # 在交易日历中寻找 >= 月初第一天的日期
             found_days = price_index[price_index >= month_start]
             if not found_days.empty:
                  actual_exec_date = found_days[0]
                  # 确保没跨月 (虽然逻辑上不太可能，防御性检查)
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
            # 1. 每日盯市 (Mark to Market): 更新持仓市值
            self.portfolio.update_portfolio_value(current_date)

            # 2. 判断是否是调仓日
            if current_date in self.trade_execution_dates:
                # 调用策略生成信号
                # 注意：策略内部已经加载了因子数据，这里只需传入“当前时间”和“可选股票池”
                target_weights = self.strategy.on_bar(current_date, self.codes_in_universe)
                
                # 3. 执行调仓 (Execution)
                # Portfolio 会自动计算 (目标持仓 - 当前持仓) 的差额并进行买卖
                self.portfolio.rebalance(current_date, current_date, target_weights)

        # 回测结束，生成结果
        final_equity = self.portfolio.get_portfolio_history()
        
        # 截取有效时间段返回
        return final_equity.loc[self.effective_start_date:], self.portfolio