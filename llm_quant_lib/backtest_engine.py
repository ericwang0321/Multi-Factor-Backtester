# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tqdm import tqdm # 引入 tqdm 來顯示進度條

# 從同級目錄導入其他模塊
from .portfolio import Portfolio
from .strategy import BaseStrategy
from .data_handler import DataHandler
from .factor_engine import FactorEngine

class BacktestEngine:
    """
    主回測引擎。
    負責協調 DataHandler, FactorEngine, Portfolio, 和 Strategy，執行回測循環。
    """
    def __init__(self, start_date, end_date, config, strategy: BaseStrategy, data_handler: DataHandler):
        """
        初始化回測引擎。

        Args:
            start_date (str): 回測開始日期 'YYYY-MM-DD'。
            end_date (str): 回測結束日期 'YYYY-MM-DD'。
            config (dict): 包含回測參數的字典，如 'INITIAL_CAPITAL', 'COMMISSION_RATE', 'SLIPPAGE', 'REBALANCE_MONTHS'。
            strategy (BaseStrategy): 策略類的實例 (例如 LLMStrategy)。
            data_handler (DataHandler): 已初始化的 DataHandler 實例。
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.config = config
        self.strategy = strategy
        self.data_handler = data_handler # 直接使用傳入的 data_handler

        print("BacktestEngine: 正在初始化...")

        # 1. 從 DataHandler 獲取回測所需的全範圍價格數據
        print("BacktestEngine: 正在獲取價格數據...")
        try:
            self.open_prices = self.data_handler.get_pivot_prices('open')
            self.close_prices = self.data_handler.get_pivot_prices('close')
            # 可以在這裡添加 high, low 如果策略需要
        except ValueError as e:
             raise ValueError(f"BacktestEngine: 無法獲取回測所需的價格數據: {e}")
        print("BacktestEngine: 價格數據獲取完畢。")

        # 過濾價格數據至回測區間
        self.open_prices = self.open_prices.loc[self.start_date:self.end_date]
        self.close_prices = self.close_prices.loc[self.start_date:self.end_date]

        if self.open_prices.empty or self.close_prices.empty:
            raise ValueError(f"在指定的日期範圍 {start_date} 到 {end_date} 內沒有找到價格數據。")

        # 2. 初始化因子引擎
        print("BacktestEngine: 正在初始化因子引擎...")
        # 因子引擎需要 DataHandler 來獲取數據
        self.factor_engine = FactorEngine(self.data_handler)
        print("BacktestEngine: 因子引擎初始化完畢。")

        # 3. 初始化投資組合
        print("BacktestEngine: 正在初始化投資組合...")
        self.portfolio = Portfolio(
            self.open_prices,
            self.close_prices,
            config.get('INITIAL_CAPITAL', 1_000_000), # 使用 .get() 提供默認值
            config.get('COMMISSION_RATE', 0.001),
            config.get('SLIPPAGE', 0.0005)
        )
        # 將 config 也存儲在 portfolio 中，方便 performance 模塊訪問
        self.portfolio.config = config
        print("BacktestEngine: 投資組合初始化完畢。")

        # 4. 準備回測日期序列
        self.trade_dates = self.open_prices.index # 使用實際存在的交易日
        self.rebalance_dates = self._get_rebalance_dates(self.trade_dates, months=config.get('REBALANCE_MONTHS', 1))
        print(f"BacktestEngine: 回測將在 {len(self.trade_dates)} 個交易日上運行。")
        print(f"BacktestEngine: 預計調倉次數: {len(self.rebalance_dates)}。")
        print("BacktestEngine: 初始化完成。")

    def _get_rebalance_dates(self, price_index, months=1):
        """計算實際存在於價格索引中的調倉日期 (每月最後一個交易日)"""
        s = pd.Series(index=price_index, data=price_index) # 使用價格索引創建 Series
        # 獲取每個重採樣週期的最後一個日期
        potential_dates = s.resample(f'{months}M').last()
        # 篩選出實際存在於交易日曆中的日期
        rebalance_dates = potential_dates[potential_dates.isin(price_index)].dropna()
        return pd.to_datetime(rebalance_dates.values)


    def run(self):
        """
        執行回測主循環。
        """
        print("BacktestEngine: === 開始回測循環 ===")

        # 使用 tqdm 創建進度條
        for i, current_date in enumerate(tqdm(self.trade_dates, desc="回測進度")):
            # 1. 每日更新投資組合淨值 (使用當日收盤價)
            self.portfolio.update_portfolio_value(current_date)

            # 2. 檢查是否為調倉決策日的前一天
            # 我們需要在 rebalance_date 的前一個交易日做出決策 (current_date)，
            # 以便在 rebalance_date 當天 (next_trade_date) 以開盤價交易。
            is_decision_day = False
            rebalance_date_for_today = None
            if i + 1 < len(self.trade_dates):
                next_trade_date = self.trade_dates[i+1]
                if next_trade_date in self.rebalance_dates:
                    is_decision_day = True
                    rebalance_date_for_today = next_trade_date # 記錄下實際的調倉執行日

            if is_decision_day and rebalance_date_for_today is not None:
                # --- 調倉邏輯 ---
                # a. 獲取決策日 (current_date) 的因子快照
                #    因子值是基於 current_date 及之前的數據計算的 (已做 shift(1))
                factor_snapshot = self.factor_engine.get_factor_snapshot(current_date)

                # 如果當天因子數據為空，則跳過決策
                if factor_snapshot.empty:
                     print(f"BacktestEngine: 在決策日 {current_date.date()} 因子數據為空，跳過調倉。")
                     continue

                # b. 獲取當前投資組合狀態 (用於 AI 參考)
                portfolio_state = {
                    'cash': self.portfolio.cash,
                    'current_positions': self.portfolio.current_positions # 傳遞當前持倉股數
                }

                # c. 調用策略 (AI 或其他) 來獲取目標權重
                #    策略應基於 current_date 的因子值做出決策
                print(f"\nBacktestEngine: 在 {current_date.date()} 進行決策，準備在 {rebalance_date_for_today.date()} 調倉...")
                target_weights = self.strategy.get_target_weights(
                    current_date,    # 決策基於的數據日期
                    factor_snapshot, # 當天的因子快照
                    portfolio_state  # 當前的投資組合狀態
                )

                # d. 在下一個交易日 (rebalance_date_for_today) 以開盤價執行調倉
                self.portfolio.rebalance(
                    current_date,              # 傳遞 decision_date 以計算交易前價值
                    rebalance_date_for_today,  # 傳遞實際交易日以獲取開盤價
                    target_weights             # 傳遞目標權重
                )
                print(f"BacktestEngine: 在 {rebalance_date_for_today.date()} 調倉執行完畢。")
                # --- 調倉邏輯結束 ---

        print("BacktestEngine: === 回測循環結束 ===")

        # 返回最終的投資組合歷史記錄和 Portfolio 實例本身
        return self.portfolio.portfolio_history, self.portfolio
