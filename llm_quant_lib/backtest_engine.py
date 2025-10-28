# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tqdm import tqdm # 引入 tqdm 来显示进度条
from typing import Dict, List, Optional

# 从同级目录导入其他模块
from .portfolio import Portfolio
from .strategy import BaseStrategy
from .data_handler import DataHandler
from .factor_engine import FactorEngine

class BacktestEngine:
    """
    主回测引擎。
    负责协调 DataHandler, FactorEngine, Portfolio, 和 Strategy，执行回测循环。
    支持按资产池 (universe) 进行回测。
    """
    def __init__(self, start_date: str, end_date: str, config: Dict, strategy: BaseStrategy, data_handler: DataHandler, universe_to_run: str = 'All'):
        """
        初始化回测引擎。

        Args:
            start_date (str): 回测开始日期 'YYYY-MM-DD'。
            end_date (str): 回测结束日期 'YYYY-MM-DD'。
            config (dict): 包含回测参数的字典，如 'INITIAL_CAPITAL', 'COMMISSION_RATE', 'SLIPPAGE', 'REBALANCE_MONTHS'。
            strategy (BaseStrategy): 策略类的实例 (例如 LLMStrategy)。
            data_handler (DataHandler): 已初始化的 DataHandler 实例。
            universe_to_run (str): 指定本次回测运行的资产池名称 (例如 'equity_us', 'bond', 或 'All')。
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.config = config
        self.strategy = strategy
        self.data_handler = data_handler
        self.universe_to_run = universe_to_run

        print(f"BacktestEngine: 正在为资产池 '{universe_to_run}' 初始化...")

        # 1. 确定回测资产列表
        try:
            # get_codes_in_universe 现在由 DataHandler 提供
            self.codes_in_universe = self.data_handler.get_codes_in_universe(self.universe_to_run)
            if not self.codes_in_universe:
                 raise ValueError(f"资产池 '{universe_to_run}' 不包含任何有效的证券代码。")
            print(f"BacktestEngine: 资产池 '{universe_to_run}' 包含 {len(self.codes_in_universe)} 个证券代码。")
        except Exception as e:
             raise ValueError(f"BacktestEngine: 获取资产池 '{universe_to_run}' 的证券代码时出错: {e}")

        # 2. 从 DataHandler 获取已过滤到资产池的价格数据
        print("BacktestEngine: 正在获取并过滤价格数据...")
        try:
            # 传入 codes 参数进行过滤
            self.open_prices = self.data_handler.get_pivot_prices('open', codes=self.codes_in_universe)
            self.close_prices = self.data_handler.get_pivot_prices('close', codes=self.codes_in_universe)

            # 过滤价格数据至回测区间
            self.open_prices = self.open_prices.loc[self.start_date:self.end_date]
            self.close_prices = self.close_prices.loc[self.start_date:self.end_date]

            # 确保过滤后仍有数据
            if self.open_prices.empty or self.close_prices.empty:
                raise ValueError(f"在指定的日期范围和资产池 '{universe_to_run}' 内没有找到有效的价格数据。")
            # 移除价格数据中完全是 NaN 的列 (即在整个回测期间都没有价格的资产)
            self.open_prices.dropna(axis=1, how='all', inplace=True)
            self.close_prices.dropna(axis=1, how='all', inplace=True)
            valid_codes_after_filtering = self.open_prices.columns.tolist()
            if set(valid_codes_after_filtering) != set(self.codes_in_universe):
                 print(f"BacktestEngine: 警告 - 过滤价格数据后，有效证券代码减少到 {len(valid_codes_after_filtering)} 个。")
                 self.codes_in_universe = valid_codes_after_filtering # 更新实际参与回测的代码列表
            if not self.codes_in_universe:
                 raise ValueError(f"过滤价格数据后，资产池 '{universe_to_run}' 内没有剩余的有效证券代码。")

        except ValueError as e:
             raise ValueError(f"BacktestEngine: 准备价格数据时出错: {e}")
        print(f"BacktestEngine: 价格数据准备完毕，包含 {len(self.codes_in_universe)} 个有效证券代码。")


        # 3. 初始化因子引擎 (它内部处理所有资产，但在 get_snapshot 时可被过滤)
        print("BacktestEngine: 正在初始化因子引擎...")
        # 因子引擎需要 DataHandler 来获取数据
        self.factor_engine = FactorEngine(self.data_handler)
        print("BacktestEngine: 因子引擎初始化完毕。")

        # 4. 初始化投资组合 (只传入过滤后的价格数据)
        print("BacktestEngine: 正在初始化投资组合...")
        self.portfolio = Portfolio(
            self.open_prices, # 使用过滤后的价格
            self.close_prices, # 使用过滤后的价格
            config.get('INITIAL_CAPITAL', 1_000_000),
            config.get('COMMISSION_RATE', 0.001),
            config.get('SLIPPAGE', 0.0005)
        )
        self.portfolio.config = config # 存储配置信息
        print("BacktestEngine: 投资组合初始化完毕。")

        # 5. 准备回测日期序列 (基于过滤后的价格数据)
        self.trade_dates = self.open_prices.index
        self.rebalance_dates = self._get_rebalance_dates(self.trade_dates, months=config.get('REBALANCE_MONTHS', 1))
        print(f"BacktestEngine: 回测将在 {len(self.trade_dates)} 个交易日 ({self.trade_dates.min().date()} 到 {self.trade_dates.max().date()}) 上运行。")
        print(f"BacktestEngine: 预计调仓次数: {len(self.rebalance_dates)}。")
        print(f"BacktestEngine: 资产池 '{universe_to_run}' 初始化完成。")

    def _get_rebalance_dates(self, price_index: pd.DatetimeIndex, months: int = 1) -> pd.DatetimeIndex:
        """计算实际存在于价格索引中的调仓日期 (每月最后一个交易日)"""
        if price_index.empty:
            return pd.DatetimeIndex([])
        # 使用 Series 的 resample 方法找到每个月末的日期
        s = pd.Series(index=price_index, data=price_index)
        # 使用 'ME' 代替 'M' (Month End)
        potential_dates = s.resample(f'{months}ME').last()
        # 筛选出实际存在于交易日历中的日期
        rebalance_dates = potential_dates[potential_dates.isin(price_index)].dropna()
        return pd.to_datetime(rebalance_dates.values)

    def run(self) -> tuple[Optional[pd.DataFrame], Optional[Portfolio]]:
        """
        执行回测主循环。

        Returns:
            tuple[Optional[pd.DataFrame], Optional[Portfolio]]:
                - 投资组合每日净值历史的 DataFrame (索引为 datetime)。
                - 运行结束后的 Portfolio 实例。
                如果回测失败或无结果，则返回 (None, None)。
        """
        print(f"BacktestEngine: === 开始为资产池 '{self.universe_to_run}' 执行回测循环 ===")

        if self.trade_dates.empty:
             print("BacktestEngine: 错误 - 无有效的交易日期，无法执行回测。")
             return None, None

        # 使用 tqdm 创建进度条
        # desc 添加资产池名称
        progress_bar = tqdm(self.trade_dates, desc=f"回测进度 ({self.universe_to_run})")
        for i, current_date in enumerate(progress_bar):
            # 1. 每日更新投资组合净值 (使用当日收盘价)
            self.portfolio.update_portfolio_value(current_date)

            # 2. 检查是否为调仓决策日 (即实际调仓日的前一个交易日)
            is_decision_day = False
            rebalance_date_for_today: Optional[pd.Timestamp] = None
            if i + 1 < len(self.trade_dates):
                next_trade_date = self.trade_dates[i+1]
                # 检查下一个交易日是否在预定的调仓日期列表中
                if next_trade_date in self.rebalance_dates:
                    is_decision_day = True
                    rebalance_date_for_today = next_trade_date # 记录实际的调仓执行日

            # --- 如果是决策日 ---
            if is_decision_day and rebalance_date_for_today is not None:
                # print(f"\nBacktestEngine: {current_date.date()} 是决策日，准备在 {rebalance_date_for_today.date()} 调仓...") # 打印详细日志

                # a. 获取决策日 (current_date) 的因子快照，只包含当前资产池的资产
                try:
                    # --- 【修正】传递 codes_in_universe 参数 ---
                    factor_snapshot = self.factor_engine.get_factor_snapshot(
                        current_date,
                        codes=self.codes_in_universe # 传入资产列表进行过滤
                    )
                except Exception as e:
                     print(f"BacktestEngine: 错误 - 在 {current_date.date()} 获取因子快照失败: {e}。跳过本次调仓。")
                     continue # 跳过本次调仓

                # 如果因子数据为空，则跳过
                if factor_snapshot.empty:
                     print(f"BacktestEngine: 在决策日 {current_date.date()} 未生成有效的因子数据，跳过调仓。")
                     continue

                # b. 获取当前投资组合状态 (用于 AI 参考)
                portfolio_state = {
                    'cash': self.portfolio.cash,
                    # 传递当前的持仓股数字典
                    'current_positions': self.portfolio.current_positions.copy() # 传入副本
                }

                # c. 调用策略 (AI 或其他) 获取目标权重
                try:
                    target_weights = self.strategy.get_target_weights(
                        current_date,    # 决策基于的数据日期
                        factor_snapshot, # 当天的因子快照 (已过滤资产池)
                        portfolio_state  # 当前的投资组合状态
                    )
                except Exception as e:
                    print(f"BacktestEngine: 错误 - 策略在 {current_date.date()} 生成目标权重时失败: {e}。跳过本次调仓。")
                    import traceback
                    traceback.print_exc()
                    continue # 跳过本次调仓

                # d. 在下一个交易日 (rebalance_date_for_today) 以开盘价执行调仓
                try:
                    self.portfolio.rebalance(
                        current_date,              # 决策日，用于计算交易前价值
                        rebalance_date_for_today,  # 实际交易日，用于获取开盘价
                        target_weights             # 目标权重
                    )
                    # print(f"BacktestEngine: 在 {rebalance_date_for_today.date()} 调仓执行完毕。") # 打印详细日志
                except Exception as e:
                     print(f"BacktestEngine: 错误 - 在 {rebalance_date_for_today.date()} 执行调仓时失败: {e}。投资组合状态可能不一致。")
                     import traceback
                     traceback.print_exc()
                     # 考虑是否需要在这里停止回测或采取其他恢复措施
                     # 暂时继续，但后续结果可能受影响
                     continue
                # --- 调仓逻辑结束 ---

            # 更新进度条显示当前净值 (可选)
            # current_value = self.portfolio.get_current_value(current_date)
            # progress_bar.set_postfix({"净值": f"{current_value:,.0f}"})


        print(f"BacktestEngine: === 资产池 '{self.universe_to_run}' 回测循环结束 ===")

        # 返回最终的投资组合历史记录 DataFrame 和 Portfolio 实例本身
        final_equity = self.portfolio.get_portfolio_history()
        if final_equity.empty:
            print(f"BacktestEngine: 警告 - 资产池 '{self.universe_to_run}' 的回测未生成有效的净值记录。")
            return None, self.portfolio
        else:
            return final_equity, self.portfolio

