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
        # --- 【修改点 1】: 数据加载起始日期提前，确保第一个决策有数据 ---
        # 稍微提前加载数据，例如提前一个月，以确保第一个月初决策时有上月底数据
        buffer_start_date = pd.to_datetime(start_date) - pd.DateOffset(months=1)
        self.effective_start_date = pd.to_datetime(start_date) # 实际回测开始日期
        self.end_date = pd.to_datetime(end_date)
        # --------------------------------------------------------

        self.config = config
        self.strategy = strategy
        self.data_handler = data_handler
        self.universe_to_run = universe_to_run

        print(f"BacktestEngine: 正在为资产池 '{universe_to_run}' 初始化 (数据从 {buffer_start_date.date()} 加载)...") # 更新提示

        # 1. 确定回测资产列表 (与之前相同)
        try:
            self.codes_in_universe = self.data_handler.get_codes_in_universe(self.universe_to_run)
            if not self.codes_in_universe:
                 raise ValueError(f"资产池 '{universe_to_run}' 不包含任何有效的证券代码。")
            print(f"BacktestEngine: 资产池 '{universe_to_run}' 包含 {len(self.codes_in_universe)} 个证券代码。")
        except Exception as e:
             raise ValueError(f"BacktestEngine: 获取资产池 '{universe_to_run}' 的证券代码时出错: {e}")

        # 2. 从 DataHandler 获取价格数据 (加载范围扩大)
        print("BacktestEngine: 正在获取并过滤价格数据...")
        try:
            # --- 【修改点 2】: 传递扩大的日期范围给 DataHandler 获取价格 ---
            # get_pivot_prices 内部会根据传入的 start/end date 过滤 data_handler.raw_df
            # 我们需要在初始化 DataHandler 时就传入扩大的范围，或者修改 get_pivot_prices 逻辑
            # 为简单起见，我们假设 DataHandler 初始化时已加载足够数据 (即其 start_date <= buffer_start_date)
            # 或者我们在这里重新加载一次需要的数据范围 (但不推荐，效率低)
            # 更好的方式是调整 DataHandler 初始化时的日期处理
            # 暂时假设 DataHandler 已加载 buffer_start_date 开始的数据
            # ---
            self.all_open_prices = self.data_handler.get_pivot_prices('open', codes=self.codes_in_universe)
            self.all_close_prices = self.data_handler.get_pivot_prices('close', codes=self.codes_in_universe)

            # ---【修改点 3】: 按实际回测区间过滤用于回测循环的价格 ---
            self.open_prices = self.all_open_prices.loc[self.effective_start_date:self.end_date]
            self.close_prices = self.all_close_prices.loc[self.effective_start_date:self.end_date]
            # ---------------------------------------------------------

            if self.open_prices.empty or self.close_prices.empty:
                raise ValueError(f"在指定的实际回测日期范围 ({self.effective_start_date.date()} to {self.end_date.date()}) 和资产池 '{universe_to_run}' 内没有找到有效的价格数据。")

            self.open_prices.dropna(axis=1, how='all', inplace=True)
            self.close_prices.dropna(axis=1, how='all', inplace=True)
            valid_codes_after_filtering = self.open_prices.columns.tolist()
            if set(valid_codes_after_filtering) != set(self.codes_in_universe):
                 print(f"BacktestEngine: 警告 - 过滤价格数据后，有效证券代码减少到 {len(valid_codes_after_filtering)} 个。")
                 self.codes_in_universe = valid_codes_after_filtering
            if not self.codes_in_universe:
                 raise ValueError(f"过滤价格数据后，资产池 '{universe_to_run}' 内没有剩余的有效证券代码。")

        except ValueError as e:
             raise ValueError(f"BacktestEngine: 准备价格数据时出错: {e}")
        print(f"BacktestEngine: 价格数据准备完毕，包含 {len(self.codes_in_universe)} 个有效证券代码。")


        # 3. 初始化因子引擎 (需要包含 buffer 区间的数据)
        print("BacktestEngine: 正在初始化因子引擎...")
        # --- 【修改点 4】: 确保 FactorEngine 使用包含 buffer 区间的数据 ---
        # 假设 DataHandler.load_data() 返回了 buffer_start_date 开始的数据
        # 或者 FactorEngine 初始化时传入扩大的数据范围
        # 为简单起见，假设 DataHandler 已正确加载
        self.factor_engine = FactorEngine(self.data_handler) # FactorEngine 内部会处理所有可用数据
        # ----------------------------------------------------
        print("BacktestEngine: 因子引擎初始化完毕。")

        # 4. 初始化投资组合 (使用实际回测区间的价格)
        print("BacktestEngine: 正在初始化投资组合...")
        self.portfolio = Portfolio(
            self.open_prices, # 使用过滤后的价格
            self.close_prices, # 使用过滤后的价格
            config.get('INITIAL_CAPITAL', 1_000_000),
            config.get('COMMISSION_RATE', 0.001),
            config.get('SLIPPAGE', 0.0005)
        )
        self.portfolio.config = config
        print("BacktestEngine: 投资组合初始化完毕。")

        # 5. 准备回测日期序列 (基于实际回测区间的价格数据)
        self.trade_dates = self.open_prices.index
        # --- 【修改点 5】: 计算交易执行日期 (每月第一个交易日) ---
        self.trade_execution_dates = self._get_trade_execution_dates(self.trade_dates, months=config.get('REBALANCE_MONTHS', 1))
        # -----------------------------------------------------
        print(f"BacktestEngine: 回测将在 {len(self.trade_dates)} 个交易日 ({self.trade_dates.min().date()} 到 {self.trade_dates.max().date()}) 上运行。")
        print(f"BacktestEngine: 预计调仓次数: {len(self.trade_execution_dates)}。")
        print(f"BacktestEngine: 资产池 '{self.universe_to_run}' 初始化完成。")

    # --- 【修改点 6】: 修改函数名和逻辑，获取每月第一个交易日 ---
    def _get_trade_execution_dates(self, price_index: pd.DatetimeIndex, months: int = 1) -> pd.DatetimeIndex:
        """计算实际存在于价格索引中的交易执行日期 (每月第一个交易日)"""
        if price_index.empty:
            return pd.DatetimeIndex([])
        # 获取所有月份的起始日期
        month_starts = price_index.to_period('M').start_time.unique()
        # 对于每个月起始，找到它之后（包含它自己）的第一个存在于 price_index 中的日期
        execution_dates = []
        for month_start in month_starts:
             # 找到大于等于 month_start 的第一个交易日
             first_day_of_month_or_later = price_index[price_index >= month_start]
             if not first_day_of_month_or_later.empty:
                  # 检查是否仍在同一个月（避免跨月）
                  actual_exec_date = first_day_of_month_or_later[0]
                  if actual_exec_date.month == month_start.month:
                       execution_dates.append(actual_exec_date)
                  # 如果第一个交易日在下个月，说明这个月没有交易日（不太可能但做个保护）
                  # 或者如果需要严格按月分组，可以用 resample('MS').first()，但需要处理非交易日
        return pd.DatetimeIndex(sorted(list(set(execution_dates)))) # 去重并排序
    # ------------------------------------------------------

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

        # 将 trade_dates 转换为 Series 以便快速查找索引
        trade_dates_series = pd.Series(self.trade_dates, index=self.trade_dates)
        all_available_dates = self.all_close_prices.index # 使用包含 buffer 区间的完整日期索引

        progress_bar = tqdm(self.trade_dates, desc=f"回测进度 ({self.universe_to_run})")
        for i, current_date in enumerate(progress_bar):
            # 1. 每日更新投资组合净值 (使用当日收盘价)
            # 确保传递给 Portfolio 的是实际回测区间的日期
            self.portfolio.update_portfolio_value(current_date)

            # --- 【修改点 7】: 判断当前日期是否为交易执行日 ---
            if current_date in self.trade_execution_dates:
                # print(f"\nBacktestEngine: {current_date.date()} 是交易执行日...") # 详细日志

                # a. 找到决策日期 (交易执行日的前一个交易日)
                try:
                    # 在包含 buffer 的完整日期列表中查找 current_date 的前一个日期
                    current_date_location = all_available_dates.get_loc(current_date)
                    if current_date_location == 0:
                        print(f"BacktestEngine: 警告 - {current_date.date()} 是第一个交易日，无法找到前一天的决策数据，跳过首次调仓。")
                        continue # 跳过第一次执行

                    decision_date = all_available_dates[current_date_location - 1]
                    # print(f"BacktestEngine: 决策日期为: {decision_date.date()}") # 详细日志

                except (KeyError, IndexError) as e:
                     print(f"BacktestEngine: 错误 - 在 {current_date.date()} 查找决策日期失败: {e}。跳过本次调仓。")
                     continue

                # b. 获取决策日的因子快照
                try:
                    factor_snapshot = self.factor_engine.get_factor_snapshot(
                        decision_date, # 使用前一交易日作为决策日期
                        codes=self.codes_in_universe
                    )
                except Exception as e:
                     print(f"BacktestEngine: 错误 - 在 {decision_date.date()} 获取因子快照失败: {e}。跳过本次调仓。")
                     continue

                if factor_snapshot.empty:
                     print(f"BacktestEngine: 在决策日 {decision_date.date()} 未生成有效的因子数据，跳过调仓。")
                     continue

                # c. 获取决策日的投资组合状态 (用于 AI 参考和 rebalance 计算交易前价值)
                portfolio_state = {
                    'cash': self.portfolio.cash,
                    'current_positions': self.portfolio.current_positions.copy()
                }

                # d. 调用策略获取目标权重
                try:
                    target_weights = self.strategy.get_target_weights(
                        decision_date,    # 决策基于的数据日期
                        factor_snapshot,
                        portfolio_state
                    )
                except Exception as e:
                    print(f"BacktestEngine: 错误 - 策略在 {decision_date.date()} 生成目标权重时失败: {e}。跳过本次调仓。")
                    import traceback; traceback.print_exc(); continue

                # e. 在当前交易执行日 (current_date) 以开盘价执行调仓
                try:
                    self.portfolio.rebalance(
                        decision_date,  # 决策日，用于计算交易前价值
                        current_date,   # 实际交易日，用于获取开盘价
                        target_weights
                    )
                    # print(f"BacktestEngine: 在 {current_date.date()} 调仓执行完毕。") # 详细日志
                except Exception as e:
                     print(f"BacktestEngine: 错误 - 在 {current_date.date()} 执行调仓时失败: {e}。")
                     import traceback; traceback.print_exc(); continue
            # --- 【修改结束】---

            # 更新进度条 (可选)
            # current_value = self.portfolio.get_current_value(current_date)
            # progress_bar.set_postfix({"净值": f"{current_value:,.0f}"})

        print(f"BacktestEngine: === 资产池 '{self.universe_to_run}' 回测循环结束 ===")

        final_equity = self.portfolio.get_portfolio_history()
        if final_equity.empty:
            print(f"BacktestEngine: 警告 - 资产池 '{self.universe_to_run}' 的回测未生成有效的净值记录。")
            return None, self.portfolio
        else:
            # --- 【修改点 8】: 确保返回的净值是从实际回测开始日算起 ---
            final_equity = final_equity.loc[self.effective_start_date:]
            if final_equity.empty:
                print(f"BacktestEngine: 警告 - 过滤到实际回测开始日期 {self.effective_start_date.date()} 后，净值记录为空。")
                return None, self.portfolio
            # ----------------------------------------------------
            return final_equity, self.portfolio

