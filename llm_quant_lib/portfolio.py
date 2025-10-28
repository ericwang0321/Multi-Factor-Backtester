# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from collections import defaultdict

class Portfolio:
    """
    负责管理投资组合的状态、现金和执行交易。
    (逻辑主要来自你 backtest.py 中的 VectorizedBacktester)
    """
    def __init__(self, open_prices: pd.DataFrame, close_prices: pd.DataFrame, initial_capital: float, commission_rate: float, slippage: float):
        """
        初始化投资组合。

        Args:
            open_prices (pd.DataFrame): 开盘价 (datetime x sec_code)。确保只包含回测资產池的代码。
            close_prices (pd.DataFrame): 收盘价 (datetime x sec_code)。确保只包含回测资產池的代码。
            initial_capital (float): 初始资金。
            commission_rate (float): 手续费率 (例如 0.001)。
            slippage (float): 滑点率 (例如 0.0005)。
        """
        if not isinstance(open_prices.index, pd.DatetimeIndex) or not isinstance(close_prices.index, pd.DatetimeIndex):
             raise ValueError("价格数据的索引必须是 DatetimeIndex。")

        self.open_prices = open_prices
        self.close_prices = close_prices
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.config = None # BacktestEngine 会将配置信息存储在这里

        self.cash = initial_capital
        # {sec_code: shares} 持有股数, 只存储 > 0 的仓位
        self.current_positions: Dict[str, float] = defaultdict(float)
        # 记录每日总净值 {'datetime': date, 'total_value': value}
        self.portfolio_history: List[Dict] = []
        # 记录每次调仓后的持仓详情 {'datetime': date, 'sec_code': code, 'shares': shares, 'price': price, 'value': value, 'weight': weight}
        self.holdings_history: List[Dict] = []
        # 记录每次调仓的换手率 (min(buy_value, sell_value) / portfolio_value_before_trade)
        self.turnover_history: List[float] = []

        # 记录第一天的初始状态
        if not self.open_prices.empty:
             self.portfolio_history.append({'datetime': self.open_prices.index[0], 'total_value': initial_capital})


    def get_current_value(self, date: pd.Timestamp) -> float:
        """获取指定日期收盘后的总市值 (持仓市值 + 現金)"""
        market_value = 0.0
        # 确保日期在價格數據的索引中
        if date in self.close_prices.index:
            # 获取当天的收盘价 Series
            prices_today = self.close_prices.loc[date]
            for sec, shares in self.current_positions.items():
                # 确保该证券当在价格 Series 中且价格有效
                if sec in prices_today.index and pd.notna(prices_today[sec]) and prices_today[sec] > 0:
                    market_value += shares * prices_today[sec]
                # else: # 如果找不到價格，假设其价值为0 （更保守）
                #     print(f"警告: 在 {date.date()} 收盘时无法找到 {sec} 的有效价格，其市值计为0。")
        # else:
        #     # 如果日期不存在，使用最近一次记录的总价值？或者返回NaN？返回当前现金+持仓按上个交易日价格计算？
        #     # 为简单起见，如果日期无效，我们返回NaN或0？返回当前现金可能是最不坏的选择
        #     print(f"警告: 日期 {date.date()} 不在收盘价数据索引中，无法准确计算当日持仓市值。")
        #     # 尝试使用前一天的 portfolio_history 值？
        #     if self.portfolio_history:
        #         return self.portfolio_history[-1]['total_value']
        #     else:
        #         return self.cash # 近似值

        return market_value + self.cash

    def update_portfolio_value(self, date: pd.Timestamp):
        """记录当天的投资组合总价值到历史记录"""
        # 如果日期已存在（例如步进优化时重复处理），先移除旧记录
        if self.portfolio_history and self.portfolio_history[-1]['datetime'] == date:
             self.portfolio_history.pop()

        total_value = self.get_current_value(date)
        self.portfolio_history.append({'datetime': date, 'total_value': total_value})
        return total_value

    def rebalance(self, decision_date: pd.Timestamp, trade_date: pd.Timestamp, target_weights: Dict[str, float]):
        """
        根据策略提供的目标权重，在 trade_date 以开盘价执行调仓。

        Args:
            decision_date (pd.Timestamp): 决策日期 (用于计算交易前的总资產)。
            trade_date (pd.Timestamp): 交易日期 (用于获取开盘价并执行)。
            target_weights (dict): {sec_code: target_weight} 目标权重字典。
        """
        # 1. 获取交易前的总资產 (基于 decision_date 的收盘价)
        portfolio_value_before_trade = self.get_current_value(decision_date)

        # 如果资產耗尽或为负，则无法交易
        if portfolio_value_before_trade <= 1e-8:
            print(f"信息: {decision_date.date()} 投资组合价值过低 ({portfolio_value_before_trade:.2f})，跳过调仓。")
            self.turnover_history.append(0.0) # 记录零换手率
            self._record_holdings(trade_date) # 仍然记录（可能是空的）持仓
            return

        # 确保 trade_date 在开盘价数据中
        if trade_date not in self.open_prices.index:
            print(f"警告: 交易日 {trade_date.date()} 无开盘价数据，跳过本次调仓。")
            self.turnover_history.append(0.0)
            self._record_holdings(trade_date) # 记录未变化的持仓
            return

        # 获取当天的开盘价 Series
        prices_today = self.open_prices.loc[trade_date]

        # 2. 计算目标持有股数
        target_positions: Dict[str, float] = {} # {sec_code: target_shares}
        for sec, weight in target_weights.items():
            if weight > 1e-9: # 只处理目标权重为正的
                target_value = portfolio_value_before_trade * weight
                price = prices_today.get(sec, np.nan) # 使用 .get() 避免 KeyError
                if pd.notna(price) and price > 0:
                     # 估算买入成本（包括费用和滑点）来计算目标股数
                     cost_per_share = price * (1 + self.commission_rate + self.slippage)
                     target_shares = target_value / cost_per_share if cost_per_share > 0 else 0
                     if target_shares > 0:
                         target_positions[sec] = target_shares
                # else:
                    # print(f"警告: 无法为 {sec} 在 {trade_date.date()} 获取有效开盘价 ({price})，无法设定买入目标。")

        # 3. 生成交易列表 (需要买入/卖出的股数)
        trades: Dict[str, float] = defaultdict(float) # {sec_code: shares_to_trade (+buy, -sell)}
        # 需要卖出的：当前持有但不在目标中，或目标股数少于当前持有
        for sec, current_shares in self.current_positions.items():
            target_shares = target_positions.get(sec, 0.0)
            if target_shares < current_shares:
                trades[sec] = target_shares - current_shares # 负数表示卖出

        # 需要买入的：在目标中，且目标股数多于当前持有
        for sec, target_shares in target_positions.items():
            current_shares = self.current_positions.get(sec, 0.0)
            if target_shares > current_shares:
                trades[sec] = target_shares - current_shares # 正数表示买入

        # 4. 执行交易 (先卖后买，以释放现金)
        total_sells_nominal_value = 0.0
        total_buys_nominal_value = 0.0

        # --- 执行卖出 ---
        sell_trades = {sec: s for sec, s in trades.items() if s < -1e-9}
        for sec, shares_to_sell in sell_trades.items():
            # _execute_trade 的第二个参数是最终目标股数
            target_shares_after_sell = self.current_positions[sec] + shares_to_sell
            sell_nominal, _ = self._execute_trade(trade_date, sec, target_shares_after_sell)
            total_sells_nominal_value += sell_nominal

        # --- 执行买入 ---
        buy_trades = {sec: s for sec, s in trades.items() if s > 1e-9}
        # 简单处理：如果多个买单但现金不够，按目标权重比例缩减？或者按顺序买入？
        # 这里采用你的原始 backtest.py 的逻辑，在 _execute_trade 内部处理现金不足的情况
        for sec, shares_to_buy in buy_trades.items():
            # _execute_trade 的第二个参数是最终目标股数
            target_shares_after_buy = self.current_positions.get(sec, 0.0) + shares_to_buy
            buy_nominal, _ = self._execute_trade(trade_date, sec, target_shares_after_buy)
            total_buys_nominal_value += buy_nominal


        # 5. 记录换手率
        # 使用交易发生前的投资组合价值作为分母
        turnover = min(total_buys_nominal_value, total_sells_nominal_value) / portfolio_value_before_trade if portfolio_value_before_trade > 1e-8 else 0.0
        self.turnover_history.append(turnover)

        # 6. 记录本次调仓执行完毕后的持仓权重
        self._record_holdings(trade_date)


    def _execute_trade(self, date: pd.Timestamp, sec_code: str, target_shares: float) -> tuple[float, Optional[str]]:
        """
        执行单笔交易以达到目标股数 target_shares。
        返回 (交易的名义价值, 'buy'/'sell'/None)。
        名义价值是交易股数 * 交易时的市场价格 (不含费用滑点)。
        (这是你 backtest.py 中的 _execute_trade 核心逻辑，已整合)
        """
        # 获取当天的开盘价
        price = self.open_prices.loc[date, sec_code] if date in self.open_prices.index and sec_code in self.open_prices.columns else np.nan

        # 检查价格是否有效
        if pd.isna(price) or price <= 0:
            # print(f"警告: 无法在 {date.date()} 为 {sec_code} 获取有效开盘价 ({price})，无法执行交易。")
            return 0.0, None # 无法交易

        current_shares = self.current_positions.get(sec_code, 0.0)
        shares_to_trade = target_shares - current_shares # 需要交易的股数（正为买，负为卖）

        # 如果需要交易的股数非常接近0，则不执行
        # 使用名义价值判断，避免股数很小但价格很高的情况
        if abs(shares_to_trade * price) < 1.0: # 阈值设为1元
             return 0.0, None

        trade_nominal_value = 0.0
        trade_type = None

        if shares_to_trade > 0: # --- 买入 ---
            trade_type = 'buy'
            cost_per_share = price * (1 + self.commission_rate + self.slippage) # 每股成本
            if cost_per_share <= 0: return 0.0, None

            # 根据可用现金计算最多能买多少股
            affordable_shares = self.cash / cost_per_share if cost_per_share > 0 else 0
            # 实际能买的股数 = min(需要买的, 能负担的)
            actual_shares_to_buy = min(shares_to_trade, affordable_shares)

            # 如果实际能买的太少，就不买了
            if actual_shares_to_buy * price < 1.0: # 名义价值太低
                 # print(f"信息: {date.date()} 现金 {self.cash:.2f} 不足以按目标买入 {shares_to_trade:.2f} 股 {sec_code} (需 {shares_to_trade * cost_per_share:.2f})，或买入量过小。")
                 return 0.0, trade_type # 返回 'buy' 但交易额为0

            # 执行买入
            cost = actual_shares_to_buy * cost_per_share # 总花费
            self.cash -= cost
            self.current_positions[sec_code] = current_shares + actual_shares_to_buy # 更新持仓
            trade_nominal_value = actual_shares_to_buy * price # 记录名义价值
            # print(f"交易: {date.date()} 买入 {actual_shares_to_buy:.2f} 股 {sec_code} @ {price:.2f}, 花费 {cost:.2f}, 剩餘现金 {self.cash:.2f}")

        elif shares_to_trade < 0: # --- 卖出 ---
            trade_type = 'sell'
            shares_to_sell_target = abs(shares_to_trade)
            # 实际能卖的股数 = min(想要卖的, 当前持有的)
            actual_shares_to_sell = min(shares_to_sell_target, current_shares)

            # 如果实际能卖的太少，就不卖了
            if actual_shares_to_sell * price < 1.0: # 名义价值太低
                 # print(f"信息: {date.date()} 尝试卖出 {shares_to_sell_target:.2f} 股 {sec_code}，但实际持有 {current_shares:.2f} 或卖出量过小。")
                 return 0.0, trade_type # 返回 'sell' 但交易额为0

            # 执行卖出
            proceeds_per_share = price * (1 - self.commission_rate - self.slippage) # 每股收入
            proceeds = actual_shares_to_sell * proceeds_per_share # 总收入
            self.cash += proceeds
            self.current_positions[sec_code] = current_shares - actual_shares_to_sell # 更新持仓
            trade_nominal_value = actual_shares_to_sell * price # 记录名义价值
            # print(f"交易: {date.date()} 卖出 {actual_shares_to_sell:.2f} 股 {sec_code} @ {price:.2f}, 收入 {proceeds:.2f}, 剩餘现金 {self.cash:.2f}")

            # 如果卖出后股数接近零，则从持仓字典中移除该键
            if self.current_positions[sec_code] < 1e-6:
                del self.current_positions[sec_code]

        return trade_nominal_value, trade_type

    def _record_holdings(self, date: pd.Timestamp):
        """记录指定日期收盘后的持仓详情 (股数, 價格, 价值, 权重)"""
        current_portfolio_value = self.get_current_value(date)
        if current_portfolio_value <= 1e-8: # 如果总价值为0或负，无法计算权重
             # 仍然记录空的持仓或零价值持仓
             for sec, shares in self.current_positions.items():
                  if shares > 1e-6:
                     self.holdings_history.append({
                        'datetime': date, 'sec_code': sec, 'shares': shares,
                        'price': 0.0, 'value': 0.0, 'weight': 0.0
                     })
             return

        # 记录现金的权重
        cash_weight = self.cash / current_portfolio_value if current_portfolio_value > 1e-8 else 0.0
        self.holdings_history.append({
             'datetime': date, 'sec_code': 'CASH', 'shares': self.cash,
             'price': 1.0, 'value': self.cash, 'weight': cash_weight
        })

        # 记录各个持仓的权重
        if date in self.close_prices.index:
            prices_today = self.close_prices.loc[date]
            for sec, shares in self.current_positions.items():
                if shares > 1e-6: # 只记录有效持仓
                    price = prices_today.get(sec, 0.0) # 获取价格，没有则为0
                    holding_value = shares * price
                    weight = holding_value / current_portfolio_value if current_portfolio_value > 1e-8 else 0.0
                    self.holdings_history.append({
                        'datetime': date, 'sec_code': sec, 'shares': shares,
                        'price': price, 'value': holding_value, 'weight': weight
                    })
        # else:
            # print(f"警告: 在 {date.date()} 无法获取收盘价，当日持仓记录可能不完整。")

    def get_holdings_history(self) -> pd.DataFrame:
        """返回持仓历史记录的 DataFrame"""
        return pd.DataFrame(self.holdings_history)

    def get_portfolio_history(self) -> pd.DataFrame:
         """返回投资组合每日净值历史的 DataFrame"""
         if not self.portfolio_history:
             return pd.DataFrame(columns=['datetime', 'total_value'])
         return pd.DataFrame(self.portfolio_history).set_index('datetime')

