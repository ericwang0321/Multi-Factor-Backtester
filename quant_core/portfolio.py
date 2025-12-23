# quant_core/portfolio.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

class Portfolio:
    """
    负责管理投资组合的状态、现金和执行交易。
    已增强：支持交易成本追踪 + 持仓成本(Avg Cost)计算 + 【新增】交易日志记录
    """
    def __init__(self, open_prices: pd.DataFrame, close_prices: pd.DataFrame, initial_capital: float, commission_rate: float, slippage: float):
        if not isinstance(open_prices.index, pd.DatetimeIndex) or not isinstance(close_prices.index, pd.DatetimeIndex):
            raise ValueError("价格数据的索引必须是 DatetimeIndex。")

        self.open_prices = open_prices
        self.close_prices = close_prices
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage

        self.cash = initial_capital
        # {sec_code: shares} 持有股数
        self.current_positions: Dict[str, float] = defaultdict(float)
        
        # 追踪平均持仓成本 {sec_code: avg_cost_per_share}
        self.avg_costs: Dict[str, float] = defaultdict(float)
        
        # 记录每日总净值
        self.portfolio_history: List[Dict] = [] 
        # 记录调仓后的持仓详情
        self.holdings_history: List[Dict] = [] 
        # 记录换手率
        self.turnover_history: List[float] = [] 

        # --- 【核心新增 1】交易记录列表 ---
        self.trade_records: List[Dict] = []

        # 归因分析变量
        self.total_commission_paid = 0.0  
        self.total_slippage_paid = 0.0    

        # 记录第一天的初始状态
        if not self.open_prices.empty:
            self.portfolio_history.append({'datetime': self.open_prices.index[0], 'total_value': initial_capital})

    def get_current_value(self, date: pd.Timestamp) -> float:
        """获取指定日期收盘后的总资产 (持仓市值 + 现金)"""
        market_value = 0.0
        if date in self.close_prices.index:
            prices_today = self.close_prices.loc[date]
            for sec, shares in self.current_positions.items():
                if sec in prices_today.index:
                    p = prices_today[sec]
                    if pd.notna(p) and p > 0:
                        market_value += shares * p
        return market_value + self.cash

    def update_portfolio_value(self, date: pd.Timestamp):
        """记录当天的投资组合总价值到历史记录"""
        if self.portfolio_history and self.portfolio_history[-1]['datetime'] == date:
            self.portfolio_history.pop()

        total_value = self.get_current_value(date)
        self.portfolio_history.append({'datetime': date, 'total_value': total_value})
        return total_value

    def get_portfolio_state(self):
        """获取当前账户快照，供策略层风控使用"""
        return {
            'total_equity': self.portfolio_history[-1]['total_value'] if self.portfolio_history else self.initial_capital,
            'cash': self.cash,
            'positions': self.current_positions.copy(),
            'avg_costs': self.avg_costs.copy() # 传给策略比较止损
        }

    def rebalance(self, decision_date: pd.Timestamp, trade_date: pd.Timestamp, target_weights: Dict[str, float]):
        """根据目标权重执行调仓"""
        portfolio_value_before_trade = self.get_current_value(decision_date)

        if portfolio_value_before_trade <= 1e-8 or trade_date not in self.open_prices.index:
            self.turnover_history.append(0.0)
            self._record_holdings(trade_date)
            return

        prices_today = self.open_prices.loc[trade_date]

        # 1. 计算目标持有股数
        target_positions: Dict[str, float] = {}
        for sec, weight in target_weights.items():
            if weight > 1e-9:
                target_value = portfolio_value_before_trade * weight
                price = prices_today.get(sec, np.nan)
                if pd.notna(price) and price > 0:
                    cost_per_share = price * (1 + self.commission_rate + self.slippage)
                    target_shares = target_value / cost_per_share if cost_per_share > 0 else 0
                    if target_shares > 0:
                        target_positions[sec] = target_shares

        # 2. 生成交易列表
        trades: Dict[str, float] = defaultdict(float)
        # 现有持仓变动
        for sec, current_shares in self.current_positions.items():
            target_shares = target_positions.get(sec, 0.0)
            diff = target_shares - current_shares
            if abs(diff) > 1e-6: # 忽略微小浮点误差
                trades[sec] = diff
        
        # 新开仓
        for sec, target_shares in target_positions.items():
            if sec not in self.current_positions:
                trades[sec] = target_shares

        # 3. 执行交易 (先卖后买，释放资金)
        total_sells_nominal_value = 0.0
        total_buys_nominal_value = 0.0

        # 卖出逻辑
        sell_trades = {sec: s for sec, s in trades.items() if s < -1e-9}
        for sec, shares_to_sell in sell_trades.items():
            # 目标持仓量 = 当前 + 变动(负数)
            target_shares_after_sell = self.current_positions[sec] + shares_to_sell
            if target_shares_after_sell < 0: target_shares_after_sell = 0
            
            sell_nominal, _ = self._execute_trade(trade_date, sec, target_shares_after_sell)
            total_sells_nominal_value += sell_nominal

        # 买入逻辑
        buy_trades = {sec: s for sec, s in trades.items() if s > 1e-9}
        for sec, shares_to_buy in buy_trades.items():
            target_shares_after_buy = self.current_positions.get(sec, 0.0) + shares_to_buy
            buy_nominal, _ = self._execute_trade(trade_date, sec, target_shares_after_buy)
            total_buys_nominal_value += buy_nominal

        # 4. 记录换手率
        turnover = min(total_buys_nominal_value, total_sells_nominal_value) / portfolio_value_before_trade if portfolio_value_before_trade > 1e-8 else 0.0
        self.turnover_history.append(turnover)

        self._record_holdings(trade_date)

    def _execute_trade(self, date: pd.Timestamp, sec_code: str, target_shares: float) -> Tuple[float, Optional[str]]:
        """
        执行单笔交易并记录佣金与滑点。
        更新平均持仓成本 (Avg Cost)。
        【新增】记录交易日志到 self.trade_records。
        """
        price = self.open_prices.loc[date, sec_code] if date in self.open_prices.index and sec_code in self.open_prices.columns else np.nan

        if pd.isna(price) or price <= 0:
            return 0.0, None

        current_shares = self.current_positions.get(sec_code, 0.0)
        shares_to_trade = target_shares - current_shares

        if abs(shares_to_trade * price) < 1.0:
             return 0.0, None

        trade_nominal_value = abs(shares_to_trade * price)
        trade_type = 'buy' if shares_to_trade > 0 else 'sell'

        comm_cost = trade_nominal_value * self.commission_rate
        slip_cost = trade_nominal_value * self.slippage

        actual_shares_traded = 0.0 # 实际成交股数

        if trade_type == 'buy':
            # 买入：实际支出
            total_required = trade_nominal_value + comm_cost + slip_cost
            
            # 确定实际能买多少
            actual_buy_shares = shares_to_trade
            if self.cash < total_required:
                cost_per_share = price * (1 + self.commission_rate + self.slippage)
                actual_buy_shares = self.cash / cost_per_share if cost_per_share > 0 else 0
                trade_nominal_value = actual_buy_shares * price # 更新名义价值
                total_required = self.cash # 用光现金

            if actual_buy_shares * price < 1.0: # 金额太小不交易
                return 0.0, None

            # 计算加权平均成本
            old_cost_basis = current_shares * self.avg_costs[sec_code]
            new_cost_basis = trade_nominal_value + comm_cost + slip_cost 
            new_total_shares = current_shares + actual_buy_shares
            
            if new_total_shares > 0:
                self.avg_costs[sec_code] = (old_cost_basis + new_cost_basis) / new_total_shares

            # 更新资金和持仓
            self.cash -= total_required
            self.current_positions[sec_code] = new_total_shares
            
            # 记录归因
            self.total_commission_paid += trade_nominal_value * self.commission_rate
            self.total_slippage_paid += trade_nominal_value * self.slippage
            
            actual_shares_traded = actual_buy_shares

        elif trade_type == 'sell':
            # 卖出
            actual_shares_to_sell = min(abs(shares_to_trade), current_shares)
            real_nominal = actual_shares_to_sell * price
            
            proceeds = real_nominal - (real_nominal * self.commission_rate) - (real_nominal * self.slippage)
            self.cash += proceeds
            self.current_positions[sec_code] = current_shares - actual_shares_to_sell
            
            # 记录归因
            self.total_commission_paid += real_nominal * self.commission_rate
            self.total_slippage_paid += real_nominal * self.slippage
            trade_nominal_value = real_nominal
            
            actual_shares_traded = actual_shares_to_sell

            # 卖出不影响单位成本，只在清仓时移除
            if self.current_positions[sec_code] < 1e-6:
                del self.current_positions[sec_code]
                del self.avg_costs[sec_code] # 清仓，移除成本记录

        # --- 【核心新增 2】记录交易日志 ---
        if actual_shares_traded > 0:
            self.trade_records.append({
                'datetime': date,
                'sec_code': sec_code,
                'action': trade_type,
                'price': price,
                'shares': actual_shares_traded,
                'value': trade_nominal_value,
                'commission': trade_nominal_value * self.commission_rate,
                'slippage': trade_nominal_value * self.slippage
            })

        return trade_nominal_value, trade_type

    def _record_holdings(self, date: pd.Timestamp):
        """记录持仓详情"""
        current_portfolio_value = self.get_current_value(date)
        if current_portfolio_value <= 1e-8:
             return

        cash_weight = self.cash / current_portfolio_value
        self.holdings_history.append({
            'datetime': date, 'sec_code': 'CASH', 'shares': self.cash,
            'price': 1.0, 'value': self.cash, 'weight': cash_weight
        })

        if date in self.close_prices.index:
            prices_today = self.close_prices.loc[date]
            for sec, shares in self.current_positions.items():
                if shares > 1e-6:
                    price = prices_today.get(sec, 0.0)
                    holding_value = shares * price
                    weight = holding_value / current_portfolio_value
                    self.holdings_history.append({
                        'datetime': date, 'sec_code': sec, 'shares': shares,
                        'price': price, 'value': holding_value, 'weight': weight
                    })

    # --- 【核心新增 3】获取交易日志接口 ---
    def get_trade_log(self) -> pd.DataFrame:
        if not self.trade_records:
            return pd.DataFrame(columns=['datetime', 'sec_code', 'action', 'price', 'shares', 'value', 'commission', 'slippage'])
        return pd.DataFrame(self.trade_records)

    def get_holdings_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.holdings_history)

    def get_portfolio_history(self) -> pd.DataFrame:
        if not self.portfolio_history:
            return pd.DataFrame(columns=['datetime', 'total_value'])
        df = pd.DataFrame(self.portfolio_history)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df.set_index('datetime')