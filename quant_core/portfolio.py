# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

class Portfolio:
    """
    负责管理投资组合的状态、现金和执行交易。
    工业级增强 V5: 
    1. 显式区分 Signal Price (用于定股数) vs Execution Price (用于结算)。
    2. 增加 2% Cash Buffer 防止跳空透支。
    3. 资金不足时自动缩减买单 (Order Truncation)。
    """
    def __init__(self, initial_capital: float, commission_rate: float, slippage: float):
        # [修改] 不再需要在 init 里传全量价格数据，由 Engine 逐日传入，解耦更彻底
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage

        self.cash = initial_capital
        
        # {sec_code: shares} 持有股数
        self.current_positions: Dict[str, float] = defaultdict(float)
        
        # 追踪平均持仓成本 {sec_code: avg_cost_per_share}
        self.avg_costs: Dict[str, float] = defaultdict(float)
        
        # 历史记录容器
        self.portfolio_history: List[Dict] = [] 
        self.holdings_history: List[Dict] = [] 
        self.turnover_history: List[float] = [] 
        self.trade_records: List[Dict] = []

        # 归因分析变量
        self.total_commission_paid = 0.0  
        self.total_slippage_paid = 0.0    

        # 记录初始状态 (时间设为 NaT，等待第一次 update 更新)
        self.portfolio_history.append({'datetime': pd.NaT, 'total_value': initial_capital})

    def get_current_equity(self, current_prices: pd.Series) -> float:
        """
        根据提供的最新价格计算总权益 (Mark-to-Market)
        """
        market_value = 0.0
        for sec, shares in self.current_positions.items():
            if shares != 0:
                price = current_prices.get(sec, np.nan)
                if pd.notna(price) and price > 0:
                    market_value += shares * price
        return market_value + self.cash

    def update_portfolio_value(self, date: pd.Timestamp, current_prices: pd.Series):
        """每日更新净值"""
        # 如果当天有重复记录先移除（防止同一天多次调用）
        if self.portfolio_history and self.portfolio_history[-1]['datetime'] == date:
            self.portfolio_history.pop()

        total_value = self.get_current_equity(current_prices)
        self.portfolio_history.append({'datetime': date, 'total_value': total_value})
        return total_value

    def get_portfolio_state(self):
        """获取当前账户快照，供策略层风控使用"""
        last_val = self.portfolio_history[-1]['total_value'] if self.portfolio_history else self.initial_capital
        return {
            'total_equity': last_val,
            'cash': self.cash,
            'positions': self.current_positions.copy(),
            'avg_costs': self.avg_costs.copy()
        }

    def rebalance(self, 
                  date: pd.Timestamp, 
                  target_weights: Dict[str, float], 
                  signal_prices: pd.Series, 
                  execution_prices: pd.Series):
        """
        [工业级调仓逻辑]
        
        Args:
            date: 交易日期
            target_weights: 策略输出的目标权重
            signal_prices: 信号价格 (T-1 收盘价)，用于计算【目标股数】
            execution_prices: 执行价格 (T 开盘价)，用于计算【实际成交金额】
        """
        # 1. 基于 Signal Price 估算当前的参考总资产
        #    (这一步模拟：在昨晚收盘后，我以为我有多少钱)
        estimated_equity = self.get_current_equity(signal_prices)
        
        # 资产太少不交易
        if estimated_equity <= 1e-6: 
            self.turnover_history.append(0.0)
            self._record_holdings(date, execution_prices)
            return

        # =========================================================
        # [修改点 1] 现金缓冲 (Cash Buffer)
        # 预留 2% 现金，只分配 98% 的资金去买股票，防止次日跳空导致透支
        # =========================================================
        SAFE_BUFFER = 0.02
        available_equity = estimated_equity * (1.0 - SAFE_BUFFER)

        # 2. 计算目标持有股数 (Target Shares)
        #    公式: (可用权益 * 目标权重) / 昨收价
        target_positions: Dict[str, float] = {}
        for sec, weight in target_weights.items():
            if weight > 1e-6:
                ref_price = signal_prices.get(sec, np.nan)
                if pd.notna(ref_price) and ref_price > 0:
                    target_value = available_equity * weight
                    # 向下取整，模拟实盘只能买整数股 (这里保留浮点模拟部分成交)
                    target_shares = target_value / ref_price 
                    target_positions[sec] = target_shares

        # 3. 生成订单差额 (Diff)
        trades: Dict[str, float] = defaultdict(float)
        all_secs = set(self.current_positions.keys()) | set(target_positions.keys())
        
        for sec in all_secs:
            tgt = target_positions.get(sec, 0.0)
            curr = self.current_positions.get(sec, 0.0)
            diff = tgt - curr
            if abs(diff) > 1e-6:
                trades[sec] = diff

        # 4. 执行交易
        #    原则：先卖后买，释放资金
        total_sells_nominal = 0.0
        total_buys_nominal = 0.0
        
        # --- A. 卖出流程 ---
        sell_orders = {k: v for k, v in trades.items() if v < -1e-6}
        for sec, shares_delta in sell_orders.items():
            exec_price = execution_prices.get(sec, np.nan)
            # 停牌或价格异常无法卖出
            if pd.isna(exec_price) or exec_price <= 0: continue 
            
            # 卖出金额 = 股数 * 价格 * (1 - 费率)
            revenue, nominal = self._execute_trade(date, sec, shares_delta, exec_price, 'sell')
            total_sells_nominal += nominal

        # --- B. 买入流程 (含 Gap Risk 处理) ---
        buy_orders = {k: v for k, v in trades.items() if v > 1e-6}
        
        for sec, shares_delta in buy_orders.items():
            exec_price = execution_prices.get(sec, np.nan)
            if pd.isna(exec_price) or exec_price <= 0: continue # 停牌无法买入

            # 估算需要多少钱
            cost_per_share = exec_price * (1 + self.commission_rate + self.slippage)
            estimated_cost = shares_delta * cost_per_share
            
            # =========================================================
            # [修改点 2] 资金硬约束检查 (Hard Cash Constraint)
            # 处理跳空高开导致的资金不足：如果钱不够，自动减少购买股数
            # =========================================================
            actual_delta = shares_delta
            if estimated_cost > self.cash:
                # 钱不够了！重新计算最大能买多少股
                # 留 0.1% 的极小余量防止精度误差
                max_shares = self.cash / cost_per_share * 0.999 
                if max_shares < 1e-6: 
                    continue # 连一股都买不起，跳过
                actual_delta = max_shares # 缩减订单
            
            cost, nominal = self._execute_trade(date, sec, actual_delta, exec_price, 'buy')
            total_buys_nominal += nominal

        # 5. 记录换手率
        turnover = 0.0
        if estimated_equity > 0:
            turnover = min(total_buys_nominal, total_sells_nominal) / estimated_equity
        self.turnover_history.append(turnover)

        # 6. 记录持仓快照 (用于归因)
        self._record_holdings(date, execution_prices)


    def _execute_trade(self, date, sec, shares_delta, price, side):
        """执行单笔交易，更新现金和持仓"""
        nominal_value = abs(shares_delta) * price
        commission = nominal_value * self.commission_rate
        slippage = nominal_value * self.slippage
        total_cost = nominal_value + commission + slippage if side == 'buy' else 0
        net_proceeds = nominal_value - commission - slippage if side == 'sell' else 0

        if side == 'buy':
            self.cash -= total_cost
            self.total_commission_paid += commission
            self.total_slippage_paid += slippage
            
            # 更新平均成本 (加权平均)
            curr_shares = self.current_positions[sec]
            if curr_shares + shares_delta > 0:
                old_cost = curr_shares * self.avg_costs[sec]
                new_cost = nominal_value + commission + slippage
                self.avg_costs[sec] = (old_cost + new_cost) / (curr_shares + shares_delta)
            
            self.current_positions[sec] += shares_delta

        elif side == 'sell':
            self.cash += net_proceeds
            self.total_commission_paid += commission
            self.total_slippage_paid += slippage
            self.current_positions[sec] += shares_delta # shares_delta is negative
            
            # 清理微小碎股
            if self.current_positions[sec] <= 1e-6:
                del self.current_positions[sec]
                if sec in self.avg_costs: del self.avg_costs[sec]

        # 记录日志
        self.trade_records.append({
            'datetime': date,
            'sec_code': sec,
            'action': side,
            'price': price,
            'shares': abs(shares_delta),
            'value': nominal_value,
            'commission': commission,
            'slippage': slippage
        })
        
        # 返回 (实际现金变动, 名义价值)
        return (net_proceeds if side == 'sell' else total_cost), nominal_value

    def _record_holdings(self, date, current_prices):
        """记录每日持仓比例"""
        equity = self.get_current_equity(current_prices)
        if equity <= 0: return

        # 记录现金
        self.holdings_history.append({
            'datetime': date, 'sec_code': 'CASH', 'shares': self.cash,
            'price': 1.0, 'value': self.cash, 'weight': self.cash/equity
        })
        # 记录股票
        for sec, shares in self.current_positions.items():
            price = current_prices.get(sec, 0)
            val = shares * price
            self.holdings_history.append({
                'datetime': date, 'sec_code': sec, 'shares': shares,
                'price': price, 'value': val, 'weight': val/equity
            })

    # Getters
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