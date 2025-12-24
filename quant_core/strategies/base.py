# quant_core/strategies/base.py
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

class BaseStrategy(ABC):
    """
    策略基类 (Abstract Base Class) - V4 (支持因子依赖声明)
    
    新增功能:
    1. get_required_factors: 策略主动声明所需因子列表 (依赖倒置)。
    
    保留功能:
    2. Circuit Breaker (熔断): 净值回撤超过阈值，强制空仓。
    3. Stop Loss (个股止损): 个股亏损超过阈值，强制剔除。
    4. Position Limit (限仓): 单票权重上限。
    
    流水线:
    OnBar -> 熔断检查 -> 算分 -> 选股 -> 定权 -> 限仓检查 -> 止损覆盖
    """
    
    def __init__(self, name: str, top_k: int = 5, 
                 stop_loss_pct: Optional[float] = None,      # e.g., 0.10 for 10%
                 max_pos_weight: Optional[float] = None,     # e.g., 0.30 for 30%
                 max_drawdown_pct: Optional[float] = None):  # e.g., 0.20 for 20%
        self.name = name
        self.top_k = top_k
        self.factor_data: Optional[pd.DataFrame] = None
        self.price_data: Optional[pd.DataFrame] = None 
        
        # --- 风控参数 ---
        self.stop_loss_pct = stop_loss_pct
        self.max_pos_weight = max_pos_weight
        self.max_drawdown_pct = max_drawdown_pct
        
        # 内部状态记录 (用于熔断计算)
        self.peak_equity = 0.0
        
        print(f"[{self.name}] 初始化完成。Top-K: {self.top_k}")
        if any([stop_loss_pct, max_pos_weight, max_drawdown_pct]):
            print(f"🛡️ 风控开启: 止损={stop_loss_pct}, 限仓={max_pos_weight}, 熔断={max_drawdown_pct}")

    # =========================================================================
    # [新增] 核心接口：依赖倒置
    # =========================================================================
    @abstractmethod
    def get_required_factors(self) -> List[str]:
        """
        【新增抽象方法】
        策略必须声明它依赖哪些因子名 (e.g., ['RSI', 'Momentum'] 或 ['feature_1', ...])
        RunLiveStrategy 会根据这个列表去 Bridge 取数据。
        """
        pass

    def load_data(self, factor_df: pd.DataFrame, price_df: Optional[pd.DataFrame] = None):
        """注入数据 (因子 + 可选的价格数据)"""
        self.factor_data = factor_df
        if price_df is not None:
            self.price_data = price_df
        print(f"[{self.name}] 数据加载完成。")

    def get_day_factors(self, date, universe_codes: List[str]) -> pd.DataFrame:
        """获取当日因子切片"""
        if self.factor_data is None: return pd.DataFrame()
        
        # 兼容性处理：确保 factor_data 是 MultiIndex (Date, Code)
        # 如果不是 MultiIndex，说明数据加载有问题，直接返回空
        if not isinstance(self.factor_data.index, pd.MultiIndex):
            return pd.DataFrame()

        # 检查日期是否在索引 Level 0 中
        if date not in self.factor_data.index.get_level_values(0): 
            return pd.DataFrame()
        
        try:
            day_df = self.factor_data.loc[date]
            # 筛选出 universe 里的代码，防止 KeyError
            valid_codes = [c for c in universe_codes if c in day_df.index]
            return day_df.loc[valid_codes]
        except KeyError:
            return pd.DataFrame()

    @abstractmethod
    def calculate_scores(self, factor_df: pd.DataFrame) -> pd.Series:
        """【抽象方法】计算打分 (Step 1)"""
        pass

    def calculate_weights(self, selected_codes: List[str], date) -> Dict[str, float]:
        """【虚方法】计算权重 (Step 2)"""
        if not selected_codes:
            return {}
        w = 1.0 / len(selected_codes)
        return {code: w for code in selected_codes}

    def _check_circuit_breaker(self, current_equity: float) -> bool:
        """
        检查是否触发账户级熔断
        Returns: True 表示触发熔断 (应空仓)，False 表示正常
        """
        if self.max_drawdown_pct is None:
            return False
            
        # 更新历史最高净值
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            
        if self.peak_equity <= 0: return False
            
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if drawdown > self.max_drawdown_pct:
            # print(f"⚠️ [{self.name}] 触发熔断! 回撤 {drawdown:.2%} > {self.max_drawdown_pct:.2%}")
            return True
        return False

    def on_bar(self, date, universe_codes: List[str], 
               portfolio_state: Dict[str, Any] = None, 
               current_prices: pd.Series = None) -> Dict[str, float]:
        """
        标准执行流水线 (含风控)
        
        Args:
            portfolio_state: 从 Engine 传入的账户状态 {'total_equity', 'positions', 'avg_costs'}
            current_prices: 当日所有股票的收盘价/开盘价 Series (用于算止损)
        """
        
        # --- 1. 账户级风控 (熔断) ---
        if portfolio_state and self._check_circuit_breaker(portfolio_state.get('total_equity', 0)):
            return {} # 触发熔断，返回空仓 (全卖)

        # --- 2. 正常选股逻辑 ---
        factors_df = self.get_day_factors(date, universe_codes)
        target_positions = {}
        
        if not factors_df.empty:
            scores = self.calculate_scores(factors_df).dropna()
            if not scores.empty:
                k = min(self.top_k, len(scores))
                selected_codes = scores.nlargest(k).index.tolist()
                target_positions = self.calculate_weights(selected_codes, date)

        # 如果没有选出股票且没有风控需求，直接返回
        if not target_positions and (not portfolio_state or not portfolio_state['positions']):
            return {}

        # --- 3. 持仓级风控 (限仓 & 止损) ---
        # 即使 target_positions 是空的，我们也可能需要处理现有的持仓进行止损
        
        # A. 单票限仓 (Position Limit)
        if self.max_pos_weight is not None:
            # 将所有目标权重截断到上限
            # 注意：这会导致总仓位 < 100%，多余部分变成现金，这是符合风控逻辑的
            for code in list(target_positions.keys()):
                if target_positions[code] > self.max_pos_weight:
                    target_positions[code] = self.max_pos_weight

        # B. 止损 (Stop Loss) - 最优先逻辑
        if self.stop_loss_pct is not None and portfolio_state and current_prices is not None:
            current_positions = portfolio_state.get('positions', {})
            avg_costs = portfolio_state.get('avg_costs', {})
            
            for code, shares in current_positions.items():
                if shares > 0 and code in avg_costs:
                    cost = avg_costs[code]
                    # 获取当前价格 (如果在 current_prices 里没有，尝试用 cost 避免报错，或者跳过)
                    price = current_prices.get(code, np.nan)
                    
                    if pd.notna(price) and cost > 0:
                        ret = (price - cost) / cost
                        if ret < -self.stop_loss_pct:
                            # 触发止损!
                            # 逻辑: 无论模型是否选中它，强制将其目标权重设为 0 (卖出)
                            # print(f"🛑 [{date}] {code} 触发止损 (亏损 {ret:.2%}), 强制平仓。")
                            target_positions[code] = 0.0

        return target_positions