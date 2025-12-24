# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# =========================================================================
# 1. 注册与工厂模块 (Factory & Registry)
#    这里实现了“依赖倒置”：基类不知道有哪些子类，但子类会自己注册上来。
# =========================================================================

STRATEGY_REGISTRY = {}

def register_strategy(name):
    """
    策略注册装饰器
    用法: @register_strategy('linear')
    """
    def decorator(cls):
        STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator

def create_strategy_instance(strat_config: dict):
    """
    通用策略生产工厂
    """
    strat_type = strat_config.get('type')
    
    if strat_type not in STRATEGY_REGISTRY:
        raise ValueError(f"❌ 未知策略类型: '{strat_type}'。已注册: {list(STRATEGY_REGISTRY.keys())}")
    
    # 1. 获取对应的类
    strat_class = STRATEGY_REGISTRY[strat_type]
    
    # 2. 准备参数
    #   a. 提取 common 参数
    common_cfg = strat_config.get('common', {})
    
    #   b. 提取 type 特有的参数 (约定配置里的 key 必须是 "{type}_params")
    #      例如 type='linear', 则去找 'linear_params'
    specific_key = f"{strat_type}_params"
    specific_cfg = strat_config.get(specific_key, {})
    
    #   c. 提取风控参数 (从 common.risk 提取并平铺)
    risk_cfg = common_cfg.get('risk', {})
    
    # 3. 合并所有参数
    #    优先级: 风控参数 > 特有参数 > 通用参数
    #    注意: 我们把 key 平铺传入，这就要求策略类的 __init__ 参数名要和 config 里的 key 一致
    init_params = {
        'name': common_cfg.get('name', f'{strat_type}_strategy'),
        'top_k': common_cfg.get('top_k', 5),
        'stop_loss_pct': risk_cfg.get('stop_loss_pct'),
        'max_pos_weight': risk_cfg.get('max_pos_weight'),
        'max_drawdown_pct': risk_cfg.get('max_drawdown_pct'),
        **specific_cfg  # 比如 linear 的 'weights', ml 的 'model_path' 都在这里
    }
    
    print(f"🏭 工厂正在生产策略: {strat_type} | 参数 keys: {list(init_params.keys())}")
    return strat_class(**init_params)


# =========================================================================
# 2. 策略基类 (BaseStrategy)
# =========================================================================

class BaseStrategy(ABC):
    """
    策略基类 (Abstract Base Class)
    """
    
    def __init__(self, name: str, top_k: int = 5, 
                 stop_loss_pct: Optional[float] = None,
                 max_pos_weight: Optional[float] = None,
                 max_drawdown_pct: Optional[float] = None,
                 **kwargs): # <--- [关键修改] 必须加 **kwargs，吃掉多余参数
        
        self.name = name
        self.top_k = top_k
        self.factor_data: Optional[pd.DataFrame] = None
        
        # --- 风控参数 ---
        self.stop_loss_pct = stop_loss_pct
        self.max_pos_weight = max_pos_weight
        self.max_drawdown_pct = max_drawdown_pct
        
        # 内部状态
        self.peak_equity = 0.0
        
        # 打印被忽略的额外参数 (调试用)
        if kwargs:
            # 比如 BaseStrategy 不关心 weights，但它会被传进来，这里直接忽略即可
            pass

        print(f"[{self.name}] 基类初始化完成。Top-K: {self.top_k}")
        if any([stop_loss_pct, max_pos_weight, max_drawdown_pct]):
            print(f"🛡️ 风控开启: 止损={stop_loss_pct}, 限仓={max_pos_weight}, 熔断={max_drawdown_pct}")

    @abstractmethod
    def get_required_factors(self) -> List[str]:
        """策略声明所需因子"""
        pass

    def load_data(self, factor_df: pd.DataFrame, price_df: Optional[pd.DataFrame] = None):
        """注入数据"""
        self.factor_data = factor_df
        # price_df 若需使用可自行赋值
        print(f"[{self.name}] 数据加载完成。")

    def get_day_factors(self, date, universe_codes: List[str]) -> pd.DataFrame:
        """获取当日因子切片"""
        if self.factor_data is None: return pd.DataFrame()
        
        # 兼容性检查
        if not isinstance(self.factor_data.index, pd.MultiIndex):
            return pd.DataFrame()

        # 检查 Level 0 (日期)
        if date not in self.factor_data.index.get_level_values(0): 
            return pd.DataFrame()
        
        try:
            day_df = self.factor_data.loc[date]
            valid_codes = day_df.index.intersection(universe_codes)
            return day_df.loc[valid_codes]
        except KeyError:
            return pd.DataFrame()

    @abstractmethod
    def calculate_scores(self, factor_df: pd.DataFrame) -> pd.Series:
        """计算打分"""
        pass

    def calculate_weights(self, selected_codes: List[str], date) -> Dict[str, float]:
        """计算权重 (默认等权)"""
        if not selected_codes: return {}
        w = 1.0 / len(selected_codes)
        return {code: w for code in selected_codes}

    def _check_circuit_breaker(self, current_equity: float) -> bool:
        """账户熔断检查"""
        if self.max_drawdown_pct is None: return False
        if current_equity > self.peak_equity: self.peak_equity = current_equity
        if self.peak_equity <= 0: return False
            
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        return drawdown > self.max_drawdown_pct

    def on_bar(self, date, universe_codes: List[str], 
               portfolio_state: Dict[str, Any] = None, 
               current_prices: pd.Series = None) -> Dict[str, float]:
        """标准执行流水线"""
        
        # 1. 熔断检查
        if portfolio_state and self._check_circuit_breaker(portfolio_state.get('total_equity', 0)):
            return {} 

        # 2. 选股
        factors_df = self.get_day_factors(date, universe_codes)
        target_positions = {}
        
        if not factors_df.empty:
            scores = self.calculate_scores(factors_df).dropna()
            if not scores.empty:
                k = min(self.top_k, len(scores))
                # 使用 nlargest 选择前 K 个
                selected_codes = scores.nlargest(k).index.tolist()
                target_positions = self.calculate_weights(selected_codes, date)

        if not target_positions and (not portfolio_state or not portfolio_state.get('positions')):
            return {}

        # 3. 持仓风控 (限仓 & 止损)
        
        # A. 限仓
        if self.max_pos_weight is not None:
            for code in list(target_positions.keys()):
                if target_positions[code] > self.max_pos_weight:
                    target_positions[code] = self.max_pos_weight

        # B. 止损 (覆盖掉目标持仓)
        if self.stop_loss_pct is not None and portfolio_state and current_prices is not None:
            positions = portfolio_state.get('positions', {})
            avg_costs = portfolio_state.get('avg_costs', {})
            
            for code, shares in positions.items():
                if shares > 0 and code in avg_costs:
                    cost = avg_costs[code]
                    price = current_prices.get(code, np.nan)
                    
                    if pd.notna(price) and cost > 0:
                        ret = (price - cost) / cost
                        if ret < -self.stop_loss_pct:
                            # 触发止损，强制设为 0
                            target_positions[code] = 0.0

        return target_positions