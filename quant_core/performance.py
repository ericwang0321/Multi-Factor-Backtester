# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Dict

def get_drawdown_info(equity_curve: pd.Series) -> tuple[float, int, int]:
    """计算最大回撤及其相关信息"""
    peak = equity_curve.cummax()
    safe_peak = peak.replace(to_replace=0, value=np.nan).ffill().bfill()
    drawdown = (equity_curve - safe_peak) / safe_peak
    max_drawdown = drawdown.min()
    
    if pd.isna(max_drawdown) or max_drawdown >= 0:
        return 0.0, 0, 0

    mdd_end_date = drawdown.idxmin()
    mdd_start_date = equity_curve.loc[:mdd_end_date].idxmax()
    mdd_duration = (mdd_end_date - mdd_start_date).days
    
    underwater = drawdown < -1e-9
    if not underwater.any(): return 0.0, 0, 0
    
    groups = (underwater != underwater.shift()).cumsum()
    durations = underwater.groupby(groups).apply(lambda x: (x.index[-1] - x.index[0]).days if x.all() else 0)
    longest_water = durations.max()
    
    return float(max_drawdown), int(mdd_duration), int(longest_water)

def calculate_extended_metrics(
    portfolio_equity: pd.Series,
    benchmark_equity: pd.Series,
    portfolio_instance
) -> Dict:
    """计算综合指标，统一英文键名以适配前端显示"""
    metrics: Dict = {}
    
    # 对齐与归一化
    p_norm = portfolio_equity / portfolio_equity.iloc[0]
    b_norm = benchmark_equity.reindex(p_norm.index).ffill()
    b_norm = b_norm / b_norm.iloc[0]
    
    p_rets = p_norm.pct_change().dropna()
    b_rets = b_norm.pct_change().dropna()

    # --- 1. 回报指标 ---
    metrics['Strategy Return'] = p_norm.iloc[-1] - 1
    metrics['Benchmark Return'] = b_norm.iloc[-1] - 1
    metrics['Alpha'] = metrics['Strategy Return'] - metrics['Benchmark Return']
    
    days = (p_norm.index[-1] - p_norm.index[0]).days
    ann_p = (1 + metrics['Strategy Return']) ** (365.25 / max(days, 1)) - 1
    ann_b = (1 + metrics['Benchmark Return']) ** (365.25 / max(days, 1)) - 1
    
    metrics['Annual Return'] = ann_p
    metrics['Benchmark Annual Return'] = ann_b
    
    # --- 2. 风险与对比指标 ---
    ann_vol = p_rets.std() * np.sqrt(252)
    metrics['Volatility'] = ann_vol
    metrics['Sharpe Ratio'] = ann_p / ann_vol if ann_vol > 1e-8 else 0.0
    
    cov = p_rets.cov(b_rets)
    b_var = b_rets.var()
    metrics['Beta'] = cov / b_var if b_var > 1e-8 else 1.0
    
    active_ret = p_rets - b_rets
    tracking_error = active_ret.std() * np.sqrt(252)
    metrics['Info Ratio'] = (ann_p - ann_b) / tracking_error if tracking_error > 1e-8 else 0.0

    mdd, _, _ = get_drawdown_info(p_norm)
    metrics['Max Drawdown'] = mdd

    # --- 3. VaR & ES (95%) ---
    conf = 0.95
    metrics['VaR_95'] = p_rets.quantile(1 - conf)
    metrics['ES_95'] = p_rets[p_rets <= metrics['VaR_95']].mean()
    if len(p_rets) >= 60:
        metrics['rolling_var_series'] = p_rets.rolling(60).quantile(1 - conf)

    # --- 4. 交易成本归因 (核心修复：尝试多种可能的属性名) ---
    # 佣金
    comm = getattr(portfolio_instance, 'total_commission_paid', 
                   getattr(portfolio_instance, 'total_commission', 0.0))
    # 滑点
    slip = getattr(portfolio_instance, 'total_slippage_paid', 
                   getattr(portfolio_instance, 'total_slippage', 0.0))
    
    metrics['Total Cost'] = comm + slip
    metrics['Commission'] = comm
    metrics['Slippage'] = slip

    # 存储曲线
    metrics['strategy_curve'] = p_norm
    metrics['benchmark_curve'] = b_norm
    metrics['excess_curve'] = p_norm - b_norm

    return metrics

def display_metrics(metrics: Dict, benchmark_loaded: bool = False):
    """格式化打印性能指标"""
    print("\n" + "="*20 + " PERFORMANCE SUMMARY " + "="*20)
    # 按顺序展示主要指标
    order = [
        '总回报率', '年化回报率', '夏普比率', '最大回撤', 
        '总交易成本', '累计佣金支出', '累计滑点支出', '交易成本对收益损耗', '年化换手率'
    ]
    for key in order:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                fmt = ".2%" if "率" in key or "损耗" in key or "回报" in key else ".2f"
                print(f"{key:<20}: {val:{fmt}}")
            else:
                print(f"{key:<20}: {val}")
    print("="*55)