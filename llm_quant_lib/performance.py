# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, Optional

class Portfolio:
    """临时空类用于类型提示"""
    pass

def get_drawdown_info(equity_curve: pd.Series) -> tuple[float, int, int]:
    """计算最大回撤及其相关信息"""
    peak = equity_curve.cummax()
    safe_peak = peak.replace(to_replace=0, value=np.nan).ffill().bfill()
    
    if safe_peak.isnull().all() or (safe_peak <= 0).any():
         first_valid_peak = safe_peak[safe_peak > 0].first_valid_index()
         if first_valid_peak is None:
              return 0.0, 0, 0
         safe_peak = safe_peak.fillna(safe_peak[first_valid_peak])
         safe_peak[safe_peak <= 0] = safe_peak[first_valid_peak]

    drawdown = (equity_curve - safe_peak) / safe_peak
    max_drawdown = drawdown.min()

    if pd.isna(max_drawdown) or max_drawdown >= 0:
        return 0.0, 0, 0

    try:
        mdd_end_date = drawdown.idxmin()
        peak_at_mdd_end = peak.loc[mdd_end_date]
        mdd_start_date_candidates = equity_curve.loc[:mdd_end_date][equity_curve == peak_at_mdd_end]
        if mdd_start_date_candidates.empty:
             mdd_start_date = peak.loc[:mdd_end_date][peak == peak_at_mdd_end].index[-1]
        else:
             mdd_start_date = mdd_start_date_candidates.index[-1]
        mdd_duration = (mdd_end_date - mdd_start_date).days
    except:
         mdd_start_date = equity_curve.index[0]
         mdd_duration = (mdd_end_date - mdd_start_date).days

    underwater_periods = drawdown < -1e-9
    if not underwater_periods.any():
        longest_dd_duration = 0
    else:
        underwater_days_list = []
        current_streak_start = None
        extended_periods = pd.concat([underwater_periods, pd.Series([False], index=[underwater_periods.index[-1] + pd.Timedelta(days=1)])])

        for date, is_underwater in extended_periods.items():
            if is_underwater and current_streak_start is None:
                current_streak_start = date
            elif not is_underwater and current_streak_start is not None:
                duration = (date - current_streak_start).days
                underwater_days_list.append(duration)
                current_streak_start = None
        longest_dd_duration = max(underwater_days_list) if underwater_days_list else 0

    return float(max_drawdown), int(mdd_duration), int(longest_dd_duration)


def calculate_extended_metrics(
    portfolio_equity: pd.Series,
    benchmark_equity: pd.Series,
    portfolio_instance 
) -> Dict:
    """计算详细的回测性能指标，包含成本归因及 VaR/ES"""
    metrics: Dict = {}
    if portfolio_equity.empty or len(portfolio_equity) < 2:
        return {"错误": "净值序列不足"}

    portfolio_equity.index = pd.to_datetime(portfolio_equity.index)
    daily_returns = portfolio_equity.pct_change().dropna()

    metrics['开始日期'] = portfolio_equity.index[0].strftime('%Y-%m-%d')
    metrics['结束日期'] = portfolio_equity.index[-1].strftime('%Y-%m-%d')

    # --- 1. 基础回报与风险指标 ---
    initial_value = portfolio_equity.iloc[0]
    final_value = portfolio_equity.iloc[-1]
    total_return = (final_value / initial_value) - 1
    metrics['总回报率'] = total_return

    days = (portfolio_equity.index[-1] - portfolio_equity.index[0]).days
    annualized_return = (1 + total_return) ** (365.25 / max(days, 1)) - 1
    metrics['年化回报率'] = annualized_return

    ann_vol = daily_returns.std() * np.sqrt(252)
    metrics['年化波动率'] = ann_vol
    metrics['夏普比率'] = annualized_return / ann_vol if ann_vol > 1e-8 else 0.0

    mdd, mdd_days, water_days = get_drawdown_info(portfolio_equity)
    metrics['最大回撤'] = mdd
    metrics['卡玛比率'] = annualized_return / abs(mdd) if abs(mdd) > 1e-8 else 0.0

    # --- 2. 【核心新增】风险度量 (95% 置信度) ---
    confidence_level = 0.95
    # 历史 VaR: 5% 分位数
    metrics['历史 VaR (95%)'] = daily_returns.quantile(1 - confidence_level)
    # 历史 ES: 损失超过 VaR 时的平均损失
    metrics['预期缺口 ES (95%)'] = daily_returns[daily_returns <= metrics['历史 VaR (95%)']].mean()
    
    # 每日 60 日滚动 VaR (用于绘图)
    if len(daily_returns) >= 60:
        metrics['rolling_var_series'] = daily_returns.rolling(60).quantile(1 - confidence_level)

    # --- 3. 交易成本归因 ---
    comm_paid = getattr(portfolio_instance, 'total_commission_paid', 0.0)
    slip_paid = getattr(portfolio_instance, 'total_slippage_paid', 0.0)
    total_cost = comm_paid + slip_paid
    metrics['总交易成本'] = total_cost
    metrics['累计佣金支出'] = comm_paid
    metrics['累计滑点支出'] = slip_paid
    
    theoretical_return = ((final_value + total_cost) / initial_value) - 1
    metrics['理论无成本总回报'] = theoretical_return
    metrics['交易成本对收益损耗'] = theoretical_return - total_return
    
    # --- 4. 换手率 ---
    turnover_hist = getattr(portfolio_instance, 'turnover_history', [])
    avg_turnover = np.mean(turnover_hist) if turnover_hist else 0.0
    rebal_days = 20
    if hasattr(portfolio_instance, 'config') and portfolio_instance.config:
        rebal_days = portfolio_instance.config.get('REBALANCE_DAYS', 20)
    metrics['年化换手率'] = avg_turnover * (252 / rebal_days)

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