# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm # 用于计算 Alpha, Beta
from typing import Dict, Optional

# ======================================================================
# 性能指标计算模块 (Performance Metrics Calculation Module)
# ======================================================================

def get_drawdown_info(equity_curve: pd.Series) -> tuple[float, int, int]:
    """计算最大回撤及其相关信息"""
    peak = equity_curve.cummax()
    # 避免除以零或负的 peak 值
    drawdown = (equity_curve - peak) / peak.replace(to_replace=0, value=np.nan).fillna(method='ffill')
    max_drawdown = drawdown.min()

    # 如果没有回撤或数据无效
    if pd.isna(max_drawdown) or max_drawdown >= 0:
        return 0.0, 0, 0

    try:
        mdd_end_date = drawdown.idxmin()
        # 找到 mdd_end_date 对应的 peak 值
        peak_at_mdd_end = peak.loc[mdd_end_date]
        # 找到在 mdd_end_date 之前，净值等于 peak_at_mdd_end 的最后一个日期
        mdd_start_date = equity_curve.loc[:mdd_end_date][equity_curve == peak_at_mdd_end].index[-1]
        mdd_duration = (mdd_end_date - mdd_start_date).days
    except (IndexError, KeyError):
         # 如果找不到精确匹配，可能发生在序列开头
         mdd_start_date = equity_curve.index[0]
         mdd_duration = (mdd_end_date - mdd_start_date).days

    # 计算最长回撤持续时间（水下时间）
    underwater_periods = drawdown < -1e-9 # 使用一个小阈值避免浮点误差
    if not underwater_periods.any():
        longest_dd_duration = 0
    else:
        # 计算连续为 True 的区段长度（天数）
        underwater_days_list = []
        current_streak_start = None
        for date, is_underwater in underwater_periods.items():
            if is_underwater and current_streak_start is None:
                current_streak_start = date # 记录水下开始日期
            elif not is_underwater and current_streak_start is not None:
                # 计算水下持续天数
                duration = (date - current_streak_start).days # 结束日当天不算在水下
                underwater_days_list.append(duration)
                current_streak_start = None # 重置
        # 处理最后一段可能仍在水下的情况
        if current_streak_start is not None:
             duration = (underwater_periods.index[-1] - current_streak_start).days + 1 # 加1包含最后一天
             underwater_days_list.append(duration)

        longest_dd_duration = max(underwater_days_list) if underwater_days_list else 0

    return float(max_drawdown), int(mdd_duration), int(longest_dd_duration)


def calculate_extended_metrics(
    portfolio_equity: pd.Series,
    benchmark_equity: pd.Series,
    portfolio_instance # 传入 Portfolio 实例以获取换手率
) -> Dict:
    """
    计算详细的回测性能指标。

    Args:
        portfolio_equity (pd.Series): 策略每日总净值时间序列。
        benchmark_equity (pd.Series): 基準每日总净值时间序列 (应预先对齐日期)。
        portfolio_instance (Portfolio): 运行完毕的 Portfolio 实例。

    Returns:
        dict: 包含各种性能指标的字典。
    """
    metrics: Dict = {}
    if portfolio_equity.empty or len(portfolio_equity) < 2:
        return {"错误": "投资组合净值曲线过短或为空。"}

    # 确保索引是 DatetimeIndex
    portfolio_equity.index = pd.to_datetime(portfolio_equity.index)
    benchmark_equity.index = pd.to_datetime(benchmark_equity.index)

    # 再次对齐基準数据到策略的日期索引 (ffill 保守填充)
    benchmark_equity_aligned = benchmark_equity.reindex(portfolio_equity.index, method='ffill')
    # 如果对齐后基準仍然有 NaN (例如回测开始时基準数据缺失)，则用第一个有效值回填
    benchmark_equity_aligned = benchmark_equity_aligned.fillna(method='bfill')
    # 如果基準仍然全是 NaN，标记为无效
    benchmark_is_valid = not benchmark_equity_aligned.isnull().all()
    if not benchmark_is_valid:
        print("警告: 基準数据无法对齐或全是 NaN，部分相对指标将为 N/A。")

    metrics['开始日期'] = portfolio_equity.index[0].strftime('%Y-%m-%d')
    metrics['结束日期'] = portfolio_equity.index[-1].strftime('%Y-%m-%d')

    # --- 绝对指标 ---
    initial_value = portfolio_equity.iloc[0]
    final_value = portfolio_equity.iloc[-1]
    if pd.isna(initial_value) or initial_value == 0: initial_value = 1.0 # 避免除零

    total_return = (final_value / initial_value) - 1
    metrics['总回报率'] = total_return

    days = (portfolio_equity.index[-1] - portfolio_equity.index[0]).days
    if days <= 0: days = 1 # 避免除以零
    # 使用 365.25 更精确地年化
    annualized_return = (1 + total_return) ** (365.25 / days) - 1
    metrics['年化回报率'] = annualized_return

    daily_returns = portfolio_equity.pct_change().dropna()
    if daily_returns.empty:
        annualized_volatility = 0.0
        sharpe_ratio = 0.0
    else:
        annualized_volatility = daily_returns.std() * np.sqrt(252) # 假设每年252交易日
        # 夏普比率 (假设无风险利率为0)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 1e-8 else 0.0
    metrics['年化波动率'] = annualized_volatility
    metrics['夏普比率'] = sharpe_ratio

    downside_returns = daily_returns[daily_returns < 0.0]
    if downside_returns.empty:
        downside_deviation = 0.0
        sortino_ratio = 0.0
    else:
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 1e-8 else 0.0
    metrics['下行标准差'] = downside_deviation
    metrics['索提诺比率'] = sortino_ratio

    max_drawdown, mdd_duration, longest_dd_duration = get_drawdown_info(portfolio_equity)
    metrics['最大回撤'] = max_drawdown
    # Calmar 比率 = 年化收益 / abs(最大回撤)
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < -1e-8 else 0.0
    metrics['卡玛比率'] = calmar_ratio
    metrics['最大回撤持续天数'] = f"{mdd_duration} 天"
    metrics['最长回撤持续天数'] = f"{longest_dd_duration} 天"

    # 月度指标 (至少需要两个月数据)
    monthly_equity = portfolio_equity.resample('M').last()
    win_rate_monthly: float = 0.0
    profit_loss_ratio: float = 0.0
    if len(monthly_equity) >= 2:
        monthly_returns = monthly_equity.pct_change().dropna()
        if not monthly_returns.empty:
            win_rate_monthly = (monthly_returns > 0).mean()
            avg_win = monthly_returns[monthly_returns > 0].mean()
            avg_loss = monthly_returns[monthly_returns < 0].mean()
            if pd.isna(avg_loss) or avg_loss == 0:
                 profit_loss_ratio = np.inf if not pd.isna(avg_win) and avg_win > 0 else 0.0
            else:
                 profit_loss_ratio = abs(avg_win / avg_loss) if not pd.isna(avg_win) else 0.0
    metrics['月度胜率'] = win_rate_monthly
    metrics['盈亏比'] = profit_loss_ratio

    # --- 相对指标 ---
    if benchmark_is_valid:
        benchmark_initial = benchmark_equity_aligned.iloc[0]
        benchmark_final = benchmark_equity_aligned.iloc[-1]
        if pd.isna(benchmark_initial) or benchmark_initial == 0: benchmark_initial = 1.0

        benchmark_total_return = (benchmark_final / benchmark_initial) - 1
        benchmark_ann_return = (1 + benchmark_total_return) ** (365.25 / days) - 1
        benchmark_daily_returns = benchmark_equity_aligned.pct_change().dropna()

        # 对齐策略和基準的日收益率
        aligned_returns = pd.concat([daily_returns, benchmark_daily_returns], axis=1, join='inner')
        aligned_returns.columns = ['portfolio', 'benchmark']

        if not aligned_returns.empty and len(aligned_returns) >= 2:
            # Alpha 和 Beta (年化)
            try:
                alpha, beta, alpha_p_value, r_squared = perform_regression_analysis(
                    aligned_returns['portfolio'], aligned_returns['benchmark']
                )
                metrics['年化 Alpha'] = alpha
                metrics['Beta'] = beta
                metrics['Alpha P-value'] = alpha_p_value
                metrics['R-squared'] = r_squared
            except Exception as e:
                 print(f"警告: 回归分析失败: {e}")
                 metrics['年化 Alpha'] = np.nan
                 metrics['Beta'] = np.nan
                 metrics['Alpha P-value'] = np.nan
                 metrics['R-squared'] = np.nan


            # 信息比率 = (策略年化 - 基準年化) / 年化跟踪误差
            excess_returns = aligned_returns['portfolio'] - aligned_returns['benchmark']
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = (annualized_return - benchmark_ann_return) / tracking_error if tracking_error > 1e-8 else 0.0
            metrics['信息比率'] = information_ratio
        else:
             # 如果无法对齐收益率或数据不足
             metrics['年化 Alpha'] = np.nan
             metrics['Beta'] = np.nan
             metrics['Alpha P-value'] = np.nan
             metrics['R-squared'] = np.nan
             metrics['信息比率'] = np.nan

        metrics['年化超额回报'] = annualized_return - benchmark_ann_return
        metrics['总超额回报'] = total_return - benchmark_total_return
    else:
        # 如果基準无效
        metrics['年化 Alpha'] = np.nan
        metrics['Beta'] = np.nan
        metrics['Alpha P-value'] = np.nan
        metrics['R-squared'] = np.nan
        metrics['信息比率'] = np.nan
        metrics['年化超额回报'] = np.nan
        metrics['总超额回报'] = np.nan

    # --- 换手率 ---
    avg_turnover = np.mean(portfolio_instance.turnover_history) if portfolio_instance.turnover_history else 0.0
    # 从 portfolio 的 config 中获取调仓频率
    rebalance_months = 1 # 默认值
    if portfolio_instance.config and 'REBALANCE_MONTHS' in portfolio_instance.config:
        rebalance_months = portfolio_instance.config['REBALANCE_MONTHS']
    ann_turnover = avg_turnover * (12 / rebalance_months) if rebalance_months > 0 else 0.0
    metrics['年化换手率'] = ann_turnover

    return metrics

def perform_regression_analysis(strategy_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.0) -> tuple[float, float, float, float]:
    """
    执行策略对基準的 OLS 回归分析。
    返回: 年化 Alpha, Beta, Alpha P-value, R-squared。
    (来自你的 backtest.py)
    """
    # 确保输入的 Series 名称不同，以避免合并问题
    strategy_returns = strategy_returns.rename('strategy')
    benchmark_returns = benchmark_returns.rename('benchmark')

    # 计算日无风险利率
    daily_risk_free = (1 + risk_free_rate)**(1/252) - 1

    # 合并并移除 NaN
    df = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()

    if len(df) < 2: # 回归需要至少两个数据点
        print("警告: 用于回归分析的数据点不足 (<2)。")
        return 0.0, 0.0, 1.0, 0.0

    # 计算超额收益
    df['strategy_excess'] = df['strategy'] - daily_risk_free
    df['benchmark_excess'] = df['benchmark'] - daily_risk_free

    # 准备回归变量
    Y = df['strategy_excess']
    X = sm.add_constant(df['benchmark_excess']) # 添加常数项 (截距项 Alpha)

    # 执行 OLS 回归
    try:
        model = sm.OLS(Y, X, missing='drop').fit() # missing='drop' 确保稳健性

        # 提取结果
        alpha_daily = model.params.get('const', 0.0) # 获取截距项 alpha (日)
        beta = model.params.get('benchmark_excess', 0.0) # 获取斜率项 beta
        p_value_alpha = model.pvalues.get('const', 1.0) # 获取 alpha 的 p 值
        r_squared = model.rsquared

        # 年化 Alpha: (1 + 日 alpha)^252 - 1
        alpha_annualized = (1 + alpha_daily)**252 - 1

        return float(alpha_annualized), float(beta), float(p_value_alpha), float(r_squared)
    except Exception as e:
        print(f"回归分析执行失败: {e}")
        return 0.0, 0.0, 1.0, 0.0 # 失败时返回默认值

def display_metrics(metrics: Dict):
    """格式化并打印性能指标字典"""
    print("\n" + "="*25 + " 回测性能指标 " + "="*25)
    print(f"回测期间: {metrics.get('开始日期', 'N/A')} 到 {metrics.get('结束日期', 'N/A')}")
    print("-" * 65)

    # 定义指标显示名称和格式
    METRICS_DISPLAY_ORDER = [
        ('总回报率', '.2%'), ('年化回报率', '.2%'), ('年化波动率', '.2%'),
        ('夏普比率', '.2f'), ('索提诺比率', '.2f'), ('卡玛比率', '.2f'),
        ('最大回撤', '.2%'), ('最大回撤持续天数', 's'), ('最长回撤持续天数', 's'),
        ('月度胜率', '.2%'), ('盈亏比', '.2f'), ('年化换手率', '.2%'),
        ('Beta', '.2f'), ('年化 Alpha', '.2%'), ('Alpha P-value', '.4f'),
        ('R-squared', '.2%'), ('信息比率', '.2f'), ('年化超额回报', '.2%')
    ]

    for display_name, fmt in METRICS_DISPLAY_ORDER:
        if display_name in metrics:
            value = metrics[display_name]
            # 统一处理 NaN 和格式化
            if pd.isna(value):
                value_str = "N/A"
            elif fmt == 's': # 字符串直接显示
                 value_str = str(value)
            elif isinstance(value, (int, float)):
                try:
                    value_str = format(value, fmt)
                except (TypeError, ValueError):
                    value_str = f"{value:.4f}" # 默认格式
            else:
                 value_str = str(value) # 其他类型直接转字符串

            # 为 p-value 添加显著性标记
            if display_name == 'Alpha P-value' and isinstance(value, (float, int)) and not pd.isna(value):
                 if value < 0.01: value_str += " (***)" # 极显著
                 elif value < 0.05: value_str += " (**)"  # 显著
                 elif value < 0.10: value_str += " (*)"   # 弱显著

            print(f"{display_name:<25}: {value_str}")
        # else:
            # 如果某个指标计算失败或不存在，可以不显示或显示 N/A
            # print(f"{display_name:<25}: N/A")

    print("-" * 65)

