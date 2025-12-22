# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm # 用于计算 Alpha, Beta
from typing import Dict, Optional
# 导入 Portfolio 类型提示，避免循环导入问题，这里用 'Portfolio' 字符串
# from .portfolio import Portfolio # 通常不推荐直接导入，可能会循环

class Portfolio: # 创建一个临时的空类用于类型提示
     pass

# ======================================================================
# 性能指标计算模块 (Performance Metrics Calculation Module)
# ======================================================================

def get_drawdown_info(equity_curve: pd.Series) -> tuple[float, int, int]:
    """计算最大回撤及其相关信息"""
    peak = equity_curve.cummax()
    # 避免除以零或负的 peak 值
    # 使用 ffill 填充 peak 中可能因 equity_curve 开始有 NaN 导致的 NaN
    safe_peak = peak.replace(to_replace=0, value=np.nan).ffill().bfill() # 先前向填充再后向填充
    if safe_peak.isnull().all() or (safe_peak <= 0).any(): # 如果全是NaN或有非正值
         # 尝试找到第一个非 NaN 且 > 0 的值作为基准
         first_valid_peak = safe_peak[safe_peak > 0].first_valid_index()
         if first_valid_peak is None: # 如果找不到，无法计算
              print("警告: 无法计算有效的峰值用于回撤计算。")
              return 0.0, 0, 0
         # 将所有无效 peak 填充为第一个有效 peak
         safe_peak = safe_peak.fillna(safe_peak[first_valid_peak])
         safe_peak[safe_peak <= 0] = safe_peak[first_valid_peak] # 将非正值也替换

    drawdown = (equity_curve - safe_peak) / safe_peak
    max_drawdown = drawdown.min()

    # 如果没有回撤或数据无效
    if pd.isna(max_drawdown) or max_drawdown >= 0:
        return 0.0, 0, 0

    try:
        mdd_end_date = drawdown.idxmin()
        # 找到 mdd_end_date 对应的 peak 值
        peak_at_mdd_end = peak.loc[mdd_end_date]
        # 找到在 mdd_end_date 之前，净值等于 peak_at_mdd_end 的最后一个日期
        mdd_start_date_candidates = equity_curve.loc[:mdd_end_date][equity_curve == peak_at_mdd_end]
        if mdd_start_date_candidates.empty:
             # 如果找不到完全相等的（可能因为浮点数），找最接近的峰值日期
             mdd_start_date = peak.loc[:mdd_end_date][peak == peak_at_mdd_end].index[-1]
        else:
             mdd_start_date = mdd_start_date_candidates.index[-1]

        mdd_duration = (mdd_end_date - mdd_start_date).days
    except (IndexError, KeyError, Exception) as e:
         # 如果查找失败，使用序列开始作为近似起点
         print(f"警告: 计算最大回撤起始日期时出错 ({e})，使用近似值。")
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
        # 添加一个哨兵日期，确保最后一个水下期能被计算
        extended_periods = pd.concat([underwater_periods, pd.Series([False], index=[underwater_periods.index[-1] + pd.Timedelta(days=1)])])

        for date, is_underwater in extended_periods.items():
            if is_underwater and current_streak_start is None:
                current_streak_start = date # 记录水下开始日期
            elif not is_underwater and current_streak_start is not None:
                # 计算水下持续天数 (结束日期 - 开始日期)
                duration = (date - current_streak_start).days # 结束日当天已不再水下
                underwater_days_list.append(duration)
                current_streak_start = None # 重置

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
        benchmark_equity (pd.Series): 基准每日总净值时间序列 (应预先对齐日期)。
        portfolio_instance (Portfolio): 运行完毕的 Portfolio 实例。

    Returns:
        dict: 包含各种性能指标的字典。
    """
    metrics: Dict = {}
    if portfolio_equity.empty or len(portfolio_equity) < 2:
        metrics['错误'] = "投资组合净值曲线过短或为空。"
        # 返回包含基本信息和错误标记的字典
        metrics['开始日期'] = 'N/A'
        metrics['结束日期'] = 'N/A'
        return metrics

    # 确保索引是 DatetimeIndex
    portfolio_equity.index = pd.to_datetime(portfolio_equity.index)
    benchmark_equity.index = pd.to_datetime(benchmark_equity.index)

    # 再次对齐基准数据到策略的日期索引 (ffill 保守填充)
    # 确保benchmark_equity有有效的起始值，如果起始值是NaN则填充为策略的起始值
    initial_strategy_value = portfolio_equity.iloc[0]
    if pd.isna(initial_strategy_value): initial_strategy_value = 1.0 # 备用值

    benchmark_equity_aligned = benchmark_equity.reindex(portfolio_equity.index, method='ffill')
    benchmark_equity_aligned = benchmark_equity_aligned.fillna(method='bfill') # 先后向填充
    benchmark_equity_aligned = benchmark_equity_aligned.fillna(initial_strategy_value) # 用策略起始值填充剩余 NaN

    benchmark_is_valid = not benchmark_equity_aligned.isnull().all() and len(benchmark_equity_aligned) > 1
    if not benchmark_is_valid:
        print("警告: 基准数据无法对齐或全是 NaN，部分相对指标将为 N/A。")
        # 创建一个常量基准用于兼容性
        benchmark_equity_aligned = pd.Series(initial_strategy_value, index=portfolio_equity.index)


    metrics['开始日期'] = portfolio_equity.index[0].strftime('%Y-%m-%d')
    metrics['结束日期'] = portfolio_equity.index[-1].strftime('%Y-%m-%d')

    # --- 绝对指标 ---
    initial_value = portfolio_equity.iloc[0]
    final_value = portfolio_equity.iloc[-1]
    if pd.isna(initial_value) or initial_value == 0:
         print("警告: 策略起始净值无效，回报率计算可能不准确。")
         initial_value = 1.0 # 使用备用值

    total_return = (final_value / initial_value) - 1
    metrics['总回报率'] = total_return

    days = (portfolio_equity.index[-1] - portfolio_equity.index[0]).days
    if days <= 0: days = 1 # 避免除以零
    annualized_return = (1 + total_return) ** (365.25 / days) - 1
    metrics['年化回报率'] = annualized_return

    daily_returns = portfolio_equity.pct_change().dropna()
    if daily_returns.empty or daily_returns.isnull().all():
        annualized_volatility = 0.0
        sharpe_ratio = 0.0
    else:
        # ---【健壮性】计算波动率前移除 Inf/-Inf ---
        daily_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        annualized_volatility = daily_returns.std(skipna=True) * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 1e-8 else 0.0
    metrics['年化波动率'] = annualized_volatility
    metrics['夏普比率'] = sharpe_ratio

    downside_returns = daily_returns[daily_returns < 0.0].dropna() # 再次 dropna
    if downside_returns.empty or downside_returns.isnull().all():
        downside_deviation = 0.0
        sortino_ratio = 0.0
    else:
        downside_deviation = downside_returns.std(skipna=True) * np.sqrt(252)
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 1e-8 else 0.0
    metrics['下行标准差'] = downside_deviation
    metrics['索提诺比率'] = sortino_ratio

    max_drawdown, mdd_duration, longest_dd_duration = get_drawdown_info(portfolio_equity)
    metrics['最大回撤'] = max_drawdown
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < -1e-8 else 0.0
    metrics['卡玛比率'] = calmar_ratio
    metrics['最大回撤持续天数'] = f"{mdd_duration} 天"
    metrics['最长水下持续天数'] = f"{longest_dd_duration} 天" # 修改名称更清晰

    # 月度指标
    # ---【健壮性】确保索引唯一且排序 ---
    portfolio_equity_unique = portfolio_equity[~portfolio_equity.index.duplicated(keep='last')].sort_index()
    monthly_equity = portfolio_equity_unique.resample('ME').last() # 使用 'M' (Month End)
    win_rate_monthly: float = np.nan # 默认为 NaN
    profit_loss_ratio: float = np.nan # 默认为 NaN
    if len(monthly_equity) >= 2:
        monthly_returns = monthly_equity.pct_change().dropna()
        if not monthly_returns.empty:
            win_rate_monthly = (monthly_returns > 0).mean()
            avg_win = monthly_returns[monthly_returns > 0].mean()
            avg_loss = monthly_returns[monthly_returns < 0].mean()

            # ---【健壮性】处理 avg_win 或 avg_loss 为 NaN 或 0 的情况 ---
            if pd.isna(avg_loss) or avg_loss == 0:
                profit_loss_ratio = np.inf if not pd.isna(avg_win) and avg_win > 0 else np.nan
            elif pd.isna(avg_win): # 如果没有盈利月份
                 profit_loss_ratio = 0.0
            else:
                profit_loss_ratio = abs(avg_win / avg_loss)

    metrics['月度胜率'] = win_rate_monthly
    metrics['月度盈亏比'] = profit_loss_ratio # 修改名称

    # --- 相对指标 ---
    if benchmark_is_valid:
        benchmark_initial = benchmark_equity_aligned.iloc[0]
        benchmark_final = benchmark_equity_aligned.iloc[-1]
        if pd.isna(benchmark_initial) or benchmark_initial == 0:
             print("警告: 基准起始净值无效，相对指标计算可能不准确。")
             benchmark_initial = 1.0

        benchmark_total_return = (benchmark_final / benchmark_initial) - 1
        benchmark_ann_return = (1 + benchmark_total_return) ** (365.25 / days) - 1
        benchmark_daily_returns = benchmark_equity_aligned.pct_change().dropna()
        # ---【健壮性】处理 Inf/-Inf ---
        benchmark_daily_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        benchmark_daily_returns.dropna(inplace=True)

        # 对齐策略和基准的日收益率
        # 确保 daily_returns 也处理了 Inf/-Inf
        daily_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        daily_returns.dropna(inplace=True)

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


            # 信息比率 = 年化超额回报 / 年化跟踪误差
            excess_returns = aligned_returns['portfolio'] - aligned_returns['benchmark']
            tracking_error = excess_returns.std(skipna=True) * np.sqrt(252)
            annualized_excess_return = annualized_return - benchmark_ann_return
            information_ratio = annualized_excess_return / tracking_error if tracking_error > 1e-8 else 0.0
            metrics['信息比率'] = information_ratio
            metrics['年化跟踪误差'] = tracking_error # 添加跟踪误差
        else:
             print("警告: 策略和基准收益率无法对齐或数据不足，无法计算 Alpha, Beta, IR。")
             metrics['年化 Alpha'] = np.nan
             metrics['Beta'] = np.nan
             metrics['Alpha P-value'] = np.nan
             metrics['R-squared'] = np.nan
             metrics['信息比率'] = np.nan
             metrics['年化跟踪误差'] = np.nan

        metrics['年化超额回报'] = annualized_return - benchmark_ann_return
        metrics['总超额回报'] = total_return - benchmark_total_return
    else:
        # 如果基准无效
        metrics['年化 Alpha'] = np.nan
        metrics['Beta'] = np.nan
        metrics['Alpha P-value'] = np.nan
        metrics['R-squared'] = np.nan
        metrics['信息比率'] = np.nan
        metrics['年化跟踪误差'] = np.nan
        metrics['年化超额回报'] = np.nan
        metrics['总超额回报'] = np.nan

    # --- 换手率 ---
    # ---【修改】从 portfolio_instance 获取换手率历史 ---
    turnover_hist = []
    if hasattr(portfolio_instance, 'turnover_history'):
         turnover_hist = portfolio_instance.turnover_history
    avg_turnover = np.mean(turnover_hist) if turnover_hist else 0.0
    rebalance_months = 1 # 默认值
    if hasattr(portfolio_instance, 'config') and portfolio_instance.config and 'REBALANCE_MONTHS' in portfolio_instance.config:
        rebalance_months = portfolio_instance.config['REBALANCE_MONTHS']
        if not isinstance(rebalance_months, (int, float)) or rebalance_months <= 0:
             print(f"警告: 无效的调仓频率 ({rebalance_months})，换手率计算可能不准确。")
             rebalance_months = 1 # 使用默认值
    ann_turnover = avg_turnover * (12 / rebalance_months)
    metrics['年化换手率'] = ann_turnover

    return metrics

def perform_regression_analysis(strategy_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.0) -> tuple[float, float, float, float]:
    """
    执行策略对基准的 OLS 回归分析。
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
        # ---【修改】返回 NaN 而不是 0 ---
        return np.nan, np.nan, np.nan, np.nan

    # 计算超额收益
    df['strategy_excess'] = df['strategy'] - daily_risk_free
    df['benchmark_excess'] = df['benchmark'] - daily_risk_free

    # 准备回归变量
    Y = df['strategy_excess']
    X = sm.add_constant(df['benchmark_excess']) # 添加常数项 (截距项 Alpha)
    # ---【修改】确保列名唯一，防止与 statsmodels 内部冲突 ---
    X.columns = ['const', 'benchmark_excess_beta']

    # 执行 OLS 回归
    try:
        model = sm.OLS(Y, X, missing='drop').fit() # missing='drop' 确保稳健性

        # 提取结果
        alpha_daily = model.params.get('const', 0.0) # 获取截距项 alpha (日)
        beta = model.params.get('benchmark_excess_beta', 0.0) # 获取斜率项 beta
        p_value_alpha = model.pvalues.get('const', 1.0) # 获取 alpha 的 p 值
        r_squared = model.rsquared

        # 年化 Alpha: (1 + 日 alpha)^252 - 1
        alpha_annualized = (1 + alpha_daily)**252 - 1

        return float(alpha_annualized), float(beta), float(p_value_alpha), float(r_squared)
    except Exception as e:
        print(f"回归分析执行失败: {e}")
        # ---【修改】返回 NaN ---
        return np.nan, np.nan, np.nan, np.nan

# --- 【修改】函数签名添加 benchmark_loaded 参数 ---
def display_metrics(metrics: Dict, benchmark_loaded: bool = False):
    """
    格式化并打印性能指标字典。

    Args:
        metrics (Dict): 包含性能指标的字典。
        benchmark_loaded (bool): 指示基准数据是否成功加载，用于决定是否显示相对指标。
    """
    print("\n" + "="*25 + " 回测性能指标 " + "="*25)
    print(f"回测期间: {metrics.get('开始日期', 'N/A')} 到 {metrics.get('结束日期', 'N/A')}")
    print("-" * 65)

    # ---【修改】将指标分为绝对指标和相对指标 ---
    ABSOLUTE_METRICS_ORDER = [
        ('总回报率', '.2%'), ('年化回报率', '.2%'), ('年化波动率', '.2%'),
        ('夏普比率', '.2f'), ('索提诺比率', '.2f'), ('卡玛比率', '.2f'),
        ('最大回撤', '.2%'), ('最大回撤持续天数', 's'), ('最长水下持续天数', 's'),
        ('月度胜率', '.2%'), ('月度盈亏比', '.2f'), ('年化换手率', '.2%')
    ]

    RELATIVE_METRICS_ORDER = [
        ('Beta', '.2f'), ('年化 Alpha', '.2%'), ('Alpha P-value', '.4f'),
        ('R-squared', '.2%'), ('信息比率', '.2f'), ('年化跟踪误差', '.2%'),
        ('年化超额回报', '.2%'), ('总超额回报', '.2%')
    ]

    # --- 打印绝对指标 ---
    for display_name, fmt in ABSOLUTE_METRICS_ORDER:
        if display_name in metrics:
            value = metrics[display_name]
            # (格式化逻辑不变)
            if pd.isna(value): value_str = "N/A"
            elif fmt == 's': value_str = str(value)
            elif isinstance(value, (int, float)):
                try: value_str = format(value, fmt)
                except (TypeError, ValueError): value_str = f"{value:.4f}"
            else: value_str = str(value)
            print(f"{display_name:<25}: {value_str}")

    # --- 根据 benchmark_loaded 决定是否打印相对指标 ---
    if benchmark_loaded:
        print("-" * 65) # 添加分隔线
        for display_name, fmt in RELATIVE_METRICS_ORDER:
            if display_name in metrics:
                value = metrics[display_name]
                # (格式化逻辑不变)
                if pd.isna(value): value_str = "N/A"
                elif fmt == 's': value_str = str(value)
                elif isinstance(value, (int, float)):
                    try: value_str = format(value, fmt)
                    except (TypeError, ValueError): value_str = f"{value:.4f}"
                else: value_str = str(value)

                # 为 p-value 添加显著性标记
                if display_name == 'Alpha P-value' and isinstance(value, (float, int)) and not pd.isna(value):
                     if value < 0.01: value_str += " (***)"
                     elif value < 0.05: value_str += " (**)"
                     elif value < 0.10: value_str += " (*)"
                print(f"{display_name:<25}: {value_str}")
    else:
         print("-" * 65)
         print("相对指标 (Alpha, Beta 等): N/A (未加载或处理基准数据)")


    print("-" * 65)

