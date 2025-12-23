import pandas as pd
import numpy as np
import plotly.graph_objects as go
from .theme import VisualTheme

class PerformanceCharts:
    """
    策略绩效相关图表 (对应 Strategy Explorer 结果展示)
    """
    
    @staticmethod
    def plot_equity_curve(strategy_nav: pd.Series, benchmark_nav: pd.Series, 
                          benchmark_name: str = "Benchmark") -> go.Figure:
        """
        绘制 策略 vs 基准 净值曲线 + 超额收益面积图 (双轴)
        (完全恢复你提供的原始逻辑)
        """
        fig = go.Figure()

        # 1. 策略曲线
        # 归一化处理
        s_norm = strategy_nav / strategy_nav.iloc[0]
        fig.add_trace(go.Scatter(
            x=s_norm.index, y=s_norm, 
            name='Strategy', 
            line=dict(color=VisualTheme.COLOR_STRATEGY, width=2.5)
        ))
        
        # 2. 基准曲线
        b_norm = benchmark_nav / benchmark_nav.iloc[0]
        fig.add_trace(go.Scatter(
            x=b_norm.index, y=b_norm, 
            name=benchmark_name, 
            line=dict(color=VisualTheme.COLOR_BENCHMARK, width=2, dash='dot')
        ))
        
        # 3. 超额收益 (右轴 + 阴影)
        excess = s_norm - b_norm
        fig.add_trace(go.Scatter(
            x=excess.index, y=excess, 
            name='Excess Return', 
            yaxis='y2', 
            fill='tozeroy', 
            line=dict(color=VisualTheme.COLOR_EXCESS, width=1.5), 
            fillcolor=VisualTheme.COLOR_EXCESS_FILL
        ))
        
        # 布局 (使用 VisualTheme 默认布局，不添加额外后缀)
        VisualTheme.apply_layout(fig, title=f"Strategy vs {benchmark_name}", y_title="Normalized Value")
        
        # 配置双轴 (完全恢复原始配置，去掉我之前添加的 legend 位置参数)
        fig.update_layout(
            yaxis2=dict(
                title=dict(text="Cumulative Excess Return", font=dict(color=VisualTheme.COLOR_EXCESS)),
                tickfont=dict(color=VisualTheme.COLOR_EXCESS),
                overlaying="y",
                side="right",
                showgrid=False
            )
        )
        return fig

    @staticmethod
    def plot_drawdown_underwater(strategy_nav: pd.Series) -> go.Figure:
        """
        绘制水下回撤图 (补充缺失函数，防止 Dashboard 报错)
        """
        # 计算回撤
        running_max = strategy_nav.cummax()
        drawdown = (strategy_nav / running_max) - 1
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index, 
            y=drawdown, 
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='#DC3545', width=1.5),
            fillcolor='rgba(220, 53, 69, 0.2)'
        ))
        
        VisualTheme.apply_layout(fig, title="Drawdown (Underwater Plot)", y_title="Drawdown (%)")
        fig.update_yaxes(tickformat=".1%")
        return fig

    @staticmethod
    def plot_monthly_heatmap(strategy_returns: pd.Series) -> go.Figure:
        """
        绘制月度收益热力图 (补充缺失函数，防止 Dashboard 报错)
        """
        # 确保是 datetime 索引
        returns = strategy_returns.copy()
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)
            
        # 按月重采样并计算月度收益
        monthly_ret = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        
        # 构造透视表：Year vs Month
        monthly_ret_df = pd.DataFrame({
            'Year': monthly_ret.index.year,
            'Month': monthly_ret.index.month,
            'Return': monthly_ret.values
        })
        
        pivot_table = monthly_ret_df.pivot(index='Year', columns='Month', values='Return')
        
        # 补全缺失月份
        all_months = range(1, 13)
        for m in all_months:
            if m not in pivot_table.columns:
                pivot_table[m] = np.nan
        
        # 排序
        pivot_table = pivot_table.sort_index(ascending=False).sort_index(axis=1)
        
        # 绘图
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values * 100, 
            x=[pd.to_datetime(f'2000-{m}-01').strftime('%b') for m in pivot_table.columns],
            y=pivot_table.index,
            colorscale='RdYlGn',
            zmid=0,
            texttemplate="%{z:.2f}%",
            xgap=1, ygap=1
        ))
        
        VisualTheme.apply_layout(fig, title="Monthly Returns Heatmap", y_title="Year")
        fig.update_layout(xaxis_title=None)
        return fig

    @staticmethod
    def plot_rolling_var(rolling_var_series: pd.Series, confidence_level: float = 0.95) -> go.Figure:
        """
        绘制滚动 VaR (使用你提供的原始逻辑)
        """
        fig = go.Figure()
        
        # VaR 区域图
        fig.add_trace(go.Scatter(
            x=rolling_var_series.index, 
            y=rolling_var_series.values * 100, 
            fill='tozeroy', 
            name=f'{confidence_level:.0%} Rolling VaR', 
            line=dict(color='red', width=1),
            fillcolor=VisualTheme.COLOR_RISK_FILL
        ))
        
        VisualTheme.apply_layout(fig, title="Historical Rolling VaR (Value at Risk)", y_title="Potential Loss (%)")
        return fig