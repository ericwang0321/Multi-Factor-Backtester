# quant_core/visualization/factor.py
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from .theme import VisualTheme

class FactorCharts:
    """
    因子分析相关图表 (对应 Analysis Explorer & Correlation)
    """
    
    @staticmethod
    def plot_ic_series(ic_data: pd.Series, ma_window: int = 20) -> go.Figure:
        """绘制 IC 时序柱状图及移动平均线"""
        fig = go.Figure()
        
        # IC 柱状图
        fig.add_trace(go.Bar(
            x=ic_data.index, y=ic_data.values,
            name="Daily IC",
            marker_color=[VisualTheme.COLOR_UP if x > 0 else VisualTheme.COLOR_DOWN for x in ic_data.values]
        ))
        
        # IC 均线
        if len(ic_data) > ma_window:
            ic_ma = ic_data.rolling(ma_window).mean()
            fig.add_trace(go.Scatter(
                x=ic_data.index, y=ic_ma.values,
                name=f"{ma_window}-Day IC MA",
                line=dict(color='black', width=2)
            ))
        
        # 辅助线
        fig.add_hline(y=0, line_dash="solid", line_color="gray")
        mean_ic = ic_data.mean()
        fig.add_hline(y=mean_ic, line_dash="dash", line_color="blue", 
                     annotation_text=f"Mean: {mean_ic:.3f}")

        VisualTheme.apply_layout(fig, title="Information Coefficient (IC) Series", y_title="IC Value")
        return fig

    @staticmethod
    def plot_quantile_layers(cumulative_returns_by_layer: pd.DataFrame) -> go.Figure:
        """绘制分层回测累积收益图"""
        fig = go.Figure()
        
        # 使用 Plotly 预设的 Viridis 色盘，适合展示层级顺序
        colors = px.colors.sequential.Viridis
        n_layers = len(cumulative_returns_by_layer.columns)
        
        for i, col in enumerate(cumulative_returns_by_layer.columns):
            # 动态计算颜色索引
            color_idx = int(i / max(1, n_layers - 1) * (len(colors) - 1))
            color = colors[color_idx]
            
            # 突出显示 Top 和 Bottom 层
            is_edge = "Top" in str(col) or "Bottom" in str(col) or str(col) == "0" or str(col) == str(n_layers-1)
            width = 3 if is_edge else 1
            
            fig.add_trace(go.Scatter(
                x=cumulative_returns_by_layer.index, 
                y=cumulative_returns_by_layer[col],
                mode='lines',
                name=str(col),
                line=dict(color=color, width=width)
            ))

        VisualTheme.apply_layout(fig, title="Quantile Layered Backtest (Fixed Wealth)", y_title="Cumulative Net Value")
        return fig

    @staticmethod
    def plot_correlation_matrix(corr_matrix: pd.DataFrame) -> go.Figure:
        """绘制因子相关性热力图"""
        fig = px.imshow(
            corr_matrix, 
            text_auto=".2f", 
            color_continuous_scale=VisualTheme.PALETTE, 
            zmin=-1, zmax=1
        )
        VisualTheme.apply_layout(fig, title="Dynamic Factor Correlation Matrix")
        return fig