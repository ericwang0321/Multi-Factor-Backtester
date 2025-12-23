# quant_core/visualization/market.py
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from .theme import VisualTheme

class MarketCharts:
    """
    市场行情相关图表 (对应 Data Explorer)
    """
    
    @staticmethod
    def plot_price_history(df: pd.DataFrame, symbol: str) -> go.Figure:
        """绘制历史价格走勢"""
        fig = px.line(df, x='datetime', y='close', title=f"{symbol} Historical Price")
        # 使用统一主题覆盖默认样式
        fig.update_traces(line_color=VisualTheme.COLOR_STRATEGY)
        VisualTheme.apply_layout(fig, title=f"{symbol} Price History", y_title="Price")
        return fig

    @staticmethod
    def plot_volume(df: pd.DataFrame) -> go.Figure:
        """绘制成交量图"""
        # 简单着色：涨红跌绿 (这里简化为单一颜色或根据 close 变化)
        # 如果 df 有 open/close，可以计算颜色，这里为了通用简单处理
        fig = px.bar(df, x='datetime', y='volume')
        fig.update_traces(marker_color=VisualTheme.COLOR_BENCHMARK)
        VisualTheme.apply_layout(fig, title="Trading Volume", y_title="Volume")
        return fig