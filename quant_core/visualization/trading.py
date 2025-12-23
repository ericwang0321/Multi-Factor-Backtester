# quant_core/visualization/trading.py
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, List, Dict
from .theme import VisualTheme
from .base import PlotBase

class TradingCharts(PlotBase):
    """
    交易视角相关图表 (对应 Trading/Signals 分析)
    包含 K线、技术指标叠加、买卖点标记。
    """
    
    @staticmethod
    def plot_candlestick(df: pd.DataFrame, 
                         symbol: str = "Security", 
                         title: str = None) -> go.Figure:
        """
        绘制基础 K 线图
        Args:
            df: 包含 open, high, low, close 列的 DataFrame
            symbol: 标的代码
        """
        # 1. 数据校验 (使用基类功能)
        df = TradingCharts.ensure_datetime_index(df)
        TradingCharts.check_required_columns(df, ['open', 'high', 'low', 'close'])
        
        # 2. 绘制 K 线
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol,
            increasing_line_color=VisualTheme.COLOR_UP,
            decreasing_line_color=VisualTheme.COLOR_DOWN
        )])
        
        # 3. 应用主题
        final_title = title if title else f"{symbol} Price Action"
        VisualTheme.apply_layout(fig, title=final_title, y_title="Price")
        
        # 移除底部的 Range Slider (Plotly 默认带这个，通常比较占地)
        fig.update_layout(xaxis_rangeslider_visible=False)
        
        return fig

    @staticmethod
    def add_bollinger_bands(fig: go.Figure, 
                            df: pd.DataFrame, 
                            col_upper: str = 'upper', 
                            col_lower: str = 'lower', 
                            col_mid: str = 'mid') -> go.Figure:
        """
        在现有图表上叠加布林带
        """
        # 检查列是否存在
        if not {col_upper, col_lower}.issubset(df.columns):
            return fig
            
        # 1. 绘制下轨 (无填充)
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col_lower],
            name='BB Lower',
            line=dict(color='rgba(100, 100, 100, 0.3)', width=1),
            showlegend=False
        ))
        
        # 2. 绘制上轨 (填充到下轨)
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col_upper],
            name='Bollinger Bands',
            line=dict(color='rgba(100, 100, 100, 0.3)', width=1),
            fill='tonexty', # 核心：填充到前一个 Trace (即下轨)
            fillcolor='rgba(100, 100, 100, 0.05)', # 极淡的灰色背景
            mode='lines'
        ))
        
        # 3. 绘制中轨 (可选)
        if col_mid in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_mid],
                name='BB Mid',
                line=dict(color='rgba(100, 100, 100, 0.5)', width=1, dash='dot')
            ))
            
        return fig

    @staticmethod
    def add_trade_markers(fig: go.Figure, 
                          trade_log: pd.DataFrame) -> go.Figure:
        """
        在 K 线图上标记买卖点
        Args:
            trade_log: 包含 'datetime', 'action'(buy/sell), 'price' 的 DataFrame
        """
        if trade_log.empty:
            return fig
            
        # 筛选买入和卖出记录
        buys = trade_log[trade_log['action'] == 'buy']
        sells = trade_log[trade_log['action'] == 'sell']
        
        # 1. 标记买入 (绿色向上三角)
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys['datetime'], y=buys['price'],
                mode='markers',
                name='Buy Signal',
                marker=dict(symbol='triangle-up', size=10, color=VisualTheme.COLOR_UP),
                hovertemplate="Buy: %{y:.2f}<br>Date: %{x}"
            ))
            
        # 2. 标记卖出 (红色向下三角)
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells['datetime'], y=sells['price'],
                mode='markers',
                name='Sell Signal',
                marker=dict(symbol='triangle-down', size=10, color=VisualTheme.COLOR_DOWN),
                hovertemplate="Sell: %{y:.2f}<br>Date: %{x}"
            ))
            
        return fig

    @staticmethod
    def plot_strategy_view(df_price: pd.DataFrame, 
                           df_signals: pd.DataFrame = None, 
                           df_bands: pd.DataFrame = None) -> go.Figure:
        """
        [工厂方法] 一键生成综合交易视图 (K线 + 信号 + 指标)
        """
        # 1. 基础 K 线
        fig = TradingCharts.plot_candlestick(df_price)
        
        # 2. 叠加布林带 (如果有)
        if df_bands is not None and not df_bands.empty:
            # 假设 df_bands 索引与 df_price 对齐
            fig = TradingCharts.add_bollinger_bands(fig, df_bands)
            
        # 3. 叠加交易信号 (如果有)
        if df_signals is not None and not df_signals.empty:
            fig = TradingCharts.add_trade_markers(fig, df_signals)
            
        return fig