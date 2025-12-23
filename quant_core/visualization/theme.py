# quant_core/visualization/theme.py
import plotly.graph_objects as go
import plotly.express as px  # <--- 修复了这里

class VisualTheme:
    """
    统一的视觉主题配置 (Industrial Style)
    """
    # 品牌/策略颜色系统
    COLOR_STRATEGY = '#0B3D59'   # 深蓝 (策略主线)
    COLOR_BENCHMARK = '#5EA9CE'  # 浅蓝 (基准虚线)
    COLOR_EXCESS = '#8E44AD'     # 紫色 (超额收益)
    
    COLOR_UP = '#27AE60'         # 上涨/正值 (绿)
    COLOR_DOWN = '#C0392B'       # 下跌/负值 (红)
    
    # 风险/回撤颜色
    COLOR_RISK_FILL = 'rgba(255, 0, 0, 0.6)'
    COLOR_EXCESS_FILL = 'rgba(142, 68, 173, 0.2)'
    
    # 配色盘 (用于多因子相关性等)
    PALETTE = px.colors.diverging.RdBu_r

    @staticmethod
    def apply_layout(fig: go.Figure, title: str = "", x_title: str = "", y_title: str = "", legend_top: bool = True):
        """
        应用统一的 Plotly Layout 风格
        """
        layout_args = dict(
            title=dict(text=title, font=dict(size=18)),
            xaxis_title=x_title,
            yaxis_title=y_title,
            template="plotly_white", # 使用简洁白底
            hovermode="x unified",   # 统一的十字光标提示
            margin=dict(l=40, r=40, t=60, b=40),
            height=500, # 默认高度
        )
        
        if legend_top:
            layout_args['legend'] = dict(
                orientation="h",     # 水平图例
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
            
        fig.update_layout(**layout_args)
        return fig