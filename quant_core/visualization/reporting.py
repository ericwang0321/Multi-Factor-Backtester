# quant_core/visualization/reporting.py
import pandas as pd
import numpy as np
import plotly.io as pio
import os
from datetime import datetime
from io import BytesIO
from typing import Dict, Any, Optional

# 引用同级目录下的可视化组件
from .performance import PerformanceCharts
from .theme import VisualTheme

class ReportGenerator:
    """
    报告生成器 (Reporting Engine)
    负责将回测结果打包成 Excel 或 HTML 静态报告。
    """
    
    def __init__(self, strategy_name: str = "Strategy"):
        self.strategy_name = strategy_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_excel_report(self, 
                              metrics: Dict[str, Any],
                              equity_df: pd.DataFrame,
                              positions_df: Optional[pd.DataFrame] = None,
                              trade_log: Optional[pd.DataFrame] = None) -> BytesIO:
        """
        生成多 Sheet 的 Excel 报告 (返回 BytesIO 对象，方便 Streamlit 下载或存盘)
        """
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # 1. Sheet: Performance Summary (指标摘要)
            # 过滤掉非数值型的 Series (例如 equity curve)
            scalar_metrics = {k: v for k, v in metrics.items() if not isinstance(v, (pd.Series, pd.DataFrame))}
            df_metrics = pd.DataFrame.from_dict(scalar_metrics, orient='index', columns=['Value'])
            df_metrics.to_excel(writer, sheet_name='Summary')
            
            # 2. Sheet: Equity Curve (每日净值)
            if not equity_df.empty:
                equity_df.to_excel(writer, sheet_name='Daily Equity')
            
            # 3. Sheet: Holdings (持仓历史)
            if positions_df is not None and not positions_df.empty:
                positions_df.to_excel(writer, sheet_name='Holdings', index=False)
                
            # 4. Sheet: Trade Log (交易记录)
            if trade_log is not None and not trade_log.empty:
                trade_log.to_excel(writer, sheet_name='Trade Log', index=False)
                
        output.seek(0)
        return output

    def generate_html_report(self, 
                             metrics: Dict[str, Any],
                             equity_curve: pd.Series,
                             benchmark_curve: Optional[pd.Series] = None,
                             drawdown_series: Optional[pd.Series] = None,
                             trade_log: Optional[pd.DataFrame] = None,
                             output_path: Optional[str] = None) -> str:
        """
        生成单文件 HTML 交互式报告 (包含嵌入的 Plotly 图表)
        """
        # 1. 准备图表对象
        fig_equity = PerformanceCharts.plot_equity_curve(
            equity_curve, benchmark_curve, benchmark_name="Benchmark"
        )
        # 转换为 HTML div string (不包含完整的 html 标签，只包含图表部分)
        plot_equity_div = pio.to_html(fig_equity, full_html=False, include_plotlyjs='cdn')
        
        plot_dd_div = ""
        if drawdown_series is not None:
            fig_dd = PerformanceCharts.plot_drawdown(drawdown_series)
            plot_dd_div = pio.to_html(fig_dd, full_html=False, include_plotlyjs=False)

        # 2. 准备指标表格 HTML
        scalar_metrics = {k: v for k, v in metrics.items() if not isinstance(v, (pd.Series, pd.DataFrame))}
        metrics_html = "<table class='metrics-table'><tr><th>Metric</th><th>Value</th></tr>"
        for k, v in scalar_metrics.items():
            # 格式化数值
            if isinstance(v, float):
                val_str = f"{v:.2%}" if "Ratio" not in k and "Beta" not in k and "Sharpe" not in k else f"{v:.4f}"
            else:
                val_str = str(v)
            metrics_html += f"<tr><td>{k}</td><td>{val_str}</td></tr>"
        metrics_html += "</table>"

        # 3. 准备交易记录 HTML (取前 50 条)
        trades_html = "<p>No trades executed.</p>"
        if trade_log is not None and not trade_log.empty:
            trades_html = trade_log.head(50).to_html(classes='metrics-table', index=False)
            if len(trade_log) > 50:
                trades_html += f"<p>... and {len(trade_log)-50} more trades.</p>"

        # 4. 组装完整 HTML
        # 使用简单的 CSS 美化
        html_content = f"""
        <html>
        <head>
            <title>{self.strategy_name} - Backtest Report</title>
            <style>
                body {{ font-family: -apple-system, sans-serif; background-color: #f4f4f9; color: #333; margin: 0; padding: 20px; }}
                .container {{ max_width: 1200px; margin: 0 auto; background: white; padding: 40px; box-shadow: 0 0 20px rgba(0,0,0,0.05); border-radius: 8px; }}
                h1, h2 {{ color: {VisualTheme.COLOR_STRATEGY}; }}
                .header {{ border-bottom: 2px solid #eee; padding-bottom: 20px; margin-bottom: 30px; }}
                .metrics-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 14px; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .metrics-table th {{ background-color: #f8f9fa; font-weight: 600; }}
                .chart-container {{ margin-bottom: 40px; border: 1px solid #eee; padding: 10px; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Backtest Report: {self.strategy_name}</h1>
                    <p>Generated at: {self.timestamp}</p>
                </div>
                
                <h2>1. Key Performance Metrics</h2>
                {metrics_html}
                
                <h2>2. Equity Curve</h2>
                <div class="chart-container">
                    {plot_equity_div}
                </div>
                
                <h2>3. Drawdown Analysis</h2>
                <div class="chart-container">
                    {plot_dd_div}
                </div>

                <h2>4. Recent Trades (Top 50)</h2>
                {trades_html}
            </div>
        </body>
        </html>
        """

        # 5. 保存或返回
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return output_path
        else:
            return html_content