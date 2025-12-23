# quant_core/visualization/base.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Union, Optional

class ValidationMixin:
    """
    可视化数据校验混入类 (Defensive Programming)
    用于在绘图前清洗数据，防止 'KeyError' 或空图表。
    """

    @staticmethod
    def ensure_datetime_index(data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """
        确保数据的索引是 DatetimeIndex，并且已排序。
        """
        data = data.copy()
        
        # 如果索引不是时间，尝试查找名为 'datetime' 或 'date' 的列
        if not isinstance(data.index, pd.DatetimeIndex):
            if isinstance(data, pd.DataFrame):
                for col in ['datetime', 'date', 'Date', 'Time']:
                    if col in data.columns:
                        data[col] = pd.to_datetime(data[col])
                        data = data.set_index(col)
                        break
        
        # 再次检查
        if not isinstance(data.index, pd.DatetimeIndex):
            # 尝试强转索引
            try:
                data.index = pd.to_datetime(data.index)
            except Exception:
                raise ValueError("Data Validation Error: Index must be DatetimeIndex or convertable to datetime.")
        
        return data.sort_index()

    @staticmethod
    def sanitize_series(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
        """
        清洗 Series 数据：处理 NaN, Inf, 确保数值类型。
        """
        if series.empty:
            return series
            
        s = series.replace([np.inf, -np.inf], np.nan)
        s = s.fillna(fill_value)
        return s

    @staticmethod
    def check_required_columns(df: pd.DataFrame, required_columns: List[str]):
        """
        检查 DataFrame 是否包含必要的列。
        """
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Visualization Error: Input DataFrame is missing columns: {missing}")

class PlotBase(ValidationMixin):
    """
    所有绘图类的基类。
    定义了通用的绘图接口规范。
    """
    
    @staticmethod
    def create_empty_figure(text: str = "No Data Available") -> go.Figure:
        """
        当数据为空时，返回一个带有提示文字的空白图表，而不是报错。
        """
        fig = go.Figure()
        fig.add_annotation(
            text=text,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            template="plotly_white"
        )
        return fig