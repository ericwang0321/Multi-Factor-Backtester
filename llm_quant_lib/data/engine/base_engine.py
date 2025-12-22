# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional

class BaseEngine(ABC):
    """
    数据引擎抽象基类，定义统一的数据拉取接口。
    """
    @abstractmethod
    def connect(self):
        """建立连接"""
        pass

    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, duration: str, bar_size: str) -> pd.DataFrame:
        """获取 OHLCV 原始行情数据"""
        pass

    @abstractmethod
    def fetch_fundamentals(self, symbol: str) -> dict:
        """获取基本面数据（用于计算市值和股本）"""
        pass