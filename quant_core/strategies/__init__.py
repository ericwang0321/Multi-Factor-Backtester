# quant_core/strategies/__init__.py
# -*- coding: utf-8 -*-

# 1. 暴露核心工厂函数和注册表 (给 run_backtest.py 用)
from .base import create_strategy_instance, STRATEGY_REGISTRY

# 2. 导入策略模块 (为了触发 @register_strategy 装饰器)
#    每当你新建一个策略文件，必须在这里 import 一下！
from . import rules

# 如果你有 ml_strategy.py，也要在这里 import
# from . import ml_strategy