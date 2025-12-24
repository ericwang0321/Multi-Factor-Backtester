# quant_core/strategies/__init__.py
from .rules import LinearWeightedStrategy
from .ml_strategy import XGBoostStrategy 

# 策略注册表：建立“字符串”与“类”的映射
STRATEGY_REGISTRY = {
    'linear': LinearWeightedStrategy,
    'ml': XGBoostStrategy
}

def get_strategy_instance(strat_config: dict):
    """
    策略生产工厂
    """
    strat_type = strat_config.get('type')
    common_cfg = strat_config.get('common', {})
    
    if strat_type not in STRATEGY_REGISTRY:
        raise ValueError(f"❌ 不支持的策略类型: {strat_type}")
    
    strat_class = STRATEGY_REGISTRY[strat_type]
    
    # 提取所有策略共有的初始化参数
    init_params = {
        'name': common_cfg.get('name', 'Live_Strategy'),
        'top_k': common_cfg.get('top_k', 3),
        'stop_loss_pct': common_cfg.get('risk', {}).get('stop_loss_pct'),
        'max_pos_weight': common_cfg.get('risk', {}).get('max_pos_weight'),
        'max_drawdown_pct': common_cfg.get('risk', {}).get('max_drawdown_pct'),
    }
    
    # 根据类型注入特有参数
    if strat_type == 'linear':
        init_params['weights'] = strat_config.get('linear_params', {}).get('weights', {})
    elif strat_type == 'ml':
        init_params['model_path'] = strat_config.get('ml_params', {}).get('model_path')
        init_params['feature_list'] = strat_config.get('ml_params', {}).get('feature_list', [])
        
    return strat_class(**init_params)