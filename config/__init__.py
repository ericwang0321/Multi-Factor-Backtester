# config/__init__.py
import yaml
import os
from typing import Dict, Any

def deep_update(base_dict, update_dict):
    """递归合并字典，防止子层级被直接覆盖"""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def load_config(mode: str = 'live') -> Dict[str, Any]:
    """
    配置加载工厂
    作用：合并 base.yaml + {mode}.yaml + secrets.yaml
    """
    config_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 加载基础策略逻辑 (base.yaml)
    base_path = os.path.join(config_dir, 'base.yaml')
    with open(base_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
        
    # 2. 加载环境覆盖配置 (live.yaml 或 backtest.yaml)
    env_path = os.path.join(config_dir, f'{mode}.yaml')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            env_config = yaml.safe_load(f) or {}
            config = deep_update(config, env_config)

    # 3. 加载隐私密钥 (secrets.yaml)
    secrets_path = os.path.join(config_dir, 'secrets.yaml')
    if os.path.exists(secrets_path):
        with open(secrets_path, 'r', encoding='utf-8') as f:
            secrets = yaml.safe_load(f) or {}
            config = deep_update(config, secrets)
            
    return config