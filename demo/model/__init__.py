from typing import Dict, Type
from .base_model import BaseModel

# 模型注册表
_MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}

def register_model(name: str):
    """
    模型注册装饰器
    """
    def register_model_cls(cls):
        if name in _MODEL_REGISTRY:
            raise ValueError(f'模型名称 {name} 已被注册')
        if not issubclass(cls, BaseModel):
            raise ValueError(f'模型类 {cls.__name__} 必须继承 BaseModel')
        _MODEL_REGISTRY[name] = cls
        return cls
    return register_model_cls

def get_model(name: str) -> Type[BaseModel]:
    """
    根据名称获取模型类
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f'未找到名为 {name} 的模型，可用的模型有: {list(_MODEL_REGISTRY.keys())}')
    return _MODEL_REGISTRY[name]

# 导出 Model 类用于实例化
def Model(config):
    """
    根据配置创建模型实例
    """
    model_name = config.get('name')
    if not model_name:
        raise ValueError('配置中必须指定 "name" 字段')
    
    model_cls = get_model(model_name)
    return model_cls(config)

from .sub_model_Dn import SubModelDn  # 导入子模型
from .tcpdnet import TCPDNet  # 导入 TCPDNet 模型
from .sub_model_Dn_rgb2rgb import SubModelDnRgb2Rgb  # 导入子模型
from .pidndm_model import PIDNDM  # 导入 PIDNDM 模型
