import torchvision.models as models
from transformers import ViTForImageClassification
from typing import Union
import torch.nn as nn

def load_pretrained(model_name: str, pretrained: bool = True) -> Union[nn.Module, None]:
    """加载预训练模型"""
    model_map = {
        'resnet18': models.resnet18,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'vgg19': models.vgg19,
        'mobilenet_v2': models.mobilenet_v2,
        'squeezenet': models.squeezenet1_0,
        'vit': ViTForImageClassification.from_pretrained
    }

    try:
        if model_name == 'vit':
            return model_map[model_name]('google/vit-base-patch16-224')
        return model_map[model_name](pretrained=pretrained)
    except KeyError:
        raise ValueError(f"不支持的模型: {model_name}")
    except Exception as e:
        print(f"模型加载错误: {str(e)}")
        return None