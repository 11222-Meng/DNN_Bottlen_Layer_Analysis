import torchvision.models as models
from transformers import ViTForImageClassification


def load_pretrained_model(model_name, pretrained=True):
    """
    加载预训练模型。
    :param model_name: 模型名称（如 'resnet101', 'vgg19' 等）
    :param pretrained: 是否加载预训练权重
    :return: 加载的模型
    """
    if model_name == 'resnet18':
        return models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet50':
        return models.resnet50(pretrained=pretrained)
    elif model_name == 'resnet101':
        return models.resnet101(pretrained=pretrained)
    elif model_name == 'resnet152':
        return models.resnet152(pretrained=pretrained)

    elif model_name == 'vgg19':
        return models.vgg19(pretrained=pretrained)
    elif model_name == 'mobilenet_v2':
        return models.mobilenet_v2(pretrained=pretrained)
    elif model_name == 'squeezenet':
        return models.squeezenet1_0(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
