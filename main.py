import torch
import warnings

from models.model_tester import test_model_performance
from models.pretrained import load_pretrained_model
from tasks.data_loader import load_cifar10

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    """主程序入口"""
    # 加载 CIFAR-10 数据集
    train_loader = load_cifar10(batch_size=16)
    data_iter = iter(train_loader)
    images, _ = next(data_iter)

    # 测试预训练模型
    pretrained_models = ['resnet101', 'vgg19', 'mobilenet_v2']
    for model_name in pretrained_models:
        print(f"\n{'='*50}")
        print(f"Testing model: {model_name}")
        model = load_pretrained_model(model_name, pretrained=True)
        model.eval()
        test_model_performance(model_name, model, images)

if __name__ == "__main__":
    main()