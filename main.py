import torch
import torch.nn as nn
from models.pretrained import load_pretrained_model
from tasks.data_loader import load_cifar10
from utils.memory_monitor import monitor_memory_usage
import threading
import psutil
import os
import time
from models.pretrained import load_pretrained_model
from tasks.data_loader import load_cifar10
from models.model_tester import test_model_performance

def main():
    """主程序入口"""
    # 加载 CIFAR-10 数据集
    train_loader = load_cifar10(batch_size=16)
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # 测试预训练模型
    pretrained_models = ['resnet101', 'vgg19', 'mobilenet_v2', 'squeezenet']
    for model_name in pretrained_models:
        print(f"\nLoading pretrained model: {model_name}")
        model = load_pretrained_model(model_name, pretrained=True)
        model.eval()
        test_model_performance(model_name, model, images)

if __name__ == "__main__":
    main()