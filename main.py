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

# 修改后（更简洁的导入）
from utils import SlicePointIdentifier

def main():
    """主程序入口"""
    # 加载 CIFAR-10 数据集
    train_loader = load_cifar10(batch_size=16)
    data_iter = iter(train_loader)
    images, _ = next(data_iter)

    # 测试的模型列表
    models_to_test = ['resnet101', 'vgg19', 'mobilenet_v2', 'squeezenet']

    for model_name in models_to_test:
        print(f"\n===== Testing {model_name} =====")
        model = load_pretrained_model(model_name, pretrained=True)

        # 测试性能并生成DAG
        metrics, dags = test_model_performance(model_name, model, images)

        # 让用户选择优化指标
        print("\nAvailable optimization metrics:")
        for i, metric in enumerate(dags.keys(), 1):
            print(f"{i}. {metric}")

        choice = int(input("Select metric to optimize (1-3): ")) - 1
        selected_metric = list(dags.keys())[choice]

        # 初始化切片点识别器
        slice_identifier = SlicePointIdentifier(
            threshold=0.3,  # 参数名与类定义完全一致
            cost_ratio_max=2.0
        )

        # 对每个瓶颈层DAG识别切片点
        print(f"\nIdentifying slice points for {selected_metric} bottlenecks:")
        for layer_name, dag in dags[selected_metric].items():
            candidates = slice_identifier.identify_slice_points(dag, selected_metric)

            # 调试模式逻辑（原代码中的if-else分支）
            if not candidates:
                print("\n[DEBUG] No slice points found...")
                # 显示调试信息...
            else:
                slice_identifier.visualize_candidates(dag, candidates, selected_metric)
            for i, (u, v, props) in enumerate(candidates[:5], 1):  # 只显示前5个
                print(f"{i}. Edge: {u} -> {v}")
                print(f"   Feature size ratio: {props['feature_size']:.2f}")
                print(f"   Cut cost: {props['cut_cost']:.2f}")
                print(f"   Fused latency: {props['fused_latency']:.2f}")
                print(f"   Score: {props['score']:.2f}")

if __name__ == "__main__":
    main()