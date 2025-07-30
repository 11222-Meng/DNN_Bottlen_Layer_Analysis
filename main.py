import torch
import torch.nn as nn
from models.pretrained import load_pretrained_model
from tasks.data_loader import load_cifar10
from utils.memory_monitor import monitor_memory_usage
from utils.bottleneck_analyzer import LayerProfiler, analyze_bottlenecks
from utils.dag_generator import DAGGenerator, SlicePointIdentifier
import threading
import psutil
import os
import time


def test_model_performance(model_name, model, input_tensor):
    """Test model performance and generate DAGs"""
    print(f"\n=== Profiling {model_name} ===")

    # Profile model
    profiler = LayerProfiler()
    layer_metrics = profiler.profile_model(model, input_tensor)
    analyze_bottlenecks(layer_metrics, model_name, k=1.5)

    # Generate DAGs for all metrics
    dag_generator = DAGGenerator(expansion_N=2)  # Can be adjusted to 7
    dags = {}

    for metric in ['memory', 'flops', 'latency']:
        print(f"\nGenerating {metric} DAGs...")
        dags[metric] = dag_generator.generate_submodel_dags(
            model, model_name, layer_metrics, metric)

        # Print DAG info
        if dags[metric]:
            print(f"Generated {len(dags[metric])} {metric} DAGs")
            for name, dag in dags[metric].items():
                print(f"  {name}: {len(dag.nodes)} operators")
        else:
            print(f"No {metric} DAGs generated")

    return layer_metrics, dags


def main():
    # Load data
    train_loader = load_cifar10(batch_size=16)
    data_iter = iter(train_loader)
    images, _ = next(data_iter)

    # Models to test
    models_to_test = ['resnet101', 'vgg19', 'mobilenet_v2', 'squeezenet']

    for model_name in models_to_test:
        print(f"\n===== Testing {model_name} =====")
        model = load_pretrained_model(model_name, pretrained=True)

        # Test performance and generate DAGs
        metrics, dags = test_model_performance(model_name, model, images)

        # User interaction for optimization
        print("\nAvailable optimization metrics:")
        for i, metric in enumerate(['memory', 'flops', 'latency'], 1):
            print(f"{i}. {metric}")

        choice = int(input("Select metric to optimize (1-3): "))
        selected_metric = ['memory', 'flops', 'latency'][choice - 1]

        # Slice point identification (commented out as in original)
        # slice_identifier = SlicePointIdentifier(
        #     threshold=0.3,
        #     cost_ratio_max=2.0
        # )
        # print(f"\nIdentifying slice points for {selected_metric} bottlenecks:")
        # for layer_name, dag in dags[selected_metric].items():
        #     candidates = slice_identifier.identify_slice_points(dag, selected_metric)
        #     ... (rest of the original code)


if __name__ == "__main__":
    main()