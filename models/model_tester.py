import torch
import threading
import psutil
import os
from utils.memory_monitor import monitor_memory_usage
from utils.bottleneck_analyzer import LayerProfiler, analyze_bottlenecks

def test_model_performance(model_name, model, input_tensor):
    """测试模型的综合性能"""
    print(f"\nTesting performance for {model_name}...")

    # 记录初始内存
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    # 创建停止事件
    stop_event = threading.Event()

    # 启动内存监控线程
    monitor_thread = threading.Thread(
        target=monitor_memory_usage,
        kwargs={"stop_event": stop_event, "interval": 0.01}
    )
    monitor_thread.start()

    # 创建性能分析器
    profiler = LayerProfiler()

    # 运行模型并记录性能指标
    layer_metrics = profiler.profile_model(model, input_tensor)

    # 记录最终内存
    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"Final memory usage: {final_memory:.2f} MB")
    print(f"Memory delta: {final_memory - initial_memory:.2f} MB")

    # 停止内存监控
    stop_event.set()
    monitor_thread.join(timeout=10)

    # 打印每层性能指标
    print("\nPerformance metrics by convolutional layers:")
    for layer_name, metrics in layer_metrics.items():
        print(f"Layer: {layer_name}")
        print(f"  Memory: {metrics.get('memory', 0):.2f} MB")
        print(f"  FLOPs: {metrics.get('flops', 0):.2e}")
        print(f"  Latency: {metrics.get('latency', 0):.2f} ms")
        print("-" * 50)

    # 分析瓶颈层，传入 k=1.5
    analyze_bottlenecks(layer_metrics, model_name, k=1.5)