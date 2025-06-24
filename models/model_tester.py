import torch
import threading
import psutil
import os
import numpy as np
import traceback
from collections import defaultdict
from utils.memory_monitor import monitor_memory_usage
from utils.bottleneck_analyzer import LayerProfiler, analyze_bottlenecks
from optimizations.fusion_optimizer import FusionOptimizer


def test_model_performance(model_name, model, input_tensor, recursion_level=0, max_recursion=2):
    """
    改进的模型性能测试函数，包含：
    - 精确的FLOPs检测（科学计数法）
    - 递归深度控制
    - 全面的错误处理
    - 多维度瓶颈分析
    """
    try:
        # ==================== 初始化检查 ====================
        if recursion_level >= max_recursion:
            print(f"⚠️ 达到最大优化递归深度 {max_recursion}，停止进一步优化")
            return

        # ==================== 内存监控 ====================
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        print(f"\n{'=' * 60}")
        print(f"Testing {model_name} (Recursion level: {recursion_level})")
        print(f"Initial memory: {initial_memory:.2f} MB")

        stop_event = threading.Event()
        memory_thread = threading.Thread(
            target=lambda: monitor_memory_usage(stop_event=stop_event),
            daemon=True
        )
        memory_thread.start()

        # ==================== 性能分析 ====================
        profiler = LayerProfiler()
        try:
            with torch.no_grad():
                # 同时获取层指标和FLOPs
                output = model(input_tensor)
                layer_metrics = profiler.profile_model(model, input_tensor)
        except Exception as e:
            print(f"❌ 性能分析失败: {str(e)}")
            layer_metrics = {}
            traceback.print_exc()

        # ==================== 结果收集 ====================
        stop_event.set()
        memory_thread.join(timeout=5)

        final_memory = process.memory_info().rss / 1024 / 1024
        print(f"\nMemory usage:")
        print(f"- Initial: {initial_memory:.2f} MB")
        print(f"- Final: {final_memory:.2f} MB")
        print(f"- Delta: {final_memory - initial_memory:.2f} MB")

        # ==================== 瓶颈分析 ====================
        if layer_metrics:
            print("\nLayer metrics:")
            for layer, metrics in layer_metrics.items():
                print(f"\n{layer}:")
                print(f"- Memory: {metrics.get('memory', 0):.2f} MB")
                print(f"- FLOPs: {metrics.get('flops', 0):.2e}")
                print(f"- Latency: {metrics.get('latency', 0):.2f} ms")

            # 改进的瓶颈检测（包含FLOPs科学计数法处理）
            bottlenecks = analyze_bottlenecks(layer_metrics, model_name)
        else:
            print("⚠️ 无有效的层指标数据")
            return

        # ==================== 优化阶段 ====================
        if not any(bottlenecks.values()):
            print("\n✅ 未检测到显著瓶颈层")
            return

        # 用户交互优化
        try:
            if input("\n是否进行优化? (y/n): ").lower() != 'y':
                return

            print("\n选择优化目标:")
            print("1. 内存瓶颈")
            print("2. 延迟瓶颈")
            print("3. 计算量(FLOPs)瓶颈")
            print("4. 综合优化")
            choice = input("输入选择 (1-4): ").strip()

            # 根据选择筛选目标层
            target_layers = []
            if choice in ['1', '4'] and bottlenecks['memory']:
                target_layers.extend([x[0] for x in bottlenecks['memory']])
            if choice in ['2', '4'] and bottlenecks['latency']:
                target_layers.extend([x[0] for x in bottlenecks['latency']])
            if choice in ['3', '4'] and bottlenecks['flops']:
                target_layers.extend([x[0] for x in bottlenecks['flops']])

            if not target_layers:
                print("⚠️ 没有符合选择条件的瓶颈层")
                return

            # 显示选择的瓶颈层
            print("\n选中的瓶颈层:")
            for layer in set(target_layers):  # 去重
                metrics = layer_metrics[layer]
                info = []
                if layer in [x[0] for x in bottlenecks['memory']]:
                    info.append(f"内存: {metrics['memory']:.2f} MB")
                if layer in [x[0] for x in bottlenecks['latency']]:
                    info.append(f"延迟: {metrics['latency']:.2f} ms")
                if layer in [x[0] for x in bottlenecks['flops']]:
                    info.append(f"FLOPs: {metrics['flops']:.2e}")
                print(f"- {layer}: {', '.join(info)}")

            # 优化策略选择
            priority = input("\n优化优先级 [1] 速度 [2] 内存: ").strip()
            priority = 'speedup' if priority == '1' else 'memory'

            # 执行优化
            optimizer = FusionOptimizer(model, layer_metrics)
            optimizations = optimizer.optimize(target_layers, priority=priority)

            if optimizations:
                print("\n优化方案:")
                for opt in optimizations:
                    print(f"- 融合 {opt['layers'][0]} + {opt['layers'][1]} 为 {opt['combo']}")
                    print(f"  预估加速: {opt['speedup']:.2f}x")
                    print(f"  内存减少: {opt['memory_reduction']:.2f} MB")

                # 应用优化并测试新模型
                fused_model = optimizer.apply_optimizations(optimizations)
                print("\n测试优化后的模型...")
                test_model_performance(
                    f"{model_name}_optimized",
                    fused_model,
                    input_tensor,
                    recursion_level=recursion_level + 1
                )
            else:
                print("\n⚠️ 未找到有效的融合方案")
                print("可能原因:")
                print("- 层类型不匹配（如单独的卷积层）")
                print("- 特殊结构（如深度可分离卷积）")
                print("- 层连接不连续")

        except KeyboardInterrupt:
            print("\n🛑 用户中断优化流程")
        except Exception as e:
            print(f"\n❌ 优化过程出错: {str(e)}")
            traceback.print_exc()

    except Exception as e:
        print(f"\n🔥 测试过程中发生严重错误: {str(e)}")
        traceback.print_exc()