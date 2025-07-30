import torch
import torch.nn as nn
import psutil
import os
import time
from collections import defaultdict
import numpy as np


class LayerProfiler:
    def __init__(self):
        self.hooks = []
        self.layer_metrics = defaultdict(dict)
        self.input_shapes = {}
        self.start_times = {}
        self.process = psutil.Process(os.getpid())
        self.model = None

    def _register_hooks(self, model):
        """Register hooks for all computational layers"""
        print("[DEBUG] Registering hooks...")
        layer_types = (nn.Conv2d, nn.Linear, nn.MaxPool2d,
                       nn.AvgPool2d, nn.BatchNorm2d, nn.ReLU,
                       nn.AdaptiveAvgPool2d)

        for name, layer in model.named_modules():
            if isinstance(layer, layer_types):
                print(f"[HOOK] Registered: {name} ({layer.__class__.__name__})")

                def memory_hook(module, input, output, layer_name=name):
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    output_memory = output.nelement() * output.element_size() / 1024 / 1024
                    input_memory = sum(
                        [i.nelement() * i.element_size() for i in input if torch.is_tensor(i)]) / 1024 / 1024
                    memory_increment = output_memory + input_memory * 0.2

                    self.layer_metrics[layer_name]['memory'] = memory_increment
                    self.input_shapes[layer_name] = input[0].shape
                    print(f"[MEMORY] {layer_name}: {memory_increment:.2f}MB")

                def timing_pre_hook(module, input, layer_name=name):
                    self.start_times[layer_name] = time.time()

                def timing_hook(module, input, output, layer_name=name):
                    elapsed = (time.time() - self.start_times.get(layer_name, time.time())) * 1000
                    self.layer_metrics[layer_name]['latency'] = elapsed
                    print(f"[TIMING] {layer_name}: {elapsed:.2f}ms")

                self.hooks.append(layer.register_forward_pre_hook(timing_pre_hook))
                self.hooks.append(layer.register_forward_hook(memory_hook))
                self.hooks.append(layer.register_forward_hook(timing_hook))
        print("[DEBUG] All hooks registered")

    def calculate_flops(self):
        """Calculate FLOPs for all layers"""
        print("[DEBUG] Calculating FLOPs...")
        for layer_name, metrics in self.layer_metrics.items():
            if layer_name in self.input_shapes:
                module = self._get_module_by_name(layer_name)
                if module is None:
                    continue

                try:
                    if isinstance(module, nn.Conv2d):
                        metrics['flops'] = self._calculate_conv_flops(module, self.input_shapes[layer_name])
                    elif isinstance(module, nn.Linear):
                        metrics['flops'] = self._calculate_linear_flops(module, self.input_shapes[layer_name])
                    print(f"[FLOPS] {layer_name}: {metrics['flops']:.2e}")
                except Exception as e:
                    print(f"[ERROR] FLOPs calculation failed for {layer_name}: {e}")
            else:
                print(f"[SKIP] No input shape for {layer_name}")
        print("[DEBUG] FLOPs calculation completed")

    def _get_module_by_name(self, layer_name):
        """Get module by name with debug info"""
        names = layer_name.split('.')
        module = self.model
        for name in names:
            try:
                if name.isdigit():
                    module = module[int(name)]
                else:
                    module = getattr(module, name)
            except (AttributeError, IndexError, TypeError) as e:
                print(f"[ERROR] Failed to resolve {layer_name} at {name}: {e}")
                return None
        return module

    @staticmethod
    def _calculate_conv_flops(conv_module, input_shape):
        batch_size, in_channels, height, width = input_shape
        out_channels = conv_module.out_channels
        kernel_size = conv_module.kernel_size[0] if isinstance(conv_module.kernel_size,
                                                               tuple) else conv_module.kernel_size
        stride = conv_module.stride[0] if isinstance(conv_module.stride, tuple) else conv_module.stride
        padding = conv_module.padding[0] if isinstance(conv_module.padding, tuple) else conv_module.padding
        groups = conv_module.groups

        out_height = (height + 2 * padding - kernel_size) // stride + 1
        out_width = (width + 2 * padding - kernel_size) // stride + 1

        if groups == 1:
            flops = batch_size * out_channels * out_height * out_width * in_channels * kernel_size * kernel_size
        else:
            flops = batch_size * out_channels * out_height * out_width * (
                    in_channels // groups) * kernel_size * kernel_size * groups

        return flops

    @staticmethod
    def _calculate_linear_flops(linear_module, input_shape):
        batch_size = input_shape[0]
        return batch_size * linear_module.in_features * linear_module.out_features

    def profile_model(self, model, input_tensor):
        """Profile model performance"""
        self.model = model
        self._register_hooks(model)

        with torch.no_grad():
            output = model(input_tensor)

        for name, module in model.named_modules():
            if hasattr(module, 'output_shape'):
                self.layer_metrics[name]['output_shape'] = module.output_shape.shape

        self.calculate_flops()

        for hook in self.hooks:
            hook.remove()

        return self.layer_metrics


def find_bottlenecks_with_adaptive_k(values, names, metric_name, k_init=2.0, k_min=0.1, step=0.1):
    """Improved adaptive k finding with dynamic initialization"""
    mean = np.mean(values)
    std = np.std(values)

    # Dynamic k initialization based on data distribution
    if std < 0.1 * mean:
        k_init = min(k_init, 1.0)

    k = k_init
    while k >= k_min:
        bottlenecks = [(name, val) for name, val in zip(names, values)
                       if val > mean + k * std]
        if bottlenecks:
            print(f"[Adaptive k] Found {len(bottlenecks)} {metric_name} bottlenecks with k={k:.2f}")
            return bottlenecks, k
        k -= step

    print(f"[Adaptive k] No {metric_name} bottlenecks found (min k={k_min:.2f})")
    return [], k_init


def analyze_bottlenecks(layer_metrics, model_name, k=1.5):
    """Analyze and print bottleneck information with adaptive k"""
    if not layer_metrics:
        print("[WARNING] No layer metrics available")
        return

    names = list(layer_metrics.keys())
    memory_values = [metrics.get('memory', 0) for metrics in layer_metrics.values()]
    flops_values = [metrics.get('flops', 0) for metrics in layer_metrics.values()]
    latency_values = [metrics.get('latency', 0) for metrics in layer_metrics.values()]

    print(f"\n===== {model_name} Bottleneck Analysis =====")

    # Memory bottlenecks
    memory_bottlenecks, km = find_bottlenecks_with_adaptive_k(
        memory_values, names, 'memory', k_init=k)
    if memory_bottlenecks:
        print("\nMemory Bottlenecks:")
        for layer, memory in sorted(memory_bottlenecks, key=lambda x: -x[1]):
            print(f"  {layer}: {memory:.2f} MB (k={km:.2f})")
    else:
        print("\nNo significant memory bottlenecks found")

    # FLOPs bottlenecks
    flops_bottlenecks, kf = find_bottlenecks_with_adaptive_k(
        flops_values, names, 'FLOPs', k_init=k)
    if flops_bottlenecks:
        print("\nFLOPs Bottlenecks:")
        for layer, flops in sorted(flops_bottlenecks, key=lambda x: -x[1]):
            print(f"  {layer}: {flops:.2e} (k={kf:.2f})")
    else:
        print("\nNo significant FLOPs bottlenecks found")

    # Latency bottlenecks
    latency_bottlenecks, kl = find_bottlenecks_with_adaptive_k(
        latency_values, names, 'latency', k_init=k)
    if latency_bottlenecks:
        print("\nLatency Bottlenecks:")
        for layer, latency in sorted(latency_bottlenecks, key=lambda x: -x[1]):
            print(f"  {layer}: {latency:.2f} ms (k={kl:.2f})")
    else:
        print("\nNo significant latency bottlenecks found")