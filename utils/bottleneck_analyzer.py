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

    def _register_hooks(self, model):
        """注册钩子到所有卷积层"""
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d):
                # 为内存测量注册前向钩子
                def memory_hook(module, input, output, layer_name=name):
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    output_memory = output.nelement() * output.element_size() / 1024 / 1024
                    input_memory = sum(
                        [i.nelement() * i.element_size() for i in input if torch.is_tensor(i)]) / 1024 / 1024
                    memory_increment = output_memory + input_memory * 0.2

                    self.layer_metrics[layer_name]['memory'] = memory_increment
                    self.input_shapes[layer_name] = input[0].shape

                # 为延迟测量注册前向pre-hook
                def timing_pre_hook(module, input, layer_name=name):
                    self.start_times[layer_name] = time.time()

                # 为延迟测量注册前向hook
                def timing_hook(module, input, output, layer_name=name):
                    end_time = time.time()
                    start_time = self.start_times.get(layer_name, end_time)
                    elapsed = end_time - start_time
                    self.layer_metrics[layer_name]['latency'] = elapsed * 1000  # 转换为毫秒

                # 注册钩子
                self.hooks.append(layer.register_forward_pre_hook(timing_pre_hook))
                self.hooks.append(layer.register_forward_hook(memory_hook))
                self.hooks.append(layer.register_forward_hook(timing_hook))

    def calculate_flops(self):
        """计算每层的FLOPs"""
        for layer_name, metrics in self.layer_metrics.items():
            if layer_name in self.input_shapes:
                input_shape = self.input_shapes[layer_name]
                module = self._get_module_by_name(layer_name)
                if module is not None:
                    flops = self._calculate_conv_flops(module, input_shape)
                    metrics['flops'] = flops

    def _get_module_by_name(self, layer_name):
        """通过名称获取模块"""
        names = layer_name.split('.')
        module = self.model
        for name in names:
            module = getattr(module, name)
        return module

    @staticmethod
    def _calculate_conv_flops(conv_module, input_shape):
        """计算卷积层的FLOPs"""
        batch_size, in_channels, height, width = input_shape
        out_channels = conv_module.out_channels
        kernel_size = conv_module.kernel_size[0]
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

    def profile_model(self, model, input_tensor):
        """分析模型性能"""
        self.model = model
        self._register_hooks(model)

        with torch.no_grad():
            output = model(input_tensor)

        self.calculate_flops()

        for hook in self.hooks:
            hook.remove()

        return self.layer_metrics

def analyze_bottlenecks(layer_metrics, model_name, k=1.5):
    """分析并打印瓶颈层信息"""
    if not layer_metrics:
        return

    memory_values = []
    flops_values = []
    latency_values = []

    for metrics in layer_metrics.values():
        memory_values.append(metrics.get('memory', 0))
        flops_values.append(metrics.get('flops', 0))
        latency_values.append(metrics.get('latency', 0))

    memory_mean = np.mean(memory_values)
    memory_std = np.std(memory_values)
    flops_mean = np.mean(flops_values)
    flops_std = np.std(flops_values)
    latency_mean = np.mean(latency_values)
    latency_std = np.std(latency_values)

    memory_bottlenecks = []
    flops_bottlenecks = []
    latency_bottlenecks = []

    for layer_name, metrics in layer_metrics.items():
        if metrics.get('memory', 0) > memory_mean + k * memory_std:
            memory_bottlenecks.append((layer_name, metrics['memory']))
        if metrics.get('flops', 0) > flops_mean + k * flops_std:
            flops_bottlenecks.append((layer_name, metrics['flops']))
        if metrics.get('latency', 0) > latency_mean + k * latency_std:
            latency_bottlenecks.append((layer_name, metrics['latency']))

    print(f"\n{model_name} Performance Summary:")
    if memory_bottlenecks:
        print("Memory Bottleneck Layers:")
        for layer, memory in memory_bottlenecks:
            print(f"  {layer}, Memory: {memory:.2f} MB")
    else:
        print("No memory bottleneck layers found.")
    if flops_bottlenecks:
        print("FLOPs Bottleneck Layers:")
        for layer, flops in flops_bottlenecks:
            print(f"  {layer}, FLOPs: {flops:.2e}")
    else:
        print("No FLOPs bottleneck layers found.")
    if latency_bottlenecks:
        print("Latency Bottleneck Layers:")
        for layer, latency in latency_bottlenecks:
            print(f"  {layer}, Latency: {latency:.2f} ms")
    else:
        print("No latency bottleneck layers found.")