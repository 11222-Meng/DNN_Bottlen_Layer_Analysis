import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from scipy import stats


class LayerProfiler:
    def __init__(self):
        self.hooks = []
        self.layer_metrics = defaultdict(dict)
        self.input_shapes = {}

    def _calculate_conv_flops(self, conv_module, input_shape):
        """精确的卷积FLOPs计算（包含分组卷积和dilation支持）"""
        batch_size, in_channels, height, width = input_shape
        out_channels = conv_module.out_channels
        groups = conv_module.groups

        # 计算输出特征图尺寸
        def _output_size(input_size, padding, dilation, kernel_size, stride):
            return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        output_height = _output_size(
            height,
            conv_module.padding[0],
            conv_module.dilation[0],
            conv_module.kernel_size[0],
            conv_module.stride[0]
        )
        output_width = _output_size(
            width,
            conv_module.padding[1],
            conv_module.dilation[1],
            conv_module.kernel_size[1],
            conv_module.stride[1]
        )

        # 每个位置的计算量 (考虑分组)
        kernel_ops = conv_module.kernel_size[0] * conv_module.kernel_size[1] * (in_channels // groups)

        # 总FLOPs (乘加算两次)
        flops = batch_size * out_channels * output_height * output_width * kernel_ops * 2

        # 添加bias项
        if conv_module.bias is not None:
            flops += batch_size * out_channels * output_height * output_width

        return flops

    def _register_hooks(self, model):
        """注册FLOPs计算钩子"""
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                def forward_hook(m, input, output, layer_name=name):
                    self.input_shapes[layer_name] = input[0].shape
                    flops = self._calculate_conv_flops(m, input[0].shape)
                    self.layer_metrics[layer_name]['flops'] = flops

                self.hooks.append(module.register_forward_hook(forward_hook))

    def profile_model(self, model, input_tensor):
        """分析模型各层FLOPs"""
        self._register_hooks(model)
        with torch.no_grad():
            model(input_tensor)

        for hook in self.hooks:
            hook.remove()

        return self.layer_metrics


def analyze_bottlenecks(layer_metrics, model_name, k=1.5):
    """改进的瓶颈检测方法（科学计数法+对数尺度）"""
    # 提取各指标值
    flops_values = [m.get('flops', 1e-6) for m in layer_metrics.values()]
    memory_values = [m.get('memory', 0) for m in layer_metrics.values()]
    latency_values = [m.get('latency', 0) for m in layer_metrics.values()]

    # 转换FLOPs到对数尺度（避免大数值问题）
    log_flops = np.log10(np.maximum(flops_values, 1e-6))

    # 计算各指标的Z-score
    def calc_z_scores(values, log_scale=False):
        if log_scale:
            values = np.log10(np.maximum(values, 1e-6))
        mean = np.mean(values)
        std = np.std(values)
        return np.abs((values - mean) / std)

    flops_z = calc_z_scores(flops_values, log_scale=True)
    memory_z = calc_z_scores(memory_values)
    latency_z = calc_z_scores(latency_values)

    # 动态调整阈值（大模型更严格）
    total_flops = sum(flops_values)
    adaptive_k = k * (1 + np.log10(max(total_flops / 1e9, 1)))  # 每增加10GFLOPs，k增加0.1

    # 检测瓶颈层
    bottlenecks = defaultdict(list)
    for i, (layer_name, metrics) in enumerate(layer_metrics.items()):
        if flops_z[i] > adaptive_k:
            bottlenecks['flops'].append((layer_name, metrics['flops']))
        if memory_z[i] > adaptive_k:
            bottlenecks['memory'].append((layer_name, metrics['memory']))
        if latency_z[i] > adaptive_k:
            bottlenecks['latency'].append((layer_name, metrics['latency']))

    # 打印结果（科学计数法）
    print(f"\n{model_name} Performance Summary (Threshold k={adaptive_k:.1f}):")
    for metric, layers in bottlenecks.items():
        if layers:
            print(f"{metric.capitalize()} Bottleneck Layers:")
            for layer, value in sorted(layers, key=lambda x: x[1], reverse=True):
                if metric == 'flops':
                    print(f"  {layer}: {value:.2e} FLOPs (Z-score: {flops_z[i]:.1f})")
                else:
                    print(
                        f"  {layer}: {value:.2f} {'MB' if metric == 'memory' else 'ms'} (Z-score: {eval(f'{metric}_z[i]'):.1f})")
        else:
            print(f"No {metric} bottleneck layers found")

    return bottlenecks