import copy
from collections import defaultdict

import torch

from .dag_generator import DAGGenerator
from .fusion_evaluator import FusionEvaluator
import torch.nn as nn


class FusionOptimizer:
    def __init__(self, model, layer_metrics):
        self.model = model
        self.layer_metrics = layer_metrics
        self.dag_generator = DAGGenerator(model)
        self.evaluator = FusionEvaluator(model, layer_metrics)

    def optimize(self, bottlenecks, priority='speedup'):
        expanded = self.dag_generator.expand_bottlenecks(set(bottlenecks))
        subgraphs = self.dag_generator.generate_subgraphs(expanded)
        optimizations = []

        for subgraph in subgraphs:
            # 尝试不同深度的组合
            for depth in [3, 2]:  # 先尝试3层组合，再尝试2层
                combinations = self.evaluator.enumerate_combinations(subgraph)
                if not combinations and depth == 3:
                    continue  # 如果找不到3层组合，尝试2层

                evaluations = self.evaluator.evaluate_combinations(combinations)
                if evaluations:
                    best = max(evaluations, key=lambda x: x[priority])
                    # 检查是否已经包含相同的层
                    if not any(opt['layers'][0] == best['layers'][0] for opt in optimizations):
                        optimizations.append(best)
                    break

        # 如果还是找不到，尝试放宽条件
        if not optimizations and len(bottlenecks) > 0:
            print("No standard fusion patterns found, trying fallback strategies...")
            return self._fallback_optimize(bottlenecks)

        return optimizations

    def _fallback_optimize(self, bottlenecks):
        """备选优化策略"""
        optimizations = []
        for layer in bottlenecks:
            try:
                module = self.model.get_submodule(layer)
                if isinstance(module, nn.Conv2d):
                    # 尝试找到后续可融合的层
                    next_layers = self._find_next_layers(layer)
                    for next_layer in next_layers:
                        next_module = self.model.get_submodule(next_layer)
                        if isinstance(next_module, nn.BatchNorm2d):
                            optimizations.append({
                                'layers': (layer, next_layer),
                                'combo': 'Conv+BN',
                                'speedup': 1.3,
                                'memory_reduction': 0.3
                            })
                        elif isinstance(next_module, nn.ReLU):
                            optimizations.append({
                                'layers': (layer, next_layer),
                                'combo': 'Conv+ReLU',
                                'speedup': 1.2,
                                'memory_reduction': 0.2
                            })
            except AttributeError:
                continue
        return optimizations
    def apply_optimizations(self, optimizations):
        try:
            fused_model = copy.deepcopy(self.model)
            for opt in optimizations:
                layer1_name, layer2_name = opt['layers']
                try:
                    layer1 = fused_model.get_submodule(layer1_name)
                    layer2 = fused_model.get_submodule(layer2_name)

                    if opt['combo'] == 'Conv+ReLU':
                        fused = nn.Sequential(layer1, layer2)
                        self._replace_module(fused_model, layer1_name, fused)
                        self._remove_module(fused_model, layer2_name)
                    elif opt['combo'] == 'Conv+BatchNorm':
                        # 实现Conv+BN融合逻辑
                        fused = self._fuse_conv_bn(layer1, layer2)
                        self._replace_module(fused_model, layer1_name, fused)
                        self._remove_module(fused_model, layer2_name)
                except AttributeError:
                    continue
            return fused_model
        except Exception as e:
            print(f"Failed to apply optimizations: {str(e)}")
            return self.model

    def _fuse_conv_bn(self, conv, bn):
        """融合Conv和BN层"""
        fused_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=True
        )

        # 计算融合后的权重和偏置
        fused_conv.weight.data = (bn.weight / torch.sqrt(bn.running_var + bn.eps)) \
                                     .view(-1, 1, 1, 1) * conv.weight.data
        fused_conv.bias.data = bn.bias - bn.weight * bn.running_mean / \
                               torch.sqrt(bn.running_var + bn.eps)

        return fused_conv

    def _replace_module(self, model, module_name, new_module):
        parts = module_name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def _remove_module(self, model, module_name):
        parts = module_name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        if hasattr(parent, parts[-1]):
            delattr(parent, parts[-1])