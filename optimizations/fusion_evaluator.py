import torch.nn as nn
from .static_analyzer import StaticAnalyzer


class FusionEvaluator:
    def __init__(self, model, layer_metrics):
        self.model = model
        self.layer_metrics = layer_metrics
        self.supported_combinations = {
            ('Conv2d', 'BatchNorm2d'): 'Conv+BN',
            ('Conv2d', 'ReLU'): 'Conv+ReLU',
            ('Conv2d', 'BatchNorm2d', 'ReLU'): 'Conv+BN+ReLU',
            ('Linear', 'ReLU'): 'Linear+ReLU',
            ('Conv2d', 'Hardswish'): 'Conv+Hardswish'  # MobileNetV3支持
        }

    def enumerate_combinations(self, subgraph):
        """支持更灵活的层组合检测"""
        layers = sorted(subgraph, key=lambda x: len(x.split('.')))
        combinations = []

        # 检查2层组合
        for i in range(len(layers) - 1):
            layer1 = self._get_layer_type(layers[i])
            layer2 = self._get_layer_type(layers[i + 1])
            combo = self.supported_combinations.get((layer1, layer2))
            if combo:
                combinations.append((layers[i], layers[i + 1], combo))

        # 检查3层组合 (Conv+BN+ReLU)
        for i in range(len(layers) - 2):
            layer1 = self._get_layer_type(layers[i])
            layer2 = self._get_layer_type(layers[i + 1])
            layer3 = self._get_layer_type(layers[i + 2])
            combo = self.supported_combinations.get((layer1, layer2, layer3))
            if combo:
                combinations.append((layers[i], layers[i + 2], combo))

        return combinations

    def _get_layer_type(self, layer_name):
        """获取层的类型名称"""
        try:
            layer = self.model.get_submodule(layer_name)
            return layer.__class__.__name__
        except AttributeError:
            return None

    def evaluate_combinations(self, combinations):
        evaluations = []
        for layer1, layer2, combo in combinations:
            static_analysis = StaticAnalyzer.analyze_fusion(combo,
                                                            self.model.get_submodule(layer1),
                                                            self.model.get_submodule(layer2))
            evaluations.append({
                'layers': (layer1, layer2),
                'combo': combo,
                'speedup': StaticAnalyzer.estimate_speedup(static_analysis),
                'memory_reduction': static_analysis['memory_access']
            })
        return evaluations