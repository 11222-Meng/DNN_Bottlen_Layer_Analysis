import torch.nn as nn


class StaticAnalyzer:
    @staticmethod
    def analyze_fusion(combo_type, layer1, layer2):
        analysis = {'kernel_launch_reduction': 0, 'redundant_computation': 0, 'memory_access': 0}
        if combo_type == 'Conv+ReLU':
            analysis.update({
                'kernel_launch_reduction': 1,
                'redundant_computation': 0.2,
                'memory_access': 0.3
            })
        elif combo_type == 'Conv+BatchNorm':
            analysis.update({
                'kernel_launch_reduction': 1,
                'redundant_computation': 0.3,
                'memory_access': 0.4
            })
        return analysis

    @staticmethod
    def estimate_speedup(analysis):
        return 1 + analysis['kernel_launch_reduction'] * 0.1 + \
            analysis['redundant_computation'] * 0.3 + \
            analysis['memory_access'] * 0.2