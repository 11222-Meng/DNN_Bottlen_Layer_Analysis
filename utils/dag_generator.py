import os

import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    from networkx.drawing.nx_agraph import write_dot

    DOT_AVAILABLE = True
except ImportError:
    DOT_AVAILABLE = False


class DAGGenerator:
    def __init__(self, expansion_N=5):
        self.expansion_N = expansion_N
        self.dag_dir = "dags"
        os.makedirs(self.dag_dir, exist_ok=True)

        # 算子类型颜色映射
        self.op_colors = {
            'Conv2d': '#FF9AA2',
            'Linear': '#FFB7B2',
            'MaxPool2d': '#FFDAC1',
            'AvgPool2d': '#E2F0CB',
            'ReLU': '#B5EAD7',
            'BatchNorm2d': '#C7CEEA',
            'AdaptiveAvgPool2d': '#F8B195',
            'InvertedResidual': '#F67280',
            'FireModule': '#6C5B7B'
        }

    def _get_operator_type(self, module: nn.Module) -> str:
        """标准化算子类型名称"""
        op_map = {
            nn.Conv2d: 'Conv2d',
            nn.Linear: 'Linear',
            nn.MaxPool2d: 'MaxPool2d',
            nn.AvgPool2d: 'AvgPool2d',
            nn.ReLU: 'ReLU',
            nn.BatchNorm2d: 'BatchNorm2d',
            nn.AdaptiveAvgPool2d: 'AdaptiveAvgPool2d'
        }
        return op_map.get(type(module), module.__class__.__name__)

    def _build_operator_graph(self, model: nn.Module, layer_set: List[str]) -> nx.DiGraph:
        """构建算子级计算图"""
        G = nx.DiGraph()
        op_counters = defaultdict(int)
        prev_op = None

        for name, module in model.named_modules():
            if name not in layer_set:
                continue

            # 处理特殊模块
            if 'InvertedResidual' in str(type(module)):
                op_type = 'InvertedResidual'
            elif 'Fire' in str(type(module)):
                op_type = 'FireModule'
            else:
                op_type = self._get_operator_type(module)

            op_id = f"{op_type}_{op_counters[op_type]}"
            op_counters[op_type] += 1

            # 添加节点属性
            G.add_node(op_id,
                       type=op_type,
                       layer_name=name,
                       params=sum(p.numel() for p in module.parameters()),
                       color=self.op_colors.get(op_type, '#DDDDDD'))

            # 添加边
            if prev_op is not None:
                G.add_edge(prev_op, op_id)
            prev_op = op_id

        return G

    def _expand_bottleneck_layers(self, model, bottlenecks, all_layers):
        """双向扩展瓶颈层范围"""
        expanded_sets = {}

        for metric, layers in bottlenecks.items():
            expanded_sets[metric] = {}

            for layer_name, _ in layers:
                try:
                    idx = all_layers.index(layer_name)

                    # 向前扩展
                    start_idx = max(0, idx - self.expansion_N)
                    # 向后扩展
                    end_idx = min(idx + self.expansion_N + 1, len(all_layers))

                    # 取消截断逻辑，让扩展完整执行
                    expanded_sets[metric][layer_name] = all_layers[start_idx:end_idx]
                except ValueError:
                    continue

        return expanded_sets

    def generate_submodel_dags(self, model, model_name, layer_metrics, k=1.5):
        """生成算子级DAG的主方法"""
        # 1. 识别瓶颈层
        bottlenecks = self._identify_bottlenecks(layer_metrics, k)

        # 2. 获取所有层列表
        all_layers = [name for name, _ in model.named_modules()
                      if isinstance(_, (nn.Conv2d, nn.Linear, nn.MaxPool2d,
                                        nn.AvgPool2d, nn.ReLU, nn.BatchNorm2d))]

        # 3. 扩展瓶颈层
        expanded_sets = self._expand_bottleneck_layers(model, bottlenecks, all_layers)

        # 4. 生成DAG
        dags = {}
        for metric, layers in expanded_sets.items():
            dags[metric] = {}
            for layer_name, layer_set in layers.items():
                op_dag = self._build_operator_graph(model, layer_set)

                if len(op_dag.nodes) > 0:
                    self._save_dag(op_dag, model_name, metric, layer_name)
                    self._visualize_dag(op_dag, model_name, metric, layer_name)
                    dags[metric][layer_name] = op_dag

        return dags

    def _identify_bottlenecks(self, layer_metrics, k):
        """识别三种指标下的瓶颈层"""
        bottlenecks = {'memory': [], 'flops': [], 'latency': []}

        for metric in bottlenecks.keys():
            values = [m.get(metric, 0) for m in layer_metrics.values()]
            if not values:
                continue

            mean = sum(values) / len(values)
            std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5

            for layer_name, m in layer_metrics.items():
                if m.get(metric, 0) > mean + k * std:
                    bottlenecks[metric].append((layer_name, m[metric]))

        return bottlenecks

    def _save_dag(self, dag: nx.DiGraph, model_name: str, metric: str, layer_name: str):
        """保存DAG到文件"""
        filename = f"{model_name}_{metric}_{layer_name.replace('.', '_')}_op"
        filepath = os.path.join(self.dag_dir, filename)

        if DOT_AVAILABLE:
            write_dot(dag, f"{filepath}.dot")
        else:
            nx.write_gml(dag, f"{filepath}.gml")

    def _visualize_dag(self, dag: nx.DiGraph, model_name: str, metric: str, layer_name: str):
        """可视化DAG"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch

            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(dag, seed=42)

            # 绘制节点和边
            node_colors = [dag.nodes[n]['color'] for n in dag.nodes]
            nx.draw_networkx_nodes(dag, pos, node_color=node_colors, node_size=1500)
            nx.draw_networkx_edges(dag, pos, arrowstyle='->', arrowsize=20)

            # 添加标签
            labels = {n: f"{dag.nodes[n]['type']}\n{dag.nodes[n]['layer_name']}"
                      for n in dag.nodes}
            nx.draw_networkx_labels(dag, pos, labels, font_size=8)

            # 添加图例
            legend_handles = [
                Patch(color=color, label=op_type)
                for op_type, color in self.op_colors.items()
            ]
            plt.legend(handles=legend_handles, loc='upper right')

            plt.title(f"{model_name} {metric} bottleneck at {layer_name}")
            plt.tight_layout()

            # 保存图像
            filename = f"{model_name}_{metric}_{layer_name.replace('.', '_')}_op.png"
            plt.savefig(os.path.join(self.dag_dir, filename))
            plt.close()
        except Exception as e:
            print(f"Visualization failed: {str(e)}")

#切片点识别功能
class SlicePointIdentifier:
    def __init__(self,
                 threshold=0.3,  # 统一使用threshold作为参数名
                 cost_ratio_max=2.0,
                 min_metric_value=1e-3):
        self.feat_thresh = threshold  # 内部变量名可以不同
        self.cost_ratio_max = cost_ratio_max
        self.min_val = min_metric_value

    def identify_slice_points(self, dag: nx.DiGraph, metric: str):
        """增强版切片点识别"""
        candidates = []

        # 预处理：计算全图指标均值
        all_metrics = [dag.nodes[n].get(metric, 0) for n in dag.nodes]
        avg_metric = max(sum(all_metrics) / len(all_metrics), self.min_val)

        for u, v in dag.edges():
            u_metric = dag.nodes[u].get(metric, 0)
            v_metric = dag.nodes[v].get(metric, 0)

            # 动态阈值调整（基于当前边指标的相对大小）
            dynamic_thresh = self.feat_thresh * (avg_metric / (v_metric + self.min_val))

            # 特征尺寸计算（考虑张量形状）
            feature_size = self._calc_real_feature_size(dag, u, v)

            # 成本计算（考虑实际通信开销）
            cut_cost = self._calc_cut_cost(dag, u, v)
            fused_cost = self._calc_fused_cost(dag, u, v)
            cost_ratio = cut_cost / (fused_cost + self.min_val)

            if (feature_size > dynamic_thresh and
                    cost_ratio < self.cost_ratio_max):
                candidates.append({
                    'edge': (u, v),
                    'feature_size': feature_size,
                    'cut_cost': cut_cost,
                    'fused_cost': fused_cost,
                    'score': feature_size / (cost_ratio + self.min_val)
                })

        return sorted(candidates, key=lambda x: -x['score'])

    def _calc_real_feature_size(self, dag, u, v):
        """基于实际输出张量形状计算特征尺寸"""
        u_output_shape = dag.nodes[u].get('output_shape', [1])  # 假设有记录输出形状
        v_output_shape = dag.nodes[v].get('output_shape', [1])
        return (np.prod(v_output_shape) + 1) / (np.prod(u_output_shape) + 1)

    def _calc_cut_cost(self, dag, u, v):
        """考虑实际通信开销的切割成本"""
        u_params = dag.nodes[u].get('params', 0)
        v_params = dag.nodes[v].get('params', 0)
        return 0.5 * u_params + 1.5 * v_params  # 模拟传输开销

    def _calc_fused_cost(self, dag, u, v):
        """考虑缓存命中的融合成本"""
        return dag.nodes[u].get('latency', 0) + dag.nodes[v].get('latency', 0) * 0.8