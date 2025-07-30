import os
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash

try:
    from networkx.drawing.nx_agraph import write_dot

    DOT_AVAILABLE = True
except ImportError:
    DOT_AVAILABLE = False


class DAGGenerator:
    def __init__(self, expansion_N=2):
        self.expansion_N = expansion_N
        self.dag_dir = "dags"
        os.makedirs(self.dag_dir, exist_ok=True)

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
        G = nx.DiGraph()
        op_counters = defaultdict(int)
        prev_op = None

        for name in layer_set:
            module = dict(model.named_modules())[name]

            if 'InvertedResidual' in str(type(module)):
                op_type = 'InvertedResidual'
            elif 'Fire' in str(type(module)):
                op_type = 'FireModule'
            else:
                op_type = self._get_operator_type(module)

            op_id = f"{op_type}_{op_counters[op_type]}"
            op_counters[op_type] += 1

            G.add_node(op_id,
                       type=op_type,
                       layer_name=name,
                       params=sum(p.numel() for p in module.parameters()),
                       color=self.op_colors.get(op_type, '#DDDDDD'))

            if prev_op is not None:
                G.add_edge(prev_op, op_id)
            prev_op = op_id

        return G

    def _expand_bottleneck_layers(self, model, bottlenecks, all_layers, selected_metric):
        expanded_sets = {}
        candidate_ranges = []

        for name, _ in bottlenecks[selected_metric]:
            if name not in all_layers:
                continue

            idx = all_layers.index(name)
            start, end = self._get_expansion_range(model, name, idx, all_layers)
            candidate_ranges.append((start, end, name))

        merged_ranges = []
        for start, end, name in sorted(candidate_ranges, key=lambda x: x[0]):
            if not merged_ranges:
                merged_ranges.append((start, end, [name]))
            else:
                last_start, last_end, last_names = merged_ranges[-1]
                if start <= last_end:
                    new_start = min(start, last_start)
                    new_end = max(end, last_end)
                    merged_ranges[-1] = (new_start, new_end, last_names + [name])
                else:
                    merged_ranges.append((start, end, [name]))

        for start, end, names in merged_ranges:
            layer_set = all_layers[start:end + 1]
            primary_name = names[0]
            expanded_sets[primary_name] = layer_set
            print(f"Expanded bottleneck {primary_name}: {len(layer_set)} layers (N={self.expansion_N})")

        return expanded_sets

    def _get_expansion_range(self, model, layer_name, idx, all_layers):
        """基于模型结构智能扩展"""
        module = dict(model.named_modules())[layer_name]
        parent_name = layer_name.rsplit('.', 1)[0]  # 获取父模块名

        # 情况1：如果是ResNet的Bottleneck块
        if 'layer' in parent_name and isinstance(module, nn.Conv2d):
            # 找到该Bottleneck块的所有层
            block_layers = [name for name in all_layers if name.startswith(parent_name)]
            if block_layers:
                return all_layers.index(block_layers[0]), all_layers.index(block_layers[-1])

        # 情况2：如果是VGG的连续卷积层
        elif 'features' in parent_name and isinstance(module, nn.Conv2d):
            # 扩展时包含相邻的卷积层（最多N层）
            start = max(0, idx - self.expansion_N)
            end = min(len(all_layers) - 1, idx + self.expansion_N)
            return start, end

        # 默认行为：严格限制在N层内
        start = max(0, idx - self.expansion_N)
        end = min(len(all_layers) - 1, idx + self.expansion_N)
        return start, end

    def generate_submodel_dags(self, model, model_name, layer_metrics, selected_metric, k=1.5):

        bottlenecks = self._identify_bottlenecks(layer_metrics, k)

        print(f"\nGenerating {model_name} {selected_metric} subgraphs...")
        print("Identified bottlenecks:", [name for name, _ in bottlenecks[selected_metric]])

        if not bottlenecks[selected_metric]:
            print("No bottlenecks found")
            return {}

        all_layers = [
            name for name, module in model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))  # 仅保留核心计算层
        ]

        expanded_sets = self._expand_bottleneck_layers(model, bottlenecks, all_layers, selected_metric)

        dags = {}
        seen_hashes = set()

        for layer_name, layer_set in expanded_sets.items():
            op_dag = self._build_operator_graph(model, layer_set)

            if len(op_dag.nodes) == 0:
                continue

            dag_hash = weisfeiler_lehman_graph_hash(op_dag)
            if dag_hash in seen_hashes:
                continue

            seen_hashes.add(dag_hash)
            self._save_dag(op_dag, model_name, selected_metric, layer_name)
            self._visualize_dag(op_dag, model_name, selected_metric, layer_name)
            dags[layer_name] = op_dag

        return dags

    def _identify_bottlenecks(self, layer_metrics, k):
        bottlenecks = {'memory': [], 'flops': [], 'latency': []}

        for metric in bottlenecks.keys():
            values = [m.get(metric, 0) for m in layer_metrics.values()]
            if not values:
                print(f"[WARNING] No {metric} data found")
                continue

            mean = np.mean(values)
            std = np.std(values)

            print(f"[INFO] {metric} mean={mean:.4f}, std={std:.4f}")

            current_k = k
            while current_k >= 0.5:
                bottlenecks[metric] = [(name, m[metric]) for name, m in layer_metrics.items()
                                       if m.get(metric, 0) > mean + current_k * std]
                if bottlenecks[metric]:
                    print(f"[Adaptive k] Using k={current_k:.2f} found {len(bottlenecks[metric])} bottlenecks")
                    break
                current_k -= 0.2
            else:
                print(f"[WARNING] No {metric} bottlenecks found (min k=0.5)")

        return bottlenecks

    def _save_dag(self, dag: nx.DiGraph, model_name: str, metric: str, layer_name: str):
        filename = f"{model_name}_{metric}_{layer_name.replace('.', '_')}_op"
        filepath = os.path.join(self.dag_dir, filename)

        if DOT_AVAILABLE:
            write_dot(dag, f"{filepath}.dot")
        else:
            nx.write_gml(dag, f"{filepath}.gml")

    def _visualize_dag(self, dag: nx.DiGraph, model_name: str, metric: str, layer_name: str):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch

            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(dag, seed=42)

            node_colors = [dag.nodes[n]['color'] for n in dag.nodes]
            nx.draw_networkx_nodes(dag, pos, node_color=node_colors, node_size=1500)
            nx.draw_networkx_edges(dag, pos, arrowstyle='->', arrowsize=20)

            labels = {n: f"{dag.nodes[n]['type']}\n{dag.nodes[n]['layer_name']}"
                      for n in dag.nodes}
            nx.draw_networkx_labels(dag, pos, labels, font_size=8)

            legend_handles = [
                Patch(color=color, label=op_type)
                for op_type, color in self.op_colors.items()
            ]
            plt.legend(handles=legend_handles, loc='upper right')

            plt.title(f"{model_name} {metric} bottleneck: {layer_name}\n{len(dag.nodes)} ops (N={self.expansion_N})")
            plt.tight_layout()

            filename = f"{model_name}_{metric}_{layer_name.replace('.', '_')}_op.png"
            plt.savefig(os.path.join(self.dag_dir, filename))
            plt.close()
        except Exception as e:
            print(f"Visualization failed: {str(e)}")


class SlicePointIdentifier:
    def __init__(self, threshold=0.3, cost_ratio_max=2.0, min_metric_value=1e-3):
        self.feat_thresh = threshold
        self.cost_ratio_max = cost_ratio_max
        self.min_val = min_metric_value

    def identify_slice_points(self, dag: nx.DiGraph, metric: str):
        candidates = []
        all_metrics = [dag.nodes[n].get(metric, 0) for n in dag.nodes]
        avg_metric = max(sum(all_metrics) / len(all_metrics), self.min_val)

        for u, v in dag.edges():
            u_metric = dag.nodes[u].get(metric, 0)
            v_metric = dag.nodes[v].get(metric, 0)
            dynamic_thresh = self.feat_thresh * (avg_metric / (v_metric + self.min_val))
            feature_size = self._calc_real_feature_size(dag, u, v)
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
        u_output_shape = dag.nodes[u].get('output_shape', [1])
        v_output_shape = dag.nodes[v].get('output_shape', [1])
        return (np.prod(v_output_shape) + 1) / (np.prod(u_output_shape) + 1)

    def _calc_cut_cost(self, dag, u, v):
        u_params = dag.nodes[u].get('params', 0)
        v_params = dag.nodes[v].get('params', 0)
        return 0.5 * u_params + 1.5 * v_params

    def _calc_fused_cost(self, dag, u, v):
        return dag.nodes[u].get('latency', 0) + dag.nodes[v].get('latency', 0) * 0.8