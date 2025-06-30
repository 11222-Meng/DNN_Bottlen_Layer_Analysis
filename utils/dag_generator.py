import os
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
    def __init__(self, expansion_N=3):
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
        """扩展瓶颈层范围"""
        expanded_sets = {}

        for metric, layers in bottlenecks.items():
            expanded_sets[metric] = {}

            for layer_name, _ in layers:
                try:
                    idx = all_layers.index(layer_name)
                    end_idx = min(idx + self.expansion_N + 1, len(all_layers))

                    # 动态调整扩展范围
                    for i in range(idx + 1, end_idx):
                        if any(all_layers[i] == b[0] for b_list in bottlenecks.values() for b in b_list):
                            end_idx = i
                            break

                    expanded_sets[metric][layer_name] = all_layers[idx:end_idx]
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