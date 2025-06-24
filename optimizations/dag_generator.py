from collections import defaultdict, deque


class DAGGenerator:
    def __init__(self, model):
        self.model = model
        self.named_modules = dict(model.named_modules())
        self.adjacency_list = self._build_adjacency_list()

    def _build_adjacency_list(self):
        adj_list = defaultdict(list)
        for name, module in self.named_modules.items():
            parts = name.split('.')
            if len(parts) > 1:
                parent = '.'.join(parts[:-1])
                adj_list[parent].append(name)
        return adj_list

    def expand_bottlenecks(self, bottlenecks):
        expanded = set(bottlenecks)
        for layer in bottlenecks:
            next_layers = self.adjacency_list.get(layer, [])
            if next_layers and next_layers[0] not in bottlenecks:
                expanded.add(next_layers[0])
            elif not next_layers or next_layers[0] in bottlenecks:
                parts = layer.split('.')
                if len(parts) > 1:
                    parent = '.'.join(parts[:-1])
                    if parent not in bottlenecks:
                        expanded.add(parent)
        return expanded

    def generate_subgraphs(self, expanded_bottlenecks):
        subgraphs = []
        visited = set()
        for layer in expanded_bottlenecks:
            if layer not in visited:
                queue = deque([layer])
                subgraph = set()
                while queue:
                    current = queue.popleft()
                    if current in visited:
                        continue
                    visited.add(current)
                    subgraph.add(current)
                    parts = current.split('.')
                    if len(parts) > 1:
                        parent = '.'.join(parts[:-1])
                        if parent in expanded_bottlenecks:
                            queue.append(parent)
                    for child in self.adjacency_list.get(current, []):
                        if child in expanded_bottlenecks:
                            queue.append(child)
                subgraphs.append(subgraph)
        return subgraphs