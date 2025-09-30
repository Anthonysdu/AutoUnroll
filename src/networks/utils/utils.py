import os
import re
import torch
import json
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import Dataset as GDataset


class MultiBoundGraphDataset(GDataset):
    def __init__(
        self, node_files, edge_files, loop_tokens, result_files, mode, full_encode, edges, transform=None,
    ):
        self.transform = transform
        self.graph_infos = []
        self.mode = mode
        self.full_encode = full_encode
        self.edges_to_use = edges

        edge_type_map = {0: "AST", 1: "ICFG", 2: "Data"}

        for node_file, edge_file, loop_token, result_file in zip(
            node_files, edge_files, loop_tokens, result_files
        ):
            print(node_file)
            # node data
            node_data = np.load(node_file)
            x = torch.tensor(node_data["node_rep"], dtype=torch.float)

            edge_data = np.load(edge_file)
            edge_index_list = []
            edge_attr_list = []
            for e_type in self.edges_to_use:
                key = edge_type_map[e_type]
                if key in edge_data:
                    print(key)
                    edges = torch.tensor(edge_data[key], dtype=torch.long).t().contiguous()
                    edge_index_list.append(edges)
                    edge_attr_type = torch.full((edges.size(1), 1), e_type, dtype=torch.float)
                    edge_attr_list.append(edge_attr_type)
            if edge_index_list:
                edge_index = torch.cat(edge_index_list, dim=1)
                edge_attr = torch.cat(edge_attr_list, dim=0)
            else:
                assert(0)

            src_nodes = edge_index[0].tolist()
            token_data = json.load(open(loop_token))

            result_json = json.load(open(result_file))
            labels = [item["result"] for item in result_json["results"]]
            bounds = [
                {int(k): int(v) for k, v in item.items() if k != "result"}
                for item in result_json["results"]
            ]

            self.graph_infos.append(
                (
                    x,
                    edge_index,
                    edge_attr,
                    src_nodes,
                    token_data,
                    bounds,
                    labels,
                    result_file,
                )
            )

        self.global_index = []
        for g_idx, (_, _, _, _, _, bounds, _, _) in enumerate(self.graph_infos):
            for b_idx in range(len(bounds)):
                self.global_index.append((g_idx, b_idx))

    def len(self):
        return len(self.global_index)

    def indices(self):
        return list(range(len(self.global_index)))

    def get(self, idx):
        g_idx, b_idx = self.global_index[idx]
        (
            x,
            edge_index,
            edge_attr,
            src_nodes,
            token_data,
            bounds,
            labels,
            result_file,
        ) = self.graph_infos[g_idx]
        bound = bounds[b_idx]
        label = torch.tensor([labels[b_idx]], dtype=torch.float)

        graph_max_unwind = max(bound.values()) if bound else 0
        graph_max_unwind_tensor = torch.full((x.size(0), 1), float(graph_max_unwind))
        x = torch.cat([x, graph_max_unwind_tensor], dim=1)
        node_unwind = torch.zeros(x.size(0), 1) 
        if self.mode == 1:
            unwind_attr = []
            if self.full_encode:
                for src in src_nodes:
                    unwind_val = 0
                    for token_str, token_info in token_data.items():
                        loopid = token_info['loop_id']
                        reachable = token_info.get('reachable', [])
                        if src in reachable:
                            unwind_val += bound.get(loopid, 0)
                    unwind_attr.append(unwind_val)
            else:
                unwind_attr = []
                for src in src_nodes:
                    token_str = str(src)
                    if token_str in token_data:
                        loopid = token_data[token_str]["loop_id"]
                        unwind_val = bound.get(loopid, 0)
                    else:
                        unwind_val = 0
                    unwind_attr.append(unwind_val)
            unwind_attr = torch.tensor(unwind_attr, dtype=torch.float).unsqueeze(1)

            edge_attr = torch.cat([edge_attr, unwind_attr], dim=1)
        else:
            for node_id in range(x.size(0)):
                val = 0
                for token_str, token_info in token_data.items():
                    loopid = token_info['loop_id']
                    reachable = token_info.get('reachable', [])
                    if node_id in reachable:
                        val += bound.get(loopid, 0)
                node_unwind[node_id, 0] = val
            edge_attr = edge_attr

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label, node_unwind=node_unwind)
        data.result_file = result_file
        data.unwind_dict = json.dumps(bound)
        if self.transform:
            data = self.transform(data)
        return data