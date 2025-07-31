import os
import re
import torch
import json
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch_geometric.data import Dataset as GDataset

class MultiBoundGraphDataset(GDataset):
    def __init__(self, node_files, edge_files, loop_tokens, result_files, transform=None):
        self.transform = transform
        self.graph_infos = []

        for node_file, edge_file, loop_token, result_file in zip(
            node_files, edge_files, loop_tokens, result_files
        ):
            print(node_file)
            # node data
            node_data = np.load(node_file)
            x = torch.tensor(node_data["node_rep"], dtype=torch.float)

            # edge data — 只保留 ICFG 边
            edge_data = np.load(edge_file)
            icfg_edges = torch.tensor(edge_data["ICFG"], dtype=torch.long).t().contiguous()
            edge_index = icfg_edges

            # edge_attr_type：ICFG 类型设为 2
            edge_attr_type = torch.full((icfg_edges.size(1), 1), 2, dtype=torch.float)

            # 记录 source nodes
            src_nodes = edge_index[0].tolist()
            token_data = json.load(open(loop_token))

            # 结果与展开因子
            result_json = json.load(open(result_file))
            labels = [item["result"] for item in result_json["results"]]
            bounds = [
                {int(k): int(v) for k, v in item.items() if k != "result"}
                for item in result_json["results"]
            ]

            self.graph_infos.append(
                (x, edge_index, edge_attr_type, src_nodes, token_data, bounds, labels, result_file)
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
        x, edge_index, edge_attr_type, src_nodes, token_data, bounds, labels, result_file = (
            self.graph_infos[g_idx]
        )
        bound = bounds[b_idx]
        label = torch.tensor([labels[b_idx]], dtype=torch.float)

        # 添加展开因子作为 node 特征（图级最大值）
        graph_max_unwind = max(bound.values()) if bound else 0
        graph_max_unwind_tensor = torch.full((x.size(0), 1), float(graph_max_unwind))
        x = torch.cat([x, graph_max_unwind_tensor], dim=1)

        # 计算每条边的 unwind 属性（upper_bound 被移除）
        unwind_attr = []
        for src in src_nodes:
            token_str = str(src)
            if token_str in token_data:
                loopid = token_data[token_str]['loop_id']
                unwind_val = bound.get(loopid, 0)
            else:
                unwind_val = 0
            unwind_attr.append(unwind_val)

        unwind_attr = torch.tensor(unwind_attr, dtype=torch.float).unsqueeze(1)

        # 拼接最终 edge_attr: [edge_type, unwind_val]
        edge_attr = torch.cat([edge_attr_type, unwind_attr], dim=1)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label)
        data.result_file = result_file
        data.unwind_dict = json.dumps(bound)
        if self.transform:
            data = self.transform(data)
        return data

# class MultiBoundGraphDataset(GDataset):
#     def __init__(self, node_files, edge_files, loop_tokens, result_files, transform=None):
#         self.transform = transform
#         self.graph_infos = (
#             []
#         )  # shape: [(x, edge_index, edge_attr_type, src_nodes, json_data, [bounds], [verdicts])]
        
#         for node_file, edge_file, loop_token, result_file in zip(
#             node_files, edge_files, loop_tokens, result_files
#         ):
#             # node data
#             node_data = np.load(node_file)
#             x = torch.tensor(node_data["node_rep"], dtype=torch.float)

#             # edge data
#             edge_data = np.load(edge_file)
#             ast_edges = (
#                 torch.tensor(edge_data["AST"], dtype=torch.long).t().contiguous()
#             )
#             data_edges = (
#                 torch.tensor(edge_data["Data"], dtype=torch.long).t().contiguous()
#             )
#             icfg_edges = (
#                 torch.tensor(edge_data["ICFG"], dtype=torch.long).t().contiguous()
#             )
#             edge_index = torch.cat([ast_edges, data_edges, icfg_edges], dim=1)

#             # edge attribute. shape: [edge type, unwind number]
#             edge_attr_ast = torch.full((ast_edges.size(1), 1), 0, dtype=torch.float)
#             edge_attr_data = torch.full((data_edges.size(1), 1), 1, dtype=torch.float)
#             edge_attr_icfg = torch.full((icfg_edges.size(1), 1), 2, dtype=torch.float)
#             edge_attr_type = torch.cat(
#                 [edge_attr_ast, edge_attr_data, edge_attr_icfg], dim=0
#             )

#             # starting nodes (token) among edges, used to locate the loops nodes. token is the key, loop line_number is the value.
#             src_nodes = edge_index[0].tolist()
#             token_data = json.load(open(loop_token))

#             # extract unwind number and the labels.
#             result_json = json.load(open(result_file))
#             labels = [item["result"] for item in result_json["results"]]
#             bounds = [
#                 {int(k): int(v) for k, v in item.items() if k != "result"}
#                 for item in result_json["results"]
#             ]
#             self.graph_infos.append(
#                 (x, edge_index, edge_attr_type, src_nodes, token_data, bounds, labels, result_file)
#             )

#         # global index, shape: [(graph_id, bound_id)]
#         self.global_index = []
#         for g_idx, (_, _, _, _, _, bounds, _, _) in enumerate(self.graph_infos):
#             for b_idx in range(len(bounds)):
#                 self.global_index.append((g_idx, b_idx))

#     def len(self):
#         return len(self.global_index)
#     def indices(self):
#         # This method returns the indices of the dataset
#         return list(range(len(self.global_index)))
#     def get(self, idx):
#         # find graph_id and bound id by idx.
#         g_idx, b_idx = self.global_index[idx]
#         x, edge_index, edge_attr_type, src_nodes, token_data, bounds, labels, result_file = (
#             self.graph_infos[g_idx]
#         )
#         bound = bounds[b_idx]
#         label = labels[b_idx]
#         label = torch.tensor([label], dtype=torch.float)

#         graph_max_unwind = max(bound.values()) if bound else 0
#         graph_max_unwind_tensor = torch.full((x.size(0), 1), float(graph_max_unwind))
#         x = torch.cat([x, graph_max_unwind_tensor], dim=1)

#         max_unwind_per_loop = {}
#         for b in bounds:
#             for loop_id, val in b.items():
#                 if loop_id not in max_unwind_per_loop:
#                     max_unwind_per_loop[loop_id] = val
#                 else:
#                     max_unwind_per_loop[loop_id] = max(max_unwind_per_loop[loop_id], val)

        
#         unwind_attr = []
#         upper_bound = []
#         for src in src_nodes:
#             token_str = str(src)
#             if token_str in token_data:
#                 # loopid is the loop's line number in the code.
#                 loopid = token_data[token_str]['loop_id']
#                 unwind_val = bound.get(loopid, 0)
#                 static_upper = token_data[token_str]["max_iterations"]
#                 if static_upper == "Unknown":
#                     static_upper = 1e6
#                 else:
#                     static_upper = int(static_upper) + 1
#                 dynamic_upper = max_unwind_per_loop.get(loopid, 0)
#                 # 有争议
#                 combined_upper = max(static_upper, dynamic_upper)
#             else:
#                 unwind_val = 0
#                 combined_upper = 0
#             unwind_attr.append(unwind_val)
#             upper_bound.append(combined_upper)
#         unwind_attr = torch.tensor(unwind_attr, dtype=torch.float).unsqueeze(1)
#         upper_bound = torch.tensor(upper_bound, dtype=torch.float).unsqueeze(1)
#         edge_attr = torch.cat([edge_attr_type, unwind_attr, upper_bound], dim=1)
#         data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label)
#         data.result_file = result_file
#         if self.transform:
#             data = self.transform(data)
#         return data

# Function to load graph data (node features and edges)
def load_graph_data(node_file, edge_file):
    node_data = np.load(node_file)
    x = torch.tensor(
        node_data["node_rep"], dtype=torch.float
    )  # node feature (one-hot encoding)

    edge_data = np.load(edge_file)
    ast_edges = edge_data["AST"]
    data_edges = edge_data["Data"]
    icfg_edges = edge_data["ICFG"]
    # lcfg_edges = edge_data["LCFG"]

    ast_edges = torch.tensor(ast_edges, dtype=torch.long).t().contiguous()
    data_edges = torch.tensor(data_edges, dtype=torch.long).t().contiguous()
    icfg_edges = torch.tensor(icfg_edges, dtype=torch.long).t().contiguous()

    edge_attr_ast = torch.full((ast_edges.size(1), 1), 0, dtype=torch.float)
    edge_attr_data = torch.full((data_edges.size(1), 1), 1, dtype=torch.float)
    edge_attr_icfg = torch.full((icfg_edges.size(1), 1), 2, dtype=torch.float)

    edge_index = torch.cat([ast_edges, data_edges, icfg_edges], dim=1)
    json_data = json.load(open("../../data/final_graphs/test.c.json"))
    edge_attr_type = torch.cat([edge_attr_ast, edge_attr_data, edge_attr_icfg], dim=0)

    src_nodes = edge_index[0].tolist()
    file_path = "../../test.c.txt"
    bounds, benchmark, verdit = extract_bounds_and_labels(file_path)
    print(verdit)
    data_list = []
    for i, bound in enumerate(bounds):
        unwind_attr = []
        for src in src_nodes:
            token_str = str(src)
            if token_str in json_data:
                loopid = json_data[token_str]
                unwind_val = bound.get(loopid, 0)
            else:
                unwind_val = 0
            unwind_attr.append(unwind_val)
        unwind_attr = torch.tensor(unwind_attr, dtype=torch.float).unsqueeze(1)
        edge_attr = torch.cat(
            [edge_attr_type, unwind_attr], dim=1
        )  # [edge_type, unwind]
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=verdit[i])
        data_list.append(data)
    return data_list


# Function to load labels from a file
def load_labels(label_file):
    with open(label_file, "r") as f:
        label_data = json.load(f)

    labels_dict = {}
    for item in label_data:
        program_name = item["program_name"]
        labels = item["labels"]
        labels_dict[program_name] = labels
    return labels_dict


# Function to prepare data for training, validation, and testing
def prepare_data(node_file, edge_file, label_file):
    print(label_file)
    return load_graph_data(node_file, edge_file)
    labels_dict = load_labels(label_file)
    file_name = os.path.basename(node_file)
    program_name, _ = os.path.splitext(file_name)
    labels = labels_dict.get(program_name.replace(".json", ""), [])
    # print(labels_dict)
    file_path = "../../test.c.txt"
    bound, benchmark, verdit = extract_bounds_from_file(file_path)
    for i, unwind in enumerate(bound):
        print(unwind)

    exit(0)
    data_list = []
    for i, lcfg_edge in enumerate(lcfg_edges):
        lcfg_edge_index = torch.tensor(lcfg_edge, dtype=torch.long)

        if lcfg_edge_index.dim() == 1:
            lcfg_edge_index = lcfg_edge_index.unsqueeze(0).repeat(2, 1)

        lcfg_edge_index = lcfg_edge_index.contiguous()

        # edge_list = [ast_edges, data_edges, icfg_edges, lcfg_edge_index]
        # combined_edge_index = torch.cat(edge_list, dim=1)

        if i < len(labels):
            label = torch.tensor(labels[i], dtype=torch.float)

            label = label.unsqueeze(0) if label.dim() == 1 else label
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label)
            data_list.append(data)

    return data_list


# Function to load data in batches
def load_data_in_batches(label_file, batch_size):
    labels_dict = load_labels(label_file)
    print(labels_dict)
    program_names = list(labels_dict.keys())

    for i in range(0, len(program_names), batch_size):
        batch_program_names = program_names[i : i + batch_size]
        batch_data_list = []

        for program_name in batch_program_names:
            node_file = f"../../data/final_graphs/{program_name}.json.npz"
            edge_file = f"../../data/final_graphs/{program_name}.jsonEdges.npz"
            if os.path.exists(node_file) and os.path.exists(edge_file):
                data_list = prepare_data(node_file, edge_file, label_file)
                batch_data_list.extend(data_list)

        # yield batch_data_list
        return batch_data_list


def extract_bounds_from_file(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        benchmark_with_verdict = file.readline().strip()
        text = file.read()

    pattern = r"(\d+):(\d+),(\d+):(\d+) (-?\d+) (\d+\.\d+)"

    matches = re.findall(pattern, text)
    extracted_numbers = []

    for match in matches:
        extracted_numbers.append(
            {
                int(match[0]): int(match[1]),
                int(match[2]): int(match[3]),
            }
        )
    labels = [(int(match[4]), float(match[5])) for match in matches]
    return extracted_numbers, benchmark_with_verdict, labels


def extract_bounds_and_labels(file_path):
    file = json.load(file_path)
    bounds = []
    labels = []
    return bounds, labels
