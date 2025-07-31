from collections import defaultdict
import json
from src.networks.utils.utils import MultiBoundGraphDataset
from src.networks.gnn import GATModel
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset    
from pathlib import Path
from tqdm import tqdm
from math import floor
import gc
import heapq
import random


num_sample = 1000
num_epochs = 20
batch_size = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")

base_path = Path("data/final_graphs")
result_path = Path("json_result")

loop_tokens = []
node_files = []
edge_files = []
result_files = []
step = 2
upper_bound = 21
start_bound = 2


def normalise_label(output):
    result = output[:, 0]
    time = output[:, 1]
    time = torch.clamp(time, max=60)
    n_result = (result == 1).float()
    n_time = 1 - time / 60
    return n_result, n_time

def s1(all_preds, all_preds_time, all_unwinds):
    max_idx = max(range(len(all_preds)), key=lambda i: all_preds[i])
    max_pred = torch.sigmoid(torch.tensor(all_preds[max_idx])).item()

    def unwind_sum(i):
        unwind = all_unwinds[i]
        return sum(unwind.values()) if unwind else float('inf')
    if max_pred > 0.9:
        high_conf_indices = [
        i for i in range(len(all_preds))
        if torch.sigmoid(torch.tensor(all_preds[i])).item() > 0.9
        ]
        topk_indices = sorted(high_conf_indices, key=unwind_sum)[:10]
    
    elif max_pred < 0.01:
        topk_indices = sorted(range(len(all_preds)), key=unwind_sum, reverse=True)[:10]
    else:
        topk_indices = sorted(
        range(len(all_preds)),
        key=lambda i: (all_preds[i], all_preds_time[i]),
        reverse=True
        )[:10]

    return topk_indices

def s2(all_preds, all_preds_time, all_unwinds):
    topk_indices = []
    max_idx = max(range(len(all_preds)), key=lambda i: all_preds[i])
    max_pred = torch.sigmoid(torch.tensor(all_preds[max_idx])).item()
    def unwind_sum(i):
        unwind = all_unwinds[i]
        return sum(unwind.values()) if unwind else float('inf')
    def primary_unwind_value(i):
        unwind = all_unwinds[i]
        if not unwind:
            return -float('inf')  # 保证空的排后面
        min_key = min(unwind, key=lambda k: int(k))
        return unwind[min_key]
    
    if max_pred > 0.5:
        high_conf_indices = [
        i for i in range(len(all_preds))
        if torch.sigmoid(torch.tensor(all_preds[i])).item() > 0.5
        ]
        topk_indices = sorted(
            high_conf_indices,
            key=lambda i: (unwind_sum(i), primary_unwind_value(i)),
            reverse=True
        )[:10]
    
    else:
        topk_indices = sorted(
        range(len(all_preds)),
        key=lambda i: (all_preds[i], all_preds_time[i]),
        reverse=True
        )[:10]

    for i, uw in enumerate(all_unwinds):
        if uw and all(v == 3 for v in uw.values()):
            if i in topk_indices:
                topk_indices.remove(i)
            topk_indices = [i] + topk_indices
            break
    if max_pred < 0.1:
        for i, uw in enumerate(all_unwinds):
            if uw:
                values = list(uw.values())
                if values.count(20) == 1 and values.count(1) == len(values) - 1:
                    if i in topk_indices:
                        topk_indices.remove(i)
                    topk_indices = [i]
                    break
    return topk_indices

def evaluate_topk(model, dataloader, device, top_k=5):
    model.eval()
    all_preds = []
    all_preds_time = []
    all_labels = []
    all_files = []
    all_unwinds = []
    all_times = []
    topk_indices = []

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch_idx, batch in enumerate(loop):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            result_pred = out[:, 0]
            time_pred = out[:, 1]
            result_label, _ = normalise_label(batch.y.to(device))
            exec_times = batch.y[:, 1].tolist()
            result_label = result_label.to(device)

            data_list = batch.to_data_list()
            for i, data in enumerate(data_list):
                all_preds.append(result_pred[i].item())
                all_preds_time.append(time_pred[i].item())
                all_labels.append(result_label[i].item())
                all_files.append(getattr(data, "result_file", f"unknown_{batch_idx}_{i}"))
                all_unwinds.append(json.loads(getattr(data, "unwind_dict", "{}")))
                all_times.append(exec_times[i])

    total_time = 0.0

    topk_indices = s2(all_preds, all_preds_time, all_unwinds)

    # topk_indices = heapq.nlargest(top_k, range(len(all_preds)), key=lambda i: all_preds[i])

    print(f"\nTop {top_k} Predictions:")
    for rank, idx in enumerate(topk_indices, 1):
        pred_prob = torch.sigmoid(torch.tensor(all_preds[idx])).item()
        print(f"[{rank}] File: {all_files[idx]}")
        print(f"     Predicted Score: {pred_prob:.4f}")
        print(f"     Predicted Time:  {(1- all_preds_time[idx]) * 60}")
        print(f"     True Label:      {all_labels[idx]}")
        print(f"     Unwind Factors:  {all_unwinds[idx]}")
        print(f"     Execution Time:  {all_times[idx]}s")
        total_time += all_times[idx]
        if all_labels[idx] == 1:
            print("Label is 1, stopping early and returning.")
            return 1, total_time
    return 0, total_time


def find_results_with_unwind(results, unwind):
    matched = []
    for entry in results:
        unwind_values = [v for k, v in entry.items() if k.isdigit()]
        if all(u == unwind for u in unwind_values):
            matched.append(entry["result"])
    return matched


# Load file paths
for subdir in base_path.iterdir():
    if subdir.is_dir():
        name = subdir.name
        json_file = subdir / f"{name}.json"
        npz_file = subdir / f"{name}.json.npz"
        edge_file = subdir / f"{name}.jsonEdges.npz"
        result_file = result_path / f"{name}.json"

        if (
            json_file.exists()
            and npz_file.exists()
            and edge_file.exists()
            and result_file.exists()
        ):
            loop_tokens.append(str(json_file))
            node_files.append(str(npz_file))
            edge_files.append(str(edge_file))
            result_files.append(str(result_file))

assert len(loop_tokens) == len(node_files) == len(edge_files) == len(result_files), \
    f"Array lengths differ: loop_tokens={len(loop_tokens)}, node_files={len(node_files)}, edge_files={len(edge_files)}, result_files={len(result_files)}"

Bugs_found = 0
Model_Bugs_found = 0
split_idx = floor(len(node_files) * 0.8)

loop_tokens = loop_tokens[split_idx:]
result_files = result_files[split_idx:]
node_files = node_files[split_idx:]
edge_files = edge_files[split_idx:]

# Main processing loop with progress bar
for loop, resultfile, node, edge in tqdm(zip(loop_tokens, result_files, node_files, edge_files),
                                         total=len(loop_tokens), desc="Processing Graph Files"):

    total_time = 0
    unwind = 0

    with open(resultfile, "r", encoding="utf-8") as f:
        data = json.load(f)

    found = False
    for itreation in range(start_bound, upper_bound, step):
        matched_results = find_results_with_unwind(data["results"], itreation)
        for result in matched_results:
            res, time = result
            total_time += time
            if res == 1:
                found = True
                if total_time <= 61.0:
                    Bugs_found += 1
                    unwind = itreation
                break
        if found:
            break

    if not found:
        print(f'{resultfile} NO Bug')
    else:
        print(f'{resultfile} Bug found with unlimited timeout')
    print(f'total_time: {total_time}, unwind: {unwind}')

    dataset = MultiBoundGraphDataset([node], [edge], [loop], [resultfile])
    inputLayerSize = dataset[0].x.size(1)
    model = GATModel(in_channels=inputLayerSize, hidden_channels=inputLayerSize).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    
    # choose random num_sample candidates to ask the model
    dataset_size = len(dataset)
    num_samples = min(num_sample, dataset_size)
    random.seed(42)
    indices = random.sample(range(dataset_size), num_samples)
    val_subset = Subset(dataset, indices)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True, num_workers=4)

    model_result, model_time = evaluate_topk(model, val_loader, device,
                                             top_k=len(range(start_bound, upper_bound, step)))

    if model_result == 1 and model_time <= 61.0:
        Model_Bugs_found += 1

    print(f'model_time: {model_time}')
    print('------------------------------------------')

print(f'ESBMC Baseline bugs found: {Bugs_found}')
print(f'GNN bugs found: {Model_Bugs_found}')