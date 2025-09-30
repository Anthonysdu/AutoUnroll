def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 2],
        default=1,
        help="Encoding mode for unroll factor (1: add to edge_attr; 2: add to node features).",
    )
    parser.add_argument(
        "--no-full-encode",
        action="store_false",
        help="Disable encoding unwind factors for edges/nodes inside loops."
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=None,
        help="Input feature dimension for unroll factor. "
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size during training. Default to 2",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=20,
        help="nums of epochs during training. Default to 20",
    )
    parser.add_argument(
        "--edges",
        type=int,
        nargs="+",
        choices=[0, 1, 2],
        default=[0],
        help="Edge(s) to encode: 0=AST, 1=ICFG, 2=DATA. Can select multiple, e.g. 0 2",
    )
    parser.add_argument(
        "--result-weight",
        type=lambda x: (
            float(x)
            if 0.0 <= float(x) <= 1.0
            else argparse.ArgumentTypeError(f"result_weight {x} not in range [0, 1]")
        ),
        default=0.9,
        help="Hyperparameter for setting result loss weight (range [0,1]). "
        "Default=0.9 for result label, 0.1 for time weight.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model.pth",
        help="Filename to save the best model. Default='model.pth'."
    )

    parser.add_argument(
        "--gradnorm",
        action="store_true",
        help="Use gradnorm to adjust loss weight."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=0,
        help="GPU device ID for traning. Currently just supports single device",
    )
    return parser


args = get_args().parse_args()
if args.input_dim is None:
    if args.mode == 2:
        args.input_dim = 16
    else:
        args.input_dim = 0
num_epochs = args.epoch
batch_size = args.batch_size
result_weight = args.result_weight
sample_set = []
unwind_dim = args.input_dim
mode = args.mode
grad_norm = args.gradnorm
full_encode = args.no_full_encode
device_ID = args.device

from collections import defaultdict
import json
import os
from src.networks.utils.utils import MultiBoundGraphDataset
from src.networks.gnn import GATModel
import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
import gc
import random

device = torch.device(f"cuda:{device_ID}" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")

base_path = Path("data/final_graphs")
result_path = Path("json_result")

loop_tokens = []
node_files = []
edge_files = []
result_files = []

class GradNorm:
    def __init__(self, model, alpha=1.5):
        self.alpha = alpha
        # 记录每个任务的初始损失
        self.initial_losses = None
        self.model = model
        # 每个任务的权重，初始化为 1
        self.task_weights = torch.nn.Parameter(torch.ones(2, device=device))

    def compute_gradnorm_loss(self, losses):
        """
        losses: list of torch scalar losses [loss_task1, loss_task2]
        """
        if self.initial_losses is None:
            self.initial_losses = torch.tensor([l.item() for l in losses], device=device)

        weighted_losses = [self.task_weights[i] * losses[i] for i in range(len(losses))]
        total_loss = sum(weighted_losses)

        # 计算任务的梯度范数
        G_list = []
        for i, l in enumerate(weighted_losses):
            grads = torch.autograd.grad(l, self.model.parameters(), retain_graph=True, create_graph=True)
            G = torch.cat([g.view(-1) for g in grads]).norm()
            G_list.append(G)

        G_avg = sum(G_list) / len(G_list)
        # 计算任务的相对比例
        r_list = [losses[i].item() / self.initial_losses[i].item() for i in range(len(losses))]
        target_G_list = [G_avg * (r ** self.alpha) for r in r_list]

        gradnorm_loss = sum(torch.abs(G_list[i] - target_G_list[i]) for i in range(len(losses)))
        return total_loss, gradnorm_loss


def compute_global_weights_from_json(result_files):
    labels = []
    for file in result_files:
        with open(file, "r") as f:
            data = json.load(f)
            for item in data.get("results", []):
                label = item.get("result", [0])[0]
                if label == -1:
                    label = 0
                labels.append(label)

    labels = torch.tensor(labels, dtype=torch.float32)
    num_pos = (labels == 1).sum().item()
    num_neg = (labels == 0).sum().item()

    # Avoid division by zero
    if num_pos == 0:
        pos_weight = torch.tensor([1.0])
    else:
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)

    print(f"Positive samples: {num_pos}, Negative samples: {num_neg}")
    print(f"pos_weight for BCEWithLogitsLoss: {pos_weight.item()}")

    return pos_weight


def normalise_label(output):
    result = output[:, 0]
    time = output[:, 1]
    time = torch.clamp(time, max=60)
    n_result = (result == 1).float()
    n_time = 1 - time / 60
    return n_result, n_time


def select_key_samples_balanced(data, max_middle=19, seed=None):
    if seed is not None:
        random.seed(seed)

    results = data["results"]
    total_len = len(results)

    if total_len <= 21:
        return results

    first_bug_idx = next(
        (i for i, r in enumerate(results) if r["result"][0] == 1), None
    )
    non_bug_indices = [i for i, r in enumerate(results) if r["result"][0] != 1]
    last_non_bug_idx = non_bug_indices[-1] if non_bug_indices else None

    key_indices = set()
    if first_bug_idx is not None:
        key_indices.add(first_bug_idx)
    if last_non_bug_idx is not None:
        key_indices.add(last_non_bug_idx)

    remaining_indices = [i for i in range(len(results)) if i not in key_indices]
    label_1_indices = [i for i in remaining_indices if results[i]["result"][0] == 1]
    label_non1_indices = [i for i in remaining_indices if results[i]["result"][0] != 1]

    half_max = max_middle // 2
    sampled_1 = random.sample(label_1_indices, min(len(label_1_indices), half_max))
    sampled_non1 = random.sample(
        label_non1_indices, min(len(label_non1_indices), max_middle - len(sampled_1))
    )

    selected_indices = list(key_indices) + sampled_1 + sampled_non1
    selected_indices = sorted(selected_indices)
    selected_results = [results[i] for i in selected_indices]
    return selected_results


def count_results_in_dir(directory):
    results_summary = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"[Skip] {filename} not valid JSON")
                    continue

            counts = {-1: 0, 0: 0, 1: 0}
            for entry in data.get("results", []):
                result = entry.get("result")
                if isinstance(result, list) and len(result) > 0:
                    val = result[0]
                    if val in counts:
                        counts[val] += 1

            total = sum(counts.values())
            if total > 0:
                percents = {k: counts[k] / total * 100 for k in counts}
            else:
                percents = {k: 0 for k in counts}

            results_summary.append((filename, counts, percents, total))

    results_summary.sort(key=lambda x: x[2][1])

    for filename, counts, percents, total in results_summary:
        if percents[1] > 0:
            sample_set.append(filename)


def train(model, dataloader, optimizer, criterion_class, criterion_reg, gradnorm, device):
    model.train()
    total_loss = 0

    loop = tqdm(dataloader, desc="Training", leave=False)
    for batch in loop:
        batch = batch.to(device)
        optimizer.zero_grad()
        if mode == 1:
            out = model(batch.x, [], batch.edge_index, batch.edge_attr, batch.batch)
        else:
            out = model(batch.x, batch.node_unwind, batch.edge_index, batch.edge_attr, batch.batch)
        result_pred = out[:, 0]
        time_pred = out[:, 1]
        result_label, time_label = normalise_label(batch.y.to(device))
        result_label = result_label.to(device)
        time_label = time_label.to(device)
        loss_result = criterion_class(result_pred, result_label).to(device)
        loss_time = criterion_reg(time_pred, time_label).to(device)

        if gradnorm:
            total_task_loss, gradnorm_loss = gradnorm.compute_gradnorm_loss([loss_result, loss_time])
            loss = total_task_loss + gradnorm_loss
        else:
            loss = result_weight * loss_result + (1 - result_weight) * loss_time
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion_class, criterion_reg, device):
    model.eval()
    total_loss = 0
    result_file_groups = defaultdict(list)

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch_idx, batch in enumerate(loop):
            batch = batch.to(device)
            if mode == 1:
                out = model(batch.x, [], batch.edge_index, batch.edge_attr, batch.batch)
            else:
                out = model(batch.x, batch.node_unwind, batch.edge_index, batch.edge_attr, batch.batch)
            result_pred = out[:, 0]
            time_pred = out[:, 1]

            result_label, time_label = normalise_label(batch.y.to(device))
            result_label = result_label.to(device)
            time_label = time_label.to(device)
            loss_result = criterion_class(result_pred, result_label).to(device)
            loss_time = criterion_reg(time_pred, time_label).to(device)
            loss = result_weight * loss_result + (1 - result_weight) * loss_time

            total_loss += loss.item()

            data_list = batch.to_data_list()
            for i, data in enumerate(data_list):
                result_file = getattr(data, "result_file", f"unknown_{batch_idx}_{i}")
                result_file_groups[result_file].append(
                    {
                        "result_pred": result_pred[i].item(),
                        "result_label": result_label[i].item(),
                        "time_pred": time_pred[i].item(),
                        "time_label": time_label[i].item(),
                    }
                )

            loop.set_postfix(loss=loss.item())

    correct = 0
    total = 0
    for result_file, results in result_file_groups.items():
        labels = [item["result_label"] for item in results]
        preds = [item["result_pred"] for item in results]

        print(f"\nResult file: {result_file}")
        for i, (p, l) in enumerate(zip(preds, labels)):
            prob = torch.sigmoid(torch.tensor(p)).item()
            print(f"  Instance {i}: Pred = {prob:.4f}, Label = {l}")

        if 1 in labels:
            total += 1
            max_index = preds.index(max(preds))
            if labels[max_index] == 1:
                correct += 1

    print(f"\nBugs found (correct predictions): {correct}")
    print(f"Bugs missed: {total - correct}")

    return total_loss / len(dataloader)


count_results_in_dir(result_path)

for subdir in base_path.iterdir():
    if subdir.is_dir() and f"{subdir.name}.json" in sample_set:
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
            with open(result_file, "r") as f:
                data = json.load(f)
            sampled_results = select_key_samples_balanced(data, max_middle=19, seed=42)
            data["results"] = sampled_results
            tmp_result_file = result_path / f"{name}_sampled.json"
            with open(tmp_result_file, "w") as f:
                json.dump(data, f)
            loop_tokens.append(str(json_file))
            node_files.append(str(npz_file))
            edge_files.append(str(edge_file))
            result_files.append(str(tmp_result_file))

assert (
    len(loop_tokens) == len(node_files) == len(edge_files) == len(result_files)
), f"Array lengths differ: loop_tokens={len(loop_tokens)}, node_files={len(node_files)}, edge_files={len(edge_files)}, result_files={len(result_files)}"

total_samples = len(node_files)
indices = list(range(total_samples))
random.seed(42)
random.shuffle(indices)

train_end = int(0.6 * total_samples)
val_end = int(0.8 * total_samples)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

train_dataset = MultiBoundGraphDataset(
    node_files=[node_files[i] for i in train_idx],
    edge_files=[edge_files[i] for i in train_idx],
    loop_tokens=[loop_tokens[i] for i in train_idx],
    result_files=[result_files[i] for i in train_idx],
    mode=mode,
    full_encode=full_encode,
    edges = args.edges,
)

val_dataset = MultiBoundGraphDataset(
    node_files=[node_files[i] for i in val_idx],
    edge_files=[edge_files[i] for i in val_idx],
    loop_tokens=[loop_tokens[i] for i in val_idx],
    result_files=[result_files[i] for i in val_idx],
    mode=mode,
    full_encode=full_encode,
    edges = args.edges,
)

test_dataset = MultiBoundGraphDataset(
    node_files=[node_files[i] for i in test_idx],
    edge_files=[edge_files[i] for i in test_idx],
    loop_tokens=[loop_tokens[i] for i in test_idx],
    result_files=[result_files[i] for i in test_idx],
    mode=mode,
    full_encode=full_encode,
    edges = args.edges,
)

print(
    f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
)
print("calulating weight...")
train_result_files = [result_files[i] for i in train_idx]
pos_weight = compute_global_weights_from_json(train_result_files)
pos_weight = pos_weight.to(device)
inputLayerSize = train_dataset[0].x.size(1) + unwind_dim
model = GATModel(in_channels=inputLayerSize, hidden_channels=inputLayerSize, mode = mode).to(device)
loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)

criterion_class = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion_reg = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-5)
best_val_loss = float("inf")

if grad_norm:
    gradnorm = GradNorm(model, alpha=1.5)
else:
    gradnorm = None
for epoch in range(1, num_epochs + 1):
    print(f"\n Epoch {epoch}/{num_epochs}")

    train_loss = train(model, loader, optimizer, criterion_class, criterion_reg, gradnorm, device)
    torch.cuda.empty_cache()
    gc.collect()
    val_loss = evaluate(model, val_loader, criterion_class, criterion_reg, device)
    print("Valset accuracy:")
    print(f" Train Loss: {train_loss:.4f} |  Val Loss: {val_loss:.4f}")
    #scheduler.step(val_loss)

    # save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_file = f"{args.model}.pth"
        torch.save(model.state_dict(), model_file)
        print(f"Saved best model as {model_file}!")

print("\n=== Testing on test set ===")
test_loss = evaluate(model, test_loader, criterion_class, criterion_reg, device)
print(f"Test Loss: {test_loss:.4f}")
