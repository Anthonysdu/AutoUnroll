from collections import defaultdict
import json
from src.networks.utils.utils import MultiBoundGraphDataset
from src.networks.gnn import GATModel
import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.loader import DataLoader
from pathlib import Path
from tqdm import tqdm
from math import floor
import gc

num_epochs = 20
batch_size = 4
result_weight = 0.9

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")

base_path = Path("data/final_graphs")
result_path = Path("json_result")

loop_tokens = []
node_files = []
edge_files = []
result_files = []


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


def train(model, dataloader, optimizer, criterion_class, criterion_reg, device):
    model.train()
    total_loss = 0

    loop = tqdm(dataloader, desc="Training", leave=False)
    for batch in loop:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )  # [batch_size, 2]
        result_pred = out[:, 0]
        time_pred = out[:, 1]
        result_label, time_label = normalise_label(batch.y.to(device))
        result_label = result_label.to(device)
        time_label = time_label.to(device)
        loss_result = criterion_class(result_pred, result_label).to(device)
        loss_time = criterion_reg(time_pred, time_label).to(device)
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
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

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


for subdir in base_path.iterdir():
    if subdir.is_dir() and subdir.name.startswith("pals_lcr"):
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

assert (
    len(loop_tokens) == len(node_files) == len(edge_files) == len(result_files)
), f"Array lengths differ: loop_tokens={len(loop_tokens)}, node_files={len(node_files)}, edge_files={len(edge_files)}, result_files={len(result_files)}"

print(len(node_files))
split_idx = floor(len(node_files) * 0.8)
print(len(node_files[:split_idx]))
print(len(node_files[split_idx:]))

train_dataset = MultiBoundGraphDataset(
    node_files=node_files[:split_idx],
    edge_files=edge_files[:split_idx],
    loop_tokens=loop_tokens[:split_idx],
    result_files=result_files[:split_idx],
)

val_dataset = MultiBoundGraphDataset(
    node_files=node_files[split_idx:],
    edge_files=edge_files[split_idx:],
    loop_tokens=loop_tokens[split_idx:],
    result_files=result_files[split_idx:],
)
print("calulating weight...")
pos_weight = compute_global_weights_from_json(result_files[:split_idx]).to(device)
inputLayerSize = train_dataset[0].x.size(1)
model = GATModel(in_channels=inputLayerSize, hidden_channels=inputLayerSize).to(device)
loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)
train_val_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)
criterion_class = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion_reg = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
best_val_loss = float("inf")

for epoch in range(1, num_epochs + 1):
    print(f"\n Epoch {epoch}/{num_epochs}")

    train_loss = train(model, loader, optimizer, criterion_class, criterion_reg, device)
    torch.cuda.empty_cache()
    gc.collect()
    val_loss = evaluate(model, val_loader, criterion_class, criterion_reg, device)
    print("Valset accuracy:")
    print(f" Train Loss: {train_loss:.4f} |  Val Loss: {val_loss:.4f}")

    # save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "model2.pth")
        print(" Saved best model!")
