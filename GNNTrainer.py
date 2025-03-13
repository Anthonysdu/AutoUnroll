import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import json
import time
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler


# GAT Model definition
class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads=4):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.conv2 = GATConv(
            hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True
        )

        self.result = nn.Linear(hidden_channels * num_heads, 1)
        self.time = nn.Linear(hidden_channels * num_heads, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        x = global_mean_pool(x, batch)
        result = self.result(x)
        time = self.time(x)

        return torch.cat([result, time], dim=1)  # Shape: [batch_size, 2]


def load_graph_data(node_file, edge_file):
    node_data = np.load(node_file)
    x = torch.tensor(
        node_data["node_rep"], dtype=torch.float
    )  # node feature (one-hot encoding)

    edge_data = np.load(edge_file)
    ast_edges = edge_data["AST"]
    data_edges = edge_data["Data"]
    icfg_edges = edge_data["ICFG"]
    lcfg_edges = edge_data["LCFG"]

    ast_edges = torch.tensor(ast_edges, dtype=torch.long).t().contiguous()
    data_edges = torch.tensor(data_edges, dtype=torch.long).t().contiguous()
    icfg_edges = torch.tensor(icfg_edges, dtype=torch.long).t().contiguous()

    return x, ast_edges, data_edges, icfg_edges, lcfg_edges


def load_labels(label_file):
    with open(label_file, "r") as f:
        label_data = json.load(f)

    labels_dict = {}
    for item in label_data:
        program_name = item["program_name"]
        labels = item["labels"]
        labels_dict[program_name] = labels

    return labels_dict


def prepare_data(node_file, edge_file, label_file):
    x, ast_edges, data_edges, icfg_edges, lcfg_edges = load_graph_data(
        node_file, edge_file
    )
    labels_dict = load_labels(label_file)
    file_name = os.path.basename(node_file)
    program_name, _ = os.path.splitext(file_name)
    labels = labels_dict.get(program_name.replace(".json", ""), [])
    data_list = []
    for i, lcfg_edge in enumerate(lcfg_edges):
        lcfg_edge_index = torch.tensor(lcfg_edge, dtype=torch.long)

        if lcfg_edge_index.dim() == 1:
            lcfg_edge_index = lcfg_edge_index.unsqueeze(0).repeat(2, 1)

        lcfg_edge_index = lcfg_edge_index.contiguous()

        edge_list = [ast_edges, data_edges, icfg_edges, lcfg_edge_index]
        combined_edge_index = torch.cat(edge_list, dim=1)

        if i < len(labels):
            label = torch.tensor(labels[i], dtype=torch.float)

            label = label.unsqueeze(0) if label.dim() == 1 else label
            data = Data(x=x, edge_index=combined_edge_index, y=label)
            data_list.append(data)

    return data_list


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch.x, batch.edge_index, batch.batch)
        result, time = output[:, 0], output[:, 1]
        y_result, y_time = cal_label(batch.y)
        loss_result = criterion(result, y_result)
        loss_time = criterion(time, y_time)
        loss = 0.8 * loss_result + 0.2 * loss_time
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def cal_label(output):
    result = output[:, 0]
    time = output[:, 1]
    time = torch.clamp(time, max=60)
    n_result = (result + 1) / 2
    n_time = 1 - time / 60
    return n_result, n_time


def evaluate(model, loader, criterion, device, rank, world_size):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)
            result, time = output[:, 0], output[:, 1]
            y_result, y_time = cal_label(batch.y)
            loss_result = criterion(result, y_result)
            loss_time = criterion(time, y_time)
            loss = 0.8 * loss_result + 0.2 * loss_time
            total_loss += loss.item()

    return total_loss / len(loader)


def load_all_data(label_file):
    labels_dict = load_labels(label_file)
    all_data_list = []
    for program_name in labels_dict.keys():
        node_file = f"../../data/final_graphs/{program_name}.json.npz"
        edge_file = f"../../data/final_graphs/{program_name}.jsonEdges.npz"
        if os.path.exists(node_file) and os.path.exists(edge_file):
            data_list = prepare_data(node_file, edge_file, label_file)
            all_data_list.extend(data_list)
    return all_data_list


def load_data_in_batches(label_file, batch_size, rank, world_size):
    labels_dict = load_labels(label_file)
    program_names = list(labels_dict.keys())
    program_names = program_names[rank::world_size]

    for i in range(0, len(program_names), batch_size):
        batch_program_names = program_names[i : i + batch_size]
        batch_data_list = []

        for program_name in batch_program_names:
            node_file = f"../../data/final_graphs/{program_name}.json.npz"
            edge_file = f"../../data/final_graphs/{program_name}.jsonEdges.npz"
            if os.path.exists(node_file) and os.path.exists(edge_file):
                data_list = prepare_data(node_file, edge_file, label_file)
                batch_data_list.extend(data_list)

        yield batch_data_list


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main_worker(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    model = GATModel(in_channels=92, hidden_channels=64).to(device)
    model = DDP(model, device_ids=[rank])  # Wrap model in DDP

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_label_file = "../../data/trainset.json"
    val_label_file = "../../data/valset.json"
    test_label_file = "../../data/testset.json"
    batch_size = 8
    epochs = 10

    # Training loop
    for epoch in range(epochs):
        train_loss = 0
        for train_data_list in load_data_in_batches(
            train_label_file, batch_size, rank, world_size
        ):
            if len(train_data_list) == 0:
                continue
            train_sampler = DistributedSampler(
                train_data_list, num_replicas=world_size, rank=rank, shuffle=True
            )
            train_sampler.set_epoch(epoch)
            train_loader = DataLoader(
                train_data_list, batch_size=2, shuffle=False, sampler=train_sampler
            )
            current_device = next(model.parameters()).device
            start_time = time.time()
            train_loss += train(model, train_loader, optimizer, criterion, device)
            end_time = time.time()
            epoch_duration = end_time - start_time
            print(
                f"Epoch {epoch+1}/{epochs}, Cal Train Loss: {train_loss:.4f} , Duration: {epoch_duration:.2f} seconds, "
                f"Using GPU: {current_device}"
            )
            torch.cuda.empty_cache()
        if len(train_data_list) > 0:
            train_loss /= len(train_loader)
        else:
            train_loss = 0
        print(f"Epoch {epoch+1}/{epochs}, Avg Train Loss: {train_loss:.4f}")

        val_loss = 0
        for val_data_list in load_data_in_batches(
            val_label_file, batch_size, rank, world_size
        ):
            val_sampler = DistributedSampler(
                val_data_list, num_replicas=world_size, rank=rank, shuffle=False
            )
            val_loader = DataLoader(val_data_list, batch_size=2, sampler=val_sampler)
            val_loss += evaluate(model, val_loader, criterion, device, rank, world_size)
            torch.cuda.empty_cache()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Val Loss: {val_loss:.4f}")

    # Test loop
    test_loss = 0
    for test_data_list in load_data_in_batches(
        test_label_file, batch_size, rank, world_size
    ):
        test_sampler = DistributedSampler(
            test_data_list, num_replicas=world_size, rank=rank, shuffle=False
        )
        test_loader = DataLoader(test_data_list, batch_size=2, sampler=test_sampler)
        test_loss += evaluate(model, test_loader, criterion, device, rank, world_size)
        torch.cuda.empty_cache()
    test_loss /= len(test_loader)
    print(f"Avg Test Loss: {test_loss:.4f}")
    # dist.barrier()
    # Save model
    if rank == 0:
        print("Saving model...")
        torch.save(model.state_dict(), "trained_gat_model.pth")
        print("Model saved to trained_gat_model.pth")
        cleanup()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.cuda.empty_cache()

    world_size = 4
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
