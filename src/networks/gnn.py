import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool


# class GATModel(nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_heads=4, num_layers=2, edge_dim=6, dropout_p=0.2):
#         super(GATModel, self).__init__()

#         self.dropout_p = dropout_p
#         self.edge_type_proj = nn.Linear(1, edge_dim // 3)
#         self.unwind_proj = nn.Linear(1, edge_dim // 3)
#         self.upperbound_proj = nn.Linear(1, edge_dim // 3)
#         self.unwind_attention_layer = nn.Linear(1, 1)
#         self.gats = nn.ModuleList([
#             GATv2Conv(
#                 in_channels if i == 0 else hidden_channels,
#                 hidden_channels,
#                 heads=num_heads,
#                 concat=False,
#                 edge_dim=edge_dim
#             )
#             for i in range(num_layers)
#         ])

#         self.pool = global_mean_pool
#         self.fc = nn.Linear(hidden_channels, hidden_channels)
#         self.dropout = nn.Dropout(p=dropout_p)
#         self.result = nn.Linear(hidden_channels, 1)
#         self.time = nn.Linear(hidden_channels, 1)

#     def forward(self, x, edge_index, edge_attr, batch):
#         edge_type = edge_attr[:, 0:1]
#         unwind_factor = edge_attr[:, 1:2]
#         upperbound = edge_attr[:, 2:3]
#         # print(upperbound)
#         # print(unwind_factor)
#         edge_type_emb = self.edge_type_proj(edge_type)
#         unwind_emb = self.unwind_proj(unwind_factor)
#         unwind_attention = torch.sigmoid(self.unwind_attention_layer(unwind_factor))
#         unwind_emb = unwind_emb * unwind_attention
#         upperbound_emb = self.upperbound_proj(upperbound)
#         edge_attr_proj = torch.cat([edge_type_emb, unwind_emb, upperbound_emb], dim=-1)
        

#         for conv in self.gats:
#             x = conv(x, edge_index, edge_attr=edge_attr_proj)
#             x = torch.relu(x)

#         x = self.pool(x, batch)
#         x = torch.relu(self.fc(x))
#         result = torch.sigmoid(self.result(x))
#         time = self.time(x)
#         time = torch.sigmoid(self.time(x))

#         return torch.cat([result, time], dim=1)

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads=4, num_layers=2, edge_dim=4, dropout_p=0.2):
        super(GATModel, self).__init__()

        self.dropout_p = dropout_p

        self.edge_type_proj = nn.Linear(1, edge_dim // 2)
        self.unwind_proj = nn.Linear(1, edge_dim // 2)
        self.unwind_attention_layer = nn.Linear(1, 1)

        self.gats = nn.ModuleList([
            GATv2Conv(
                in_channels if i == 0 else hidden_channels,
                hidden_channels,
                heads=num_heads,
                concat=False,
                edge_dim=edge_dim
            )
            for i in range(num_layers)
        ])

        self.pool = global_mean_pool
        self.fc = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(p=dropout_p)
        self.result = nn.Linear(hidden_channels, 1)
        self.time = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_type = edge_attr[:, 0:1]
        unwind_factor = edge_attr[:, 1:2]

        edge_type_emb = self.edge_type_proj(edge_type)
        unwind_emb = self.unwind_proj(unwind_factor)

        # 注意力权重用于调节 unwind 信息
        unwind_attention = torch.sigmoid(self.unwind_attention_layer(unwind_factor))
        unwind_emb = unwind_emb * unwind_attention

        # 拼接得到最终的 edge 特征
        edge_attr_proj = torch.cat([edge_type_emb, unwind_emb], dim=-1)

        for conv in self.gats:
            x = conv(x, edge_index, edge_attr=edge_attr_proj)
            x = torch.relu(x)

        x = self.pool(x, batch)
        x = torch.relu(self.fc(x))
        x = self.dropout(x)

        # result = torch.sigmoid(self.result(x))
        # time = torch.sigmoid(self.time(x))
        result = self.result(x)
        time = self.time(x)

        return torch.cat([result, time], dim=1)
    
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        result, time = output[:, 0], output[:, 1]
        y_result, y_time = cal_label(batch.y)
        loss_result = criterion(result, y_result)
        loss_time = criterion(time, y_time)
        loss = 0.95 * loss_result + 0.05 * loss_time
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            result, time = output[:, 0], output[:, 1]
            y_result, y_time = cal_label(batch.y)
            loss_result = criterion(result, y_result)
            loss_time = criterion(time, y_time)
            loss = 0.8 * loss_result + 0.2 * loss_time
            total_loss += loss.item()

    return total_loss / len(loader)


def cal_label(output):
    result = output[:, 0]
    time = output[:, 1]
    time = torch.clamp(time, max=60)
    n_result = (result + 1) / 2
    n_time = 1 - time / 60
    return n_result, n_time
