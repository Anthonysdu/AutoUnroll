import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATModel(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_heads=4,
        num_layers=2,
        edge_dim=4,
        dropout_p=0.2,
    ):
        super(GATModel, self).__init__()

        self.dropout_p = dropout_p

        self.edge_type_proj = nn.Linear(1, edge_dim // 2)
        self.unwind_proj = nn.Linear(1, edge_dim // 2)
        self.unwind_attention_layer = nn.Linear(1, 1)

        self.gats = nn.ModuleList(
            [
                GATv2Conv(
                    in_channels if i == 0 else hidden_channels,
                    hidden_channels,
                    heads=num_heads,
                    concat=False,
                    edge_dim=edge_dim,
                )
                for i in range(num_layers)
            ]
        )

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

        unwind_attention = torch.sigmoid(self.unwind_attention_layer(unwind_factor))
        unwind_emb = unwind_emb * unwind_attention

        edge_attr_proj = torch.cat([edge_type_emb, unwind_emb], dim=-1)

        for conv in self.gats:
            x = conv(x, edge_index, edge_attr=edge_attr_proj)
            x = torch.relu(x)

        x = self.pool(x, batch)
        x = torch.relu(self.fc(x))
        x = self.dropout(x)

        result = self.result(x)
        time = self.time(x)

        return torch.cat([result, time], dim=1)
