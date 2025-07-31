import torch
import torch.nn as nn

# 假设你的模型是 model
model = GATModel(in_channels=inputLayerSize, hidden_channels=inputLayerSize).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))

# 总参数数量
total_params = sum(p.numel() for p in model.parameters())
# 可训练参数数量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
