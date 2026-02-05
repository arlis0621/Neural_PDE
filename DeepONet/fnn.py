# fnn.py
import torch
import torch.nn as nn

#Simple fully-connected feed-forward network (regressor).
#Input: concatenated [sensor_values, trunk_input] -> shape (B, input_dim)
#Output: scalar (B, 1)
class FNN(nn.Module):

    def __init__(self, input_dim, hidden=(256, 256), out_dim=1, activation=nn.ReLU, dropout=0.0):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predict(self, x, device="cpu"):
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device=device)
            else:
                x = x.to(device).float()
            out = self.forward(x).cpu().numpy()
        return out