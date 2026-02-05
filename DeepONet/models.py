# models.py
import torch
import torch.nn as nn


class BranchNet(nn.Module):
    def __init__(self, m, hidden, out_dim, activation=nn.ReLU):
        super().__init__()
        layers = []
        in_dim = m
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x shape (B, m)
        return self.net(x)  # (B, out_dim)


class TrunkNet(nn.Module):
    def __init__(self, dim_x, hidden, out_dim, activation=nn.ReLU):
        super().__init__()
        layers = []
        in_dim = dim_x
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x shape (B, dim_x)
        return self.net(x)  # (B, out_dim)


#DeepONet: branch(sensor_values) -> b (p-dim),trunk(location) -> t (p-dim) output = <b, t> + b0 (broadcast)


class DeepONet(nn.Module):
    

    def __init__(
        self,
        m,
        dim_x,
        branch_hidden=[128, 128],
        trunk_hidden=[128, 128],
        p=100,
        activation=nn.ReLU,
        use_bias=True,
    ):
        super().__init__()
        self.branch = BranchNet(m, branch_hidden, p, activation)
        self.trunk = TrunkNet(dim_x, trunk_hidden, p, activation)
        self.use_bias = use_bias
        if use_bias:
            self.b0 = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("b0", None)

    def forward(self, sensor_vals, x):
        b = self.branch(sensor_vals)  # (B, p)
        t = self.trunk(x)  # (B, p)
        # elementwise multiply then sum along p
        prod = b * t
        out = torch.sum(prod, dim=1, keepdim=True)  # (B, 1)
        if self.use_bias:
            out = out + self.b0
        return out

    def predict_from_parts(self, sensor_vals, x):
        self.eval()
        with torch.no_grad():
            return self.forward(sensor_vals, x)
