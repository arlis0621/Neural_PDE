# model.py
import torch
import torch.nn as nn
import config

def make_mlp(sizes, activation='relu', use_bias=True):
    layers = []
    for i in range(len(sizes)-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1], bias=use_bias))
        if i < len(sizes)-2:
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class DeepONet(nn.Module):
    def __init__(self,
                 m_sensors=config.M_SENSORS,
                 p_latent=config.P_LATENT,
                 branch_hidden=config.BRANCH_HIDDEN,
                 trunk_hidden=config.TRUNK_HIDDEN,
                 activation=config.ACTIVATION,
                 use_bias=config.USE_BIAS):
        super().__init__()
        # Branch: input m -> ... -> p
        branch_sizes = [m_sensors] + branch_hidden + [p_latent]
        self.branch = make_mlp(branch_sizes, activation=activation, use_bias=True)
        # Trunk: input 1 -> ... -> p
        trunk_sizes = [1] + trunk_hidden + [p_latent]
        self.trunk = make_mlp(trunk_sizes, activation=activation, use_bias=True)
        # optional final bias
        if use_bias:
            self.b0 = nn.Parameter(torch.zeros(1))
        else:
            self.b0 = None

    def forward(self, u, y):
        """
        u: (B, m)
        y: (B, 1)
        returns: (B, 1)
        """
        b = self.branch(u)        # (B, p)
        t = self.trunk(y)         # (B, p)
        out = torch.sum(b * t, dim=1, keepdim=True)  # (B,1)
        if self.b0 is not None:
            out = out + self.b0
        return out
