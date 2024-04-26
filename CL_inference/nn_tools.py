import torch
from torch import nn

###############################
def define_MLP_model(mlp_units, representation_len, bn=False, last_bias=False):
    num_inn_layers = len(mlp_units) - 1
    num_units = [representation_len] + mlp_units
    
    layers = []
    for i in range(num_inn_layers):
        layers.append(nn.Linear(num_units[i], num_units[i + 1]))
        if bn:
            layers.append(nn.BatchNorm1d(num_units[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(num_units[-2], num_units[-1], bias=last_bias))
    return nn.Sequential(*layers)