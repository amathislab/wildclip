# 
# WildCLIP by Gabeff et al.
# © ECEO and A. Mathis Lab
# https://github.com/amathislab/wildclip
# 
# Licensed under GNU Lesser General Public License v3.0
# 

from collections import OrderedDict
from typing import List

import torch
from torchvision.ops import MLP


def add_MLP_head(
    model: "torch.nn.Module",
    embed_dim: int,
    num_out_classes: int,
    hidden_layers_dim: List,
) -> "torch.nn.Module":
    in_features = embed_dim
    print(in_features)

    mlp = MLP(
        in_channels=in_features, hidden_channels=[*hidden_layers_dim, num_out_classes], dropout=0.2, inplace=False
    )
    model_with_MLP = torch.nn.Sequential(OrderedDict([("backbone", model), ("head", mlp)]))

    return model_with_MLP
