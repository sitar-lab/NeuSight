import torch.nn as nn
import torch.nn.functional as F
from ..model import ModelBase
from .transformer_block import TransformerBlock
import torch

class MLPBlock(nn.Module):
    def __init__(self, layers, layer_size, dropout_rate, act):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layers = nn.ModuleList()

        for idx in range(layers):
            self.layers.append(nn.Linear(layer_size, layer_size))
            self.layers.append(act)
            self.layers.append(self.dropout)

    def forward(self, x, kname=None, culib=None):
        for layer in self.layers:
            x = layer(x) 

        return x

class HABITATLINEAR(ModelBase):
    def __init__(self, config, tag, device):
        super().__init__(config,tag=tag,device=device) # name and feature

        self.lr = self.config["lr"]
        self.train_batch = int(self.config["train_batch"])
        self.val_batch = int(self.config["val_batch"])
        self.loss = self.config["loss"]

        self.arch = self.config["arch"]

        if self.arch == "MLP":
            self.layer_size = config["hidden_size"]
            self.num_layers = config["num_layers"]
            self.dropout_rate = config["dropout_rate"]

            self.sigmoid = nn.Sigmoid()
            self.leakyrelu = nn.LeakyReLU()
            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()
            self.selu = nn.SELU()

            act_dict = {
                "sigmoid" : self.sigmoid,
                "leakyrelu" : self.leakyrelu,
                "relu" : self.relu,
                "tanh" : self.tanh,
                "selu" : self.selu,
            }
            self.act = config["act"]
            self.act = act_dict[self.act]

            self.fc1 = nn.Linear(len(self.features), self.layer_size)
            self.mlp = MLPBlock(self.num_layers-2, self.layer_size, self.dropout_rate, self.act)
            self.fc2 = nn.Linear(self.layer_size, 1)

        elif self.arch == "TRANS":
            num_input = len(self.features)
            self.net = TransformerBlock(config, num_inputs=num_input, num_output=1)
        else:
            assert(0)

    def forward(self, opname, x, device=None, kname=None, culib=None):

        # "B", "M", "N", "K", "Mem_Bw", "Dev_Mem", "Num_Sm", "SingleFLOPs"

        mean = [1, 2010.042720734127, 8634.46595982143, 8678.46791294643, 740.0004030257936, 22.212208581349206, 64.8203125, 11380.213944692461]
        mean = torch.Tensor(mean).to(self.device)
        std = [1, 2151.465458085419, 14334.074093649471, 14342.228195426867, 483.3678506338057, 11.87851808195382, 26.112328172121956, 4882.489567616248]
        std = torch.Tensor(std).to(self.device)
        x = (x-mean) / std

        if self.arch == "MLP":
            x = self.fc1(x)
            x = self.relu(x)
            x = self.mlp(x)
            x = self.fc2(x)
            return x
        elif self.arch == "TRANS":
            return self.net(x)
        else:
            assert(0)
            