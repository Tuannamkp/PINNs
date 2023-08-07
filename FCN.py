import torch
import torch.nn as nn
class FCN(nn.Module):
    def __init__(self, Input_layers,Output_layers,Hidden_layers,Layers) :
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                            nn.Linear(Input_layers,Hidden_layers),
                            activation()])
        self.fch =nn.Sequential(*[
                            nn.Sequential(*[nn.Linear(Hidden_layers,Hidden_layers),
                            activation()]) for _ in range(Layers-1)])
        self.fce = nn.Linear(Hidden_layers,Output_layers)
    def forward(self,x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


