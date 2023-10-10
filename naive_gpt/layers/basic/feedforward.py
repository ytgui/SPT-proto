import torch
from torch import nn


class Feedforward(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_feedforward: int,
                 p_dropout: float,
                 activation: nn.Module):
        nn.Module.__init__(self)
        #
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.Dropout(p_dropout),
            activation,
            nn.Linear(d_feedforward, d_model)
        )

    def forward(self, x: torch.Tensor):
        return self.fc(x)


class LLaMaFeedforward(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_feedforward: int,
                 p_dropout: float,
                 activation: nn.Module):
        nn.Module.__init__(self)
        #
        self.gate = nn.Linear(
            d_model, d_feedforward, bias=False
        )
        self.side = nn.Linear(
            d_model, d_feedforward, bias=False
        )
        self.down = nn.Linear(
            d_feedforward, d_model, bias=False
        )
        self.activation = activation

    def forward(self, x: torch.Tensor):
        return self.down(
            self.activation(self.gate(x)) * self.side(x)
        )
