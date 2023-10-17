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
        self.d_model = d_model
        self.d_feedforward = d_feedforward
        self.activation = activation
        self.p_dropout = p_dropout
        #
        self.fc1 = nn.Linear(
            d_model, d_feedforward
        )
        self.fc2 = nn.Linear(
            d_feedforward, d_model
        )
        self.dropout = nn.Dropout(
            p=p_dropout
        )
        self.activation = activation

    def forward(self, x: torch.Tensor):
        h = self.fc1(x)
        h = self.activation(
            self.dropout(h)
        )
        return self.fc2(h)


class LLaMaFeedforward(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_feedforward: int,
                 activation: nn.Module):
        nn.Module.__init__(self)
        #
        self.d_model = d_model
        self.d_feedforward = d_feedforward
        self.activation = activation
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
