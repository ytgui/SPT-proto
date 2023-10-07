import torch
from torch import nn


class Feedforward(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_feedforward: int,
                 p_dropout: float,
                 activation: nn.Module = nn.SiLU()):
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


class ConvFeedforward(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_feedforward: int,
                 p_dropout: float,
                 activation: nn.Module = nn.SiLU()):
        nn.Module.__init__(self)
        #
        self.fc = nn.Sequential(
            nn.Conv2d(
                d_model, d_feedforward,
                kernel_size=3, stride=1, padding=1
            ),
            nn.Dropout2d(p=p_dropout),
            activation,
            nn.Conv2d(
                d_feedforward, d_model,
                kernel_size=3, stride=1, padding=1
            )
        )

    def forward(self, x: torch.Tensor):
        return self.fc(x)
