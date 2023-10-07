import time
import torch
from torch import nn


class RoutedFFN(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 block_size: int,
                 actication: nn.Module,
                 bias: bool = True):
        nn.Module.__init__(self)
        #
        self.block_size = block_size
        self.in_features = in_features
        self.out_features = out_features
        assert out_features % block_size == 0
        self.n_blocks = out_features // block_size
        #
        self.router = nn.Sequential(
            nn.Linear(in_features, self.n_blocks),
            nn.Softmax(dim=-1)
        )
        self.fc1 = nn.Linear(
            in_features, out_features, bias=bias
        )
        self.fc2 = nn.Linear(
            out_features, in_features, bias=bias
        )
        self.activation = actication

    def forward(self, x: torch.Tensor):
        x_size = x.size()
        x = x.view(
            [-1, self.in_features]
        )
        prob = self.router(x)
        topk = torch.topk(
            prob, k=self.n_blocks // 4,
            dim=-1, sorted=False
        )
        indices = topk.indices.tolist()

        # grouping
        grouping: list[list] = [
            [] for _ in range(self.n_blocks)
        ]
        for b, items in enumerate(indices):
            for expert in items:
                grouping[expert].append(b)

        # fc1
        h = torch.zeros(
            [x.size(0), self.n_blocks, self.block_size],
            dtype=x.dtype, device=x.device
        )
        for i, batches in enumerate(grouping):
            if not batches:
                continue
            #
            x_i = x[batches]
            h_i = h[batches, i]
            w_i = self.fc1.weight[
                i * self.block_size:(i + 1) * self.block_size
            ]
            h[batches, i] = torch.addmm(
                h_i, x_i, w_i.T, beta=1.0, alpha=1.0
            )
        h = self.activation(h)

        # fc2
        y = torch.zeros_like(x)
        for i, batches in enumerate(grouping):
            if not batches:
                continue
            #
            y_i = y[batches]
            h_i = h[batches, i]
            w_i = self.fc2.weight[
                :, i * self.block_size:(i + 1) * self.block_size
            ]
            y[batches] = torch.addmm(
                y_i, h_i, w_i.T, beta=1.0, alpha=1.0
            )

        #
        return y.view(x_size)
