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

    def _apply_ffn(self,
                   x: torch.Tensor,
                   bias_1: torch.Tensor,
                   weight_1: torch.Tensor,
                   weight_2: torch.Tensor):
        #
        x_size = x.size()
        x = x.view(
            [-1, self.in_features]
        )
        prob = self.router(x)
        topk = torch.topk(
            prob, k=self.n_blocks // 4,
            dim=-1, sorted=False
        )
        indices = topk.indices

        #
        y = torch.zeros_like(x)
        for i in range(self.n_blocks):
            cmp = torch.eq(indices, i)
            mask = torch.sum(
                cmp, dim=-1, dtype=torch.bool
            )
            # fc1
            x_i = x[mask]
            b_i, w_i = bias_1[i], weight_1[i]
            h = self.activation(
                torch.addmm(
                    b_i, x_i, w_i.T, beta=1.0, alpha=1.0
                )
            )
            # fc2
            w_i = weight_2[i]
            y[mask] += torch.matmul(h, w_i)
        y += self.fc2.bias.view([1, -1])

        #
        return y.view(x_size)

    def forward(self, x: torch.Tensor):
        bias_1 = self.fc1.bias.view(
            [self.n_blocks, self.block_size]
        )
        weight_1 = self.fc1.weight.view(
            [self.n_blocks, self.block_size, -1]
        )
        weight_2 = self.fc2.weight.view(
            [-1, self.n_blocks, self.block_size]
        )
        weight_2 = torch.permute(
            weight_2, dims=[1, 2, 0]
        )
        return self._apply_ffn(
            x, bias_1=bias_1, weight_1=weight_1,
            weight_2=weight_2.contiguous()
        )
