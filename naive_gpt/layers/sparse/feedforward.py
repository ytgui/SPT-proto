import torch
from torch import nn
from naive_gpt import layers


class RoutedFFN(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_feedforward: int,
                 block_size: int,
                 activation: nn.Module,
                 bias: bool = True):
        nn.Module.__init__(self)
        #
        self.d_model = d_model
        self.block_size = block_size
        self.d_feedforward = d_feedforward
        assert d_feedforward % block_size == 0
        self.n_blocks = d_feedforward // block_size
        #
        self.router = nn.Sequential(
            nn.Linear(d_model, self.n_blocks),
            nn.Softmax(dim=-1)
        )
        self.fc1 = nn.Linear(
            d_model, d_feedforward, bias=bias
        )
        self.fc2 = nn.Linear(
            d_feedforward, d_model, bias=bias
        )
        self.activation = activation

    @staticmethod
    def from_pretrained(block_size: int,
                        source: layers.Feedforward):
        assert isinstance(
            source, layers.Feedforward
        )
        model = RoutedFFN(
            block_size=block_size,
            d_model=source.d_model,
            d_feedforward=source.d_feedforward,
            activation=source.activation,
        )
        output = model.load_state_dict(
            source.state_dict(), strict=False
        )
        if len(output.missing_keys) != 2:
            raise RuntimeError
        return model

    def _apply_ffn(self,
                   x: torch.Tensor,
                   bias_1: torch.Tensor,
                   weight_1: torch.Tensor,
                   weight_2: torch.Tensor):
        #
        x_size = x.size()
        x = x.view(
            [-1, self.d_model]
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
