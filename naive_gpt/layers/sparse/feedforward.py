import torch
from torch import nn, autograd


class RoutedLinearRow(autograd.Function):
    @staticmethod
    def forward(ctx,
                x: torch.Tensor,
                bias: torch.Tensor,
                weight: torch.Tensor,
                grouping: list[list]):
        # check
        assert x.dim() == 2
        assert bias.dim() == 2
        assert weight.dim() == 3
        assert x.size(-1) == weight.size(-1)
        block_size = weight.size(1)
        n_blocks = weight.size(0)

        # fc1
        y = torch.zeros(
            [x.size(0), n_blocks, block_size],
            dtype=x.dtype, device=x.device
        )
        for i, batches in enumerate(grouping):
            if not batches:
                continue
            #
            x_i = x[batches]
            b_i, w_i = bias[i], weight[i]
            y[batches, i] = torch.addmm(
                b_i, x_i, w_i.T, beta=1.0, alpha=1.0
            )

        #
        ctx.grouping = grouping
        ctx.save_for_backward(x, bias, weight)
        return y

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        # ctx
        x, bias, weight = ctx.saved_tensors
        n_blocks = weight.size(0)
        grouping = ctx.grouping

        # weight
        grad_x = torch.zeros_like(x)
        grad_bias = torch.zeros_like(bias)
        grad_weight = torch.empty_like(weight)
        for i, batches in enumerate(grouping):
            if not batches:
                continue
            #
            x_i, w_i = x[batches], weight[i]
            grad_output_i = grad_output[batches, i]
            grad_x[batches] += torch.matmul(grad_output_i, w_i)
            grad_weight[i] = torch.matmul(grad_output_i.T, x_i)
            grad_bias[i] += torch.sum(grad_output_i, dim=0)

        #
        return grad_x, grad_bias, grad_weight, None


class RoutedLinearCol(autograd.Function):
    @staticmethod
    def forward(ctx,
                x: torch.Tensor,
                bias: torch.Tensor,
                weight: torch.Tensor,
                grouping: list[list]):
        # check
        assert x.dim() == 3
        assert bias.dim() == 1
        assert weight.dim() == 3
        assert x.size(-1) == weight.size(-1)
        out_features = weight.size(0)

        # fc1
        y = torch.zeros(
            [x.size(0), out_features],
            dtype=x.dtype, device=x.device
        )
        weight = torch.permute(
            weight, dims=[1, 2, 0]
        )
        weight = weight.contiguous()
        for i, batches in enumerate(grouping):
            if not batches:
                continue
            #
            x_i, w_i = x[batches, i], weight[i]
            y[batches] += torch.matmul(x_i, w_i)
        y += bias.view([1, -1])

        #
        ctx.grouping = grouping
        ctx.save_for_backward(x, weight)
        return y

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        # ctx
        x, weight = ctx.saved_tensors
        grouping = ctx.grouping

        # bias
        grad_bias = torch.sum(grad_output, dim=0)

        # weight
        grad_x = torch.zeros_like(x)
        grad_weight = torch.empty_like(weight)
        for i, batches in enumerate(grouping):
            if not batches:
                continue
            #
            x_i, w_i = x[batches, i], weight[i]
            grad_output_i = grad_output[batches]
            grad_x[batches, i] = torch.matmul(grad_output_i, w_i.T)
            grad_weight[i] = torch.matmul(x_i.T, grad_output_i)

        #
        grad_weight = torch.permute(
            grad_weight, dims=[2, 0, 1]
        )
        return grad_x, grad_bias, grad_weight, None


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

        #
        h = RoutedLinearRow.apply(
            x,
            self.fc1.bias.view(
                [self.n_blocks, self.block_size]
            ),
            self.fc1.weight.view(
                [self.n_blocks, self.block_size, -1]
            ),
            grouping
        )
        h = self.activation(h)
        y = RoutedLinearCol.apply(
            h, self.fc2.bias,
            self.fc2.weight.view(
                [-1, self.n_blocks, self.block_size]
            ),
            grouping
        )
        return y.view(x_size)
