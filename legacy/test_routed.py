import time
import torch
import random
from torch import autograd
from torch import nn, profiler
from naive_gpt import kernels
from naive_gpt import ext


class RoutedFFN(autograd.Function):
    @staticmethod
    def forward(ctx,
                indices: torch.Tensor,
                weight: torch.Tensor,
                x: torch.Tensor):
        return ext.routed_forward_cuda(indices, weight, x)

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        raise NotImplementedError


def routed_ffn(indices: torch.Tensor,
               weight: torch.Tensor,
               x: torch.Tensor):
    return RoutedFFN.apply(indices, weight, x)


def test_routed():
    block_size = 4
    in_features = 4
    out_features = 16
    n_blocks = out_features // block_size
    batch_size = 16

    #
    x = torch.randn(
        [batch_size, in_features]
    )
    indices = torch.randint(
        high=n_blocks, size=[batch_size, 2]
    )
    fc_1 = nn.Linear(in_features, out_features)

    #
    y_1 = routed_ffn(indices, weight=fc_1.weight, x=x)

    #
    print('[PASS] test_routed()')


def bench_routed():
    #
    print('[PASS] bench_routed()')


def main():
    test_routed()
    bench_routed()


if __name__ == '__main__':
    main()
