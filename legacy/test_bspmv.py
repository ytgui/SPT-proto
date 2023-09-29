import time
import torch
import random
from torch import profiler
from torch import autograd
from naive_gpt import ext


class LinearFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        y = ext.bspmv_forward_cuda(x, weight.T.contiguous())
        return y

    @staticmethod
    def backward(ctx, grad_y):
        raise NotImplementedError


linear = LinearFunction.apply


def test_fc():
    in_features = 64 * random.randint(1, 16)
    out_features = 256 * random.randint(1, 64)
    batch_size = 256 * random.randint(1, 16)

    #
    x = torch.rand(
        size=[batch_size, in_features],
        device='cuda', requires_grad=True
    )
    weight = torch.rand(
        size=[out_features, in_features],
        device='cuda', requires_grad=True
    )

    # builtin
    y_1 = torch.matmul(x, weight.T)
    # y_1.sum().backward()
    # grad_x_1 = x.grad.detach().clone()
    # grad_w_1 = weight.grad.detach().clone()

    # custom
    x.grad = None
    weight.grad = None
    y_2 = linear(x, weight)
    # y_2.sum().backward()
    # grad_x_2 = x.grad.detach().clone()
    # grad_w_2 = weight.grad.detach().clone()

    # check
    assert torch.allclose(y_1, y_2, atol=1e-3)
    # assert torch.allclose(grad_x_1, grad_x_2, atol=1e-3)
    # assert torch.allclose(grad_w_1, grad_w_2, atol=1e-3)
    print('[PASS] test_fc()')


def bench_fc():
    in_features = 512
    out_features = 2048
    batch_size = 64 * 256
    x = torch.rand(
        size=[batch_size, in_features],
        device='cuda', requires_grad=True
    )
    weight = torch.rand(
        size=[out_features, in_features],
        device='cuda', requires_grad=True
    )

    #
    time.sleep(2.0)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        profile_memory=True, with_flops=True
    ) as prof:
        for _ in range(20):
            y_1 = torch.matmul(x, weight.T)
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )

    #
    time.sleep(2.0)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        profile_memory=True, with_flops=True
    ) as prof:
        for _ in range(20):
            y_2 = linear(x, weight)
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )


def main():
    test_fc()
    bench_fc()


if __name__ == "__main__":
    main()
