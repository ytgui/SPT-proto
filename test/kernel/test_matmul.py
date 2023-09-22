import time
import torch
import random
from torch import autograd
from torch import profiler
from naive_gpt import ext


class GEMMFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        y = ext.matmul_cuda(
            x, weight.T.contiguous()
        )
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, weight = ctx.saved_tensors
        #
        grad_y = grad_y.contiguous()
        grad_x = ext.matmul_cuda(grad_y, weight)
        grad_weight = ext.matmul_cuda(
            x.T.contiguous(), grad_y
        )
        return grad_x, grad_weight.T.contiguous()


cuda_matmul = GEMMFunction.apply


def test_fc():
    in_features = 128 * random.randint(1, 16)
    out_features = 128 * random.randint(1, 64)
    batch_size = 128 * random.randint(1, 16)
    cuda_device = 'cuda'

    #
    x = torch.rand(
        size=[batch_size, in_features],
        device=cuda_device, requires_grad=True
    )
    weight = torch.rand(
        size=[out_features, in_features],
        device=cuda_device, requires_grad=True
    )

    # builtin
    y_1 = torch.matmul(x, weight.T)
    y_2 = cuda_matmul(x, weight)
    assert torch.allclose(y_1, y_2, atol=1e-3)


def bench_fc():
    in_features = 512
    out_features = 2048
    batch_size = 64 * 256
    cuda_device = 'cuda'

    #
    x = torch.rand(
        size=[batch_size, in_features],
        device=cuda_device, requires_grad=True
    )
    weight = torch.rand(
        size=[out_features, in_features],
        device=cuda_device, requires_grad=True
    )

    #
    time.sleep(2.0)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        profile_memory=True, with_flops=True
    ) as prof:
        for _ in range(20):
            torch.matmul(x, weight.T)
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
            cuda_matmul(x, weight)
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
