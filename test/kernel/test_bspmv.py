import time
import torch
import random
from torch import nn
from torch.autograd import Function
from naive_gpt import ext


class LinearFunction(Function):
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


linear = LinearFunction.apply


def test_fc():
    in_features = 128 * random.randint(1, 16)
    out_features = 128 * random.randint(1, 64)
    batch_size = 128 * random.randint(1, 16)

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
    y_1.sum().backward()
    grad_x_1 = x.grad.detach().clone()
    grad_w_1 = weight.grad.detach().clone()

    # custom
    x.grad = None
    weight.grad = None
    y_2 = linear(x, weight)
    y_2.sum().backward()
    grad_x_2 = x.grad.detach().clone()
    grad_w_2 = weight.grad.detach().clone()

    # check
    torch.cuda.synchronize()
    assert torch.allclose(y_1, y_2, atol=1e-3)
    assert torch.allclose(grad_x_1, grad_x_2, atol=1e-3)
    assert torch.allclose(grad_w_1, grad_w_2, atol=1e-3)


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
    torch.cuda.synchronize()
    before = time.time()
    for _ in range(200):
        y_1 = torch.matmul(x, weight.T)
        y_1.sum().backward()
    torch.cuda.synchronize()
    print('timing 0:', time.time() - before)

    #
    time.sleep(5.0)
    torch.cuda.synchronize()
    before = time.time()
    for _ in range(200):
        y_2 = linear(x, weight)
        y_2.sum().backward()
    torch.cuda.synchronize()
    print('timing 1:', time.time() - before)


def main():
    test_fc()
    bench_fc()


if __name__ == "__main__":
    main()
