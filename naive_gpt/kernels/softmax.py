import torch
from torch import autograd
from naive_gpt import ext


class Softmax(autograd.Function):
    @staticmethod
    def forward(ctx,
                indptr: torch.Tensor,
                indices: torch.Tensor,
                values: torch.Tensor):
        ctx.save_for_backward(
            indptr, indices, values
        )
        return ext.softmax_forward_cuda(
            indptr, indices, values
        )

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        raise NotImplementedError


def softmax(indptr: torch.Tensor,
            indices: torch.Tensor,
            values: torch.Tensor):
    return Softmax.apply(
        indptr, indices, values
    )
