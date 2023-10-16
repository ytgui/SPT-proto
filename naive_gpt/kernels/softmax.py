import torch
from torch import autograd
from naive_gpt import ext


class Softmax(autograd.Function):
    @staticmethod
    def forward(ctx,
                indptr: torch.Tensor,
                indices: torch.Tensor,
                values: torch.Tensor):
        output = ext.softmax_forward_cuda(
            indptr, indices, values
        )
        ctx.save_for_backward(
            indptr, indices, output
        )
        return output

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        indptr = ctx.saved_tensors[0]
        indices = ctx.saved_tensors[1]
        output = ctx.saved_tensors[2]
        grad_values = ext.softmax_backward_cuda(
            indptr, indices, output,
            grad_output.contiguous()
        )
        return None, None, grad_values


def softmax(indptr: torch.Tensor,
            indices: torch.Tensor,
            values: torch.Tensor):
    return Softmax.apply(
        indptr, indices, values
    )
