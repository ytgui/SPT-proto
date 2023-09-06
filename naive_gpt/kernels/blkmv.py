import torch
from torch import autograd
from naive_gpt import ext


class BLKMV(autograd.Function):
    @staticmethod
    def forward(ctx,
                config: torch.Tensor,
                dense: torch.Tensor,
                indptr: torch.Tensor,
                indices: torch.Tensor,
                x: torch.Tensor):
        ctx.save_for_backward(
            config, dense, indptr, indices, x
        )
        return ext.blkmv_forward_cuda(
            config, dense, indptr, indices, x
        )

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        config = ctx.saved_tensors[0]
        dense = ctx.saved_tensors[1]
        indptr = ctx.saved_tensors[2]
        indices = ctx.saved_tensors[3]
        x = ctx.saved_tensors[4]
        #
        output = ext.blkmv_backward_cuda(
            config, dense, indptr, indices,
            x, grad_output.contiguous()
        )
        grad_weight, grad_x = output
        return None, grad_weight, None, None, grad_x


def blkmv(config: torch.Tensor,
          dense: torch.Tensor,
          indptr: torch.Tensor,
          indices: torch.Tensor,
          x: torch.Tensor):
    return BLKMV.apply(
        config, dense, indptr, indices, x
    )
