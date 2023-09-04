import torch
from torch import autograd
from naive_gpt import ext


class SPMM(autograd.Function):
    @staticmethod
    def forward(ctx,
                indptr: torch.Tensor,
                indices: torch.Tensor,
                values: torch.Tensor,
                x: torch.Tensor):
        s_false = torch.scalar_tensor(False)
        return ext.spmm_forward_cuda(
            s_false, s_false,
            indptr, indices, values, x
        )

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        raise NotImplementedError


def spmm(indptr: torch.Tensor,
         indices: torch.Tensor,
         values: torch.Tensor,
         x: torch.Tensor):
    return SPMM.apply(
        indptr, indices, values, x
    )
