import torch
from torch import autograd
from naive_gpt import ext


class SDDMM(autograd.Function):
    @staticmethod
    def forward(ctx,
                indptr: torch.Tensor,
                indices: torch.Tensor,
                query: torch.Tensor,
                key: torch.Tensor):
        return ext.sddmm_forward_cuda(
            indptr, indices, query, key
        )

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        raise NotImplementedError


def sddmm(indptr: torch.Tensor,
          indices: torch.Tensor,
          query: torch.Tensor,
          key: torch.Tensor):
    return SDDMM.apply(
        indptr, indices, query, key
    )
