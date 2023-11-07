import torch
from torch import autograd
from naive_gpt import ext


class CSR2CSC(autograd.Function):
    @staticmethod
    def forward(ctx,
                indptr: torch.Tensor,
                indices: torch.Tensor,
                values: torch.Tensor):
        output = ext.csr2csc_cuda(
            indptr, indices, values
        )
        return output

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        raise NotImplementedError


def csr2csc(indptr: torch.Tensor,
            indices: torch.Tensor,
            values: torch.Tensor):
    return CSR2CSC.apply(
        indptr, indices, values
    )
