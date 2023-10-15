import torch
from torch import autograd
from naive_gpt import ext


class CSR2CSC(autograd.Function):
    @staticmethod
    def forward(ctx,
                indptr: torch.Tensor,
                indices: torch.Tensor,
                values: torch.Tensor,
                n_cols: int):
        config = torch.empty([n_cols])
        output = ext.csr2csc_cuda(
            config, indptr, indices, values
        )
        return output

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        raise NotImplementedError


def csr2csc(indptr: torch.Tensor,
            indices: torch.Tensor,
            values: torch.Tensor,
            n_cols: int):
    return CSR2CSC.apply(
        indptr, indices, values, n_cols
    )
