import torch
from torch import autograd
from naive_gpt import ext


class Lookup(autograd.Function):
    @staticmethod
    def forward(ctx,
                config: torch.Tensor,
                indptr: torch.Tensor,
                query: torch.Tensor,
                store: torch.Tensor):
        return ext.lookup_forward_cuda(
            config, indptr, query, store
        )

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        raise NotImplementedError


def lookup(indptr: torch.Tensor,
           query: torch.Tensor,
           key: torch.Tensor,
           sparse_coeff: int):
    config = torch.empty([sparse_coeff])
    return Lookup.apply(config, indptr, query, key)
