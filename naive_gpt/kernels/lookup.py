import torch
from torch import autograd
from naive_gpt import ext


class Lookup(autograd.Function):
    @staticmethod
    def forward(ctx,
                config: torch.Tensor,
                query: torch.Tensor,
                store: torch.Tensor):
        return ext.lookup_forward_cuda(config, query, store)

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        raise NotImplementedError


def lookup(query: torch.Tensor,
           store: torch.Tensor,
           sparsity: int):
    config = torch.empty([sparsity])
    return Lookup.apply(config, query, store)
