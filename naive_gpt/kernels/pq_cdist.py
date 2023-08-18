import torch
from torch import autograd
from naive_gpt import ext


class PQCDist(autograd.Function):
    @staticmethod
    def forward(ctx,
                query: torch.Tensor,
                table: torch.Tensor):
        return ext.cdist_forward_cuda(query, table)

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        raise NotImplementedError


def pq_cdist(query: torch.Tensor,
             table: torch.Tensor):
    return PQCDist.apply(query, table)
