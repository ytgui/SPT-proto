import torch
from torch import autograd
from naive_gpt import ext


class PQCDist(autograd.Function):
    @staticmethod
    def forward(ctx,
                query: torch.Tensor,
                table: torch.Tensor):
        ctx.save_for_backward(query, table)
        return ext.cdist_forward_cuda(query, table)

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        query, table = ctx.saved_tensors
        output = ext.cdist_backward_cuda(
            query, table, grad_output
        )
        grad_query, grad_table = output
        return grad_query, grad_table


def pq_cdist(query: torch.Tensor,
             table: torch.Tensor):
    return PQCDist.apply(query, table)
