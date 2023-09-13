import torch
from torch import autograd
from naive_gpt import ext


class CDist(autograd.Function):
    @staticmethod
    def forward(ctx,
                query: torch.Tensor,
                table: torch.Tensor):
        ctx.save_for_backward(query, table)
        output = ext.cdist_forward_cuda(query, table)
        distance, indices = output
        return distance, indices

    @staticmethod
    def backward(ctx,
                 grad_distance: torch.Tensor,
                 grad_indices: torch.Tensor):
        query, table = ctx.saved_tensors
        output = ext.cdist_backward_cuda(
            query, table, grad_distance.contiguous()
        )
        grad_query, grad_table = output
        return grad_query, grad_table


def cdist(query: torch.Tensor,
          table: torch.Tensor):
    return CDist.apply(query, table)
