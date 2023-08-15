import torch
from torch import autograd
from naive_gpt import ext


class SparseMHA(autograd.Function):
    @staticmethod
    def forward(ctx,
                indptr: torch.Tensor,
                indices: torch.Tensor,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor):
        attn, output = ext.sparse_mha_forward(
            indptr, indices, query, key, value
        )
        ctx.save_for_backward(
            indptr, indices, query, key, value, attn, output
        )
        return output

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        indptr, indices = ctx.saved_tensors[:2]
        query, key, value = ctx.saved_tensors[2:5]
        attn, output = ctx.saved_tensors[5:]
        #
        grad_output = grad_output.contiguous()
        grad_query, grad_key, grad_value = ext.sparse_mha_backward(
            indptr, indices, query, key, value, attn, output, grad_output
        )
        return None, None, grad_query, grad_key, grad_value


def sparse_mha(indptr: torch.Tensor,
               indices: torch.Tensor,
               query: torch.Tensor,
               key: torch.Tensor,
               value: torch.Tensor):
    return SparseMHA.apply(
        indptr, indices, query, key, value
    )
