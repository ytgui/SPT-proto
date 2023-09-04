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
        ctx.save_for_backward(
            indptr, indices,
            query, key
        )
        return ext.sddmm_forward_cuda(
            indptr, indices, query, key
        )

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        indptr = ctx.saved_tensors[0]
        indices = ctx.saved_tensors[1]
        query = ctx.saved_tensors[2]
        key = ctx.saved_tensors[3]
        #
        s_true = torch.scalar_tensor(True)
        s_false = torch.scalar_tensor(False)
        grad_key = ext.spmm_forward_cuda(
            s_true, s_false, indptr, indices,
            grad_output.contiguous(), query
        )
        grad_query = ext.spmm_forward_cuda(
            s_false, s_false, indptr, indices,
            grad_output.contiguous(), key
        )
        return None, None, grad_query, grad_key


def sddmm(indptr: torch.Tensor,
          indices: torch.Tensor,
          query: torch.Tensor,
          key: torch.Tensor):
    return SDDMM.apply(
        indptr, indices, query, key
    )
