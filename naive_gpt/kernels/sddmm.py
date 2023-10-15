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
        s_true = torch.scalar_tensor(True)
        s_false = torch.scalar_tensor(False)
        return ext.sddmm_forward_cuda(
            s_false, s_true, indptr,
            indices, query, key
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
        grad_output = grad_output.contiguous()
        #
        s0 = torch.cuda.current_stream()
        with torch.cuda.stream(s0):
            grad_query = ext.spmm_forward_cuda(
                s_false, s_false, indptr, indices,
                grad_output, key
            )
        #
        s1 = torch.cuda.Stream(priority=-1)
        with torch.cuda.stream(s1):
            s1.wait_stream(s0)
            grad_key = ext.spmm_forward_cuda(
                s_true, s_false, indptr, indices,
                grad_output, query
            )
        s0.wait_stream(s1)
        return None, None, grad_query, grad_key


def sddmm(indptr: torch.Tensor,
          indices: torch.Tensor,
          query: torch.Tensor,
          key: torch.Tensor):
    return SDDMM.apply(
        indptr, indices, query, key
    )
