import torch
from torch import autograd
from naive_gpt import ext


class SPMM(autograd.Function):
    @staticmethod
    def forward(ctx,
                indptr: torch.Tensor,
                indices: torch.Tensor,
                values: torch.Tensor,
                x: torch.Tensor):
        ctx.save_for_backward(
            indptr, indices, values, x
        )
        s_false = torch.scalar_tensor(False)
        return ext.spmm_forward_cuda(
            s_false, s_false,
            indptr, indices, values, x
        )

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        indptr = ctx.saved_tensors[0]
        indices = ctx.saved_tensors[1]
        values = ctx.saved_tensors[2]
        x = ctx.saved_tensors[3]
        #
        s_true = torch.scalar_tensor(True)
        s_false = torch.scalar_tensor(False)
        grad_output = grad_output.contiguous()
        grad_a = ext.sddmm_forward_cuda(
            s_false, s_true, indptr, indices,
            grad_output, x
        )
        grad_x = ext.spmm_forward_cuda(
            s_true, s_false, indptr, indices,
            values, grad_output
        )
        return None, None, grad_a, grad_x


def spmm(indptr: torch.Tensor,
         indices: torch.Tensor,
         values: torch.Tensor,
         x: torch.Tensor):
    return SPMM.apply(
        indptr, indices, values, x
    )
