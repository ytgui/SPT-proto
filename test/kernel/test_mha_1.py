import torch
import random
from torch import autograd
from naive_gpt import ext


class SparseMHA(autograd.Function):
    @staticmethod
    def forward(ctx,
                indptr: torch.Tensor,
                indices: torch.Tensor,
                query: torch.Tensor,
                key: torch.Tensor):
        ctx.save_for_backward(
            indptr, indices, query, key
        )
        return ext.sparse_mha_forward(
            indptr, indices, query, key
        )

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        indptr, indices, query, key = \
            ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_query, grad_key = ext.sparse_mha_backward(
            indptr, indices, query, key, grad_output
        )
        return None, None, grad_query, grad_key


def sparse_mha(indptr: torch.Tensor,
               indices: torch.Tensor,
               query: torch.Tensor,
               key: torch.Tensor):
    return SparseMHA.apply(
        indptr, indices, query, key
    )


def main():
    d_head = random.choice(
        [16, 32, 48, 64]
    )
    n_heads = random.randint(1, 8)
    seq_length = random.randint(1, 256)
    cuda_device = 'cuda'

    #
    q = torch.ones(
        [seq_length, d_head],
        requires_grad=True,
        device=cuda_device
    )
    k = torch.ones(
        [seq_length, d_head],
        requires_grad=True,
        device=cuda_device
    )

    # mask
    prob = torch.rand(
        [seq_length, seq_length],
        device=cuda_device
    )
    mask = torch.where(
        prob < 0.15, True, False
    )
    sparse = mask.to_sparse_csr()

    # built-in
    y_1 = torch.matmul(q, k.T)
    y_1 = torch.masked_select(
        y_1, mask=mask
    )
    torch.sum(y_1).backward()
    grad_q_1 = q.grad.detach().clone()
    grad_k_1 = k.grad.detach().clone()

    # custom kernel
    q.grad = None
    k.grad = None
    indptr_2 = sparse.crow_indices()
    indices_2 = sparse.col_indices()
    y_2 = sparse_mha(
        indptr_2, indices_2, q, k
    )
    torch.sum(y_2).backward()
    grad_q_2 = q.grad.detach().clone()
    grad_k_2 = k.grad.detach().clone()

    # check
    assert torch.allclose(
        y_1, y_2, atol=1e-3
    )
    assert torch.allclose(
        grad_q_1, grad_q_2, atol=1e-3
    )
    assert torch.allclose(
        grad_k_1, grad_k_2, atol=1e-3
    )


if __name__ == '__main__':
    main()
