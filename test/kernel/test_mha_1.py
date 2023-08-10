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
    seq_length = 16 * random.randint(1, 16)
    cuda_device = 'cuda'

    #
    q = torch.randn(
        [seq_length, d_head],
        requires_grad=True,
        device=cuda_device
    )
    k = torch.randn(
        [seq_length, d_head],
        requires_grad=True,
        device=cuda_device
    )
    attn = torch.einsum(
        'ie, je -> ij', q, k
    )

    # sparse
    top_k = attn.size(-1) // 4
    top_output = torch.topk(
        attn, k=top_k, dim=-1,
        largest=True, sorted=False
    )
    top_values, top_indices = top_output
    csr_indices = torch.flatten(top_indices)
    fixed_indptr = torch.arange(
        0, top_k * (attn.size(-2) + 1),
        step=top_k, device=cuda_device
    )
    sparse_attn = torch.sparse_csr_tensor(
        crow_indices=fixed_indptr, col_indices=top_indices,
        values=top_values, size=attn.size()
    )

    # built-in
    y_1 = torch.flatten(top_values)
    torch.sum(y_1).backward()
    grad_q_1 = q.grad.detach().clone()
    grad_k_1 = k.grad.detach().clone()

    # custom kernel
    q.grad = None
    k.grad = None
    y_2 = sparse_mha(
        fixed_indptr, csr_indices, q, k
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
