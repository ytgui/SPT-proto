import torch
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
    n_heads = 4
    d_model = 64
    seq_length = 16
    batch_size = 16
    d_head = d_model // n_heads
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
    y_1 = torch.matmul(
        q, k.transpose(0, 1)
    )
    y_1 = torch.masked_select(
        y_1, mask=mask
    )
    torch.sum(y_1).backward()
    grad_q_1 = q.grad.detach().clone()
    grad_k_1 = k.grad.detach().clone()

    # custom kernel
    q.grad.zero_()
    k.grad.zero_()
    indptr = sparse.crow_indices()
    indices = sparse.col_indices()
    y_2 = sparse_mha(
        indptr, indices, q, k
    )
    torch.sum(y_2).backward()
    grad_q_2 = q.grad.detach().clone()
    grad_k_2 = k.grad.detach().clone()

    # check
    print('-- y --')
    print(
        torch.allclose(
            y_1, y_2, atol=1e-3
        )
    )
    print('-- grad --')
    print(
        torch.allclose(
            grad_q_1, grad_q_2, atol=1e-3
        )
    )


if __name__ == '__main__':
    main()
