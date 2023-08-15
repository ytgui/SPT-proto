import torch
import random
from torch import autograd
from naive_gpt import ext
from tqdm import tqdm


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


def test_sparse_mha():
    d_head = random.choice(
        [16, 32, 48, 64]
    )
    n_heads = random.randint(1, 16)
    seq_length = random.randint(16, 256)
    batch_size = random.randint(1, 16)
    cuda_device = 'cuda'

    #
    q = torch.randn(
        [batch_size, seq_length, n_heads, d_head],
        requires_grad=True, device=cuda_device
    )
    k = torch.randn(
        [batch_size, seq_length, n_heads, d_head],
        requires_grad=True, device=cuda_device
    )
    v = torch.randn(
        [batch_size, seq_length, n_heads, d_head],
        requires_grad=True, device=cuda_device
    )
    attn = torch.einsum('niae, njae -> naij', q, k)

    # sparse
    top_k = attn.size(-1) // 4
    top_output = torch.topk(
        attn, k=top_k, dim=-1,
        largest=True, sorted=False
    )
    top_values, top_indices = top_output
    csr_indices = torch.flatten(
        top_indices, start_dim=2
    ).transpose(-1, -2).contiguous()
    fixed_indptr = torch.arange(
        0, top_k * (attn.size(-2) + 1),
        step=top_k, device=cuda_device
    )

    # built-in
    attn = torch.scatter(
        torch.full_like(
            attn, fill_value=float('-inf')
        ), dim=-1,
        index=top_indices, src=top_values
    )
    attn = torch.softmax(attn, dim=-1)
    y_1 = torch.einsum(
        'naij, njae -> niae', attn, v
    )
    torch.sum(y_1).backward()
    grad_q_1 = q.grad.detach().clone()
    grad_k_1 = k.grad.detach().clone()
    grad_v_1 = v.grad.detach().clone()

    # custom kernel
    q.grad, k.grad, v.grad = None, None, None
    y_2 = sparse_mha(
        fixed_indptr, csr_indices, q, k, v
    )
    torch.sum(y_2).backward()
    grad_q_2 = q.grad.detach().clone()
    grad_k_2 = k.grad.detach().clone()
    grad_v_2 = v.grad.detach().clone()

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
    assert torch.allclose(
        grad_v_1, grad_v_2, atol=1e-3
    )


def main():
    for _ in tqdm(range(1024)):
        test_sparse_mha()


if __name__ == '__main__':
    main()
