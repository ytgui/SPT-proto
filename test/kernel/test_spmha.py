import time
import torch
import random
from torch import profiler
from naive_torch import layers
from naive_gpt import kernels


def test_spmha():
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
    topk_output = torch.topk(
        attn, k=top_k, dim=-1,
        largest=True, sorted=False
    )
    topk_values, topk_indices = topk_output
    csr_indices = torch.flatten(
        topk_indices, start_dim=2
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
        index=topk_indices, src=topk_values
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
    y_2 = kernels.sparse_mha(
        fixed_indptr, csr_indices, q, k, v
    )
    torch.sum(y_2).backward()
    grad_q_2 = q.grad.detach().clone()
    grad_k_2 = k.grad.detach().clone()
    grad_v_2 = v.grad.detach().clone()

    # check
    assert torch.allclose(y_1, y_2, atol=1e-3)
    assert torch.allclose(grad_q_1, grad_q_2, atol=1e-3)
    assert torch.allclose(grad_k_1, grad_k_2, atol=1e-3)
    assert torch.allclose(grad_v_1, grad_v_2, atol=1e-3)

    #
    print('[PASS] test_spmha()')


def bench_spmha():
    d_head = 64
    n_heads = 16
    seq_length = 512
    batch_size = 64
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
    attn_fn = layers.VanillaAttention(
        d_head=d_head, p_dropout=0.0
    )
    attn_dense = torch.einsum('niae, njae -> naij', q, k)

    # sparse
    top_k = attn_dense.size(-1) // 4
    topk_output = torch.topk(
        attn_dense, k=top_k, dim=-1,
        largest=True, sorted=False
    )
    csr_indices = torch.flatten(
        topk_output.indices, start_dim=2
    ).transpose(-1, -2).contiguous()
    fixed_indptr = torch.arange(
        0, top_k * (attn_dense.size(-2) + 1),
        step=top_k, device=cuda_device
    )

    # einsum
    time.sleep(2.0)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        profile_memory=True, with_flops=True
    ) as prof:
        for _ in range(20):
            y_1 = attn_fn(q, k, v)
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )

    # kernel
    time.sleep(2.0)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        profile_memory=True, with_flops=True
    ) as prof:
        for _ in range(20):
            y_2 = kernels.sparse_mha(
                fixed_indptr, csr_indices, q, k, v
            )
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )

    #
    print('[PASS] bench_spmha()')


def main():
    test_spmha()
    bench_spmha()


if __name__ == '__main__':
    main()
