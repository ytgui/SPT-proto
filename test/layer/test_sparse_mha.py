import time
import torch
from torch import profiler
from naive_gpt import layers


def test_sparse_mha():
    d_head = 64
    n_heads = 4
    batch_size = 16
    seq_length = 1024
    cuda_device = 'cuda'

    #
    q = torch.ones(
        [batch_size, seq_length, n_heads, d_head],
        requires_grad=True, device=cuda_device
    )
    k = torch.ones(
        [batch_size, seq_length, n_heads, d_head],
        requires_grad=True, device=cuda_device
    )
    v = torch.ones(
        [batch_size, seq_length, n_heads, d_head],
        requires_grad=True, device=cuda_device
    )
    dense_fn = layers.VanillaAttention(
        d_head=d_head, p_dropout=0.0
    )
    dense_fn = dense_fn.to(cuda_device)
    sparse_fn = layers.SparseVanillaAttentionV2(
        d_head=d_head, d_codeword=8,
        n_codewords=16, p_dropout=0.0
    )
    sparse_fn = sparse_fn.to(cuda_device)

    # check
    y_1 = dense_fn(q, k, v, attn_mask=None)
    y_2 = sparse_fn(q, k, v, attn_mask=None)
    assert torch.allclose(y_1, y_2, atol=1e-3)

    #
    print('[PASS] test_sparse_mha()')


def bench_sparse_mha():
    d_head = 64
    n_heads = 4
    batch_size = 16
    seq_length = 1024
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
    dense_fn = layers.VanillaAttention(
        d_head=d_head, p_dropout=0.0
    )
    dense_fn = dense_fn.to(cuda_device)
    sparse_fn = layers.SparseVanillaAttentionV2(
        d_head=d_head, d_codeword=8,
        n_codewords=16, p_dropout=0.0
    )
    sparse_fn = sparse_fn.to(cuda_device)

    # pre-warm
    for _ in range(20):
        y_1 = dense_fn(q, k, v, attn_mask=None)
        y_2 = sparse_fn(q, k, v, attn_mask=None)
        torch.sum(y_1).backward()
        torch.sum(y_2).backward()
        torch.cuda.synchronize()
    
    # simple full
    torch.cuda.synchronize()
    before = time.time()
    y_1 = dense_fn(q, k, v, attn_mask=None)
    torch.cuda.synchronize()
    print('timing 0', 1000.0 * (time.time() - before))

    # simple sparse
    time.sleep(2.0)
    torch.cuda.synchronize()
    before = time.time()
    y_2 = sparse_fn(q, k, v, attn_mask=None)
    torch.cuda.synchronize()
    print('timing 1', 1000.0 * (time.time() - before))

    # full
    time.sleep(2.0)
    with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CUDA
            ],
            record_shapes=True,
            profile_memory=True,
            with_modules=True
    ) as prof:
        for _ in range(20):
            y_1 = dense_fn(q, k, v, attn_mask=None)
            torch.sum(y_1).backward()
            torch.cuda.synchronize()
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )

    # sparse
    time.sleep(2.0)
    with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CUDA
            ],
            record_shapes=True,
            profile_memory=True,
            with_modules=True
    ) as prof:
        for _ in range(20):
            y_2 = sparse_fn(q, k, v, attn_mask=None)
            torch.sum(y_2).backward()
            torch.cuda.synchronize()
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )

    #
    print('[PASS] bench_sparse_mha()')


def main():
    test_sparse_mha()
    bench_sparse_mha()


if __name__ == '__main__':
    main()
