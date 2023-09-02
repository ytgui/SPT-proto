import time
import torch
import random
from torch import profiler
from naive_gpt import kernels


def test_sddmm():
    d_model = 16 * random.randint(1, 16)
    seq_length = 64 * random.randint(1, 64)
    cuda_device = 'cuda'

    # mask
    prob = torch.rand(
        [seq_length, seq_length],
        device=cuda_device
    )
    topk = torch.topk(
        prob, k=seq_length // 4, dim=-1
    )
    mask = torch.scatter(
        torch.zeros_like(prob),
        dim=-1, index=topk.indices,
        src=torch.ones_like(topk.values)
    )
    sparse_mask = mask.to_sparse_csr()
    indptr = sparse_mask.crow_indices()
    indices = sparse_mask.col_indices()
    indices = indices.type(torch.int32)
    indptr = indptr.type(torch.int32)

    # query
    q = torch.randn(
        [seq_length, d_model], device=cuda_device
    )
    k = torch.randn(
        [seq_length, d_model], device=cuda_device
    )

    # check
    y_1 = torch.multiply(mask, torch.matmul(q, k.T))
    y_2: torch.Tensor = torch.sparse.sampled_addmm(
        sparse_mask, q, k.T, alpha=1.0, beta=0.0
    )
    y_3: torch.Tensor = kernels.sddmm(
        indptr=indptr, indices=indices, query=q, key=k
    )
    assert torch.allclose(y_1, y_2.to_dense(), atol=1e-3)
    assert torch.allclose(y_2.values(), y_3, atol=1e-3)

    #
    print('[PASS] test_sddmm()')


def bench_sddmm():
    d_model = 64
    seq_length = 1024
    cuda_device = 'cuda'

    # mask
    prob = torch.rand(
        [seq_length, seq_length],
        device=cuda_device
    )
    topk = torch.topk(
        prob, k=seq_length // 8, dim=-1
    )
    mask = torch.scatter(
        torch.zeros_like(prob),
        dim=-1, index=topk.indices,
        src=torch.ones_like(topk.values)
    )
    sparse_mask = mask.to_sparse_csr()
    indptr = sparse_mask.crow_indices()
    indices = sparse_mask.col_indices()
    indices = indices.type(torch.int32)
    indptr = indptr.type(torch.int32)

    # query
    q = torch.randn(
        [seq_length, d_model], device=cuda_device
    )
    k = torch.randn(
        [seq_length, d_model], device=cuda_device
    )

    # dense
    time.sleep(2.0)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        profile_memory=True, with_flops=True
    ) as prof:
        for _ in range(200):
            y_1 = torch.multiply(
                mask, torch.matmul(q, k.T)
            )
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )

    # sparse
    time.sleep(2.0)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        profile_memory=True, with_flops=True
    ) as prof:
        for _ in range(200):
            y_2 = torch.sparse.sampled_addmm(
                sparse_mask, q, k.T, alpha=1.0, beta=0.0
            )
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
        for _ in range(200):
            y_3 = kernels.sddmm(
                indptr=indptr, indices=indices, query=q, key=k
            )
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )

    #
    print('[PASS] bench_sddmm()')


def main():
    test_sddmm()
    bench_sddmm()


if __name__ == '__main__':
    main()
