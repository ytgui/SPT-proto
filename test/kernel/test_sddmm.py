import time
import torch
import random
from torch import autograd
from torch import profiler
from naive_gpt import ext


class SDDMM(autograd.Function):
    @staticmethod
    def forward(ctx,
                indptr: torch.Tensor,
                indices: torch.Tensor,
                query: torch.Tensor,
                key: torch.Tensor):
        return ext.sddmm_forward_cuda(
            indptr, indices, query, key
        )

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        raise NotImplementedError


def sddmm_fn(indptr: torch.Tensor,
             indices: torch.Tensor,
             query: torch.Tensor,
             key: torch.Tensor):
    return SDDMM.apply(indptr, indices, query, key)


def test_sddmm():
    d_model = 16 * random.randint(1, 16)
    seq_length = 16 * random.randint(1, 256)
    cuda_device = 'cuda'

    # mask
    prob = torch.rand(
        [seq_length, seq_length], device=cuda_device
    )
    mask = torch.where(prob < 0.25, 1.0, 0.0)
    sparse_mask = mask.to_sparse_csr()
    indptr = sparse_mask.crow_indices()
    indices = sparse_mask.col_indices()

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
    y_3: torch.Tensor = sddmm_fn(
        indptr=indptr, indices=indices, query=q, key=k
    )
    assert torch.allclose(y_1, y_2.to_dense(), atol=1e-3)
    assert torch.allclose(y_2.values(), y_3, atol=1e-3)

    #
    print('[PASS] test_sddmm()')


def bench_sddmm():
    d_model = 256
    seq_length = 2048
    cuda_device = 'cuda'

    # mask
    prob = torch.rand(
        [seq_length, seq_length], device=cuda_device
    )
    mask = torch.where(prob < 0.25, 1.0, 0.0)
    sparse_mask = mask.to_sparse_csr()
    indptr = sparse_mask.crow_indices()
    indices = sparse_mask.col_indices()

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
            y_1 = torch.softmax(y_1, dim=-1)
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
            y_3 = sddmm_fn(
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
