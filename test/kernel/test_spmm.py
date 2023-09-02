import time
import torch
import random
from torch import autograd
from torch import profiler
from naive_gpt import ext


class SPMM(autograd.Function):
    @staticmethod
    def forward(ctx,
                indptr: torch.Tensor,
                indices: torch.Tensor,
                values: torch.Tensor,
                x: torch.Tensor):
        return ext.spmm_forward_cuda(
            indptr, indices, values, x
        )

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        raise NotImplementedError


def spmm_fn(indptr: torch.Tensor,
            indices: torch.Tensor,
            values: torch.Tensor,
            x: torch.Tensor):
    return SPMM.apply(indptr, indices, values, x)


def test_spmm():
    d_model = 16 * random.randint(1, 4)
    seq_length = 64 * random.randint(1, 16)
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

    # x
    x = torch.randn(
        [seq_length, d_model], device=cuda_device
    )

    # check
    y_1 = torch.matmul(mask, x)
    y_2 = torch.sparse.mm(sparse_mask, x)
    y_3: torch.Tensor = spmm_fn(
        indptr, indices, values=sparse_mask.values(), x=x
    )
    assert torch.allclose(y_1, y_2, atol=1e-3)
    assert torch.allclose(y_2, y_3, atol=1e-3)

    #
    print('[PASS] test_spmm()')


def bench_spmm():
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

    # x
    x = torch.randn(
        [seq_length, d_model], device=cuda_device
    )

    # dense
    time.sleep(2.0)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        profile_memory=True, with_flops=True
    ) as prof:
        for _ in range(200):
            y_1 = torch.matmul(mask, x)
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
            y_2 = torch.sparse.mm(sparse_mask, x)
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
            y_3: torch.Tensor = spmm_fn(
                indptr, indices, values=sparse_mask.values(), x=x
            )
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )

    #
    print('[PASS] bench_spmm()')


def main():
    test_spmm()
    bench_spmm()


if __name__ == '__main__':
    main()
