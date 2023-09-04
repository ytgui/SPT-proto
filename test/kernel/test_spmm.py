import time
import torch
import random
from torch import profiler
from naive_gpt import kernels


def test_spmm():
    d_model = 16 * random.randint(1, 4)
    seq_length = 64 * random.randint(1, 16)
    cuda_device = 'cuda'

    # mask
    prob = torch.rand(
        [seq_length, seq_length],
        requires_grad=True,
        device=cuda_device
    )
    topk = torch.topk(
        prob, k=seq_length // 4, dim=-1,
        sorted=False
    )
    mask = torch.scatter(
        torch.zeros_like(prob),
        dim=-1, index=topk.indices,
        src=torch.ones_like(prob)
    )
    sparse_mask = mask.to_sparse_csr()
    indptr = sparse_mask.crow_indices()
    indices = sparse_mask.col_indices()
    indices = indices.type(torch.int32)
    indptr = indptr.type(torch.int32)

    # x
    x = torch.randn(
        [seq_length, d_model],
        requires_grad=True,
        device=cuda_device
    )

    # matmul
    mask.requires_grad = True
    y_1 = torch.matmul(mask, x)
    torch.sum(y_1).backward()
    grad_a_1 = torch.multiply(
        mask, mask.grad.detach()
    )
    grad_x_1 = x.grad.detach().clone()

    # torch.sparse
    x.grad = None
    sparse_mask = torch.clone(
        sparse_mask.detach()
    )
    sparse_mask.requires_grad = True
    y_2 = torch.sparse.mm(sparse_mask, x)
    torch.sum(y_2).backward()
    grad_a_2 = sparse_mask.grad.clone()
    grad_x_2 = x.grad.detach().clone()

    # kernel
    x.grad = None
    sparse_values = torch.clone(
        sparse_mask.values().detach()
    )
    sparse_values.requires_grad = True
    y_3 = kernels.spmm(
        indptr, indices, sparse_values, x=x
    )
    torch.sum(y_3).backward()
    grad_a_3 = sparse_values.grad.detach().clone()
    grad_x_3 = x.grad.detach().clone()

    # check
    assert torch.allclose(y_1, y_2, atol=1e-3)
    assert torch.allclose(y_1, y_3, atol=1e-3)
    assert torch.allclose(
        grad_a_1, grad_a_2.to_dense(), atol=1e-3
    )
    assert torch.allclose(
        grad_a_2.values(), grad_a_3, atol=1e-3
    )
    assert torch.allclose(grad_x_1, grad_x_2, atol=1e-3)
    assert torch.allclose(grad_x_1, grad_x_3, atol=1e-3)

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
        prob, k=seq_length // 8, dim=-1,
        sorted=False
    )
    mask = torch.scatter(
        torch.zeros_like(prob),
        dim=-1, index=topk.indices,
        src=torch.ones_like(topk.values)
    )
    sparse_mask = torch.clone(
        mask.detach().to_sparse_csr()
    )
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
            sparse_mask.requires_grad = True
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
            y_3: torch.Tensor = kernels.spmm(
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
