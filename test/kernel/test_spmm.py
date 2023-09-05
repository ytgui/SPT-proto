import time
import torch
import random
from torch import profiler
from naive_gpt import kernels


def get_input(batch_size: int,
              seq_length: int,
              n_features: int):
    device = 'cuda'

    # mask
    prob = torch.rand(
        [batch_size, seq_length, seq_length],
        requires_grad=True, device=device
    )
    topk = torch.topk(
        prob, k=seq_length // 8, dim=-1,
        largest=True, sorted=False
    )
    dense = torch.scatter(
        torch.zeros_like(prob), dim=-1,
        index=topk.indices, src=topk.values
    )

    # sort indices
    order = torch.argsort(
        topk.indices, dim=-1
    )
    topk_indices = torch.gather(
        topk.indices, dim=-1, index=order
    )
    topk_values = torch.gather(
        topk.values, dim=-1, index=order
    )

    # sparse
    sparse = dense.to_sparse_csr()
    indptr = sparse.crow_indices()
    indices = sparse.col_indices()
    indptr = indptr.type(torch.int32)
    indices = indices.type(torch.int32)

    #
    x = torch.randn(
        [batch_size, seq_length, n_features],
        requires_grad=True, device=device
    )
    return dense, [topk_indices, topk_values], [
        indptr, indices, sparse.values()
    ], x


def test_spmm():
    dense, topk, sparse, x = get_input(
        batch_size=random.randint(1, 16),
        seq_length=16 * random.randint(1, 16),
        n_features=16 * random.randint(1, 4)
    )
    topk_indices, topk_values = topk
    indptr, indices, values = sparse

    # matmul
    dense = dense.detach().clone()
    dense.requires_grad = True
    y_1 = torch.bmm(dense, x)
    torch.sum(y_1).backward()
    grad_a_1 = torch.gather(
        dense.grad, dim=-1, index=topk_indices
    )
    grad_a_1 = torch.flatten(grad_a_1, start_dim=1)
    grad_x_1 = x.grad.detach().clone()

    # kernel
    x.grad = None
    values = values.detach().clone()
    values.requires_grad = True
    y_2 = kernels.spmm(
        indptr, indices, values, x=x
    )
    # torch.sum(y_2).backward()
    # grad_a_2 = values.grad.detach().clone()
    # grad_x_2 = x.grad.detach().clone()

    # check
    assert torch.allclose(y_1, y_2, atol=1e-3)
    # assert torch.allclose(grad_a_1, grad_a_2, atol=1e-3)
    # assert torch.allclose(grad_x_1, grad_x_2, atol=1e-3)

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
