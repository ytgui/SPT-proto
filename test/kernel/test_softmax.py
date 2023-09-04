import time
import torch
import random
from torch import profiler
from naive_gpt import kernels


def get_input(seq_length: int):
    device = 'cuda'

    # mask
    prob = torch.randn(
        [seq_length, seq_length],
        requires_grad=True, device=device
    )
    topk = torch.topk(
        prob, k=seq_length // 8,
        largest=True, sorted=False
    )
    mask = torch.scatter(
        torch.zeros_like(prob), dim=-1,
        index=topk.indices, src=topk.values
    )

    # sparse
    sparse_mask = mask.to_sparse_csr()
    indptr = sparse_mask.crow_indices()
    indices = sparse_mask.col_indices()
    indices = indices.type(torch.int32)
    indptr = indptr.type(torch.int32)

    #
    return prob, topk, [
        indptr, indices, sparse_mask.values()
    ]


def test_softmax():
    prob, topk, sparse = get_input(
        seq_length=256 * random.randint(1, 16)
    )

    # torch
    dense = torch.scatter(
        torch.full_like(
            prob, fill_value=float('-inf')
        ), dim=-1,
        index=topk.indices, src=topk.values
    )
    dense = dense.detach().clone()
    dense.requires_grad = True
    y_1 = torch.softmax(dense, dim=-1)
    y_1 = torch.gather(
        y_1, dim=-1, index=topk.indices
    )
    y_1 = torch.flatten(y_1)
    torch.max(y_1).backward()
    grad_1 = torch.gather(
        dense.grad, dim=-1, index=topk.indices
    )
    grad_1 = torch.flatten(grad_1)

    # kernel
    indptr, indices, values = sparse
    values = torch.clone(
        values.detach()
    )
    values.requires_grad = True
    y_2 = kernels.softmax(
        indptr, indices, values
    )
    torch.max(y_2).backward()
    grad_2 = values.grad.detach().clone()

    # exp precision
    diff = torch.abs(y_1 - y_2)
    assert diff.mean().item() < 1e-2
    assert torch.allclose(
        grad_1, grad_2, atol=1e-2
    )

    #
    print('[PASS] test_softmax()')


def bench_softmax():
    prob, topk, sparse = get_input(
        seq_length=2048
    )
    dense = torch.scatter(
        torch.full_like(
            prob, fill_value=float('-inf')
        ), dim=-1,
        index=topk.indices, src=topk.values
    )
    dense = dense.detach().clone()
    dense.requires_grad = True
    indptr, indices, values = sparse
    values = torch.clone(
        values.detach()
    )
    values.requires_grad = True

    # torch
    time.sleep(2.0)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        profile_memory=True, with_flops=True
    ) as prof:
        for _ in range(20):
            y_1 = torch.softmax(dense, dim=-1)
            torch.sum(y_1).backward()
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
            y_2 = kernels.softmax(
                indptr, indices, values
            )
            torch.sum(y_2).backward()
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )

    #
    print('[PASS] bench_softmax()')


def main():
    test_softmax()
    bench_softmax()


if __name__ == '__main__':
    main()
