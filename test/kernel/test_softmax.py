import time
import torch
import random
from torch import profiler
from naive_gpt import kernels


def get_input(batch_size: int, seq_length: int):
    cuda_device = 'cuda'

    # mask
    prob = torch.rand(
        [batch_size, seq_length, seq_length],
        device=cuda_device
    )
    topk = torch.topk(
        prob, k=seq_length // 8, dim=-1,
        largest=True, sorted=False
    )
    mask = torch.scatter(
        torch.zeros_like(prob),
        dim=-1, index=topk.indices,
        src=torch.ones_like(topk.values)
    )
    dense = torch.where(
        mask > 0.0, prob, torch.full_like(
            prob, fill_value=float('-inf')
        )
    )
    dense.requires_grad = True

    # sparse
    sparse_mask = mask.to_sparse_csr()
    indptr = sparse_mask.crow_indices()
    indices = sparse_mask.col_indices()
    sparse_csr = [
        indptr.type(torch.int32),
        indices.type(torch.int32),
        topk.values.view_as(indices)
    ]
    sparse_csr[-1].requires_grad = True

    #
    return dense, sparse_csr


def test_softmax():
    dense, sparse_csr = get_input(
        batch_size=random.randint(1, 64),
        seq_length=64 * random.randint(1, 16)
    )
    indptr, indices, values = sparse_csr

    # torch
    y_1 = torch.softmax(dense, dim=-1)
    torch.max(y_1).backward()
    grad_1 = dense.grad.detach().clone()

    # kernel
    y_2 = kernels.softmax(
        indptr, indices, values
    )
    torch.max(y_2).backward()
    grad_2 = values.grad.detach().clone()

    # check
    y_2 = torch.sparse_csr_tensor(
        indptr, col_indices=indices, values=y_2
    )
    grad_2 = torch.sparse_csr_tensor(
        indptr, col_indices=indices, values=grad_2
    )
    assert torch.allclose(y_1, y_2.to_dense(), atol=1e-2)
    assert torch.allclose(grad_1, grad_2.to_dense(), atol=1e-3)

    #
    print('[PASS] test_softmax()')


def bench_softmax():
    dense, sparse_csr = get_input(
        batch_size=64, seq_length=1024
    )
    indptr, indices, values = sparse_csr

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
