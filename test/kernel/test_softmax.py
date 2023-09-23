import time
import torch
import random
from torch import profiler
from naive_gpt import kernels


def get_input(batch_size: int, seq_length: int):
    cuda_device = 'cuda'

    # dense
    x = torch.randn(
        [batch_size, seq_length, 16]
    )
    dense = torch.matmul(
        x, x.transpose(-1, -2)
    )
    dense = torch.clamp(
        dense, min=-5.0, max=5.0
    )

    # sparsify
    for b in range(batch_size):
        for row in range(seq_length):
            topk = min(
                row + 1, seq_length // 8
            )
            topk_output = torch.topk(
                dense[b, row, :(row + 1)],
                k=topk, dim=-1, largest=True
            )
            dense[b, row] = torch.scatter(
                torch.zeros_like(dense[b, row]),
                dim=-1, index=topk_output.indices,
                src=topk_output.values
            )
    dense = dense.to(cuda_device)

    # sparse
    sparse = dense.to_sparse_csr()
    indptr = sparse.crow_indices()
    indices = sparse.col_indices()
    sparse_csr = [
        indptr[0].type(torch.int32),
        indices.type(torch.int32),
        sparse.values().type(torch.float)
    ]
    sparse_csr[-1].requires_grad = True

    # fill -inf
    dense = torch.where(
        dense != 0.0, dense, float('-inf')
    )
    dense.requires_grad = True

    #
    return dense, sparse_csr


def test_softmax():
    dense, sparse_csr = get_input(
        batch_size=random.randint(1, 16),
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
    indptr = torch.expand_copy(
        indptr.view(1, -1), size=[indices.size(0), -1]
    )
    y_2 = torch.sparse_csr_tensor(
        indptr, col_indices=indices, values=y_2
    )
    grad_2 = torch.sparse_csr_tensor(
        indptr, col_indices=indices, values=grad_2
    )
    assert torch.allclose(y_1, y_2.to_dense(), atol=1e-3)
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
