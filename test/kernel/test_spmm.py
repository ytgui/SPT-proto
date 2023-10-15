import time
import torch
import random
from torch import profiler
from naive_gpt import kernels


def get_input(batch_size: int,
              seq_length: int,
              n_features: int):
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
    dense = torch.scatter(
        torch.zeros_like(prob), dim=-1,
        index=topk.indices, src=topk.values
    )

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
    dense.requires_grad = True

    #
    x = torch.randn(
        [batch_size, seq_length, n_features],
        requires_grad=True, device=cuda_device
    )
    return dense, sparse_csr, x


def test_spmm():
    dense, sparse_csr, x = get_input(
        batch_size=random.randint(1, 16),
        seq_length=16 * random.randint(1, 16),
        n_features=16 * random.randint(1, 4)
    )
    indptr, indices, values = sparse_csr

    # matmul
    y_1 = torch.matmul(dense, x)
    torch.sum(y_1).backward()
    grad_a_1 = dense.grad.detach().clone()
    grad_x_1 = x.grad.detach().clone()

    # kernel
    x.grad = None
    y_2 = kernels.spmm(
        indptr, indices, values, x=x
    )
    torch.sum(y_2).backward()
    grad_a_2 = values.grad.detach().clone()
    grad_x_2 = x.grad.detach().clone()

    # check
    assert torch.allclose(y_1, y_2, atol=1e-3)
    assert torch.allclose(grad_x_1, grad_x_2, atol=1e-3)
    grad_a_1 = torch.where(
        dense > 0.0, grad_a_1, 0.0
    )
    indptr = torch.expand_copy(
        indptr.view(1, -1), size=[indices.size(0), -1]
    )
    grad_a_2 = torch.sparse_csr_tensor(
        indptr, col_indices=indices, values=grad_a_2
    )
    assert torch.allclose(grad_a_1, grad_a_2.to_dense(), atol=1e-3)

    #
    print('[PASS] test_spmm()')


def bench_spmm():
    dense, sparse_csr, x = get_input(
        64, seq_length=1024, n_features=128
    )
    indptr, indices, values = sparse_csr

    # dense
    time.sleep(2.0)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        profile_memory=True, with_flops=True
    ) as prof:
        for _ in range(20):
            y_1 = torch.matmul(dense, x)
            torch.sum(y_1).backward()
            torch.cuda.synchronize()
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
            y_2: torch.Tensor = kernels.spmm(
                indptr, indices, values, x=x
            )
            torch.sum(y_2).backward()
            torch.cuda.synchronize()
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
