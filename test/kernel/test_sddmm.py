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
    mask = torch.scatter(
        torch.zeros_like(prob),
        dim=-1, index=topk.indices,
        src=torch.ones_like(topk.values)
    )

    # sparse
    sparse = mask.to_sparse_csr()
    indptr = sparse.crow_indices()
    indices = sparse.col_indices()
    sparse_csr = [
        indptr[0].type(torch.int32),
        indices.type(torch.int32)
    ]

    # query and key
    q = torch.randn(
        [batch_size, seq_length, n_features],
        requires_grad=True, device=device
    )
    k = torch.randn(
        [batch_size, seq_length, n_features],
        requires_grad=True, device=device
    )
    return mask, sparse_csr, q, k


def test_sddmm():
    mask, sparse_csr, q, k = get_input(
        batch_size=random.randint(1, 16),
        seq_length=16 * random.randint(1, 16),
        n_features=16 * random.randint(1, 4)
    )
    indptr, indices = sparse_csr

    # matmul
    y_1 = torch.multiply(
        mask, torch.matmul(
            q, k.transpose(-1, -2)
        )
    )
    torch.sum(y_1).backward()
    grad_q_1 = q.grad.detach().clone()
    grad_k_1 = k.grad.detach().clone()

    # kernel
    q.grad, k.grad = None, None
    y_2: torch.Tensor = kernels.sddmm(
        indptr, indices, query=q, key=k
    )
    torch.sum(y_2).backward()
    grad_q_2 = q.grad.detach().clone()
    grad_k_2 = k.grad.detach().clone()

    # check
    indptr = torch.expand_copy(
        indptr.view(1, -1), size=[indices.size(0), -1]
    )
    y_2 = torch.sparse_csr_tensor(
        indptr, col_indices=indices, values=y_2
    )
    assert torch.allclose(y_1, y_2.to_dense(), atol=1e-3)
    assert torch.allclose(grad_q_1, grad_q_2, atol=1e-3)
    assert torch.allclose(grad_k_1, grad_k_2, atol=1e-3)

    #
    print('[PASS] test_sddmm()')


def bench_sddmm():
    mask, sparse_csr, q, k = get_input(
        batch_size=64, seq_length=1024,
        n_features=128
    )
    indptr, indices = sparse_csr

    # dense
    time.sleep(2.0)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        profile_memory=True, with_flops=True
    ) as prof:
        for _ in range(20):
            y_1 = torch.multiply(
                mask, torch.matmul(
                    q, k.transpose(-1, -2)
                )
            )
            torch.sum(y_1).backward()
            torch.cuda.synchronize()
    print(prof.key_averages().table(
        sort_by='cuda_time_total', row_limit=5
    ))

    # kernel
    time.sleep(2.0)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        profile_memory=True, with_flops=True
    ) as prof:
        for _ in range(20):
            y_2 = kernels.sddmm(
                indptr, indices, query=q, key=k
            )
            torch.sum(y_2).backward()
            torch.cuda.synchronize()
    print(prof.key_averages().table(
        sort_by='cuda_time_total', row_limit=5
    ))

    #
    print('[PASS] bench_sddmm()')


def main():
    test_sddmm()
    bench_sddmm()


if __name__ == '__main__':
    main()
