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
    sparse = mask.to_sparse_csr()
    indptr = sparse.crow_indices()
    indices = sparse.col_indices()
    indptr = indptr.type(torch.int32)
    indices = indices.type(torch.int32)

    #
    q = torch.randn(
        [batch_size, seq_length, n_features],
        requires_grad=True, device=device
    )
    k = torch.randn(
        [batch_size, seq_length, n_features],
        requires_grad=True, device=device
    )
    return mask, [topk_indices, topk_values], [
        indptr, indices, sparse.values()
    ], q, k


def test_sddmm():
    mask, topk, sparse, q, k = get_input(
        batch_size=random.randint(1, 16),
        seq_length=16 * random.randint(1, 16),
        n_features=16 * random.randint(1, 4)
    )
    topk_indices, topk_values = topk
    indptr, indices, values = sparse

    # matmul
    y_1 = torch.multiply(
        mask, torch.matmul(
            q, k.transpose(-1, -2)
        )
    )
    y_1 = torch.gather(
        y_1, dim=-1, index=topk_indices
    )
    y_1 = torch.flatten(y_1, start_dim=1)
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
    assert torch.allclose(y_1, y_2, atol=1e-3)
    assert torch.allclose(grad_q_1, grad_q_2, atol=1e-3)
    assert torch.allclose(grad_k_1, grad_k_2, atol=1e-3)

    #
    print('[PASS] test_sddmm()')


def bench_sddmm():
    mask, topk, sparse, q, k = get_input(
        64, seq_length=1024, n_features=64
    )
    indptr, indices, values = sparse

    # dense
    time.sleep(2.0)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        profile_memory=True, with_flops=True
    ) as prof:
        for _ in range(200):
            y_1 = torch.multiply(
                mask, torch.matmul(
                    q, k.transpose(-1, -2)
                )
            )
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
        for _ in range(200):
            y_2 = kernels.sddmm(
                indptr, indices, query=q, key=k
            )
            torch.sum(y_2).backward()
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
