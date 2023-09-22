import time
import torch
import random
from torch import profiler
from naive_gpt import kernels


def test_cdist():
    d_code = random.choice([4, 8])
    n_queries = 64 * random.randint(1, 64)
    n_codewords = 16 * random.randint(1, 16)
    n_subspaces = random.randint(1, 16)
    cuda_device = 'cuda'

    #
    query = torch.randn(
        [n_subspaces, n_queries, d_code],
        device=cuda_device, requires_grad=True
    )
    table = torch.randn(
        [n_subspaces, n_codewords, d_code],
        device=cuda_device, requires_grad=True
    )

    #
    y_1a = torch.cdist(query, table, p=1.0)
    indices_1 = torch.argmin(y_1a, dim=-1)
    y_1b = torch.gather(
        y_1a, dim=-1, index=indices_1.unsqueeze(-1)
    )
    torch.sum(y_1b).backward()
    grad_q_1 = query.grad.detach().clone()
    grad_t_1 = table.grad.detach().clone()

    #
    query.grad = None
    table.grad = None
    y_2a, indices_2 = kernels.cdist(query, table)
    indices_2 = indices_2.type(torch.long)
    y_2b = torch.gather(
        y_2a, dim=-1, index=indices_2.unsqueeze(-1)
    )
    torch.sum(y_2b).backward()
    grad_q_2 = query.grad.detach().clone()
    grad_t_2 = table.grad.detach().clone()

    # check
    assert torch.allclose(y_1a, y_2a, atol=1e-3)
    assert torch.allclose(indices_1, indices_2)
    assert torch.allclose(y_1b, y_2b, atol=1e-3)
    assert torch.allclose(grad_q_1, grad_q_2, atol=1e-3)
    assert torch.allclose(grad_t_1, grad_t_2, atol=1e-3)

    #
    print('[PASS] test_cdist()')


def bench_cdist():
    d_code = 8
    n_queries = 16384
    n_codewords = 64
    n_subspaces = 8
    cuda_device = 'cuda'

    #
    query = torch.randn(
        [n_subspaces, n_queries, d_code],
        device=cuda_device, requires_grad=True
    )
    table = torch.randn(
        [n_subspaces, n_codewords, d_code],
        device=cuda_device, requires_grad=True
    )

    # dot
    time.sleep(2.0)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        profile_memory=True, with_flops=True
    ) as prof:
        for _ in range(20):
            y_1 = torch.matmul(
                query, table.transpose(-1, -2)
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
        for _ in range(20):
            y_2 = kernels.cdist(query, table)[0]
            torch.sum(y_2).backward()
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )

    #
    print('[PASS] bench_cdist()')


def main():
    test_cdist()
    bench_cdist()


if __name__ == '__main__':
    main()
