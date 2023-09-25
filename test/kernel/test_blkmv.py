import time
import torch
import random
from torch import profiler
from naive_gpt import kernels


def get_input(in_blocks: int,
              out_blocks: int,
              block_size: int,
              batch_size: int,
              skip_mask: bool = False):
    cuda_device = 'cuda'

    # mask
    prob = torch.rand(
        [batch_size, out_blocks, in_blocks],
        device=cuda_device
    )
    topk = torch.topk(
        prob, k=in_blocks // 4,
        largest=True, sorted=False
    )
    mask = torch.scatter(
        torch.zeros_like(
            prob, dtype=torch.bool
        ),
        dim=-1, index=topk.indices,
        src=torch.ones_like(
            topk.values, dtype=torch.bool
        )
    )
    sparse_mask = mask.to_sparse_csr()
    csr_indices = sparse_mask.col_indices()
    csr_indptr = sparse_mask.crow_indices()
    indptr = csr_indptr.cpu().type(torch.int32)
    indices = csr_indices.cpu().type(torch.int32)

    #
    dense = torch.randn(
        [out_blocks * block_size, in_blocks * block_size],
        requires_grad=True, device=cuda_device
    )

    #
    if not skip_mask:
        mask = mask.type(torch.float)
        mask = mask.repeat_interleave(block_size, dim=-1)
        mask = mask.repeat_interleave(block_size, dim=-2)
    x = torch.randn(
        [batch_size, in_blocks * block_size],
        requires_grad=True, device=cuda_device
    )
    config = torch.empty(
        [out_blocks, in_blocks, block_size]
    )

    #
    return config, mask, [indptr, indices, dense], x


def test_blkmv():
    config, mask, sparse, x = get_input(
        in_blocks=4 * random.randint(1, 4),
        out_blocks=4 * random.randint(1, 4),
        block_size=4 * random.randint(1, 4),
        batch_size=random.randint(1, 16)
    )
    indptr, indices, dense = sparse

    # dense
    y_1 = torch.matmul(
        torch.multiply(
            mask, dense.unsqueeze(0)
        ), x.unsqueeze(-1)
    )
    y_1 = y_1.squeeze(-1)
    torch.sum(y_1).backward()
    grad_w_1 = dense.grad.detach().clone()
    grad_x_1 = x.grad.detach().clone()

    # custom
    x.grad = None
    dense.grad = None
    y_2 = kernels.blkmv(
        config, dense, indptr, indices, x
    )
    torch.sum(y_2).backward()
    grad_w_2 = dense.grad.detach().clone()
    grad_x_2 = x.grad.detach().clone()

    # check
    assert torch.allclose(y_1, y_2, atol=1e-3)
    assert torch.allclose(grad_w_1, grad_w_2, atol=1e-3)
    assert torch.allclose(grad_x_1, grad_x_2, atol=1e-3)

    #
    print('[PASS] test_blkmv()')


def bench_blkmv():
    config, _, sparse, x = get_input(
        in_blocks=4, out_blocks=16,
        block_size=256, batch_size=1024,
        skip_mask=True
    )
    indptr, indices, dense = sparse

    # dense
    time.sleep(2.0)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        profile_memory=True, with_flops=True
    ) as prof:
        for _ in range(20):
            y_1 = torch.matmul(x, dense.T)
            # torch.sum(y_1).backward()
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
        for _ in range(20):
            y_2 = kernels.blkmv(
                config, dense, indptr, indices, x
            )
            # torch.sum(y_2).backward()
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )

    #
    print('[PASS] bench_blkmv()')


def main():
    test_blkmv()
    bench_blkmv()


if __name__ == '__main__':
    main()
