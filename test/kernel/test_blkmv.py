import time
import torch
import random
from torch import profiler
from naive_gpt import kernels


def get_input(in_blocks: int,
              out_blocks: int,
              block_size: int):
    device = 'cpu'

    # mask
    prob = torch.rand(
        [out_blocks, in_blocks],
        device=device
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

    #
    dense = torch.randn(
        [out_blocks * block_size,
         in_blocks * block_size]
    )
    blocks = [
        dense[
            y * block_size:(y + 1) * block_size,
            x * block_size:(x + 1) * block_size]
        for y in range(out_blocks) for x in range(in_blocks)
    ]
    blocks = torch.stack(blocks, dim=0)
    blocks = blocks.view(
        [out_blocks, in_blocks, block_size, block_size]
    )
    bsr_tensor = torch.sparse_bsr_tensor(
        csr_indptr, csr_indices, values=blocks[mask]
    )

    #
    mask = mask.view(
        [out_blocks, in_blocks]
    ).type(torch.float)
    mask = mask.repeat_interleave(
        repeats=block_size, dim=-1
    )
    mask = mask.repeat_interleave(
        repeats=block_size, dim=-2
    )
    x = torch.randn(
        [in_blocks * block_size, 1],
        device=device
    )

    #
    return mask, dense, bsr_tensor, x


def test_blkmv():
    mask, dense, sparse, x = get_input(
        in_blocks=4 * random.randint(1, 4),
        out_blocks=4 * random.randint(1, 4),
        block_size=4 * random.randint(1, 4)
    )

    # dense
    y_1 = torch.matmul(
        torch.multiply(mask, dense), x
    )

    # sparse
    y_2 = torch.sparse.mm(sparse, x)

    # check
    assert torch.allclose(y_1, y_2, atol=1e-3)

    #
    print('[PASS] test_blkmv()')


def bench_blkmv():
    mask, dense, sparse, x = get_input(
        in_blocks=16, out_blocks=64,
        block_size=64
    )

    # dense
    time.sleep(2.0)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU],
        profile_memory=True, with_flops=True
    ) as prof:
        for _ in range(20):
            y_1 = torch.matmul(
                torch.multiply(mask, dense), x
            )
    print(
        prof.key_averages().table(
            sort_by='cpu_time_total', row_limit=5
        )
    )

    # sparse
    time.sleep(2.0)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU],
        profile_memory=True, with_flops=True
    ) as prof:
        for _ in range(20):
            y_2 = torch.sparse.mm(sparse, x)
    print(
        prof.key_averages().table(
            sort_by='cpu_time_total', row_limit=5
        )
    )

    #
    print('[PASS] bench_blkmv()')


def main():
    test_blkmv()
    bench_blkmv()


if __name__ == '__main__':
    main()
