import time
import torch
import random
from torch import profiler
from naive_gpt import kernels


from torch import autograd
from naive_gpt import ext


class BLKMV(autograd.Function):
    @staticmethod
    def forward(ctx,
                config: torch.Tensor,
                dense: torch.Tensor,
                indptr: torch.Tensor,
                indices: torch.Tensor,
                x: torch.Tensor):
        ctx.save_for_backward(
            config, dense, indptr, indices, x
        )
        return ext.blkmv_forward_cuda(
            config, dense, indptr, indices, x
        )

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        raise NotImplementedError


def blkmv(config: torch.Tensor,
          dense: torch.Tensor,
          indptr: torch.Tensor,
          indices: torch.Tensor,
          x: torch.Tensor):
    return BLKMV.apply(
        config, dense, indptr, indices, x
    )


def get_input(in_blocks: int,
              out_blocks: int,
              block_size: int):
    cuda_device = 'cuda'

    # mask
    prob = torch.rand(
        [out_blocks, in_blocks],
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
        device=cuda_device
    )

    #
    mask = mask.type(torch.float)
    mask = mask.repeat_interleave(block_size, dim=-1)
    mask = mask.repeat_interleave(block_size, dim=-2)
    x = torch.randn(
        [in_blocks * block_size], device=cuda_device
    )

    #
    return mask, dense, [indptr, indices], x


def test_blkmv():
    in_blocks = 4 # * random.randint(1, 4)
    out_blocks = 2 # * random.randint(1, 4)
    block_size = 2  # * random.randint(1, 4)
    mask, dense, sparse, x = get_input(
        in_blocks=in_blocks,
        out_blocks=out_blocks,
        block_size=block_size
    )
    indptr, indices = sparse

    # dense
    y_1 = torch.matmul(
        torch.multiply(mask, dense), x
    )

    # custom
    config = torch.empty(
        [out_blocks, in_blocks, block_size]
    )
    y_2 = blkmv(config, dense, indptr, indices, x)

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
    # bench_blkmv()


if __name__ == '__main__':
    main()