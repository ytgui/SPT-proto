import time
import torch
from torch import profiler
from torch import autograd
from naive_gpt import ext


class Lookup(autograd.Function):
    @staticmethod
    def forward(ctx,
                config: torch.Tensor,
                query: torch.Tensor,
                store: torch.Tensor):
        return ext.lookup_forward_cuda(config, query, store)

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        raise NotImplementedError


def lookup(query: torch.Tensor,
           store: torch.Tensor,
           sparsity: int):
    config = torch.empty([sparsity])
    return Lookup.apply(config, query, store)


def get_input(n_subspaces: int,
              n_codewords: int,
              seq_length: int,
              batch_size: int):
    cuda_device = 'cuda'

    #
    query = torch.randint(
        high=n_codewords, size=[
            batch_size, seq_length, n_subspaces
        ],
        dtype=torch.int32, device=cuda_device
    )
    store = torch.randint(
        high=n_codewords, size=[
            batch_size, seq_length, n_subspaces
        ],
        dtype=torch.int32, device=cuda_device
    )
    return query, store


def test_lookup():
    query, store = get_input(
        n_subspaces=8, n_codewords=4,
        seq_length=512, batch_size=64
    )

    # builtin
    cmp = torch.eq(
        query.unsqueeze(-2),
        store.unsqueeze(-3)
    )
    reduced = torch.sum(cmp, dim=-1)
    topk_output = torch.topk(
        reduced, k=reduced.size(-1) // 8, dim=-1
    )
    topk_indices = topk_output.indices
    y_1 = torch.flatten(topk_indices, end_dim=-2)

    # kernel
    lookup_indices = lookup(query, store, sparsity=8)
    y_2 = torch.flatten(lookup_indices, end_dim=-2)

    # check
    recall = []
    assert y_1.size() == y_2.size()
    for lhs, rhs in zip(y_1.tolist(), y_2.tolist()):
        count = set(lhs).intersection(rhs)
        recall.append(len(count) / len(lhs))
    recall = sum(recall) / len(recall)
    print('recall:', recall)
    assert recall > 0.8

    #
    print('[PASS] test_lookup()')


def bench_lookup():
    batch_size = 64
    n_subspaces = 8
    query, store = get_input(
        n_subspaces=n_subspaces, n_codewords=4,
        seq_length=512, batch_size=batch_size
    )
    x_1 = torch.randn(
        [batch_size, 512, n_subspaces * 8],
        device=query.device
    )
    x_2 = torch.randn(
        [batch_size, n_subspaces * 8, 512],
        device=query.device
    )

    # matmul
    time.sleep(2.0)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        profile_memory=True, with_flops=True
    ) as prof:
        for _ in range(20):
            torch.matmul(x_1, x_2)
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
            lookup(query, store, sparsity=8)
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )

    #
    print('[PASS] bench_lookup()')


def main():
    # test_lookup()
    bench_lookup()


if __name__ == '__main__':
    main()
