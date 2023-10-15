import time
import torch
import random
from torch import profiler
from naive_gpt import kernels


def get_input(n_subspaces: int,
              n_codewords: int,
              seq_length: int,
              batch_size: int):
    cuda_device = 'cuda'

    #
    mask = torch.tril(
        torch.ones(
            [batch_size, seq_length, seq_length],
            dtype=torch.bool, device=cuda_device
        )
    )
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
    return mask, query, store


def test_lookup():
    batch_size = 16
    n_subspaces = random.choice([8])
    seq_length = random.choice([512, 1024])
    mask, query, key = get_input(
        n_subspaces=n_subspaces, n_codewords=8,
        seq_length=seq_length, batch_size=batch_size
    )

    # builtin
    cmp = torch.eq(
        query.unsqueeze(-2), key.unsqueeze(-3)
    )
    reduced = torch.sum(cmp, dim=-1)
    y_1 = torch.where(mask, reduced, -1)

    # kernel
    y_2: torch.Tensor = kernels.lookup(
        query, key, sparse_coeff=8
    )

    # check
    recall = []
    assert y_1.size(0) == y_2.size(0)
    assert y_1.size(1) == y_2.size(1)
    for b in range(batch_size):
        for row in range(seq_length):
            k = min(
                row + 1, seq_length // 8
            )
            gt = torch.topk(
                y_1[b, row, :(row + 1)],
                k=k, dim=-1, largest=True
            ).indices.tolist()
            pred = y_2[b, row, :k].tolist()
            hit = set(gt).intersection(pred)
            recall.append(len(hit) / len(gt))
    recall = sum(recall) / len(recall)
    print('recall:', recall)
    assert recall > 0.8

    #
    print('[PASS] test_lookup()')


def bench_lookup():
    batch_size = 64
    n_subspaces = 8
    _, query, key = get_input(
        n_subspaces=n_subspaces, n_codewords=4,
        seq_length=1024, batch_size=batch_size
    )
    x_1 = torch.randn(
        [batch_size, 1024, n_subspaces * 8],
        device=query.device
    )
    x_2 = torch.randn(
        [batch_size, n_subspaces * 8, 1024],
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
            kernels.lookup(query, key, sparse_coeff=8)
            torch.cuda.synchronize()
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )

    #
    print('[PASS] bench_lookup()')


def main():
    test_lookup()
    bench_lookup()


if __name__ == '__main__':
    main()
