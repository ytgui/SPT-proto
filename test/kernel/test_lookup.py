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
    indptr = torch.arange(
        0, seq_length + 1, device=cuda_device
    )
    indptr = torch.cumsum(
        torch.clamp(indptr, max=seq_length // 8), -1
    )
    indptr = indptr.type(torch.int)

    #
    query = torch.randint(
        high=n_codewords, size=[
            batch_size, seq_length, n_subspaces
        ],
        dtype=torch.int32, device=cuda_device
    )
    key = torch.randint(
        high=n_codewords, size=[
            batch_size, seq_length, n_subspaces
        ],
        dtype=torch.int32, device=cuda_device
    )

    #
    return mask, indptr, query, key


def test_lookup():
    batch_size = 64
    seq_length = random.choice(
        [512, 1024]
    )
    mask, indptr, query, key = get_input(
        n_subspaces=8, n_codewords=8,
        seq_length=seq_length,
        batch_size=batch_size
    )

    # builtin
    y_1 = []
    cmp = torch.eq(
        query.unsqueeze(-2), key.unsqueeze(-3)
    )
    reduced = torch.sum(cmp, dim=-1)
    reduced = torch.where(mask, reduced, -1)
    for b in range(batch_size):
        y_1.append([])
        for row in range(seq_length):
            topk = min(
                row + 1, reduced.size(-1) // 8
            )
            topk_indices = torch.topk(
                reduced[b, row], k=topk, dim=-1
            ).indices
            y_1[b].extend(topk_indices.tolist())
    y_1 = torch.IntTensor(y_1).to(query.device)
    assert y_1.size(-1) == indptr[-1].item()

    # kernel
    y_2 = kernels.lookup(
        indptr, query, key, sparse_coeff=8
    )

    # check
    recall = []
    assert y_1.size() == y_2.size()
    for b in range(batch_size):
        for row in range(seq_length):
            gt = y_1[
                b, indptr[row]:indptr[row + 1]
            ].tolist()
            pred = y_2[
                b, indptr[row]:indptr[row + 1]
            ].tolist()
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
    seq_length = 1024
    _, indptr, query, store = get_input(
        n_subspaces=n_subspaces, n_codewords=4,
        seq_length=seq_length, batch_size=batch_size
    )
    x_1 = torch.randn(
        [batch_size, seq_length, n_subspaces * 8],
        device=query.device
    )
    x_2 = torch.randn(
        [batch_size, n_subspaces * 8, seq_length],
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
            kernels.lookup(
                indptr, query, store, sparse_coeff=8
            )
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
