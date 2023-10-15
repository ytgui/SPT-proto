import torch
from torch import autograd
from naive_gpt import ext

import time
import torch
import random
from torch import profiler
from naive_gpt import kernels


class CSR2CSC(autograd.Function):
    @staticmethod
    def forward(ctx,
                indptr: torch.Tensor,
                indices: torch.Tensor,
                values: torch.Tensor,
                n_cols: int):
        config = torch.empty([n_cols])
        output = ext.csr2csc_cuda(
            config, indptr, indices, values
        )
        return output

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        raise NotImplementedError


def csr2csc(indptr: torch.Tensor,
            indices: torch.Tensor,
            values: torch.Tensor,
            n_cols: int):
    return CSR2CSC.apply(
        indptr, indices, values, n_cols
    )


def get_input(batch_size: int,
              seq_length: int):
    cuda_device = 'cuda'

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
        indptr.type(torch.int32),
        indices.type(torch.int32),
        sparse.values().type(torch.float)
    ]
    sparse_csr[-1].requires_grad = True
    dense.requires_grad = True

    #
    return dense, sparse_csr


def test_csr2csc():
    batch_size = random.randint(1, 16)
    seq_length = random.randint(1, 64)
    dense, sparse_csr = get_input(
        batch_size=batch_size, seq_length=seq_length
    )
    indptr, indices, values = sparse_csr

    # to csc
    output = csr2csc(
        indptr, indices, values, n_cols=seq_length
    )
    rev_indptr, rev_indices, rev_values = output

    # check
    y = torch.sparse_csr_tensor(
        rev_indptr, rev_indices, values=rev_values,
        size=[batch_size, seq_length, seq_length]
    )
    y = y.to_dense().transpose(-1, -2)
    assert torch.allclose(y, dense, atol=1e-3)

    #
    print('[PASS] test_csr2csc()')


def bench_csr2csc():
    seq_length = 1024
    _, sparse_csr = get_input(
        batch_size=64, seq_length=seq_length
    )
    indptr, indices, values = sparse_csr

    # kernel
    time.sleep(2.0)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        profile_memory=True, with_flops=True
    ) as prof:
        for _ in range(20):
            y = csr2csc(
                indptr, indices,
                values, n_cols=seq_length
            )
            torch.cuda.synchronize()
    print(prof.key_averages().table(
        sort_by='cuda_time_total', row_limit=5
    ))

    #
    print('[PASS] bench_csr2csc()')


def main():
    test_csr2csc()
    bench_csr2csc()


if __name__ == '__main__':
    main()
