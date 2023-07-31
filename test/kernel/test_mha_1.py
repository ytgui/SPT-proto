import torch
from naive_gpt import ext


def main():
    n_heads = 4
    d_model = 64
    seq_length = 16
    batch_size = 16
    d_head = d_model // n_heads
    cuda_device = 'cuda'

    #
    q = torch.randn(
        [seq_length, d_head],
        device=cuda_device
    )
    k = torch.randn(
        [seq_length, d_head],
        device=cuda_device
    )

    # mask
    prob = torch.rand(
        [seq_length, seq_length],
        device=cuda_device
    )
    mask = torch.where(
        prob < 0.05, True, False
    )
    sparse = mask.to_sparse_csr()

    # built-in
    y_1 = torch.matmul(
        q, k.transpose(0, 1)
    )
    y_1 = torch.masked_select(
        y_1, mask=mask
    )

    # custom kernel
    indptr = sparse.crow_indices()
    indices = sparse.col_indices()
    y_2 = ext.sparse_mha_forward(
        indptr, indices, q, k
    )

    # check
    print('y_1:', y_1)
    print('y_2:', y_2)
    print(torch.allclose(y_1, y_2, atol=1e-3))

    return


if __name__ == '__main__':
    main()
