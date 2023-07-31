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

    # built-in
    y_1 = torch.matmul(
        q, k.transpose(0, 1)
    )
    # y_1 = torch.where(
    #     mask, y_1, torch.zeros_like(y_1)
    # )

    # custom kernel
    y_2 = ext.sparse_mha_forward(q, k)

    # check
    print(torch.allclose(y_1, y_2, atol=1e-3))

    return


if __name__ == '__main__':
    main()
