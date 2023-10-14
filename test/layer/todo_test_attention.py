import torch
from naive_gpt import tuning


def test_attention():
    d_head = 64
    n_heads = 4
    seq_length = 64
    batch_size = 16
    cuda_device = 'cuda'

    #
    x = torch.ones(
        [batch_size, seq_length,
         n_heads, d_head],
        device=cuda_device
    )
    sparse_fn = tuning.SparseAttention(
        d_head=d_head, d_codeword=4,
        n_codewords=64, p_dropout=0.0
    ).to(device=cuda_device)

    #
    y_1 = sparse_fn(x, x, x)
    assert torch.allclose(y_1, x, atol=1e-3)

    #
    print('[PASS] test_attention()')


def main():
    test_attention()


if __name__ == '__main__':
    main()
