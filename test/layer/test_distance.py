import torch
import random
from naive_gpt import layers


def test_pq_table_1():
    d_model = 4
    n_subspaces = 1
    x_eye = torch.eye(d_model)
    quantizer = layers.PQ(
        d_codeword=d_model, n_codewords=d_model,
        n_subspaces=n_subspaces
    )
    with torch.no_grad():
        for param in quantizer.parameters():
            param.set_(x_eye.unsqueeze(dim=0))

    #
    x_code: torch.Tensor
    x_code = quantizer('encode', x_eye)
    assert x_code.tolist() == [
        [0], [1], [2], [3]
    ]

    #
    table = layers.PQTable(quantizer)
    x_dist_1 = torch.cdist(
        x_eye, x_eye, p=2.0
    )
    x_dist_2 = table(
        q_code=x_code, k_code=x_code
    )
    assert torch.allclose(
        x_dist_1, x_dist_2, atol=1e-5
    )

    #
    print('[PASS] test_pq_table_1()')


def test_pq_table_2():
    d_codeword = random.randint(1, 16)
    n_codewords = random.randint(1, 64)
    n_subspaces = random.randint(1, 16)
    seq_length = random.randint(1, 64)
    batch_size = random.randint(1, 16)
    d_model = n_subspaces * d_codeword

    #
    for loss_method in ['vq-vae', 'k-means']:
        quantizer = layers.PQ(
            d_codeword=d_codeword,
            n_codewords=n_codewords,
            n_subspaces=n_subspaces,
            method=loss_method
        )
        table = layers.PQTable(quantizer)
        assert torch.allclose(
            table.table.transpose(-1, -2),
            table.table, atol=1e-5
        )

        # 2d
        x = torch.randn(
            [batch_size, d_model]
        )
        x_dist: torch.Tensor
        x_code = quantizer('encode', z=x)
        x_dist = table(x_code, x_code)
        assert x_dist.size() == torch.Size(
            [batch_size, batch_size]
        )
        x_diag = torch.diagonal(x_dist)
        assert torch.all(
            torch.less_equal(x_diag, 0.1)
        )

        # 3d - batch dim
        x = torch.randn(
            [batch_size, seq_length, d_model]
        )
        x_dist: torch.Tensor
        x_code = quantizer('encode', z=x)
        x_dist = table(x_code, x_code)
        assert x_dist.size() == torch.Size(
            [batch_size, seq_length, batch_size]
        )
        x_diag = torch.diagonal(
            x_dist, dim1=0, dim2=2
        )
        assert torch.all(
            torch.less_equal(x_diag, 0.1)
        )

        # 3d - sequence dim
        x = torch.randn(
            [batch_size, seq_length, d_model]
        )
        table = layers.PQTable(quantizer, dim=1)
        assert torch.allclose(
            table.table.transpose(-1, -2),
            table.table, atol=1e-5
        )
        x_dist: torch.Tensor
        x_code = quantizer('encode', z=x)
        x_dist = table(x_code, x_code)
        assert x_dist.size() == torch.Size(
            [batch_size, seq_length, seq_length]
        )
        x_diag = torch.diagonal(
            x_dist, dim1=1, dim2=2
        )
        assert torch.all(
            torch.less_equal(x_diag, 0.1)
        )

    #
    print('[PASS] test_pq_table_2()')


def main():
    test_pq_table_1()
    test_pq_table_2()


if __name__ == '__main__':
    main()
