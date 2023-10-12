import torch
import random
from torch import nn
from naive_gpt import layers


def test_pq_basic():
    d_codeword = random.choice([4, 8, 16])
    n_codewords = 64 * random.randint(1, 4)
    n_queries = 64 * random.randint(1, 16)
    n_subspaces = random.choice([8, 16])
    d_model = n_subspaces * d_codeword
    cuda_device = 'cuda'

    #
    for PQModule in [layers.PQv1,
                     layers.PQv2]:
        mse_fn = nn.MSELoss()
        quantizer = PQModule(
            d_codeword=d_codeword,
            n_codewords=n_codewords,
            n_subspaces=n_subspaces
        ).to(cuda_device)

        #
        for x in [
            # 2d
            torch.randn(
                [n_queries, d_model],
                device=cuda_device
            ),
            # 3d
            torch.randn(
                [n_queries, n_queries, d_model],
                device=cuda_device
            )
        ]:
            # quant
            y_1 = quantizer('encode', x)
            y_2 = quantizer('decode', y_1)
            y_3 = quantizer('quantize', x)
            assert y_1.size(0) == n_queries
            assert y_1.size(-1) == n_subspaces
            assert y_2.size() == x.size()
            assert mse_fn(y_2, target=x) < 2.5
            assert torch.allclose(y_2, y_3)

            # train
            loss = quantizer('train', x)
            assert loss.numel() == 1

    #
    print('[PASS] test_pq_basic()')


def main():
    test_pq_basic()


if __name__ == '__main__':
    main()
