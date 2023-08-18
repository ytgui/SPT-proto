import torch
import random
from naive_gpt import kernels
from tqdm import tqdm


def test_cdist():
    d_code = 16 * random.randint(1, 4)
    n_queries = 16 * random.randint(1, 1024)
    n_codewords = 16 * random.randint(1, 16)
    cuda_device = 'cuda'

    #
    query = torch.randn(
        [n_queries, d_code], device=cuda_device,
        requires_grad=True
    )
    table = torch.randn(
        [n_codewords, d_code], device=cuda_device,
        requires_grad=True
    )

    #
    y_1 = torch.cdist(query, table, p=1.0)
    y_2 = kernels.pq_cdist(query, table)
    assert torch.allclose(y_1, y_2, atol=1e-3)


def main():
    for _ in tqdm(range(1024)):
        test_cdist()


if __name__ == '__main__':
    main()
