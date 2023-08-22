import torch
import random
from naive_gpt import kernels


def test_cdist():
    d_code = random.choice([4, 8, 16])
    n_queries = 4 * random.randint(1, 64)
    n_codewords = 4 * random.randint(1, 64)
    n_subspaces = random.randint(1, 16)
    cuda_device = 'cuda'

    #
    query = torch.randn(
        [n_subspaces, n_queries, d_code],
        device=cuda_device, requires_grad=True
    )
    table = torch.randn(
        [n_subspaces, n_codewords, d_code],
        device=cuda_device, requires_grad=True
    )

    #
    y_1 = torch.cdist(query, table, p=1.0)
    torch.mean(y_1).backward()
    grad_q_1 = query.grad.detach().clone()
    grad_t_1 = table.grad.detach().clone()

    #
    query.grad = None
    table.grad = None
    y_2 = kernels.pq_cdist(query, table)
    torch.mean(y_2).backward()
    grad_q_2 = query.grad.detach().clone()
    grad_t_2 = table.grad.detach().clone()

    # check
    assert torch.allclose(y_1, y_2, atol=1e-3)
    assert torch.allclose(grad_q_1, grad_q_2, atol=1e-3)
    assert torch.allclose(grad_t_1, grad_t_2, atol=1e-3)

    #
    print('[PASS] test_cdist()')


def main():
    test_cdist()


if __name__ == '__main__':
    main()
