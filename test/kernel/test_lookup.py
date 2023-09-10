import torch
from torch import autograd
from naive_gpt import ext


class Lookup(autograd.Function):
    @staticmethod
    def forward(ctx,
                config: torch.Tensor,
                query: torch.Tensor,
                store: torch.Tensor):
        # ctx.save_for_backward(query, store)
        return ext.lookup_forward_cuda(config, query, store)

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        # query, table = ctx.saved_tensors
        raise NotImplementedError


def lookup(query: torch.Tensor,
           store: torch.Tensor,
           sparsity: int):
    config = torch.empty([sparsity])
    return Lookup.apply(config, query, store)


def test_lookup():
    n_subspaces = 8
    n_codewords = 4
    seq_length = 256
    batch_size = 16
    cuda_device = 'cuda'

    #
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

    # builtin
    cmp = torch.eq(
        query.unsqueeze(-2),
        store.unsqueeze(-3)
    )
    reduced = torch.sum(cmp, dim=-1)
    topk_output = torch.topk(
        reduced, k=seq_length // 8, dim=-1
    )
    topk_indices = topk_output.indices
    y_1 = torch.flatten(topk_indices, end_dim=-2)

    # kernel
    lookup_indices = lookup(query, store, sparsity=8)
    y_2 = torch.flatten(lookup_indices, end_dim=-2)

    # check
    accuracy = []
    assert y_1.size() == y_2.size()
    for lhs, rhs in zip(y_1.tolist(), y_2.tolist()):
        union = set(lhs).union(rhs)
        count = set(lhs).intersection(rhs)
        accuracy.append(len(count) / len(union))
    accuracy = sum(accuracy) / len(accuracy)
    print('accuracy:', accuracy)
    assert accuracy > 0.8

    #
    print('[PASS] test_lookup()')


def main():
    test_lookup()


if __name__ == '__main__':
    main()
