import torch
from torch import nn
from naive_torch import layers, models


class SparseAttention(layers.VanillaAttention):
    def __init__(self,
                 d_head: int,
                 d_codeword: int,
                 n_codewords: int,
                 p_dropout: float):
        layers.VanillaAttention.__init__(
            self, d_head=d_head,
            p_dropout=p_dropout
        )
        #
        self.quantizer = models.PQ(
            d_codeword=d_codeword,
            n_codewords=n_codewords,
            n_subspaces=d_head // d_codeword
        )
        self.table = models.PQTable(
            quantizer=self.quantizer, dim=1
        )

    def _get_attn(self,
                  q: torch.Tensor,
                  k: torch.Tensor,
                  attn_mask: torch.Tensor):
        # loss
        loss_q = self.quantizer('train', z=q)[-1]
        loss_k = self.quantizer('train', z=k)[-1]
        self.register_buffer(
            'loss', loss_q + loss_k, persistent=False
        )

        # distance
        q_code = self.quantizer('encode', z=q)
        k_code = self.quantizer('encode', z=k)
        distance = torch.transpose(
            self.table(q_code, k_code), 1, 2
        )
        if attn_mask is not None:
            distance -= attn_mask

        # indptr and indices
        top_k = q.size(1) // 4
        top_indices = torch.topk(
            distance, k=top_k, dim=-1,
            largest=False, sorted=False
        )[-1]
        top_indices = torch.flatten(
            top_indices, start_dim=2
        )
        fixed_indptr = torch.arange(
            0, top_k * (q.size(1) + 1), step=top_k
        ).view(1, 1, -1)
        fixed_indptr = fixed_indptr.repeat(
            [distance.size(0), distance.size(1), 1]
        )
        sparse_mask = torch.sparse_csr_tensor(
            crow_indices=fixed_indptr,
            col_indices=top_indices,
            values=torch.ones_like(
                top_indices, dtype=q.dtype
            ),
            size=[distance.size(0), distance.size(1),
                  distance.size(2), distance.size(2)]
        )

        # TODO: CSR SDDMM
        raise NotImplementedError


def main():
    d_head = 64
    n_heads = 4
    seq_length = 64
    batch_size = 16

    #
    x = torch.randn(
        [batch_size, seq_length,
         n_heads, d_head]
    )
    sparse_fn = SparseAttention(
        d_head=d_head, d_codeword=4,
        n_codewords=64, p_dropout=0.0
    )
    y_1 = sparse_fn(x, x, x)

    #
    return


if __name__ == '__main__':
    main()
