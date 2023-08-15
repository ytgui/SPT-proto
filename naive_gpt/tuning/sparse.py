import torch
from naive_torch import layers, models
from naive_gpt import kernels


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
            pq=self.quantizer, dim=1
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
        ).transpose(-1, -2).contiguous()
        fixed_indptr = torch.arange(
            0, top_k * (q.size(1) + 1),
            step=top_k, device=q.device
        )

        #
        return [fixed_indptr, top_indices, q, k]

    def _apply_attn(self, attn: tuple, v: torch.Tensor):
        fixed_indptr, top_indices, q, k = attn
        return kernels.sparse_mha(
            fixed_indptr, top_indices, q, k, v
        )
