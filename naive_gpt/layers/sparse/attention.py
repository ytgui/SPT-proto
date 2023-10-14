import torch
from naive_gpt import layers


class SparseVanillaAttentionV1(layers.VanillaAttention):
    def __init__(self,
                 d_head: int,
                 p_dropout: float,
                 d_codeword: int,
                 n_codewords: int,
                 n_subspaces: int):
        layers.VanillaAttention.__init__(
            self, d_head=d_head,
            p_dropout=p_dropout
        )
        #
        self.quantizer = layers.PQV1(
            d_codeword=d_codeword,
            n_codewords=n_codewords,
            n_subspaces=n_subspaces
        )

    def _get_attn(self,
                  q: torch.Tensor,
                  k: torch.Tensor,
                  attn_mask: torch.Tensor):
        loss_q = self.quantizer('train', z=q)[-1]
        loss_k = self.quantizer('train', z=k)[-1]
        self.register_buffer(
            'loss', loss_q + loss_k, persistent=False
        )
        return layers.VanillaAttention._get_attn(
            self, q, k, attn_mask=attn_mask
        )
