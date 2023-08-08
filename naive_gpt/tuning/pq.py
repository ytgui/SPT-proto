import torch
from torch import nn
from naive_torch import layers, models


class QuantizedAttention(layers.VanillaAttention):
    def __init__(self,
                 d_head: int,
                 d_codeword: int,
                 n_codewords: int,
                 p_dropout: float):
        layers.VanillaAttention.__init__(
            self, d_head=d_head, p_dropout=p_dropout
        )
        #
        self.quantizer = models.PQ(
            d_codeword=d_codeword,
            n_codewords=n_codewords,
            n_subspaces=d_head // d_codeword
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
            self, q=q, k=k, attn_mask=attn_mask
        )


class QuantizedUpgrader:
    def __init__(self,
                 d_codeword: int,
                 n_codewords: int):
        self.d_codeword = d_codeword
        self.n_codewords = n_codewords

    def default(self,
                name: str,
                child: nn.Module):
        print('[SKIP]', name, type(child).__name__)

    def onVanillaAttention(self,
                           name: str,
                           child: nn.Module):
        assert isinstance(
            child, layers.VanillaAttention
        )
        new_model = QuantizedAttention(
            d_head=child.d_head,
            d_codeword=self.d_codeword,
            n_codewords=self.n_codewords,
            p_dropout=child.p_dropout
        )
        print('[UPGRADE]', name, type(child).__name__)
        return new_model
