import torch
from torch import nn
from naive_gpt import layers


class VanillaAttention(nn.Module):
    def __init__(self,
                 d_head: int,
                 p_dropout: float):
        nn.Module.__init__(self)
        #
        self.d_head = d_head
        self.p_dropout = p_dropout
        self.scaling = float(d_head) ** -0.5
        self.dropout = nn.Dropout(p_dropout)

    def _get_attn(self,
                  q: torch.Tensor,
                  k: torch.Tensor,
                  attn_mask: torch.Tensor):
        # q, k, v: [N, S, A, E]
        attn = torch.einsum(
            'niae, njae -> naij', q, k
        )
        if attn_mask is not None:
            attn += attn_mask
        attn = torch.softmax(
            self.scaling * attn, dim=-1
        )
        return self.dropout(attn)

    def _apply_attn(self,
                    attn: torch.Tensor,
                    v: torch.Tensor):
        # apply attn: [N, A, S, E]
        y = torch.einsum(
            'naij, njae -> niae', attn, v
        )
        return y.contiguous()

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                attn_mask: torch.Tensor = None):
        assert q.dim() == 4
        assert k.dim() == 4
        assert v.dim() == 4
        assert q.size(0) == k.size(0)
        assert q.size(0) == v.size(0)

        #
        attn = self._get_attn(
            q=q, k=k, attn_mask=attn_mask
        )
        y = self._apply_attn(attn, v=v)
        return y


class RotaryAttention(VanillaAttention):
    def __init__(self,
                 d_head: int,
                 p_dropout: float,
                 max_length: int = 2048):
        VanillaAttention.__init__(
            self, d_head=d_head, p_dropout=p_dropout
        )
        #
        self.embedding = layers.RotaryEmbedding(
            n_embeddings=max_length, d_model=d_head
        )
        self.cached_ids: torch.Tensor
        self.register_buffer(
            'cached_ids', torch.arange(max_length)
        )
        self.max_length = max_length

    def _get_attn(self,
                  q: torch.Tensor,
                  k: torch.Tensor,
                  attn_mask: torch.Tensor):
        # rotary
        q = self.embedding(
            q, ids=self.cached_ids[:q.size(1)]
        )
        k = self.embedding(
            k, ids=self.cached_ids[:k.size(1)]
        )
        return VanillaAttention._get_attn(
            self, q, k, attn_mask=attn_mask
        )
