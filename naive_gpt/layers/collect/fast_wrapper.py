import torch
from torch import nn
from naive_gpt import layers
from fast_transformers import attention, masking


class LocalAttention(nn.Module):
    def __init__(self,
                 d_head: int,
                 p_dropout: float,
                 local_context: int):
        nn.Module.__init__(self)
        #
        self.d_head = d_head
        self.p_dropout = p_dropout
        self.local_context = local_context
        self.scaling = float(d_head) ** -0.5

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
        N = q.size(0)
        L = q.size(1)
        length_mask = masking.LengthMask(
            q.new_full((N,), L, dtype=torch.int64)
        )
        attention_fn = attention.LocalAttention(
            local_context=self.local_context,
            attention_dropout=self.p_dropout,
            softmax_temp=self.scaling
        )
        if attn_mask is None:
            attn_mask = masking.FullMask(
                L, device=q.device
            )
        y = attention_fn(
            q, k, v, attn_mask=attn_mask,
            query_lengths=length_mask,
            key_lengths=length_mask
        )
        return y


class RotaryLocalAttention(LocalAttention):
    def __init__(self,
                 d_head: int,
                 p_dropout: float,
                 local_context: int,
                 max_length: int = 2048):
        LocalAttention.__init__(
            self, d_head=d_head,
            p_dropout=p_dropout,
            local_context=local_context
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

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                attn_mask: torch.Tensor = None):
        # rotary
        q = self.embedding(
            q, ids=self.cached_ids[:q.size(1)]
        )
        k = self.embedding(
            k, ids=self.cached_ids[:k.size(1)]
        )
        return LocalAttention.forward(
            self, q, k, v, attn_mask=attn_mask
        )


class ReformerAttention(nn.Module):
    def __init__(self,
                 d_head: int,
                 p_dropout: float):
        nn.Module.__init__(self)
        #
        self.d_head = d_head
        self.p_dropout = p_dropout
        self.scaling = float(d_head) ** -0.5

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
        N = q.size(0)
        L = q.size(1)
        length_mask = masking.LengthMask(
            q.new_full((N,), L, dtype=torch.int64)
        )
        attention_fn = attention.ReformerAttention(
            masked=attn_mask is not None,
            attention_dropout=self.p_dropout,
            softmax_temp=self.scaling
        )
        if attn_mask is None:
            attn_mask = masking.FullMask(
                L, device=q.device
            )
        y = attention_fn(
            q, k, v, attn_mask=attn_mask,
            query_lengths=length_mask,
            key_lengths=length_mask
        )
        return y


class RotaryReformerAttention(ReformerAttention):
    def __init__(self,
                 d_head: int,
                 p_dropout: float,
                 max_length: int = 2048):
        ReformerAttention.__init__(
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

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                attn_mask: torch.Tensor = None):
        # rotary
        q = self.embedding(
            q, ids=self.cached_ids[:q.size(1)]
        )
        k = self.embedding(
            k, ids=self.cached_ids[:k.size(1)]
        )
        return ReformerAttention.forward(
            self, q, k, v, attn_mask=attn_mask
        )
