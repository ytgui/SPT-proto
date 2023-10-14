import copy
import torch
from torch import nn


class MultiheadAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 attention_fn: nn.Module,
                 bias: bool):
        nn.Module.__init__(self)
        #
        self.d_model = d_model
        self.n_heads = n_heads
        self.attn_fn = attention_fn
        #
        self.linear_q = nn.Linear(d_model, d_model, bias=bias)
        self.linear_k = nn.Linear(d_model, d_model, bias=bias)
        self.linear_v = nn.Linear(d_model, d_model, bias=bias)
        self.linear_o = nn.Linear(d_model, d_model, bias=bias)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                attn_mask: torch.Tensor = None):
        # TODO: cross attention
        assert q.size(0) == k.size(0) == v.size(0)

        # [N, S, E]
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # [N, S, H, E]
        q = q.view([q.size(0), q.size(1), self.n_heads, -1])
        k = k.view([k.size(0), k.size(1), self.n_heads, -1])
        v = v.view([v.size(0), v.size(1), self.n_heads, -1])

        # [N, S, H, E]
        y: torch.Tensor = self.attn_fn(
            q, k, v, attn_mask=attn_mask
        )

        # [N, S, H * E]
        y = y.view([y.size(0), y.size(1), -1])
        y = self.linear_o(y)

        return y


class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 layernorm_fn: nn.Module,
                 attention_fn: nn.Module,
                 feedforward_fn: nn.Module,
                 attention_bias: bool,
                 pre_norm: bool):
        nn.Module.__init__(self)
        #
        self.pre_norm = pre_norm
        # mha
        self.mha = MultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            attention_fn=attention_fn,
            bias=attention_bias
        )
        # ffn
        self.ffd = copy.deepcopy(feedforward_fn)
        # norm
        self.norm1 = copy.deepcopy(layernorm_fn)
        self.norm2 = copy.deepcopy(layernorm_fn)

    def forward(self,
                x: torch.Tensor,
                attn_mask: torch.Tensor = None):
        assert x.dim() == 3
        #
        if self.pre_norm:
            h = self.norm1(x)
            x = x + self.mha(
                h, h, h, attn_mask=attn_mask
            )
            h = self.norm2(x)
            x = x + self.ffd(h)
        else:
            h = self.mha(
                x, x, x, attn_mask=attn_mask
            )
            x = self.norm1(x + h)
            h = self.ffd(x)
            x = self.norm2(x + h)
        return x
