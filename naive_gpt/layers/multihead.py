import torch
from torch import nn


class MultiheadAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 attention_fn: nn.Module):
        nn.Module.__init__(self)
        #
        self.d_model = d_model
        self.n_heads = n_heads
        self.attn_fn = attention_fn
        #
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_o = nn.Linear(d_model, d_model)

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
