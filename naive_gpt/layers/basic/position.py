import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self,
                 n_embeddings: int,
                 d_model: int,
                 base: float = 10000.0):
        nn.Module.__init__(self)
        #
        assert d_model % 2 == 0
        i = torch.arange(0, d_model, step=2)
        inv_freq = 1.0 / (base ** (i / d_model))
        #
        t = torch.arange(n_embeddings)
        freqs = torch.einsum(
            'i, j -> ij', t, inv_freq
        )
        # embed: [S, E]
        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor
        embed = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', embed.cos())
        self.register_buffer('sin_cached', embed.sin())

    def rotate_half(self, x: torch.Tensor):
        x1, x2 = torch.chunk(
            x, chunks=2, dim=-1
        )
        y = torch.cat([-x2, x1], dim=-1)
        return y

    def forward(self,
                x: torch.Tensor,
                ids: torch.Tensor):
        assert x.dim() == 4
        assert ids.dim() == 1

        # [S, E] -> [1, S, 1, E]
        cos = self.cos_cached[ids][None, :, None, :]
        sin = self.sin_cached[ids][None, :, None, :]

        # Rotary * x = [
        #     [x1 cos - x2 sin],
        #     [x1 sin + x2 cos]
        # ]
        return cos * x + sin * self.rotate_half(x)
