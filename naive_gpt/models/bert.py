import copy
import torch
from torch import nn
from naive_gpt import layers


class BertBase(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_layers: int,
                 max_length: int,
                 vocab_size: int,
                 block: nn.Module):
        nn.Module.__init__(self)
        # embeddings
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim=d_model
        )
        self.learned_pe = nn.Embedding(
            max_length, embedding_dim=d_model
        )
        self.token_type = nn.Embedding(
            2, embedding_dim=d_model
        )
        self.init_norm = nn.LayerNorm(d_model)
        # encoder layers
        self.encoders = nn.ModuleList([
            copy.deepcopy(block)
            for _ in range(n_layers)
        ])

    def forward(self,
                x: torch.Tensor,
                token_types: torch.Tensor = None):
        assert x.dim() == 2
        batch_size = x.size(0)
        seq_length = x.size(-1)

        # types
        if token_types is None:
            token_types = torch.zeros_like(
                x, dtype=torch.long
            )

        # position
        indices = torch.arange(
            seq_length, device=x.device
        )
        indices = torch.tile(
            indices, dims=[batch_size]
        ).view_as(x)

        # forward
        h = self.embedding(x)
        h += self.token_type(token_types)
        h += self.learned_pe(indices)
        h = self.init_norm(h)
        for layer in self.encoders:
            h = layer(h, attn_mask=None)
        return h


class BertModel(BertBase):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 n_layers: int,
                 max_length: int,
                 vocab_size: int,
                 d_feedforward: int,
                 p_dropout: float):
        BertBase.__init__(
            self,
            d_model=d_model,
            n_layers=n_layers,
            max_length=max_length,
            vocab_size=vocab_size,
            block=layers.TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                layernorm_fn=nn.LayerNorm(
                    d_model, eps=1e-12
                ),
                attention_fn=layers.VanillaAttention(
                    d_head=d_model // n_heads,
                    p_dropout=p_dropout
                ),
                feedforward_fn=layers.Feedforward(
                    d_model=d_model,
                    d_feedforward=d_feedforward,
                    activation=nn.GELU(),
                    p_dropout=p_dropout
                ),
                attention_bias=True,
                head_first=False,
                pre_norm=False
            )
        )
