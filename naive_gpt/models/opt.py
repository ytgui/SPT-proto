import copy
import torch
from torch import nn
from naive_gpt import layers


class OPTBase(nn.Module):
    PE_OFFSET = 2

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
            max_length + self.PE_OFFSET,
            embedding_dim=d_model
        )
        # decoder layers
        self.decoders = nn.ModuleList([
            copy.deepcopy(block)
            for _ in range(n_layers)
        ])
        # output layer norm
        self.final_norm = nn.LayerNorm(d_model)
        # casual LM output
        self.lm_output = nn.Linear(
            d_model, vocab_size, bias=False
        )
        # attention mask
        ones = torch.tril(
            torch.ones(
                [max_length + self.PE_OFFSET,
                 max_length + self.PE_OFFSET],
                dtype=torch.bool
            )
        )
        attn_mask = torch.where(
            ones, 0.0, float('-inf')
        )
        self.attn_mask: torch.Tensor
        self.register_buffer(
            'attn_mask', attn_mask
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2

        # mask
        seq_length = x.size(-1)
        attn_mask = self.attn_mask[
            :seq_length, :seq_length
        ]

        # position
        batch_size = x.size(0)
        indices = torch.arange(
            seq_length, device=x.device
        ) + self.PE_OFFSET
        indices = torch.tile(
            indices, dims=[batch_size]
        ).view_as(x)

        # forward
        h = self.embedding(x)
        h += self.learned_pe(indices)
        for layer in self.decoders:
            h = layer(h, attn_mask=attn_mask)
        h = self.final_norm(h)
        h = self.lm_output(h)
        return h


class OPTModel(OPTBase):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 n_layers: int,
                 max_length: int,
                 vocab_size: int,
                 d_feedforward: int,
                 p_dropout: float):
        OPTBase.__init__(
            self,
            d_model=d_model,
            n_layers=n_layers,
            max_length=max_length,
            vocab_size=vocab_size,
            block=layers.TransformerBlock(
                d_model=d_model, n_heads=n_heads,
                layernorm_fn=nn.LayerNorm(d_model),
                attention_fn=layers.VanillaAttention(
                    d_head=d_model // n_heads,
                    p_dropout=p_dropout
                ),
                feedforward_fn=layers.Feedforward(
                    d_model=d_model,
                    d_feedforward=d_feedforward,
                    activation=nn.ReLU(),
                    p_dropout=p_dropout
                ),
                attention_bias=True,
                pre_norm=True
            )
        )
