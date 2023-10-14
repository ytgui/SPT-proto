import copy
import torch
from torch import nn
from naive_gpt import layers


class LLaMABase(nn.Module):
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
        # decoder layers
        self.decoders = nn.ModuleList([
            copy.deepcopy(block)
            for _ in range(n_layers)
        ])
        # output layer norm
        self.final_norm = layers.LlamaRMSNorm(d_model)
        # casual LM output
        self.lm_output = nn.Linear(
            d_model, vocab_size, bias=False
        )
        # attention mask
        ones = torch.tril(
            torch.ones(
                [max_length, max_length],
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

        # forward
        h = self.embedding(x)
        for layer in self.decoders:
            h = layer(h, attn_mask=attn_mask)
        h = self.final_norm(h)
        h = self.lm_output(h)
        return h


class LLaMAModel(LLaMABase):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 n_layers: int,
                 max_length: int,
                 vocab_size: int,
                 d_feedforward: int,
                 p_dropout: float):
        LLaMABase.__init__(
            self,
            d_model=d_model,
            n_layers=n_layers,
            max_length=max_length,
            vocab_size=vocab_size,
            block=layers.TransformerBlock(
                d_model=d_model, n_heads=n_heads,
                layernorm_fn=layers.LlamaRMSNorm(d_model),
                attention_fn=layers.RotaryAttention(
                    d_head=d_model // n_heads,
                    p_dropout=p_dropout
                ),
                feedforward_fn=layers.LLaMaFeedforward(
                    d_model=d_model,
                    d_feedforward=d_feedforward,
                    activation=nn.SiLU()
                ),
                attention_bias=False,
                pre_norm=True
            )
        )
