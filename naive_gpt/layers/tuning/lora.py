import torch
from torch import nn


class LoRABase(nn.Module):
    def __init__(self,
                 d_model: int,
                 in_features: int,
                 out_features: int,
                 p_dropout: float,
                 device: any = None,
                 dtype: any = None):
        nn.Module.__init__(self)
        #
        self.left = nn.Embedding(
            in_features, embedding_dim=d_model,
            device=device, dtype=dtype
        )
        self.right = nn.Embedding(
            out_features, embedding_dim=d_model,
            device=device, dtype=dtype
        )
        self.dropout = nn.Dropout(p_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.right.weight)


class LoRALinear(nn.Linear):
    def __init__(self,
                 d_model: int,
                 in_features: int,
                 out_features: int,
                 lora_dropout: float,
                 bias: bool = True,
                 *args, **kwargs):
        nn.Linear.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            bias=bias, *args, **kwargs
        )
        for param in self.parameters():
            param.requires_grad = False
        # LoRA
        self.lora = LoRABase(
            d_model=d_model,
            in_features=in_features,
            out_features=out_features,
            p_dropout=lora_dropout
        )

    @staticmethod
    def from_pretrained(d_model: int,
                        p_dropout: float,
                        source: nn.Linear):
        #
        model = LoRALinear(
            d_model=d_model,
            in_features=source.in_features,
            out_features=source.out_features,
            bias=source.bias is not None,
            lora_dropout=p_dropout
        )
        output = model.load_state_dict(
            source.state_dict(), strict=False
        )
        if len(output.missing_keys) != 2:
            raise RuntimeError
        return model

    def forward(self, x: torch.Tensor):
        w = self.lora.dropout(
            torch.matmul(
                self.lora.right.weight,
                self.lora.left.weight.T
            )
        )
        return nn.functional.linear(
            x, bias=self.bias,
            weight=self.weight + w
        )


class LoRAEmbedding(nn.Embedding):
    def __init__(self,
                 d_model: int,
                 num_embeddings: int,
                 embedding_dim: int,
                 lora_dropout: float,
                 *args, **kwargs):
        nn.Embedding.__init__(
            self,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            *args, **kwargs
        )
        for param in self.parameters():
            param.requires_grad = False
        # LoRA
        self.lora = LoRABase(
            d_model=d_model,
            in_features=num_embeddings,
            out_features=embedding_dim,
            p_dropout=lora_dropout
        )

    @staticmethod
    def from_pretrained(d_model: int,
                        p_dropout: float,
                        source: nn.Embedding):
        model = LoRAEmbedding(
            d_model=d_model,
            num_embeddings=source.num_embeddings,
            embedding_dim=source.embedding_dim,
            lora_dropout=p_dropout
        )
        output = model.load_state_dict(
            source.state_dict(), strict=False
        )
        if len(output.missing_keys) != 2:
            raise RuntimeError
        return model

    def forward(self, x: torch.Tensor):
        h = self.lora.dropout(
            torch.matmul(
                self.lora.left(x),
                self.lora.right.weight.T
            )
        )
        x = nn.functional.embedding(
            x, weight=self.weight
        )
        return x + h
