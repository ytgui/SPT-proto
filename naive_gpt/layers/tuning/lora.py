import torch
from torch import nn


class LoRABase(nn.Module):
    def __init__(self,
                 d_lora: int,
                 in_features: int,
                 out_features: int,
                 device: any = None,
                 dtype: any = None):
        nn.Module.__init__(self)
        #
        self.left = nn.Embedding(
            in_features, embedding_dim=d_lora,
            device=device, dtype=dtype
        )
        self.right = nn.Embedding(
            out_features, embedding_dim=d_lora,
            device=device, dtype=dtype
        )
        # scaling not used
        self.scaling = 1.0 / d_lora
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.right.weight)


class LoRALinear(nn.Linear):
    def __init__(self,
                 d_lora: int,
                 in_features: int,
                 out_features: int,
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
            d_lora=d_lora,
            in_features=in_features,
            out_features=out_features
        )

    @staticmethod
    def from_pretrained(d_lora: int,
                        source: nn.Linear):
        #
        model = LoRALinear(
            d_lora=d_lora,
            in_features=source.in_features,
            out_features=source.out_features,
            bias=source.bias is not None
        )
        output = model.load_state_dict(
            source.state_dict(), strict=False
        )
        if len(output.missing_keys) != 2:
            raise RuntimeError
        return model

    def forward(self, x: torch.Tensor):
        y = nn.functional.linear(
            x, bias=self.bias, weight=self.weight
        )
        y += torch.matmul(
            torch.matmul(
                x, self.lora.left.weight
            ),
            self.lora.right.weight.T
        )
        return y


class LoRAEmbedding(nn.Embedding):
    def __init__(self,
                 d_lora: int,
                 num_embeddings: int,
                 embedding_dim: int,
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
            d_lora=d_lora,
            in_features=num_embeddings,
            out_features=embedding_dim
        )

    @staticmethod
    def from_pretrained(d_lora: int,
                        source: nn.Embedding):
        model = LoRAEmbedding(
            d_lora=d_lora,
            num_embeddings=source.num_embeddings,
            embedding_dim=source.embedding_dim
        )
        output = model.load_state_dict(
            source.state_dict(), strict=False
        )
        if len(output.missing_keys) != 2:
            raise RuntimeError
        return model

    def forward(self, x: torch.Tensor):
        h = torch.matmul(
            self.lora.left(x),
            self.lora.right.weight.T
        )
        x = nn.functional.embedding(
            x, weight=self.weight
        )
        return x + h
