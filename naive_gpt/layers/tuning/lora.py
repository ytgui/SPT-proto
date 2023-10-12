import torch
from torch import nn
from naive_gpt import layers


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


class LoRARoutedFFN(layers.RoutedFFN):
    def __init__(self,
                 d_model: int,
                 block_size: int,
                 in_features: int,
                 out_features: int,
                 lora_dropout: float,
                 actication: nn.Module,
                 bias: bool = True):
        layers.RoutedFFN.__init__(
            self,
            block_size=block_size,
            in_features=in_features,
            out_features=out_features,
            actication=actication,
            bias=bias
        )
        for param in self.parameters():
            param.requires_grad = False
        # LoRA
        self.lora1 = LoRABase(
            d_model=d_model,
            in_features=in_features,
            out_features=out_features,
            p_dropout=lora_dropout
        )
        self.lora2 = LoRABase(
            d_model=d_model,
            in_features=out_features,
            out_features=in_features,
            p_dropout=lora_dropout
        )

    @staticmethod
    def from_pretrained(d_model: int,
                        p_dropout: float,
                        source: layers.RoutedFFN):
        assert isinstance(source, layers.RoutedFFN)
        model = LoRARoutedFFN(
            d_model=d_model,
            block_size=source.block_size,
            in_features=source.in_features,
            out_features=source.out_features,
            actication=source.activation,
            lora_dropout=p_dropout
        )
        output = model.load_state_dict(
            source.state_dict(), strict=False
        )
        if len(output.missing_keys) != 4:
            raise RuntimeError
        return model

    def forward(self, x: torch.Tensor):
        x_size = x.size()

        # lora
        weight_1 = self.lora1.dropout(
            torch.matmul(
                self.lora1.right.weight,
                self.lora1.left.weight.T
            )
        ) + self.fc1.weight
        weight_2 = self.lora1.dropout(
            torch.matmul(
                self.lora2.right.weight,
                self.lora2.left.weight.T
            )
        ) + self.fc2.weight

        # route
        x = x.view(
            [-1, self.in_features]
        )
        prob = self.router(x)
        topk = torch.topk(
            prob, k=self.n_blocks // 4,
            dim=-1, sorted=False
        )
        indices = topk.indices.tolist()

        # grouping
        grouping: list[list] = [
            [] for _ in range(self.n_blocks)
        ]
        for b, items in enumerate(indices):
            for expert in items:
                grouping[expert].append(b)

        #
        h = layers.RoutedLinearRow.apply(
            x,
            self.fc1.bias.view(
                [self.n_blocks, self.block_size]
            ),
            weight_1.view(
                [self.n_blocks, self.block_size, -1]
            ),
            grouping
        )
        h = self.activation(h)
        y = layers.RoutedLinearCol.apply(
            h, self.fc2.bias,
            weight_2.view(
                [-1, self.n_blocks, self.block_size]
            ),
            grouping
        )
        return y.view(x_size)
