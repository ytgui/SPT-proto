import torch
from torch import nn
from naive_gpt import layers
from .lora import LoRABase


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

        #
        bias_1 = self.fc1.bias.view(
            [self.n_blocks, self.block_size]
        )
        weight_1 = weight_1.view(
            [self.n_blocks, self.block_size, -1]
        )
        weight_2 = weight_2.view(
            [-1, self.n_blocks, self.block_size]
        )
        weight_2 = torch.permute(
            weight_2, dims=[1, 2, 0]
        )
        return self._apply_ffn(
            x, bias_1=bias_1, weight_1=weight_1,
            weight_2=weight_2.contiguous()
        )
