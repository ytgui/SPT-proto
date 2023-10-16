import torch
from torch import nn
from naive_gpt import layers


class LoRARoutedFFN(layers.RoutedFFN):
    def __init__(self,
                 d_lora: int,
                 block_size: int,
                 d_model: int,
                 d_feedforward: int,
                 lora_dropout: float,
                 activation: nn.Module,
                 bias: bool = True):
        layers.RoutedFFN.__init__(
            self,
            block_size=block_size,
            d_model=d_model,
            d_feedforward=d_feedforward,
            activation=activation,
            bias=bias
        )
        # LoRA
        self.fc1 = layers.LoRALinear(
            d_model=d_lora, in_features=d_model,
            out_features=d_feedforward, lora_dropout=lora_dropout
        )
        self.fc2 = layers.LoRALinear(
            d_model=d_lora, in_features=d_feedforward,
            out_features=d_model, lora_dropout=lora_dropout
        )

    @staticmethod
    def from_pretrained(d_lora: int,
                        block_size: int,
                        p_dropout: float,
                        source: layers.Feedforward):
        assert isinstance(
            source, layers.Feedforward
        )
        model = LoRARoutedFFN(
            d_lora=d_lora,
            block_size=block_size,
            d_model=source.d_model,
            d_feedforward=source.d_feedforward,
            activation=source.activation,
            lora_dropout=p_dropout
        )
        output = model.load_state_dict(
            source.state_dict(), strict=False
        )
        if len(output.missing_keys) != 2:
            raise RuntimeError
        return model

    def forward(self, x: torch.Tensor):
        # lora
        weight_1 = self.fc1.lora.dropout(
            torch.matmul(
                self.fc1.lora.right.weight,
                self.fc1.lora.left.weight.T
            )
        ) + self.fc1.weight
        weight_2 = self.fc2.lora.dropout(
            torch.matmul(
                self.fc2.lora.right.weight,
                self.fc2.lora.left.weight.T
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
