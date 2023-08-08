import torch
from torch import nn
from naive_torch import layers


class LoRAUpgrader:
    def __init__(self,
                 lora_r: int,
                 lora_dropout: float):
        self.lora_r = lora_r
        self.lora_dropout = lora_dropout

    def default(self,
                name: str,
                child: nn.Module):
        print('[SKIP]', name, type(child).__name__)

    def onLinear(self,
                 name: str,
                 child: nn.Module):
        Module = layers.LoRALinear
        new_model = Module.from_pretrained(
            d_model=self.lora_r,
            p_dropout=self.lora_dropout,
            source=child
        )
        print('[UPGRADE]', name, type(child).__name__)
        return new_model

    def onEmbedding(self,
                    name: str,
                    child: nn.Module):
        Module = layers.LoRAEmbedding
        new_model = Module.from_pretrained(
            d_model=self.lora_r,
            p_dropout=self.lora_dropout,
            source=child
        )
        print('[UPGRADE]', name, type(child).__name__)
        return new_model
