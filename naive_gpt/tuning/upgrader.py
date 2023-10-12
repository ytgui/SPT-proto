import torch
from torch import nn
from naive_gpt import layers


class LoRAHandler:
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
                 child: nn.Linear):
        assert isinstance(
            child, nn.Linear
        )
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
                    child: nn.Embedding):
        assert isinstance(
            child, nn.Embedding
        )
        Module = layers.LoRAEmbedding
        new_model = Module.from_pretrained(
            d_model=self.lora_r,
            p_dropout=self.lora_dropout,
            source=child
        )
        print('[UPGRADE]', name, type(child).__name__)
        return new_model


class ModuleUpgrader:
    def __init__(self,
                 handler: object):
        if not hasattr(handler, 'default'):
            raise RuntimeError('requires default method')
        self.handler = handler

    def visit(self, root: nn.Module) -> nn.Module:
        # visit
        named_upgrades = {}
        for name, child in root.named_modules():
            # module
            cls_name = type(child).__name__
            attr_name = 'on' + cls_name
            if hasattr(self.handler, attr_name):
                fn = getattr(self.handler, attr_name)
            else:
                fn = getattr(self.handler, 'default')

            # upgrade
            new_child = fn(name=name, child=child)
            if not isinstance(new_child, nn.Module):
                continue
            named_upgrades[name] = new_child

        # replace
        for path in named_upgrades:
            parent, child = None, root
            for name in path.split('.'):
                parent, child = \
                    child, child.get_submodule(name)
            parent.add_module(
                name, module=named_upgrades[path]
            )

        #
        return root
