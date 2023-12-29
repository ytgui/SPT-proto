import torch
from torch import nn
from naive_gpt import layers


class LoRAHandler:
    def __init__(self,
                 d_lora: int):
        self.d_lora = d_lora

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
            d_lora=self.d_lora, source=child
        )
        print('[UPGRADE]', name, type(child).__name__,
              '->', type(new_model).__name__)
        return new_model

    def onEmbedding(self,
                    name: str,
                    child: nn.Embedding):
        assert isinstance(
            child, nn.Embedding
        )
        Module = layers.LoRAEmbedding
        new_model = Module.from_pretrained(
            d_lora=self.d_lora, source=child
        )
        print('[UPGRADE]', name, type(child).__name__,
              '->', type(new_model).__name__)
        return new_model


class SparseLoRAHandler(LoRAHandler):
    def __init__(self,
                 d_lora: int,
                 stage: str):
        LoRAHandler.__init__(
            self, d_lora=d_lora
        )
        #
        assert stage in [
            'lora', 'pq-v1', 'pq-v2'
        ]
        self.stage = stage

    def onLinear(self,
                 name: str,
                 child: nn.Linear):
        assert isinstance(
            child, nn.Linear
        )
        if self.stage != 'lora':
            print('[SKIP]', name, type(child).__name__)
            return
        return LoRAHandler.onLinear(
            self, name=name, child=child
        )

    def onEmbedding(self,
                    name: str,
                    child: nn.Embedding):
        assert isinstance(
            child, nn.Embedding
        )
        if self.stage != 'lora':
            print('[SKIP]', name, type(child).__name__)
            return
        return LoRAHandler.onEmbedding(
            self, name=name, child=child
        )

    def onVanillaAttention(self,
                           name: str,
                           child: layers.VanillaAttention):
        if self.stage == 'pq-v1':
            Module = layers.SparseVanillaAttentionV1
        elif self.stage == 'pq-v2':
            Module = layers.SparseVanillaAttentionV2
        else:
            print('[SKIP]', name, type(child).__name__)
            return
        assert isinstance(
            child, layers.VanillaAttention
        )
        #
        new_model = Module(
            d_head=child.d_head,
            p_dropout=child.p_dropout,
            d_codeword=8, n_codewords=16
        )
        print('[UPGRADE]', name, type(child).__name__,
              '->', type(new_model).__name__)
        return new_model

    def onRotaryAttention(self,
                          name: str,
                          child: layers.RotaryAttention):
        if self.stage == 'pq-v1':
            Module = layers.SparseRotaryAttentionV1
        elif self.stage == 'pq-v2':
            Module = layers.SparseRotaryAttentionV2
        else:
            print('[SKIP]', name, type(child).__name__)
            return
        assert isinstance(
            child, layers.VanillaAttention
        )
        #
        new_model = Module(
            d_head=child.d_head,
            p_dropout=child.p_dropout,
            d_codeword=8, n_codewords=16
        )
        print('[UPGRADE]', name, type(child).__name__,
              '->', type(new_model).__name__)
        return new_model


class ModuleUpgrader:
    def __init__(self,
                 handler: object):
        if not hasattr(handler, 'default'):
            raise RuntimeError('requires default handler')
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
            if new_child is None or new_child is child:
                continue
            assert isinstance(new_child, nn.Module)
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
