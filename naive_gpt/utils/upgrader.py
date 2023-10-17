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
            d_model=self.lora_r,
            p_dropout=self.lora_dropout,
            source=child
        )
        print('[UPGRADE]', name, type(child).__name__,
              '->', type(new_model).__name__)
        return new_model


class SparseLoRAHandler(LoRAHandler):
    def __init__(self,
                 lora_r: int,
                 lora_dropout: float,
                 stage: int):
        LoRAHandler.__init__(
            self, lora_r=lora_r,
            lora_dropout=lora_dropout
        )
        #
        assert stage in [1, 2]
        self.stage = stage

    def onLinear(self,
                 name: str,
                 child: nn.Linear):
        assert isinstance(
            child, nn.Linear
        )
        if self.stage != 1:
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
        if self.stage != 1:
            print('[SKIP]', name, type(child).__name__)
            return
        return LoRAHandler.onEmbedding(
            self, name=name, child=child
        )

    def onVanillaAttention(self,
                           name: str,
                           child: layers.VanillaAttention):
        assert self.stage == 1
        assert isinstance(
            child, layers.VanillaAttention
        )
        Module = layers.SparseVanillaAttentionV1
        new_model = Module(
            d_head=child.d_head, p_dropout=child.p_dropout,
            d_codeword=8, n_codewords=16, n_subspaces=child.d_head // 8
        )
        print('[UPGRADE]', name, type(child).__name__,
              '->', type(new_model).__name__)
        return new_model

    def onRotaryAttention(self,
                          name: str,
                          child: layers.RotaryAttention):
        assert self.stage == 1
        assert isinstance(
            child, layers.RotaryAttention
        )
        Module = layers.SparseRotaryAttentionV1
        new_model = Module(
            d_head=child.d_head, p_dropout=child.p_dropout,
            d_codeword=8, n_codewords=16, n_subspaces=child.d_head // 8
        )
        print('[UPGRADE]', name, type(child).__name__,
              '->', type(new_model).__name__)
        return new_model

    def onSparseVanillaAttentionV1(self,
                                   name: str,
                                   child: layers.SparseVanillaAttentionV1):
        assert self.stage == 2
        assert isinstance(
            child, layers.SparseVanillaAttentionV1
        )
        Module = layers.SparseVanillaAttentionV2
        new_model = Module.from_pretrained(
            source=child
        )
        print('[UPGRADE]', name, type(child).__name__,
              '->', type(new_model).__name__)
        return new_model

    def onSparseRotaryAttentionV1(self,
                                  name: str,
                                  child: layers.SparseRotaryAttentionV1):
        assert self.stage == 2
        assert isinstance(
            child, layers.SparseRotaryAttentionV1
        )
        Module = layers.SparseRotaryAttentionV2
        new_model = Module.from_pretrained(
            source=child
        )
        print('[UPGRADE]', name, type(child).__name__,
              '->', type(new_model).__name__)
        return new_model

    def onFeedforward(self,
                      name: str,
                      child: layers.Feedforward):
        if self.stage != 2:
            print('[SKIP]', name, type(child).__name__)
            return
        assert isinstance(child, layers.Feedforward)
        new_model = layers.LoRARoutedFFN.from_pretrained(
            d_lora=self.lora_r, block_size=child.d_model // 2,
            p_dropout=self.lora_dropout, source=child
        )
        return new_model

    def onFeedforward(self,
                      name: str,
                      child: layers.LLaMaFeedforward):
        if self.stage != 2:
            print('[SKIP]', name, type(child).__name__)
            return
        raise NotImplementedError


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
