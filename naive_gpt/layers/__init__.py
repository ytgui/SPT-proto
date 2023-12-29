# 3rd
from .collect.triton_flash import FlashAttention
try:
    from .collect.fast_wrapper import LocalAttention
    from .collect.fast_wrapper import ReformerAttention
    from .collect.fast_wrapper import RotaryLocalAttention
    from .collect.fast_wrapper import RotaryReformerAttention
except ImportError as e:
    print('import fast transformer error', e)

# utils
from .basic.utils import FnModule
from .basic.utils import LlamaRMSNorm

# PQ
from .basic.quantizer import PQV1
from .basic.quantizer import PQV2

# Attention
from .basic.position import RotaryEmbedding
from .basic.attention import VanillaAttention
from .basic.attention import RotaryAttention

# Feed-forward
from .basic.feedforward import Feedforward
from .basic.feedforward import LLaMaFeedforward

# Routed FFN
from .sparse.feedforward import RoutedFFN
from .sparse.feedforward import RoutedLLaMaFFN

# Transformer
from .basic.transformer import MultiheadAttention
from .basic.transformer import TransformerBlock

# LoRA
from .tuning.lora import LoRALinear
from .tuning.lora import LoRAEmbedding
from .tuning.lora_ffn import LoRARoutedFFN
from .tuning.lora_ffn import LoRARoutedLLaMaFFN

# PQ Attention
from .sparse.attention import SparseVanillaAttentionV1
from .sparse.attention import SparseVanillaAttentionV2
from .sparse.attention import SparseRotaryAttentionV1
from .sparse.attention import SparseRotaryAttentionV2
