import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from transformers import PreTrainedModel, PretrainedConfig

from modules import Attention

try:
    from flash_attn.modules.mlp import GatedMlp as MLP
    triton_mlp = True
except ImportError as e:
    print(f"Unable to import Triton-based MLP: {e}. Falling back to vanilla SwiGLU MLP instead.")
    from modules import MLP
    triton_mlp = False

try:
    from flash_attn.ops.triton.layer_norm import RMSNorm
except ImportError as e:
    print(f"Unable to import Triton-based RMSNorm: {e}. Falling back to PyTorch implementation.")
    from torch.nn import RMSNorm


class Layer(nn.Module):
    def __init__(self, config) -> None:
        super(Layer, self).__init__()
        # For more complex %-split arrangements, see https://arxiv.org/pdf/2406.07887
        self.rn_1 = RMSNorm(config.n_embd, dtype=config.torch_dtype)
        self.attn_1 = Attention(config)
        self.rn_2 = RMSNorm(config.n_embd, dtype=config.torch_dtype)
        self.attn_2 = Attention(config)
        self.rn_3 = RMSNorm(config.n_embd, dtype=config.torch_dtype)
        self.mlp = MLP(
            config.n_embd, 
            config.n_embd * config.mlp_scale, 
            activation=F.silu,
            bias1=config.bias,
            bias2=config.bias,
            dtype=config.torch_dtype,
        ) if triton_mlp else MLP(config, dtype=config.torch_dtype)
        self.rn_4 = RMSNorm(config.n_embd, dtype=config.torch_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_1(self.rn_1(x))
        x = x + self.mlp(self.rn_2(x))
        x = x + self.attn_2(self.rn_3(x))
        x = x + self.mlp(self.rn_4(x))
        return x

class TransformerConfig(PretrainedConfig):
    model_type = "llama"

    def __init__(
        self,
        bsz: int = 8,
        n_embd: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        seq_len: int = 4096,
        vocab_size: int = 200064,
        mlp_scale: int = 4,
        bias: bool = False,
        dropout: float = 0.0,
        softcap: bool = 50.0,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bsz = bsz
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.mlp_scale = mlp_scale
        self.bias = bias
        self.dropout = dropout
        self.softcap = softcap
        self.torch_dtype = torch_dtype


class Transformer(PreTrainedModel):
    config_class = TransformerConfig

    def __init__(self, config) -> None:
        super(Transformer, self).__init__(config)
        self.config = config
        self.n_layers = config.n_layers
        self.n_embd = config.n_embd
        self.mlp_scale = config.mlp_scale
        self.seq_len = config.seq_len
        self.vocab_size = config.vocab_size
        self.dropout = config.dropout
        self.bias = config.bias

        self.transformer = nn.ModuleDict(
            dict(
                tok_emb=nn.Embedding(self.vocab_size, self.n_embd, dtype=config.torch_dtype),
                dropout=nn.Dropout(self.dropout),
                hidden=nn.ModuleList(
                    [
                        Layer(self.config)
                        for _ in range(self.n_layers)
                    ]
                ),
                rn_f=RMSNorm(self.n_embd, dtype=config.torch_dtype)
            )
        )

        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=self.bias, dtype=config.torch_dtype)

        self.std = (self.n_embd) ** -0.5
        self.apply(self._init_weights)
        print("Model Parameter Count: %.2fM\n" % (self._get_num_params() / 1e6,))

    def forward(
        self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tok_emb = self.transformer.tok_emb(x)
        x = self.transformer.dropout(tok_emb)

        for block in self.transformer.hidden:
            x = block(x)
        x = self.transformer.rn_f(x)
        y_hat = self.lm_head(x)

        return y_hat

    def _get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                self.std *= (2 * self.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            weight_norm(module)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            weight_norm(module)
