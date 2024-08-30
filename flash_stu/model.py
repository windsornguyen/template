import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel

from stu import STU
from modules import Attention
from utils import get_spectral_filters, nearest_power_of_two
from flash_stu.config import FlashSTUConfig

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

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
except ImportError as e:
    print(f"Unable to import Triton-based cross entropy loss: {e}. Falling back to PyTorch implementation.")
    from torch.nn import CrossEntropyLoss

class Block(nn.Module):
    def __init__(self, config, phi, n) -> None:
        super(Block, self).__init__()
        # For more complex %-split arrangements, see https://arxiv.org/pdf/2406.07887
        self.rn_1 = RMSNorm(config.n_embd, dtype=config.torch_dtype)
        self.stu = STU(config, phi, n)
        self.rn_2 = RMSNorm(config.n_embd, dtype=config.torch_dtype)
        self.attn = Attention(config)
        self.rn_3 = RMSNorm(config.n_embd, dtype=config.torch_dtype)
        self.mlp = MLP(
            config.n_embd, 
            config.n_embd * config.mlp_scale, 
            activation=F.silu, # Use SwiGLU
            bias1=config.bias,
            bias2=config.bias,
            dtype=config.torch_dtype,
        ) if triton_mlp else MLP(config, dtype=config.torch_dtype)
        self.rn_4 = RMSNorm(config.n_embd, dtype=config.torch_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.stu(self.rn_1(x))
        x = x + self.mlp(self.rn_2(x))
        x = x + self.attn(self.rn_3(x))
        x = x + self.mlp(self.rn_4(x))
        return x

class FlashSTU(PreTrainedModel):
    config_class = FlashSTUConfig

    def __init__(self, config) -> None:
        super(FlashSTU, self).__init__(config)
        self.config = config
        self.n_layers = config.n_layers
        self.n_embd = config.n_embd
        self.mlp_scale = config.mlp_scale
        self.seq_len = config.seq_len
        self.n = nearest_power_of_two(self.seq_len * 2 - 1, round_up=True)
        self.vocab_size = config.vocab_size
        self.K = config.num_eigh
        self.use_hankel_L = config.use_hankel_L
        self.phi = get_spectral_filters(self.seq_len, self.K, self.use_hankel_L)
        self.use_approx = config.use_approx
        self.dropout = config.dropout
        self.bias = config.bias
        self.loss_fn = CrossEntropyLoss()

        self.flash_stu = nn.ModuleDict(
            dict(
                tok_emb=nn.Embedding(self.vocab_size, self.n_embd, dtype=config.torch_dtype),
                dropout=nn.Dropout(self.dropout),
                hidden=nn.ModuleList(
                    [
                        Block(self.config, self.phi, self.n)
                        for _ in range(self.n_layers)
                    ]
                ),
                rn_f=RMSNorm(config.n_embd, dtype=config.torch_dtype)
            )
        )
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=self.bias, dtype=config.torch_dtype)
        self.flash_stu.tok_emb.weight = self.lm_head.weight

        self.std = (self.n_embd) ** -0.5
        self.apply(self._init_weights)
        print("Model Parameter Count: %.2fM\n" % (self._get_num_params() / 1e6,))

    def forward(self, x: torch.Tensor) -> torch.tensor:
        tok_emb = self.flash_stu.tok_emb(x)
        x = self.flash_stu.dropout(tok_emb)

        for block in self.flash_stu.hidden:
            x = block(x)
        x = self.flash_stu.rn_f(x)
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
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
        elif isinstance(module, STU):
            if self.use_approx:
                torch.nn.init.xavier_normal_(module.M_inputs)
                torch.nn.init.xavier_normal_(module.M_filters)
            else:
                torch.nn.init.xavier_normal_(module.M_phi_plus)
                torch.nn.init.xavier_normal_(module.M_phi_minus)
