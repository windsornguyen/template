import torch

from transformers import PretrainedConfig


class FlashSTUConfig(PretrainedConfig):
    model_type = "FlashSTU"

    def __init__(
        self,
        bsz: int = 4,
        n_embd: int = 2304,
        n_heads: int = 9,
        n_layers: int = 7,
        seq_len: int = 8192,
        window_size: int = 1024,
        vocab_size: int = 200064,
        mlp_scale: int = 12,
        bias: bool = False,
        dropout: float = 0.0,
        num_eigh: int = 24,
        use_hankel_L: bool = False,
        use_flash_fft: bool = True,
        use_approx: bool = True,
        softcap: float = 50.0,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bsz = bsz
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.mlp_scale = mlp_scale
        self.bias = bias
        self.dropout = dropout
        self.num_eigh = num_eigh
        self.use_hankel_L = use_hankel_L
        self.use_flash_fft = use_flash_fft
        self.use_approx = use_approx
        self.softcap = softcap
        self.torch_dtype = torch_dtype
