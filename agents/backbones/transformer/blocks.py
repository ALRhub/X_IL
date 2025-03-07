from matplotlib.pyplot import cla
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from inspect import isfunction

from typing import Optional, Tuple

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
import einops


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# RMSNorm -- Better, simpler alternative to LayerNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale, self.eps = dim ** -0.5, eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


# SwishGLU -- A Gated Linear Unit (GLU) with the Swish activation; always better than GELU MLP!
class SwishGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.act, self.project = nn.SiLU(), nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected, gate = self.project(x).tensor_split(2, dim=-1)
        return projected * self.act(gate)


class Attention(nn.Module):
    def __init__(
            self,
            n_embd: int,
            n_head: int,
            attn_pdrop: float,
            resid_pdrop: float,
            block_size: int = 100,
            causal: bool = False,
            bias=False,
            qk_norm: bool = False,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd
        self.causal = causal

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash and causal:
            print("WARNING: Using slow attention. Flash Attention requires PyTorch >= 2.0")
        # Dynamically compute causal mask instead of using a fixed bias buffer
        self.block_size = block_size
        self.qk_norm = qk_norm
        # init qk norm if enabled
        if self.qk_norm:
            self.q_norm = RMSNorm(n_embd // self.n_head, eps=1e-6)
            self.k_norm = RMSNorm(n_embd // self.n_head, eps=1e-6)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x, context=None, custom_attn_mask=None):
        B, T, C = x.size()

        if context is not None:
            k = self.key(context).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.value(context).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        else:
            k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=custom_attn_mask,
                                                                 dropout_p=self.attn_dropout.p if self.training else 0,
                                                                 is_causal=self.causal)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # Optimize custom attention masking
            if custom_attn_mask is not None:
                att = att.masked_fill(custom_attn_mask == 0, float('-inf'))
            elif self.causal:
                # Dynamically compute causal mask based on current sequence length T
                causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
                att = att.masked_fill(causal_mask == 0, float('-inf'))

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(
            self,
            n_embd: int,
            bias: bool,
            use_swish: bool = True,
            use_relus: bool = False,
            dropout: float = 0,
    ):
        super().__init__()
        layers = []

        if use_swish:
            layers.append(SwishGLU(n_embd, 4 * n_embd))
        else:
            layers.append(nn.Linear(n_embd, 4 * n_embd, bias=bias))
            if use_relus:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.GELU())

        layers.append(nn.Linear(4 * n_embd, n_embd, bias=bias))
        layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Block(nn.Module):
    def __init__(
            self,
            n_embd: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            mlp_pdrop: float,
            block_size: int = 100,
            causal: bool = True,
            use_cross_attention: bool = False,
            bias: bool = False,  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
            qk_norm: bool = True,
    ):
        super().__init__()
        self.ln_1 = RMSNorm(n_embd, eps=1e-6)
        self.attn = Attention(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size, causal, bias, qk_norm)
        self.use_cross_attention = use_cross_attention

        if self.use_cross_attention:
            self.cross_att = Attention(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size, causal, bias, qk_norm)
            self.ln3 = RMSNorm(n_embd, eps=1e-6)

        self.ln_2 = RMSNorm(n_embd, eps=1e-6)
        self.mlp = MLP(n_embd, bias=bias, dropout=mlp_pdrop)

    def forward(self, x, context=None, custom_attn_mask=None):
        x = x + self.attn(self.ln_1(x), custom_attn_mask=custom_attn_mask)
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.ln3(x), context, custom_attn_mask=custom_attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class AdaLNZero(nn.Module):
    """
    AdaLN-Zero modulation for conditioning.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # Initialize weights and biases to zero
        # nn.init.zeros_(self.modulation[1].weight)
        # nn.init.zeros_(self.modulation[1].bias)

    def forward(self, c):
        return self.modulation(c).chunk(6, dim=-1)


def modulate(x, shift, scale):
    return shift + (x * (scale))


class ConditionedBlock(Block):
    """
    Block with AdaLN-Zero conditioning.
    """

    def __init__(
            self,
            n_embd: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            mlp_pdrop: float,
            block_size: int = 100,
            causal: bool = True,
            use_cross_attention: bool = False,
            bias: bool = False,  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
            qk_norm: bool = True,
    ):
        super().__init__(n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop, block_size, causal,
                         use_cross_attention, bias, qk_norm)

        self.adaLN_zero = AdaLNZero(n_embd)

    def forward(self, x, c, context=None, custom_attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_zero(c)

        # Attention with modulation
        x_attn = self.ln_1(x)
        x_attn = modulate(x_attn, shift_msa, scale_msa)
        x = x + gate_msa * self.attn(x_attn, custom_attn_mask=custom_attn_mask)

        # Cross attention if used
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.ln3(x), context, custom_attn_mask=custom_attn_mask)

        # MLP with modulation
        x_mlp = self.ln_2(x)
        x_mlp = modulate(x_mlp, shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_mlp)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            n_layers: int,
            block_size: int = 100,
            causal: bool = True,
            qk_norm: bool = True,
            bias: bool = False,
            mlp_pdrop: float = 0,
            use_cross_attention: bool = False
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[Block(
                embed_dim,
                n_heads,
                attn_pdrop,
                resid_pdrop,
                mlp_pdrop,
                block_size,
                causal=causal,
                use_cross_attention=use_cross_attention,
                bias=bias,
                qk_norm=qk_norm,
            )
                for _ in range(n_layers)]
        )
        self.ln = RMSNorm(embed_dim, eps=1e-6)

    def forward(self, x, custom_attn_mask=None):
        for layer in self.blocks:
            x = layer(x, custom_attn_mask=custom_attn_mask)
        x = self.ln(x)
        return x


class TransformerFiLMEncoder(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            n_layers: int,
            block_size: int = 100,
            causal: bool = True,
            qk_norm: bool = True,
            bias: bool = False,
            mlp_pdrop: float = 0,
            use_cross_attention: bool = False
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ConditionedBlock(
                embed_dim,
                n_heads,
                attn_pdrop,
                resid_pdrop,
                mlp_pdrop,
                block_size,
                causal=causal,
                use_cross_attention=use_cross_attention,
                bias=bias,
                qk_norm=qk_norm,
            )
                for _ in range(n_layers)]
        )
        self.ln = RMSNorm(embed_dim, eps=1e-6)

    def forward(self, x, c):
        for layer in self.blocks:
            x = layer(x, c)
        x = self.ln(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            n_layers: int,
            block_size: int = 100,
            causal: bool = True,
            qk_norm: bool = True,
            bias: bool = False,
            mlp_pdrop: float = 0,
            use_cross_attention: bool = True
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[Block(
                embed_dim,
                n_heads,
                attn_pdrop,
                resid_pdrop,
                mlp_pdrop,
                block_size,
                causal=causal,
                use_cross_attention=use_cross_attention,
                bias=bias,
                qk_norm=qk_norm,
            )
                for _ in range(n_layers)]
        )
        self.ln = RMSNorm(embed_dim, eps=1e-6)

    def forward(self, x, cond=None, custom_attn_mask=None):
        for layer in self.blocks:
            x = layer(x, cond, custom_attn_mask=custom_attn_mask)
        x = self.ln(x)
        return x


class TransformerFiLMDecoder(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            n_layers: int,
            block_size: int = 100,
            causal: bool = True,
            qk_norm: bool = True,
            bias: bool = False,
            mlp_pdrop: float = 0,
            use_cross_attention: bool = True
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ConditionedBlock(
                embed_dim,
                n_heads,
                attn_pdrop,
                resid_pdrop,
                mlp_pdrop,
                block_size,
                causal=causal,
                use_cross_attention=use_cross_attention,
                bias=bias,
                qk_norm=qk_norm,
            )
                for _ in range(n_layers)]
        )
        self.ln = RMSNorm(embed_dim, eps=1e-6)

    def forward(self, x, c, cond=None, custom_attn_mask=None):
        for layer in self.blocks:
            x = layer(x, c, cond, custom_attn_mask=custom_attn_mask)
        x = self.ln(x)
        return x

