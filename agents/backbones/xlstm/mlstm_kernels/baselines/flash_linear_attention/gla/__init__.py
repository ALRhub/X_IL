# This module is copied from https://github.com/sustcsonglin/flash-linear-attention/blob/fee90b2e72366a46c60e3ef16431133aa5aced8d/fla/ops/gla
# Adapted to make it work in this codebase

from .chunk import chunk_gla
from .chunk_fuse import fused_chunk_gla
from .recurrent_fuse import fused_recurrent_gla

__all__ = ["chunk_gla", "fused_chunk_gla", "fused_recurrent_gla"]
