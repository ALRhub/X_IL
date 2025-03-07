#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from .flash_attention_triton import attention_causal as attention_causal_triton_flash
from .torch_sdp_attention import (
    attention_causal_pt_cudnn as attention_causal_torch_cudnn,
    attention_causal_pt_efficient as attention_causal_torch_efficient,
    attention_causal_pt_fa2 as attention_causal_torch_flash,
    attention_causal_pt_math as attention_causal_torch_math,
)
from .triton_tutorial import attention_causal as attention_causal_triton_tutorial

registry = {
    "torch_flash": attention_causal_torch_flash,
    "torch_cudnn": attention_causal_torch_cudnn,
    "torch_math": attention_causal_torch_math,
    "torch_efficient": attention_causal_torch_efficient,
    "triton_flash": attention_causal_triton_flash,
    "triton_tutorial": attention_causal_triton_tutorial,
}
