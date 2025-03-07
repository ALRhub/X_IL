#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from .native_sequence import (
    mlstm_recurrent_sequence__native_fw,
    mlstm_recurrent_sequence__triton_step_fused_fw,
)
from .native_step import mlstm_recurrent_step__native
from .triton_step import mlstm_recurrent_step__triton

registry_step = {
    "native": mlstm_recurrent_step__native,
    "triton": mlstm_recurrent_step__triton,
}
