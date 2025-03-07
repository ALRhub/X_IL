#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from .native_sequence import (
    mlstm_recurrent_sequence__native_fw,
    mlstm_recurrent_sequence__triton_alternate_step_fw,
    mlstm_recurrent_sequence__triton_step_fused_fw,
)
from .native_step import mlstm_recurrent_step__native
from .triton_step import mlstm_recurrent_step__triton
from .triton_step_alternate import mlstm_recurrent_step__triton_alternate

registry_step = {
    "native": mlstm_recurrent_step__native,
    # "triton_alternate": mlstm_recurrent_step__triton_alternate,
    "triton": mlstm_recurrent_step__triton,
}


registry_sequence = {
    "native_sequence__native": mlstm_recurrent_sequence__native_fw,
    # "native_sequence__triton_alternate": mlstm_recurrent_sequence__triton_alternate_step_fw,
    "native_sequence__triton": mlstm_recurrent_sequence__triton_step_fused_fw,
}
