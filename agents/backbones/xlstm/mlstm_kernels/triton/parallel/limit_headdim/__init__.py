#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from .fw_kernel import mlstm_parallel_fw_kernel
from .bw_kernel import mlstm_parallel_bw_dKdV_kernel, mlstm_parallel_bw_dQ_kernel
