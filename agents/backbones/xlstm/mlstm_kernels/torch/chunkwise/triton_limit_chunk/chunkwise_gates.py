#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck


import torch

# Note: we separate this into a extra function for torch.compile. 
# torch.compile will compile this into a single kernel with ca. 0.2 ms runtime (compared to 2.5 ms non-fused kernels)
# for a 1.3B sized model with ctx8192.
@torch.compile
def compute_gate_grads_vecDeltaI_vecDeltaF(
    matQ: torch.Tensor, matK: torch.Tensor, matDeltaQ: torch.Tensor, matDeltaK: torch.Tensor, vecF: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    #! postprocessing: compute deltaF and deltaI gradients
    ## ? postprocessing
    # vecF = rearrange(vecF, "b nh nc l -> b nh (nc l)")
    # compute the vecDeltaFbar values with dfbar = rev_cumsum((q*dq - k*dk).sum(-1))
    matQ = matQ.to(torch.float32)
    matK = matK.to(torch.float32)
    matDeltaQ = matDeltaQ.to(torch.float32)
    matDeltaK = matDeltaK.to(torch.float32)
    vecDeltaFbar_acc = ((matQ * matDeltaQ) - (matK * matDeltaK)).sum(-1)
    vecDeltaFbar = vecDeltaFbar_acc.flip(-1).to(torch.float32).cumsum(-1).flip(-1)
    vecDeltaF = vecDeltaFbar * torch.sigmoid(-vecF)
    ## ? end postprocessing
    # compute deltaI
    # both are equivalent:
    # vecDeltaI = (matV * matDeltaV).sum(-1)
    vecDeltaI = (matK * matDeltaK).sum(-1)
    return vecDeltaI, vecDeltaF
