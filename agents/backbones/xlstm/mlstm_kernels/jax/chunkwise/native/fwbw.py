#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import jax

from .fw import mlstm_chunkwise_fw


def mlstm_chunkwise__native_autograd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    i: jax.Array,
    f: jax.Array,
    c_initial: jax.Array = None,
    n_initial: jax.Array = None,
    m_initial: jax.Array = None,
    return_last_states: bool = False,
    eps: float = 1e-6,
    chunk_size: int = 64,
    **kwargs,
) -> jax.Array | tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    matH_out, _, _, last_states, _ = mlstm_chunkwise_fw(
        matQ=q,
        matK=k,
        matV=v,
        vecI=i,
        vecF=f,
        matC_initial=c_initial,
        vecN_initial=n_initial,
        scaM_initial=m_initial,
        return_last_states=return_last_states,
        return_all_states=False,
        eps=eps,
        chunk_size=chunk_size,
    )
    if return_last_states:
        return matH_out, last_states
    else:
        return matH_out

# TODO bring this into jax
# def mlstm_chunkwise_custbw(
#     q: jax.Array,
#     k: jax.Array,
#     v: jax.Array,
#     i: jax.Array,
#     f: jax.Array,
#     c_initial: jax.Array = None,
#     n_initial: jax.Array = None,
#     m_initial: jax.Array = None,
#     return_last_states: bool = False,
#     eps: float = 1e-6,
#     chunk_size: int = 64,
#     autocast_kernel_dtype: torch.dtype = torch.float32,
# ) -> jax.Array | tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
#     _mlstm_chunkwise_fwbw = _get_chunkwise_fwbw_kernel(autocast_kernel_dtype)
#     matH_out, matC_last, vecN_last, scaM_last = _mlstm_chunkwise_fwbw.apply(
#         q,
#         k,
#         v,
#         i,
#         f,
#         c_initial,
#         n_initial,
#         m_initial,
#         None,
#         return_last_states,
#         True,
#         chunk_size,
#         eps,
#     )
#     if return_last_states:
#         return matH_out, (matC_last, vecN_last, scaM_last)
#     else:
#         return matH_out
