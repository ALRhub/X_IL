#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import jax
import jax.numpy as jnp
import numpy as np
import triton.language as tl


def dtype2str(dtype: jnp.dtype) -> str:
    if dtype == jnp.float32:
        return "fp32"
    elif dtype == jnp.float16:
        return "fp16"
    elif dtype == jnp.float64:
        return "fp64"
    elif dtype == jnp.bfloat16:
        return "bf16"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def jax2triton_dtype(dtype):
    """
    Converts a JAX dtype to a Triton dtype.

    Args:
        dtype: JAX dtype.

    Returns:
        Triton dtype.
    """
    # We need to grab the dtype from the dtype object in jax
    # >> dt = jnp.float32
    # >> str(dt), str(dt.dtype)
    # Output:
    # ("<class 'jax.numpy.float32'>", 'float32')
    if hasattr(dtype, "dtype"):
        dtype = dtype.dtype
    return getattr(tl, str(dtype))


def to_numpy(tensor: jnp.ndarray) -> np.ndarray:
    return jax.device_get(tensor)
