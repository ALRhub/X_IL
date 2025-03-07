#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from collections.abc import Callable


def _create_module_sequence_backend_registry() -> dict[str, dict[str, Callable]]:
    from .chunkwise import registry as mlstm_chunkwise_registry
    from .parallel import registry as mlstm_parallel_registry

    module_backend_registry = {
        "chunkwise": mlstm_chunkwise_registry,
        "parallel": mlstm_parallel_registry,
    }
    return module_backend_registry


def get_available_mlstm_kernels() -> list[str]:
    """
    Get a list of available mlstm sequence kernels.
    These kernels process a sequence in the parallel or chunkwise parallel mode of the mLSTM. 
    They do not support arbitrary sequence lengths. 
    They are used for training and prefill processing during inference of the mLSTM during.
    """
    module_backend_registry = _create_module_sequence_backend_registry()

    backend_names = [
        f"{module_key}--{kernel_key}"
        for module_key in module_backend_registry.keys()
        for kernel_key in module_backend_registry[module_key].keys()
    ]
    return backend_names


def get_mlstm_kernel(name: str) -> Callable:
    """
    Get a mlstm sequence kernel function by name.

    Naming convention:
    name = "<module_name>--<backend_name>"

    module_name: The name of the module containing the kernel function.
        Example: "chunkwise", "parallel", "recurrent"

    backend_name: The name of the kernel function as defined in the registry in the __init__.py file of the module.
    """

    module_backend_registry = _create_module_sequence_backend_registry()

    module_name, backend_name = name.split("--")

    if module_name not in module_backend_registry:
        raise ValueError(
            f"Unknown module name: {module_name}. Available module names: {list(module_backend_registry.keys())}"
        )

    if backend_name not in module_backend_registry[module_name]:
        raise ValueError(
            f"Unknown mlstm kernel backend name: {backend_name}. Available backend names: {list(module_backend_registry[module_name].keys())}"
        )

    return module_backend_registry[module_name][backend_name]


def get_available_mlstm_step_kernels() -> list[str]:
    """Returns the available mlstm step kernels. 
    These kernels can be used to compute a single time step of the mLSTM, i.e. for generation.
    """
    from .recurrent import registry_step as mlstm_recurrent_step_registry

    backend_names = list(mlstm_recurrent_step_registry.keys())
    return backend_names


def get_mlstm_step_kernel(name: str) -> Callable:
    """
    Get a mlstm step kernel function by name.

    Naming convention:
    name = "<backend_name>"

    backend_name: The name of the kernel function as defined in the registry in the __init__.py file of the module.
    """
    from .recurrent import registry_step as mlstm_recurrent_step_registry

    if name not in mlstm_recurrent_step_registry:
        raise ValueError(
            f"Unknown step kernel backend name: {name}. Available backend names: {list(mlstm_recurrent_step_registry.keys())}"
        )

    return mlstm_recurrent_step_registry[name]


def get_available_mlstm_sequence_kernels() -> list[str]:
    """Returns the available mlstm sequence kernels.
    These kernels process a sequence in the recurrent mode of the mLSTM and hence support any sequence length.
    """
    from .recurrent import registry_sequence as mlstm_recurrent_sequence_registry

    backend_names = list(mlstm_recurrent_sequence_registry.keys())
    return backend_names


def get_mlstm_sequence_kernel(name: str) -> Callable:
    """
    Get a mlstm sequence kernel function by name.

    Naming convention:
    name = "<backend_name>"

    backend_name: The name of the kernel function as defined in the registry in the __init__.py file of the module.
    """
    from .recurrent import registry_sequence as mlstm_recurrent_sequence_registry

    if name not in mlstm_recurrent_sequence_registry:
        raise ValueError(
            f"Unknown backend name: {name}. Available backend names: {list(mlstm_recurrent_sequence_registry.keys())}"
        )

    return mlstm_recurrent_sequence_registry[name]
