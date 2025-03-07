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
    Get a list of available mlstm sequence kernel names.
    """
    module_backend_registry = _create_module_sequence_backend_registry()

    backend_names = [
        f"{module_key}--{kernel_key}"
        for module_key in module_backend_registry.keys()
        for kernel_key in module_backend_registry[module_key].keys()
    ]
    return backend_names


def get_available_mlstm_step_kernels() -> list[str]:
    from .recurrent import registry_step as mlstm_recurrent_step_registry

    backend_names = list(mlstm_recurrent_step_registry.keys())
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
            f"Unknown backend name: {backend_name}. Available backend names: {list(module_backend_registry[module_name].keys())}"
        )

    return module_backend_registry[module_name][backend_name]


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
            f"Unknown backend name: {name}. Available backend names: {list(mlstm_recurrent_step_registry.keys())}"
        )

    return mlstm_recurrent_step_registry[name]
