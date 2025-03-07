#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import os


def simulate_CPU_devices(device_count: int = 8):
    """
    Simulate a CPU with a given number of devices.

    Args:
        device_count: The number of devices to simulate.
    """
    # Set XLA flags to simulate a CPU with a given number of devices
    flags = os.environ.get("XLA_FLAGS", "")
    flags += f" --xla_force_host_platform_device_count={device_count}"
    os.environ["XLA_FLAGS"] = flags
    # Disable CUDA to force XLA to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""