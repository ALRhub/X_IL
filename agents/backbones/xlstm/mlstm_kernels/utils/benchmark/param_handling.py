#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from dataclasses import dataclass
from itertools import product
from typing import Any, Literal


def strip_to_first_chars_of_split(input_str: str, split_str: str = "_") -> str:
    """Strips a string to the sequence first characters after the `sep_str`.

    Args:
        s (str): The string to strip.
        sep_str (str, optional): The separator string. Defaults to "_".
    """
    in_str = input_str.lstrip(split_str)  # remove all leading underscores
    in_str = "".join([s[0] for s in in_str.split(split_str)])
    return in_str


@dataclass
class KernelSpec:
    """Specification for a kernel to benchmark."""

    kernel_name: str
    dtype: Literal["float16", "float32", "float64", "bfloat16"]
    fwbw: bool = False
    use_torch_compile: bool = None

    additional_params: dict[str, Any] = None

    def to_string(self, short_param_name: bool = True) -> str:
        str_name = f"{self.kernel_name}"
        if self.use_torch_compile is not None:
            str_name += f"__{'tc' if self.use_torch_compile else ''}"
        str_name += f"__{self.dtype}__{'fwbw' if self.fwbw else 'fw'}"
        if self.additional_params is not None:
            str_name += "_"
            for k, v in self.additional_params.items():
                if short_param_name:
                    k = strip_to_first_chars_of_split(k)
                str_name += f"_{k}-{v}"
        return str_name


@dataclass
class ModelSpec:
    """Specification for a model to benchmark."""

    model_name: str
    amp_enabled: bool = True
    amp_dtype: str = "bfloat16"
    weight_dtype: str = "float32"
    use_torch_compile_model: bool = False

    additional_params: dict[str, Any] = None

    def to_string(self, short_param_name: bool = True) -> str:
        str_name = f"{self.model_name}"
        if self.use_torch_compile_model:
            str_name += "__tcm"
        if self.amp_enabled:
            str_name += f"__ampdt-{self.amp_dtype}"
        str_name += f"__wdt-{self.weight_dtype}"
        if self.additional_params is not None:
            str_name += "_"
            for k, v in self.additional_params.items():
                if short_param_name:
                    k = strip_to_first_chars_of_split(k)
                str_name += f"_{k}-{v}"
        return str_name


@dataclass
class BenchmarkConfig:
    vary_type: Literal["grid", "sequence"]
    vary_params: dict[str, list[Any]] | None

    fixed_params: dict[str, Any]

    # TODO rename kernel_specs to a generic name
    kernel_specs: list[KernelSpec | ModelSpec]

    benchmark_name: str

    x_axis_param: str = None

    def _get_vary_param_dicts(self) -> list[dict[str, Any]]:
        vary_dicts = []
        if self.vary_params is not None:
            if self.vary_type == "grid":
                for val_tuple in product(*self.vary_params.values()):
                    vary_dict = dict(zip(self.vary_params.keys(), val_tuple))
                    vary_dicts.append(vary_dict)
            elif self.vary_type == "sequence":
                num_vals = len(list(self.vary_params.values())[0])
                for k, v in self.vary_params.items():
                    if len(v) != num_vals:
                        raise ValueError(
                            f"All vary parameters must have the same length. Not matching for {k}"
                        )

                for val_tuple in zip(*self.vary_params.values()):
                    vary_dict = dict(zip(self.vary_params.keys(), val_tuple))
                    vary_dicts.append(vary_dict)
            else:
                raise ValueError(f"Unknown vary type: {self.vary_type}")
        return vary_dicts

    def get_param_dicts(self) -> list[dict[str, Any]]:
        """
        Get all the parameter dictionaries for the benchmark.
        """
        param_dicts = []
        vary_dicts = self._get_vary_param_dicts()
        if len(vary_dicts) == 0:
            param_dicts.append(self.fixed_params)
            return param_dicts
        else:
            for vary_dict in vary_dicts:
                param_dict = self.fixed_params.copy()
                for k, v in vary_dict.items():
                    param_dict[k] = v
                param_dicts.append(param_dict)
            return param_dicts

    def get_plot_title(self) -> str:
        title = self.benchmark_name
        for k, v in self.fixed_params.items():
            title += f"__{k}-{v}"
        return title
