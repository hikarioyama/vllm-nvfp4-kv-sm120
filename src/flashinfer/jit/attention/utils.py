"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import List

# NVFP4 KV-cache scale-factor tensors that need EXPLICIT (page/h/n) gmem strides
# plumbed into the FA2 kernel (attn/prefill.cuh page_produce_kv_sf / produce_kv_sf).
# Previously the kernel derived SF strides as data_stride/SF_CONTAINERS(=8), which
# forced a 2.25x over-allocated SF scratch (+25% of the KV pool). With explicit
# strides, K is read directly from the interleaved cache view (zero scratch) and V
# from a separate CONTIGUOUS de-swizzled cache (+5.5% only). For each such tensor we
# auto-emit `<name>_stride_page/h/n` Params fields + a layout-aware setter that reads
# the strides from the tensor itself (no run()/wrapper/binding changes needed).
_SF_STRIDE_TENSORS = ("maybe_k_cache_sf", "maybe_v_cache_sf")


def generate_additional_params(
    additional_tensor_names: List[str],
    additional_tensor_dtypes: List[str],
    additional_scalar_names: List[str],
    additional_scalar_dtypes: List[str],
    is_sm90_template: bool = False,
):
    # SF tensors present in this variant (empty for non-NVFP4 / hopper variants).
    sf_stride_vars = [v for v in additional_tensor_names if v in _SF_STRIDE_TENSORS]

    additional_params_decl = "".join(
        [
            f"{dtype}* {var};\n"
            for dtype, var in zip(
                additional_tensor_dtypes,
                additional_tensor_names,
                strict=True,
            )
        ]
        + [
            f"{dtype} {var};\n"
            for dtype, var in zip(
                additional_scalar_dtypes, additional_scalar_names, strict=True
            )
        ]
        + [
            f"uint32_t {var}_stride_page;\n"
            f"uint32_t {var}_stride_h;\n"
            f"uint32_t {var}_stride_n;\n"
            for var in sf_stride_vars
        ]
    )
    additional_func_params = "".join(
        [
            (
                f", Optional<ffi::Tensor> {var}"
                if var.startswith("maybe")
                else f", ffi::Tensor {var}"
            )
            for var in additional_tensor_names
        ]
        + [
            f", {dtype} {var}"
            for dtype, var in zip(
                additional_scalar_dtypes, additional_scalar_names, strict=True
            )
        ]
    )

    # Layout-aware SF-stride setter lines. `layout` (int64_t) is in scope in every
    # prefill Run function; page=stride(0), and (n, h) map per QKV layout exactly as
    # the data tensor does (the SF view shares the data dim order). The SF tensor may
    # be the interleaved cache view (K, real strides) or a contiguous cache (V).
    def _sf_setter(prefix: str) -> List[str]:
        lines: List[str] = []
        n_dim = "static_cast<QKVLayout>(layout) == QKVLayout::kNHD ? 1 : 2"
        h_dim = "static_cast<QKVLayout>(layout) == QKVLayout::kNHD ? 2 : 1"
        for var in sf_stride_vars:
            lines += [
                f"{prefix}{var}_stride_page = {var} ? (uint32_t)({var}.value().stride(0)) : 0;",
                f"{prefix}{var}_stride_n = {var} ? (uint32_t)({var}.value().stride({n_dim})) : 0;",
                f"{prefix}{var}_stride_h = {var} ? (uint32_t)({var}.value().stride({h_dim})) : 0;",
            ]
        return lines

    if is_sm90_template:
        additional_params_setter = " \\\n".join(
            [
                (
                    f"params.additional_params.{var} = {var} ? static_cast<{dtype}*>({var}.value().data_ptr()): nullptr;"
                    if var.startswith("maybe")
                    else f"params.additional_params.{var} = static_cast<{dtype}*>({var}.data_ptr());"
                )
                for dtype, var in zip(
                    additional_tensor_dtypes, additional_tensor_names, strict=True
                )
            ]
            + [
                f"params.additional_params.{var} = {var};"
                for var in additional_scalar_names
            ]
            + _sf_setter("params.additional_params.")
        )
    else:
        additional_params_setter = " \\\n".join(
            [
                (
                    f"params.{var} = {var} ? static_cast<{dtype}*>({var}.value().data_ptr()): nullptr;"
                    if var.startswith("maybe")
                    else f"params.{var} = static_cast<{dtype}*>({var}.data_ptr());"
                )
                for dtype, var in zip(
                    additional_tensor_dtypes, additional_tensor_names, strict=True
                )
            ]
            + [f"params.{var} = {var};" for var in additional_scalar_names]
            + _sf_setter("params.")
        )
    return (additional_params_decl, additional_func_params, additional_params_setter)
