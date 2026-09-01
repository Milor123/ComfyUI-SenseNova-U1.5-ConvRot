"""Runtime guards that keep QuantizedTensors intact through ComfyUI weight streaming.

ComfyUI's patcher feeds patched weights through ``cast_to_device`` /
``tensor.to(dtype)`` whenever a LoRA touches a layer. On packed quantized
tensors those calls either relabel ``orig_dtype`` while keeping the packed
bytes or route the packing into float math, which silently corrupts 4-bit
formats (int8 survives because its values are representable after a cast).
The guards strip dtype requests for QuantizedTensors so they reach kernels
pristine, matching the fix the community validated for other convrot models.

Set ``SENSENOVA_NO_QT_GUARDS=1`` to disable.
"""

import logging
import os

import torch

_guard_installed = False

# Only SenseNova's convrot layouts need the dtype strip. Other quantized
# models (e.g. MiniMax H3's text encoder) must keep their dtype casts or
# they fail with "mat1 and mat2 have different dtype".
_SENSENOVA_LAYOUTS = {"TensorCoreConvRotW4A4Layout", "AsymW4A8Int8Layout"}


def _is_sensenova_qt(qt) -> bool:
    """True only for SenseNova's own QuantizedTensors."""
    try:
        # 4-bit layouts are exclusive to SenseNova
        if getattr(qt, "_layout_cls", None) in _SENSENOVA_LAYOUTS:
            return True
        # int8 convrot is TensorWiseINT8Layout with convrot=True
        params = getattr(qt, "_params", None)
        return bool(params is not None and getattr(params, "convrot", False))
    except Exception:
        return False


def _strip_dtype_args(args):
    head, rest = args[:1], args[1:]
    return head + tuple(a for a in rest if not isinstance(a, torch.dtype))


def install_quant_guards():
    """Patch dtype-stripping wrappers around QuantizedTensor conversion paths."""
    global _guard_installed
    if _guard_installed:
        return True
    if os.environ.get("SENSENOVA_NO_QT_GUARDS"):
        return False

    try:
        from comfy import model_management  # type: ignore[import-not-found]
        from comfy_kitchen.tensor import (  # type: ignore[import-not-found]
            base as kitchen_base,  # type: ignore[import-not-found]
        )
        from comfy_kitchen.tensor.base import (  # type: ignore[import-not-found]
            QuantizedTensor,  # type: ignore[import-not-found]
        )
    except ImportError:
        return False

    orig_cast_to_device = model_management.cast_to_device

    def cast_to_device_qt_safe(tensor, device, dtype=None, copy=False):
        if isinstance(tensor, QuantizedTensor) and _is_sensenova_qt(tensor):
            dtype = None
        return orig_cast_to_device(tensor, device, dtype, copy)

    model_management.cast_to_device = cast_to_device_qt_safe

    orig_handle_to = kitchen_base._handle_to

    def handle_to_dtype_safe(qt, args, kwargs, force_copy=False):
        if isinstance(qt, QuantizedTensor) and _is_sensenova_qt(qt):
            args = _strip_dtype_args(args)
            kwargs = {k: v for k, v in kwargs.items() if k != "dtype"}
        return orig_handle_to(qt, args, kwargs, force_copy=force_copy)

    kitchen_base._handle_to = handle_to_dtype_safe

    orig_handle_empty_like = kitchen_base._handle_empty_like

    def handle_empty_like_dtype_safe(qt, args, kwargs):
        if isinstance(qt, QuantizedTensor) and _is_sensenova_qt(qt):
            kwargs = {k: v for k, v in kwargs.items() if k != "dtype"}
        return orig_handle_empty_like(qt, args, kwargs)

    kitchen_base._handle_empty_like = handle_empty_like_dtype_safe

    dispatch = getattr(kitchen_base, "_DISPATCH_TABLE", None)
    if dispatch is not None:
        for op_key, handler in list(dispatch.items()):
            if handler is orig_handle_to:
                dispatch[op_key] = handle_to_dtype_safe
            elif handler is orig_handle_empty_like:
                dispatch[op_key] = handle_empty_like_dtype_safe

    _guard_installed = True
    logging.info(
        "[sensenova-u15] QuantizedTensor dtype guards installed (scoped to SenseNova layouts)."
    )
    return True
