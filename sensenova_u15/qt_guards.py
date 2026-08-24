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
        from comfy import model_management
        from comfy_kitchen.tensor import base as kitchen_base
        from comfy_kitchen.tensor.base import QuantizedTensor
    except ImportError:
        return False

    orig_cast_to_device = model_management.cast_to_device

    def cast_to_device_qt_safe(tensor, device, dtype=None, copy=False):
        if isinstance(tensor, QuantizedTensor):
            dtype = None
        return orig_cast_to_device(tensor, device, dtype, copy)

    model_management.cast_to_device = cast_to_device_qt_safe

    orig_handle_to = kitchen_base._handle_to

    def handle_to_dtype_safe(qt, args, kwargs, force_copy=False):
        if isinstance(qt, QuantizedTensor):
            args = _strip_dtype_args(args)
            kwargs = {k: v for k, v in kwargs.items() if k != "dtype"}
        return orig_handle_to(qt, args, kwargs, force_copy=force_copy)

    kitchen_base._handle_to = handle_to_dtype_safe

    orig_handle_empty_like = kitchen_base._handle_empty_like

    def handle_empty_like_dtype_safe(qt, args, kwargs):
        if isinstance(qt, QuantizedTensor):
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
    logging.info("[sensenova-u15] QuantizedTensor dtype guards installed.")
    return True
