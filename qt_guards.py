"""Runtime guards that keep SenseNova QuantizedTensors intact through weight streaming.

ComfyUI's patcher feeds patched weights through ``cast_to_device`` /
``tensor.to(dtype)`` whenever a LoRA or hook touches a layer. On packed convrot
tensors those calls relabel ``orig_dtype`` while keeping the packed bytes, and
the subsequent ``dequantize()`` then produces the wrong dtype (e.g. bf16
instead of fp32), which breaks float math and silently corrupts 4-bit formats.

The dtype strip must ONLY apply while SenseNova's own patcher moves SenseNova
weights. Layout/params cannot identify the owner: ComfyUI core also creates
``int8_tensorwise`` convrot tensors for unrelated quantized models (MiniMax H3
text encoder, Flux2 Klein's Qwen3), and stripping their dtype makes
``to_dequant()`` produce bf16 where fp32 was requested -- exactly the
"expected mat1 and mat2 to have the same dtype" breakage reported in issues
#2 and #4.

Therefore the guards are invocation-scoped: they are inert process-wide
unless a ``SenseNovaModelPatcher`` method activates
``sensenova_quant_scope()`` while moving SenseNova weights
(``patch_weight_to_device`` / ``patch_hook_weight_to_device`` -- the only
ComfyUI paths that pass an explicit dtype to model weights).

Set ``SENSENOVA_NO_QT_GUARDS=1`` to disable.
"""

import contextlib
import logging
import os
import threading

import torch

_guard_installed = False
_scope_state = threading.local()

# Packed layouts whose dequantize() output dtype follows params.orig_dtype.
_PACKED_LAYOUTS = {"TensorCoreConvRotW4A4Layout", "AsymW4A8Int8Layout"}


def _scope_active():
    return getattr(_scope_state, "active", False)


@contextlib.contextmanager
def sensenova_quant_scope():
    """Activate the dtype-strip guards for the current thread only."""
    previous = _scope_active()
    _scope_state.active = True
    try:
        yield
    finally:
        _scope_state.active = previous


def _needs_guard(qt) -> bool:
    """True for packed convrot tensors whose dequant depends on orig_dtype."""
    try:
        if getattr(qt, "_layout_cls", None) in _PACKED_LAYOUTS:
            return True
        params = getattr(qt, "_params", None)
        return bool(params is not None and getattr(params, "convrot", False))
    except Exception:
        return False


def _strip_dtype_args(args):
    head, rest = args[:1], args[1:]
    return head + tuple(a for a in rest if not isinstance(a, torch.dtype))


def install_quant_guards():
    """Install invocation-scoped dtype-strip wrappers.

    The wrappers are inert (exact passthrough) unless
    ``sensenova_quant_scope()`` is active on the calling thread, so unrelated
    quantized models keep ComfyUI's native behavior even after SenseNova has
    been loaded in the same session.
    """
    global _guard_installed
    if _guard_installed:
        return True
    if os.environ.get("SENSENOVA_NO_QT_GUARDS"):
        return False

    try:
        from comfy import model_management  # type: ignore[import-not-found]
        from comfy_kitchen.tensor import (  # type: ignore[import-not-found]
            base as kitchen_base,
        )
        from comfy_kitchen.tensor.base import (  # type: ignore[import-not-found]
            QuantizedTensor,
        )
    except ImportError:
        return False

    orig_cast_to_device = model_management.cast_to_device

    def cast_to_device_qt_safe(tensor, device, dtype=None, copy=False):
        if (
            dtype is not None
            and _scope_active()
            and isinstance(tensor, QuantizedTensor)
            and _needs_guard(tensor)
        ):
            dtype = None
        return orig_cast_to_device(tensor, device, dtype, copy)

    model_management.cast_to_device = cast_to_device_qt_safe

    orig_handle_to = kitchen_base._handle_to

    def handle_to_dtype_safe(qt, args, kwargs, force_copy=False):
        if _scope_active() and isinstance(qt, QuantizedTensor) and _needs_guard(qt):
            args = _strip_dtype_args(args)
            kwargs = {k: v for k, v in kwargs.items() if k != "dtype"}
        return orig_handle_to(qt, args, kwargs, force_copy=force_copy)

    kitchen_base._handle_to = handle_to_dtype_safe

    orig_handle_empty_like = kitchen_base._handle_empty_like

    def handle_empty_like_dtype_safe(qt, args, kwargs):
        if _scope_active() and isinstance(qt, QuantizedTensor) and _needs_guard(qt):
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
        "[sensenova-u15] QuantizedTensor dtype guards installed (invocation-scoped to SenseNova patcher)."
    )
    return True
