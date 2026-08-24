"""ConvRot-aware Linear forwards for quantized SenseNova checkpoints.

ComfyUI core stores convrot metadata on quantized weights but its generic
dispatch computes plain (dequantized) linears without rotating activations,
so offline-folded checkpoints evaluate against the wrong basis. This factory
subclasses ComfyUI's mixed-precision ops and routes convrot weights through
comfy-kitchen kernels (which rotate internally); if anything upstream already
materialized the weight as float, it rotates the activation explicitly - the
math is identical.
"""
import torch

import comfy.ops
import comfy.quant_ops
from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout

_REPORTED_NAN = set()


def _warn_nan(name, where, out):
    import logging

    if (torch.isnan(out).any() or torch.isinf(out).any()) and name not in _REPORTED_NAN:
        _REPORTED_NAN.add(name)
        logging.warning(
            f"[sensenova-quant] NaN/Inf detected in {where} output of '{name}' "
            f"(first occurrence; further occurrences suppressed)"
        )


def _rotate_input(x, group_size):
    from comfy_kitchen.backends.eager.convrot_w4a4 import _build_hadamard

    h = _build_hadamard(group_size, x.device, x.dtype)
    features = x.shape[-1]
    rotated = torch.matmul(x.reshape(-1, features // group_size, group_size), h)
    return rotated.reshape(x.shape)


def make_sensenova_quant_ops():
    base = comfy.ops.mixed_precision_ops({"mixed_ops": True})

    class _Linear(base.Linear):
        def _load_from_state_dict(self, *args, **kwargs):
            prefix = args[1] if len(args) > 1 else kwargs.get("prefix", "")
            super()._load_from_state_dict(*args, **kwargs)
            params = getattr(self.weight, "_params", None)
            fmt = getattr(self, "quant_format", None)
            rotated = params is not None and (
                getattr(params, "convrot", False) or fmt == "convrot_w4a4"
            )
            self._sensenova_convrot_gs = int(getattr(params, "convrot_groupsize", 256)) if rotated else None
            self._sensenova_name = prefix.rstrip(".")

        def _forward(self, input, weight, bias):
            gs = getattr(self, "_sensenova_convrot_gs", None)
            if gs is None:
                return torch.nn.functional.linear(input, weight, bias)

            fmt = getattr(self, "quant_format", None)
            if isinstance(weight, QuantizedTensor):
                if fmt == "asym_w4a8_int8":
                    # TRUE W4A8: 4-bit weights, int8 runtime activations.
                    # Eager forced for the same consistency reason as w4a4.
                    from comfy_kitchen.backends.eager.w4a8_int8 import (
                        w4a8_int8_linear as eager_w4a8,
                    )
                    from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout

                    qdata, s_rel, s_ch, corr, cb = AsymW4A8Int8Layout.get_plain_tensors(weight)
                    out = eager_w4a8(
                        input, qdata, s_rel, s_ch,
                        codebook=cb, correction=corr, bias=bias,
                        group_size=int(getattr(weight._params, "group_size", 16)),
                        convrot_groupsize=gs,
                        out_dtype=input.dtype,
                    )
                    _warn_nan(getattr(self, "_sensenova_name", "?"), "w4a8 eager", out)
                    return out
                if fmt == "convrot_w4a4":
                    # Kitchen's W4A4 linear rotates activations internally.
                    # The EAGER implementation is forced deliberately: the CUDA
                    # kernel diverges from eager by ~15% per call on this
                    # build, which compounds destructively across 42 layers.
                    from comfy_kitchen.backends.eager.convrot_w4a4 import (
                        convrot_w4a4_linear as eager_w4a4_linear,
                    )
                    from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout

                    qdata, wscales = TensorCoreConvRotW4A4Layout.get_plain_tensors(weight)
                    out = eager_w4a4_linear(
                        input, qdata, wscales, bias,
                        convrot_groupsize=gs,
                        quant_group_size=int(getattr(weight._params, "quant_group_size", 64)),
                        linear_dtype=getattr(weight._params, "linear_dtype", "int4"),
                    )
                    _warn_nan(getattr(self, "_sensenova_name", "?"), "w4a4 eager", out)
                    return out
                # NOTE: this comfy-kitchen build accepts convrot kwargs on
                # ck.int8_linear but ignores them, so int8 goes through the
                # exact float path below instead.

            # Exact path: rotate activations into the folded basis, then a
            # plain linear against the rotated weight. Dequantization is done
            # MANUALLY from raw qdata/scales: QuantizedTensor.dequantize()
            # dispatch is unreliable on CPU in current builds (wrong results
            # or native crashes) and weight streaming hits CPU constantly.
            #
            # Float materializations differ per format: comfy's int8 dequant
            # keeps the ROTATED basis (input must rotate), while kitchen's
            # w4a4 dequant already restores the ORIGINAL basis (plain linear).
            if isinstance(weight, QuantizedTensor):
                qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
                weight_float = qdata.to(input.dtype) * scale.to(input.dtype).reshape(-1, 1)
                return torch.nn.functional.linear(_rotate_input(input, gs), weight_float, bias)
            if fmt == "convrot_w4a4" or fmt == "asym_w4a8_int8":
                # Both kitchen formats dequantize back to the ORIGINAL basis.
                return torch.nn.functional.linear(input, weight, bias)
            return torch.nn.functional.linear(_rotate_input(input, gs), weight, bias)

    class Ops(base):
        Linear = _Linear

    return Ops()
