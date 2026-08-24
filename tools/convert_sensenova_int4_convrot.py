"""Convert a SenseNova-U1.5 bf16/fp32 checkpoint to native convrot_w4a4 int4.

Packing is delegated to comfy-kitchen's own TensorCoreConvRotW4A4Layout (the
exact code ComfyUI runs at inference), so rotation, symmetric per-row int4
quantization and nibble packing are guaranteed bit-compatible with the runtime.

Run inside the ComfyUI venv:
  python convert_sensenova_int4_convrot.py -i <in.safetensors> -o <out.safetensors> [--device cpu]
"""
import argparse
import json
import re
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file

from comfy_kitchen.backends.eager.convrot_w4a4 import _build_hadamard
from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout

# Mirrors sensenova_u15/loader.py::_is_quant_candidate policy.
_EXCLUDED = ("norm", "embed_tokens", "lm_head")

# Mixed-mode policy (mirrors QuantizationToolkit's architecture-aware tiers):
# write-back projections (o_proj, down_proj) and the fm conditioning MLPs stay
# at convrot int8; the bulk goes to convrot int4.
_INT8_STEMS = ("o_proj", "down_proj")
_INT8_EMBEDDERS = ("fm_modules.timestep_embedder", "fm_modules.noise_scale_embedder")

_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I64": torch.int64,
    "I8": torch.int8,
    "U8": torch.uint8,
}


def _iter_tensors(path):
    """Yield (name, cpu_tensor) using sequential reads - NO whole-file mmap.

    Windows error 1455 (page file too small) makes safetensors' 50 GB mmap
    unreliable under memory pressure; explicit reads only ever hold one tensor.
    """
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        header.pop("__metadata__", None)
        data_start = 8 + header_len
        ordered = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
        for name, info in ordered:
            begin, end = info["data_offsets"]
            f.seek(data_start + begin)
            buf = f.read(end - begin)
            dtype = _DTYPES[info["dtype"]]
            if dtype == torch.bfloat16:
                t = torch.frombuffer(bytearray(buf), dtype=torch.int16).view(torch.bfloat16)
            else:
                t = torch.frombuffer(bytearray(buf), dtype=dtype)
            yield name, t.reshape(info["shape"])


def _is_quant_candidate(name, tensor):
    return (
        tensor.dim() == 2
        and name.endswith(".weight")
        and not any(token in name for token in _EXCLUDED)
    )


def _quantize_int8_convrot(w_fp32, groupsize, device):
    """Row-wise symmetric int8 with the Hadamard fold folded in (R1 layout)."""
    h = _build_hadamard(groupsize, device, torch.float32)
    out_f, in_f = w_fp32.shape
    ng = in_f // groupsize
    w_rot = torch.einsum("ong,gh->onh", w_fp32.view(out_f, ng, groupsize),
                         h.t()).reshape(out_f, in_f)
    scale = w_rot.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
    q = (w_rot / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--device", default="cpu", help="compute device for rotation math")
    parser.add_argument("--convrot-groupsize", type=int, default=256)
    parser.add_argument("--linear-dtype", choices=["int4", "int8"], default="int4",
                        help="MMA accumulation precision for w4a4 layers (activations are "
                             "always int4 in this format)")
    parser.add_argument("--mode", choices=["mixed", "all-w4a4"], default="mixed",
                        help="mixed: o_proj/down_proj + fm embedders at convrot int8, "
                             "rest at w4a4 (official quality recipe). all-w4a4: everything int4.")
    args = parser.parse_args()

    device = torch.device(args.device)
    dst = Path(args.output)

    out = {}
    counts = {"w4a4": 0, "int8": 0, "copy": 0}
    n_total = 0
    for key, tensor in _iter_tensors(args.input):
        n_total += 1
        if not _is_quant_candidate(key, tensor):
            out[key] = tensor
            counts["copy"] += 1
            continue
        stem = key[: -len(".weight")]
        w = tensor.to(device=device, dtype=torch.float32)

        use_int8 = args.mode == "mixed" and (
            any(tok in key for tok in _INT8_STEMS)
            or any(key.startswith(pfx) for pfx in _INT8_EMBEDDERS)
        )
        if use_int8:
            qdata, scale = _quantize_int8_convrot(w, args.convrot_groupsize, device)
            conf = {
                "format": "int8_tensorwise",
                "orig_dtype": "torch.bfloat16",
                "convrot": True,
                "convrot_groupsize": args.convrot_groupsize,
                "per_row": True,
            }
            out[stem + ".weight"] = qdata.to("cpu")
            out[stem + ".weight_scale"] = scale.to("cpu").to(torch.float32)
            fmt_key = "int8"
        else:
            qdata, params = TensorCoreConvRotW4A4Layout.quantize(
                w, convrot_groupsize=args.convrot_groupsize, stochastic_rounding=0
            )
            conf = {"format": "convrot_w4a4", "convrot_groupsize": args.convrot_groupsize}
            if args.linear_dtype != "int4":
                conf["linear_dtype"] = args.linear_dtype
            out[stem + ".weight"] = qdata.to("cpu")
            out[stem + ".weight_scale"] = params.scale.to("cpu").to(torch.float32).reshape(-1)
            fmt_key = "w4a4"
        out[stem + ".comfy_quant"] = torch.tensor(
            list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8
        )
        counts[fmt_key] += 1
        if n_total % 100 == 0:
            print(f"[{n_total}] {counts}", flush=True)

    save_file(out, str(dst))
    print(f"done: {counts} -> {dst}")


if __name__ == "__main__":
    main()

