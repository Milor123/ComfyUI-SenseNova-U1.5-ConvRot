"""Convert a SenseNova-U1.5 bf16/fp32 checkpoint to native convrot_w4a4 int4.

Packing is delegated to comfy-kitchen's own TensorCoreConvRotW4A4Layout (the
exact code ComfyUI runs at inference), so rotation, symmetric per-row int4
quantization and nibble packing are guaranteed bit-compatible with the runtime.

Run inside the ComfyUI venv:
  python convert_sensenova_int4_convrot.py -i <in.safetensors> -o <out.safetensors> [--device cpu]
"""
import argparse
import json
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file

from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout

# Mirrors sensenova_u15/loader.py::_is_quant_candidate policy.
_EXCLUDED = ("norm", "embed_tokens", "lm_head")

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--device", default="cpu", help="compute device for rotation math")
    parser.add_argument("--convrot-groupsize", type=int, default=256)
    parser.add_argument("--linear-dtype", choices=["int4", "int8"], default="int4",
                        help="activation precision for the kernel: int4 (W4A4, fastest/coarsest) "
                             "or int8 (W4A8, much better quality, same 4-bit weights)")
    args = parser.parse_args()

    device = torch.device(args.device)
    dst = Path(args.output)

    out = {}
    n_quant, n_copy, n_total = 0, 0, 0
    for key, tensor in _iter_tensors(args.input):
        n_total += 1
        if not _is_quant_candidate(key, tensor):
            out[key] = tensor
            n_copy += 1
            continue
        stem = key[: -len(".weight")]
        w = tensor.to(device=device, dtype=torch.float32)
        qdata, params = TensorCoreConvRotW4A4Layout.quantize(
            w, convrot_groupsize=args.convrot_groupsize, stochastic_rounding=0
        )
        conf = {"format": "convrot_w4a4", "convrot_groupsize": args.convrot_groupsize}
        if args.linear_dtype != "int4":
            conf["linear_dtype"] = args.linear_dtype
        out[stem + ".weight"] = qdata.to("cpu")
        out[stem + ".weight_scale"] = params.scale.to("cpu").to(torch.float32).reshape(-1)
        out[stem + ".comfy_quant"] = torch.tensor(
            list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8
        )
        n_quant += 1
        if n_total % 100 == 0:
            print(f"[{n_total}] quantized={n_quant} copied={n_copy}", flush=True)

    save_file(out, str(dst))
    print(f"done: {n_quant} weights packed to convrot_w4a4, {n_copy} passthrough -> {dst}")


if __name__ == "__main__":
    main()

