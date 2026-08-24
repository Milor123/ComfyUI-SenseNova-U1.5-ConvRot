"""Smoke test for the convrot_w4a4 offline converter on a tiny synthetic checkpoint."""
import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
from smoke_test import build_sd, linear_keys  # reuse the synthetic builder

CONVERTER = Path(
    r"C:\Users\User\Documents\Clonitaditos\experimento\Comfyui-SenseNova-U1.5-Wrapper-T8"
    r"\tools\convert_sensenova_int4_convrot.py"
)
IN_F = WORK / "in.safetensors"  # rebuilt identical to the int8 run
OUT_F = WORK / "out_int4_convrot.safetensors"

from comfy_kitchen.tensor.convrot_w4a4 import dequantize_convrot_w4a4_weight


def main():
    sd_in = build_sd()
    save_in = WORK / "in.safetensors"
    from safetensors.torch import save_file

    save_file(sd_in, str(save_in))
    expect_quant = linear_keys(sd_in)
    print(f"[setup] tensors={len(sd_in)} expected-quantized={len(expect_quant)}")

    r = subprocess.run(
        [sys.executable, str(CONVERTER), "-i", str(IN_F), "-o", str(OUT_F), "--device", "cpu"],
        capture_output=True, text=True,
    )
    print((r.stdout or "")[-800:])
    if r.returncode != 0:
        print("STDERR:\n" + (r.stderr or "")[-3000:])
        sys.exit("CONVERTER FAILED")

    sd_out = load_file(str(OUT_F))
    failures = []

    def check(name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
        if not ok:
            failures.append(name)

    exp_keys = set(sd_in.keys())
    for k in expect_quant:
        stem = k[: -len(".weight")]
        out_f, in_f = sd_in[k].shape
        exp_keys.add(stem + ".weight")
        exp_keys.add(stem + ".weight_scale")
        exp_keys.add(stem + ".comfy_quant")
    check("key-set exact", set(sd_out) == exp_keys,
          f"diff={sorted(set(sd_out) ^ exp_keys)[:4]}")

    bad, worst = [], 0.0
    for k in expect_quant:
        stem = k[: -len(".weight")]
        out_f, in_f = sd_in[k].shape
        w = sd_out[stem + ".weight"]
        s = sd_out[stem + ".weight_scale"]
        c = sd_out[stem + ".comfy_quant"]
        j = json.loads(bytes(c.tolist()).decode())
        ok_shape = tuple(w.shape) == (out_f, in_f // 2)
        if not (w.dtype == torch.int8 and ok_shape
                and s.dtype == torch.float32 and tuple(s.shape) == (out_f,)
                and j.get("format") == "convrot_w4a4" and j.get("convrot_groupsize") == 256):
            bad.append(stem)
            continue
        deq = dequantize_convrot_w4a4_weight(w, s, 256, 64, torch.float32)
        rel = ((deq - sd_in[k].float()).norm() / sd_in[k].float().norm()).item()
        worst = max(worst, rel)
        # Per-row signed int4 on gaussian data sits ~13-15%; flag structural bugs only.
        if not (0.05 < rel < 0.20):
            bad.append(f"{stem}(rel={rel:.3f})")
    check("w4a4 layout+numerics all", not bad, f"bad={bad[:2]} worst_rel_err={worst:.4f}")

    passthrough = [k for k in sd_in if k not in expect_quant]
    drifted = [k for k in passthrough
               if sd_out[k].dtype != sd_in[k].dtype or not torch.equal(sd_out[k], sd_in[k])]
    check("passthrough identical", not drifted, f"drifted={drifted[:3]}")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
