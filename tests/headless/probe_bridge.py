"""Verify quant_bridge forwards match convrot-invariance math on CUDA."""
import sys

import torch

sys.path.insert(0, r"C:\Users\User\Documents\TEO\ComfyUI")
sys.path.insert(0, r"C:\Users\User\Documents\Clonitaditos\experimento\Comfyui-SenseNova-U1.5-Wrapper-T8")

from safetensors import safe_open  # noqa: E402

from sensenova_u15.quant_bridge import make_sensenova_quant_ops  # noqa: E402
from comfy_kitchen.backends.eager.convrot_w4a4 import _build_hadamard  # noqa: E402

BASE = (
    r"C:\Users\User\Documents\TEO\ComfyUI\models\diffusion_models\SenseNovaU1.5"
)
STEM = "language_model.model.layers.0.self_attn.q_proj"
GS = 256

ops = make_sensenova_quant_ops()
torch.manual_seed(0)
x = torch.randn(8, 4096, dtype=torch.bfloat16, device="cuda")

for fname, expect_fmt in (
    ("SenseNova-U1.5-8B-MoT-T8-int8-convrot-tagged.safetensors", "int8_tensorwise"),
    ("SenseNova-U1.5-8B-MoT-T8-int4-convrot-tagged.safetensors", "convrot_w4a4"),
):
    lin = ops.Linear(4096, 4096, bias=False, device="cuda", dtype=torch.bfloat16)
    with safe_open(BASE + "\\" + fname, framework="pt", device="cpu") as f:
        sd = {
            "weight": f.get_tensor(STEM + ".weight"),
            "weight_scale": f.get_tensor(STEM + ".weight_scale"),
            "comfy_quant": f.get_tensor(STEM + ".comfy_quant"),
        }
    lin.load_state_dict(sd, strict=False)
    assert getattr(lin, "quant_format", None) == expect_fmt
    with torch.no_grad():
        y = lin(x)
        wdq = lin.weight.dequantize().float()
        Hm = _build_hadamard(GS, x.device, torch.float32)
        xf = x.float()
        if expect_fmt == "int8_tensorwise":
            # dequant keeps the ROTATED basis: correct math needs x rotated too
            w_ref, x_ref = wdq, torch.einsum(
                "bng,gk->bnk", xf.view(-1, 16, GS), Hm).reshape(xf.shape)
            w_wrong = wdq
        else:
            # kitchen dequant UNROTATES to the original basis: plain linear
            w_ref, x_ref = wdq, xf
            w_wrong = None
        ref_correct = x_ref @ w_ref.t()
        eA = ((y.float() - ref_correct).norm() / ref_correct.norm()).item()
        gA = (y.float().norm() / ref_correct.norm()).item()
        if w_wrong is not None:
            eB = ((y.float() - (xf @ w_wrong.t())).norm() / (xf @ w_wrong.t()).norm()).item()
        else:
            eB = float("nan")
        print(f"{expect_fmt:16s}: vs CORRECT={eA:.4f} (gain {gA:.4f}) | vs other-basis={eB:.4f}")
        assert eA < 0.05, f"{expect_fmt} still wrong!"
print("BRIDGE OK - both formats now compute rotation-consistent outputs")
