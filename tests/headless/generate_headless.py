"""Minimal flow-matching sampling loop, headless. Usage:
python generate_headless.py <model_path> <out.png> <steps> <no_bridge 0|1> [seed]
"""
import os
import sys
import types

import torch

COMFY = r"C:\Users\User\Documents\TEO\ComfyUI"
WRAP = r"C:\Users\User\Documents\Clonitaditos\experimento\Comfyui-SenseNova-U1.5-Wrapper-T8"

sys.path.insert(0, COMFY)
pkg = types.ModuleType("sensenova_pkg")
pkg.__path__ = [os.path.join(WRAP, "sensenova_u15")]
sys.modules["sensenova_pkg"] = pkg

import comfy.model_management as mm  # noqa: E402

mm.get_torch_device = lambda: torch.device("cpu")
mm.unet_inital_load_device = lambda p, d: torch.device("cpu")
mm.unet_offload_device = lambda: torch.device("cpu")
mm.text_encoder_device = lambda: torch.device("cpu")

import comfy.ldm.modules.attention as _A  # noqa: E402
import sensenova_pkg.model as _smodel  # noqa: E402

_smodel.optimized_attention = _A.attention_pytorch

loader = __import__("sensenova_pkg.loader", fromlist=["x"])
from sensenova_pkg.sampling import SenseNovaModelSampling  # noqa: E402

tag, path, out_png, steps, no_bridge = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5] == "1"
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 42
if no_bridge:
    os.environ["SENSENOVA_NO_BRIDGE"] = "1"
else:
    os.environ.pop("SENSENOVA_NO_BRIDGE", None)

patcher = loader.load_sensenova_model(path, dtype=torch.bfloat16)
dm = patcher.model.diffusion_model
base_model = patcher.model

# flow-matching schedule via the wrapper's own model_sampling (shift=3.0 default)
ms = SenseNovaModelSampling(base_model.model_config)
sigmas = ms.sigma(torch.linspace(1000, 0, steps + 1)).flip(0)  # sigmas ~1 -> 0
print("sigmas:", [round(float(s), 4) for s in sigmas])

torch.manual_seed(seed)
x = torch.randn(1, 3, 128, 128, dtype=torch.bfloat16)  # pixel latent (no VAE!)
text_ids = torch.randint(1000, 100000, (1, 16))

with torch.no_grad():
    for i in range(steps):
        t = sigmas[i].repeat(1)
        v = dm._forward(x, t, text_input_ids=text_ids)
        x = x + (sigmas[i + 1] - sigmas[i]) * v
        print(f"step {i}: t={float(sigmas[i]):.4f} -> {float(sigmas[i+1]):.4f} "
              f"| v_std={v.std().item():.3f} x_std={x.std().item():.3f}", flush=True)

img = x[0].float().clamp(-1, 1)
img = ((img + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
from PIL import Image  # noqa: E402

Image.fromarray(img).save(out_png)
print("saved", out_png)
