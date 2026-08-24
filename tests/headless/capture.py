"""Capture per-layer outputs for ONE checkpoint. Usage: capture.py <tag> <path> <out.pt> <no_bridge 0|1>"""
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
mm.unet_inital_load_device = lambda parameters, dtype: torch.device("cpu")
mm.unet_offload_device = lambda: torch.device("cpu")
mm.text_encoder_device = lambda: torch.device("cpu")

import comfy.ldm.modules.attention as _A  # noqa: E402
import sensenova_pkg.model as _smodel  # noqa: E402

_smodel.optimized_attention = _A.attention_pytorch

loader = __import__("sensenova_pkg.loader", fromlist=["x"])

tag, path, out_pt, no_bridge = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
if no_bridge:
    os.environ["SENSENOVA_NO_BRIDGE"] = "1"
else:
    os.environ.pop("SENSENOVA_NO_BRIDGE", None)

patcher = loader.load_sensenova_model(path, dtype=torch.bfloat16)
dm = patcher.model.diffusion_model
caps = {}
for i, layer in enumerate(dm.language_model.model.layers):
    orig = layer.forward_generation

    def wrapped(*a, _orig=orig, _i=i, **k):
        out = _orig(*a, **k)
        caps[f"layer_{_i}"] = out.detach().float().clone()
        return out

    layer.forward_generation = wrapped
dm.language_model.model.norm_mot_gen.register_forward_hook(
    lambda m, a, o: caps.update(final_norm=o.detach().float().clone()))
dm.fm_modules["fm_head"].register_forward_hook(
    lambda m, a, o: caps.update(fm_head=o.detach().float().clone()))

torch.manual_seed(7)
x = torch.rand(1, 3, 64, 64, dtype=torch.bfloat16) * 2 - 1
t = torch.tensor([0.5])
ids = torch.randint(1000, 100000, (1, 16))
with torch.no_grad():
    caps["velocity"] = dm._forward(x, t, text_input_ids=ids).detach().float().clone()

torch.save(caps, out_pt)
print(f"[{tag}] saved {len(caps)} captures -> {out_pt}; velocity std={caps['velocity'].std().item():.4f}")
