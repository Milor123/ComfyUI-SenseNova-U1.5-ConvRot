"""Find the FIRST divergent tensor: capture pre-layer0 pieces + layer0 internals."""
import os
import sys
import types

import torch

COMFY = r"C:\Users\User\Documents\TEO\ComfyUI"
WRAP = r"C:\Users\User\Documents\Clonitaditos\experimento\Comfyui-SenseNova-U1.5-Wrapper-T8"
BASE = os.path.join(COMFY, r"models\diffusion_models\SenseNovaU1.5")

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


def capture(tag, path, no_bridge):
    if no_bridge:
        os.environ["SENSENOVA_NO_BRIDGE"] = "1"
    else:
        os.environ.pop("SENSENOVA_NO_BRIDGE", None)
    patcher = loader.load_sensenova_model(path, dtype=torch.bfloat16)
    dm = patcher.model.diffusion_model
    caps = {}

    dm.fm_modules["timestep_embedder"].register_forward_hook(
        lambda m, a, o: caps.update(timestep=o.detach().float().clone()))
    dm.fm_modules["noise_scale_embedder"].register_forward_hook(
        lambda m, a, o: caps.update(noise=o.detach().float().clone()))
    dm.fm_modules["vision_model_mot_gen"].register_forward_hook(
        lambda m, a, o: caps.update(vision_stem=o.detach().float().clone()))

    layer0 = dm.language_model.model.layers[0]
    orig_gen = layer0.forward_generation

    def gen_wrap(*a, **k):
        caps["layer0_in"] = a[0].detach().float().clone()
        out = orig_gen(*a, **k)
        caps["layer0_out"] = out.detach().float().clone()
        return out

    layer0.forward_generation = gen_wrap
    attn = layer0.self_attn
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        mod = getattr(attn, name)
        mod.register_forward_hook(
            (lambda nm: lambda m, a, o: caps.update({nm: o.detach().float().clone()}))(name))
    for name in ("gate_proj", "up_proj", "down_proj"):
        mod = layer0.mlp.__getattr__(name) if hasattr(layer0.mlp, name) else getattr(layer0.mlp, name)
        mod.register_forward_hook(
            (lambda nm: lambda m, a, o: caps.update({nm: o.detach().float().clone()}))(name))

    torch.manual_seed(7)
    x = torch.rand(1, 3, 64, 64, dtype=torch.bfloat16) * 2 - 1
    t = torch.tensor([0.5])
    ids = torch.randint(1000, 100000, (1, 16))
    with torch.no_grad():
        dm._forward(x, t, text_input_ids=ids)
    del patcher, dm
    return caps


bf = capture("bf16", os.path.join(BASE, "SenseNova-U1.5-8B-MoT-T8.safetensors"), False)
q = capture("int8b", os.path.join(BASE, "SenseNova-U1.5-8B-MoT-T8-int8-convrot-tagged.safetensors"), False)

print("\n=== first-divergence hunt (rel err vs bf16) ===")
for k in sorted(bf):
    if k in q:
        r = bf[k]
        e = ((q[k] - r).norm() / r.norm()).item()
        print(f"{k:>12}: {e:.6f}")
