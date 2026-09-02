import hashlib
from pathlib import Path

import comfy.model_management  # type: ignore[import-not-found]
import comfy.model_patcher  # type: ignore[import-not-found]
import comfy.sd  # type: ignore[import-not-found]
import comfy.utils  # type: ignore[import-not-found]
import torch
from safetensors import safe_open

from .model import NUM_LAYERS
from .model_config import SenseNovaModelConfig


_qt_guards = None


def _get_qt_guards():
    """Import the sibling top-level qt_guards module, whatever the parent
    package is called on this install (ComfyUI sanitizes folder names)."""
    global _qt_guards
    if _qt_guards is None:
        try:
            import importlib

            parent = __package__.rsplit(".", 1)[0]
            _qt_guards = importlib.import_module(".qt_guards", parent)
        except Exception:
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            import qt_guards as _mod

            _qt_guards = _mod
    return _qt_guards


def _make_sensenova_patcher_class(disable_dynamic):
    """Patcher subclass that activates the quant guards only while moving
    SenseNova weights.

    ComfyUI passes an explicit dtype to model weights in exactly two patcher
    methods (``patch_weight_to_device`` for LoRA patches and
    ``patch_hook_weight_to_device`` for hooks). Wrapping only those with
    ``sensenova_quant_scope()`` keeps the guards inert for every other model
    in the process, so quantized text encoders from other nodes (MiniMax H3,
    Flux2 Klein) keep ComfyUI's native dtype handling in the same session.
    """
    guards = _get_qt_guards()
    guards.install_quant_guards()
    sensenova_quant_scope = guards.sensenova_quant_scope

    base_class = (
        comfy.model_patcher.ModelPatcher
        if disable_dynamic
        else comfy.model_patcher.CoreModelPatcher
    )

    class SenseNovaModelPatcher(base_class):
        def patch_weight_to_device(self, *args, **kwargs):
            with sensenova_quant_scope():
                return super().patch_weight_to_device(*args, **kwargs)

        def patch_hook_weight_to_device(self, *args, **kwargs):
            with sensenova_quant_scope():
                return super().patch_hook_weight_to_device(*args, **kwargs)

    return SenseNovaModelPatcher


CONFIG_SHA256 = "6497591f64cb0dd6917fbb10c0cd13024e5817179a9aa3700998eb137a553d6b"
MODEL_REVISION = "1f6ec60423d29939dde4202fd82ae340b144e280"
MODEL_REPO = "sensenova/SenseNova-U1.5-8B-MoT"
SFT_MODEL_REVISION = "661834c5b5aee0f89958353511d6ac0ccaacb646"
SFT_MODEL_REPO = "sensenova/SenseNova-U1.5-8B-MoT-SFT"
# Compat alias for older lora.py that imports FINAL_MODEL_REVISIONS
FINAL_MODEL_REVISIONS = (MODEL_REVISION, "19bc874ef6ffc97fda9837b40fc1d1301806158a")
MODEL_FORMAT = "sensenova-u1.5-mot"
MODEL_VARIANTS = {
    "final": {
        "source_repo": MODEL_REPO,
        "source_revision": MODEL_REVISION,
    },
    "sft": {
        "source_repo": SFT_MODEL_REPO,
        "source_revision": SFT_MODEL_REVISION,
    },
}
TOKENIZER_ASSET_SHA256 = {
    "config.json": CONFIG_SHA256,
    "tokenizer_config.json": "7433b95cec590c7d687259e81bca1bc4630ff39773dbf7f30f7df27a99748077",
    "special_tokens_map.json": "529306ff26be5cf190b4d96781e63c7dccd03ef0a39f87c0f1289d2d5a67a02f",
    "added_tokens.json": "d0ff3acec259fabfafc1ffa67638aeaf58203e5e604648fb44f072e4efe040c4",
    "vocab.json": "87a257b04b17642a0688c98cd1df89c398bda4fee532d6f88b38a659ecb4ac8d",
    "merges.txt": "455e0caaa06abffc663e9282dfe71dde07fd1991eaf24146bf08793c4dba4497",
}


def _validate_metadata(metadata):
    if metadata.get("format") != MODEL_FORMAT:
        raise ValueError(
            "SenseNova-U1.5 checkpoint format does not match this node version"
        )
    if metadata.get("config_sha256") != CONFIG_SHA256:
        raise ValueError(
            "SenseNova-U1.5 config digest does not match this node version"
        )
    source_repo = metadata.get("source_repo")
    source_revision = metadata.get("source_revision")
    for variant, contract in MODEL_VARIANTS.items():
        if source_repo == contract["source_repo"]:
            if source_revision != contract["source_revision"]:
                raise ValueError(
                    f"SenseNova-U1.5 {variant.upper()} model revision does not match this node version"
                )
            return variant
    raise ValueError("SenseNova-U1.5 checkpoint is not a supported Final or SFT model")


_QUANT_EXCLUDED_SUBSTRINGS = ("norm", "embed_tokens", "lm_head")


def _is_quant_candidate(name, shape):
    """Rank-2 linear weights that convert_to_quant's sensenova policy quantizes."""
    return (
        len(shape) == 2
        and name.endswith(".weight")
        and not any(token in name for token in _QUANT_EXCLUDED_SUBSTRINGS)
    )


def _checkpoint_contract(quant_formats=None):
    # Static table generated from the official checkpoint header; building a
    # meta model here proved fragile inside full ComfyUI sessions where other
    # extensions patch module construction.
    from .checkpoint_contract import BASE_CONTRACT

    contract = dict(BASE_CONTRACT)
    if not quant_formats:
        return contract
    # Quantized checkpoints keep every base key but store rank-2 linear weights
    # packed plus fp32 scales and a uint8 JSON config tensor. Sidecar keys
    # REPLACE the trailing ".weight" segment (converter convention). Formats
    # are per-layer (mixed int8/w4a4 checkpoints are supported).
    quant_formats = quant_formats or {}
    quant_contract = {}
    for name, shape in contract.items():
        if _is_quant_candidate(name, shape):
            out_f, in_f = shape
            stem = name[: -len(".weight")]
            fmt = quant_formats.get(stem, "int8_tensorwise")
            if fmt == "convrot_w4a4":
                # Packed W4 halves K; per-row scales are flat (out,).
                quant_contract[name] = (out_f, in_f // 2)
                quant_contract[stem + ".weight_scale"] = (out_f,)
            elif fmt == "asym_w4a8_int8":
                # Packed W4 halves K; fp8 per-group scales (group 16) + fp32
                # per-channel + Lloyd-Max codebook.
                quant_contract[name] = (out_f, in_f // 2)
                quant_contract[stem + ".weight_s_rel"] = (out_f, in_f // 16)
                quant_contract[stem + ".weight_s_channel"] = (out_f,)
                quant_contract[stem + ".weight_codebook"] = (16,)
            else:
                quant_contract[name] = shape
                quant_contract[stem + ".weight_scale"] = (out_f, 1)
            # JSON payload length varies per layer; only dtype is pinned.
            quant_contract[stem + ".comfy_quant"] = None
        else:
            quant_contract[name] = shape
    return quant_contract


def _read_quant_formats(checkpoint):
    """Per-layer format map from every comfy_quant payload; empty = not quantized."""
    import json

    formats = {}
    for key in checkpoint.keys():
        if not key.endswith(".comfy_quant"):
            continue
        try:
            payload = checkpoint.get_tensor(key)
            conf = json.loads(bytes(payload.numpy()).decode("utf-8"))
            formats[key[: -len(".comfy_quant")]] = conf.get("format")
        except Exception:
            return {}
    return formats


def _expected_storage_dtype(name, variant, quantized, quant_weight_stems):
    if quantized:
        if name.endswith(".comfy_quant"):
            return "U8"
        if name.endswith(".weight_scale"):
            return "F32"
        if name.endswith(".weight_s_rel"):
            return "F8_E4M3"
        if name.endswith(".weight_s_channel"):
            return "F32"
        if name.endswith(".weight_codebook"):
            return "F32"
        if name.endswith(".weight") and name[: -len(".weight")] in quant_weight_stems:
            return "I8"
    return _storage_dtype(name, variant)


def _storage_dtype(name, variant="final"):
    if variant == "sft":
        return "BF16"
    if variant != "final":
        raise ValueError(f"unsupported SenseNova-U1.5 checkpoint variant: {variant}")
    if name.startswith(
        (
            "fm_modules.vision_model_mot_gen.",
            "fm_modules.timestep_embedder.",
            "fm_modules.noise_scale_embedder.",
        )
    ):
        return "F32"
    layer_prefix = "language_model.model.layers."
    if name.startswith(layer_prefix) and "_mot_gen" in name:
        layer = int(name[len(layer_prefix) :].split(".", 1)[0])
        if layer < NUM_LAYERS - 3:
            return "F32"
    return "BF16"


def _validate_checkpoint_header(checkpoint):
    variant = _validate_metadata(checkpoint.metadata() or {})
    actual_keys = set(checkpoint.keys())
    quant_formats = _read_quant_formats(checkpoint)
    quantized = bool(quant_formats)
    contract = _checkpoint_contract(quant_formats=quant_formats)
    quant_weight_stems = {
        name[: -len(".weight")]
        for name, shape in contract.items()
        if shape is not None and _is_quant_candidate(name, shape)
    }
    expected_keys = set(contract)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)[:5]
        unexpected = sorted(actual_keys - expected_keys)[:5]
        raise ValueError(
            f"SenseNova-U1.5 checkpoint key mismatch: quant_formats={sorted(set(quant_formats.values()))}, "
            f"contract_keys={len(expected_keys)}, file_keys={len(actual_keys)}, "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, shape in contract.items():
        tensor = checkpoint.get_slice(name)
        actual_shape = tuple(tensor.get_shape())
        if shape is not None and actual_shape != shape:
            raise ValueError(
                f"SenseNova-U1.5 checkpoint shape mismatch for {name}: {actual_shape} != {shape}"
            )
        expected_dtype = _expected_storage_dtype(
            name, variant, quantized, quant_weight_stems
        )
        if tensor.get_dtype() != expected_dtype:
            raise ValueError(
                f"SenseNova-U1.5 checkpoint dtype mismatch for {name}: {tensor.get_dtype()} != {expected_dtype}"
            )
    return expected_keys, variant


def _validate_tokenizer_assets():
    asset_dir = Path(__file__).resolve().parent / "tokenizer"
    for name, expected in TOKENIZER_ASSET_SHA256.items():
        path = asset_dir / name
        digest = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        if digest != expected:
            raise ValueError(f"SenseNova-U1.5 tokenizer asset digest mismatch: {name}")


def load_sensenova_model(model_path, dtype=torch.bfloat16, disable_dynamic=False):
    if Path(model_path).suffix.lower() not in (".safetensors", ".sft"):
        raise ValueError("SenseNova-U1.5 loader accepts safetensors files only")
    with safe_open(model_path, framework="pt", device="cpu") as checkpoint:
        expected_keys, variant = _validate_checkpoint_header(checkpoint)
    state_dict, metadata = comfy.utils.load_torch_file(model_path, return_metadata=True)
    loaded_variant = _validate_metadata(metadata)
    if loaded_variant != variant:
        raise ValueError("SenseNova-U1.5 checkpoint metadata changed while loading")
    if set(state_dict) != expected_keys:
        raise ValueError(
            "SenseNova-U1.5 loaded state dict does not match the validated header"
        )

    load_device = comfy.model_management.get_torch_device()
    model_config = SenseNovaModelConfig({})
    manual_cast_dtype = comfy.model_management.unet_manual_cast(
        dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(dtype, manual_cast_dtype, device=load_device)

    parameters = comfy.utils.calculate_parameters(state_dict)
    initial_load_device = comfy.model_management.unet_inital_load_device(
        parameters, dtype
    )
    model = model_config.get_model(state_dict, device=initial_load_device)
    patcher_class = _make_sensenova_patcher_class(disable_dynamic)
    patcher = patcher_class(
        model,
        load_device=load_device,
        offload_device=comfy.model_management.unet_offload_device(),
    )
    model.load_model_weights(state_dict, assign=patcher.is_dynamic())
    if state_dict:
        raise ValueError(
            f"SenseNova-U1.5 unused checkpoint keys after load: {sorted(state_dict)[:5]}"
        )
    patcher.cached_patcher_init = (load_sensenova_model, (model_path, dtype))
    patcher.set_attachments(
        "sensenova_checkpoint",
        {
            "variant": variant,
            "source_repo": MODEL_VARIANTS[variant]["source_repo"],
            "source_revision": MODEL_VARIANTS[variant]["source_revision"],
        },
    )
    return patcher


def load_sensenova_clip():
    _validate_tokenizer_assets()
    target = SenseNovaModelConfig({}).clip_target()
    return comfy.sd.CLIP(target, parameters=0, state_dict=[])


def load_pixel_vae():
    return comfy.sd.VAE(sd={"pixel_space_vae": torch.tensor(1.0)})
