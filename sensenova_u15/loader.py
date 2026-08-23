import hashlib
from pathlib import Path

import torch
from safetensors import safe_open

import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.utils

from .model import NUM_LAYERS
from .model_config import SenseNovaModelConfig


CONFIG_SHA256 = "6497591f64cb0dd6917fbb10c0cd13024e5817179a9aa3700998eb137a553d6b"
MODEL_REVISION = "1f6ec60423d29939dde4202fd82ae340b144e280"
MODEL_REPO = "sensenova/SenseNova-U1.5-8B-MoT"
SFT_MODEL_REVISION = "661834c5b5aee0f89958353511d6ac0ccaacb646"
SFT_MODEL_REPO = "sensenova/SenseNova-U1.5-8B-MoT-SFT"
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
        raise ValueError("SenseNova-U1.5 checkpoint format does not match this node version")
    if metadata.get("config_sha256") != CONFIG_SHA256:
        raise ValueError("SenseNova-U1.5 config digest does not match this node version")
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


def _checkpoint_contract(quantized=False, quant_format="int8_tensorwise"):
    # Static table generated from the official checkpoint header; building a
    # meta model here proved fragile inside full ComfyUI sessions where other
    # extensions patch module construction.
    from .checkpoint_contract import BASE_CONTRACT

    contract = dict(BASE_CONTRACT)
    if not quantized:
        return contract
    # Quantized checkpoints keep every base key but store rank-2 linear weights
    # packed plus fp32 scales and a uint8 JSON config tensor. Sidecar keys
    # REPLACE the trailing ".weight" segment (converter convention).
    quant_contract = {}
    for name, shape in contract.items():
        if _is_quant_candidate(name, shape):
            out_f, in_f = shape
            stem = name[: -len(".weight")]
            # Packed W4 halves K; int8 keeps full shape.
            quant_contract[name] = (out_f, in_f // 2) if quant_format == "convrot_w4a4" else shape
            quant_contract[stem + ".weight_scale"] = (
                (out_f,) if quant_format == "convrot_w4a4" else (out_f, 1)
            )
            # JSON payload length varies per layer; only dtype is pinned.
            quant_contract[stem + ".comfy_quant"] = None
        else:
            quant_contract[name] = shape
    return quant_contract


def _detect_quant_format(checkpoint):
    """Peek the first comfy_quant JSON payload; None means not quantized."""
    import json

    for key in checkpoint.keys():
        if not key.endswith(".comfy_quant"):
            continue
        try:
            payload = checkpoint.get_tensor(key)
            conf = json.loads(bytes(payload.numpy()).decode("utf-8"))
            return conf.get("format")
        except Exception:
            return None
    return None


def _expected_storage_dtype(name, variant, quantized, quant_weight_stems):
    if quantized:
        if name.endswith(".comfy_quant"):
            return "U8"
        if name.endswith(".weight_scale"):
            return "F32"
        if name.endswith(".weight") and name[: -len(".weight")] in quant_weight_stems:
            return "I8"
    return _storage_dtype(name, variant)


def _storage_dtype(name, variant="final"):
    if variant == "sft":
        return "BF16"
    if variant != "final":
        raise ValueError(f"unsupported SenseNova-U1.5 checkpoint variant: {variant}")
    if name.startswith((
        "fm_modules.vision_model_mot_gen.",
        "fm_modules.timestep_embedder.",
        "fm_modules.noise_scale_embedder.",
    )):
        return "F32"
    layer_prefix = "language_model.model.layers."
    if name.startswith(layer_prefix) and "_mot_gen" in name:
        layer = int(name[len(layer_prefix):].split(".", 1)[0])
        if layer < NUM_LAYERS - 3:
            return "F32"
    return "BF16"


def _validate_checkpoint_header(checkpoint):
    variant = _validate_metadata(checkpoint.metadata() or {})
    actual_keys = set(checkpoint.keys())
    quant_format = _detect_quant_format(checkpoint)
    quantized = quant_format is not None
    contract = _checkpoint_contract(quantized=quantized, quant_format=quant_format)
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
            f"SenseNova-U1.5 checkpoint key mismatch: quant_format={quant_format}, "
            f"contract_keys={len(expected_keys)}, file_keys={len(actual_keys)}, "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, shape in contract.items():
        tensor = checkpoint.get_slice(name)
        actual_shape = tuple(tensor.get_shape())
        if shape is not None and actual_shape != shape:
            raise ValueError(f"SenseNova-U1.5 checkpoint shape mismatch for {name}: {actual_shape} != {shape}")
        expected_dtype = _expected_storage_dtype(name, variant, quantized, quant_weight_stems)
        if tensor.get_dtype() != expected_dtype:
            raise ValueError(
                f"SenseNova-U1.5 checkpoint dtype mismatch for {name}: {tensor.get_dtype()} != {expected_dtype}"
            )
    return expected_keys, variant


def _validate_tokenizer_assets():
    asset_dir = Path(__file__).resolve().parent / "tokenizer"
    for name, expected in TOKENIZER_ASSET_SHA256.items():
        path = asset_dir / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
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
        raise ValueError("SenseNova-U1.5 loaded state dict does not match the validated header")

    load_device = comfy.model_management.get_torch_device()
    model_config = SenseNovaModelConfig({})
    manual_cast_dtype = comfy.model_management.unet_manual_cast(
        dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(dtype, manual_cast_dtype, device=load_device)

    parameters = comfy.utils.calculate_parameters(state_dict)
    initial_load_device = comfy.model_management.unet_inital_load_device(parameters, dtype)
    model = model_config.get_model(state_dict, device=initial_load_device)
    patcher_class = comfy.model_patcher.ModelPatcher if disable_dynamic else comfy.model_patcher.CoreModelPatcher
    patcher = patcher_class(
        model,
        load_device=load_device,
        offload_device=comfy.model_management.unet_offload_device(),
    )
    model.load_model_weights(state_dict, assign=patcher.is_dynamic())
    if state_dict:
        raise ValueError(f"SenseNova-U1.5 unused checkpoint keys after load: {sorted(state_dict)[:5]}")
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
