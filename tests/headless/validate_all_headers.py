import importlib
import json
import struct
import sys
import types

sys.path.insert(0, r"C:\Users\User\Documents\TEO\ComfyUI")
pkg = types.ModuleType("sensenova_pkg")
pkg.__path__ = [r"C:\Users\User\Documents\Clonitaditos\experimento\Comfyui-SenseNova-U1.5-Wrapper-T8\sensenova_u15"]
sys.modules["sensenova_pkg"] = pkg
loader = importlib.import_module("sensenova_pkg.loader")
BASE = r"C:\Users\User\Documents\TEO\ComfyUI\models\diffusion_models\SenseNovaU1.5"


def manual_validate(path, expect_format=None):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", {})
    variant = loader._validate_metadata(meta)
    quantized = any(k.endswith(".comfy_quant") for k in header)
    fmt = None
    if quantized:
        key = next(k for k in header if k.endswith(".comfy_quant"))
        begin, end = header[key]["data_offsets"]
        with open(path, "rb") as f:
            f.seek(8 + n + begin)
            payload = f.read(end - begin)
        fmt = json.loads(payload.decode("utf-8")).get("format")
    assert (fmt is not None) == quantized, "format detection mismatch"
    if expect_format:
        assert fmt == expect_format, f"format {fmt} != {expect_format}"
    contract = loader._checkpoint_contract(quantized=quantized, quant_format=fmt)
    stems = {
        name[: -len(".weight")]
        for name, shape in contract.items()
        if shape is not None and loader._is_quant_candidate(name, shape)
    }
    assert set(header) == set(contract), f"key diff: {sorted(set(header) ^ set(contract))[:4]}"
    for name, shape in contract.items():
        info = header[name]
        if shape is not None:
            assert tuple(info["shape"]) == shape, f"shape mismatch {name}"
        expected_dtype = loader._expected_storage_dtype(name, variant, quantized, stems)
        assert info["dtype"] == expected_dtype, f"dtype mismatch {name}: {info['dtype']} != {expected_dtype}"
    return len(contract), variant, fmt


n, v, _ = manual_validate(BASE + r"\SenseNova-U1.5-8B-MoT-T8.safetensors")
print(f"BF16 original : VALID variant={v} keys={n}")

n, v, fmt = manual_validate(
    BASE + r"\SenseNova-U1.5-8B-MoT-T8-int8-convrot-tagged.safetensors", expect_format="int8_tensorwise"
)
print(f"INT8 tagged   : VALID variant={v} keys={n} format={fmt}")

n, v, fmt = manual_validate(
    BASE + r"\SenseNova-U1.5-8B-MoT-T8-int4-convrot-tagged.safetensors", expect_format="convrot_w4a4"
)
print(f"INT4 tagged   : VALID variant={v} keys={n} format={fmt}")
