"""Build hybrid int8+w4a8 checkpoints by merging two existing converted files.

Takes the validated all-int8 and all-w4a8 checkpoints and emits rungs where a
range of transformer layers uses w4a8 storage while everything else keeps the
int8-convrot format. Pure byte-level remix of already-validated tensors: no
requantization happens here.

Usage (ComfyUI venv python):
    python tools/make_hybrid_ladder.py [--outdir DIR] [--dry-run]
"""

import argparse
import json
import struct
import sys
from pathlib import Path

from safetensors import safe_open

MODELS_DIR = Path(
    r"C:\Users\User\Documents\TEO\ComfyUI\models\diffusion_models\SenseNovaU1.5"
)
INT8_FILE = MODELS_DIR / "SenseNova-U1.5-8B-MoT-T8-int8-convrot-tagged.safetensors"
W4A8_FILE = MODELS_DIR / "SenseNova-U1.5-8B-MoT-T8-w4a8-convrot-tagged.safetensors"

SIDECAR_SUFFIXES = (
    ".weight",
    ".weight_scale",
    ".weight_s_rel",
    ".weight_s_channel",
    ".weight_codebook",
    ".comfy_quant",
    ".bias",
)

RUNGS = [
    # (tag, predicate(layer_idx) -> bool for w4a8 migration)
    ("hybw4a8-L00-07", lambda i: i < 8),
    ("hybw4a8-L34-41", lambda i: i >= 34),
    ("hybw4a8-L26-41", lambda i: i >= 26),
    ("hybw4a8-L18-41", lambda i: i >= 18),
    ("hybw4a8-L10-41", lambda i: i >= 10),
]


def read_header(path):
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    meta = header.pop("__metadata__", None)
    return header, meta, 8 + header_len


def stem_of(key):
    for suffix in SIDECAR_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def group_by_stem(header):
    groups = {}
    for key, entry in header.items():
        groups.setdefault(stem_of(key), {})[key] = entry
    return groups


def quant_format_of(int8_groups, stem):
    entry = int8_groups.get(stem, {}).get(stem + ".comfy_quant")
    if entry is None:
        return None
    return entry  # placeholder; real JSON parsed lazily by caller if needed


def layer_index(stem):
    marker = "language_model.model.layers."
    if not stem.startswith(marker):
        return None
    rest = stem[len(marker):]
    token = rest.split(".", 1)[0]
    return int(token) if token.isdigit() else None


def build_rung(tag, predicate, outdir, dry_run=False):
    out_path = Path(outdir) / f"SenseNova-U1.5-8B-MoT-T8-{tag}.safetensors"
    header_i, meta, data_start_i = read_header(INT8_FILE)
    header_w, _, data_start_w = read_header(W4A8_FILE)
    groups_i = group_by_stem(header_i)
    groups_w = group_by_stem(header_w)

    migrated, kept = [], []
    plan = []  # (from_w4a8, absolute_offset, length)
    new_header = {}
    if meta:
        new_header["__metadata__"] = meta

    cursor = 0
    for stem in sorted(groups_i):
        idx = layer_index(stem)
        use_w = (
            idx is not None
            and predicate(idx)
            and stem + ".comfy_quant" in groups_w.get(stem, {})
        )
        src_start = data_start_w if use_w else data_start_i
        src_groups = groups_w if use_w else groups_i
        if use_w:
            migrated.append(stem)
        else:
            kept.append(stem)
        for key in sorted(src_groups[stem]):
            entry = src_groups[stem][key]
            rel_start, rel_end = entry["data_offsets"]
            new_header[key] = {
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "data_offsets": [cursor, cursor + (rel_end - rel_start)],
            }
            plan.append((use_w, src_start + rel_start, rel_end - rel_start))
            cursor += rel_end - rel_start
        # sanity: migrated stems must expose their w4a8 sidecar set
        if use_w:
            needed = {stem + s for s in (".weight", ".weight_s_rel", ".weight_s_channel", ".comfy_quant")}
            missing = needed - set(groups_w[stem])
            if missing:
                raise SystemExit(f"{tag}: w4a8 source missing sidecars for {stem}: {missing}")

    print(f"[{tag}] migrated={len(migrated)} layers -> w4a8 | kept={len(kept)} stems on int8")
    print(f"[{tag}] tensors={len(plan)} bytes={cursor / 2**30:.2f}GiB -> {out_path.name}")
    if dry_run:
        return out_path

    header_bytes = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(header_bytes) % 8) % 8
    with open(out_path, "wb") as fo, open(INT8_FILE, "rb") as fi, open(W4A8_FILE, "rb") as fw:
        fo.write(struct.pack("<Q", len(header_bytes) + pad))
        fo.write(header_bytes)
        fo.write(b"\x20" * pad)
        written = 0
        for is_w4a8_src, offset, length in plan:
            src = fw if is_w4a8_src else fi
            src.seek(offset)
            remaining = length
            while remaining > 0:
                chunk = src.read(min(remaining, 1 << 24))
                if not chunk:
                    raise SystemExit(f"{tag}: short read at offset {offset}")
                fo.write(chunk)
                remaining -= len(chunk)
                written += len(chunk)
    assert written == cursor
    return out_path


def validate(path, expect_migrated_count):
    sys.path.insert(0, r"C:\Users\User\Documents\TEO\ComfyUI")  # for comfy imports
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sensenova_u15.loader import _read_quant_formats, _validate_checkpoint_header

    with safe_open(str(path), framework="pt", device="cpu") as f:
        formats = _read_quant_formats(f)
        _validate_checkpoint_header(f)
    counts = {}
    for fmt in formats.values():
        counts[fmt] = counts.get(fmt, 0) + 1
    assert counts.get("asym_w4a8_int8", 0) == expect_migrated_count * 14, (
        f"{path.name}: expected {expect_migrated_count * 14} w4a8 layers, got {counts}"
    )
    print(f"VALIDATED {path.name}: {counts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(MODELS_DIR))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for tag, predicate in RUNGS:
        count = sum(1 for i in range(42) if predicate(i))
        path = Path(args.outdir) / f"SenseNova-U1.5-8B-MoT-T8-{tag}.safetensors"
        if path.exists():
            print(f"[{tag}] already exists, skipping build")
        else:
            build_rung(tag, predicate, args.outdir, dry_run=args.dry_run)
        validate(path, count)


if __name__ == "__main__":
    main()
