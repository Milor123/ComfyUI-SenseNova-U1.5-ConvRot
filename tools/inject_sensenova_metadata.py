"""Inject SenseNova provenance metadata into a converted safetensors file.

convert_to_quant writes `_quantization_metadata` but not the SenseNova tags the
wrapper's strict loader requires. This tool rewrites ONLY the JSON header and
streams the data section unchanged: safetensors buffer offsets are relative to
the start of the data section, so a different header size never invalidates
tensor offsets.

Usage:
  python inject_sensenova_metadata.py -i in.safetensors -o out.safetensors [--variant final|sft]
"""
import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

# Keep in sync with sensenova_u15/loader.py pinning.
VARIANTS = {
    "final": {
        "format": "sensenova-u1.5-mot",
        "config_sha256": "6497591f64cb0dd6917fbb10c0cd13024e5817179a9aa3700998eb137a553d6b",
        "source_repo": "sensenova/SenseNova-U1.5-8B-MoT",
        "source_revision": "1f6ec60423d29939dde4202fd82ae340b144e280",
    },
    "sft": {
        "format": "sensenova-u1.5-mot",
        "config_sha256": "6497591f64cb0dd6917fbb10c0cd13024e5817179a9aa3700998eb137a553d6b",
        "source_repo": "sensenova/SenseNova-U1.5-8B-MoT-SFT",
        "source_revision": "661834c5b5aee0f89958353511d6ac0ccaacb646",
    },
}
QUANTIZATION_TAG = "int8_tensorwise_convrot"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="final")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    if dst.resolve() == src.resolve():
        sys.exit("output must differ from input")

    with src.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))

    metadata = dict(header.get("__metadata__") or {})
    metadata.update(VARIANTS[args.variant])
    metadata["quantization"] = QUANTIZATION_TAG
    header["__metadata__"] = metadata

    new_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with dst.open("wb") as out:
        out.write(struct.pack("<Q", len(new_header)))
        out.write(new_header)
        with src.open("rb") as f:
            f.seek(8 + header_len)
            shutil.copyfileobj(f, out, length=64 * 1024 * 1024)

    print(f"tagged {src.name} -> {dst.name} ({dst.stat().st_size} bytes, variant={args.variant})")


if __name__ == "__main__":
    main()
