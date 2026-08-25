# Comfyui-SenseNova-U1.5-Wrapper-T8 (ConvRot Quantization Fork)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97-Weights-yellow)](https://huggingface.co/Milor123/ComfyUI-ConvRot-SenseNova-U1.5-8B-MoT-T8)
[![Upstream](https://img.shields.io/badge/upstream-T8mars-8A2BE2)](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8)

Native ComfyUI nodes for **SenseNova-U1.5-8B-MoT** (any-to-any: text-to-image, single-image editing, multi-reference editing, batch generation), forked from [T8mars' wrapper](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8) and extended with **ConvRot quantization support** so the 50 GB bf16 model runs on a 12 GB GPU.

**Quantized weights live here:** [Milor123/ComfyUI-ConvRot-SenseNova-U1.5-8B-MoT-T8](https://huggingface.co/Milor123/ComfyUI-ConvRot-SenseNova-U1.5-8B-MoT-T8)

## What this fork adds

- **ConvRot-aware quantized inference** (`sensenova_u15/quant_bridge.py`): explicit activation rotation + manual dequantization for INT8 ConvRot, and comfy-kitchen kernel routing for W4A4 / W4A8 — ComfyUI's generic dispatch silently skips the rotation these checkpoints require.
- **QuantizedTensor runtime guards** (`qt_guards.py`): neutralizes the dtype-cast paths in ComfyUI weight streaming (`cast_to_device` / `Tensor.to(dtype)`) that corrupt packed quantized tensors when LoRAs are applied.
- **Static checkpoint contract** (`sensenova_u15/checkpoint_contract.py`): deterministic key/shape/dtype validation immune to the partial `state_dict` that meta-model construction returns inside full ComfyUI sessions.
- **Hybrid checkpoint tooling** (`tools/make_hybrid_ladder.py`): byte-level merging of validated INT8 and W4A8 checkpoints into hybrid files — no re-quantization involved.
- **INT4 converter** (`tools/convert_sensenova_int4_convrot.py`): self-contained W4A4 / W4A8 / mixed-mode converter built directly on comfy-kitchen layouts, with sequential (mmap-free) reads for large files.
- **Headless test harness** (`tests/headless/`): capture, compare, first-divergence and validation scripts used to debug quantization numerically without launching the ComfyUI UI.

All upstream features are retained: text-to-image, single-image edit, 1-10 reference-image editing, 1-16 outputs per prompt, standard `KSampler`, U1.5 Final and SFT weights, the official 8-step speed LoRA, three-path `img_cfg` guidance, CFG Norm, and structured edit prompts. Workflows live in `examples/`.

## Install

Clone into `ComfyUI/custom_nodes/`:

```bash
git clone https://github.com/Milor123/ComfyUI-SenseNova-U1.5-ConvRot.git
```

Requires ComfyUI with **comfy-kitchen >= 0.2.31**. Download quantized weights from the [model repo](https://huggingface.co/Milor123/ComfyUI-ConvRot-SenseNova-U1.5-8B-MoT-T8) into `ComfyUI/models/diffusion_models/SenseNovaU1.5/`.

## Quantization findings

This model tolerates W4A8 activation quantization in its later transformer layers but **not in its earliest ones**: the hybrid release anchors layers 0-17 in INT8 (bf16 activations) and runs layers 18-41 in W4A8. The boundary was located empirically with a bisect ladder of hybrid checkpoints. See the model card for measured numbers (INT8: 0.43% pixel diff vs bf16; hybrid: visually indistinguishable in same-seed A/B).

## Credits

- [T8mars](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8) - upstream wrapper (Apache-2.0)
- [sensenova](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT) - original model (Apache-2.0)
- ConvRot - Hanzuliang et al., technical notes in [Comfy-Org/ComfyUI#14735](https://github.com/Comfy-Org/ComfyUI/issues/14735)
- [silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant) - INT8 conversion lineage
- [Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) - quantized tensor runtimes

## License

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).
