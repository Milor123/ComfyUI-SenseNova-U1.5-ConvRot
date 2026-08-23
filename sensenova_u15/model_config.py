import math

import torch

import comfy.conds
import comfy.latent_formats
import comfy.model_base
import comfy.supported_models_base

from .model import SenseNovaU15
from .conditioning import block_causal_mask, condition_input_ids, conditioned_input_length, preprocess_references, smart_resize, thw_indexes
from .sampling import SenseNovaModelSampling, time_snr_shift
from .text_encoder import SenseNovaTextEncoder, SenseNovaTokenizer


class CONDSharedRegular(comfy.conds.CONDRegular):
    """Keep a prefix tensor at one copy per guidance branch.

    ComfyUI normally repeats every condition to the latent batch. SenseNova's
    text/reference prefix is identical for all generated variants, so the model
    computes it once and expands only the much smaller per-layer KV tensors.
    """

    def process_cond(self, batch_size, **kwargs):
        return self._copy_with(self.cond)


class CONDSharedList(comfy.conds.CONDList):
    """List counterpart to :class:`CONDSharedRegular` for reference images."""

    def process_cond(self, batch_size, **kwargs):
        return self._copy_with(self.cond)


class SenseNovaBaseModel(comfy.model_base.BaseModel):
    def __init__(self, model_config, device=None):
        super().__init__(
            model_config,
            comfy.model_base.ModelType.FLOW,
            device=device,
            unet_model=SenseNovaU15,
        )
        self.model_sampling = SenseNovaModelSampling(model_config)
        self.memory_usage_factor_conds = ("reference_images",)

    def process_timestep(self, timestep, **kwargs):
        base_timestep = timestep / self.model_sampling.multiplier
        return 1.0 - time_snr_shift(self.model_sampling.shift, 1.0 - base_timestep)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        text_input_ids = kwargs.get("text_input_ids")
        if text_input_ids is not None:
            reference_images = kwargs.get("sensenova_reference_images")
            if reference_images is not None:
                reference_images = preprocess_references(reference_images)
                reference_grids = [(image.shape[-2] // 32, image.shape[-1] // 32) for image in reference_images]
                text_input_ids = condition_input_ids(
                    text_input_ids,
                    reference_grids,
                    image_only=kwargs.get("sensenova_reference_mode") == "image_only",
                )
                indexes = thw_indexes(text_input_ids, reference_grids)
                out["prefix_indexes"] = CONDSharedRegular(indexes)
                out["prefix_mask"] = CONDSharedRegular(block_causal_mask(indexes))
                out["reference_images"] = CONDSharedList(reference_images)
            out["text_input_ids"] = CONDSharedRegular(text_input_ids)
        return out

    def extra_conds_shapes(self, **kwargs):
        images = kwargs.get("sensenova_reference_images")
        if images is None:
            return {}
        max_pixels = min(2048 * 2048, (4096 * 4096) // len(images))
        resized = [smart_resize(*image.shape[1:3], max_pixels=max_pixels) for image in images]
        reference_grids = [(height // 32, width // 32) for height, width in resized]
        out = {"reference_images": [1, 3, sum(height * width for height, width in resized)]}
        text_input_ids = kwargs.get("text_input_ids")
        if text_input_ids is not None:
            length = conditioned_input_length(
                text_input_ids.shape[1],
                reference_grids,
                image_only=kwargs.get("sensenova_reference_mode") == "image_only",
            )
            out["prefix_mask"] = [1, 1, length, length]
        return out

    def memory_required(self, input_shape, cond_shapes={}):
        memory = super().memory_required(input_shape, cond_shapes)
        mask_shapes = cond_shapes.get("prefix_mask", ())
        return memory + sum(math.prod(shape) * 4 for shape in mask_shapes)


class SenseNovaModelConfig(comfy.supported_models_base.BASE):
    unet_config = {"image_model": "sensenova_u15"}
    sampling_settings = {"shift": 3.0, "noise_scale": 1.0}
    latent_format = comfy.latent_formats.HiDreamO1Pixel
    memory_usage_factor = 0.033
    supported_inference_dtypes = [torch.bfloat16, torch.float32]
    optimizations = {"fp8": False}

    def get_model(self, state_dict, prefix="", device=None):
        if any(key.endswith(".comfy_quant") for key in state_dict):
            # Quantized checkpoints need convrot-aware forwards: ComfyUI's
            # generic dispatch skips the activation rotation, so route through
            # our bridge instead of stock mixed_precision_ops.
            from .quant_bridge import make_sensenova_quant_ops

            self.custom_operations = make_sensenova_quant_ops()
        return SenseNovaBaseModel(self, device=device)

    def process_unet_state_dict(self, state_dict):
        state_dict.pop("language_model.lm_head.weight", None)
        return state_dict

    def process_vae_state_dict(self, state_dict):
        return {"pixel_space_vae": torch.tensor(1.0)}

    def clip_target(self, state_dict={}):
        return comfy.supported_models_base.ClipTarget(SenseNovaTokenizer, SenseNovaTextEncoder)
