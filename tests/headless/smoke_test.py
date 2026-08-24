"""Smoke test: SenseNova preset for convert_to_quant (tiny synthetic checkpoint).

Builds a minimal state dict replicating SenseNova-U1.5-8B-MoT structure,
runs the CLI converter with --sensenova --int8 --scaling-mode row --convrot,
then validates the output layout + numerics.
"""
import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(r"C:\Users\User\Documents\Clonitaditos\experimento\convert_to_quant")
WORK = Path(__file__).parent
IN_F = WORK / "in.safetensors"
OUT_F = WORK / "out_int8_convrot.safetensors"

H, INTER, KV, HDIM, VOCAB = 512, 1536, 128, 16, 1000


def lin(out_f, in_f, dtype):
    return torch.randn(out_f, in_f, dtype=dtype)


def build_sd():
    sd = {}
    for n in range(2):
        p = f"language_model.model.layers.{n}"
        twin_dt = torch.float32 if n == 0 else torch.bfloat16
        # norms (passthrough, 1D)
        sd[f"{p}.input_layernorm.weight"] = torch.randn(H, dtype=torch.bfloat16)
        sd[f"{p}.input_layernorm_mot_gen.weight"] = torch.randn(H, dtype=twin_dt)
        sd[f"{p}.post_attention_layernorm.weight"] = torch.randn(H, dtype=torch.bfloat16)
        sd[f"{p}.post_attention_layernorm_mot_gen.weight"] = torch.randn(H, dtype=twin_dt)
        for nm in ("q_norm", "k_norm", "q_norm_hw", "k_norm_hw"):
            sd[f"{p}.self_attn.{nm}.weight"] = torch.randn(HDIM, dtype=torch.bfloat16)
            sd[f"{p}.self_attn.{nm}_mot_gen.weight"] = torch.randn(HDIM, dtype=twin_dt)
        # attention linears (quantize candidates): base + mot_gen twins
        for nm, shape in (("q_proj", (H, H)), ("o_proj", (H, H)), ("k_proj", (KV, H)), ("v_proj", (KV, H))):
            sd[f"{p}.self_attn.{nm}.weight"] = lin(*shape, torch.bfloat16)
            sd[f"{p}.self_attn.{nm}_mot_gen.weight"] = lin(*shape, twin_dt)
        # mlp linears
        for nm, shape in (("gate_proj", (INTER, H)), ("up_proj", (INTER, H)), ("down_proj", (H, INTER))):
            sd[f"{p}.mlp.{nm}.weight"] = lin(*shape, torch.bfloat16)
            sd[f"{p}.mlp_mot_gen.{nm}.weight"] = lin(*shape, twin_dt)
    sd["language_model.model.norm.weight"] = torch.randn(H, dtype=torch.bfloat16)
    sd["language_model.model.norm_mot_gen.weight"] = torch.randn(H, dtype=torch.float32)
    sd["language_model.model.embed_tokens.weight"] = lin(VOCAB, H, torch.bfloat16)
    sd["language_model.lm_head.weight"] = lin(VOCAB, H, torch.bfloat16)
    # flow-matching embedder MLPs (fp32, quantizable 2D)
    for emb in ("timestep_embedder", "noise_scale_embedder"):
        sd[f"fm_modules.{emb}.mlp.0.weight"] = lin(128, 256, torch.float32)
        sd[f"fm_modules.{emb}.mlp.0.bias"] = torch.randn(128, dtype=torch.float32)
        sd[f"fm_modules.{emb}.mlp.2.weight"] = lin(256, 512, torch.float32)
        sd[f"fm_modules.{emb}.mlp.2.bias"] = torch.randn(256, dtype=torch.float32)
    # 4D convs (passthrough)
    sd["vision_model.embeddings.patch_embedding.weight"] = torch.randn(8, 3, 4, 4, dtype=torch.bfloat16)
    sd["vision_model.embeddings.patch_embedding.bias"] = torch.randn(8, dtype=torch.bfloat16)
    sd["vision_model.embeddings.dense_embedding.weight"] = torch.randn(16, 8, 2, 2, dtype=torch.bfloat16)
    sd["vision_model.embeddings.dense_embedding.bias"] = torch.randn(16, dtype=torch.bfloat16)
    sd["fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight"] = torch.randn(
        8, 3, 4, 4, dtype=torch.float32)
    sd["fm_modules.vision_model_mot_gen.embeddings.patch_embedding.bias"] = torch.randn(8, dtype=torch.float32)
    sd["fm_modules.vision_model_mot_gen.embeddings.dense_embedding.weight"] = torch.randn(
        16, 8, 2, 2, dtype=torch.float32)
    sd["fm_modules.vision_model_mot_gen.embeddings.dense_embedding.bias"] = torch.randn(16, dtype=torch.float32)
    sd["fm_modules.fm_head.conv1.weight"] = torch.randn(8, 8, 3, 3, dtype=torch.bfloat16)
    sd["fm_modules.fm_head.conv1.bias"] = torch.randn(8, dtype=torch.bfloat16)
    sd["fm_modules.fm_head.conv2.weight"] = torch.randn(4, 8, 3, 3, dtype=torch.bfloat16)
    sd["fm_modules.fm_head.conv2.bias"] = torch.randn(4, dtype=torch.bfloat16)
    return sd


def linear_keys(sd):
    """All 2D .weight keys eligible for quantization under the sensenova policy."""
    avoid = ("norm", "bias", "embed_tokens", "lm_head")
    return sorted(k for k, v in sd.items()
                  if k.endswith(".weight") and v.dim() == 2 and not any(a in k for a in avoid))


def comfy_json(t):
    return json.loads(bytes(t.tolist()).decode("utf-8"))


def main():
    sd_in = build_sd()
    save_file(sd_in, str(IN_F))
    expect_quant = linear_keys(sd_in)
    print(f"[setup] input tensors={len(sd_in)}  expected-quantized={len(expect_quant)}")

    cmd = [
        sys.executable, "-m", "convert_to_quant.cli.main",
        "-i", str(IN_F), "-o", str(OUT_F),
        "--int8", "--scaling-mode", "row", "--convrot",
        "--sensenova", "--simple", "--device", "cpu",
    ]
    print("[run]", " ".join(cmd[1:]))
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    tail = "\n".join((r.stdout or "").splitlines()[-25:])
    print(tail)
    if r.returncode != 0:
        print("STDERR:\n" + (r.stderr or "")[-3000:])
        sys.exit("CONVERTER FAILED")

    sd_out = load_file(str(OUT_F))
    results, failures = [], []

    def check(name, ok, detail=""):
        results.append((name, ok, detail))
        if not ok:
            failures.append(name)

    # 1) key-set integrity
    exp_out_keys = set(sd_in.keys())
    for k in expect_quant:
        exp_out_keys.add(k.replace(".weight", ".weight_scale"))
        exp_out_keys.add(k.replace(".weight", ".comfy_quant"))
    missing, extra = exp_out_keys - set(sd_out), set(sd_out) - exp_out_keys
    check("key-set exact", not missing and not extra,
          f"missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")

    # 2) every expected linear got proper int8 layout + convrot metadata
    bad = []
    for k in expect_quant:
        wkey = k
        skey = k.replace(".weight", ".weight_scale")
        ckey = k.replace(".weight", ".comfy_quant")
        w, s, c = sd_out.get(wkey), sd_out.get(skey), sd_out.get(ckey)
        ok = (w is not None and w.dtype == torch.int8
              and s is not None and s.dtype == torch.float32 and s.dim() == 2 and s.shape[0] == w.shape[0]
              and c is not None and c.dtype == torch.uint8)
        j = comfy_json(c) if ok else {}
        gin = sd_in[k].shape[1]
        div256 = (gin % 256 == 0)
        meta_ok = j.get("format") == "int8_tensorwise"
        rot_ok = (j.get("convrot") is True and j.get("convrot_groupsize") == 256) if div256 else True
        if not (ok and meta_ok and rot_ok):
            bad.append(f"{k}(w={w.dtype if w is not None else None},j={j})")
    check("int8 layout + comfy_quant meta (all 32)", not bad, f"bad={len(bad)} e.g. {bad[:2]}")

    # 3) passthrough byte-identical (excluding biases of quantized linears,
    #    which legitimately receive bias correction)
    quant_stems = {k[: -len(".weight")] for k in expect_quant}
    passthrough = [k for k in sd_in if k not in expect_quant and k.rsplit(".", 1)[0] not in quant_stems]
    drifted = [k for k in passthrough
               if sd_out[k].dtype != sd_in[k].dtype or not torch.equal(sd_out[k], sd_in[k])]
    check("passthrough identical (norms/embed/lm_head/convs)", not drifted, f"drifted={drifted[:5]}")
    corrected_biases = [k.replace(".weight", ".bias") for k in expect_quant
                        if k.replace(".weight", ".bias") in sd_in]
    bad_bias = [k for k in corrected_biases
                if sd_out[k].dtype != sd_in[k].dtype or not torch.isfinite(sd_out[k].float()).all()]
    check("corrected biases: dtype kept + finite", not bad_bias, f"bad={bad_bias[:5]}")

    # 4) numerics: dequant vs manual grouped Hadamard reference on layer1 base q_proj
    sys.path.insert(0, str(REPO_ROOT))
    from convert_to_quant.utils.convrot import build_hadamard

    kref = "language_model.model.layers.1.self_attn.q_proj.weight"
    W = sd_in[kref].float()
    Gsd = sd_out[kref.replace(".weight", ".comfy_quant")]
    G = comfy_json(Gsd)["convrot_groupsize"]
    Hm = build_hadamard(G, torch.device("cpu"), torch.float32).to(torch.float32)
    ng = W.shape[1] // G
    Wr = W.view(W.shape[0], ng, G)
    W_rot = torch.einsum("ong,gh->onh", Wr, Hm.t()).reshape(W.shape)
    q = sd_out[kref].float()
    s = sd_out[kref.replace(".weight", ".weight_scale")].float()
    deq = q * s
    ref_q = (W_rot * (W_rot.abs().amax(dim=1, keepdim=True) / 127.0)).round().clamp(-127, 127)
    err_deq_vs_rot = ((deq - W_rot).norm() / W_rot.norm()).item()
    check("numerics: dequant ~= W@H^T (<2%)", err_deq_vs_rot < 0.02, f"rel_err={err_deq_vs_rot:.5f}")
    print(f"[numerics] rel_err(deq vs rotated_ref)={err_deq_vs_rot:.5f}")

    # summary table
    print("\n=== SMOKE TEST RESULTS ===")
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
