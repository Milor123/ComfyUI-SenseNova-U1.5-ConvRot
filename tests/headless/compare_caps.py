import torch

W = r"C:\Users\User\AppData\Local\Temp\opencode\sensenova_smoke"
b = torch.load(W + r"\caps_bf16.pt")
g = torch.load(W + r"\caps_int8b.pt")
s = torch.load(W + r"\caps_int8s.pt")

print(f"{'layer':>6} {'bridge':>10} {'stock':>10}")
prev = 0.0
for i in range(42):
    ref = b[f"layer_{i}"]
    eb = ((g[f"layer_{i}"] - ref).norm() / ref.norm()).item()
    es = ((s[f"layer_{i}"] - ref).norm() / ref.norm()).item()
    flag = " <-- jump" if eb > max(3 * prev, 0.05) else ""
    print(f"{i:>6} {eb:>10.4f} {es:>10.4f}{flag}")
    prev = eb
for k in ("final_norm", "fm_head"):
    r = b[k]
    print(f"{k:>10}: bridge={((g[k]-r).norm()/r.norm()).item():.4f} "
          f"stock={((s[k]-r).norm()/r.norm()).item():.4f}")
print(f"{'velocity':>10}: bridge={((g['velocity']-b['velocity']).norm()/b['velocity'].norm()).item():.4f} "
      f"stock={((s['velocity']-b['velocity']).norm()/b['velocity'].norm()).item():.4f}")
