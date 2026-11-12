"""Segment draft_trace.json (chrome trace) into per-round draft/verify windows.
Anchors: cuda_recurrent_gated_delta_rule_kernel events = target GDN scan (verify);
topk/gather clusters between = draft tail. Sums GPU time per phase + launch counts."""
import json, sys
from collections import defaultdict

path = "/home/aisever/exl3_pipeline/draft_trace.json"
data = json.load(open(path))
evs = data["traceEvents"] if isinstance(data, dict) else data
kern = [e for e in evs if e.get("ph") == "X" and e.get("cat", "").lower() in ("kernel", "gpu_op", "cuda_runtime", "kernel_details")]
if not kern:
    kern = [e for e in evs if e.get("ph") == "X" and "kernel" in str(e.get("cat", "")).lower()]
print(f"kernel events: {len(kern)}")
kern.sort(key=lambda e: e["ts"])

GDN = "cuda_recurrent_gated_delta_rule"
gdn_ts = [e for e in kern if GDN in e["name"]]
print(f"GDN scan events: {len(gdn_ts)}")

# Cluster GDN events into verify windows (gap > 2ms starts new cluster)
gdn_sorted = sorted(e["ts"] for e in gdn_ts)
clusters = []
start = prev = gdn_sorted[0]
for t in gdn_sorted[1:]:
    if t - prev > 2000:  # >2ms gap (ts is microseconds)
        clusters.append((start, prev)); start = t
    prev = t
clusters.append((start, prev))
print(f"verify clusters: {len(clusters)}")

# Round = [end of verify cluster i, end of verify cluster i+1): tail(verify_i) + draft + verify_{i+1}
# Simpler: verify window = cluster span + 300us pad; draft window = between clusters
def in_win(e, a, b):
    return a <= e["ts"] <= b

stats = {"verify": [0.0, 0], "draft": [0.0, 0]}
names_verify = defaultdict(float)
names_draft = defaultdict(float)
for i in range(len(clusters) - 1):
    va, vb = clusters[i]
    va, vb = va - 400, vb + 400                     # verify window
    da, db = vb, clusters[i + 1][0] - 400           # draft window = until next verify
    for e in kern:
        if va <= e["ts"] <= vb:
            stats["verify"][0] += e["dur"]; stats["verify"][1] += 1
            names_verify[e["name"][:60]] += e["dur"]
        elif da < e["ts"] < db:
            stats["draft"][0] += e["dur"]; stats["draft"][1] += 1
            names_draft[e["name"][:60]] += e["dur"]
n = len(clusters) - 1
print(f"\nrounds segmented: {n}")
for ph in ("verify", "draft"):
    tot_ms, cnt = stats[ph]
    print(f"{ph:7s}: {tot_ms/1000/n:7.2f} ms/round GPU, {cnt/n:6.1f} launches/round")
print("\n== top verify kernels (ms/round) ==")
for nm, d in sorted(names_verify.items(), key=lambda kv: -kv[1])[:10]:
    print(f"  {d/1000/n:7.3f}  {nm}")
print("\n== top draft kernels (ms/round) ==")
for nm, d in sorted(names_draft.items(), key=lambda kv: -kv[1])[:10]:
    print(f"  {d/1000/n:7.3f}  {nm}")
