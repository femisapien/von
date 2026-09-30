"""Render the HF model-card figure from a JevBench v1.5.x results JSON.

    uv run python benchmarks/plot_hf_card.py ~/scratch/v151.json hf/assets/jevbench_v151_encoders.png

Left: sealed-half competence (chance-corrected, 100 = perfect, 0 = chance) per tier for
Von and the other <=600M encoders on the board. Right: open-vs-sealed Choice competence,
the generalization gap the v1.5 penalty is built on.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))
want = {"Von": "Von 1.2 (395M)", "jeff": "jeff (400M)", "GLiNER2 (": "GLiNER2 (194M)",
        "openJev Verdict 1.4": "Verdict 1.4 (151M)", "Laya": "Laya (421M)"}
rows = {}
for r in d["systems"]:
    for k, name in want.items():
        if r["display"].startswith(k) and name not in rows:
            rows[name] = r["intelligence"]["per_type_split"]
tiers = ["easy", "standard", "judge", "hard"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
w = 0.16
for i, (name, pt) in enumerate(rows.items()):
    vals = [(pt["sealed|choice"]["tiers"][t] + pt["sealed|score"]["tiers"][t]) / 2 for t in tiers]
    ax[0].bar([x + (i - 2) * w for x in range(4)], vals, w, label=name)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(tiers)
ax[0].set_ylabel("sealed competence (Choice+Score)/2"); ax[0].axhline(0, color="k", lw=0.5)
ax[0].set_title("JevBench v1.5.1, sealed half, ≤600M encoders"); ax[0].legend(fontsize=7)
for name, pt in rows.items():
    o = [pt["open|choice"]["tiers"][t] for t in tiers]
    s = [pt["sealed|choice"]["tiers"][t] for t in tiers]
    ln, = ax[1].plot(range(4), o, marker="o", label=f"{name} open")
    ax[1].plot(range(4), s, marker="x", ls="--", color=ln.get_color())
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(tiers); ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_ylabel("Choice competence"); ax[1].set_title("open (solid) vs sealed (dashed) Choice")
ax[1].legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(out); print("wrote", out)
