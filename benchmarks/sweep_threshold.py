"""Measure the confidence threshold instead of guessing it.

Runs a checkpoint over held-out suites recording the calibrated confidence per item
(Choice: answer.confidence; Noul: |noul_raw-0.5|*2, same as patterns.confidence_gate;
Score: answer.confidence), then sweeps 101 cutoffs and reports, per suite and pooled:

  coverage          share of items Von keeps (conf >= t)
  selective acc     accuracy on the kept items
  wilson90 lower    one-sided 90% lower bound on that accuracy
  deferred acc      accuracy on the items Von hands off (what the fallback inherits)

Threshold rules (all frozen numbers, fitted on these rows):
  honest   highest t where wilson90(kept) >= t         (stated confidence is earned up to here)
  target   lowest t where wilson90(kept) >= --target  (e.g. 0.90 kept accuracy)
  cascade  (only if --fallback dump given) lowest t where cascade acc >= fallback acc - margin

Usage:
  uv run python benchmarks/sweep_threshold.py --suites judge_heldout,probes,jabr_v2 --out results/threshold_sweep.json
  uv run python benchmarks/sweep_threshold.py --suites jev_easy,jev_standard,jev_hard \
      --fallback benchmarks/data/native_logit_q35_4b_q4km_full.json --out results/threshold_sweep_jev.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
from gate_standard import hit, jabr_rows, jev_rows, judge_rows, probe_rows  # noqa: E402


def wilson_lo(k: int, n: int, z: float = 1.2816) -> float:
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - s) / d


def conf_of(ans, qtype: str) -> float:
    if qtype == "noul":
        p = ans.noul_raw if getattr(ans, "noul_raw", None) is not None else ans.noul
        return abs(float(p) - 0.5) * 2.0
    return float(getattr(ans, "confidence", 0.0) or 0.0)


def run(ckpt: str, rows: List[dict], device: str, cache: str) -> Dict[str, dict]:
    done: Dict[str, dict] = {}
    if os.path.exists(cache):
        done = {r["id"]: r for r in json.load(open(cache))}
    todo = [r for r in rows if r["id"] not in done]
    if not todo:
        return done
    os.environ["VON_CHAINS_DIR"] = "off"
    from von.backends.option_marker_backend import OptionMarkerBackend
    b = OptionMarkerBackend(checkpoint_dir=ckpt, device=device)
    b._get_model()
    for i, r in enumerate(todo):
        payload = r["criteria"] if r["type"] != "score" else [r["criteria"][k] for k in sorted(r["criteria"], key=int)]
        q = {"type": r["type"], "instructions": r["instructions"]}
        if payload:
            q["criteria"] = payload
        t0 = time.perf_counter()
        a = b.evaluate(state=r["state"], questions={"q": q}).answers["q"]
        pick = ("yes" if a.noul >= 0.5 else "no") if r["type"] == "noul" else (a.score if r["type"] == "score" else a.choice)
        done[r["id"]] = {"id": r["id"], "suite": r["suite"], "family": r["family"], "type": r["type"], "pick": pick,
                         "expected": r["expected"], "hit": hit(r, pick), "conf": conf_of(a, r["type"]),
                         "seconds": time.perf_counter() - t0}
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"  {i+1}/{len(todo)}", flush=True)
            json.dump(list(done.values()), open(cache, "w"))
    json.dump(list(done.values()), open(cache, "w"))
    return done


def curve(items: List[dict], fallback: Dict[str, bool] | None) -> List[dict]:
    n = len(items)
    out = []
    for k in range(0, 101):
        t = k / 100
        kept = [r for r in items if r["conf"] >= t]
        deferred = [r for r in items if r["conf"] < t]
        kh = sum(r["hit"] for r in kept)
        row = {"t": t, "coverage": len(kept) / n, "n_kept": len(kept),
               "sel_acc": kh / len(kept) if kept else None,
               "wilson90_lo": wilson_lo(kh, len(kept)) if kept else None,
               "deferred_acc": (sum(r["hit"] for r in deferred) / len(deferred)) if deferred else None}
        if fallback is not None:
            ch = kh + sum(fallback.get(r["id"], False) for r in deferred)
            row["cascade_acc"] = ch / n
        out.append(row)
    return out


def pick_rules(c: List[dict], target: float, fallback_acc: float | None, margin: float) -> dict:
    """honest: highest t at which the kept set still earns its stated confidence (bound >= t).
    target: lowest t at which the kept set clears --target with 90% one-sided confidence.
    cascade: lowest t at which Von-then-fallback matches fallback-alone within --margin."""
    rules: dict = {}
    honest = [r["t"] for r in c if r["wilson90_lo"] is not None and r["wilson90_lo"] >= r["t"]]
    rules["honest"] = max(honest) if honest else None
    tgt = [r["t"] for r in c if r["wilson90_lo"] is not None and r["wilson90_lo"] >= target]
    rules["target"] = min(tgt) if tgt else None
    if fallback_acc is not None:
        cas = [r["t"] for r in c if r["cascade_acc"] >= fallback_acc - margin]
        rules["cascade"] = min(cas) if cas else None
    return rules


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints/von-1.2"))
    ap.add_argument("--device", default="openvino:cpu")
    ap.add_argument("--suites", default="judge_heldout,probes,jabr_v2")
    ap.add_argument("--fallback", default="", help="per-item dump of the fallback model ({'results':[{id,hit}]}) for cascade rule")
    ap.add_argument("--target", type=float, default=0.90)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows: List[dict] = []
    for s in a.suites.split(","):
        rows += (jabr_rows() if s == "jabr_v2" else probe_rows() if s == "probes"
                 else judge_rows() if s == "judge_heldout" else jev_rows(s.split("_", 1)[1]))
    cache = os.path.join(ROOT, "benchmarks/data/gate_cache", os.path.basename(a.ckpt.rstrip("/")) + ".conf.json")
    done = run(a.ckpt, rows, a.device, cache)
    items = [done[r["id"]] for r in rows]

    fb: Dict[str, bool] | None = None
    if a.fallback:
        fbd = json.load(open(a.fallback))["results"]
        # jev dump ids are bare ("hard-opus-a-long_policy-01"); cache ids are "jev:hard:<id>-<k>"
        fb = {}
        for r in fbd:
            fb[r["id"]] = bool(r["hit"])
        def key(i: str) -> str:
            return i.split(":", 2)[-1] if i.startswith("jev:") else i
        fb_mapped = {}
        for it in items:
            kk = key(it["id"])
            if kk in fb:
                fb_mapped[it["id"]] = fb[kk]
            elif kk.rsplit("-", 1)[0] in fb:
                fb_mapped[it["id"]] = fb[kk.rsplit("-", 1)[0]]
        items = [it for it in items if it["id"] in fb_mapped]
        fb = fb_mapped
        print(f"fallback matched {len(items)} items")

    groups = {"pooled": items}
    for s in sorted({r["suite"] for r in items}):
        groups[s] = [r for r in items if r["suite"] == s]
    for ty in sorted({r["type"] for r in items}):
        groups[f"type:{ty}"] = [r for r in items if r["type"] == ty]

    report = {"ckpt": a.ckpt, "suites": a.suites, "target": a.target, "fallback": a.fallback or None, "groups": {}}
    for name, g in groups.items():
        c = curve(g, fb)
        fba = (sum(fb[r["id"]] for r in g) / len(g)) if fb else None
        rules = pick_rules(c, a.target, fba, a.margin)
        report["groups"][name] = {"n": len(g), "acc_all": sum(r["hit"] for r in g) / len(g), "fallback_acc": fba,
                                  "rules": rules, "curve": c}
        print(f"\n== {name}  n={len(g)}  acc(all)={report['groups'][name]['acc_all']:.3f}"
              + (f"  fallback={fba:.3f}" if fba is not None else ""))
        print(f"  rules: {rules}")
        print("   t    cover  sel_acc  w90lo  defer_acc" + ("  cascade" if fb else ""))
        for r in c:
            if round(r["t"] * 100) % 10 == 0 or r["t"] in (0.8, 0.85, 0.95):
                sa = f"{r['sel_acc']:.3f}" if r["sel_acc"] is not None else "  -  "
                wl = f"{r['wilson90_lo']:.3f}" if r["wilson90_lo"] is not None else "  -  "
                da = f"{r['deferred_acc']:.3f}" if r["deferred_acc"] is not None else "  -  "
                print(f"  {r['t']:.2f}  {r['coverage']:.3f}  {sa}   {wl}  {da}" + (f"    {r['cascade_acc']:.3f}" if fb else ""))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
