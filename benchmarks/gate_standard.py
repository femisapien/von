"""Standard-tier ship gate for a continue-trained checkpoint vs von-1.2.

Paired exact McNemar (benchmarks/stat_gate.py) on:
  * JevBench public standard tier (original.jsonl, n=72)  -- the target
  * jabr v2 suite (benchmarks/jabr_cases_v2.py, 869 cases) -- out-of-sample
  * the pooled union                                        -- most power
plus hard tier (n=111) as a regression guard.

Both checkpoints are run through the same OptionMarkerBackend so the pair
is exact. Reads cached per-item results when present so a rerun of one
side does not redo the other.

    uv run python benchmarks/gate_standard.py --cand checkpoints/von-1.3-standard-spines --out benchmarks/data/gate_standard.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from benchmarks.eval_hard_fast import build_request  # noqa: E402
from benchmarks.jabr_cases_v2 import TASKS_V2  # noqa: E402
from benchmarks.stat_gate import mcnemar_gate  # noqa: E402

PUBLIC = os.environ.get("JEVBENCH_PUBLIC", os.path.expanduser("~/scratch/jevbench/datasets/public"))


def jev_rows(tier: str) -> List[dict]:
    fn = {"standard": "original.jsonl", "hard": "hard.jsonl", "easy": "easy.jsonl"}[tier]
    out = []
    with open(os.path.join(PUBLIC, fn), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                state, instructions, criteria = build_request(r, "plain")
                qtype = r["question"].get("type", "choice")
                out.append({"id": f"jev:{tier}:{r['id']}", "suite": f"jev_{tier}", "family": r.get("family"),
                            "type": qtype, "state": state, "instructions": instructions, "criteria": criteria,
                            "expected": r["expected"]})
    return out


def jabr_rows() -> List[dict]:
    out = []
    for task in TASKS_V2:
        crit = task.question.criteria
        if task.type == "score":
            criteria = {str(i): c for i, c in enumerate(crit)}
        elif task.type == "noul":
            criteria = dict(crit) if isinstance(crit, dict) else {}  # plain noul path, as von.judge
        else:
            criteria = dict(crit)
        for i, c in enumerate(task.cases):
            exp = c.expected
            if task.type == "noul":
                exp = "yes" if exp in (True, "yes", "true") else "no"
            out.append({"id": f"jabr:{task.id}:{i}", "suite": "jabr_v2", "family": task.id, "type": task.type,
                        "state": c.state, "instructions": task.question.instructions, "criteria": criteria,
                        "expected": exp})
    return out


def probe_rows() -> List[dict]:
    """Hagetino's von-shadow coding-agent probes (issue #21): 38 Choice + 14 Noul, held out from all training."""
    out = []
    with open(os.path.join(ROOT, "benchmarks/data/probes_von_shadow.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                r = json.loads(line)
                r["id"] = f"probe:{r['family']}:{i}"
                out.append(r)
    return out


def judge_rows() -> List[dict]:
    """Locked judge-shaped held-out set (benchmarks/data/judge_heldout.jsonl, sha 9107f051ec0de753):
    pairwise response preference from HelpSteer3 validation + preference-test-sets summarize/shp/pku/hhh.
    None of these splits are used by any Von training corpus."""
    with open(os.path.join(ROOT, "benchmarks/data/judge_heldout.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def hit(row: dict, pick) -> bool:
    exp = row["expected"]
    if row["type"] == "noul":
        return (str(pick).lower() in ("yes", "true")) == (str(exp).lower() in ("yes", "true"))
    if row["type"] == "score":
        try:
            return int(round(float(pick))) == int(exp)
        except (TypeError, ValueError):
            return False
    return pick == exp


def run_ckpt(ckpt: str, rows: List[dict], device: str, cache: str, chains: str = "") -> Dict[str, dict]:
    done: Dict[str, dict] = {}
    if os.path.exists(cache):
        done = {r["id"]: r for r in json.load(open(cache))}
    todo = [r for r in rows if r["id"] not in done]
    if not todo:
        return done
    if chains:
        os.environ["VON_CHAINS_DIR"] = chains
        os.environ.setdefault("VON_CHAINS_MODE", "bindall")
    else:
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
        res = b.evaluate(state=r["state"], questions={"q": q})
        a = res.answers["q"]
        pick = ("yes" if a.noul >= 0.5 else "no") if r["type"] == "noul" else (a.score if r["type"] == "score" else a.choice)
        done[r["id"]] = {"id": r["id"], "suite": r["suite"], "family": r["family"], "type": r["type"], "pick": pick,
                         "expected": r["expected"], "hit": hit(r, pick), "seconds": time.perf_counter() - t0}
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"  [{os.path.basename(ckpt)}] {i+1}/{len(todo)}", flush=True)
            json.dump(list(done.values()), open(cache, "w"))
    json.dump(list(done.values()), open(cache, "w"))
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(ROOT, "checkpoints/von-1.2"))
    ap.add_argument("--cand", default="", help="candidate checkpoint (default: same as --base)")
    ap.add_argument("--cand-chains", default="", help="run the candidate with this chains dir (bindall); cache key gets a +chains suffix")
    ap.add_argument("--device", default="openvino:cpu")
    ap.add_argument("--suites", default="jev_standard,jabr_v2,jev_hard")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows: List[dict] = []
    for s in a.suites.split(","):
        rows += (jabr_rows() if s == "jabr_v2" else probe_rows() if s == "probes"
                 else judge_rows() if s == "judge_heldout" else jev_rows(s.split("_", 1)[1]))
    print(f"{len(rows)} items across {a.suites}")
    cache_dir = os.path.join(ROOT, "benchmarks/data/gate_cache")
    os.makedirs(cache_dir, exist_ok=True)
    base = run_ckpt(a.base, rows, a.device, os.path.join(cache_dir, os.path.basename(a.base.rstrip("/")) + ".json"))
    cand_ckpt = a.cand or a.base
    cand_key = os.path.basename(cand_ckpt.rstrip("/")) + ("+chains" if a.cand_chains else "")
    cand = run_ckpt(cand_ckpt, rows, a.device, os.path.join(cache_dir, cand_key + ".json"), chains=a.cand_chains)

    gates = {}
    groups = {"jev_standard": [r for r in rows if r["suite"] == "jev_standard"],
              "jabr_v2": [r for r in rows if r["suite"] == "jabr_v2"],
              "jev_hard": [r for r in rows if r["suite"] == "jev_hard"]}
    groups["jev_easy"] = [r for r in rows if r["suite"] == "jev_easy"]
    groups["probes"] = [r for r in rows if r["suite"] == "probes"]
    groups["judge_heldout"] = [r for r in rows if r["suite"] == "judge_heldout"]
    groups["heldout_jabr+probes"] = groups["jabr_v2"] + groups["probes"]
    groups["pooled_standard+jabr"] = groups["jev_standard"] + groups["jabr_v2"]
    groups["pooled_all"] = rows
    groups = {k: v for k, v in groups.items() if v}
    for suite in ("jev_standard", "probes", "judge_heldout"):
        for fam in sorted({r["family"] for r in groups.get(suite, []) if r["family"]}):
            groups[f"{suite}/{fam}"] = [r for r in groups[suite] if r["family"] == fam]
    for name, g in groups.items():
        if not g:
            continue
        ids = [r["id"] for r in g]
        gates[name] = mcnemar_gate([base[i]["hit"] for i in ids], [cand[i]["hit"] for i in ids], label=name)
        gg = gates[name]
        print(f"{name:32s} n={gg['n']:4d} base={gg['acc_baseline']:.3f} cand={gg['acc_candidate']:.3f} "
              f"d={gg['delta_pp']:+.1f}pp p={gg['mcnemar_p']:.3f} MDE={gg['mde_pp_80pct']:.1f}pp {gg['verdict']}")
    json.dump({"base": a.base, "cand": cand_ckpt, "cand_chains": a.cand_chains, "gates": gates}, open(a.out, "w"), indent=1, default=str)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
