"""Paired McNemar comparison of two gate_cache dumps (benchmarks/data/gate_cache/<name>.json).

    python benchmarks/compare_dumps.py --base von-1.2 --cand von-2-nano-r1 [--also jeff-0.8b-v1.2]

Rows are matched by id; groups are pooled, per suite, per question type, and per probe family.
Dumps come from gate_standard.py (Von checkpoints), run_http_baseline.py (any Jev-wire server),
or training/eval_decoder.py (decoder checkpoints).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stat_gate import mcnemar_gate  # noqa: E402


def load(name: str) -> Dict[str, dict]:
    path = name if name.endswith(".json") else os.path.join(HERE, "data/gate_cache", name + ".json")
    return {r["id"]: r for r in json.load(open(path))}


def _get(r: dict, *keys):
    for k in keys:
        if k in r:
            return r[k]
    return None


def compare(base: Dict[str, dict], cand: Dict[str, dict], base_name: str, cand_name: str) -> None:
    ids = [i for i in cand if i in base]
    if not ids:
        print("no overlapping ids"); return
    groups: Dict[str, List[str]] = {"pooled": ids}
    for s in sorted({cand[i]["suite"] for i in ids}):
        groups[s] = [i for i in ids if cand[i]["suite"] == s]
    for t in sorted({cand[i]["type"] for i in ids}):
        groups["type:" + t] = [i for i in ids if cand[i]["type"] == t]
    for f in sorted({str(cand[i]["family"]) for i in ids if cand[i]["suite"] == "probes"}):
        groups["probes/" + f] = [i for i in ids if cand[i]["suite"] == "probes" and str(cand[i]["family"]) == f]
    print(f"{'group':28s} {'n':>5s} {base_name[:10]:>10s} {cand_name[:10]:>10s} {'delta':>7s}  p      verdict")
    for name, ii in groups.items():
        r = mcnemar_gate([base[i]["hit"] for i in ii], [cand[i]["hit"] for i in ii], label=name)
        print(f"{name:28s} {len(ii):5d} {sum(base[i]['hit'] for i in ii)/len(ii):10.3f} {sum(cand[i]['hit'] for i in ii)/len(ii):10.3f} "
              f"{_get(r, 'delta_pp', 'delta'):+7.1f}  {_get(r, 'p', 'p_value', 'mcnemar_p'):.3f}  {_get(r, 'verdict')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--also", default="", help="second baseline to compare the candidate against")
    a = ap.parse_args()
    cand = load(a.cand)
    compare(load(a.base), cand, a.base, a.cand)
    if a.also:
        print()
        compare(load(a.also), cand, a.also, a.cand)


if __name__ == "__main__":
    main()
