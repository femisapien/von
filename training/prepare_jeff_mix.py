"""Convert firelex/jeff training rows (Example format) into Von rows, with a leak filter against our held-out suites.

Jeff (MIT code, Apache weights; data sources listed in its docs/data-sources.md, each under its own licence) ships
converters for public QA / NLI / hallucination / sentiment sets, code-built probability questions with exact targets,
and 20-254-option long lists. We reuse its converters' *output* only. This script:

  1. reads Jeff Example rows ({id, suite, state, question:{type, instructions, criteria}, label, target, source})
  2. drops any row that touches our held-out gates: jabr v2 states, Hagetino probes, judge_heldout, and the
     JevBench public tiers (exact normalised state, or a normalised option text unique to one held-out item)
  3. writes Von rows {state, question, options:[{id, description}], label[, target], source, licence}

Von's trainer packs one question per row; Jeff's long-list rows (up to 254 options) exceed the 4096-token schema
budget, so lists are capped at --max-options (gold kept, distractors subsampled).

    python training/prepare_jeff_mix.py --jeff ~/scratch/ext/jeff/data --out ~/scratch/data_jeff_mix/train.jsonl
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import re
import sys
from typing import Any, Dict, Iterable, List, Set

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

_WS = re.compile(r"\s+")


def norm(text: Any) -> str:
    return _WS.sub(" ", str(text or "")).strip().lower()


def heldout_rows() -> List[dict]:
    from benchmarks.gate_standard import jabr_rows, jev_rows, judge_rows, probe_rows
    rows = jabr_rows() + probe_rows() + judge_rows()
    for tier in ("easy", "standard", "hard"):
        try:
            rows += jev_rows(tier)
        except FileNotFoundError:
            print(f"warning: jev {tier} public file missing; leak filter does not cover it", file=sys.stderr)
    return rows


class LeakGuard:
    """Exact normalised state, or an option text (5+ words) that appears in exactly one held-out item."""

    def __init__(self, rows: Iterable[dict]) -> None:
        self.states: Set[str] = set()
        opts: collections.Counter = collections.Counter()
        for r in rows:
            s = norm(r.get("state"))
            if s:
                self.states.add(s)
            crit = r.get("criteria") or {}
            vals = crit.values() if isinstance(crit, dict) else crit
            for v in set(norm(v) for v in vals):
                if len(v.split()) >= 5:
                    opts[v] += 1
        self.options = {t for t, c in opts.items() if c == 1}

    def leaks(self, state: str, descriptions: Iterable[str]) -> bool:
        if norm(state) in self.states:
            return True
        return any(norm(d) in self.options for d in descriptions)


def convert(row: dict, rng: random.Random, max_options: int) -> Dict[str, Any] | None:
    q = row["question"]
    qtype = q["type"]
    label = str(row["label"])
    if qtype == "noul":
        crit = q.get("criteria") or {}
        options = [{"id": "yes", "description": crit.get("true") or "Yes."},
                   {"id": "no", "description": crit.get("false") or "No."}]
        gold = "yes" if label.lower() in ("true", "yes") else "no"
        out = {"state": row["state"], "question": q["instructions"], "options": options, "label": gold}
        t = row.get("target")
        if isinstance(t, (int, float)) and not isinstance(t, bool):
            out["target"] = [float(t), 1.0 - float(t)]  # exact probability from the code-built questions
        return out
    if qtype == "choice":
        crit = q["criteria"]
        if label not in crit:
            return None
        ids = list(crit)
        if len(ids) > max_options:
            keep = [i for i in ids if i != label]
            rng.shuffle(keep)
            ids = sorted([label] + keep[: max_options - 1], key=ids.index)
        # The trainer infers the question type from option ids: contiguous 0..K-1 reads as a Score ladder.
        # cosmos_qa / truthful_qa use numeric ids for plain choices, so relabel those to letters.
        if all(str(i).isdigit() for i in ids):
            letters = {i: chr(ord("A") + n) for n, i in enumerate(ids)}
            options = [{"id": letters[i], "description": str(crit[i])} for i in ids]
            return {"state": row["state"], "question": q["instructions"], "options": options, "label": letters[label]}
        options = [{"id": i, "description": str(crit[i])} for i in ids]
        return {"state": row["state"], "question": q["instructions"], "options": options, "label": label}
    return None  # Jeff's open data carries no score rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jeff", required=True, help="Jeff data dir holding extra/, benchmark-train/, probability/, longlists/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-options", type=int, default=24)
    ap.add_argument("--cap-per-suite", type=int, default=12000, help="rows per source suite (winogrande alone is 40k)")
    ap.add_argument("--seed", type=int, default=23)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    guard = LeakGuard(heldout_rows())
    print(f"leak guard: {len(guard.states)} states, {len(guard.options)} unique option texts", file=sys.stderr)

    by_suite: Dict[str, List[dict]] = collections.defaultdict(list)
    stats: collections.Counter = collections.Counter()
    for path in sorted(glob.glob(os.path.join(a.jeff, "*", "train.jsonl"))):
        part = os.path.basename(os.path.dirname(path))
        for line in open(path, encoding="utf-8").read().split("\n"):  # not splitlines(): U+2028 inside JSON strings
            if not line.strip():
                continue
            row = json.loads(line)
            suite = f"{part}/{row.get('suite', '?')}"
            von = convert(row, rng, a.max_options)
            if von is None:
                stats["unconvertible"] += 1
                continue
            if guard.leaks(von["state"], (o["description"] for o in von["options"])):
                stats["leak:" + suite] += 1
                continue
            src = row.get("source") or {}
            von["source"] = f"jeff:{suite}"
            von["licence"] = src.get("license") or "see firelex/jeff docs/data-sources.md"
            by_suite[suite].append(von)

    out_rows: List[dict] = []
    for suite, rows in sorted(by_suite.items()):
        rng.shuffle(rows)
        kept = rows[: a.cap_per_suite]
        out_rows += kept
        print(f"{suite:40s} {len(rows):6d} -> {len(kept):6d}", file=sys.stderr)
    rng.shuffle(out_rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(out_rows), "suites": len(by_suite), **stats}), file=sys.stderr)


if __name__ == "__main__":
    main()
