"""Score a train_decoder_head.py checkpoint on the gate suites; write a gate_cache dump.

Same output shape as benchmarks/run_http_baseline.py (id, suite, family, type, pick, expected,
hit, conf, seconds), so benchmarks/gate_standard.py and sweep_threshold.py can compare it
against Von-1.2 and the Jeff-0.8B dump without an HTTP server in the loop.

    python training/eval_decoder.py --checkpoint checkpoints/von-2-nano --name von-2-nano-r1 \
        --suites judge_heldout,probes,jabr_v2,jev_easy,jev_standard,jev_hard
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
from benchmarks.gate_standard import hit, jabr_rows, jev_rows, judge_rows, probe_rows  # noqa: E402
from training.clef_head import collate_records, encode_record, systemone_answer  # noqa: E402
from training.train_decoder_head import build_model  # noqa: E402


def gate_question(r: dict) -> Dict[str, Any]:
    """gate_standard row -> Jev-wire question (same mapping run_http_baseline.ask uses)."""
    q: Dict[str, Any] = {"type": r["type"], "instructions": r["instructions"]}
    if r["type"] == "score":
        q["criteria"] = [r["criteria"][k] for k in sorted(r["criteria"], key=int)]
    elif r["criteria"]:
        q["criteria"] = r["criteria"]
    return q


def load_checkpoint(ckpt: str, device: torch.device):
    cfg = json.load(open(os.path.join(ckpt, "decoder_config.json")))
    head = {k: v for k, v in cfg["head"].items() if k != "hidden_size"}
    head["dropout"] = 0.0
    model, tokenizer = build_model(cfg["base_model_id"], head, 8, 16, 0.0, device,
                                   adapter_dir=os.path.join(ckpt, "adapter"), head_path=os.path.join(ckpt, "head.safetensors"))
    model.eval()
    return model, tokenizer, cfg


@torch.inference_mode()
def answer(model, tokenizer, r: dict, max_length: int, device: torch.device) -> Dict[str, Any]:
    q = gate_question(r)
    enc = encode_record(tokenizer, {"state": r["state"], "questions": {"q": q}}, max_length=max_length)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(collate_records([enc], tokenizer.pad_token_id, device))[0][0]
    probs = dict(zip(enc.questions[0].option_ids, logits.float().softmax(-1).tolist()))
    return systemone_answer(q, probs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True, help="gate_cache key, e.g. von-2-nano-r1")
    ap.add_argument("--suites", default="judge_heldout,probes,jabr_v2")
    ap.add_argument("--max_length", type=int, default=0, help="0 = checkpoint's training max_length")
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "benchmarks/data/gate_cache"))
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, cfg = load_checkpoint(a.checkpoint, device)
    max_length = a.max_length or int(cfg.get("max_length", 4096))

    rows: List[dict] = []
    for s in a.suites.split(","):
        rows += (jabr_rows() if s == "jabr_v2" else probe_rows() if s == "probes"
                 else judge_rows() if s == "judge_heldout" else jev_rows(s.split("_", 1)[1]))
    os.makedirs(a.out_dir, exist_ok=True)
    cache = os.path.join(a.out_dir, a.name + ".json")
    done = {r["id"]: r for r in json.load(open(cache))} if os.path.exists(cache) else {}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} rows, {len(todo)} to do, max_length {max_length}, device {device}", flush=True)
    for i, r in enumerate(todo):
        t0 = time.perf_counter()
        try:
            ans = answer(model, tokenizer, r, max_length, device)
        except Exception as e:  # noqa: BLE001
            print(f"  ERR {r['id']}: {str(e)[:160]}", flush=True)
            continue
        if r["type"] == "noul":
            p = float(ans["noul"]); pick = "yes" if p >= 0.5 else "no"; conf = abs(p - 0.5) * 2
        elif r["type"] == "score":
            pick = float(ans["score"]); conf = float(ans.get("confidence") or 0)
        else:
            pick = ans["choice"]; conf = float(ans.get("confidence") or 0)
        done[r["id"]] = {"id": r["id"], "suite": r["suite"], "family": r["family"], "type": r["type"], "pick": pick,
                         "expected": r["expected"], "hit": hit(r, pick), "conf": conf, "seconds": time.perf_counter() - t0}
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            print(f"  {i+1}/{len(todo)}", flush=True)
            json.dump(list(done.values()), open(cache, "w"))
    json.dump(list(done.values()), open(cache, "w"))
    by: Dict[str, List[bool]] = {}
    for r in done.values():
        by.setdefault(r["suite"], []).append(r["hit"])
    for s, h in sorted(by.items()):
        print(f"{s:14s} n={len(h):4d} acc={sum(h)/len(h):.3f}")
    if done:
        secs = sorted(r["seconds"] for r in done.values())
        print(f"pooled n={len(done)} acc={sum(r['hit'] for r in done.values())/len(done):.3f}  median s={secs[len(secs)//2]:.3f}")


if __name__ == "__main__":
    main()
