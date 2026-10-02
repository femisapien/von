"""Score any Jev-wire HTTP server on the gate suites; writes a gate_cache-compatible dump
(id, suite, family, type, pick, expected, hit, conf, seconds) so gate_standard.py /
sweep_threshold.py --fallback can compare it against Von."""
from __future__ import annotations
import argparse, json, os, sys, time
import urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
from gate_standard import hit, jabr_rows, jev_rows, judge_rows, probe_rows  # noqa: E402

def ask(url: str, model: str, r: dict) -> dict:
    q = {"type": r["type"], "instructions": r["instructions"]}
    if r["type"] == "score":
        q["criteria"] = [r["criteria"][k] for k in sorted(r["criteria"], key=int)]
    elif r["criteria"]:
        q["criteria"] = r["criteria"]
    body = json.dumps({"model": model, "state": r["state"], "questions": {"q": q}}).encode()
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)["answers"]["q"]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765/v1/systemone")
    ap.add_argument("--model", default="jeff-latest")
    ap.add_argument("--name", required=True, help="cache key, e.g. jeff-0.8b-v1.2")
    ap.add_argument("--suites", default="judge_heldout,probes,jabr_v2")
    a = ap.parse_args()
    rows = []
    for s in a.suites.split(","):
        rows += (jabr_rows() if s == "jabr_v2" else probe_rows() if s == "probes"
                 else judge_rows() if s == "judge_heldout" else jev_rows(s.split("_", 1)[1]))
    cache = os.path.join(ROOT, "benchmarks/data/gate_cache", a.name + ".json")
    done = {r["id"]: r for r in json.load(open(cache))} if os.path.exists(cache) else {}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} rows, {len(todo)} to do")
    for i, r in enumerate(todo):
        t0 = time.perf_counter()
        try:
            ans = ask(a.url, a.model, r)
        except Exception as e:  # noqa: BLE001
            print(f"  ERR {r['id']}: {str(e)[:120]}"); continue
        if r["type"] == "noul":
            p = float(ans["noul"]); pick = "yes" if p >= 0.5 else "no"; conf = abs(p - 0.5) * 2
        elif r["type"] == "score":
            pick = float(ans["score"]); conf = float(ans.get("confidence") or 0)
        else:
            pick = ans["choice"]; conf = float(ans.get("confidence") or 0)
        done[r["id"]] = {"id": r["id"], "suite": r["suite"], "family": r["family"], "type": r["type"], "pick": pick,
                         "expected": r["expected"], "hit": hit(r, pick), "conf": conf, "seconds": time.perf_counter() - t0}
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"  {i+1}/{len(todo)}", flush=True); json.dump(list(done.values()), open(cache, "w"))
    json.dump(list(done.values()), open(cache, "w"))
    by = {}
    for r in done.values():
        by.setdefault(r["suite"], []).append(r["hit"])
    for s, h in sorted(by.items()):
        print(f"{s:14s} n={len(h):4d} acc={sum(h)/len(h):.3f}")
    print(f"pooled n={len(done)} acc={sum(r['hit'] for r in done.values())/len(done):.3f}  median s={sorted(r['seconds'] for r in done.values())[len(done)//2]:.2f}")

if __name__ == "__main__":
    main()
