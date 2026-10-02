"""Refit Von's confidence on your own labels, with frozen weights and no GPU.

Von reports confidence through a temperature map fitted on JevBench items:
    T = bias + entropy*H_norm + log_tokens*log10(state_tokens)/4 + n_options*(K/8)
Temperature never changes what Von answers, only how sure it says it is, and
a map fitted on one distribution does not carry to another. This module
replays labelled requests through the backend, captures the raw option logits
and the exact features the serving path reads, and refits the map by negative
log-likelihood on those labels. It writes a `marker_calibration.json` that the
backend prefers over the shipped one when placed in the checkpoint directory.

A scalar temperature (map with only `bias`) is always fitted too; the winner
under k-fold cross-validated NLL is written, so a few dozen labels give you a
safe scalar and a few hundred can earn the feature map.

Label rows are JSON lines with the wire question plus `gold`:

    {"state": "...", "question": {"type": "choice", "instructions": "...",
     "criteria": {"auth": "...", "sandbox": "..."}}, "gold": "auth"}
    {"state": "...", "question": {"type": "noul", "instructions": "...",
     "criteria": {"true": "...", "false": "..."}}, "gold": "yes"}
    {"state": "...", "question": {"type": "score", "instructions": "...",
     "criteria": ["low", "mid", "high"]}, "gold": 2}

Shorthand `{"state", "instructions", "choices": {...}, "gold"}` is accepted
for Choice.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

FEATURES = ("bias", "entropy", "log_tokens", "n_options")
LO, HI = 0.3, 12.0


# ----------------------------------------------------------------------------
# Rows
# ----------------------------------------------------------------------------


def parse_row(raw: dict, line_no: int) -> Tuple[str, dict, Any]:
    """Normalise one label row to (state_text, question_dict, gold)."""
    if "state" not in raw or "gold" not in raw:
        raise ValueError(f"line {line_no}: needs 'state' and 'gold'")
    state = raw["state"]
    state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    q = raw.get("question")
    if q is None:
        if "choices" not in raw:
            raise ValueError(f"line {line_no}: needs 'question' or shorthand 'choices'")
        choices = raw["choices"]
        if isinstance(choices, list):
            choices = {c: c for c in choices}
        q = {"type": "choice", "instructions": raw.get("instructions", ""), "criteria": choices}
    if q.get("type") not in ("choice", "noul", "score"):
        raise ValueError(f"line {line_no}: question.type must be choice|noul|score")
    return state_text, q, raw["gold"]


def gold_index(q: dict, gold: Any, line_no: int) -> int:
    """Index of the gold answer in the option order the backend scores."""
    t = q["type"]
    if t == "choice":
        keys = list(q["criteria"].keys())
        if gold not in keys:
            raise ValueError(f"line {line_no}: gold {gold!r} not in criteria keys {keys}")
        return keys.index(gold)
    if t == "noul":
        g = str(gold).strip().lower()
        if g in ("yes", "true", "1"):
            return 0
        if g in ("no", "false", "0"):
            return 1
        raise ValueError(f"line {line_no}: noul gold must be yes/no, got {gold!r}")
    k = len(q["criteria"])
    try:
        i = int(gold)
    except (TypeError, ValueError):
        raise ValueError(f"line {line_no}: score gold must be a level index, got {gold!r}")
    if not 0 <= i < k:
        raise ValueError(f"line {line_no}: score gold {i} outside 0..{k - 1}")
    return i


def load_rows(path: str) -> List[Tuple[str, dict, int]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            state_text, q, gold = parse_row(json.loads(line), n)
            out.append((state_text, q, gold_index(q, gold, n)))
    if not out:
        raise ValueError(f"{path}: no rows")
    return out


# ----------------------------------------------------------------------------
# Collection
# ----------------------------------------------------------------------------


def collect(rows: Sequence[Tuple[str, dict, int]], checkpoint: str, device: str, log=print) -> List[dict]:
    """Replay rows through the backend; return [{logits, feats, gold, type}]."""
    # Chains answer through their own matcher and bypass the temperature;
    # the raw Noul posterior is what the map sees, not the band rule.
    os.environ["VON_CHAINS_DIR"] = "off"
    os.environ["VON_NOUL_DECISION"] = "raw"
    from von.backends.option_marker_backend import OptionMarkerBackend
    from von.types import Choice, Noul, Score

    if not os.path.exists(os.path.join(checkpoint, "marker_calibration.json")) and \
            os.path.exists(os.path.join(checkpoint, "option_marker.pt")):
        # The shipped file also carries `independent_options`, a forward-pass
        # flag; without it the backend scores in a different mode and the
        # fitted map would not match what `von serve` reports.
        raise ValueError(f"{checkpoint} has option_marker.pt but no marker_calibration.json; "
                         "restore the shipped file first (it sets independent_options), then refit")
    backend = OptionMarkerBackend(checkpoint_dir=checkpoint, device=device)
    backend._get_model()
    records: List[dict] = []
    t0 = time.perf_counter()
    for i, (state_text, q, gold) in enumerate(rows):
        sink: List[dict] = []
        backend._capture_local.sink = sink
        try:
            if q["type"] == "choice":
                backend.evaluate_choice("q", state_text, Choice(**q))
            elif q["type"] == "noul":
                backend.evaluate_noul("q", state_text, Noul(**q))
            else:
                backend.evaluate_score("q", state_text, Score(**q))
        finally:
            backend._capture_local.sink = None
        if not sink:
            raise RuntimeError("backend produced no logits for a row; is chains off and the model loaded?")
        rec = sink[-1]
        records.append({"logits": rec["logits"], "feats": rec["feats"], "gold": gold, "type": q["type"]})
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            log(f"  {i + 1}/{len(rows)} rows, {time.perf_counter() - t0:.0f}s")
    return records


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------


def _temp(params: Dict[str, float], feats: Dict[str, float]) -> float:
    raw = sum(params.get(k, 0.0) * feats.get(k, 0.0) for k in FEATURES)
    return min(HI, max(LO, raw))


def _log_softmax(logits: Sequence[float], t: float) -> List[float]:
    z = [x / t for x in logits]
    m = max(z)
    lse = m + math.log(sum(math.exp(v - m) for v in z))
    return [v - lse for v in z]


def nll(params: Dict[str, float], records: Sequence[dict]) -> float:
    total = 0.0
    for r in records:
        lp = _log_softmax(r["logits"], _temp(params, r["feats"]))
        total -= lp[r["gold"]]
    return total / max(len(records), 1)


def ece(params: Optional[Dict[str, float]], records: Sequence[dict], bins: int = 10) -> float:
    """Top-label expected calibration error, equal-width bins."""
    buckets: Dict[int, List[Tuple[float, bool]]] = {}
    for r in records:
        t = _temp(params, r["feats"]) if params else 1.0
        lp = _log_softmax(r["logits"], t)
        top = max(range(len(lp)), key=lambda i: lp[i])
        conf = math.exp(lp[top])
        buckets.setdefault(min(int(conf * bins), bins - 1), []).append((conf, top == r["gold"]))
    n = len(records)
    out = 0.0
    for b in buckets.values():
        out += len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(h for _, h in b) / len(b))
    return out


def _fit(records: Sequence[dict], keys: Sequence[str], seed: int = 0, restarts: int = 4, iters: int = 400) -> Dict[str, float]:
    """Minimise NLL over the map's `keys` with torch (Adam + LBFGS polish)."""
    import torch

    X = torch.tensor([[r["feats"][k] for k in keys] for r in records], dtype=torch.float64)
    L = [torch.tensor(r["logits"], dtype=torch.float64) for r in records]
    gold = [r["gold"] for r in records]

    def loss_fn(w: "torch.Tensor") -> "torch.Tensor":
        t = (X @ w).clamp(LO, HI)
        acc = torch.zeros((), dtype=torch.float64)
        for i, lg in enumerate(L):
            acc = acc - torch.log_softmax(lg / t[i], dim=-1)[gold[i]]
        return acc / len(L)

    best_w, best_loss = None, float("inf")
    rng = random.Random(seed)
    for _ in range(restarts):
        w = torch.zeros(len(keys), dtype=torch.float64)
        w[0] = rng.uniform(0.8, 3.0)  # bias
        for j in range(1, len(keys)):
            w[j] = rng.uniform(-1.0, 1.0)
        w.requires_grad_(True)
        opt = torch.optim.Adam([w], lr=0.05)
        for _ in range(iters):
            opt.zero_grad()
            loss = loss_fn(w)
            loss.backward()
            opt.step()
        lb = torch.optim.LBFGS([w], max_iter=50, line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad()
            loss = loss_fn(w)
            loss.backward()
            return loss

        try:
            lb.step(closure)
        except RuntimeError:
            pass
        final = float(loss_fn(w).detach())
        if final < best_loss:
            best_loss, best_w = final, w.detach().clone()
    assert best_w is not None  # loop always runs at least once
    return {k: float(v) for k, v in zip(keys, best_w)}


def cross_validate(records: Sequence[dict], keys: Sequence[str], folds: int, seed: int = 0) -> float:
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    folds = max(2, min(folds, len(records)))
    total = 0.0
    for f in range(folds):
        test = set(idx[f::folds])
        train = [records[i] for i in idx if i not in test]
        held = [records[i] for i in sorted(test)]
        params = _fit(train, keys, seed=seed)
        total += nll(params, held) * len(held)
    return total / len(records)


def fit_map(records: Sequence[dict], folds: int = 5, seed: int = 0, log=print,
            shipped_map: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Fit scalar and feature maps; return the CV winner plus a report."""
    scalar_cv = cross_validate(records, ("bias",), folds, seed)
    map_cv = cross_validate(records, FEATURES, folds, seed)
    scalar = _fit(records, ("bias",), seed=seed)
    full = _fit(records, FEATURES, seed=seed)
    winner_name = "map" if map_cv < scalar_cv else "scalar"
    winner = dict(full if winner_name == "map" else scalar)
    for k in FEATURES:
        winner.setdefault(k, 0.0)
    winner["lo"], winner["hi"] = LO, HI
    report = {
        "n": len(records),
        "folds": folds,
        "cv_nll": {"scalar": round(scalar_cv, 4), "map": round(map_cv, 4)},
        "in_sample_nll": {"raw_T1": round(nll({"bias": 1.0}, records), 4),
                          "scalar": round(nll(scalar, records), 4), "map": round(nll(full, records), 4)},
        "ece": {"raw_T1": round(ece(None, records), 4), "scalar": round(ece(scalar, records), 4),
                "map": round(ece(full, records), 4)},
        "scalar_temperature": round(scalar["bias"], 4),
        "winner": winner_name,
    }
    if shipped_map:
        report["in_sample_nll"]["shipped"] = round(nll(shipped_map, records), 4)
        report["ece"]["shipped"] = round(ece(shipped_map, records), 4)
    log(f"  scalar T={scalar['bias']:.3f}  CV-NLL scalar={scalar_cv:.4f} map={map_cv:.4f}  -> {winner_name}")
    return {"calibration_map": winner, "report": report}


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------


def write_calibration(out_path: str, checkpoint: str, fitted: Dict[str, Any], labels_path: str, shipped: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write marker_calibration.json, carrying over the shipped file's other fields."""
    base: Dict[str, Any] = {}
    src = shipped if shipped is not None else _shipped_calibration(checkpoint)
    if src:
        base = {k: v for k, v in src.items() if not k.startswith("calibration_")}
    base["calibration_map"] = {k: round(v, 4) for k, v in fitted["calibration_map"].items()}
    base["calibration_map_features"] = ("T = bias + entropy*H_norm + log_tokens*log10(state_tokens)/4 + "
                                        "n_options*(K/8), clamped to [lo,hi]")
    base["calibration_fitted_on"] = (f"von calibrate: {fitted['report']['n']} labelled rows from "
                                     f"{os.path.basename(labels_path)}; objective NLL; "
                                     f"{fitted['report']['folds']}-fold CV chose {fitted['report']['winner']}")
    base["calibration_report"] = fitted["report"]
    base["calibration_timestamp"] = time.time()
    # model_id in the shipped file names the weights (von-1.2.0, unchanged since);
    # record the engine that fitted this map separately so the two stop being confused.
    from von import __version__ as _engine_version
    base["calibrated_with_engine"] = f"von-sdk {_engine_version}"
    base.setdefault("weights_id", base.get("model_id"))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if os.path.islink(out_path):
        # Writing through a link would silently overwrite the shipped file it
        # points at; replace the link with a real file instead.
        os.unlink(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=2)
    return base


def _shipped_calibration(checkpoint: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(checkpoint, "marker_calibration.json")
    if not os.path.exists(p):
        # Weights served from the Hub: the shipped file lives in the HF cache.
        try:
            from huggingface_hub import hf_hub_download

            from von.backends.option_marker_backend import VON_HF_REPO

            p = hf_hub_download(repo_id=VON_HF_REPO, filename="marker_calibration.json")
        except Exception:
            return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def run(labels: str, out: str, checkpoint: str, device: str = "cpu", folds: int = 5, seed: int = 0, log=print) -> Dict[str, Any]:
    rows = load_rows(labels)
    log(f"{len(rows)} labelled rows; replaying through {checkpoint} on {device}")
    records = collect(rows, checkpoint, device, log=log)
    shipped = _shipped_calibration(checkpoint)
    shipped_map = (shipped or {}).get("calibration_map") if shipped else None
    fitted = fit_map(records, folds=folds, seed=seed, log=log,
                     shipped_map=shipped_map if isinstance(shipped_map, dict) else None)
    written = write_calibration(out, checkpoint, fitted, labels, shipped=shipped)
    log(f"wrote {out}")
    return written
