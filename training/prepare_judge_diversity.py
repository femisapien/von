"""Judge-shaped and label-diverse public data for Von's option-marker corpus.

Why this exists (JevBench v1.5.1, sealed Choice by tier): Von easy 50.2, standard
30.7, judge 7.2, hard 11.3. Judge is the worst in the encoder class (GLiNER2 20.5,
jeff 17.8, Verdict 16.5) and the only tier where Von's open score (36.7) collapses
on the sealed half. The 290k universal corpus contains no response-quality or
pairwise-preference data at all, and only two multi-class label spaces
(Banking77, Emotion); the peers that hold up on sealed all sit on broad
label-conditioned pretraining.

Two blocks, both from permissively licensed Hub datasets (licence recorded per row):

  judge      HelpSteer2 (Score 0-4 per attribute; pairwise helpfulness Choice),
             HelpSteer3 (pairwise, soft target from preference strength),
             MT-Bench human judgments, RewardBench, UltraFeedback (soft target
             from rating gap), Anthropic HH-RLHF.
  diversity  CLINC150 (+oos -> other), MASSIVE intents/scenarios, GoEmotions,
             DBpedia-14, LEDGAR (100 clause types), CaseHOLD (5 described
             holdings), Twitter financial topics, Bitext support intents,
             Civil Comments toxicity as Noul.

Output rows use the trainer's format: {state, question, options:[{id,description}],
label, source, licence[, target]}. Every row is a described-option Choice.

    python training/prepare_judge_diversity.py --out data_judge/train.jsonl \
        --judge_per_source 8000 --diversity_per_source 5000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from datasets import load_dataset

MAX_STATE_CHARS = 12000


def _clip(s: str, n: int = MAX_STATE_CHARS) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: int(n * 0.6)] + " ... " + s[-int(n * 0.4):]


def _humanise(label: str) -> str:
    return re.sub(r"[_\-/]+", " ", str(label)).strip().lower()


def _row(state: str, question: str, options: List[dict], label: str, source: str, licence: str,
         target: Optional[List[float]] = None) -> dict:
    r = {"state": _clip(state), "question": question, "options": options, "label": label,
         "source": source, "licence": licence}
    if target is not None:
        r["target"] = target
    return r


def _take(it: Iterable, n: int, seed: int) -> List:
    rows = list(it)
    random.Random(seed).shuffle(rows)
    return rows[:n]


# ----------------------------------------------------------------------------- judge

def _pair_rows(a: str, b: str, question: str, prefer_a: bool, source: str, licence: str,
               p_a: Optional[float] = None, context: str = "") -> dict:
    """Two responses as described options, order randomised, optional soft target."""
    first_is_a = random.random() < 0.5
    ra, rb = ("A", "B") if first_is_a else ("B", "A")
    opts = [{"id": ra, "description": _clip(a, 4000)}, {"id": rb, "description": _clip(b, 4000)}]
    opts.sort(key=lambda o: o["id"])
    label = ra if prefer_a else rb
    target = None
    if p_a is not None:
        target = [p_a if o["id"] == ra else 1.0 - p_a for o in opts]
    return _row(context, question, opts, label, source, licence, target)


HS2_ATTRS = {
    "helpfulness": ("How helpful is the response to the user's request?",
                    ["Not helpful at all; misses or ignores the request.",
                     "Slightly helpful; mostly misses the point.",
                     "Partially helpful; addresses some of the request.",
                     "Mostly helpful; addresses the request with minor gaps.",
                     "Fully helpful; completely and directly addresses the request."]),
    "correctness": ("How factually correct and complete is the response?",
                    ["Mostly incorrect or misleading.",
                     "Several errors or major omissions.",
                     "Some errors or notable omissions.",
                     "Minor inaccuracies or small omissions.",
                     "Accurate and complete; no errors found."]),
    "coherence": ("How clear and internally consistent is the response?",
                  ["Incoherent; contradictory or unreadable.",
                   "Hard to follow; frequent inconsistencies.",
                   "Somewhat clear; some confusing passages.",
                   "Clear with minor lapses.",
                   "Perfectly clear, consistent and well organised."]),
    "verbosity": ("How verbose is the response relative to what the request needed?",
                  ["Far too short; missing needed content.",
                   "Somewhat too short.",
                   "About the right length.",
                   "Somewhat too long; some padding.",
                   "Far too long; heavily padded or repetitive."]),
}


def judge_helpsteer2(n: int, seed: int) -> List[dict]:
    lic = "cc-by-4.0"
    ds = load_dataset("nvidia/HelpSteer2", split="train")
    rows: List[dict] = []
    per_attr = max(1, n // (len(HS2_ATTRS) + 1))
    by_prompt: Dict[str, List[dict]] = {}
    for r in ds:
        by_prompt.setdefault(r["prompt"], []).append(r)
    items = list(ds)
    rnd = random.Random(seed)
    rnd.shuffle(items)
    for attr, (q, levels) in HS2_ATTRS.items():
        k = 0
        for r in items:
            if k >= per_attr:
                break
            v = r.get(attr)
            if v is None:
                continue
            state = f"USER REQUEST:\n{r['prompt']}\n\nRESPONSE:\n{r['response']}"
            opts = [{"id": str(i), "description": d} for i, d in enumerate(levels)]
            rows.append(_row(state, q, opts, str(int(v)), f"helpsteer2/{attr}", lic))
            k += 1
    k = 0
    for prompt, group in by_prompt.items():
        if k >= per_attr or len(group) < 2:
            continue
        a, b = group[0], group[1]
        if a["helpfulness"] == b["helpfulness"]:
            continue
        gap = abs(a["helpfulness"] - b["helpfulness"])
        p = {1: 0.7, 2: 0.85, 3: 0.93, 4: 0.97}[min(gap, 4)]
        a_wins = a["helpfulness"] > b["helpfulness"]
        rows.append(_pair_rows(a["response"], b["response"],
                               "Which response is more helpful for this request?",
                               a_wins, "helpsteer2/pairwise", lic,
                               p_a=p if a_wins else 1.0 - p, context=f"USER REQUEST:\n{prompt}"))
        k += 1
    return rows


def judge_helpsteer3(n: int, seed: int) -> List[dict]:
    lic = "cc-by-4.0"
    ds = load_dataset("nvidia/HelpSteer3", "preference", split="train")
    rows = []
    for r in _take((r for r in ds if r.get("language") == "english" and r.get("overall_preference", 0) != 0), n, seed):
        ctx = r["context"]
        if isinstance(ctx, list):
            ctx = "\n".join(f"{t.get('role','user').upper()}: {t.get('content','')}" for t in ctx)
        pref = int(r["overall_preference"])  # <0 prefers response1, >0 prefers response2
        p1 = {3: 0.95, 2: 0.85, 1: 0.7}[abs(pref)]
        rows.append(_pair_rows(r["response1"], r["response2"],
                               "Which response is the better continuation of this conversation?",
                               prefer_a=pref < 0, source="helpsteer3", licence=lic,
                               p_a=p1 if pref < 0 else 1.0 - p1, context=f"CONVERSATION:\n{ctx}"))
    return rows


def _conv_text(conv) -> str:
    if isinstance(conv, list):
        return "\n".join(f"{t.get('role','user').upper()}: {t.get('content','')}" for t in conv)
    return str(conv)


def judge_mtbench(n: int, seed: int) -> List[dict]:
    lic = "cc-by-4.0"
    ds = load_dataset("lmsys/mt_bench_human_judgments", split="human")
    rows = []
    for r in _take((r for r in ds if r.get("winner") in ("model_a", "model_b")), n, seed):
        ca, cb = r["conversation_a"], r["conversation_b"]
        q = ca[0]["content"] if isinstance(ca, list) and ca else ""
        ans_a = "\n".join(t["content"] for t in ca if t.get("role") == "assistant") if isinstance(ca, list) else str(ca)
        ans_b = "\n".join(t["content"] for t in cb if t.get("role") == "assistant") if isinstance(cb, list) else str(cb)
        rows.append(_pair_rows(ans_a, ans_b, "Which assistant answered the user's question better?",
                               r["winner"] == "model_a", "mt_bench_human", lic,
                               context=f"USER QUESTION:\n{q}"))
    return rows


def judge_rewardbench(n: int, seed: int) -> List[dict]:
    lic = "odc-by"
    ds = load_dataset("allenai/reward-bench", split="filtered")
    rows = []
    for r in _take(ds, n, seed):
        rows.append(_pair_rows(r["chosen"], r["rejected"],
                               "Which response is the better answer to this prompt (more correct, helpful and safe)?",
                               True, f"reward_bench/{r.get('subset','')}", lic, context=f"PROMPT:\n{r['prompt']}"))
    return rows


def judge_ultrafeedback(n: int, seed: int) -> List[dict]:
    lic = "mit"
    ds = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train")
    rows = []
    for r in _take(ds, n, seed):
        ch = r["chosen"][-1]["content"] if isinstance(r["chosen"], list) else str(r["chosen"])
        rj = r["rejected"][-1]["content"] if isinstance(r["rejected"], list) else str(r["rejected"])
        gap = float(r.get("chosen-rating", 5) or 5) - float(r.get("rejected-rating", 1) or 1)
        p = min(0.97, 0.55 + 0.1 * max(gap, 0.5))
        rows.append(_pair_rows(ch, rj, "Which response better follows the instruction and is more accurate?",
                               True, "ultrafeedback", lic, p_a=p, context=f"INSTRUCTION:\n{r['prompt']}"))
    return rows


def judge_hh(n: int, seed: int) -> List[dict]:
    lic = "mit"
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    rows = []
    for r in _take(ds, n * 2, seed):
        ch, rj = r["chosen"], r["rejected"]
        i, j = ch.rfind("\n\nAssistant:"), rj.rfind("\n\nAssistant:")
        if i < 0 or j < 0 or ch[:i] != rj[:j]:
            continue
        rows.append(_pair_rows(ch[i + 13:], rj[j + 13:],
                               "Which final assistant reply is more helpful and harmless in this conversation?",
                               True, "hh_rlhf", lic, context=f"CONVERSATION:{ch[:i]}"))
        if len(rows) >= n:
            break
    return rows


# ------------------------------------------------------------------------- diversity

def _label_space_rows(items: Sequence[tuple], label_desc: Dict[str, str], questions: Sequence[str],
                      source: str, licence: str, n: int, seed: int, other_rate: float = 0.15,
                      other_label: Optional[str] = None) -> List[dict]:
    """items: (text, gold). Samples K in 3..6 described options around gold; `other_rate`
    of rows drop gold and target a catch-all. Rows whose gold is `other_label` always target it."""
    rnd = random.Random(seed)
    labels = list(label_desc)
    rows = []
    for text, gold in _take(items, n, seed):
        k = rnd.randint(3, 6)
        gold_is_other = other_label is not None and gold == other_label
        if gold_is_other or rnd.random() < other_rate:
            pool = rnd.sample([l for l in labels if l != gold], min(k, len(labels) - 1))
            opts = [{"id": l, "description": label_desc[l]} for l in pool]
            opts.append({"id": "other", "description": "None of the listed categories fits."})
            target = "other"
        else:
            pool = [gold] + rnd.sample([l for l in labels if l != gold], min(k - 1, len(labels) - 1))
            rnd.shuffle(pool)
            opts = [{"id": l, "description": label_desc[l]} for l in pool]
            target = gold
        rows.append(_row(text, rnd.choice(questions), opts, target, source, licence))
    return rows


def div_clinc(n: int, seed: int) -> List[dict]:
    ds = load_dataset("clinc/clinc_oos", "plus", split="train")
    names = ds.features["intent"].names
    desc = {nm: f"The user's intent is: {_humanise(nm)}." for nm in names if nm != "oos"}
    items = [(r["text"], names[r["intent"]]) for r in ds]
    return _label_space_rows(items, desc, ["Which intent does this user utterance express?",
                                            "What is the user asking the assistant to do?"],
                             "clinc150", "cc-by-3.0", n, seed, other_label="oos")


def div_massive(n: int, seed: int) -> List[dict]:
    ds = load_dataset("AmazonScience/massive", "en-US", split="train")
    inames = ds.features["intent"].names
    snames = ds.features["scenario"].names
    idesc = {nm: f"Intent: {_humanise(nm)}." for nm in inames}
    sdesc = {nm: f"Domain: {_humanise(nm)}." for nm in snames}
    half = n // 2
    rows = _label_space_rows([(r["utt"], inames[r["intent"]]) for r in ds], idesc,
                             ["Which intent does this voice-assistant request express?"],
                             "massive/intent", "cc-by-4.0", half, seed)
    rows += _label_space_rows([(r["utt"], snames[r["scenario"]]) for r in ds], sdesc,
                              ["Which domain does this request belong to?"],
                              "massive/scenario", "cc-by-4.0", n - half, seed + 1)
    return rows


GO_EMOTIONS = {
    "admiration": "Admiration: finding something impressive or worthy of respect.",
    "amusement": "Amusement: finding something funny or entertaining.",
    "anger": "Anger: strong displeasure or hostility.",
    "annoyance": "Annoyance: mild irritation.",
    "approval": "Approval: agreeing with or endorsing something.",
    "caring": "Caring: concern or kindness toward someone.",
    "confusion": "Confusion: not understanding, asking what is going on.",
    "curiosity": "Curiosity: wanting to know or learn more.",
    "desire": "Desire: wanting something to happen or to have something.",
    "disappointment": "Disappointment: sadness that expectations were not met.",
    "disapproval": "Disapproval: disagreeing with or rejecting something.",
    "disgust": "Disgust: revulsion or strong distaste.",
    "embarrassment": "Embarrassment: feeling awkward or ashamed.",
    "excitement": "Excitement: eager enthusiasm.",
    "fear": "Fear: being afraid or worried about a threat.",
    "gratitude": "Gratitude: thanking or feeling thankful.",
    "grief": "Grief: sorrow over a loss.",
    "joy": "Joy: happiness or delight.",
    "love": "Love: affection or deep fondness.",
    "nervousness": "Nervousness: anxiety or apprehension.",
    "optimism": "Optimism: hopefulness about the future.",
    "pride": "Pride: satisfaction in one's own or others' achievements.",
    "realization": "Realization: suddenly understanding or becoming aware.",
    "relief": "Relief: reassurance after worry or stress ends.",
    "remorse": "Remorse: regret or guilt about something done.",
    "sadness": "Sadness: unhappiness or sorrow.",
    "surprise": "Surprise: being startled by something unexpected.",
    "neutral": "Neutral: no particular emotion expressed.",
}


def div_goemotions(n: int, seed: int) -> List[dict]:
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")
    names = ds.features["labels"].feature.names
    items = [(r["text"], names[r["labels"][0]]) for r in ds if len(r["labels"]) == 1]
    return _label_space_rows(items, GO_EMOTIONS, ["Which emotion does this comment primarily express?"],
                             "go_emotions", "apache-2.0", n, seed)


DBPEDIA = {
    "Company": "A business or commercial organisation.", "EducationalInstitution": "A school, college or university.",
    "Artist": "A musician, painter, actor or other creative person.", "Athlete": "A sportsperson.",
    "OfficeHolder": "A politician or holder of public office.", "MeanOfTransportation": "A vehicle, ship, aircraft or train model.",
    "Building": "A building or man-made structure.", "NaturalPlace": "A natural geographic feature such as a river, mountain or lake.",
    "Village": "A village or small settlement.", "Animal": "An animal species.", "Plant": "A plant species.",
    "Album": "A music album.", "Film": "A film or movie.", "WrittenWork": "A book, novel, journal or other written work.",
}


def div_dbpedia(n: int, seed: int) -> List[dict]:
    ds = load_dataset("fancyzhx/dbpedia_14", split="train")
    names = ds.features["label"].names
    items = [(r["content"], names[r["label"]]) for r in _take(ds, n * 2, seed)]
    return _label_space_rows(items, DBPEDIA, ["What kind of entity does this encyclopedia text describe?"],
                             "dbpedia_14", "cc-by-sa-3.0", n, seed)


def div_ledgar(n: int, seed: int) -> List[dict]:
    ds = load_dataset("coastalcph/lex_glue", "ledgar", split="train")
    names = ds.features["label"].names
    desc = {nm: f"Contract clause type: {_humanise(nm)}." for nm in names}
    items = [(r["text"], names[r["label"]]) for r in _take(ds, n * 2, seed)]
    return _label_space_rows(items, desc, ["Which type of contract provision is this clause?"],
                             "ledgar", "cc-by-4.0", n, seed)


def div_casehold(n: int, seed: int) -> List[dict]:
    ds = load_dataset("coastalcph/lex_glue", "case_hold", split="train")
    rows = []
    for r in _take(ds, n, seed):
        opts = [{"id": str(i), "description": e} for i, e in enumerate(r["endings"])]
        rows.append(_row(r["context"], "Which holding statement correctly completes the citing text?",
                         opts, str(r["label"]), "case_hold", "cc-by-4.0"))
    return rows


def div_fin_topics(n: int, seed: int) -> List[dict]:
    ds = load_dataset("zeroshot/twitter-financial-news-topic", split="train")
    names = ["Analyst Update", "Fed | Central Banks", "Company | Product News", "Treasuries | Corporate Debt",
             "Dividend", "Earnings", "Energy | Oil", "Financials", "Currencies", "General News | Opinion",
             "Gold | Metals | Materials", "IPO", "Legal | Regulation", "M&A | Investments", "Macro",
             "Markets", "Politics", "Personnel Change", "Stock Commentary", "Stock Movement"]
    desc = {nm: f"Financial news topic: {nm.replace('|', 'or').lower()}." for nm in names}
    items = [(r["text"], names[r["label"]]) for r in ds]
    return _label_space_rows(items, desc, ["Which financial news topic does this headline belong to?"],
                             "twitter_fin_topic", "mit", n, seed)


def div_bitext(n: int, seed: int) -> List[dict]:
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    intents = sorted({r["intent"] for r in ds})
    desc = {nm: f"Customer support intent: {_humanise(nm)}." for nm in intents}
    items = [(r["instruction"], r["intent"]) for r in ds]
    return _label_space_rows(items, desc, ["Which support intent does this customer message express?"],
                             "bitext_support", "cdla-sharing-1.0", n, seed)


def div_civil_noul(n: int, seed: int) -> List[dict]:
    ds = load_dataset("google/civil_comments", split="train", streaming=True)
    pos, neg = [], []
    for r in ds:
        t = r["toxicity"]
        if t >= 0.6 and len(pos) < n // 2:
            pos.append(r["text"])
        elif t <= 0.1 and len(neg) < n // 2 and random.random() < 0.05:
            neg.append(r["text"])
        if len(pos) >= n // 2 and len(neg) >= n // 2:
            break
    opts = [{"id": "yes", "description": "Toxic: rude, disrespectful, hateful or likely to make someone leave the discussion."},
            {"id": "no", "description": "Not toxic: civil, even if critical or disagreeing."}]
    rows = [_row(t, "Is this comment toxic?", opts, "yes", "civil_comments", "cc0-1.0") for t in pos]
    rows += [_row(t, "Is this comment toxic?", opts, "no", "civil_comments", "cc0-1.0") for t in neg]
    return rows


JUDGE: Dict[str, Callable[[int, int], List[dict]]] = {
    "helpsteer2": judge_helpsteer2, "helpsteer3": judge_helpsteer3, "mt_bench": judge_mtbench,
    "reward_bench": judge_rewardbench, "ultrafeedback": judge_ultrafeedback, "hh_rlhf": judge_hh,
}
DIVERSITY: Dict[str, Callable[[int, int], List[dict]]] = {
    "clinc150": div_clinc, "go_emotions": div_goemotions, "dbpedia_14": div_dbpedia,
    "ledgar": div_ledgar, "case_hold": div_casehold, "twitter_fin_topic": div_fin_topics,
    "bitext_support": div_bitext, "civil_comments": div_civil_noul,
}


def build(judge_per_source: int, diversity_per_source: int, seed: int, only: Optional[Sequence[str]] = None) -> List[dict]:
    random.seed(seed)
    rows: List[dict] = []
    for i, (name, fn) in enumerate({**JUDGE, **DIVERSITY}.items()):
        if only and name not in only:
            continue
        n = judge_per_source if name in JUDGE else diversity_per_source
        try:
            got = fn(n, seed + i)
        except Exception as e:  # one dataset outage must not sink the build
            print(f"  !! {name}: {type(e).__name__}: {e}")
            continue
        for r in got:
            ids = [o["id"] for o in r["options"]]
            assert r["label"] in ids, (name, r["label"], ids)
            if "target" in r:
                assert ids[max(range(len(ids)), key=lambda k: r["target"][k])] == r["label"], (name, r["target"], r["label"])
        print(f"  {name:20} {len(got):6d} rows")
        rows.extend(got)
    random.Random(seed).shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="data_judge/train.jsonl")
    ap.add_argument("--judge_per_source", type=int, default=8000)
    ap.add_argument("--diversity_per_source", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--only", default="", help="comma-separated source names to build (default all)")
    a = ap.parse_args()
    rows = build(a.judge_per_source, a.diversity_per_source, a.seed, [s for s in a.only.split(",") if s] or None)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_src: Dict[str, int] = {}
    for r in rows:
        by_src[r["source"].split("/")[0]] = by_src.get(r["source"].split("/")[0], 0) + 1
    print(json.dumps({"rows": len(rows), "by_source": by_src,
                      "licences": sorted({r["licence"] for r in rows})}, indent=2))


if __name__ == "__main__":
    main()
