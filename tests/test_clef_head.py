"""training/clef_head.py: packing invariants and head forward, with a stub tokenizer (no downloads)."""
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.clef_head import (  # noqa: E402
    JointSchemaHead, collate_records, encode_record, infer_row_type, row_to_record, soft_target_for, systemone_answer)


class _Tok:
    """Byte-level stub: one token per character, pad = 0."""
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        class R:  # noqa: D401
            input_ids = [ord(c) % 60000 + 1 for c in text]
        return R()


def test_infer_row_type():
    assert infer_row_type(["yes", "no"]) == "noul"
    assert infer_row_type(["True", "False"]) == "noul"
    assert infer_row_type(["0", "1", "2"]) == "score"
    assert infer_row_type(["2", "0", "1"]) == "score"
    assert infer_row_type(["0", "2"]) == "choice"  # not contiguous: a real option id space
    assert infer_row_type(["a", "b"]) == "choice"


def test_row_to_record_each_type():
    rec, target = row_to_record({"state": "s", "question": "q?", "options": [{"id": "no", "description": "n"}, {"id": "yes", "description": "y"}], "label": "yes"})
    assert rec["questions"]["q"]["type"] == "noul" and target == "true"
    assert rec["questions"]["q"]["criteria"] == {"false": "n", "true": "y"}
    rec, target = row_to_record({"state": "s", "question": "q?", "options": [{"id": "1", "description": "mid"}, {"id": "0", "description": "low"}], "label": "1"})
    assert rec["questions"]["q"]["type"] == "score" and rec["questions"]["q"]["criteria"] == ["low", "mid"] and target == "1"
    rec, target = row_to_record({"state": "s", "question": "q?", "options": [{"id": "b", "description": "B"}, {"id": "a", "description": "A"}], "label": "b"})
    assert rec["questions"]["q"]["type"] == "choice" and target == "b"


def test_spans_point_at_option_text_and_choice_order_is_sorted():
    tok = _Tok()
    rec, target = row_to_record({"state": "the state", "question": "pick", "options": [{"id": "zeta", "description": "Z"}, {"id": "alpha", "description": "A"}], "label": "zeta"})
    enc = encode_record(tok, rec, max_length=4096)
    q = enc.questions[0]
    assert q.option_ids == ("alpha", "zeta")  # sorted by id: order-invariant scoring
    assert q.option_ids.index(target) == 1
    chars = "".join(chr(i - 1) for i in enc.input_ids)
    for (s, e), oid in zip(q.option_spans, q.option_ids):
        assert chars[s:e] == '{"description":"%s","option_id":"%s"}' % ("A" if oid == "alpha" else "Z", oid)
    qs, qe = q.question_span
    assert chars[qs:qe] == "pick"
    assert chars.startswith("<|im_start|>system") and chars.endswith("JOINT SCHEMA DECISIONS:")


def test_state_truncation_only_touches_state():
    tok = _Tok()
    rec, _ = row_to_record({"state": "x" * 5000, "question": "q", "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}], "label": "a"})
    enc = encode_record(tok, rec, max_length=600)
    assert len(enc.input_ids) == 600
    chars = "".join(chr(i - 1) for i in enc.input_ids)
    assert chars.endswith("JOINT SCHEMA DECISIONS:")
    s, e = enc.questions[0].option_spans[0]
    assert chars[s:e].endswith('"option_id":"a"}')


def test_soft_target_reordered_to_packed_option_order():
    row = {"state": "s", "question": "q", "options": [{"id": "zeta", "description": "Z"}, {"id": "alpha", "description": "A"}], "label": "zeta", "target": [3, 1]}
    assert soft_target_for(row, ("alpha", "zeta")) == [0.25, 0.75]
    row = {"state": "s", "question": "q", "options": [{"id": "yes", "description": "y"}, {"id": "no", "description": "n"}], "label": "yes", "target": [0.9, 0.1]}
    assert soft_target_for(row, ("true", "false")) == [0.9, 0.1]


def test_head_forward_shapes_and_systemone_answer():
    torch.manual_seed(0)
    tok = _Tok()
    rows = [
        {"state": "alpha beta", "question": "which", "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}, {"id": "c", "description": "C"}], "label": "a"},
        {"state": "gamma", "question": "ok?", "options": [{"id": "yes", "description": "y"}, {"id": "no", "description": "n"}], "label": "no"},
    ]
    recs = [encode_record(tok, row_to_record(r)[0], max_length=4096) for r in rows]
    batch = collate_records(recs, tok.pad_token_id, torch.device("cpu"))
    hidden_size, vocab = 32, 60001
    head = JointSchemaHead(hidden_size=hidden_size, width=16, routing_layers=1, layers=1, heads=2, feedforward=32)
    hidden = torch.randn(batch["input_ids"].shape + (hidden_size,))
    emb = torch.randn(vocab, hidden_size)
    out = head(hidden, batch["input_ids"], batch["attention_mask"], recs, emb)
    assert len(out) == 2 and out[0][0].shape == (3,) and out[1][0].shape == (2,)
    probs = dict(zip(recs[0].questions[0].option_ids, out[0][0].softmax(-1).tolist()))
    ans = systemone_answer({"type": "choice", "criteria": {"a": "A", "b": "B", "c": "C"}}, probs)
    assert ans["choice"] in ("a", "b", "c") and abs(sum(ans["probabilities"].values()) - 1) < 1e-3
    probs = dict(zip(recs[1].questions[0].option_ids, out[1][0].softmax(-1).tolist()))
    ans = systemone_answer({"type": "noul"}, probs)
    assert 0.0 <= ans["noul"] <= 1.0
