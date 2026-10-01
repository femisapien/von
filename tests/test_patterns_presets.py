import pytest
import von
from von.presets import triage_preset, email_preset, moderation_preset, security_preset
from von.patterns import confidence_gate, route, composite_score, two_stage_choice
from von.types import Choice, Noul, Score


def test_presets_structure():
    triage = triage_preset()
    assert "intent" in triage
    assert "is_urgent" in triage
    assert "frustration" in triage
    assert "churn_risk" in triage
    assert isinstance(triage["intent"], Choice)
    assert isinstance(triage["is_urgent"], Noul)
    assert isinstance(triage["frustration"], Score)

    email = email_preset()
    assert "destination" in email
    assert "is_spam_or_phishing" in email
    assert "priority" in email

    mod = moderation_preset()
    assert "policy_violation" in mod
    assert "should_block" in mod

    sec = security_preset()
    assert "event_type" in sec
    assert "is_threat" in sec


def test_preset_nouls_carry_explicit_criteria():
    # Without explicit criteria the backend falls back to zero-shot debiasing,
    # which is what the presets silently did while passing pos/neg_criteria.
    for preset in (triage_preset(), email_preset(), moderation_preset(), security_preset()):
        for q_id, q in preset.items():
            if isinstance(q, Noul):
                assert q.criteria and q.criteria.get("true") and q.criteria.get("false"), q_id


def test_noul_legacy_criteria_kwargs_are_folded():
    with pytest.warns(DeprecationWarning):
        q = Noul(instructions="Is it down?", pos_criteria="Down", neg_criteria="Up")
    assert q.criteria == {"true": "Down", "false": "Up"}
    assert "pos_criteria" not in q.model_dump()

    with pytest.warns(DeprecationWarning):
        q = Noul.model_validate({"type": "noul", "instructions": "Is it down?", "pos_criteria": "Down"})
    assert q.criteria == {"true": "Down"}


def test_noul_legacy_criteria_conflict_raises():
    with pytest.raises(ValueError):
        Noul(instructions="Is it down?", criteria={"true": "Down"}, pos_criteria="Also down")


def test_patterns_route():
    state = "The customer wants an immediate refund for their unused subscription."
    q = Choice(
        instructions="Route customer request",
        criteria={
            "refund": "Customer asks for refund or payment reversal",
            "support": "Customer asks for technical support",
        }
    )

    dispatched = []

    def handle_refund(ans):
        dispatched.append("refund_handled")
        return "REFUND_PROCESSED"

    def handle_support(ans):
        dispatched.append("support_handled")
        return "SUPPORT_OPENED"

    res = route(
        state,
        question=q,
        routes={"refund": handle_refund, "support": handle_support},
    )
    assert res == "REFUND_PROCESSED"
    assert dispatched == ["refund_handled"]


def test_patterns_confidence_gate():
    state = "Urgent: database cluster crashed, connection pool completely exhausted."
    questions = {
        "is_outage": Noul(
            instructions="Is there an active database outage?",
            criteria={
                "true": "Database crash, pool exhausted, downtime",
                "false": "Normal operational query, no crash",
            }
        )
    }

    gated = confidence_gate(state, questions, threshold=0.1)
    assert "automatic" in gated
    assert "escalate" in gated
    assert len(gated["automatic"]) + len(gated["escalate"]) == 1


def test_confidence_gate_routes_nouls_by_distance_from_half(monkeypatch):
    from von import patterns
    from von.types import NoulAnswer, SystemOneResponse, Usage

    answers = {
        "sure_yes": NoulAnswer(noul=0.95),
        "sure_no": NoulAnswer(noul=0.05),
        "unsure": NoulAnswer(noul=0.55),
    }
    fake = SystemOneResponse(model="fake", answers=answers, usage=Usage())
    monkeypatch.setattr(patterns, "system_one", lambda state, questions: fake)

    gated = confidence_gate("state", {}, threshold=0.8)
    assert set(gated["automatic"]) == {"sure_yes", "sure_no"}
    assert set(gated["escalate"]) == {"unsure"}


def test_confidence_gate_uses_pre_band_probability(monkeypatch):
    """Under the band rule every committed `noul` lands in [0.8,0.85] or [0.15,0.2],
    so |noul-0.5|*2 is ~0.6-0.7 for every answer and a 0.8 gate escalates all of them.
    The gate must read `noul_raw` when present."""
    from von import patterns
    from von.backends.option_marker_backend import _noul_decide
    from von.types import NoulAnswer, SystemOneResponse, Usage

    def banded(p: float) -> NoulAnswer:
        return NoulAnswer(noul=_noul_decide(p, 0.8, 0.1), noul_raw=p)

    answers = {"sure_yes": banded(0.97), "sure_no": banded(0.03), "unsure": banded(0.55)}
    assert all(0.6 <= abs(a.noul - 0.5) * 2 <= 0.7 for a in answers.values())
    fake = SystemOneResponse(model="fake", answers=answers, usage=Usage())
    monkeypatch.setattr(patterns, "system_one", lambda state, questions: fake)

    gated = confidence_gate("state", {}, threshold=0.8)
    assert set(gated["automatic"]) == {"sure_yes", "sure_no"}
    assert set(gated["escalate"]) == {"unsure"}


def test_patterns_composite_score():
    state = "Catastrophic multi-region outage affecting all enterprise payments and databases."
    questions = {
        "severity": Score(
            instructions="Rate outage severity",
            criteria=["Minor", "Moderate", "Critical emergency"]
        ),
        "blocking": Noul(
            instructions="Is this blocking?",
            criteria={
                "true": "Critical blocking outage",
                "false": "Non-blocking",
            }
        )
    }

    scored = composite_score(state, questions)
    assert "score" in scored
    assert 0.0 <= scored["score"] <= 1.0
    assert "breakdown" in scored
    assert "severity" in scored["breakdown"]
    assert "blocking" in scored["breakdown"]
