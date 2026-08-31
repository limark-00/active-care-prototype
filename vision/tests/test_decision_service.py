from decision_service import apply_decision_guardrails


def predictions(**patch):
    values = {
        "derived_risk_level": {"label": "L1", "confidence": 0.9},
        "intervention_level": {"label": "I1", "confidence": 0.9},
        "alert_mode": {"label": "NONE", "confidence": 0.9},
        "manual_review": {"label": False, "confidence": 0.9},
        "abstain": {"label": False, "confidence": 0.9},
    }
    for target, value in patch.items():
        values[target].update(value)
    return values


def test_low_confidence_in_any_head_requires_review():
    guarded, reasons = apply_decision_guardrails(
        predictions(abstain={"confidence": 0.51})
    )
    assert guarded["manual_review"] is True
    assert any("低置信度" in reason for reason in reasons)


def test_l4_and_i4_invariants_are_applied_without_mutating_raw_heads():
    raw = predictions(
        derived_risk_level={"label": "L4"},
        intervention_level={"label": "I2"},
        alert_mode={"label": "NONE"},
    )
    guarded, reasons = apply_decision_guardrails(raw)
    assert guarded["intervention_level"] == "I4"
    assert guarded["alert_mode"] == "URGENT_HELP"
    assert guarded["manual_review"] is True
    assert raw["intervention_level"]["label"] == "I2"
    assert reasons


def test_l2_l3_cannot_silently_return_no_page_warning():
    guarded, reasons = apply_decision_guardrails(
        predictions(derived_risk_level={"label": "L3"}, alert_mode={"label": "NONE"})
    )
    assert guarded["alert_mode"] == "PAGE_WARNING"
    assert guarded["manual_review"] is True
    assert any("L2/L3" in reason for reason in reasons)


def test_abstention_is_explicitly_non_automatic():
    guarded, reasons = apply_decision_guardrails(
        predictions(abstain={"label": True})
    )
    assert guarded["abstain"] is True
    assert guarded["manual_review"] is True
    assert any("拒绝自动判断" in reason for reason in reasons)
