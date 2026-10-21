"""Tests for RevioIQ crisis detection engine.

Validates detection rules, category classification, severity levels, and human routing.
No network calls.
"""

import pytest
from revioiq.crisis import CrisisDetector, CrisisResult


@pytest.fixture
def detector():
    return CrisisDetector()


def test_clean_positive_review(detector):
    text = "The food was delectable and our waiter Marcus was extraordinarily attentive."
    res: CrisisResult = detector.detect(text)
    assert res.crisis_flag is False
    assert res.crisis_type == "none"
    assert res.severity == "none"
    assert len(res.matched_evidence) == 0
    assert res.routing_action == "auto_reply"


def test_general_negative_review_non_crisis(detector):
    text = "The fries were cold and it took twenty minutes to receive our dessert."
    res = detector.detect(text)
    assert res.crisis_flag is False
    assert res.crisis_type == "none"
    assert res.routing_action == "auto_reply"


def test_food_poisoning_detection(detector):
    text = "I ended up in the emergency room with severe Salmonella poisoning after dining here."
    res = detector.detect(text)
    assert res.crisis_flag is True
    assert res.crisis_type == "food_poisoning"
    assert res.severity == "critical"
    assert res.routing_action == "human_review"
    assert any("salmonella" in ev.lower() for ev in res.matched_evidence)


def test_legal_threat_detection(detector):
    text = "Our attorney is filing a formal negligence lawsuit in civil court tomorrow."
    res = detector.detect(text)
    assert res.crisis_flag is True
    assert res.crisis_type == "legal_threat"
    assert res.severity == "critical"
    assert res.routing_action == "human_review"
    assert any("attorney" in ev.lower() or "lawsuit" in ev.lower() for ev in res.matched_evidence)


def test_discrimination_detection(detector):
    text = "The host used a racist slur and barred a disabled patron with a service dog."
    res = detector.detect(text)
    assert res.crisis_flag is True
    assert res.crisis_type == "discrimination"
    assert res.severity == "critical"
    assert res.routing_action == "human_review"


def test_fraud_accusation_detection(detector):
    text = "This cashier stole my money with an unauthorized tip and fraudulent duplicate billing."
    res = detector.detect(text)
    assert res.crisis_flag is True
    assert res.crisis_type == "fraud_accusation"
    assert res.routing_action == "human_review"


def test_media_threat_detection(detector):
    text = "An investigative reporter from Channel 5 News is broadcasting an expose video tonight."
    res = detector.detect(text)
    assert res.crisis_flag is True
    assert res.crisis_type == "media_threat"
    assert res.routing_action == "human_review"


def test_empty_text_handled_gracefully(detector):
    res = detector.detect("")
    assert res.crisis_flag is False
    assert res.routing_action == "auto_reply"
