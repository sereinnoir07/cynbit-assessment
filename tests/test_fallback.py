"""Tests for RevioIQ offline fallback and no-key behavior.

Verifies that RevioIQ operates fully offline with no API key without attempting network requests.
"""

import pytest
from revioiq.drafter import ReplyDrafter, ReplyDraftResult
from revioiq.validator import ReplyValidator


def test_offline_mode_returns_validated_template(monkeypatch):
    # Ensure has_gemini_api_key returns False
    monkeypatch.setattr("revioiq.drafter.has_gemini_api_key", lambda: False)

    drafter = ReplyDrafter()
    res: ReplyDraftResult = drafter.draft_reply(
        review_text="The dessert was phenomenal and service was top notch.",
        sentiment="positive",
        location="Downtown",
        crisis_flag=False,
    )

    assert res.source == "offline_template"
    assert res.model_used is None
    assert res.validator_passed is True
    assert len(res.violations) == 0
    assert res.routing_action == "auto_reply"
    assert "wonderful feedback" in res.reply_text or "positive feedback" in res.reply_text


def test_offline_mode_crisis_routing(monkeypatch):
    monkeypatch.setattr("revioiq.drafter.has_gemini_api_key", lambda: False)

    drafter = ReplyDrafter()
    res: ReplyDraftResult = drafter.draft_reply(
        review_text="Hospitalized with severe salmonella poisoning after eating raw chicken.",
        sentiment="negative",
        location="Airport",
        crisis_flag=True,
    )

    assert res.source == "offline_template"
    assert res.crisis_flag is True
    assert res.routing_action == "human_review"
    assert res.validator_passed is True
    assert "leadership team" in res.reply_text or "utmost seriousness" in res.reply_text


def test_all_safe_templates_pass_validator():
    from revioiq.drafter import SAFE_TEMPLATES
    validator = ReplyValidator()

    for sentiment_key, template_text in SAFE_TEMPLATES.items():
        val = validator.validate(template_text)
        assert val.passed is True, f"Template for {sentiment_key} failed validator: {val.violations}"
