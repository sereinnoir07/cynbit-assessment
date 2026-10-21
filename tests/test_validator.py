"""Tests for RevioIQ pure-Python guardrail validator.

Ensures replies adhere to corporate compliance guidelines without network calls.
"""

import pytest
from revioiq.validator import ReplyValidator, ValidationResult


@pytest.fixture
def validator():
    return ReplyValidator(min_len=20, max_len=750)


def test_valid_reply_passes(validator):
    valid_text = (
        "Thank you for sharing your feedback with us. We take pride in delivering great "
        "experiences, and our management team will review your comments with our staff."
    )
    res: ValidationResult = validator.validate(valid_text)
    assert res.passed is True
    assert len(res.violations) == 0
    assert res.reply_text == valid_text


@pytest.mark.parametrize(
    "phrase",
    [
        "We are truly sorry and promise a full refund right away.",
        "We will issue a refund to your card immediately.",
        "Our manager will give a 100% refund for this experience.",
        "We offer automatic monetary compensation to make up for this.",
        "We will wire $500 directly to your checking account.",
        "Enjoy free meals for life on us as an apology.",
    ],
)
def test_banned_financial_promises_fail(validator, phrase):
    res = validator.validate(phrase)
    assert res.passed is False
    assert any("financial compensation promise" in v for v in res.violations)


@pytest.mark.parametrize(
    "phrase",
    [
        "The company admits full legal liability for your accident.",
        "We acknowledge our gross negligence in handling the food.",
        "Our staff was at fault and we admit guilt in this matter.",
        "We accept legal liability and want to settle out of court.",
    ],
)
def test_banned_legal_admissions_fail(validator, phrase):
    res = validator.validate(phrase)
    assert res.passed is False
    assert any("legal liability admission" in v for v in res.violations)


def test_pii_detection_email(validator):
    text_with_email = (
        "Please contact our operations manager directly at sarah.jenkins@cynbit-corp.com "
        "to resolve this problem."
    )
    res = validator.validate(text_with_email)
    assert res.passed is False
    assert any("Personal email address detected" in v for v in res.violations)


def test_pii_detection_phone(validator):
    text_with_phone = (
        "Call our general manager on his cell phone at 555-839-2019 to discuss "
        "your dining experience."
    )
    res = validator.validate(text_with_phone)
    assert res.passed is False
    assert any("Phone number detected" in v for v in res.violations)


def test_too_short_reply_fails(validator):
    res = validator.validate("Too short!")
    assert res.passed is False
    assert any("too short" in v for v in res.violations)


def test_too_long_reply_fails(validator):
    long_text = "This is a detailed response. " * 35  # ~1000 characters
    res = validator.validate(long_text)
    assert res.passed is False
    assert any("too long" in v for v in res.violations)


def test_empty_input_fails(validator):
    res = validator.validate("")
    assert res.passed is False
    assert "Empty or invalid reply string" in res.violations[0]
