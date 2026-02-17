"""Tests for reply draft quality.

Validates that AI-generated drafts meet the consultative framework checklist:
- Word count limits
- No filler words
- No setup language
- Contains exit ramp
- Stage-appropriate
"""

import re

import pytest

from sdr.models import (
    ConversationStage,
    DraftReply,
    LeadCategory,
    LeadClassification,
    SourceChannel,
)
from sdr.outbound import compute_edit_distance


# --- Quality check helpers ---

FILLER_WORDS = {"just", "really", "actually", "basically", "honestly", "simply", "literally"}
SETUP_PHRASES = [
    "i wanted to reach out",
    "hope this finds you well",
    "i hope you're doing well",
    "i'd love to connect",
    "i came across your profile",
    "i noticed your profile",
]
EXIT_RAMP_PHRASES = [
    "no worries",
    "if not",
    "not the right time",
    "either way",
    "ignore me",
    "no agenda",
    "totally fine",
    "not relevant",
]


def check_word_count(text: str, source: str) -> tuple[bool, int]:
    """Check if word count is within limits."""
    word_count = len(text.split())
    if source == "LinkedIn":
        return word_count <= 65, word_count  # Small buffer over 60
    else:
        return word_count <= 160, word_count  # Small buffer over 150


def check_no_filler_words(text: str) -> tuple[bool, list[str]]:
    """Check for filler words."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    found = words & FILLER_WORDS
    return len(found) == 0, list(found)


def check_no_setup_language(text: str) -> tuple[bool, list[str]]:
    """Check for setup language."""
    lower_text = text.lower()
    found = [phrase for phrase in SETUP_PHRASES if phrase in lower_text]
    return len(found) == 0, found


def check_has_exit_ramp(text: str) -> tuple[bool, list[str]]:
    """Check for exit ramp language."""
    lower_text = text.lower()
    found = [phrase for phrase in EXIT_RAMP_PHRASES if phrase in lower_text]
    return len(found) > 0, found


def check_has_question(text: str) -> bool:
    """Check if the reply contains a question (CTA)."""
    return "?" in text


# --- Tests ---

class TestReplyQualityChecks:
    """Test the quality check helpers themselves."""

    def test_good_linkedin_reply(self):
        reply = (
            "Noticed you're scaling your agency — curious if LinkedIn outbound "
            "is something you've tested, or if most of your pipeline comes from "
            "referrals right now? If not on your radar, no worries at all."
        )
        ok, count = check_word_count(reply, "LinkedIn")
        assert ok, f"Word count {count} exceeds LinkedIn limit"

        ok, found = check_no_filler_words(reply)
        assert ok, f"Found filler words: {found}"

        ok, found = check_no_setup_language(reply)
        assert ok, f"Found setup language: {found}"

        ok, found = check_has_exit_ramp(reply)
        assert ok, "Missing exit ramp"

        assert check_has_question(reply), "Missing question/CTA"

    def test_bad_reply_with_filler_words(self):
        reply = "I just really wanted to basically reach out and honestly see if you need help."
        ok, found = check_no_filler_words(reply)
        assert not ok
        assert "just" in found
        assert "really" in found
        assert "basically" in found
        assert "honestly" in found

    def test_bad_reply_with_setup_language(self):
        reply = "Hope this finds you well! I wanted to reach out because we offer LinkedIn services."
        ok, found = check_no_setup_language(reply)
        assert not ok
        assert any("hope this finds you well" in p for p in found)
        assert any("i wanted to reach out" in p for p in found)

    def test_reply_without_exit_ramp(self):
        reply = "We help agencies book more meetings through LinkedIn. Let's chat?"
        ok, found = check_has_exit_ramp(reply)
        assert not ok

    def test_reply_without_question(self):
        reply = "We help agencies book more meetings through LinkedIn."
        assert not check_has_question(reply)


class TestEditDistance:
    def test_identical_strings(self):
        assert compute_edit_distance("hello world", "hello world") == 0.0

    def test_completely_different(self):
        dist = compute_edit_distance("hello", "xyz")
        assert dist > 0.5

    def test_small_edit(self):
        dist = compute_edit_distance(
            "Thanks for reaching out — curious about your setup?",
            "Thanks for reaching out — curious about your current setup?",
        )
        assert dist < 0.2  # Small edit

    def test_empty_strings(self):
        assert compute_edit_distance("", "") == 0.0
        assert compute_edit_distance("hello", "") == 1.0
        assert compute_edit_distance("", "hello") == 1.0

    def test_moderate_edit(self):
        original = (
            "Noticed you're scaling your agency — curious if LinkedIn outbound "
            "is something you've tested? If not, no worries."
        )
        edited = (
            "Congrats on scaling the agency — have you explored LinkedIn outbound "
            "as a channel? If it's not on your radar right now, totally fine."
        )
        dist = compute_edit_distance(original, edited)
        assert 0.2 < dist < 0.8  # Significant but not total rewrite


class TestDraftReplyModel:
    def test_creates_draft(self):
        draft = DraftReply(
            reply_text="Test reply",
            strategy_notes="Used qualification-led approach",
        )
        assert draft.reply_text == "Test reply"
        assert draft.strategy_notes != ""

    def test_draft_with_empty_notes(self):
        draft = DraftReply(reply_text="Test reply")
        assert draft.strategy_notes == ""
