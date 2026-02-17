"""Tests for the lead classifier.

Uses mock Claude API responses to verify classification parsing and
labeled test fixtures for accuracy validation.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from sdr.ai.classifier import LeadClassifier
from sdr.models import (
    ConversationStage,
    InboundMessage,
    LeadCategory,
    LeadClassification,
    SourceChannel,
)


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Claude API response with tool_use."""
    def _make_response(tool_input: dict):
        response = MagicMock()
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.input = tool_input
        response.content = [tool_block]
        return response
    return _make_response


@pytest.fixture
def classifier():
    """Create a classifier with mocked Anthropic client."""
    with patch("sdr.ai.classifier.anthropic") as mock_anthropic:
        c = LeadClassifier(
            api_key="test-key",
            model="claude-sonnet-4-5-20250929",
            temperature=0.1,
        )
        yield c, mock_anthropic


class TestLeadClassifier:
    def test_classifies_hot_lead(self, classifier, mock_anthropic_response, sample_gmail_message):
        c, mock_module = classifier
        c.client.messages.create.return_value = mock_anthropic_response({
            "category": "Hot",
            "confidence": 0.92,
            "reasoning": "Direct pricing inquiry from ICP-matching CEO",
            "detected_intent": "pricing inquiry",
            "detected_signals": ["direct_inquiry", "icp_match"],
            "should_reply": True,
            "conversation_stage": "New",
            "icp_match_score": 0.95,
        })

        result = c.classify(sample_gmail_message)

        assert isinstance(result, LeadClassification)
        assert result.category == LeadCategory.HOT
        assert result.confidence == 0.92
        assert result.should_reply is True
        assert result.icp_match_score == 0.95

    def test_classifies_not_a_lead(self, classifier, mock_anthropic_response, sample_job_seeker_message):
        c, _ = classifier
        c.client.messages.create.return_value = mock_anthropic_response({
            "category": "Not a Lead",
            "confidence": 0.98,
            "reasoning": "Job seeker, not a potential customer",
            "detected_intent": "job seeking",
            "detected_signals": ["job_seeker"],
            "should_reply": True,
            "conversation_stage": "New",
            "icp_match_score": 0.0,
        })

        result = c.classify(sample_job_seeker_message)

        assert result.category == LeadCategory.NOT_A_LEAD
        assert result.should_reply is True  # Still polite to reply to job seekers

    def test_raises_on_missing_tool_use(self, classifier, sample_gmail_message):
        c, _ = classifier
        response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        response.content = [text_block]
        c.client.messages.create.return_value = response

        with pytest.raises(ValueError, match="tool_use"):
            c.classify(sample_gmail_message)

    def test_passes_enrichment_data_to_prompt(self, classifier, mock_anthropic_response, sample_gmail_message):
        c, _ = classifier
        c.client.messages.create.return_value = mock_anthropic_response({
            "category": "Warm",
            "confidence": 0.7,
            "reasoning": "test",
            "detected_intent": "test",
            "detected_signals": [],
            "should_reply": True,
            "conversation_stage": "New",
            "icp_match_score": 0.5,
        })

        with patch("sdr.ai.classifier.build_classification_prompt") as mock_prompt:
            mock_prompt.return_value = "test prompt"
            c.classify(sample_gmail_message, enrichment_data='{"company_size": 25}')
            mock_prompt.assert_called_once_with(
                message=sample_gmail_message,
                enrichment_data='{"company_size": 25}',
                current_stage="",
            )


class TestClassificationModel:
    def test_valid_classification(self):
        c = LeadClassification(
            category=LeadCategory.WARM,
            confidence=0.75,
            reasoning="test",
            detected_intent="test",
            detected_signals=["signal1"],
            should_reply=True,
            conversation_stage=ConversationStage.NEW,
            icp_match_score=0.6,
        )
        assert c.category == LeadCategory.WARM
        assert 0 <= c.confidence <= 1
        assert 0 <= c.icp_match_score <= 1

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            LeadClassification(
                category=LeadCategory.HOT,
                confidence=1.5,  # Out of bounds
                reasoning="test",
                detected_intent="test",
                detected_signals=[],
                should_reply=True,
                conversation_stage=ConversationStage.NEW,
                icp_match_score=0.5,
            )
