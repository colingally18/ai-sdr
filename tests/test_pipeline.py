"""Tests for the inbound pipeline with mocked APIs."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from sdr.crm.dedup import ContactDeduplicator
from sdr.models import (
    ContactRecord,
    ConversationStage,
    DraftReply,
    InboundMessage,
    LeadCategory,
    LeadClassification,
    MessageDirection,
    MessageRecord,
    MessageStatus,
    SourceChannel,
)
from sdr.pipeline import InboundPipeline


@pytest.fixture
def pipeline(mock_crm, mock_classifier, mock_drafter):
    dedup = ContactDeduplicator(mock_crm)
    # Make dedup return None (new contact) by default
    mock_crm.find_contact_by_email.return_value = None
    mock_crm.find_contact_by_linkedin_url.return_value = None
    mock_crm.find_contacts_by_name.return_value = []

    return InboundPipeline(
        crm=mock_crm,
        dedup=dedup,
        classifier=mock_classifier,
        drafter=mock_drafter,
    )


class TestProcessMessage:
    @patch("sdr.pipeline.db")
    def test_processes_new_message_successfully(
        self, mock_db, pipeline, sample_gmail_message
    ):
        mock_db.is_message_processed.return_value = False
        mock_db.mark_message_processed.return_value = None
        mock_db.log_local_audit.return_value = None

        result = pipeline.process_message(sample_gmail_message)

        assert result is not None  # Returns message record ID
        pipeline.crm.upsert_contact.assert_called_once()
        pipeline.classifier.classify.assert_called_once()
        pipeline.drafter.draft.assert_called_once()
        pipeline.crm.create_message.assert_called_once()

    @patch("sdr.pipeline.db")
    def test_skips_already_processed_message(
        self, mock_db, pipeline, sample_gmail_message
    ):
        mock_db.is_message_processed.return_value = True

        result = pipeline.process_message(sample_gmail_message)

        assert result is None
        pipeline.crm.upsert_contact.assert_not_called()
        pipeline.classifier.classify.assert_not_called()

    @patch("sdr.pipeline.db")
    def test_does_not_draft_when_should_not_reply(
        self, mock_db, pipeline, sample_gmail_message
    ):
        mock_db.is_message_processed.return_value = False
        mock_db.mark_message_processed.return_value = None
        mock_db.log_local_audit.return_value = None

        pipeline.classifier.classify.return_value = LeadClassification(
            category=LeadCategory.NOT_A_LEAD,
            confidence=0.95,
            reasoning="Spam",
            detected_intent="spam",
            detected_signals=[],
            should_reply=False,
            conversation_stage=ConversationStage.NEW,
            icp_match_score=0.0,
        )

        result = pipeline.process_message(sample_gmail_message)

        assert result is not None
        pipeline.drafter.draft.assert_not_called()
        # Message should still be created but without draft
        create_call = pipeline.crm.create_message.call_args
        msg_arg = create_call[0][0]
        assert msg_arg.status == MessageStatus.NEW
        assert msg_arg.draft_reply == ""

    @patch("sdr.pipeline.db")
    def test_creates_draft_ready_status_when_should_reply(
        self, mock_db, pipeline, sample_gmail_message
    ):
        mock_db.is_message_processed.return_value = False
        mock_db.mark_message_processed.return_value = None
        mock_db.log_local_audit.return_value = None

        result = pipeline.process_message(sample_gmail_message)

        create_call = pipeline.crm.create_message.call_args
        msg_arg = create_call[0][0]
        assert msg_arg.status == MessageStatus.DRAFT_READY
        assert msg_arg.draft_reply != ""
        assert msg_arg.ai_draft_version != ""

    @patch("sdr.pipeline.db")
    def test_updates_contact_with_classification(
        self, mock_db, pipeline, sample_gmail_message
    ):
        mock_db.is_message_processed.return_value = False
        mock_db.mark_message_processed.return_value = None
        mock_db.log_local_audit.return_value = None

        pipeline.process_message(sample_gmail_message)

        # Should update contact with classification data
        pipeline.crm.update_contact.assert_called()
        update_call = pipeline.crm.update_contact.call_args
        fields = update_call[0][1]
        assert "Lead Category" in fields
        assert "Conversation Stage" in fields
        assert "AI Confidence" in fields

    @patch("sdr.pipeline.db")
    def test_marks_failed_on_error(self, mock_db, pipeline, sample_gmail_message):
        mock_db.is_message_processed.return_value = False
        mock_db.mark_message_failed.return_value = None
        pipeline.classifier.classify.side_effect = RuntimeError("API error")

        result = pipeline.process_message(sample_gmail_message)

        assert result is None
        mock_db.mark_message_failed.assert_called_once()

    @patch("sdr.pipeline.db")
    def test_existing_contact_gets_updated(
        self, mock_db, pipeline, sample_gmail_message, sample_contact
    ):
        mock_db.is_message_processed.return_value = False
        mock_db.mark_message_processed.return_value = None
        mock_db.log_local_audit.return_value = None

        # Make dedup find an existing contact
        pipeline.crm.find_contact_by_email.return_value = sample_contact

        pipeline.process_message(sample_gmail_message)

        # Should NOT call upsert_contact for new creation
        # but should call update_contact for classification
        pipeline.crm.update_contact.assert_called()


class TestProcessBatch:
    @patch("sdr.pipeline.db")
    def test_processes_batch_and_returns_stats(self, mock_db, pipeline):
        mock_db.is_message_processed.return_value = False
        mock_db.mark_message_processed.return_value = None
        mock_db.log_local_audit.return_value = None

        messages = [
            InboundMessage(
                source=SourceChannel.GMAIL,
                source_message_id=f"msg_{i}",
                sender_name=f"User {i}",
                sender_email=f"user{i}@test.com",
                body="Hello",
                received_at=datetime.utcnow(),
            )
            for i in range(3)
        ]

        stats = pipeline.process_batch(messages)
        assert stats["total"] == 3
        assert stats["processed"] == 3
