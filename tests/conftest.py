"""Shared test fixtures for the Growlancer SDR test suite."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_gmail_message() -> InboundMessage:
    """A sample inbound Gmail message from a hot lead."""
    return InboundMessage(
        source=SourceChannel.GMAIL,
        source_message_id="msg_001",
        sender_name="John Smith",
        sender_email="john@acmeagency.com",
        sender_title="CEO",
        sender_company="Acme Marketing Agency",
        subject="LinkedIn outreach services",
        body=(
            "Hi Colin,\n\n"
            "I've been looking into LinkedIn outreach services for our B2B marketing agency. "
            "We have about 25 employees and are doing around $3M in revenue. Our average client "
            "is worth about $40K/mo.\n\n"
            "We've been trying to do LinkedIn prospecting in-house but the reply rates have been "
            "pretty bad. Would love to hear how you approach this.\n\n"
            "Best,\nJohn"
        ),
        thread_context="",
        received_at=datetime(2024, 2, 16, 12, 0, 0),
    )


@pytest.fixture
def sample_linkedin_message() -> InboundMessage:
    """A sample inbound LinkedIn DM from a warm lead."""
    return InboundMessage(
        source=SourceChannel.LINKEDIN,
        source_message_id="li_msg_001",
        sender_name="Sarah Johnson",
        sender_linkedin_url="https://linkedin.com/in/sarahjohnson",
        sender_title="Managing Partner at Growth Consulting Group",
        sender_company="Growth Consulting Group",
        body=(
            "Hey Colin, saw your posts about LinkedIn outreach — we're a consulting firm "
            "and have been trying to get more clients through LinkedIn but struggling with "
            "response rates. Any tips?"
        ),
        thread_context="",
        received_at=datetime(2024, 2, 16, 14, 30, 0),
        account_id="acct_001",
    )


@pytest.fixture
def sample_job_seeker_message() -> InboundMessage:
    """A sample message from a job seeker (not a lead)."""
    return InboundMessage(
        source=SourceChannel.LINKEDIN,
        source_message_id="li_msg_002",
        sender_name="Alex Rivera",
        sender_title="Software Engineer",
        sender_company="Google",
        body="Hi! I'm looking for new opportunities in sales development. Are you guys hiring by any chance?",
        received_at=datetime(2024, 2, 16, 15, 0, 0),
    )


@pytest.fixture
def sample_competitor_message() -> InboundMessage:
    """A sample message from a competitor."""
    return InboundMessage(
        source=SourceChannel.LINKEDIN,
        source_message_id="li_msg_003",
        sender_name="Chris Davis",
        sender_title="Founder at LeadGenPro",
        sender_company="LeadGenPro",
        body="Hey! We also do LinkedIn outreach automation. Would love to explore a partnership opportunity.",
        received_at=datetime(2024, 2, 16, 15, 30, 0),
    )


@pytest.fixture
def sample_classification_hot() -> LeadClassification:
    """A hot lead classification."""
    return LeadClassification(
        category=LeadCategory.HOT,
        confidence=0.92,
        reasoning="Direct pricing inquiry from ICP-matching B2B agency CEO",
        detected_intent="pricing inquiry",
        detected_signals=["direct_inquiry", "icp_match", "budget_signal"],
        should_reply=True,
        conversation_stage=ConversationStage.NEW,
        icp_match_score=0.95,
    )


@pytest.fixture
def sample_classification_not_lead() -> LeadClassification:
    """A not-a-lead classification."""
    return LeadClassification(
        category=LeadCategory.NOT_A_LEAD,
        confidence=0.98,
        reasoning="Job seeker, not a potential customer",
        detected_intent="job seeking",
        detected_signals=["job_seeker"],
        should_reply=True,
        conversation_stage=ConversationStage.NEW,
        icp_match_score=0.0,
    )


@pytest.fixture
def sample_contact() -> ContactRecord:
    """A sample contact record."""
    return ContactRecord(
        id="rec_contact_001",
        name="John Smith",
        email="john@acmeagency.com",
        company="Acme Marketing Agency",
        title="CEO",
        source_channel=SourceChannel.GMAIL,
        lead_category=LeadCategory.HOT,
        conversation_stage=ConversationStage.NEW,
        ai_confidence=0.92,
        first_contact=datetime(2024, 2, 16, 12, 0, 0),
        last_contact=datetime(2024, 2, 16, 12, 0, 0),
        interaction_count=1,
    )


@pytest.fixture
def sample_message_record() -> MessageRecord:
    """A sample message record."""
    return MessageRecord(
        id="rec_msg_001",
        contact_id="rec_contact_001",
        source=SourceChannel.GMAIL,
        direction=MessageDirection.INBOUND,
        subject="LinkedIn outreach services",
        body="Hi Colin, interested in your services...",
        status=MessageStatus.DRAFT_READY,
        draft_reply="Thanks for reaching out...",
        ai_draft_version="Thanks for reaching out...",
        classification="Hot",
        conversation_stage="New",
        received_at=datetime(2024, 2, 16, 12, 0, 0),
        source_message_id="msg_001",
    )


@pytest.fixture
def mock_crm():
    """A mocked AirtableCRM instance."""
    crm = MagicMock()
    crm.upsert_contact.return_value = ContactRecord(
        id="rec_new_contact",
        name="Test User",
        email="test@example.com",
        source_channel=SourceChannel.GMAIL,
        conversation_stage=ConversationStage.NEW,
        interaction_count=1,
    )
    crm.create_message.return_value = MessageRecord(
        id="rec_new_msg",
        source=SourceChannel.GMAIL,
        direction=MessageDirection.INBOUND,
        body="test",
        status=MessageStatus.DRAFT_READY,
    )
    crm.get_approved_messages.return_value = []
    crm.log_audit.return_value = None
    crm.update_contact.return_value = None
    crm.update_message.return_value = None
    return crm


@pytest.fixture
def mock_classifier():
    """A mocked LeadClassifier instance."""
    classifier = MagicMock()
    classifier.classify.return_value = LeadClassification(
        category=LeadCategory.WARM,
        confidence=0.75,
        reasoning="Shows interest in LinkedIn outreach",
        detected_intent="information seeking",
        detected_signals=["icp_match"],
        should_reply=True,
        conversation_stage=ConversationStage.NEW,
        icp_match_score=0.7,
    )
    return classifier


@pytest.fixture
def mock_drafter():
    """A mocked ReplyDrafter instance."""
    drafter = MagicMock()
    drafter.draft.return_value = DraftReply(
        reply_text="Thanks for the message — curious, are you currently doing any outbound on LinkedIn, or is most of your pipeline from referrals right now?",
        strategy_notes="Qualification-led approach for warm lead",
    )
    return drafter


@pytest.fixture
def tmp_db(tmp_path):
    """A temporary SQLite database path."""
    return tmp_path / "test_sdr.db"


@pytest.fixture
def labeled_messages() -> list[dict]:
    """Load labeled messages for classification accuracy testing."""
    path = FIXTURES_DIR / "labeled_messages.json"
    with open(path) as f:
        return json.load(f)
