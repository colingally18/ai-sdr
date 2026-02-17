"""Tests for cross-channel contact deduplication."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from sdr.crm.dedup import ContactDeduplicator
from sdr.models import (
    ContactRecord,
    ConversationStage,
    InboundMessage,
    LeadCategory,
    SourceChannel,
)


@pytest.fixture
def dedup():
    crm = MagicMock()
    return ContactDeduplicator(crm), crm


class TestFindExistingContact:
    def test_match_by_email(self, dedup):
        dedup_instance, crm = dedup
        expected = ContactRecord(
            id="rec_001", name="John", email="john@test.com",
            source_channel=SourceChannel.GMAIL,
        )
        crm.find_contact_by_email.return_value = expected

        msg = InboundMessage(
            source=SourceChannel.GMAIL, source_message_id="m1",
            sender_name="John", sender_email="john@test.com",
            body="Hello", received_at=datetime.utcnow(),
        )

        result = dedup_instance.find_existing_contact(msg)
        assert result is not None
        assert result.id == "rec_001"
        crm.find_contact_by_email.assert_called_once_with("john@test.com")

    def test_match_by_linkedin_url(self, dedup):
        dedup_instance, crm = dedup
        crm.find_contact_by_email.return_value = None
        expected = ContactRecord(
            id="rec_002", name="Sarah",
            linkedin_url="https://linkedin.com/in/sarah",
            source_channel=SourceChannel.LINKEDIN,
        )
        crm.find_contact_by_linkedin_url.return_value = expected

        msg = InboundMessage(
            source=SourceChannel.LINKEDIN, source_message_id="m2",
            sender_name="Sarah",
            sender_linkedin_url="https://linkedin.com/in/sarah",
            body="Hi", received_at=datetime.utcnow(),
        )

        result = dedup_instance.find_existing_contact(msg)
        assert result is not None
        assert result.id == "rec_002"

    def test_match_by_unique_name(self, dedup):
        dedup_instance, crm = dedup
        crm.find_contact_by_email.return_value = None
        crm.find_contact_by_linkedin_url.return_value = None
        expected = ContactRecord(
            id="rec_003", name="UniqueNamePerson",
            source_channel=SourceChannel.GMAIL,
        )
        crm.find_contacts_by_name.return_value = [expected]

        msg = InboundMessage(
            source=SourceChannel.GMAIL, source_message_id="m3",
            sender_name="UniqueNamePerson",
            body="Test", received_at=datetime.utcnow(),
        )

        result = dedup_instance.find_existing_contact(msg)
        assert result is not None
        assert result.id == "rec_003"

    def test_no_match_returns_none(self, dedup):
        dedup_instance, crm = dedup
        crm.find_contact_by_email.return_value = None
        crm.find_contact_by_linkedin_url.return_value = None
        crm.find_contacts_by_name.return_value = []

        msg = InboundMessage(
            source=SourceChannel.GMAIL, source_message_id="m4",
            sender_name="Nobody", body="Test",
            received_at=datetime.utcnow(),
        )

        result = dedup_instance.find_existing_contact(msg)
        assert result is None

    def test_unknown_name_skips_name_matching(self, dedup):
        dedup_instance, crm = dedup
        crm.find_contact_by_email.return_value = None
        crm.find_contact_by_linkedin_url.return_value = None
        # Even if there's a single "Unknown" contact, we should not match on name
        crm.find_contacts_by_name.return_value = [
            ContactRecord(id="rec_unk", name="Unknown",
                          source_channel=SourceChannel.LINKEDIN),
        ]

        msg = InboundMessage(
            source=SourceChannel.LINKEDIN, source_message_id="m_unk",
            sender_name="Unknown", body="Hello",
            received_at=datetime.utcnow(),
        )

        result = dedup_instance.find_existing_contact(msg)
        assert result is None
        # find_contacts_by_name should never be called for "Unknown"
        crm.find_contacts_by_name.assert_not_called()

    def test_ambiguous_name_match_resolves_by_company(self, dedup):
        dedup_instance, crm = dedup
        crm.find_contact_by_email.return_value = None
        crm.find_contact_by_linkedin_url.return_value = None
        crm.find_contacts_by_name.return_value = [
            ContactRecord(id="rec_a", name="John Smith", company="Acme",
                          source_channel=SourceChannel.GMAIL),
            ContactRecord(id="rec_b", name="John Smith", company="Beta Corp",
                          source_channel=SourceChannel.LINKEDIN),
        ]

        msg = InboundMessage(
            source=SourceChannel.GMAIL, source_message_id="m5",
            sender_name="John Smith", sender_company="Acme",
            body="Test", received_at=datetime.utcnow(),
        )

        result = dedup_instance.find_existing_contact(msg)
        assert result is not None
        assert result.id == "rec_a"


class TestShouldUpdateSourceChannel:
    def test_gmail_to_linkedin_should_update(self, dedup):
        dedup_instance, _ = dedup
        contact = ContactRecord(
            id="rec_001", name="Test",
            source_channel=SourceChannel.GMAIL,
        )
        msg = InboundMessage(
            source=SourceChannel.LINKEDIN, source_message_id="m1",
            sender_name="Test", body="Hi",
            received_at=datetime.utcnow(),
        )
        assert dedup_instance.should_update_source_channel(contact, msg) is True

    def test_both_should_not_update(self, dedup):
        dedup_instance, _ = dedup
        contact = ContactRecord(
            id="rec_001", name="Test",
            source_channel=SourceChannel.BOTH,
        )
        msg = InboundMessage(
            source=SourceChannel.LINKEDIN, source_message_id="m1",
            sender_name="Test", body="Hi",
            received_at=datetime.utcnow(),
        )
        assert dedup_instance.should_update_source_channel(contact, msg) is False


class TestMergeContactData:
    def test_fills_missing_email(self, dedup):
        dedup_instance, _ = dedup
        existing = ContactRecord(
            id="rec_001", name="Test",
            source_channel=SourceChannel.LINKEDIN,
            interaction_count=1,
        )
        msg = InboundMessage(
            source=SourceChannel.GMAIL, source_message_id="m1",
            sender_name="Test", sender_email="test@example.com",
            body="Hi", received_at=datetime(2024, 3, 1),
        )

        updates = dedup_instance.merge_contact_data(existing, msg)
        assert updates["Email"] == "test@example.com"
        assert updates["Source Channel"] == "Both"
        assert updates["Interaction Count"] == 2

    def test_does_not_overwrite_existing_fields(self, dedup):
        dedup_instance, _ = dedup
        existing = ContactRecord(
            id="rec_001", name="Test",
            email="existing@email.com",
            company="Existing Corp",
            source_channel=SourceChannel.GMAIL,
            interaction_count=3,
        )
        msg = InboundMessage(
            source=SourceChannel.GMAIL, source_message_id="m1",
            sender_name="Test", sender_email="new@email.com",
            sender_company="New Corp",
            body="Hi", received_at=datetime(2024, 3, 1),
        )

        updates = dedup_instance.merge_contact_data(existing, msg)
        assert "Email" not in updates  # Existing email not overwritten
        assert "Company" not in updates  # Existing company not overwritten
        assert updates["Interaction Count"] == 4
