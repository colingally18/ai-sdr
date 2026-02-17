"""Cross-channel contact deduplication.

Matches incoming messages to existing contacts using email, LinkedIn URL,
and fuzzy name matching. Prevents duplicate contact creation across Gmail
and LinkedIn channels.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

import structlog

if TYPE_CHECKING:
    from sdr.crm.airtable import AirtableCRM
    from sdr.models import ContactRecord, InboundMessage

logger = structlog.get_logger()


class ContactDeduplicator:
    """Finds existing contacts matching an inbound message."""

    def __init__(self, crm: AirtableCRM):
        self.crm = crm

    def find_existing_contact(self, message: InboundMessage) -> Optional[ContactRecord]:
        """Find an existing contact that matches the message sender.

        Matching priority:
        1. Exact email match
        2. Exact LinkedIn URL match
        3. Exact name match (if unique)
        4. Returns None if no match found
        """
        # 1. Exact email match
        if message.sender_email:
            match = self.crm.find_contact_by_email(message.sender_email)
            if match:
                logger.info(
                    "dedup.match_by_email",
                    email=message.sender_email,
                    contact_id=match.id,
                )
                return match

        # 2. Exact LinkedIn URL match
        if message.sender_linkedin_url:
            match = self.crm.find_contact_by_linkedin_url(message.sender_linkedin_url)
            if match:
                logger.info(
                    "dedup.match_by_linkedin",
                    linkedin_url=message.sender_linkedin_url,
                    contact_id=match.id,
                )
                return match

        # 3. Name match (only if exactly one person with that name)
        # Skip name matching for "Unknown" to prevent merging all
        # unresolved LinkedIn contacts into a single record.
        if message.sender_name and message.sender_name != "Unknown":
            candidates = self.crm.find_contacts_by_name(message.sender_name)
            if len(candidates) == 1:
                logger.info(
                    "dedup.match_by_name",
                    name=message.sender_name,
                    contact_id=candidates[0].id,
                )
                return candidates[0]
            elif len(candidates) > 1:
                # Multiple matches — try to disambiguate by company
                if message.sender_company:
                    for c in candidates:
                        if c.company and c.company.lower() == message.sender_company.lower():
                            logger.info(
                                "dedup.match_by_name_and_company",
                                name=message.sender_name,
                                company=message.sender_company,
                                contact_id=c.id,
                            )
                            return c
                logger.warning(
                    "dedup.ambiguous_name_match",
                    name=message.sender_name,
                    candidate_count=len(candidates),
                )

        # 4. No match
        logger.info("dedup.no_match", sender_name=message.sender_name)
        return None

    def should_update_source_channel(
        self, existing: ContactRecord, message: InboundMessage
    ) -> bool:
        """Check if we should update source channel to 'Both'."""
        from sdr.models import SourceChannel

        if existing.source_channel == SourceChannel.BOTH:
            return False
        if existing.source_channel == SourceChannel.GMAIL and message.source == SourceChannel.LINKEDIN:
            return True
        if existing.source_channel == SourceChannel.LINKEDIN and message.source == SourceChannel.GMAIL:
            return True
        return False

    def merge_contact_data(
        self, existing: ContactRecord, message: InboundMessage
    ) -> dict:
        """Return fields that should be updated on the existing contact.

        Fills in missing data (email, LinkedIn URL, company, title) from the
        new message without overwriting existing values.
        """
        from sdr.models import SourceChannel

        updates: dict = {}

        # Fill missing email
        if not existing.email and message.sender_email:
            updates["Email"] = message.sender_email

        # Fill missing LinkedIn URL
        if not existing.linkedin_url and message.sender_linkedin_url:
            updates["LinkedIn URL"] = message.sender_linkedin_url

        # Fill missing company
        if not existing.company and message.sender_company:
            updates["Company"] = message.sender_company

        # Fill missing title
        if not existing.title and message.sender_title:
            updates["Title"] = message.sender_title

        # Update source channel to Both if cross-channel
        if self.should_update_source_channel(existing, message):
            updates["Source Channel"] = SourceChannel.BOTH.value

        # Always update last contact and increment interaction count
        if message.received_at:
            updates["Last Contact"] = message.received_at.strftime("%Y-%m-%d")
        updates["Interaction Count"] = existing.interaction_count + 1

        return updates
