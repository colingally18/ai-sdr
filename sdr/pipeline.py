"""Inbound pipeline: message → enrich → CRM → classify → draft.

Processes normalized InboundMessages through the full pipeline:
1. Check idempotency (skip if already processed)
2. Deduplicate / find existing contact
3. Upsert contact in Airtable
4. Enrich contact data (if enabled)
5. Classify lead (AI)
6. Draft reply (AI, if should_reply)
7. Create message record in Airtable with draft
8. Update contact with classification
9. Log everything
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import structlog

from sdr import db
from sdr.models import (
    AuditAction,
    AuditLogEntry,
    ContactRecord,
    ConversationStage,
    InboundMessage,
    MessageDirection,
    MessageRecord,
    MessageStatus,
    SourceChannel,
)

if TYPE_CHECKING:
    from sdr.ai.classifier import LeadClassifier
    from sdr.ai.reply_drafter import ReplyDrafter
    from sdr.config import AppConfig
    from sdr.crm.airtable import AirtableCRM
    from sdr.crm.dedup import ContactDeduplicator
    from sdr.enrichment.enricher import ContactEnricher

logger = structlog.get_logger()


class InboundPipeline:
    """Processes inbound messages through the full SDR pipeline."""

    def __init__(
        self,
        crm: AirtableCRM,
        dedup: ContactDeduplicator,
        classifier: LeadClassifier,
        drafter: ReplyDrafter,
        enricher: Optional[ContactEnricher] = None,
        config: Optional[AppConfig] = None,
    ):
        self.crm = crm
        self.dedup = dedup
        self.classifier = classifier
        self.drafter = drafter
        self.enricher = enricher
        self.config = config

    def process_message(self, message: InboundMessage) -> Optional[str]:
        """Process a single inbound message through the full pipeline.

        Returns the Airtable message record ID if successful, None if skipped.
        """
        trace_id = f"msg_{uuid.uuid4().hex[:8]}"
        log = logger.bind(
            trace_id=trace_id,
            source=message.source.value,
            source_message_id=message.source_message_id,
            sender=message.sender_name,
        )

        pipeline_start = time.monotonic()

        # 1. Idempotency check
        if db.is_message_processed(message.source.value, message.source_message_id):
            log.debug("pipeline.already_processed")
            return None

        log.info("pipeline.start")

        try:
            # 2. Find or create contact
            contact = self._upsert_contact(message, trace_id, log)

            # 3. Enrich (if enabled and enricher available)
            enrichment_data = ""
            if self.enricher:
                enrichment_data = self._enrich_contact(contact, message, trace_id, log)

            # 4. Classify
            classification = self._classify(message, enrichment_data, contact, trace_id, log)

            # 5. Draft reply (if warranted)
            draft_reply = ""
            ai_draft_version = ""
            status = MessageStatus.NEW
            if classification.should_reply:
                draft = self._draft_reply(message, classification, enrichment_data, trace_id, log)
                draft_reply = draft.reply_text
                ai_draft_version = draft.reply_text
                status = MessageStatus.DRAFT_READY
            else:
                log.info("pipeline.no_reply_needed", reason=classification.reasoning)

            # 6. Create message record in Airtable
            msg_record = MessageRecord(
                contact_id=contact.id,
                source=message.source,
                direction=MessageDirection.INBOUND,
                subject=message.subject,
                body=message.body,
                thread_context=message.thread_context,
                draft_reply=draft_reply,
                status=status,
                classification=classification.category.value,
                conversation_stage=classification.conversation_stage.value,
                ai_draft_version=ai_draft_version,
                received_at=message.received_at,
                account_id=message.account_id or "",
                source_message_id=message.source_message_id,
            )
            msg_record = self.crm.create_message(msg_record)
            log.info("pipeline.message_created", message_id=msg_record.id, status=status.value)

            # 7. Update contact with classification
            contact_updates = {
                "Lead Category": classification.category.value,
                "Conversation Stage": classification.conversation_stage.value,
                "AI Confidence": classification.confidence,
                "Detected Intent": classification.detected_intent,
                "Signal Stack": json.dumps(classification.detected_signals),
                "AI Reasoning": classification.reasoning,
                "Last Contact": message.received_at.strftime("%Y-%m-%d"),
            }
            self.crm.update_contact(contact.id, contact_updates)

            # 8. Mark processed in SQLite
            db.mark_message_processed(
                source=message.source.value,
                source_message_id=message.source_message_id,
                status="processed",
                airtable_message_id=msg_record.id,
                airtable_contact_id=contact.id,
            )

            # 9. Audit logs
            self.crm.log_audit(AuditLogEntry(
                action=AuditAction.MESSAGE_RECEIVED,
                contact_id=contact.id,
                message_id=msg_record.id,
                details=json.dumps({
                    "trace_id": trace_id,
                    "source": message.source.value,
                    "sender": message.sender_name,
                }),
            ))
            self.crm.log_audit(AuditLogEntry(
                action=AuditAction.CLASSIFIED,
                contact_id=contact.id,
                message_id=msg_record.id,
                details=json.dumps({
                    "trace_id": trace_id,
                    "category": classification.category.value,
                    "confidence": classification.confidence,
                    "intent": classification.detected_intent,
                    "stage": classification.conversation_stage.value,
                    "icp_score": classification.icp_match_score,
                }),
            ))
            if draft_reply:
                self.crm.log_audit(AuditLogEntry(
                    action=AuditAction.DRAFT_CREATED,
                    contact_id=contact.id,
                    message_id=msg_record.id,
                    details=json.dumps({
                        "trace_id": trace_id,
                        "word_count": len(draft_reply.split()),
                    }),
                ))

            duration_ms = int((time.monotonic() - pipeline_start) * 1000)
            db.log_local_audit(
                action="pipeline_complete",
                trace_id=trace_id,
                source=message.source.value,
                message_id=msg_record.id,
                contact_id=contact.id,
                details={
                    "category": classification.category.value,
                    "confidence": classification.confidence,
                    "should_reply": classification.should_reply,
                    "status": status.value,
                },
                duration_ms=duration_ms,
            )

            log.info(
                "pipeline.complete",
                duration_ms=duration_ms,
                category=classification.category.value,
                status=status.value,
            )
            return msg_record.id

        except Exception as e:
            log.error("pipeline.failed", error=str(e), exc_info=True)
            db.mark_message_failed(
                source=message.source.value,
                source_message_id=message.source_message_id,
                error=str(e),
            )
            return None

    def _upsert_contact(
        self, message: InboundMessage, trace_id: str, log
    ) -> ContactRecord:
        """Find existing contact or create a new one."""
        existing = self.dedup.find_existing_contact(message)

        if existing:
            # Merge new data into existing contact
            updates = self.dedup.merge_contact_data(existing, message)
            if updates:
                self.crm.update_contact(existing.id, updates)
                log.info("pipeline.contact_updated", contact_id=existing.id, updates=list(updates.keys()))
                self.crm.log_audit(AuditLogEntry(
                    action=AuditAction.CONTACT_UPDATED,
                    contact_id=existing.id,
                    details=json.dumps({"trace_id": trace_id, "updates": list(updates.keys())}),
                ))
            return existing
        else:
            # Create new contact
            contact = ContactRecord(
                name=message.sender_name,
                email=message.sender_email,
                linkedin_url=message.sender_linkedin_url,
                company=message.sender_company,
                title=message.sender_title,
                source_channel=message.source,
                conversation_stage=ConversationStage.NEW,
                first_contact=message.received_at,
                last_contact=message.received_at,
                interaction_count=1,
            )
            contact = self.crm.upsert_contact(contact)
            log.info("pipeline.contact_created", contact_id=contact.id)
            self.crm.log_audit(AuditLogEntry(
                action=AuditAction.CONTACT_CREATED,
                contact_id=contact.id,
                details=json.dumps({"trace_id": trace_id, "name": contact.name}),
            ))
            return contact

    def _enrich_contact(
        self, contact: ContactRecord, message: InboundMessage, trace_id: str, log
    ) -> str:
        """Enrich contact with external data. Returns enrichment JSON string."""
        try:
            start = time.monotonic()
            data = self.enricher.enrich(
                email=contact.email,
                linkedin_url=contact.linkedin_url,
                name=contact.name,
                company=contact.company,
            )
            duration_ms = int((time.monotonic() - start) * 1000)

            if data:
                enrichment_json = json.dumps(data)
                # Write structured fields back to contact, plus raw JSON
                contact_updates: dict = {"Enriched Data": enrichment_json}
                if data.get("title") and not contact.title:
                    contact_updates["Title"] = data["title"]
                if data.get("company") and not contact.company:
                    contact_updates["Company"] = data["company"]
                if data.get("linkedin_url") and not contact.linkedin_url:
                    contact_updates["LinkedIn URL"] = data["linkedin_url"]
                if data.get("email") and not contact.email:
                    contact_updates["Email"] = data["email"]
                self.crm.update_contact(contact.id, contact_updates)
                self.crm.log_audit(AuditLogEntry(
                    action=AuditAction.ENRICHED,
                    contact_id=contact.id,
                    details=json.dumps({"trace_id": trace_id, "duration_ms": duration_ms}),
                ))
                log.info("pipeline.enriched", contact_id=contact.id, duration_ms=duration_ms)
                return enrichment_json
        except Exception as e:
            log.warning("pipeline.enrichment_failed", error=str(e))
        return ""

    def _classify(
        self, message: InboundMessage, enrichment_data: str, contact: ContactRecord,
        trace_id: str, log
    ):
        """Classify the lead using AI."""
        start = time.monotonic()
        classification = self.classifier.classify(
            message=message,
            enrichment_data=enrichment_data,
            current_stage=contact.conversation_stage.value if contact.conversation_stage else "",
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        log.info(
            "pipeline.classified",
            category=classification.category.value,
            confidence=classification.confidence,
            intent=classification.detected_intent,
            duration_ms=duration_ms,
        )
        return classification

    def _draft_reply(self, message, classification, enrichment_data, trace_id, log):
        """Draft a reply using AI."""
        start = time.monotonic()
        draft = self.drafter.draft(
            message=message,
            classification=classification,
            enrichment_data=enrichment_data,
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        log.info(
            "pipeline.drafted",
            word_count=len(draft.reply_text.split()),
            duration_ms=duration_ms,
        )
        return draft

    def process_batch(self, messages: list[InboundMessage]) -> dict:
        """Process a batch of messages. Returns summary stats."""
        stats = {"total": len(messages), "processed": 0, "skipped": 0, "failed": 0}

        for msg in messages:
            result = self.process_message(msg)
            if result:
                stats["processed"] += 1
            else:
                # Could be skipped (already processed) or failed
                if db.is_message_processed(msg.source.value, msg.source_message_id):
                    stats["skipped"] += 1
                else:
                    stats["failed"] += 1

        logger.info("pipeline.batch_complete", **stats)
        return stats
