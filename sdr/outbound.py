"""Outbound loop: polls Airtable for approved messages and sends them.

Runs every polling interval, picks up messages with Status = "Approved",
sends via the original channel, and marks them as "Sent" or "Failed".
Also computes edit distance between AI draft and approved version.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

import structlog

from sdr.models import AuditAction, AuditLogEntry, MessageStatus

if TYPE_CHECKING:
    from sdr.crm.airtable import AirtableCRM
    from sdr.sending.sender import MessageSender

logger = structlog.get_logger()


def compute_edit_distance(original: str, edited: str) -> float:
    """Compute edit distance as percentage change (0.0 = identical, 1.0 = completely different)."""
    if not original and not edited:
        return 0.0
    if not original or not edited:
        return 1.0
    ratio = SequenceMatcher(None, original, edited).ratio()
    return round(1.0 - ratio, 3)


def process_approved_messages(crm: "AirtableCRM", sender: "MessageSender") -> int:
    """Process all approved messages: send and mark as sent.

    Returns the number of messages successfully sent.
    """
    approved = crm.get_approved_messages()
    if not approved:
        return 0

    logger.info("outbound.found_approved", count=len(approved))
    sent_count = 0

    for msg in approved:
        trace_id = f"out_{msg.id}"
        log = logger.bind(trace_id=trace_id, message_id=msg.id)

        try:
            # Guard: re-check status immediately before sending (prevent double-send)
            current = crm.get_message(msg.id)
            if not current or current.status != MessageStatus.APPROVED:
                log.warning("outbound.status_changed", current_status=current.status if current else None)
                continue

            # Get the draft reply text (user may have edited it)
            reply_text = current.draft_reply
            if not reply_text:
                log.warning("outbound.empty_reply")
                crm.update_message(msg.id, {
                    "Status": MessageStatus.FAILED.value,
                    "Send Error": "Draft reply is empty",
                })
                continue

            # Compute edit distance between AI draft and approved version
            edit_dist = None
            if current.ai_draft_version:
                edit_dist = compute_edit_distance(current.ai_draft_version, reply_text)

            # Determine channel and send
            channel = current.source.value  # "Gmail" or "LinkedIn"
            raw_data = {}
            if current.source_message_id:
                # Try to parse raw_data for thread/chat IDs
                try:
                    # The source_message_id may contain routing info
                    pass
                except Exception:
                    pass

            start_time = time.monotonic()

            if channel == "Gmail":
                # Need recipient email — get from linked contact
                contact = crm.get_contact_for_message(msg.id)
                if not contact or not contact.email:
                    log.error("outbound.no_recipient_email")
                    crm.update_message(msg.id, {
                        "Status": MessageStatus.FAILED.value,
                        "Send Error": "No recipient email found on linked contact",
                    })
                    continue

                sender.send(
                    channel="Gmail",
                    to_email=contact.email,
                    subject=f"Re: {current.subject or ''}".strip(),
                    body=reply_text,
                    thread_id=current.source_message_id,
                )
            elif channel == "LinkedIn":
                # Extract chat_id from raw data or fall back to source_message_id
                linkedin_chat_id = current.source_message_id
                sender.send(
                    channel="LinkedIn",
                    body=reply_text,
                    account_id=current.account_id,
                    chat_id=linkedin_chat_id,
                )

            duration_ms = int((time.monotonic() - start_time) * 1000)

            # Mark as sent
            update_fields = {
                "Status": MessageStatus.SENT.value,
                "Sent At": datetime.utcnow().isoformat(),
            }
            if edit_dist is not None:
                update_fields["Edit Distance"] = edit_dist
            crm.update_message(msg.id, update_fields)

            # Update contact conversation stage to Engaging (if currently New)
            # and track last outbound timestamp for follow-up cadence
            contact = crm.get_contact_for_message(msg.id)
            if contact:
                contact_updates = {"Last Outbound At": datetime.utcnow().strftime("%Y-%m-%d")}
                if contact.conversation_stage.value == "New":
                    contact_updates["Conversation Stage"] = "Engaging"
                crm.update_contact(contact.id, contact_updates)

            # Audit log
            crm.log_audit(AuditLogEntry(
                action=AuditAction.SENT,
                contact_id=contact.id if contact else None,
                message_id=msg.id,
                details=json.dumps({
                    "channel": channel,
                    "edit_distance": edit_dist,
                    "duration_ms": duration_ms,
                }),
            ))

            log.info(
                "outbound.sent",
                channel=channel,
                edit_distance=edit_dist,
                duration_ms=duration_ms,
            )
            sent_count += 1

        except Exception as e:
            log.error("outbound.send_failed", error=str(e))
            try:
                crm.update_message(msg.id, {
                    "Status": MessageStatus.FAILED.value,
                    "Send Error": str(e),
                })
            except Exception:
                log.error("outbound.failed_to_update_status")

    return sent_count
