"""Follow-up cadence engine.

Runs daily to:
1. Activate cadences for leads who've gone stale (no reply after outbound).
2. Draft personalized follow-up messages for contacts with due follow-ups.
3. Auto-approve messages when the team consistently sends AI drafts unchanged.
4. Close out leads after the final follow-up.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional

import anthropic
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sdr.ai.prompts import build_followup_prompt
from sdr.models import (
    AuditAction,
    AuditLogEntry,
    MessageDirection,
    MessageRecord,
    MessageStatus,
    SourceChannel,
)

if TYPE_CHECKING:
    from sdr.config import FollowUpConfig
    from sdr.crm.airtable import AirtableCRM

logger = structlog.get_logger(__name__)


def run_followup_cycle(
    crm: "AirtableCRM",
    api_key: str,
    config: "FollowUpConfig",
) -> dict:
    """Main entry point for the follow-up cycle.

    Returns stats: {initialized, drafted, auto_approved, paused, exhausted, skipped}.
    """
    logger.info("followup.cycle_start")

    # Step 1: Activate cadences for newly stale leads
    initialized = _activate_stale_leads(crm, config)

    # Step 2: Process due follow-ups
    stats = _process_due_followups(crm, api_key, config)
    stats["initialized"] = initialized

    logger.info("followup.cycle_complete", **stats)
    return stats


def _activate_stale_leads(crm: "AirtableCRM", config: "FollowUpConfig") -> int:
    """Find contacts that should enter the cadence and activate them."""
    stale = crm.get_stale_contacts(days_stale=config.days_before_activation)
    activated = 0

    for contact in stale:
        # Check for recent inbound — if they replied, skip
        if _has_recent_inbound(crm, contact.id, contact.last_outbound_at):
            continue

        # Determine initial channel
        channel = "LinkedIn" if contact.linkedin_url else "Email"

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        crm.update_contact(contact.id, {
            "Follow-Up Status": "Active",
            "Follow-Up Count": 0,
            "Next Follow-Up Date": today,
            "Follow-Up Channel": channel,
        })
        activated += 1
        logger.info(
            "followup.cadence_activated",
            contact_id=contact.id,
            name=contact.name,
            channel=channel,
        )

    return activated


def _process_due_followups(
    crm: "AirtableCRM",
    api_key: str,
    config: "FollowUpConfig",
) -> dict:
    """Draft follow-ups for all contacts with due dates."""
    contacts = crm.get_contacts_for_followup()
    stats = {
        "drafted": 0,
        "auto_approved": 0,
        "paused": 0,
        "exhausted": 0,
        "skipped": 0,
    }

    for contact in contacts:
        try:
            # Reply check: inbound since last outbound?
            if contact.last_outbound_at and _has_recent_inbound(
                crm, contact.id, contact.last_outbound_at
            ):
                crm.update_contact(contact.id, {"Follow-Up Status": "Paused"})
                crm.log_audit(AuditLogEntry(
                    action=AuditAction.FOLLOW_UP_PAUSED,
                    contact_id=contact.id,
                    details=json.dumps({"reason": "inbound_received"}),
                ))
                stats["paused"] += 1
                continue

            # Duplicate check: pending outbound already exists?
            if _has_pending_outbound(crm, contact.id):
                stats["skipped"] += 1
                continue

            # Channel logic
            followup_count = contact.follow_up_count or 0
            channel = _determine_channel(contact, config)
            if not channel:
                stats["skipped"] += 1
                continue

            # Get routing info
            routing = _get_routing_info(crm, contact, channel)

            # Get conversation history
            all_messages = crm.get_messages_for_contact(contact.id)
            history = _format_conversation_history(all_messages)

            followup_num = followup_count + 1

            # Draft via Claude
            reply_text = _draft_followup_message(
                api_key=api_key,
                contact=contact,
                channel=channel,
                history=history,
                followup_num=followup_num,
                config=config,
            )

            # Auto-approve check
            auto_approve = _should_auto_approve(
                crm, contact.id, threshold=config.auto_approve_threshold
            )
            status = MessageStatus.APPROVED if auto_approve else MessageStatus.DRAFT_READY

            # Create outbound message
            source = SourceChannel.LINKEDIN if channel == "LinkedIn" else SourceChannel.GMAIL
            msg = MessageRecord(
                contact_id=contact.id,
                source=source,
                direction=MessageDirection.OUTBOUND,
                body="",
                draft_reply=reply_text,
                ai_draft_version=reply_text,
                status=status,
                account_id=routing.get("account_id", ""),
                source_message_id=routing.get("chat_id", routing.get("thread_id", "")),
                follow_up_number=followup_num,
            )
            created_msg = crm.create_message(msg)

            # Update contact
            next_date = (
                datetime.now(timezone.utc) + timedelta(days=config.days_between)
            ).strftime("%Y-%m-%d")
            update_fields = {
                "Follow-Up Count": followup_num,
                "Next Follow-Up Date": next_date,
                "Follow-Up Channel": channel,
            }

            # Check if cadence is exhausted
            if followup_num >= config.total_followups:
                update_fields["Follow-Up Status"] = "Exhausted"
                update_fields["Conversation Stage"] = "Closed Lost"
                crm.log_audit(AuditLogEntry(
                    action=AuditAction.FOLLOW_UP_EXHAUSTED,
                    contact_id=contact.id,
                    details=json.dumps({
                        "total_followups": followup_num,
                    }),
                ))
                stats["exhausted"] += 1
            crm.update_contact(contact.id, update_fields)

            # Audit
            crm.log_audit(AuditLogEntry(
                action=AuditAction.FOLLOW_UP_CREATED,
                contact_id=contact.id,
                message_id=created_msg.id,
                details=json.dumps({
                    "followup_number": followup_num,
                    "channel": channel,
                    "auto_approved": auto_approve,
                }),
            ))

            if auto_approve:
                stats["auto_approved"] += 1
            else:
                stats["drafted"] += 1

            logger.info(
                "followup.message_created",
                contact_id=contact.id,
                name=contact.name,
                followup_num=followup_num,
                channel=channel,
                auto_approved=auto_approve,
            )

        except Exception:
            logger.error(
                "followup.process_contact_failed",
                contact_id=contact.id,
                exc_info=True,
            )
            stats["skipped"] += 1

    return stats


def _has_recent_inbound(
    crm: "AirtableCRM", contact_id: str, since_date: Optional[datetime]
) -> bool:
    """Check if contact has sent us a message since the given date."""
    if not since_date:
        return False
    messages = crm.get_messages_for_contact(contact_id, direction="Inbound")
    for msg in messages:
        if msg.received_at and msg.received_at >= since_date:
            return True
    return False


def _has_pending_outbound(crm: "AirtableCRM", contact_id: str) -> bool:
    """Check for existing draft/approved outbound for this contact."""
    messages = crm.get_messages_for_contact(contact_id, direction="Outbound")
    for msg in messages:
        if msg.status in (MessageStatus.DRAFT_READY, MessageStatus.APPROVED):
            return True
    return False


def _determine_channel(contact, config: "FollowUpConfig") -> Optional[str]:
    """LinkedIn if count < linkedin_followups, email if >=, with fallbacks."""
    count = contact.follow_up_count or 0
    if count < config.linkedin_followups:
        # Prefer LinkedIn, fall back to email
        if contact.linkedin_url:
            return "LinkedIn"
        if contact.email:
            return "Email"
        return None
    else:
        # Prefer email, fall back to LinkedIn
        if contact.email:
            return "Email"
        if contact.linkedin_url:
            return "LinkedIn"
        return None


def _get_routing_info(crm: "AirtableCRM", contact, channel: str) -> dict:
    """Get chat_id/account_id for LinkedIn or thread_id for email."""
    routing: dict = {}

    # Find the most recent message for this contact on the chosen channel
    messages = crm.get_messages_for_contact(contact.id)
    source_value = "LinkedIn" if channel == "LinkedIn" else "Gmail"
    channel_messages = [
        m for m in messages
        if m.source.value == source_value and m.source_message_id
    ]

    if channel_messages:
        # Sort by most recent (sent_at or received_at)
        channel_messages.sort(
            key=lambda m: m.sent_at or m.received_at or datetime.min,
            reverse=True,
        )
        latest = channel_messages[0]
        if channel == "LinkedIn":
            routing["chat_id"] = latest.source_message_id
            routing["account_id"] = latest.account_id
        else:
            routing["thread_id"] = latest.source_message_id

    return routing


def _should_auto_approve(
    crm: "AirtableCRM", contact_id: str, threshold: int = 2
) -> bool:
    """True if last N sent messages had edit_distance = 0."""
    messages = crm.get_messages_for_contact(contact_id, direction="Outbound")
    sent = [m for m in messages if m.status == MessageStatus.SENT and m.edit_distance is not None]
    if len(sent) < threshold:
        return False
    # Sort by sent_at descending
    sent.sort(key=lambda m: m.sent_at or datetime.min, reverse=True)
    recent = sent[:threshold]
    return all(m.edit_distance == 0.0 for m in recent)


@retry(
    retry=retry_if_exception_type(
        (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError)
    ),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    before_sleep=lambda retry_state: structlog.get_logger(__name__).warning(
        "followup_draft_retry",
        attempt=retry_state.attempt_number,
    ),
)
def _draft_followup_message(
    api_key: str,
    contact,
    channel: str,
    history: str,
    followup_num: int,
    config: "FollowUpConfig",
) -> str:
    """Single Claude call using draft_followup.txt. Returns reply text."""
    prompt = build_followup_prompt(
        contact=contact,
        channel=channel,
        conversation_history=history,
        followup_number=followup_num,
    )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=config.model,
        max_tokens=512,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = ""
    for block in response.content:
        if block.type == "text":
            raw_text += block.text

    return raw_text.strip()


def _format_conversation_history(messages: list[MessageRecord]) -> str:
    """Format a list of messages into a readable conversation history."""
    if not messages:
        return "No prior messages"

    # Sort chronologically
    sorted_msgs = sorted(
        messages,
        key=lambda m: m.received_at or m.sent_at or datetime.min,
    )

    lines = []
    for msg in sorted_msgs:
        direction = msg.direction.value
        channel = msg.source.value
        timestamp = (msg.sent_at or msg.received_at or datetime.min).strftime("%Y-%m-%d")
        # Use draft_reply for outbound (what was sent), body for inbound
        text = msg.draft_reply if msg.direction.value == "Outbound" and msg.draft_reply else msg.body
        lines.append(f"[{timestamp}] {direction} ({channel}): {text}")

    return "\n\n".join(lines)
