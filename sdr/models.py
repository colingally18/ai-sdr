"""Pydantic models for the Growlancer SDR system."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- Enums ---

class SourceChannel(str, Enum):
    GMAIL = "Gmail"
    LINKEDIN = "LinkedIn"
    BOTH = "Both"


class MessageDirection(str, Enum):
    INBOUND = "Inbound"
    OUTBOUND = "Outbound"


class LeadCategory(str, Enum):
    HOT = "Hot"
    WARM = "Warm"
    COLD = "Cold"
    NOT_A_LEAD = "Not a Lead"


class ConversationStage(str, Enum):
    NEW = "New"
    ENGAGING = "Engaging"
    QUALIFYING = "Qualifying"
    BOOKING = "Booking"
    FOLLOW_UP = "Follow Up"
    NURTURE = "Nurture"
    CLOSED_WON = "Closed Won"
    CLOSED_LOST = "Closed Lost"


class MessageStatus(str, Enum):
    NEW = "New"
    PROCESSING = "Processing"
    DRAFT_READY = "Draft Ready"
    APPROVED = "Approved"
    REJECTED = "Rejected"
    SENT = "Sent"
    FAILED = "Failed"


class AuditAction(str, Enum):
    MESSAGE_RECEIVED = "message_received"
    CONTACT_CREATED = "contact_created"
    CONTACT_UPDATED = "contact_updated"
    CLASSIFIED = "classified"
    DRAFT_CREATED = "draft_created"
    APPROVED = "approved"
    REJECTED = "rejected"
    SENT = "sent"
    AUTO_ACCEPTED = "auto_accepted"
    AUTO_REJECTED = "auto_rejected"
    ENRICHED = "enriched"
    FOLLOW_UP_CREATED = "follow_up_created"
    FOLLOW_UP_PAUSED = "follow_up_paused"
    FOLLOW_UP_EXHAUSTED = "follow_up_exhausted"
    LEARNING_CYCLE = "learning_cycle"


# --- Inbound Message (normalized from any source) ---

class InboundMessage(BaseModel):
    """Normalized inbound message from any source."""
    source: SourceChannel
    source_message_id: str
    sender_name: str
    sender_email: Optional[str] = None
    sender_linkedin_url: Optional[str] = None
    sender_title: Optional[str] = None
    sender_company: Optional[str] = None
    subject: Optional[str] = None  # Gmail only
    body: str
    thread_context: str = ""  # Full conversation history
    received_at: datetime
    is_connection_request: bool = False
    account_id: Optional[str] = None
    raw_data: Optional[dict] = None


# --- AI Output Models ---

class LeadClassification(BaseModel):
    """Structured output from lead classification."""
    category: LeadCategory
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    detected_intent: str
    detected_signals: list[str] = Field(default_factory=list)
    should_reply: bool
    conversation_stage: ConversationStage
    icp_match_score: float = Field(ge=0.0, le=1.0)


class DraftReply(BaseModel):
    """Output from reply drafting."""
    reply_text: str
    strategy_notes: str = ""  # Internal planning notes (not shown to user)


class ConnectionEvaluation(BaseModel):
    """Output from connection request evaluation."""
    accept: bool
    reasoning: str
    lead_category: LeadCategory = LeadCategory.COLD
    confidence: float = Field(ge=0.0, le=1.0)


# --- Airtable Record Models ---

class ContactRecord(BaseModel):
    """Represents a Contact row in Airtable."""
    id: Optional[str] = None  # Airtable record ID
    name: str
    email: Optional[str] = None
    linkedin_url: Optional[str] = None
    company: Optional[str] = None
    title: Optional[str] = None
    source_channel: SourceChannel = SourceChannel.GMAIL
    lead_category: LeadCategory = LeadCategory.COLD
    conversation_stage: ConversationStage = ConversationStage.NEW
    ai_confidence: float = 0.0
    detected_intent: str = ""
    signal_stack: str = ""
    ai_reasoning: str = ""
    first_contact: Optional[datetime] = None
    last_contact: Optional[datetime] = None
    interaction_count: int = 0
    enriched_data: str = ""
    follow_up_count: int = 0
    follow_up_channel: Optional[str] = None
    next_follow_up_date: Optional[datetime] = None
    follow_up_status: str = ""
    last_outbound_at: Optional[datetime] = None


class MessageRecord(BaseModel):
    """Represents a Message row in Airtable."""
    id: Optional[str] = None  # Airtable record ID
    contact_id: Optional[str] = None  # Linked contact record ID
    source: SourceChannel
    direction: MessageDirection
    subject: Optional[str] = None
    body: str
    thread_context: str = ""
    draft_reply: str = ""
    status: MessageStatus = MessageStatus.NEW
    classification: str = ""
    conversation_stage: str = ""
    ai_draft_version: str = ""
    edit_distance: Optional[float] = None
    received_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    send_error: str = ""
    account_id: str = ""
    source_message_id: str = ""
    follow_up_number: Optional[int] = None


class AuditLogEntry(BaseModel):
    """Represents an Audit Log row in Airtable."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    action: AuditAction
    contact_id: Optional[str] = None
    message_id: Optional[str] = None
    details: str = ""
