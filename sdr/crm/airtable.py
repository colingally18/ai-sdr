"""
Airtable CRM module for the Growlancer SDR system.

Manages three tables (Contacts, Messages, Audit Log) with automatic schema
creation, built-in request throttling, and full CRUD operations.
"""

from __future__ import annotations

import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional

import structlog
from pyairtable import Api
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from sdr.models import (
    AuditAction,
    AuditLogEntry,
    ContactRecord,
    ConversationStage,
    LeadCategory,
    MessageDirection,
    MessageRecord,
    MessageStatus,
    SourceChannel,
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Rate-limiter – Airtable enforces 5 requests / second per base.
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Simple token-bucket limiter: at most *max_rps* requests per second."""

    def __init__(self, max_rps: int = 5) -> None:
        self._lock = threading.Lock()
        self._min_interval = 1.0 / max_rps
        self._last_call: float = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# Helpers for retryable Airtable calls
# ---------------------------------------------------------------------------

# pyairtable raises requests.exceptions.HTTPError on 4xx/5xx.  We retry on
# transient server errors (429, 500, 502, 503).  Import here so tenacity can
# reference the type without an extra top-level import requirement.
try:
    from requests.exceptions import HTTPError as _HTTPError
except ImportError:  # pragma: no cover
    _HTTPError = Exception  # type: ignore[assignment,misc]

_RETRY_DECORATOR = retry(
    retry=retry_if_exception_type(_HTTPError),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)

# ---------------------------------------------------------------------------
# Field-schema constants
# ---------------------------------------------------------------------------

# Each entry: (field_name, airtable_field_type, options_dict | None)
# "link" type entries are handled separately after all tables exist.

_CONTACTS_FIELDS: list[tuple[str, str, dict | None]] = [
    ("Name", "singleLineText", None),
    ("Email", "email", None),
    ("LinkedIn URL", "url", None),
    ("Company", "singleLineText", None),
    ("Title", "singleLineText", None),
    (
        "Source Channel",
        "singleSelect",
        {"choices": [{"name": v.value} for v in SourceChannel]},
    ),
    (
        "Lead Category",
        "singleSelect",
        {"choices": [{"name": v.value} for v in LeadCategory]},
    ),
    (
        "Conversation Stage",
        "singleSelect",
        {"choices": [{"name": v.value} for v in ConversationStage]},
    ),
    ("AI Confidence", "number", {"precision": 2}),
    ("Detected Intent", "singleLineText", None),
    ("Signal Stack", "multilineText", None),
    ("AI Reasoning", "multilineText", None),
    ("First Contact", "date", {"dateFormat": {"name": "iso"}}),
    ("Last Contact", "date", {"dateFormat": {"name": "iso"}}),
    ("Interaction Count", "number", {"precision": 0}),
    ("Enriched Data", "multilineText", None),
    ("Handled By", "singleLineText", None),
    ("Follow-Up Count", "number", {"precision": 0}),
    (
        "Follow-Up Channel",
        "singleSelect",
        {"choices": [{"name": "LinkedIn"}, {"name": "Email"}]},
    ),
    ("Next Follow-Up Date", "date", {"dateFormat": {"name": "iso"}}),
    (
        "Follow-Up Status",
        "singleSelect",
        {"choices": [{"name": "Active"}, {"name": "Paused"}, {"name": "Exhausted"}]},
    ),
    ("Last Outbound At", "date", {"dateFormat": {"name": "iso"}}),
]

_MESSAGES_FIELDS: list[tuple[str, str, dict | None]] = [
    ("Subject", "singleLineText", None),
    (
        "Source",
        "singleSelect",
        {"choices": [{"name": "Gmail"}, {"name": "LinkedIn"}]},
    ),
    (
        "Direction",
        "singleSelect",
        {"choices": [{"name": v.value} for v in MessageDirection]},
    ),
    ("Body", "multilineText", None),
    ("Thread Context", "multilineText", None),
    ("Draft Reply", "multilineText", None),
    (
        "Status",
        "singleSelect",
        {"choices": [{"name": v.value} for v in MessageStatus]},
    ),
    ("Classification", "singleLineText", None),
    ("Conversation Stage", "singleLineText", None),
    ("AI Draft Version", "multilineText", None),
    ("Edit Distance", "number", {"precision": 4}),
    ("Received At", "date", {"dateFormat": {"name": "iso"}}),
    ("Sent At", "date", {"dateFormat": {"name": "iso"}}),
    ("Send Error", "multilineText", None),
    ("Account ID", "singleLineText", None),
    ("Source Message ID", "singleLineText", None),
    ("Follow-Up Number", "number", {"precision": 0}),
]

_AUDIT_LOG_FIELDS: list[tuple[str, str, dict | None]] = [
    ("Summary", "singleLineText", None),
    ("Timestamp", "date", {"dateFormat": {"name": "iso"}}),
    (
        "Action",
        "singleSelect",
        {"choices": [{"name": v.value} for v in AuditAction]},
    ),
    ("Details", "multilineText", None),
]


# ---------------------------------------------------------------------------
# Main CRM class
# ---------------------------------------------------------------------------


class AirtableCRM:
    """High-level interface to the Growlancer Airtable CRM base."""

    TABLE_CONTACTS = "Contacts"
    TABLE_MESSAGES = "Messages"
    TABLE_AUDIT_LOG = "Audit Log"

    def __init__(self, api_key: str, base_id: str) -> None:
        self._api = Api(api_key)
        self._base_id = base_id
        self._limiter = _RateLimiter(max_rps=5)

        # Table accessors (lazy – set after ensure_schema)
        self._contacts_table = self._api.table(base_id, self.TABLE_CONTACTS)
        self._messages_table = self._api.table(base_id, self.TABLE_MESSAGES)
        self._audit_table = self._api.table(base_id, self.TABLE_AUDIT_LOG)

        # Cache table-id mapping after schema is ensured.
        self._table_ids: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Rate-limited wrappers around pyairtable table operations
    # ------------------------------------------------------------------

    @_RETRY_DECORATOR
    def _create(self, table, fields: dict) -> dict:
        self._limiter.wait()
        return table.create(fields)

    @_RETRY_DECORATOR
    def _update(self, table, record_id: str, fields: dict) -> dict:
        self._limiter.wait()
        return table.update(record_id, fields)

    @_RETRY_DECORATOR
    def _all(self, table, **kwargs) -> list[dict]:
        self._limiter.wait()
        return table.all(**kwargs)

    @_RETRY_DECORATOR
    def _first(self, table, **kwargs) -> dict | None:
        self._limiter.wait()
        return table.first(**kwargs)

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def ensure_schema(self) -> None:
        """Create tables, fields, linked fields, and views if they are missing."""
        base = self._api.base(self._base_id)

        # 1. Fetch current schema -------------------------------------------
        self._limiter.wait()
        schema = base.schema()
        existing_tables: dict[str, dict] = {}  # name -> {id, fields: {name: field}}
        for tbl in schema.tables:
            field_map = {f.name: f for f in tbl.fields}
            existing_tables[tbl.name] = {"id": tbl.id, "fields": field_map}

        # 2. Create tables if missing ---------------------------------------
        tables_to_create: dict[str, list[tuple[str, str, dict | None]]] = {
            self.TABLE_CONTACTS: _CONTACTS_FIELDS,
            self.TABLE_MESSAGES: _MESSAGES_FIELDS,
            self.TABLE_AUDIT_LOG: _AUDIT_LOG_FIELDS,
        }

        for table_name, field_defs in tables_to_create.items():
            if table_name not in existing_tables:
                log.info("airtable.create_table", table=table_name)
                fields_payload = self._build_fields_payload(field_defs)
                self._limiter.wait()
                base.create_table(table_name, fields=fields_payload)
                # Re-fetch schema to get the newly created table's metadata.
                self._limiter.wait()
                refreshed = base.schema()
                for tbl in refreshed.tables:
                    if tbl.name == table_name:
                        field_map = {f.name: f for f in tbl.fields}
                        existing_tables[table_name] = {
                            "id": tbl.id,
                            "fields": field_map,
                        }
                        break
            else:
                # Ensure every expected field exists.
                self._ensure_fields(base, existing_tables[table_name], field_defs, table_name=table_name)

        # Cache table IDs.
        for tbl_name, tbl_info in existing_tables.items():
            self._table_ids[tbl_name] = tbl_info["id"]

        # 3. Linked fields ---------------------------------------------------
        contacts_info = existing_tables[self.TABLE_CONTACTS]
        messages_info = existing_tables[self.TABLE_MESSAGES]
        audit_info = existing_tables[self.TABLE_AUDIT_LOG]

        # Helper: get a table object from a table ID.
        def _tbl(table_id: str):
            return self._api.table(self._base_id, table_id)

        def _safe_create_link(table_id: str, field_name: str, linked_table_id: str, table_label: str) -> None:
            """Create a linked record field, ignoring if it already exists."""
            try:
                log.info("airtable.create_linked_field", field=field_name, table=table_label)
                self._limiter.wait()
                _tbl(table_id).create_field(
                    field_name,
                    "multipleRecordLinks",
                    options={"linkedTableId": linked_table_id},
                )
            except Exception as e:
                if "DUPLICATE" in str(e) or "422" in str(e):
                    log.info("airtable.linked_field_exists", field=field_name, table=table_label)
                else:
                    raise

        # Messages -> Contact (links to Contacts table)
        _safe_create_link(messages_info["id"], "Contact", contacts_info["id"], self.TABLE_MESSAGES)
        # Contacts -> Messages (Airtable may auto-create this; try anyway)
        _safe_create_link(contacts_info["id"], "Messages", messages_info["id"], self.TABLE_CONTACTS)
        # Audit Log -> Contact
        _safe_create_link(audit_info["id"], "Contact", contacts_info["id"], self.TABLE_AUDIT_LOG)
        # Audit Log -> Message
        _safe_create_link(audit_info["id"], "Message", messages_info["id"], self.TABLE_AUDIT_LOG)

        # 4. Views -----------------------------------------------------------
        try:
            self._ensure_views(base, existing_tables)
        except Exception:
            log.warning(
                "airtable.views_skipped",
                hint="pyairtable version does not support view creation; create views manually in Airtable UI",
            )

        log.info("airtable.schema_ensured")

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_fields_payload(
        field_defs: list[tuple[str, str, dict | None]],
    ) -> list[dict]:
        """Convert internal field definitions to the create_table payload."""
        payload: list[dict] = []
        for name, ftype, options in field_defs:
            entry: dict = {"name": name, "type": ftype}
            if options is not None:
                entry["options"] = options
            payload.append(entry)
        return payload

    def _ensure_fields(
        self,
        base,
        table_info: dict,
        field_defs: list[tuple[str, str, dict | None]],
        table_name: str = "",
    ) -> None:
        """Create any fields that are missing from an existing table."""
        existing_fields = table_info["fields"]
        table = self._api.table(self._base_id, table_info["id"])
        for name, ftype, options in field_defs:
            if name not in existing_fields:
                log.info(
                    "airtable.create_field",
                    table_id=table_info["id"],
                    field=name,
                )
                self._limiter.wait()
                table.create_field(name, ftype, options=options)

    def _ensure_views(self, base, existing_tables: dict) -> None:
        """Create standard views if they don't already exist."""

        # Re-fetch schema to get current views.
        self._limiter.wait()
        schema = base.schema()
        table_views: dict[str, set[str]] = {}
        for tbl in schema.tables:
            table_views[tbl.name] = {v.name for v in (tbl.views or [])}

        messages_table_id = existing_tables[self.TABLE_MESSAGES]["id"]
        contacts_table_id = existing_tables[self.TABLE_CONTACTS]["id"]

        views_to_create: list[tuple[str, str, dict]] = [
            # (table_id, view_name, view_config)
            (
                messages_table_id,
                "Pending Approval",
                {
                    "type": "grid",
                    "filterByFormula": '{Status} = "Draft Ready"',
                    "sorts": [{"field": "Received At", "direction": "desc"}],
                },
            ),
            (
                contacts_table_id,
                "Hot Leads",
                {
                    "type": "grid",
                    "filterByFormula": '{Lead Category} = "Hot"',
                },
            ),
            (
                contacts_table_id,
                "Active Conversations",
                {
                    "type": "grid",
                    "filterByFormula": (
                        'AND('
                        '{Conversation Stage} != "Closed Won", '
                        '{Conversation Stage} != "Closed Lost", '
                        '{Conversation Stage} != "New"'
                        ')'
                    ),
                },
            ),
            (
                messages_table_id,
                "Recently Sent",
                {
                    "type": "grid",
                    "filterByFormula": '{Status} = "Sent"',
                },
            ),
            (
                messages_table_id,
                "Rejected / Low Quality",
                {
                    "type": "grid",
                    "filterByFormula": '{Status} = "Rejected"',
                },
            ),
            # Follow-Up Queue: contacts due for follow-up today
            (
                contacts_table_id,
                "Follow-Up Queue",
                {
                    "type": "grid",
                    "filterByFormula": (
                        'AND('
                        '{Follow-Up Status} = "Active", '
                        'IS_BEFORE({Next Follow-Up Date}, DATEADD(TODAY(), 1, "days"))'
                        ')'
                    ),
                    "sorts": [{"field": "Next Follow-Up Date", "direction": "asc"}],
                },
            ),
            # Pipeline: all active leads
            (
                contacts_table_id,
                "Pipeline",
                {
                    "type": "grid",
                    "filterByFormula": (
                        'AND('
                        '{Lead Category} != "Not a Lead", '
                        '{Conversation Stage} != "Closed Lost"'
                        ')'
                    ),
                    "sorts": [{"field": "Last Contact", "direction": "desc"}],
                },
            ),
            # Follow-Up Drafts: follow-up messages pending review
            (
                messages_table_id,
                "Follow-Up Drafts",
                {
                    "type": "grid",
                    "filterByFormula": (
                        'AND('
                        '{Status} = "Draft Ready", '
                        '{Follow-Up Number} > 0'
                        ')'
                    ),
                    "sorts": [{"field": "Follow-Up Number", "direction": "asc"}],
                },
            ),
        ]

        # Map table_id back to table_name for view lookup.
        id_to_name: dict[str, str] = {}
        for tbl_name, tbl_info in existing_tables.items():
            id_to_name[tbl_info["id"]] = tbl_name

        for table_id, view_name, view_config in views_to_create:
            tbl_name = id_to_name.get(table_id, table_id)
            current_views = table_views.get(tbl_name, set())
            if view_name not in current_views:
                log.info(
                    "airtable.create_view",
                    table=tbl_name,
                    view=view_name,
                )
                try:
                    self._limiter.wait()
                    base.create_view(table_id, view_name, **view_config)
                except Exception:
                    # The Airtable REST API has limited view-creation support.
                    # If it fails we log a warning and continue; the operator
                    # can create views manually in the UI.
                    log.warning(
                        "airtable.create_view_failed",
                        table=tbl_name,
                        view=view_name,
                        exc_info=True,
                    )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _date_str(dt: datetime | None) -> str | None:
        if dt is None:
            return None
        return dt.strftime("%Y-%m-%d")

    @staticmethod
    def _datetime_str(dt: datetime | None) -> str | None:
        if dt is None:
            return None
        return dt.isoformat()

    @staticmethod
    def _parse_date(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Contact CRUD
    # ------------------------------------------------------------------

    def _contact_to_fields(self, contact: ContactRecord) -> dict:
        fields: dict = {
            "Name": contact.name,
            "Source Channel": contact.source_channel.value,
            "Lead Category": contact.lead_category.value,
            "Conversation Stage": contact.conversation_stage.value,
            "AI Confidence": contact.ai_confidence,
            "Detected Intent": contact.detected_intent,
            "Signal Stack": contact.signal_stack,
            "AI Reasoning": contact.ai_reasoning,
            "Interaction Count": contact.interaction_count,
            "Enriched Data": contact.enriched_data,
        }
        if contact.email:
            fields["Email"] = contact.email
        if contact.linkedin_url:
            fields["LinkedIn URL"] = contact.linkedin_url
        if contact.company:
            fields["Company"] = contact.company
        if contact.title:
            fields["Title"] = contact.title
        if contact.first_contact:
            fields["First Contact"] = self._date_str(contact.first_contact)
        if contact.last_contact:
            fields["Last Contact"] = self._date_str(contact.last_contact)
        if contact.follow_up_count:
            fields["Follow-Up Count"] = contact.follow_up_count
        if contact.follow_up_channel:
            fields["Follow-Up Channel"] = contact.follow_up_channel
        if contact.next_follow_up_date:
            fields["Next Follow-Up Date"] = self._date_str(contact.next_follow_up_date)
        if contact.follow_up_status:
            fields["Follow-Up Status"] = contact.follow_up_status
        if contact.last_outbound_at:
            fields["Last Outbound At"] = self._date_str(contact.last_outbound_at)
        return fields

    def _record_to_contact(self, record: dict) -> ContactRecord:
        f = record["fields"]
        return ContactRecord(
            id=record["id"],
            name=f.get("Name", ""),
            email=f.get("Email"),
            linkedin_url=f.get("LinkedIn URL"),
            company=f.get("Company"),
            title=f.get("Title"),
            source_channel=SourceChannel(f["Source Channel"]) if f.get("Source Channel") else SourceChannel.GMAIL,
            lead_category=LeadCategory(f["Lead Category"]) if f.get("Lead Category") else LeadCategory.COLD,
            conversation_stage=ConversationStage(f["Conversation Stage"]) if f.get("Conversation Stage") else ConversationStage.NEW,
            ai_confidence=f.get("AI Confidence", 0.0),
            detected_intent=f.get("Detected Intent", ""),
            signal_stack=f.get("Signal Stack", ""),
            ai_reasoning=f.get("AI Reasoning", ""),
            first_contact=self._parse_date(f.get("First Contact")),
            last_contact=self._parse_date(f.get("Last Contact")),
            interaction_count=f.get("Interaction Count", 0),
            enriched_data=f.get("Enriched Data", ""),
            follow_up_count=f.get("Follow-Up Count", 0) or 0,
            follow_up_channel=f.get("Follow-Up Channel"),
            next_follow_up_date=self._parse_date(f.get("Next Follow-Up Date")),
            follow_up_status=f.get("Follow-Up Status", "") or "",
            last_outbound_at=self._parse_date(f.get("Last Outbound At")),
        )

    def upsert_contact(self, contact: ContactRecord) -> ContactRecord:
        """Create or update a contact record.

        If the contact has an ``id``, the existing record is updated.
        Otherwise a lookup is attempted by email and then by LinkedIn URL.
        If no existing record is found, a new one is created.

        Returns the ``ContactRecord`` with its Airtable ``id`` set.
        """
        fields = self._contact_to_fields(contact)

        # Update path – record already has an id.
        if contact.id:
            log.info("airtable.update_contact", record_id=contact.id)
            record = self._update(self._contacts_table, contact.id, fields)
            return self._record_to_contact(record)

        # Try to find existing record by email or LinkedIn URL.
        existing: dict | None = None
        if contact.email:
            existing = self._first(
                self._contacts_table,
                formula=f'{{Email}} = "{contact.email}"',
            )
        if existing is None and contact.linkedin_url:
            existing = self._first(
                self._contacts_table,
                formula=f'{{LinkedIn URL}} = "{contact.linkedin_url}"',
            )

        if existing:
            log.info("airtable.update_contact", record_id=existing["id"])
            record = self._update(self._contacts_table, existing["id"], fields)
            return self._record_to_contact(record)

        # Create new contact.
        log.info("airtable.create_contact", name=contact.name)
        record = self._create(self._contacts_table, fields)
        return self._record_to_contact(record)

    def find_contact_by_email(self, email: str) -> Optional[ContactRecord]:
        """Look up a contact by email address. Returns ``None`` if not found."""
        record = self._first(
            self._contacts_table,
            formula=f'{{Email}} = "{email}"',
        )
        if record is None:
            return None
        return self._record_to_contact(record)

    def find_contact_by_linkedin_url(self, url: str) -> Optional[ContactRecord]:
        """Look up a contact by LinkedIn profile URL."""
        record = self._first(
            self._contacts_table,
            formula=f'{{LinkedIn URL}} = "{url}"',
        )
        if record is None:
            return None
        return self._record_to_contact(record)

    def find_contacts_by_name(self, name: str) -> list[ContactRecord]:
        """Find contacts whose Name field contains *name* (case-insensitive)."""
        # FIND() is case-sensitive in Airtable; use LOWER for insensitivity.
        safe_name = name.replace('"', '\\"')
        formula = f'FIND(LOWER("{safe_name}"), LOWER({{Name}})) > 0'
        records = self._all(self._contacts_table, formula=formula)
        return [self._record_to_contact(r) for r in records]

    # ------------------------------------------------------------------
    # Message CRUD
    # ------------------------------------------------------------------

    def _message_to_fields(self, message: MessageRecord) -> dict:
        fields: dict = {
            "Source": message.source.value,
            "Direction": message.direction.value,
            "Body": message.body,
            "Thread Context": message.thread_context,
            "Draft Reply": message.draft_reply,
            "Status": message.status.value,
            "Classification": message.classification,
            "Conversation Stage": message.conversation_stage,
            "AI Draft Version": message.ai_draft_version,
            "Send Error": message.send_error,
            "Account ID": message.account_id,
            "Source Message ID": message.source_message_id,
        }
        if message.subject:
            fields["Subject"] = message.subject
        if message.edit_distance is not None:
            fields["Edit Distance"] = message.edit_distance
        if message.received_at:
            fields["Received At"] = self._date_str(message.received_at)
        if message.sent_at:
            fields["Sent At"] = self._date_str(message.sent_at)
        if message.follow_up_number is not None:
            fields["Follow-Up Number"] = message.follow_up_number
        if message.contact_id:
            fields["Contact"] = [message.contact_id]
        return fields

    def _record_to_message(self, record: dict) -> MessageRecord:
        f = record["fields"]
        contact_links = f.get("Contact")
        contact_id = contact_links[0] if contact_links else None
        return MessageRecord(
            id=record["id"],
            contact_id=contact_id,
            source=SourceChannel(f["Source"]) if f.get("Source") else SourceChannel.GMAIL,
            direction=MessageDirection(f["Direction"]) if f.get("Direction") else MessageDirection.INBOUND,
            subject=f.get("Subject"),
            body=f.get("Body", ""),
            thread_context=f.get("Thread Context", ""),
            draft_reply=f.get("Draft Reply", ""),
            status=MessageStatus(f["Status"]) if f.get("Status") else MessageStatus.NEW,
            classification=f.get("Classification", ""),
            conversation_stage=f.get("Conversation Stage", ""),
            ai_draft_version=f.get("AI Draft Version", ""),
            edit_distance=f.get("Edit Distance"),
            received_at=self._parse_date(f.get("Received At")),
            sent_at=self._parse_date(f.get("Sent At")),
            send_error=f.get("Send Error", ""),
            account_id=f.get("Account ID", ""),
            source_message_id=f.get("Source Message ID", ""),
            follow_up_number=f.get("Follow-Up Number"),
        )

    def find_message_by_source_id(self, source_message_id: str) -> Optional[MessageRecord]:
        """Look up a message by its source message ID. Returns ``None`` if not found."""
        record = self._first(
            self._messages_table,
            formula='{Source Message ID} = "' + source_message_id + '"',
        )
        if record is None:
            return None
        return self._record_to_message(record)

    def create_message(self, message: MessageRecord) -> MessageRecord:
        """Insert a new message record and return it with its ``id`` set.

        If an *inbound* message with the same Source Message ID already exists
        in Airtable, returns the existing record instead of creating a duplicate.
        Outbound messages (e.g. follow-ups) skip the dedup check because multiple
        outbound messages may legitimately share the same thread/chat ID.
        """
        # Dedup: check if inbound message already exists in Airtable
        if message.source_message_id and message.direction == MessageDirection.INBOUND:
            existing = self.find_message_by_source_id(message.source_message_id)
            if existing:
                log.info(
                    "airtable.message_already_exists",
                    source_message_id=message.source_message_id,
                    record_id=existing.id,
                )
                return existing

        fields = self._message_to_fields(message)
        log.info(
            "airtable.create_message",
            source=message.source.value,
            direction=message.direction.value,
        )
        record = self._create(self._messages_table, fields)
        return self._record_to_message(record)

    def update_message(self, record_id: str, fields: dict) -> None:
        """Update arbitrary fields on an existing message record.

        ``fields`` should use Airtable field names as keys (e.g.
        ``{"Status": "Approved", "Draft Reply": "..."}``).
        """
        log.info("airtable.update_message", record_id=record_id, fields=list(fields.keys()))
        self._update(self._messages_table, record_id, fields)

    def get_message(self, record_id: str) -> Optional[MessageRecord]:
        """Fetch a single message record by its Airtable record ID."""
        try:
            self._limiter.wait()
            record = self._messages_table.get(record_id)
            return self._record_to_message(record)
        except Exception:
            log.warning("airtable.get_message_failed", record_id=record_id, exc_info=True)
            return None

    def get_approved_messages(self) -> list[MessageRecord]:
        """Return all messages with Status = "Approved"."""
        records = self._all(
            self._messages_table,
            formula='{Status} = "Approved"',
        )
        return [self._record_to_message(r) for r in records]

    def get_contact_for_message(self, message_record_id: str) -> Optional[ContactRecord]:
        """Get the linked contact for a message record."""
        msg = self.get_message(message_record_id)
        if msg and msg.contact_id:
            try:
                self._limiter.wait()
                record = self._contacts_table.get(msg.contact_id)
                return self._record_to_contact(record)
            except Exception:
                log.warning("airtable.get_contact_for_message_failed", message_id=message_record_id, exc_info=True)
        return None

    # ------------------------------------------------------------------
    # Follow-up queries
    # ------------------------------------------------------------------

    def get_contacts_for_followup(self) -> list[ContactRecord]:
        """Return contacts with active cadence and Next Follow-Up Date <= today."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        formula = (
            'AND('
            '{Follow-Up Status} = "Active", '
            f'IS_BEFORE({{Next Follow-Up Date}}, DATEADD(TODAY(), 1, "days"))'
            ')'
        )
        records = self._all(self._contacts_table, formula=formula)
        return [self._record_to_contact(r) for r in records]

    def get_stale_contacts(self, days_stale: int = 3) -> list[ContactRecord]:
        """Return contacts ready to enter follow-up cadence.

        Criteria: stage in (Engaging, Qualifying, Follow Up), no Follow-Up Status set,
        Last Outbound At exists and is older than days_stale, Lead Category != Not a Lead.
        """
        formula = (
            'AND('
            'OR('
            '{Conversation Stage} = "Engaging", '
            '{Conversation Stage} = "Qualifying", '
            '{Conversation Stage} = "Follow Up"'
            '), '
            '{Follow-Up Status} = "", '
            '{Last Outbound At} != "", '
            f'IS_BEFORE({{Last Outbound At}}, DATEADD(TODAY(), -{days_stale}, "days")), '
            '{Lead Category} != "Not a Lead"'
            ')'
        )
        records = self._all(self._contacts_table, formula=formula)
        return [self._record_to_contact(r) for r in records]

    def get_messages_for_contact(
        self, contact_id: str, direction: Optional[str] = None
    ) -> list[MessageRecord]:
        """Return all messages linked to a contact, optionally filtered by direction."""
        if direction:
            formula = f'AND(FIND("{contact_id}", ARRAYJOIN({{Contact}})), {{Direction}} = "{direction}")'
        else:
            formula = f'FIND("{contact_id}", ARRAYJOIN({{Contact}}))'
        records = self._all(self._messages_table, formula=formula)
        return [self._record_to_message(r) for r in records]

    def get_contact(self, record_id: str) -> Optional[ContactRecord]:
        """Fetch a single contact by its Airtable record ID."""
        try:
            self._limiter.wait()
            record = self._contacts_table.get(record_id)
            return self._record_to_contact(record)
        except Exception:
            log.warning("airtable.get_contact_failed", record_id=record_id, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Contact updates
    # ------------------------------------------------------------------

    def update_contact(self, record_id: str, fields: dict) -> None:
        """Update arbitrary fields on an existing contact record.

        ``fields`` should use Airtable field names as keys.
        """
        log.info("airtable.update_contact_fields", record_id=record_id, fields=list(fields.keys()))
        self._update(self._contacts_table, record_id, fields)

    # ------------------------------------------------------------------
    # Audit Log
    # ------------------------------------------------------------------

    def log_audit(self, entry: AuditLogEntry) -> None:
        """Write an entry to the Audit Log table."""
        fields: dict = {
            "Summary": f"{entry.action.value} — {entry.timestamp.strftime('%Y-%m-%d %H:%M')}",
            "Timestamp": self._date_str(entry.timestamp),
            "Action": entry.action.value,
            "Details": entry.details,
        }
        if entry.contact_id:
            fields["Contact"] = [entry.contact_id]
        if entry.message_id:
            fields["Message"] = [entry.message_id]

        log.info("airtable.log_audit", action=entry.action.value)
        self._create(self._audit_table, fields)
