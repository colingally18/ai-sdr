"""Gmail API message source implementation."""

from __future__ import annotations

import base64
import email.utils
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from sdr.config import DATA_DIR
from sdr.db import get_source_state, update_source_state
from sdr.models import InboundMessage, SourceChannel
from sdr.sources.base import MessageSource

logger = structlog.get_logger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]

TOKEN_PATH = DATA_DIR / "gmail_token.json"

# Number of messages to fetch on the initial sync (no history id yet).
INITIAL_SYNC_MAX_RESULTS = 25


class GmailSource(MessageSource):
    """Polls a Gmail inbox for new inbound messages via the Gmail API.

    Uses OAuth2 for authentication and the history.list endpoint for
    efficient incremental polling after the first sync.
    """

    def __init__(self, credentials_path: str = "./config/gmail_credentials.json") -> None:
        self._credentials_path = credentials_path
        self._service: Optional[Resource] = None
        self._user_email: Optional[str] = None

    # ------------------------------------------------------------------
    # Authentication helpers
    # ------------------------------------------------------------------

    def _get_credentials(self) -> Credentials:
        """Load or refresh OAuth credentials, running the consent flow if needed."""
        creds: Optional[Credentials] = None

        if TOKEN_PATH.exists():
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

        if creds and creds.expired and creds.refresh_token:
            logger.info("gmail.token_refresh")
            creds.refresh(Request())
        elif not creds or not creds.valid:
            logger.info("gmail.oauth_flow_start")
            flow = InstalledAppFlow.from_client_secrets_file(
                self._credentials_path, SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Persist the token for future runs.
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_PATH.write_text(creds.to_json())
        logger.info("gmail.token_saved", path=str(TOKEN_PATH))

        return creds

    def _build_service(self) -> Resource:
        """Build (and cache) the Gmail API service resource."""
        if self._service is None:
            creds = self._get_credentials()
            self._service = build("gmail", "v1", credentials=creds, cache_discovery=False)
            # Resolve the authenticated user's email so we can filter out sent
            # messages later.
            profile = self._service.users().getProfile(userId="me").execute()
            self._user_email = profile.get("emailAddress", "").lower()
            logger.info("gmail.service_ready", user=self._user_email)
        return self._service

    @property
    def service(self) -> Optional[Resource]:
        """Return the cached Gmail API service (or None if not yet built)."""
        return self._service

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the Gmail API is reachable and authenticated."""
        try:
            service = self._build_service()
            service.users().getProfile(userId="me").execute()
            return True
        except Exception:
            logger.exception("gmail.health_check_failed")
            return False

    def poll(self) -> list[InboundMessage]:
        """Fetch new inbound messages since the last poll.

        On the first call (no stored history id) a broad messages.list is
        used.  Subsequent calls use history.list for efficient incremental
        updates.
        """
        service = self._build_service()

        state = get_source_state("gmail") or {}
        history_id: Optional[str] = state.get("gmail_history_id")

        if history_id:
            message_ids = self._poll_with_history(service, history_id)
        else:
            message_ids = self._poll_initial(service)

        if not message_ids:
            logger.info("gmail.poll.no_new_messages")
            return []

        logger.info("gmail.poll.new_messages", count=len(message_ids))

        messages: list[InboundMessage] = []
        for msg_id in message_ids:
            try:
                inbound = self._process_message(service, msg_id)
                if inbound is not None:
                    messages.append(inbound)
            except Exception:
                logger.exception("gmail.process_message_failed", message_id=msg_id)

        return messages

    # ------------------------------------------------------------------
    # Polling strategies
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _poll_with_history(self, service: Resource, start_history_id: str) -> list[str]:
        """Use history.list for incremental polling."""
        logger.info("gmail.poll.history", start_history_id=start_history_id)

        message_ids: list[str] = []
        request = (
            service.users()
            .history()
            .list(userId="me", startHistoryId=start_history_id, historyTypes=["messageAdded"])
        )

        latest_history_id = start_history_id

        while request is not None:
            response = request.execute()
            latest_history_id = response.get("historyId", latest_history_id)

            for record in response.get("history", []):
                for added in record.get("messagesAdded", []):
                    msg = added.get("message", {})
                    labels = msg.get("labelIds", [])
                    if "INBOX" in labels and "SENT" not in labels:
                        message_ids.append(msg["id"])

            request = (
                service.users()
                .history()
                .list_next(previous_request=request, previous_response=response)
            )

        # Persist the latest history id.
        self._save_history_id(latest_history_id)

        return list(dict.fromkeys(message_ids))  # deduplicate, preserve order

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _poll_initial(self, service: Resource) -> list[str]:
        """Fall back to messages.list for the very first sync (last 24 hours only)."""
        logger.info("gmail.poll.initial_sync")

        # Only fetch messages from the last 24 hours
        from datetime import timedelta
        one_day_ago = datetime.now(tz=timezone.utc) - timedelta(days=1)
        after_epoch = int(one_day_ago.timestamp())

        response = (
            service.users()
            .messages()
            .list(
                userId="me",
                labelIds=["INBOX"],
                q=f"after:{after_epoch}",
                maxResults=INITIAL_SYNC_MAX_RESULTS,
            )
            .execute()
        )

        message_ids = [m["id"] for m in response.get("messages", [])]

        # Seed the history id so subsequent polls use the incremental path.
        if message_ids:
            first_msg = (
                service.users()
                .messages()
                .get(userId="me", id=message_ids[0], format="metadata", metadataHeaders=["From"])
                .execute()
            )
            history_id = first_msg.get("historyId")
            if history_id:
                self._save_history_id(history_id)

        return message_ids

    # ------------------------------------------------------------------
    # Message processing
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _fetch_message(self, service: Resource, message_id: str) -> dict[str, Any]:
        """Fetch a single message in full format."""
        return (
            service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )

    def _process_message(self, service: Resource, message_id: str) -> Optional[InboundMessage]:
        """Fetch, filter, and normalise a single Gmail message."""
        msg = self._fetch_message(service, message_id)

        labels = msg.get("labelIds", [])
        if "SENT" in labels or "INBOX" not in labels:
            return None

        headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}

        from_header = headers.get("from", "")
        sender_name, sender_email = self._parse_from_header(from_header)

        # Skip messages sent by the authenticated user.
        if sender_email and self._user_email and sender_email.lower() == self._user_email:
            return None

        subject = headers.get("subject", "")
        body = self._extract_body(msg.get("payload", {}))

        internal_date_ms = int(msg.get("internalDate", 0))
        received_at = datetime.fromtimestamp(internal_date_ms / 1000, tz=timezone.utc)

        thread_id = msg.get("threadId")
        thread_context = self._build_thread_context(service, thread_id, message_id) if thread_id else ""

        return InboundMessage(
            source=SourceChannel.GMAIL,
            source_message_id=message_id,
            sender_name=sender_name,
            sender_email=sender_email,
            subject=subject,
            body=body,
            thread_context=thread_context,
            received_at=received_at,
            raw_data=msg,
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_from_header(from_header: str) -> tuple[str, Optional[str]]:
        """Extract a display name and email address from a From header value.

        Examples:
            "Jane Doe <jane@example.com>" -> ("Jane Doe", "jane@example.com")
            "jane@example.com"            -> ("jane@example.com", "jane@example.com")
        """
        name, addr = email.utils.parseaddr(from_header)
        if not name:
            name = addr or "Unknown"
        return name, addr or None

    @staticmethod
    def _extract_body(payload: dict[str, Any]) -> str:
        """Extract the text/plain body from a Gmail message payload.

        Walks the MIME parts tree, preferring text/plain.  Falls back to
        text/html if no plain-text part is found.
        """

        def _decode(data: str) -> str:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

        # Single-part message.
        mime_type = payload.get("mimeType", "")
        body_data = payload.get("body", {}).get("data")

        if body_data and mime_type == "text/plain":
            return _decode(body_data)

        # Multi-part: depth-first search for text/plain, fall back to text/html.
        parts = payload.get("parts", [])
        plain: Optional[str] = None
        html: Optional[str] = None

        stack = list(parts)
        while stack:
            part = stack.pop()
            part_mime = part.get("mimeType", "")
            part_data = part.get("body", {}).get("data")

            if part_data and part_mime == "text/plain" and plain is None:
                plain = _decode(part_data)
            elif part_data and part_mime == "text/html" and html is None:
                html = _decode(part_data)

            stack.extend(part.get("parts", []))

        if plain is not None:
            return plain
        if html is not None:
            return html

        # Last resort: return the raw body data if present.
        if body_data:
            return _decode(body_data)

        return ""

    # ------------------------------------------------------------------
    # Thread context
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _fetch_thread(self, service: Resource, thread_id: str) -> dict[str, Any]:
        """Fetch a full thread."""
        return (
            service.users()
            .threads()
            .get(userId="me", id=thread_id, format="full")
            .execute()
        )

    def _build_thread_context(self, service: Resource, thread_id: str, current_message_id: str) -> str:
        """Fetch the full thread and format all messages chronologically.

        Excludes the current message so the context only contains prior
        conversation history.
        """
        try:
            thread = self._fetch_thread(service, thread_id)
        except Exception:
            logger.exception("gmail.thread_fetch_failed", thread_id=thread_id)
            return ""

        messages = thread.get("messages", [])

        # Sort by internalDate ascending (chronological).
        messages.sort(key=lambda m: int(m.get("internalDate", 0)))

        parts: list[str] = []
        for msg in messages:
            if msg.get("id") == current_message_id:
                continue

            headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
            from_header = headers.get("from", "Unknown")
            date_str = headers.get("date", "")
            body = self._extract_body(msg.get("payload", {}))

            parts.append(f"From: {from_header}\nDate: {date_str}\n\n{body.strip()}")

        return "\n---\n".join(parts)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_history_id(self, history_id: str) -> None:
        """Persist the Gmail history id for incremental polling."""
        update_source_state("gmail", gmail_history_id=str(history_id))
        logger.debug("gmail.history_id_saved", history_id=history_id)
