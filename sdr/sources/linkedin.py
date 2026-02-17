"""LinkedIn message source via Unipile REST API.

Polls for new DMs, connection requests, group messages, and comments.
Normalizes everything to InboundMessage format.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

import requests
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from sdr import db
from sdr.models import InboundMessage, SourceChannel
from sdr.sources.base import MessageSource

logger = structlog.get_logger()


class LinkedInSource(MessageSource):
    """Polls LinkedIn messages via Unipile REST API."""

    def __init__(self, dsn: str, api_key: str):
        self.dsn = dsn
        self.api_key = api_key
        self.base_url = f"https://{dsn}/api/v1"
        self.headers = {
            "X-API-KEY": api_key,
            "accept": "application/json",
        }

    def is_available(self) -> bool:
        """Check if Unipile API is reachable."""
        try:
            resp = requests.get(
                f"{self.base_url}/accounts",
                headers=self.headers,
                timeout=10,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.warning("linkedin.health_check_failed", error=str(e))
            return False

    def fetch_accounts(self) -> list[dict]:
        """Fetch all connected LinkedIn accounts from Unipile."""
        try:
            resp = requests.get(
                f"{self.base_url}/accounts",
                headers=self.headers,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            accounts = data.get("items", data.get("data", []))
            logger.info("linkedin.accounts_fetched", count=len(accounts))
            return accounts
        except Exception as e:
            logger.error("linkedin.fetch_accounts_failed", error=str(e))
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def poll(self) -> list[InboundMessage]:
        """Fetch new LinkedIn messages since last poll.

        Polls per-account so each account has its own cursor.
        """
        messages = []

        # Fetch all connected accounts
        accounts = self.fetch_accounts()
        if not accounts:
            # Fallback: poll without account filter using global cursor
            return self._poll_account(account_id=None)

        for account in accounts:
            account_id = account.get("id", "")
            if not account_id:
                continue
            try:
                account_msgs = self._poll_account(account_id=account_id)
                messages.extend(account_msgs)
            except Exception as e:
                logger.error(
                    "linkedin.poll_account_failed",
                    account_id=account_id,
                    error=str(e),
                )

        logger.info("linkedin.poll_complete", message_count=len(messages))
        return messages

    def _poll_account(self, account_id: Optional[str] = None) -> list[InboundMessage]:
        """Poll chats for a specific account (or all if account_id is None)."""
        messages = []

        # Per-account cursor tracking
        state_key = f"linkedin_{account_id}" if account_id else "linkedin"
        state = db.get_source_state(state_key)
        cursor = state.get("cursor") if state else None

        try:
            params: dict = {"limit": 50}
            if cursor:
                params["cursor"] = cursor
            if account_id:
                params["account_id"] = account_id

            resp = requests.get(
                f"{self.base_url}/chats",
                headers=self.headers,
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            chats = data.get("items", data.get("data", []))
            new_cursor = data.get("cursor", data.get("next_cursor"))

            for chat in chats:
                chat_messages = self._fetch_chat_messages(chat, account_id)
                messages.extend(chat_messages)

            # Update cursor for next poll
            if new_cursor:
                db.update_source_state(state_key, cursor=new_cursor)
            else:
                db.update_source_state(state_key)

        except requests.exceptions.HTTPError as e:
            logger.error(
                "linkedin.poll_failed",
                account_id=account_id,
                status=e.response.status_code,
                error=str(e),
            )
            raise
        except Exception as e:
            logger.error("linkedin.poll_failed", account_id=account_id, error=str(e))
            raise

        return messages

    def _build_attendee_map(self, chat: dict) -> dict[str, dict]:
        """Build a mapping from attendee ID to attendee info from a chat object.

        Unipile stores attendee names on the chat object's `attendees` array,
        not on individual messages. Each attendee has an `id` and name fields.
        """
        attendee_map: dict[str, dict] = {}
        attendees = chat.get("attendees", [])
        for att in attendees:
            att_id = att.get("id", "")
            if att_id:
                attendee_map[att_id] = att
        return attendee_map

    def _resolve_sender_name(self, sender_id: str, attendee_map: dict[str, dict], msg: dict) -> str:
        """Resolve a sender name from attendee map with fallback chain.

        Priority: attendee display_name → name → first_name + last_name → sender dict name → "Unknown"
        """
        att = attendee_map.get(sender_id, {})

        # Try attendee fields in priority order
        if att.get("display_name"):
            return att["display_name"]
        if att.get("name"):
            return att["name"]
        first = att.get("first_name", "")
        last = att.get("last_name", "")
        if first or last:
            return f"{first} {last}".strip()

        # Fallback to sender object on message (if Unipile provides one)
        sender = msg.get("sender", {})
        if isinstance(sender, dict):
            if sender.get("name"):
                return sender["name"]
            if sender.get("display_name"):
                return sender["display_name"]

        return "Unknown"

    def _resolve_sender_info(self, sender_id: str, attendee_map: dict[str, dict], msg: dict) -> dict:
        """Extract full sender info from attendee map and message."""
        att = attendee_map.get(sender_id, {})
        sender = msg.get("sender", {}) if isinstance(msg.get("sender"), dict) else {}

        name = self._resolve_sender_name(sender_id, attendee_map, msg)

        # LinkedIn URL: try attendee first, then sender
        linkedin_url = (
            att.get("profile_url")
            or att.get("linkedin_url")
            or sender.get("profile_url")
            or sender.get("linkedin_url")
            or ""
        )

        # Email
        email = att.get("email") or sender.get("email")

        # Headline
        headline = att.get("headline") or sender.get("headline") or ""

        return {
            "name": name,
            "linkedin_url": linkedin_url,
            "email": email,
            "headline": headline,
        }

    def _fetch_chat_messages(self, chat: dict, account_id: Optional[str] = None) -> list[InboundMessage]:
        """Fetch messages from a specific chat and normalize them."""
        messages = []
        chat_id = chat.get("id", "")
        chat_account_id = account_id or chat.get("account_id", "")

        # Build attendee map from chat object
        attendee_map = self._build_attendee_map(chat)

        try:
            resp = requests.get(
                f"{self.base_url}/chats/{chat_id}/messages",
                headers=self.headers,
                params={"limit": 10},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            chat_messages = data.get("items", data.get("data", []))

            # Build thread context from all messages using attendee map
            thread_parts = []
            for msg in reversed(chat_messages):  # Oldest first
                sender_id = msg.get("sender_id", "")
                sender_name = self._resolve_sender_name(sender_id, attendee_map, msg)
                body = msg.get("text", msg.get("body", ""))
                thread_parts.append(f"{sender_name}: {body}")
            thread_context = "\n---\n".join(thread_parts)

            # Only process messages we haven't seen (check against processed_messages)
            for msg in chat_messages:
                msg_id = msg.get("id", "")
                if not msg_id:
                    continue

                # Skip already processed
                if db.is_message_processed("LinkedIn", msg_id):
                    continue

                # Skip outbound messages (sent by us)
                if msg.get("is_sender", False) or msg.get("direction") == "outbound":
                    continue

                sender_id = msg.get("sender_id", "")
                sender_info = self._resolve_sender_info(sender_id, attendee_map, msg)
                normalized = self._normalize_message(
                    msg, sender_info, chat_id, thread_context, chat_account_id,
                )
                if normalized:
                    messages.append(normalized)

        except Exception as e:
            logger.warning("linkedin.fetch_chat_failed", chat_id=chat_id, error=str(e))

        return messages

    def _normalize_message(
        self,
        msg: dict,
        sender_info: dict,
        chat_id: str,
        thread_context: str,
        account_id: str = "",
    ) -> Optional[InboundMessage]:
        """Normalize a Unipile message to InboundMessage."""
        msg_id = msg.get("id", "")
        body = msg.get("text", msg.get("body", ""))
        if not body:
            return None

        sender_name = sender_info.get("name", "Unknown")
        sender_linkedin_url = sender_info.get("linkedin_url", "")
        sender_email = sender_info.get("email")

        # Parse headline for title/company
        headline = sender_info.get("headline", "")
        title, company = self._parse_headline(headline)

        # Parse timestamp
        received_at = datetime.utcnow()
        if msg.get("created_at") or msg.get("timestamp"):
            ts = msg.get("created_at") or msg.get("timestamp")
            try:
                if isinstance(ts, (int, float)):
                    received_at = datetime.utcfromtimestamp(ts)
                else:
                    received_at = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).replace(tzinfo=None)
            except (ValueError, TypeError):
                pass

        # Check if this is a connection request
        is_connection_request = msg.get("type") == "connection_request" or chat_id.startswith("conn_")

        return InboundMessage(
            source=SourceChannel.LINKEDIN,
            source_message_id=msg_id,
            sender_name=sender_name,
            sender_email=sender_email,
            sender_linkedin_url=sender_linkedin_url,
            sender_title=title or headline,
            sender_company=company,
            body=body,
            thread_context=thread_context,
            received_at=received_at,
            is_connection_request=is_connection_request,
            account_id=account_id,
            raw_data={"chat_id": chat_id, "account_id": account_id, **msg},
        )

    @staticmethod
    def _parse_headline(headline: str) -> tuple[Optional[str], Optional[str]]:
        """Parse a LinkedIn headline into title and company.

        Common formats:
        - "CEO at Acme Corp"
        - "VP Sales | Growth Company"
        - "Founder & CEO, Startup Inc."
        """
        if not headline:
            return None, None

        # Try "title at/@ company"
        match = re.match(r"^(.+?)\s+(?:at|@)\s+(.+)$", headline, re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()

        # Try "title | company" or "title - company"
        match = re.match(r"^(.+?)\s*[|–—-]\s*(.+)$", headline)
        if match:
            return match.group(1).strip(), match.group(2).strip()

        # Try "title, company"
        match = re.match(r"^(.+?),\s+(.+)$", headline)
        if match:
            return match.group(1).strip(), match.group(2).strip()

        return headline, None
