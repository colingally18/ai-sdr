"""Unified message sender for Gmail and LinkedIn (Unipile).

Sends approved replies back through the original channel with
rate limiting and error handling.
"""

from __future__ import annotations

import base64
from email.mime.text import MIMEText
from typing import Optional

import requests
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from sdr.sending.rate_limiter import RateLimiter

logger = structlog.get_logger()


class MessageSender:
    """Sends messages via Gmail API or Unipile (LinkedIn)."""

    def __init__(
        self,
        gmail_service=None,
        unipile_dsn: str = "",
        unipile_api_key: str = "",
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.gmail_service = gmail_service
        self.unipile_dsn = unipile_dsn
        self.unipile_api_key = unipile_api_key
        self.rate_limiter = rate_limiter or RateLimiter()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def send_gmail(
        self,
        to_email: str,
        subject: str,
        body: str,
        thread_id: Optional[str] = None,
        in_reply_to: Optional[str] = None,
    ) -> dict:
        """Send an email via Gmail API.

        Args:
            to_email: Recipient email address.
            subject: Email subject line.
            body: Email body text.
            thread_id: Gmail thread ID for threading replies.
            in_reply_to: Message-ID header for threading.

        Returns:
            Gmail API response dict with 'id' and 'threadId'.
        """
        if not self.gmail_service:
            raise RuntimeError("Gmail service not initialized")

        if not self.rate_limiter.acquire("gmail"):
            raise RuntimeError("Gmail rate limit exceeded")

        message = MIMEText(body)
        message["to"] = to_email
        message["subject"] = subject
        if in_reply_to:
            message["In-Reply-To"] = in_reply_to
            message["References"] = in_reply_to

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        send_body = {"raw": raw}
        if thread_id:
            send_body["threadId"] = thread_id

        result = (
            self.gmail_service.users()
            .messages()
            .send(userId="me", body=send_body)
            .execute()
        )

        logger.info(
            "sender.gmail_sent",
            to=to_email,
            message_id=result.get("id"),
            thread_id=result.get("threadId"),
        )
        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def send_linkedin(
        self,
        account_id: str,
        chat_id: str,
        text: str,
    ) -> dict:
        """Send a LinkedIn message via Unipile API.

        Args:
            account_id: Unipile account ID.
            chat_id: Unipile chat/conversation ID.
            text: Message text.

        Returns:
            Unipile API response dict.
        """
        if not self.unipile_dsn or not self.unipile_api_key:
            raise RuntimeError("Unipile credentials not configured")

        if not self.rate_limiter.acquire("linkedin"):
            raise RuntimeError("LinkedIn rate limit exceeded")

        url = f"https://{self.unipile_dsn}/api/v1/chats/{chat_id}/messages"
        headers = {
            "X-API-KEY": self.unipile_api_key,
            "Content-Type": "application/json",
        }
        payload = {"text": text}

        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        logger.info(
            "sender.linkedin_sent",
            chat_id=chat_id,
            message_id=result.get("id"),
        )
        return result

    def send(
        self,
        channel: str,
        *,
        to_email: str = "",
        subject: str = "",
        body: str = "",
        thread_id: str = "",
        in_reply_to: str = "",
        account_id: str = "",
        chat_id: str = "",
    ) -> dict:
        """Unified send method that routes to the correct channel.

        Args:
            channel: "Gmail" or "LinkedIn"
            Remaining kwargs: channel-specific parameters.

        Returns:
            API response dict from the sending channel.
        """
        if channel == "Gmail":
            return self.send_gmail(
                to_email=to_email,
                subject=subject,
                body=body,
                thread_id=thread_id or None,
                in_reply_to=in_reply_to or None,
            )
        elif channel == "LinkedIn":
            return self.send_linkedin(
                account_id=account_id,
                chat_id=chat_id,
                text=body,
            )
        else:
            raise ValueError(f"Unknown channel: {channel}")
