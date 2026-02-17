"""Connection request handler for LinkedIn.

Polls Unipile for pending connection requests, evaluates them against
the ICP using AI, and auto-accepts or logs rejections.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

import requests
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from sdr.models import (
    AuditAction,
    AuditLogEntry,
    ContactRecord,
    ConversationStage,
    SourceChannel,
)

if TYPE_CHECKING:
    from sdr.ai.connection_eval import ConnectionEvaluator
    from sdr.crm.airtable import AirtableCRM

logger = structlog.get_logger()


class ConnectionRequestHandler:
    """Handles LinkedIn connection requests via Unipile."""

    def __init__(
        self,
        unipile_dsn: str,
        unipile_api_key: str,
        evaluator: "ConnectionEvaluator",
        crm: "AirtableCRM",
        auto_accept: bool = True,
        min_icp_confidence: float = 0.7,
    ):
        self.unipile_dsn = unipile_dsn
        self.unipile_api_key = unipile_api_key
        self.evaluator = evaluator
        self.crm = crm
        self.auto_accept = auto_accept
        self.min_icp_confidence = min_icp_confidence

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def fetch_pending_requests(self) -> list[dict]:
        """Fetch pending connection requests from Unipile."""
        url = f"https://{self.unipile_dsn}/api/v1/connection_requests"
        headers = {
            "X-API-KEY": self.unipile_api_key,
            "Content-Type": "application/json",
        }
        params = {"status": "pending"}

        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("items", data.get("data", []))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def accept_request(self, request_id: str) -> None:
        """Accept a connection request via Unipile."""
        url = f"https://{self.unipile_dsn}/api/v1/connection_requests/{request_id}/accept"
        headers = {
            "X-API-KEY": self.unipile_api_key,
            "Content-Type": "application/json",
        }
        resp = requests.post(url, headers=headers, timeout=30)
        resp.raise_for_status()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def reject_request(self, request_id: str) -> None:
        """Reject (ignore) a connection request via Unipile."""
        url = f"https://{self.unipile_dsn}/api/v1/connection_requests/{request_id}/reject"
        headers = {
            "X-API-KEY": self.unipile_api_key,
            "Content-Type": "application/json",
        }
        resp = requests.post(url, headers=headers, timeout=30)
        resp.raise_for_status()

    def process_requests(self) -> dict:
        """Process all pending connection requests.

        Returns summary stats: {accepted, rejected, errors, total}.
        """
        stats = {"total": 0, "accepted": 0, "rejected": 0, "errors": 0}

        try:
            requests_list = self.fetch_pending_requests()
        except Exception as e:
            logger.error("connections.fetch_failed", error=str(e))
            return stats

        stats["total"] = len(requests_list)
        if not requests_list:
            return stats

        logger.info("connections.found_pending", count=len(requests_list))

        for req in requests_list:
            try:
                self._process_single_request(req, stats)
            except Exception as e:
                logger.error(
                    "connections.process_failed",
                    request_id=req.get("id"),
                    error=str(e),
                )
                stats["errors"] += 1

        logger.info("connections.batch_complete", **stats)
        return stats

    def _process_single_request(self, req: dict, stats: dict) -> None:
        """Process a single connection request."""
        request_id = req.get("id", "")
        name = req.get("name", req.get("sender_name", ""))
        headline = req.get("headline", "")
        company = req.get("company", "")
        location = req.get("location", "")
        mutual_connections = req.get("mutual_connections", 0)
        request_message = req.get("message", "")
        profile_summary = req.get("summary", "")
        linkedin_url = req.get("linkedin_url", req.get("profile_url", ""))

        log = logger.bind(request_id=request_id, name=name)

        # Evaluate against ICP
        evaluation = self.evaluator.evaluate(
            name=name,
            headline=headline,
            company=company,
            location=location,
            mutual_connections=mutual_connections,
            request_message=request_message,
            profile_summary=profile_summary,
        )

        log.info(
            "connections.evaluated",
            accept=evaluation.accept,
            confidence=evaluation.confidence,
            category=evaluation.lead_category.value,
        )

        if evaluation.accept and self.auto_accept and evaluation.confidence >= self.min_icp_confidence:
            # Auto-accept
            self.accept_request(request_id)
            log.info("connections.auto_accepted")

            # Create contact in CRM
            contact = ContactRecord(
                name=name,
                linkedin_url=linkedin_url,
                company=company,
                title=headline,
                source_channel=SourceChannel.LINKEDIN,
                lead_category=evaluation.lead_category,
                conversation_stage=ConversationStage.NEW,
                ai_confidence=evaluation.confidence,
                ai_reasoning=evaluation.reasoning,
            )
            contact = self.crm.upsert_contact(contact)

            self.crm.log_audit(AuditLogEntry(
                action=AuditAction.AUTO_ACCEPTED,
                contact_id=contact.id,
                details=json.dumps({
                    "name": name,
                    "headline": headline,
                    "company": company,
                    "confidence": evaluation.confidence,
                    "reasoning": evaluation.reasoning,
                }),
            ))
            stats["accepted"] += 1
        else:
            # Reject or log without accepting
            if not evaluation.accept:
                self.reject_request(request_id)
                log.info("connections.auto_rejected")
                self.crm.log_audit(AuditLogEntry(
                    action=AuditAction.AUTO_REJECTED,
                    details=json.dumps({
                        "name": name,
                        "headline": headline,
                        "company": company,
                        "confidence": evaluation.confidence,
                        "reasoning": evaluation.reasoning,
                    }),
                ))
            else:
                # Accept is true but confidence below threshold — accept but flag
                self.accept_request(request_id)
                log.info("connections.accepted_low_confidence")
            stats["rejected"] += 1
