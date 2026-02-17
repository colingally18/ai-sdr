"""Self-learning module: analyzes human edits to AI drafts and extracts rules.

Runs daily, reads recently edited messages from Airtable, sends (before, after)
pairs to Claude, and stores extracted patterns as rules in SQLite. These rules
are then injected into future reply and follow-up prompts.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import anthropic
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sdr.ai.prompts import load_prompt, _format_sales_context
from sdr.config import load_sales_context
from sdr.db import (
    deactivate_learned_rule,
    get_active_learned_rules,
    insert_learned_rule,
    log_local_audit,
)
from sdr.models import AuditAction, AuditLogEntry

if TYPE_CHECKING:
    from sdr.crm.airtable import AirtableCRM

logger = structlog.get_logger(__name__)

# Tool schema for structured rule extraction
LEARNING_TOOL = {
    "name": "extract_rules",
    "description": (
        "Extract writing rules from patterns observed in human edits to AI drafts. "
        "Return up to 2 rules with confidence scores."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "rules": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rule_text": {
                            "type": "string",
                            "description": "A concise, actionable writing rule (one sentence).",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "How confident you are in this pattern (0.0 to 1.0).",
                        },
                    },
                    "required": ["rule_text", "confidence"],
                },
                "maxItems": 2,
                "description": "Extracted rules (max 2). Empty array if no clear patterns found.",
            },
        },
        "required": ["rules"],
    },
}


class SelfLearner:
    """Analyzes human edits to AI drafts and extracts reusable writing rules."""

    def __init__(
        self,
        api_key: str,
        crm: "AirtableCRM",
        model: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.1,
    ) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.crm = crm
        self.model = model
        self.temperature = temperature
        logger.info("self_learner_initialized", model=model)

    def run_learning_cycle(
        self,
        lookback_days: int = 7,
        max_active_rules: int = 10,
        min_messages: int = 3,
    ) -> dict:
        """Run one learning cycle. Returns stats about the cycle."""
        logger.info("learning.cycle_start", lookback_days=lookback_days)

        # 1. Fetch edited messages
        edit_pairs = self._fetch_edited_messages(days=lookback_days)
        if len(edit_pairs) < min_messages:
            logger.info(
                "learning.skipped_insufficient_data",
                found=len(edit_pairs),
                required=min_messages,
            )
            return {
                "messages_analyzed": 0,
                "new_rules": 0,
                "skipped_reason": f"Only {len(edit_pairs)} edited messages (need {min_messages})",
            }

        # 2. Get existing rules
        existing_rules = get_active_learned_rules()

        # 3. Analyze patterns
        new_rules = self._analyze_patterns(edit_pairs, existing_rules)

        # 4. Store rules with confidence > 0.7
        stored = 0
        for rule in new_rules:
            if rule["confidence"] > 0.7:
                insert_learned_rule(rule["rule_text"], rule["confidence"])
                stored += 1
                logger.info(
                    "learning.rule_stored",
                    rule=rule["rule_text"],
                    confidence=rule["confidence"],
                )

        # 5. Cap active rules
        all_active = get_active_learned_rules()
        if len(all_active) > max_active_rules:
            # Deactivate oldest rules over the cap
            to_deactivate = all_active[: len(all_active) - max_active_rules]
            for rule in to_deactivate:
                deactivate_learned_rule(rule["id"])
                logger.info("learning.rule_deactivated", rule_id=rule["id"])

        # 6. Audit
        log_local_audit(
            action=AuditAction.LEARNING_CYCLE.value,
            details={
                "messages_analyzed": len(edit_pairs),
                "new_rules": stored,
                "total_active_rules": len(get_active_learned_rules()),
            },
        )

        result = {
            "messages_analyzed": len(edit_pairs),
            "new_rules": stored,
            "skipped_reason": None,
        }
        logger.info("learning.cycle_complete", **result)
        return result

    def _fetch_edited_messages(self, days: int = 7) -> list[dict]:
        """Fetch recently sent messages where the human edited the AI draft."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        formula = (
            'AND('
            '{Status} = "Sent", '
            '{Edit Distance} > 0.05, '
            '{AI Draft Version} != "", '
            f'IS_AFTER({{Sent At}}, "{cutoff}")'
            ')'
        )
        records = self.crm._all(self.crm._messages_table, formula=formula)

        pairs = []
        for r in records:
            f = r["fields"]
            # Get lead category from linked contact
            lead_category = ""
            contact_links = f.get("Contact")
            if contact_links:
                contact = self.crm.get_contact(contact_links[0])
                if contact:
                    lead_category = contact.lead_category.value

            pairs.append({
                "ai_draft": f.get("AI Draft Version", ""),
                "human_edit": f.get("Draft Reply", ""),
                "channel": f.get("Source", ""),
                "lead_category": lead_category,
                "edit_distance": f.get("Edit Distance", 0),
            })
        return pairs

    @retry(
        retry=retry_if_exception_type(
            (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError)
        ),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        before_sleep=lambda retry_state: structlog.get_logger(__name__).warning(
            "learner_retry",
            attempt=retry_state.attempt_number,
        ),
    )
    def _analyze_patterns(
        self, edit_pairs: list[dict], existing_rules: list[dict]
    ) -> list[dict]:
        """Send edit pairs to Claude and extract rules."""
        template = load_prompt("analyze_edits")
        sales_ctx = load_sales_context()

        # Format existing rules
        if existing_rules:
            rules_text = "\n".join(
                f"{i+1}. {r['rule_text']}" for i, r in enumerate(existing_rules)
            )
        else:
            rules_text = "No rules yet."

        # Format edit pairs
        pairs_text = ""
        for i, pair in enumerate(edit_pairs, 1):
            category_info = f", Lead Category: {pair['lead_category']}" if pair.get('lead_category') else ""
            pairs_text += (
                f"### Edit {i} (Channel: {pair['channel']}{category_info}, Edit Distance: {pair['edit_distance']:.2f})\n"
                f"**AI Draft:**\n{pair['ai_draft']}\n\n"
                f"**Human Edit:**\n{pair['human_edit']}\n\n"
            )

        prompt = template.replace("{{SALES_CONTEXT}}", _format_sales_context(sales_ctx))
        prompt = prompt.replace("{{EXISTING_RULES}}", rules_text)
        prompt = prompt.replace("{{EDIT_PAIRS}}", pairs_text)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            tools=[LEARNING_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": prompt}],
        )

        for block in response.content:
            if block.type == "tool_use":
                return block.input.get("rules", [])

        logger.warning("learning.no_tool_use_response")
        return []
