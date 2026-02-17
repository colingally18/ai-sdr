"""3-step reply drafter using Claude API.

Drafts replies to inbound messages using a single Claude call with a prompt
that guides the model through three phases:
  1. Analyze - Understand the lead, intent, and appropriate strategy.
  2. Draft - Write a tailored reply matching tone and channel constraints.
  3. Self-Critique - Review the draft for quality, then output the final version.

The prompt itself orchestrates all three steps; only the final reply text
is returned in the DraftReply model.
"""

from __future__ import annotations

import anthropic
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sdr.ai.prompts import build_reply_prompt
from sdr.models import DraftReply, InboundMessage, LeadClassification

logger = structlog.get_logger(__name__)


class ReplyDrafter:
    """Drafts replies to classified inbound messages using a 3-step prompt chain.

    The drafter sends a single prompt to Claude that contains instructions for
    all three steps (analyze, draft, self-critique). Claude processes everything
    in one call, and the drafter extracts the final reply text from the response.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.7,
        self_critique_enabled: bool = True,
    ) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.self_critique_enabled = self_critique_enabled
        logger.info(
            "reply_drafter_initialized",
            model=model,
            temperature=temperature,
            self_critique_enabled=self_critique_enabled,
        )

    @retry(
        retry=retry_if_exception_type(
            (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError)
        ),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        before_sleep=lambda retry_state: structlog.get_logger(__name__).warning(
            "reply_drafter_retry",
            attempt=retry_state.attempt_number,
            wait=retry_state.next_action.sleep,  # type: ignore[union-attr]
        ),
    )
    def draft(
        self,
        message: InboundMessage,
        classification: LeadClassification,
        enrichment_data: str = "",
    ) -> DraftReply:
        """Draft a reply to an inbound message using a 3-step prompt chain.

        The prompt instructs Claude to:
          1. **Analyze**: Identify the lead's intent, pain points, and the best
             response strategy given the classification and sales context.
          2. **Draft**: Write a reply that matches the channel (LinkedIn = concise,
             email = slightly longer), uses the appropriate tone, and includes a
             clear call-to-action.
          3. **Self-Critique**: Review the draft for issues (too salesy, too long,
             missing personalization, weak CTA) and output an improved final version.

        Only the final reply text (after self-critique) is returned.

        Args:
            message: The inbound message being replied to.
            classification: The AI classification of the lead.
            enrichment_data: Optional enrichment data for personalization.

        Returns:
            A DraftReply with the final reply_text and optional strategy_notes.

        Raises:
            ValueError: If the response cannot be parsed into a reply.
            anthropic.APIError: On unrecoverable API errors (after retries).
        """
        prompt = build_reply_prompt(
            message=message,
            classification=classification,
            enrichment_data=enrichment_data,
        )

        # If self-critique is disabled, append an instruction to skip step 3
        if not self.self_critique_enabled:
            prompt += (
                "\n\nIMPORTANT: Skip the self-critique step. "
                "Output your draft directly as the final reply."
            )

        logger.debug(
            "drafting_reply",
            sender=message.sender_name,
            category=classification.category.value,
            source=message.source.value,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract the text content from Claude's response
        raw_text = ""
        for block in response.content:
            if block.type == "text":
                raw_text += block.text

        if not raw_text.strip():
            logger.error(
                "reply_draft_empty",
                sender=message.sender_name,
                response_content=[b.type for b in response.content],
            )
            raise ValueError(
                "Claude returned an empty reply. "
                f"Response content types: {[b.type for b in response.content]}"
            )

        # Parse the response: look for a FINAL REPLY section or use the full text
        reply_text, strategy_notes = self._parse_response(raw_text)

        draft = DraftReply(reply_text=reply_text, strategy_notes=strategy_notes)

        logger.info(
            "reply_drafted",
            sender=message.sender_name,
            category=classification.category.value,
            reply_length=len(draft.reply_text),
            has_strategy_notes=bool(draft.strategy_notes),
        )

        return draft

    @staticmethod
    def _parse_response(raw_text: str) -> tuple[str, str]:
        """Parse Claude's response to extract the final reply and strategy notes.

        The prompt asks Claude to structure output with markers:
          - <STRATEGY_NOTES>...</STRATEGY_NOTES> for internal reasoning
          - <FINAL_REPLY>...</FINAL_REPLY> for the actual reply text

        If markers are not found, falls back to using the entire response as
        the reply text.

        Args:
            raw_text: The raw text response from Claude.

        Returns:
            A tuple of (reply_text, strategy_notes).
        """
        strategy_notes = ""
        reply_text = raw_text.strip()

        # Extract strategy notes if present
        strategy_start = raw_text.find("<STRATEGY_NOTES>")
        strategy_end = raw_text.find("</STRATEGY_NOTES>")
        if strategy_start != -1 and strategy_end != -1:
            strategy_notes = raw_text[
                strategy_start + len("<STRATEGY_NOTES>") : strategy_end
            ].strip()

        # Extract final reply if present
        reply_start = raw_text.find("<FINAL_REPLY>")
        reply_end = raw_text.find("</FINAL_REPLY>")
        if reply_start != -1 and reply_end != -1:
            reply_text = raw_text[
                reply_start + len("<FINAL_REPLY>") : reply_end
            ].strip()

        return reply_text, strategy_notes
