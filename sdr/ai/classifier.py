"""Lead classifier using Claude API with structured output.

Classifies inbound messages into lead categories (Hot/Warm/Cold/Not a Lead)
with confidence scores, detected signals, and conversation stage tracking.
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

from sdr.ai.prompts import build_classification_prompt
from sdr.models import (
    ConversationStage,
    InboundMessage,
    LeadCategory,
    LeadClassification,
)

logger = structlog.get_logger(__name__)

# Tool schema matching the LeadClassification Pydantic model
CLASSIFICATION_TOOL = {
    "name": "classify_lead",
    "description": (
        "Classify an inbound sales lead based on the message content, "
        "sender information, and sales context. Return structured classification."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": [c.value for c in LeadCategory],
                "description": "Lead category: Hot, Warm, Cold, or Not a Lead.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence in the classification (0.0 to 1.0).",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why this classification was chosen.",
            },
            "detected_intent": {
                "type": "string",
                "description": (
                    "The primary intent detected in the message "
                    "(e.g., 'buying signal', 'information request', 'spam')."
                ),
            },
            "detected_signals": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific buying/interest signals detected.",
            },
            "should_reply": {
                "type": "boolean",
                "description": "Whether this message warrants a reply.",
            },
            "conversation_stage": {
                "type": "string",
                "enum": [s.value for s in ConversationStage],
                "description": "Current stage in the sales conversation.",
            },
            "icp_match_score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "How well the sender matches the Ideal Customer Profile (0.0 to 1.0).",
            },
        },
        "required": [
            "category",
            "confidence",
            "reasoning",
            "detected_intent",
            "detected_signals",
            "should_reply",
            "conversation_stage",
            "icp_match_score",
        ],
    },
}


class LeadClassifier:
    """Classifies inbound messages using Claude API with tool_use for structured output.

    The classifier sends a classification prompt to Claude along with a tool definition
    that mirrors the LeadClassification schema. Claude returns the structured result
    via a tool call, which is parsed into a LeadClassification model.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.1,
    ) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        logger.info(
            "classifier_initialized",
            model=model,
            temperature=temperature,
        )

    @retry(
        retry=retry_if_exception_type(
            (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError)
        ),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        before_sleep=lambda retry_state: structlog.get_logger(__name__).warning(
            "classifier_retry",
            attempt=retry_state.attempt_number,
            wait=retry_state.next_action.sleep,  # type: ignore[union-attr]
        ),
    )
    def classify(
        self,
        message: InboundMessage,
        enrichment_data: str = "",
        current_stage: str = "",
    ) -> LeadClassification:
        """Classify an inbound message into a lead category with structured output.

        Args:
            message: The normalized inbound message to classify.
            enrichment_data: Optional enrichment data (company info, tech stack, etc.).
            current_stage: Current conversation stage if this is an ongoing thread.

        Returns:
            A LeadClassification with category, confidence, signals, and more.

        Raises:
            ValueError: If Claude's response does not contain a valid tool_use block.
            anthropic.APIError: On unrecoverable API errors (after retries).
        """
        prompt = build_classification_prompt(
            message=message,
            enrichment_data=enrichment_data,
            current_stage=current_stage,
        )

        logger.debug(
            "classifying_message",
            sender=message.sender_name,
            source=message.source.value,
            body_length=len(message.body),
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            tools=[CLASSIFICATION_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract the tool_use block from the response
        for block in response.content:
            if block.type == "tool_use":
                classification = LeadClassification(**block.input)
                logger.info(
                    "message_classified",
                    sender=message.sender_name,
                    category=classification.category.value,
                    confidence=classification.confidence,
                    should_reply=classification.should_reply,
                    stage=classification.conversation_stage.value,
                    icp_score=classification.icp_match_score,
                    signals=classification.detected_signals,
                )
                return classification

        # If no tool_use block found, raise an error
        logger.error(
            "classification_no_tool_use",
            response_content=[b.type for b in response.content],
        )
        raise ValueError(
            "Claude did not return a tool_use block for classification. "
            f"Response content types: {[b.type for b in response.content]}"
        )
