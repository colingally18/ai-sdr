"""Connection request evaluator using Claude API with structured output.

Evaluates incoming LinkedIn connection requests against the Ideal Customer
Profile (ICP) to decide whether to accept or reject them automatically.
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

from sdr.ai.prompts import build_connection_eval_prompt
from sdr.models import ConnectionEvaluation, LeadCategory

logger = structlog.get_logger(__name__)

# Tool schema matching the ConnectionEvaluation Pydantic model
CONNECTION_EVAL_TOOL = {
    "name": "evaluate_connection",
    "description": (
        "Evaluate a LinkedIn connection request against the Ideal Customer Profile. "
        "Decide whether to accept or reject and assign a lead category."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "accept": {
                "type": "boolean",
                "description": "Whether to accept the connection request.",
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "Brief explanation of why the connection should be "
                    "accepted or rejected."
                ),
            },
            "lead_category": {
                "type": "string",
                "enum": [c.value for c in LeadCategory],
                "description": "Lead category for the requester: Hot, Warm, Cold, or Not a Lead.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence in the evaluation (0.0 to 1.0).",
            },
        },
        "required": ["accept", "reasoning", "lead_category", "confidence"],
    },
}


class ConnectionEvaluator:
    """Evaluates LinkedIn connection requests using Claude API with tool_use.

    Uses structured output (tool_use) to return a ConnectionEvaluation model
    with an accept/reject decision, reasoning, lead category, and confidence.
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
            "connection_evaluator_initialized",
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
            "connection_eval_retry",
            attempt=retry_state.attempt_number,
            wait=retry_state.next_action.sleep,  # type: ignore[union-attr]
        ),
    )
    def evaluate(
        self,
        name: str,
        headline: str,
        company: str,
        location: str = "",
        mutual_connections: int = 0,
        request_message: str = "",
        profile_summary: str = "",
    ) -> ConnectionEvaluation:
        """Evaluate a LinkedIn connection request against the ICP.

        Args:
            name: Full name of the person sending the connection request.
            headline: LinkedIn headline of the requester.
            company: Company name of the requester.
            location: Geographic location of the requester.
            mutual_connections: Number of mutual connections.
            request_message: Optional message attached to the request.
            profile_summary: Optional about/summary text from the profile.

        Returns:
            A ConnectionEvaluation with accept/reject decision and metadata.

        Raises:
            ValueError: If Claude's response does not contain a valid tool_use block.
            anthropic.APIError: On unrecoverable API errors (after retries).
        """
        prompt = build_connection_eval_prompt(
            name=name,
            headline=headline,
            company=company,
            location=location,
            mutual_connections=mutual_connections,
            request_message=request_message,
            profile_summary=profile_summary,
        )

        logger.debug(
            "evaluating_connection",
            name=name,
            headline=headline,
            company=company,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            tools=[CONNECTION_EVAL_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract the tool_use block from the response
        for block in response.content:
            if block.type == "tool_use":
                evaluation = ConnectionEvaluation(**block.input)
                logger.info(
                    "connection_evaluated",
                    name=name,
                    company=company,
                    accept=evaluation.accept,
                    lead_category=evaluation.lead_category.value,
                    confidence=evaluation.confidence,
                    reasoning=evaluation.reasoning[:100],
                )
                return evaluation

        # If no tool_use block found, raise an error
        logger.error(
            "connection_eval_no_tool_use",
            response_content=[b.type for b in response.content],
        )
        raise ValueError(
            "Claude did not return a tool_use block for connection evaluation. "
            f"Response content types: {[b.type for b in response.content]}"
        )
