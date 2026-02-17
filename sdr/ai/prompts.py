"""Prompt loader for the Growlancer SDR AI layer.

Loads prompt templates from config/prompts/ and injects sales context
from config/sales_context.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog

from sdr.config import EXAMPLES_DIR, PROMPTS_DIR, load_sales_context
from sdr.db import get_active_learned_rules
from sdr.models import ContactRecord, InboundMessage, LeadClassification

logger = structlog.get_logger(__name__)


def _format_sales_context(ctx: dict) -> str:
    """Flatten the sales context dict into a readable string block."""
    lines: list[str] = []
    for key, value in ctx.items():
        if isinstance(value, dict):
            lines.append(f"\n## {key.replace('_', ' ').title()}")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list):
                    lines.append(f"  {sub_key}: {', '.join(str(v) for v in sub_value)}")
                else:
                    lines.append(f"  {sub_key}: {sub_value}")
        elif isinstance(value, list):
            lines.append(f"{key}: {', '.join(str(v) for v in value)}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def load_prompt(name: str) -> str:
    """Read a .txt prompt template from config/prompts/.

    Args:
        name: Filename (without extension) of the prompt template.

    Returns:
        The raw template string.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        logger.error("prompt_not_found", path=str(path))
        raise FileNotFoundError(f"Prompt template not found: {path}")
    text = path.read_text(encoding="utf-8")
    logger.debug("prompt_loaded", name=name, length=len(text))
    return text


def load_examples() -> str:
    """Read all .txt files from config/examples/ and concatenate them.

    Returns:
        A single string with all examples separated by double newlines.
        Returns an empty string if the directory does not exist or is empty.
    """
    if not EXAMPLES_DIR.exists():
        logger.warning("examples_dir_missing", path=str(EXAMPLES_DIR))
        return ""

    example_files = sorted(EXAMPLES_DIR.glob("*.txt"))
    if not example_files:
        logger.info("no_examples_found", path=str(EXAMPLES_DIR))
        return ""

    parts: list[str] = []
    for fpath in example_files:
        content = fpath.read_text(encoding="utf-8").strip()
        if content:
            parts.append(f"--- Example: {fpath.stem} ---\n{content}")

    combined = "\n\n".join(parts)
    logger.debug("examples_loaded", count=len(parts), total_length=len(combined))
    return combined


def build_classification_prompt(
    message: InboundMessage,
    enrichment_data: str = "",
    current_stage: str = "",
) -> str:
    """Build the lead classification prompt from the classify_lead.txt template.

    Injects sales context, message data, enrichment info, and conversation stage
    into the template placeholders.

    Args:
        message: The normalized inbound message to classify.
        enrichment_data: Optional enrichment payload (company info, etc.).
        current_stage: Current conversation stage if this is an ongoing thread.

    Returns:
        The fully rendered classification prompt string.
    """
    template = load_prompt("classify_lead")
    sales_ctx = load_sales_context()
    examples = load_examples()

    prompt = template.replace("{{SALES_CONTEXT}}", _format_sales_context(sales_ctx))
    prompt = prompt.replace("{{EXAMPLES}}", examples)

    # Message fields
    prompt = prompt.replace("{{SOURCE}}", message.source.value)
    prompt = prompt.replace("{{SENDER_NAME}}", message.sender_name or "Unknown")
    prompt = prompt.replace("{{SENDER_EMAIL}}", message.sender_email or "N/A")
    prompt = prompt.replace("{{SENDER_TITLE}}", message.sender_title or "N/A")
    prompt = prompt.replace("{{SENDER_COMPANY}}", message.sender_company or "N/A")
    prompt = prompt.replace("{{SENDER_LINKEDIN_URL}}", message.sender_linkedin_url or "N/A")
    prompt = prompt.replace("{{SUBJECT}}", message.subject or "N/A")
    prompt = prompt.replace("{{BODY}}", message.body)
    prompt = prompt.replace("{{THREAD_CONTEXT}}", message.thread_context or "N/A")
    prompt = prompt.replace("{{RECEIVED_AT}}", message.received_at.isoformat())

    # Optional enrichment and stage
    prompt = prompt.replace("{{ENRICHMENT_DATA}}", enrichment_data or "None available")
    prompt = prompt.replace("{{CURRENT_STAGE}}", current_stage or "New")

    logger.debug("classification_prompt_built", prompt_length=len(prompt))
    return prompt


def build_reply_prompt(
    message: InboundMessage,
    classification: LeadClassification,
    enrichment_data: str = "",
) -> str:
    """Build the reply drafting prompt from the draft_reply.txt template.

    Injects sales context, the original message, classification results,
    and enrichment data into the template.

    Args:
        message: The original inbound message being replied to.
        classification: AI classification result for the message.
        enrichment_data: Optional enrichment payload.

    Returns:
        The fully rendered reply drafting prompt string.
    """
    template = load_prompt("draft_reply")
    sales_ctx = load_sales_context()
    examples = load_examples()

    prompt = template.replace("{{SALES_CONTEXT}}", _format_sales_context(sales_ctx))
    prompt = prompt.replace("{{EXAMPLES}}", examples)

    # Message fields
    prompt = prompt.replace("{{SOURCE}}", message.source.value)
    prompt = prompt.replace("{{SENDER_NAME}}", message.sender_name or "Unknown")
    prompt = prompt.replace("{{SENDER_EMAIL}}", message.sender_email or "N/A")
    prompt = prompt.replace("{{SENDER_TITLE}}", message.sender_title or "N/A")
    prompt = prompt.replace("{{SENDER_COMPANY}}", message.sender_company or "N/A")
    prompt = prompt.replace("{{SUBJECT}}", message.subject or "N/A")
    prompt = prompt.replace("{{BODY}}", message.body)
    prompt = prompt.replace("{{THREAD_CONTEXT}}", message.thread_context or "N/A")

    # Classification fields
    prompt = prompt.replace("{{LEAD_CATEGORY}}", classification.category.value)
    prompt = prompt.replace("{{CONFIDENCE}}", f"{classification.confidence:.2f}")
    prompt = prompt.replace("{{DETECTED_INTENT}}", classification.detected_intent)
    prompt = prompt.replace("{{DETECTED_SIGNALS}}", ", ".join(classification.detected_signals))
    prompt = prompt.replace("{{CONVERSATION_STAGE}}", classification.conversation_stage.value)
    prompt = prompt.replace("{{ICP_MATCH_SCORE}}", f"{classification.icp_match_score:.2f}")
    prompt = prompt.replace("{{AI_REASONING}}", classification.reasoning)

    # Enrichment
    prompt = prompt.replace("{{ENRICHMENT_DATA}}", enrichment_data or "None available")

    # Learned rules from self-learning
    rules = get_active_learned_rules()
    if rules:
        rules_text = "\n".join(
            f"{i+1}. {r['rule_text']}" for i, r in enumerate(rules)
        )
    else:
        rules_text = "No learned preferences yet."
    prompt = prompt.replace("{{LEARNED_RULES}}", rules_text)

    logger.debug("reply_prompt_built", prompt_length=len(prompt))
    return prompt


def build_connection_eval_prompt(
    name: str,
    headline: str,
    company: str,
    location: str = "",
    mutual_connections: int = 0,
    request_message: str = "",
    profile_summary: str = "",
) -> str:
    """Build the connection evaluation prompt from evaluate_connection.txt.

    Args:
        name: Name of the person sending the connection request.
        headline: LinkedIn headline of the requester.
        company: Company of the requester.
        location: Geographic location of the requester.
        mutual_connections: Number of mutual connections.
        request_message: Optional message attached to the connection request.
        profile_summary: Optional summary/about text from the profile.

    Returns:
        The fully rendered connection evaluation prompt string.
    """
    template = load_prompt("evaluate_connection")
    sales_ctx = load_sales_context()

    prompt = template.replace("{{SALES_CONTEXT}}", _format_sales_context(sales_ctx))

    # Connection request fields
    prompt = prompt.replace("{{NAME}}", name)
    prompt = prompt.replace("{{HEADLINE}}", headline)
    prompt = prompt.replace("{{COMPANY}}", company)
    prompt = prompt.replace("{{LOCATION}}", location or "N/A")
    prompt = prompt.replace("{{MUTUAL_CONNECTIONS}}", str(mutual_connections))
    prompt = prompt.replace("{{REQUEST_MESSAGE}}", request_message or "No message")
    prompt = prompt.replace("{{PROFILE_SUMMARY}}", profile_summary or "N/A")

    logger.debug("connection_eval_prompt_built", prompt_length=len(prompt))
    return prompt


def build_followup_prompt(
    contact: ContactRecord,
    channel: str,
    conversation_history: str,
    followup_number: int,
) -> str:
    """Build the follow-up drafting prompt from the draft_followup.txt template.

    Args:
        contact: The contact record for the lead.
        channel: The channel to draft for ("LinkedIn" or "Email").
        conversation_history: Full conversation history with this contact.
        followup_number: Which follow-up number this is (1-8).

    Returns:
        The fully rendered follow-up drafting prompt string.
    """
    template = load_prompt("draft_followup")
    sales_ctx = load_sales_context()

    prompt = template.replace("{{SALES_CONTEXT}}", _format_sales_context(sales_ctx))

    # Contact fields
    prompt = prompt.replace("{{CONTACT_NAME}}", contact.name or "Unknown")
    prompt = prompt.replace("{{CONTACT_EMAIL}}", contact.email or "N/A")
    prompt = prompt.replace("{{CONTACT_TITLE}}", contact.title or "N/A")
    prompt = prompt.replace("{{CONTACT_COMPANY}}", contact.company or "N/A")
    prompt = prompt.replace("{{LEAD_CATEGORY}}", contact.lead_category.value)
    prompt = prompt.replace("{{CONVERSATION_STAGE}}", contact.conversation_stage.value)
    prompt = prompt.replace("{{ENRICHMENT_DATA}}", contact.enriched_data or "None available")

    # Follow-up specifics
    prompt = prompt.replace("{{CHANNEL}}", channel)
    prompt = prompt.replace("{{FOLLOWUP_NUMBER}}", str(followup_number))
    prompt = prompt.replace("{{CONVERSATION_HISTORY}}", conversation_history or "No prior messages")

    # Learned rules
    rules = get_active_learned_rules()
    if rules:
        rules_text = "\n".join(
            f"{i+1}. {r['rule_text']}" for i, r in enumerate(rules)
        )
    else:
        rules_text = "No learned preferences yet."
    prompt = prompt.replace("{{LEARNED_RULES}}", rules_text)

    logger.debug("followup_prompt_built", prompt_length=len(prompt))
    return prompt
