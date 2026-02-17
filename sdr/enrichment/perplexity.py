"""Perplexity Sonar API fallback for contact enrichment.

Used as a last-resort enrichment source when RapidAPI and Apollo
return no data. Searches the web for professional information
and returns structured results.
"""

from __future__ import annotations

import json
from typing import Optional

import requests
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger()


class PerplexityEnricher:
    """Enriches contacts using the Perplexity Sonar API."""

    BASE_URL = "https://api.perplexity.ai/chat/completions"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def is_available(self) -> bool:
        return bool(self.api_key)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        reraise=True,
    )
    def enrich(
        self,
        name: Optional[str] = None,
        company: Optional[str] = None,
        email: Optional[str] = None,
        linkedin_url: Optional[str] = None,
    ) -> Optional[dict]:
        """Search for professional information about a person.

        Returns structured enrichment data dict, or None if nothing found.
        """
        if not self.api_key:
            return None

        # Build a search query from available info
        query_parts = []
        if name and name != "Unknown":
            query_parts.append(name)
        if company:
            query_parts.append(f"at {company}")
        if email:
            query_parts.append(f"email: {email}")
        if linkedin_url:
            query_parts.append(f"LinkedIn: {linkedin_url}")

        if not query_parts:
            return None

        person_desc = " ".join(query_parts)
        prompt = (
            f"Find professional information about {person_desc}. "
            "Return a JSON object with these fields (use empty string if unknown): "
            '"name", "title", "company", "linkedin_url", "city", "country", '
            '"company_industry", "company_size_estimate", "recent_news". '
            "Only return the JSON object, no other text."
        )

        try:
            resp = requests.post(
                self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "sonar",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                },
                timeout=30,
            )

            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    parsed = self._parse_response(content)
                    if parsed:
                        logger.info("perplexity.enrichment_found", name=name)
                        return parsed
            elif resp.status_code == 429:
                logger.warning("perplexity.rate_limited")
            else:
                logger.debug("perplexity.no_result", status=resp.status_code)
        except Exception as e:
            logger.warning("perplexity.lookup_failed", error=str(e))

        return None

    @staticmethod
    def _parse_response(content: str) -> Optional[dict]:
        """Parse Perplexity response into structured enrichment data."""
        # Try to extract JSON from the response
        try:
            # Handle case where response has markdown code blocks
            if "```" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    content = content[start:end]

            data = json.loads(content)
            if isinstance(data, dict):
                data["source"] = "perplexity"
                return data
        except (json.JSONDecodeError, ValueError):
            logger.debug("perplexity.parse_failed", content=content[:200])

        return None
