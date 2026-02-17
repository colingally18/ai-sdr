"""Apollo.io people enrichment integration.

Uses the Apollo People Match API to enrich contacts by email,
LinkedIn URL, or name+company. Particularly useful for discovering
LinkedIn URLs from email addresses.
"""

from __future__ import annotations

from typing import Optional

import requests
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger()


class ApolloEnricher:
    """Enriches contacts via the Apollo.io People Match API."""

    BASE_URL = "https://api.apollo.io/api/v1/people/match"

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
        email: Optional[str] = None,
        linkedin_url: Optional[str] = None,
        name: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        organization_name: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Optional[dict]:
        """Match a person using Apollo.io.

        Returns enriched person data dict, or None if no match found.
        """
        if not self.api_key:
            return None

        payload: dict = {}
        if email:
            payload["email"] = email
        if linkedin_url:
            payload["linkedin_url"] = linkedin_url
        if name:
            # Split name into first/last for Apollo
            parts = name.strip().split(None, 1)
            if not first_name:
                payload["first_name"] = parts[0]
            if not last_name and len(parts) > 1:
                payload["last_name"] = parts[1]
        if first_name:
            payload["first_name"] = first_name
        if last_name:
            payload["last_name"] = last_name
        if organization_name:
            payload["organization_name"] = organization_name
        if domain:
            payload["domain"] = domain

        if not payload:
            return None

        try:
            resp = requests.post(
                self.BASE_URL,
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                person = data.get("person")
                if person:
                    logger.info(
                        "apollo.match_found",
                        email=email,
                        linkedin_url=linkedin_url,
                        name=person.get("name"),
                    )
                    return self._normalize(person)
            elif resp.status_code == 429:
                logger.warning("apollo.rate_limited")
            else:
                logger.debug("apollo.no_match", status=resp.status_code)
        except Exception as e:
            logger.warning("apollo.lookup_failed", error=str(e))

        return None

    @staticmethod
    def _normalize(person: dict) -> dict:
        """Normalize Apollo person response to a standard enrichment dict."""
        org = person.get("organization", {}) or {}
        return {
            "source": "apollo",
            "name": person.get("name", ""),
            "first_name": person.get("first_name", ""),
            "last_name": person.get("last_name", ""),
            "title": person.get("title", ""),
            "linkedin_url": person.get("linkedin_url", ""),
            "email": person.get("email", ""),
            "city": person.get("city", ""),
            "state": person.get("state", ""),
            "country": person.get("country", ""),
            "company": org.get("name", ""),
            "company_domain": org.get("primary_domain", ""),
            "company_industry": org.get("industry", ""),
            "company_size": org.get("estimated_num_employees"),
            "company_linkedin_url": org.get("linkedin_url", ""),
            "employment_history": person.get("employment_history", []),
        }
