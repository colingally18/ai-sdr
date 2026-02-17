"""Contact enrichment with 3-tier cascade.

Enrichment priority:
1. RapidAPI ultraapis (real-time people/company data)
2. Apollo.io (people match — great for email → LinkedIn discovery)
3. Perplexity Sonar (web search fallback)

After enrichment, structured fields are written back to the contact
record (Title, Company, LinkedIn URL) in addition to raw JSON.
"""

from __future__ import annotations

from typing import Optional

import requests
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from sdr.enrichment.apollo import ApolloEnricher
from sdr.enrichment.perplexity import PerplexityEnricher

logger = structlog.get_logger()


class ContactEnricher:
    """Enriches contact data using a 3-tier cascade."""

    RAPIDAPI_HOST = "real-time-people-company-data.p.rapidapi.com"

    def __init__(
        self,
        api_key: str,
        provider: str = "rapidapi",
        apollo_api_key: str = "",
        perplexity_api_key: str = "",
    ):
        self.api_key = api_key  # RapidAPI key
        self.provider = provider
        self.apollo = ApolloEnricher(apollo_api_key) if apollo_api_key else None
        self.perplexity = PerplexityEnricher(perplexity_api_key) if perplexity_api_key else None

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
        company: Optional[str] = None,
    ) -> Optional[dict]:
        """Enrich a contact using the 3-tier cascade.

        Cascade:
        1. If LinkedIn URL → RapidAPI ultraapis person lookup
        2. If email → Apollo.io people/match (also discovers LinkedIn URL)
        3. If Apollo found LinkedIn URL → RapidAPI ultraapis for deeper data
        4. If still no data → Perplexity search + parse
        5. Always: RapidAPI ultraapis company lookup for company intelligence

        Returns enrichment data dict or None if no data found.
        """
        if not self.api_key:
            logger.debug("enricher.no_api_key")
            return None

        result: dict = {}
        discovered_linkedin_url = linkedin_url

        # --- Tier 1: RapidAPI ultraapis ---

        # 1a. Person lookup by LinkedIn URL
        if linkedin_url:
            rapid_data = self._rapidapi_person_by_linkedin(linkedin_url)
            if rapid_data:
                result = self._merge(result, rapid_data, source="rapidapi_linkedin")

        # 1b. Person lookup by email (if no LinkedIn data yet)
        if not result and email:
            rapid_data = self._rapidapi_person_by_email(email)
            if rapid_data:
                result = self._merge(result, rapid_data, source="rapidapi_email")
                # May have discovered LinkedIn URL
                if rapid_data.get("linkedin_url"):
                    discovered_linkedin_url = rapid_data["linkedin_url"]

        # --- Tier 2: Apollo.io ---

        if self.apollo and self.apollo.is_available():
            # Use Apollo when we have email but still missing data
            if (email or name) and not result.get("title"):
                apollo_data = self.apollo.enrich(
                    email=email,
                    linkedin_url=linkedin_url,
                    name=name,
                    organization_name=company,
                )
                if apollo_data:
                    result = self._merge(result, apollo_data, source="apollo")
                    # Apollo may have discovered LinkedIn URL
                    if apollo_data.get("linkedin_url") and not discovered_linkedin_url:
                        discovered_linkedin_url = apollo_data["linkedin_url"]
                        # Now do a deeper RapidAPI lookup with the discovered URL
                        rapid_data = self._rapidapi_person_by_linkedin(discovered_linkedin_url)
                        if rapid_data:
                            result = self._merge(result, rapid_data, source="rapidapi_linkedin")

        # --- Tier 3: Perplexity fallback ---

        if not result and self.perplexity and self.perplexity.is_available():
            perplexity_data = self.perplexity.enrich(
                name=name,
                company=company,
                email=email,
                linkedin_url=linkedin_url,
            )
            if perplexity_data:
                result = self._merge(result, perplexity_data, source="perplexity")

        # --- Always: Company intelligence ---

        company_name = result.get("company") or company
        company_domain = result.get("company_domain")
        if company_name or company_domain:
            company_data = self._rapidapi_company_lookup(
                company_name=company_name,
                domain=company_domain,
            )
            if company_data:
                result["company_data"] = company_data

        # Store discovered LinkedIn URL
        if discovered_linkedin_url and not result.get("linkedin_url"):
            result["linkedin_url"] = discovered_linkedin_url

        if result:
            logger.info(
                "enricher.cascade_complete",
                sources=result.get("_sources", []),
                has_title=bool(result.get("title")),
                has_company=bool(result.get("company")),
                has_linkedin=bool(result.get("linkedin_url")),
            )
            return result

        return None

    # ------------------------------------------------------------------
    # RapidAPI ultraapis methods
    # ------------------------------------------------------------------

    def _rapidapi_person_by_linkedin(self, linkedin_url: str) -> Optional[dict]:
        """Look up a person by LinkedIn URL using ultraapis."""
        try:
            resp = requests.get(
                f"https://{self.RAPIDAPI_HOST}/search-person",
                headers={
                    "X-RapidAPI-Key": self.api_key,
                    "X-RapidAPI-Host": self.RAPIDAPI_HOST,
                },
                params={"linkedin_url": linkedin_url},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data and data.get("status") == "OK":
                    person = data.get("data", data)
                    logger.info("enricher.rapidapi_linkedin_found", url=linkedin_url)
                    return self._normalize_rapidapi_person(person)
        except Exception as e:
            logger.warning("enricher.rapidapi_linkedin_failed", url=linkedin_url, error=str(e))
        return None

    def _rapidapi_person_by_email(self, email: str) -> Optional[dict]:
        """Look up a person by email using ultraapis."""
        try:
            resp = requests.get(
                f"https://{self.RAPIDAPI_HOST}/search-person",
                headers={
                    "X-RapidAPI-Key": self.api_key,
                    "X-RapidAPI-Host": self.RAPIDAPI_HOST,
                },
                params={"email": email},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data and data.get("status") == "OK":
                    person = data.get("data", data)
                    logger.info("enricher.rapidapi_email_found", email=email)
                    return self._normalize_rapidapi_person(person)
        except Exception as e:
            logger.warning("enricher.rapidapi_email_failed", email=email, error=str(e))
        return None

    def _rapidapi_company_lookup(
        self,
        company_name: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Optional[dict]:
        """Look up company information using ultraapis."""
        params: dict = {}
        if domain:
            params["domain"] = domain
        elif company_name:
            params["name"] = company_name
        else:
            return None

        try:
            resp = requests.get(
                f"https://{self.RAPIDAPI_HOST}/search-company",
                headers={
                    "X-RapidAPI-Key": self.api_key,
                    "X-RapidAPI-Host": self.RAPIDAPI_HOST,
                },
                params=params,
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data and data.get("status") == "OK":
                    logger.info("enricher.company_found", company=company_name or domain)
                    return data.get("data", data)
        except Exception as e:
            logger.warning("enricher.company_lookup_failed", error=str(e))
        return None

    @staticmethod
    def _normalize_rapidapi_person(person: dict) -> dict:
        """Normalize ultraapis person response to standard format."""
        return {
            "source": "rapidapi",
            "name": person.get("full_name", ""),
            "first_name": person.get("first_name", ""),
            "last_name": person.get("last_name", ""),
            "title": person.get("job_title", person.get("title", "")),
            "linkedin_url": person.get("linkedin_url", ""),
            "email": person.get("email", ""),
            "city": person.get("city", ""),
            "state": person.get("state", ""),
            "country": person.get("country", ""),
            "company": person.get("company", person.get("company_name", "")),
            "company_domain": person.get("company_domain", ""),
            "company_industry": person.get("industry", ""),
            "headline": person.get("headline", ""),
        }

    @staticmethod
    def _merge(existing: dict, new_data: dict, source: str = "") -> dict:
        """Merge new enrichment data into existing, without overwriting non-empty values."""
        merged = dict(existing)

        # Track sources
        sources = merged.get("_sources", [])
        if source:
            sources.append(source)
        merged["_sources"] = sources

        for key, value in new_data.items():
            if key == "_sources":
                continue
            # Don't overwrite existing non-empty values
            if not merged.get(key) and value:
                merged[key] = value

        return merged

    def is_available(self) -> bool:
        """Check if any enrichment service is available."""
        return bool(self.api_key) or (
            self.apollo is not None and self.apollo.is_available()
        ) or (
            self.perplexity is not None and self.perplexity.is_available()
        )
