#!/usr/bin/env python3
"""Backfill script: re-resolve names and re-enrich existing Airtable contacts.

Targets LinkedIn contacts that are broken due to the sender_id / attendee ID
mismatch bug. For each contact with Name="Unknown" or missing enrichment fields:

  1. Find linked messages in Airtable -> get Source Message ID
  2. Fetch message from Unipile -> get sender_id (the LinkedIn provider ID)
  3. Fetch user profile via GET /users/{sender_id} -> name, headline, public_identifier
  4. Run enrichment cascade with the resolved LinkedIn URL
  5. Update the Airtable contact with all resolved data

Usage:
  python scripts/backfill_contacts.py              # dry run (preview changes)
  python scripts/backfill_contacts.py --apply       # apply changes to Airtable
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests
import structlog

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from sdr.config import load_config, load_secrets
from sdr.crm.airtable import AirtableCRM
from sdr.enrichment.enricher import ContactEnricher

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Headline parsing (copied from LinkedInSource to avoid circular deps)
# ---------------------------------------------------------------------------

def parse_headline(headline: str) -> tuple[Optional[str], Optional[str]]:
    if not headline:
        return None, None
    match = re.match(r"^(.+?)\s+(?:at|@)\s+(.+)$", headline, re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    match = re.match(r"^(.+?)\s*[|–—-]\s*(.+)$", headline)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    match = re.match(r"^(.+?),\s+(.+)$", headline)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return headline, None


# ---------------------------------------------------------------------------
# Unipile helpers
# ---------------------------------------------------------------------------

class UnipileClient:
    """Minimal Unipile client for backfill operations."""

    def __init__(self, dsn: str, api_key: str):
        self.base_url = f"https://{dsn}/api/v1"
        self.headers = {"X-API-KEY": api_key, "accept": "application/json"}
        self._user_cache: dict[str, Optional[dict]] = {}

    def fetch_message(self, message_id: str) -> Optional[dict]:
        """GET /messages/{id} -> message dict with sender_id."""
        try:
            resp = requests.get(
                f"{self.base_url}/messages/{message_id}",
                headers=self.headers,
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json()
            log.warning("unipile.message_fetch_failed",
                        message_id=message_id, status=resp.status_code)
        except Exception as e:
            log.warning("unipile.message_fetch_error",
                        message_id=message_id, error=str(e))
        return None

    def fetch_user_profile(self, provider_id: str) -> Optional[dict]:
        """GET /users/{provider_id} -> full profile with name, headline, etc."""
        if provider_id in self._user_cache:
            return self._user_cache[provider_id]
        try:
            resp = requests.get(
                f"{self.base_url}/users/{provider_id}",
                headers=self.headers,
                timeout=15,
            )
            if resp.status_code == 200:
                profile = resp.json()
                pub_id = profile.get("public_identifier")
                if pub_id and not profile.get("profile_url"):
                    profile["profile_url"] = f"https://linkedin.com/in/{pub_id}"
                self._user_cache[provider_id] = profile
                return profile
            log.warning("unipile.user_fetch_failed",
                        provider_id=provider_id, status=resp.status_code)
        except Exception as e:
            log.warning("unipile.user_fetch_error",
                        provider_id=provider_id, error=str(e))
        self._user_cache[provider_id] = None
        return None


# ---------------------------------------------------------------------------
# Name resolution from a profile dict
# ---------------------------------------------------------------------------

def resolve_name(profile: dict) -> str:
    for key in ("display_name", "name", "attendee_name"):
        if profile.get(key):
            return profile[key]
    first = profile.get("first_name", "")
    last = profile.get("last_name", "")
    if first or last:
        return f"{first} {last}".strip()
    return ""


def resolve_linkedin_url(profile: dict) -> str:
    return (
        profile.get("profile_url")
        or profile.get("linkedin_url")
        or profile.get("attendee_profile_url")
        or ""
    )


# ---------------------------------------------------------------------------
# Main backfill logic
# ---------------------------------------------------------------------------

def get_broken_contacts(crm: AirtableCRM) -> list[dict]:
    """Fetch contacts that need backfill: Name='Unknown' or missing key fields."""
    # Get all LinkedIn contacts with Name = "Unknown"
    unknown = crm._all(
        crm._contacts_table,
        formula='AND({Source Channel} = "LinkedIn", {Name} = "Unknown")',
    )

    # Also get LinkedIn contacts missing LinkedIn URL (can't enrich without it)
    missing_url = crm._all(
        crm._contacts_table,
        formula='AND({Source Channel} = "LinkedIn", {Name} != "Unknown", {LinkedIn URL} = "")',
    )

    # Dedup by record ID
    seen = set()
    results = []
    for rec in unknown + missing_url:
        if rec["id"] not in seen:
            seen.add(rec["id"])
            results.append(rec)

    return results


def get_linkedin_message_id(crm: AirtableCRM, contact_id: str) -> Optional[str]:
    """Find a LinkedIn Source Message ID linked to this contact."""
    messages = crm.get_messages_for_contact(contact_id)
    for msg in messages:
        if msg.source.value == "LinkedIn" and msg.source_message_id:
            return msg.source_message_id
    return None


def backfill_contact(
    record: dict,
    crm: AirtableCRM,
    unipile: UnipileClient,
    enricher: Optional[ContactEnricher],
    dry_run: bool,
) -> dict:
    """Backfill a single contact. Returns a summary dict."""
    contact_id = record["id"]
    fields = record["fields"]
    old_name = fields.get("Name", "")

    result = {
        "contact_id": contact_id,
        "old_name": old_name,
        "new_name": None,
        "linkedin_url": None,
        "title": None,
        "company": None,
        "email": None,
        "status": "skipped",
    }

    # Step 1: Find a linked LinkedIn message
    source_message_id = get_linkedin_message_id(crm, contact_id)
    if not source_message_id:
        log.warning("backfill.no_message", contact_id=contact_id, name=old_name)
        result["status"] = "no_message"
        return result

    # Step 2: Fetch message from Unipile to get sender_id
    msg_data = unipile.fetch_message(source_message_id)
    if not msg_data:
        log.warning("backfill.message_not_found",
                     contact_id=contact_id, source_message_id=source_message_id)
        result["status"] = "message_not_found"
        return result

    sender_id = msg_data.get("sender_id", "")
    if not sender_id:
        log.warning("backfill.no_sender_id",
                     contact_id=contact_id, source_message_id=source_message_id)
        result["status"] = "no_sender_id"
        return result

    # Step 3: Fetch user profile from Unipile
    profile = unipile.fetch_user_profile(sender_id)
    if not profile:
        log.warning("backfill.profile_not_found",
                     contact_id=contact_id, sender_id=sender_id)
        result["status"] = "profile_not_found"
        return result

    # Step 4: Resolve fields from profile
    new_name = resolve_name(profile)
    linkedin_url = resolve_linkedin_url(profile)
    headline = profile.get("headline", "")
    title, company = parse_headline(headline)
    email = profile.get("email") or profile.get("email_address") or ""

    result["new_name"] = new_name
    result["linkedin_url"] = linkedin_url
    result["title"] = title
    result["company"] = company
    result["email"] = email

    # Step 5: Run enrichment cascade (if we have a LinkedIn URL)
    enrichment_data = None
    if enricher and linkedin_url:
        try:
            enrichment_data = enricher.enrich(
                linkedin_url=linkedin_url,
                email=email or fields.get("Email"),
                name=new_name or old_name,
                company=company or fields.get("Company"),
            )
        except Exception as e:
            log.warning("backfill.enrichment_failed",
                        contact_id=contact_id, error=str(e))

    # Step 6: Build update payload (only update fields that improve on current data)
    updates: dict = {}

    if new_name and old_name in ("Unknown", ""):
        updates["Name"] = new_name

    if linkedin_url and not fields.get("LinkedIn URL"):
        updates["LinkedIn URL"] = linkedin_url

    if headline and not fields.get("Title"):
        updates["Title"] = title or headline

    if company and not fields.get("Company"):
        updates["Company"] = company

    if email and not fields.get("Email"):
        updates["Email"] = email

    # Overlay enrichment data (enrichment takes priority for title/company/email)
    if enrichment_data:
        if enrichment_data.get("title") and not fields.get("Title"):
            updates["Title"] = enrichment_data["title"]
        if enrichment_data.get("company") and not fields.get("Company"):
            updates["Company"] = enrichment_data["company"]
        if enrichment_data.get("email") and not fields.get("Email"):
            updates["Email"] = enrichment_data["email"]
        if enrichment_data.get("linkedin_url") and not fields.get("LinkedIn URL"):
            updates["LinkedIn URL"] = enrichment_data["linkedin_url"]
        updates["Enriched Data"] = json.dumps(enrichment_data)

    if not updates:
        result["status"] = "no_updates"
        return result

    # Step 7: Apply or preview
    if dry_run:
        log.info("backfill.preview", contact_id=contact_id,
                 old_name=old_name, updates=updates)
        result["status"] = "would_update"
    else:
        crm.update_contact(contact_id, updates)
        log.info("backfill.updated", contact_id=contact_id,
                 old_name=old_name, updates=list(updates.keys()))
        result["status"] = "updated"

    return result


def main():
    apply = "--apply" in sys.argv
    mode = "APPLY" if apply else "DRY RUN"

    log.info("backfill.start", mode=mode)

    # Load config
    secrets = load_secrets()
    config = load_config()

    if not secrets.airtable_api_key or not secrets.airtable_base_id:
        log.error("backfill.missing_airtable_config")
        sys.exit(1)

    if not secrets.unipile_dsn or not secrets.unipile_api_key:
        log.error("backfill.missing_unipile_config")
        sys.exit(1)

    # Initialize components
    crm = AirtableCRM(
        api_key=secrets.airtable_api_key,
        base_id=secrets.airtable_base_id,
    )

    unipile = UnipileClient(
        dsn=secrets.unipile_dsn,
        api_key=secrets.unipile_api_key,
    )

    enricher = None
    if config.enrichment.enabled and secrets.rapidapi_key:
        enricher = ContactEnricher(
            api_key=secrets.rapidapi_key,
            provider=config.enrichment.provider,
            apollo_api_key=secrets.apollo_api_key,
            perplexity_api_key=secrets.perplexity_api_key,
        )

    # Fetch broken contacts
    log.info("backfill.fetching_contacts")
    contacts = get_broken_contacts(crm)
    log.info("backfill.contacts_found", count=len(contacts))

    if not contacts:
        log.info("backfill.nothing_to_do")
        return

    # Process each contact
    stats = {"updated": 0, "would_update": 0, "skipped": 0, "failed": 0}
    results = []

    for i, record in enumerate(contacts, 1):
        name = record["fields"].get("Name", "")
        log.info("backfill.processing", progress=f"{i}/{len(contacts)}",
                 contact_id=record["id"], name=name)

        try:
            result = backfill_contact(record, crm, unipile, enricher, dry_run=not apply)
            results.append(result)

            if result["status"] in ("updated", "would_update"):
                stats[result["status"]] += 1
            else:
                stats["skipped"] += 1

        except Exception as e:
            log.error("backfill.contact_failed",
                      contact_id=record["id"], error=str(e))
            stats["failed"] += 1

        # Gentle rate limiting — avoid hammering APIs
        time.sleep(0.3)

    # Summary
    log.info("backfill.complete", mode=mode, **stats)

    print(f"\n{'=' * 60}")
    print(f"Backfill complete ({mode})")
    print(f"{'=' * 60}")
    print(f"  Contacts found:  {len(contacts)}")
    if apply:
        print(f"  Updated:         {stats['updated']}")
    else:
        print(f"  Would update:    {stats['would_update']}")
    print(f"  Skipped:         {stats['skipped']}")
    print(f"  Failed:          {stats['failed']}")

    # Print details for updated/would-update contacts
    actionable = [r for r in results if r["status"] in ("updated", "would_update")]
    if actionable:
        print(f"\n{'─' * 60}")
        print("Changes:")
        for r in actionable:
            old = r["old_name"] or "(empty)"
            new = r["new_name"] or "(unchanged)"
            url = r["linkedin_url"] or "(none)"
            title = r["title"] or "(none)"
            company = r["company"] or "(none)"
            print(f"\n  {r['contact_id']}")
            print(f"    Name:     {old} -> {new}")
            print(f"    LinkedIn: {url}")
            print(f"    Title:    {title}")
            print(f"    Company:  {company}")

    if not apply and stats["would_update"] > 0:
        print(f"\nRun with --apply to write these changes to Airtable.")


if __name__ == "__main__":
    main()
