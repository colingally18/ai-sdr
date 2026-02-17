"""Main entry point for the Growlancer SDR system.

Performs startup validation, initializes all components, and runs
concurrent inbound/outbound/connection polling loops.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
from pathlib import Path

import schedule
import structlog


def configure_logging(log_dir: Path) -> None:
    """Configure structured JSON logging."""
    log_dir.mkdir(parents=True, exist_ok=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def validate_startup(secrets, config) -> bool:
    """Validate all required services are accessible.

    Returns True if all checks pass, False otherwise.
    """
    logger = structlog.get_logger()
    errors = []

    # 1. Check required env vars
    from sdr.config import validate_secrets
    missing = validate_secrets(secrets)
    if missing:
        for var in missing:
            logger.error("startup.missing_env_var", var=var)
            errors.append(f"Missing required env var: {var}")

    # 2. Test Anthropic API
    if secrets.anthropic_api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=secrets.anthropic_api_key)
            client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            logger.info("startup.anthropic_ok")
        except Exception as e:
            logger.error("startup.anthropic_failed", error=str(e))
            errors.append(f"Anthropic API: {e}")

    # 3. Test Airtable API
    if secrets.airtable_api_key and secrets.airtable_base_id:
        try:
            from pyairtable import Api
            api = Api(secrets.airtable_api_key)
            # Try to access the base - this will fail if key/base is invalid
            base = api.base(secrets.airtable_base_id)
            base.schema()
            logger.info("startup.airtable_ok")
        except Exception as e:
            logger.error("startup.airtable_failed", error=str(e))
            errors.append(f"Airtable API: {e}")

    # 4. Test Gmail OAuth (if configured)
    gmail_service = None
    if secrets.gmail_credentials_path:
        try:
            from sdr.sources.gmail import GmailSource
            gmail_source = GmailSource(secrets.gmail_credentials_path)
            if gmail_source.is_available():
                gmail_service = gmail_source.service
                logger.info("startup.gmail_ok")
            else:
                logger.warning("startup.gmail_not_available", hint="Run once interactively for OAuth")
        except Exception as e:
            logger.warning("startup.gmail_not_configured", error=str(e))

    # 5. Test Unipile API (if configured)
    if secrets.unipile_dsn and secrets.unipile_api_key:
        try:
            import requests as req
            resp = req.get(
                f"https://{secrets.unipile_dsn}/api/v1/accounts",
                headers={"X-API-KEY": secrets.unipile_api_key},
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info("startup.unipile_ok")
            else:
                logger.warning("startup.unipile_auth_failed", status=resp.status_code)
        except Exception as e:
            logger.warning("startup.unipile_not_configured", error=str(e))

    if errors:
        for err in errors:
            print(f"  FAIL: {err}", file=sys.stderr)
        return False

    return True


def build_components(secrets, config):
    """Initialize all system components."""
    from sdr.ai.classifier import LeadClassifier
    from sdr.ai.connection_eval import ConnectionEvaluator
    from sdr.ai.reply_drafter import ReplyDrafter
    from sdr.config import DATA_DIR
    from sdr.connections.handler import ConnectionRequestHandler
    from sdr.crm.airtable import AirtableCRM
    from sdr.crm.dedup import ContactDeduplicator
    from sdr.enrichment.enricher import ContactEnricher
    from sdr.pipeline import InboundPipeline
    from sdr.sending.rate_limiter import RateLimiter
    from sdr.sending.sender import MessageSender
    from sdr.sources.gmail import GmailSource
    from sdr.sources.linkedin import LinkedInSource

    # CRM
    crm = AirtableCRM(
        api_key=secrets.airtable_api_key,
        base_id=secrets.airtable_base_id,
    )

    # AI components
    classifier = LeadClassifier(
        api_key=secrets.anthropic_api_key,
        model=config.classification.model,
        temperature=config.classification.temperature,
    )
    drafter = ReplyDrafter(
        api_key=secrets.anthropic_api_key,
        model=config.reply_drafting.model,
        temperature=config.reply_drafting.temperature,
        self_critique_enabled=config.reply_drafting.self_critique_enabled,
    )

    # Dedup
    dedup = ContactDeduplicator(crm)

    # Enrichment
    enricher = None
    if config.enrichment.enabled and (secrets.rapidapi_key or secrets.apollo_api_key or secrets.perplexity_api_key):
        enricher = ContactEnricher(
            api_key=secrets.rapidapi_key,
            provider=config.enrichment.provider,
            apollo_api_key=secrets.apollo_api_key,
            perplexity_api_key=secrets.perplexity_api_key,
        )

    # Pipeline
    pipeline = InboundPipeline(
        crm=crm,
        dedup=dedup,
        classifier=classifier,
        drafter=drafter,
        enricher=enricher,
        config=config,
    )

    # Sources
    gmail_source = None
    if secrets.gmail_credentials_path:
        try:
            gmail_source = GmailSource(credentials_path=secrets.gmail_credentials_path)
            if not gmail_source.is_available():
                gmail_source = None
        except Exception:
            gmail_source = None

    linkedin_source = None
    linkedin_accounts = []
    if secrets.unipile_dsn and secrets.unipile_api_key:
        linkedin_source = LinkedInSource(
            dsn=secrets.unipile_dsn,
            api_key=secrets.unipile_api_key,
        )
        if linkedin_source.is_available():
            # Fetch all connected LinkedIn accounts for multi-account support
            linkedin_accounts = linkedin_source.fetch_accounts()
            for acct in linkedin_accounts:
                structlog.get_logger().info(
                    "startup.linkedin_account",
                    account_id=acct.get("id"),
                    name=acct.get("name"),
                    provider=acct.get("provider"),
                )
        else:
            linkedin_source = None

    # Rate limiter + sender
    rate_limiter = RateLimiter(
        gmail_per_hour=config.sending.rate_limit.gmail_per_hour,
        linkedin_per_hour=config.sending.rate_limit.linkedin_per_hour,
    )
    sender = MessageSender(
        gmail_service=gmail_source.service if gmail_source else None,
        unipile_dsn=secrets.unipile_dsn,
        unipile_api_key=secrets.unipile_api_key,
        rate_limiter=rate_limiter,
    )

    # Connection handler
    connection_handler = None
    if linkedin_source and secrets.unipile_dsn:
        evaluator = ConnectionEvaluator(
            api_key=secrets.anthropic_api_key,
            model=config.classification.model,
            temperature=config.classification.temperature,
        )
        connection_handler = ConnectionRequestHandler(
            unipile_dsn=secrets.unipile_dsn,
            unipile_api_key=secrets.unipile_api_key,
            evaluator=evaluator,
            crm=crm,
            auto_accept=config.connections.auto_accept,
            min_icp_confidence=config.connections.min_icp_confidence,
        )

    # Self-learner
    learner = None
    if config.learning.enabled:
        from sdr.ai.learner import SelfLearner
        learner = SelfLearner(
            api_key=secrets.anthropic_api_key,
            crm=crm,
            model=config.classification.model,
            temperature=config.classification.temperature,
        )

    return {
        "crm": crm,
        "pipeline": pipeline,
        "gmail_source": gmail_source,
        "linkedin_source": linkedin_source,
        "sender": sender,
        "connection_handler": connection_handler,
        "learner": learner,
        "config": config,
        "secrets": secrets,
    }


class CircuitBreaker:
    """Simple circuit breaker for source polling."""

    def __init__(self, threshold: int = 5, cooldown_seconds: int = 600):
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, float] = {}

    def record_failure(self, source: str) -> None:
        self._failures[source] = self._failures.get(source, 0) + 1
        if self._failures[source] >= self.threshold:
            self._open_until[source] = time.monotonic() + self.cooldown_seconds
            structlog.get_logger().warning(
                "circuit_breaker.opened",
                source=source,
                cooldown_seconds=self.cooldown_seconds,
            )

    def record_success(self, source: str) -> None:
        self._failures[source] = 0
        self._open_until.pop(source, None)

    def is_open(self, source: str) -> bool:
        until = self._open_until.get(source)
        if until is None:
            return False
        if time.monotonic() >= until:
            # Cooldown expired, allow retry
            self._open_until.pop(source, None)
            self._failures[source] = 0
            return False
        return True


def run_inbound_cycle(components: dict, circuit_breaker: CircuitBreaker) -> None:
    """Run one inbound polling cycle for all sources."""
    logger = structlog.get_logger()
    pipeline = components["pipeline"]

    # Gmail
    gmail_source = components.get("gmail_source")
    if gmail_source and not circuit_breaker.is_open("gmail"):
        try:
            messages = gmail_source.poll()
            if messages:
                logger.info("inbound.gmail_messages", count=len(messages))
                pipeline.process_batch(messages)
            circuit_breaker.record_success("gmail")
        except Exception as e:
            logger.error("inbound.gmail_error", error=str(e))
            circuit_breaker.record_failure("gmail")

    # LinkedIn
    linkedin_source = components.get("linkedin_source")
    if linkedin_source and not circuit_breaker.is_open("linkedin"):
        try:
            messages = linkedin_source.poll()
            if messages:
                logger.info("inbound.linkedin_messages", count=len(messages))
                pipeline.process_batch(messages)
            circuit_breaker.record_success("linkedin")
        except Exception as e:
            logger.error("inbound.linkedin_error", error=str(e))
            circuit_breaker.record_failure("linkedin")


def run_outbound_cycle(components: dict) -> None:
    """Run one outbound sending cycle."""
    from sdr.outbound import process_approved_messages
    try:
        sent = process_approved_messages(components["crm"], components["sender"])
        if sent:
            structlog.get_logger().info("outbound.cycle_complete", sent=sent)
    except Exception as e:
        structlog.get_logger().error("outbound.cycle_error", error=str(e))


def run_connection_cycle(components: dict) -> None:
    """Run one connection request processing cycle."""
    handler = components.get("connection_handler")
    if handler:
        try:
            stats = handler.process_requests()
            if stats["total"]:
                structlog.get_logger().info("connections.cycle_complete", **stats)
        except Exception as e:
            structlog.get_logger().error("connections.cycle_error", error=str(e))


def run_learning_cycle_job(components: dict) -> None:
    """Run one self-learning cycle."""
    learner = components.get("learner")
    if learner:
        try:
            config = components["config"]
            stats = learner.run_learning_cycle(
                lookback_days=config.learning.lookback_days,
                max_active_rules=config.learning.max_active_rules,
                min_messages=config.learning.min_messages_for_analysis,
            )
            structlog.get_logger().info("learning.cycle_complete", **stats)
        except Exception as e:
            structlog.get_logger().error("learning.cycle_error", error=str(e))


def run_followup_cycle_job(components: dict) -> None:
    """Run one follow-up cadence cycle."""
    try:
        from sdr.followup import run_followup_cycle
        config = components["config"]
        stats = run_followup_cycle(
            crm=components["crm"],
            api_key=components["secrets"].anthropic_api_key,
            config=config.followup,
        )
        structlog.get_logger().info("followup.cycle_complete", **stats)
    except Exception as e:
        structlog.get_logger().error("followup.cycle_error", error=str(e))


def main():
    """Main entry point."""
    from sdr import db
    from sdr.config import LOG_DIR, load_config, load_secrets

    # 1. Configure logging
    configure_logging(LOG_DIR)
    logger = structlog.get_logger()
    logger.info("startup.begin")

    # 2. Load config
    config = load_config()
    secrets = load_secrets()

    # 3. Validate
    logger.info("startup.validating")
    if not validate_startup(secrets, config):
        logger.error("startup.validation_failed")
        print("\nStartup validation failed. Check errors above.", file=sys.stderr)
        sys.exit(1)

    # 4. Initialize SQLite
    db.init_db()
    logger.info("startup.db_initialized")

    # 5. Initialize Airtable schema
    components = build_components(secrets, config)
    crm = components["crm"]
    logger.info("startup.ensuring_airtable_schema")
    crm.ensure_schema()
    logger.info("startup.airtable_schema_ready")

    # 6. Log config summary
    logger.info(
        "startup.config_summary",
        polling_interval=config.polling.interval_seconds,
        auto_send=config.sending.auto_send,
        gmail_available=components.get("gmail_source") is not None,
        linkedin_available=components.get("linkedin_source") is not None,
        enrichment_enabled=config.enrichment.enabled,
        connections_auto_accept=config.connections.auto_accept,
    )

    # 7. Set up circuit breaker
    circuit_breaker = CircuitBreaker(
        threshold=config.error_handling.circuit_breaker_threshold,
        cooldown_seconds=config.error_handling.circuit_breaker_cooldown_seconds,
    )

    # 8. Schedule polling loops
    interval = config.polling.interval_seconds

    schedule.every(interval).seconds.do(run_inbound_cycle, components, circuit_breaker)
    schedule.every(interval).seconds.do(run_outbound_cycle, components)
    schedule.every(interval).seconds.do(run_connection_cycle, components)

    # Daily jobs
    if config.learning.enabled:
        schedule.every().day.at(config.learning.schedule_time).do(
            run_learning_cycle_job, components
        )
        logger.info("startup.learning_scheduled", time=config.learning.schedule_time)

    if config.followup.enabled:
        schedule.every().day.at(config.followup.schedule_time).do(
            run_followup_cycle_job, components
        )
        logger.info("startup.followup_scheduled", time=config.followup.schedule_time)

    # Run initial cycle immediately
    run_inbound_cycle(components, circuit_breaker)
    run_outbound_cycle(components)
    run_connection_cycle(components)

    # 9. Graceful shutdown
    running = True

    def handle_signal(signum, frame):
        nonlocal running
        logger.info("shutdown.signal_received", signal=signum)
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("startup.complete", message="SDR system running. Press Ctrl+C to stop.")
    print("\nGrowlancer SDR system is running.")
    print(f"  Polling every {interval}s")
    print("  Press Ctrl+C to stop.\n")

    # 10. Main loop
    while running:
        schedule.run_pending()
        time.sleep(1)

    logger.info("shutdown.complete")
    print("\nShutdown complete.")


if __name__ == "__main__":
    main()
