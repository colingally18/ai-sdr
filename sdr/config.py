"""Configuration loader for the Growlancer SDR system.

Loads secrets from .env and behavior config from config.yaml + sales_context.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class PollingConfig(BaseModel):
    interval_seconds: int = 120
    gmail_max_results: int = 50


class ModelConfig(BaseModel):
    model: str = "claude-sonnet-4-5-20250929"
    temperature: float = 0.1


class ReplyDraftingConfig(BaseModel):
    model: str = "claude-sonnet-4-5-20250929"
    temperature: float = 0.7
    max_linkedin_words: int = 60
    max_email_words: int = 150
    self_critique_enabled: bool = True


class AutoSendRules(BaseModel):
    min_confidence: float = 0.85
    allowed_categories: list[str] = Field(default_factory=lambda: ["warm", "cold"])
    max_per_day: int = 50
    require_prior_human_approval: bool = True


class RateLimitConfig(BaseModel):
    gmail_per_hour: int = 20
    linkedin_per_hour: int = 10


class SendingConfig(BaseModel):
    auto_send: bool = False
    auto_send_rules: AutoSendRules = Field(default_factory=AutoSendRules)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)


class ConnectionsConfig(BaseModel):
    auto_accept: bool = True
    min_icp_confidence: float = 0.7


class EnrichmentConfig(BaseModel):
    enabled: bool = True
    provider: str = "rapidapi"


class ErrorHandlingConfig(BaseModel):
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_cooldown_seconds: int = 600


class LearningConfig(BaseModel):
    enabled: bool = True
    schedule_time: str = "06:00"
    lookback_days: int = 7
    max_active_rules: int = 10
    min_messages_for_analysis: int = 3


class FollowUpConfig(BaseModel):
    enabled: bool = True
    schedule_time: str = "08:00"
    total_followups: int = 8
    linkedin_followups: int = 4
    days_between: int = 3
    days_before_activation: int = 3
    auto_approve_threshold: int = 2
    model: str = "claude-sonnet-4-5-20250929"
    temperature: float = 0.7


class AppConfig(BaseModel):
    """Full application configuration from config.yaml."""
    polling: PollingConfig = Field(default_factory=PollingConfig)
    classification: ModelConfig = Field(default_factory=ModelConfig)
    reply_drafting: ReplyDraftingConfig = Field(default_factory=ReplyDraftingConfig)
    sending: SendingConfig = Field(default_factory=SendingConfig)
    connections: ConnectionsConfig = Field(default_factory=ConnectionsConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    error_handling: ErrorHandlingConfig = Field(default_factory=ErrorHandlingConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    followup: FollowUpConfig = Field(default_factory=FollowUpConfig)


class Secrets(BaseSettings):
    """Environment-variable secrets."""
    gmail_credentials_path: str = "./config/gmail_credentials.json"
    unipile_dsn: str = ""
    unipile_api_key: str = ""
    airtable_api_key: str = ""
    airtable_base_id: str = ""
    anthropic_api_key: str = ""
    rapidapi_key: str = ""
    apollo_api_key: str = ""
    perplexity_api_key: str = ""

    class Config:
        env_prefix = ""
        case_sensitive = False


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config() -> AppConfig:
    """Load application config from config/config.yaml."""
    config_path = _PROJECT_ROOT / "config" / "config.yaml"
    if config_path.exists():
        data = _load_yaml(config_path)
        return AppConfig(**data)
    return AppConfig()


def load_sales_context() -> dict[str, Any]:
    """Load sales context from config/sales_context.yaml."""
    ctx_path = _PROJECT_ROOT / "config" / "sales_context.yaml"
    if ctx_path.exists():
        return _load_yaml(ctx_path)
    return {}


def load_secrets() -> Secrets:
    """Load secrets from environment variables."""
    return Secrets()


def validate_secrets(secrets: Secrets) -> list[str]:
    """Return list of missing required secrets."""
    required = {
        "airtable_api_key": "AIRTABLE_API_KEY",
        "airtable_base_id": "AIRTABLE_BASE_ID",
        "anthropic_api_key": "ANTHROPIC_API_KEY",
    }
    missing = []
    for attr, env_name in required.items():
        if not getattr(secrets, attr, ""):
            missing.append(env_name)
    return missing


# Paths
CONFIG_DIR = _PROJECT_ROOT / "config"
PROMPTS_DIR = CONFIG_DIR / "prompts"
EXAMPLES_DIR = CONFIG_DIR / "examples"
DATA_DIR = _PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "sdr.db"
LOG_DIR = DATA_DIR / "logs"
