import json
from functools import lru_cache
from os import environ
from typing import Annotated, Final, Self

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from whatsapp.jid import (
    DefaultUserServer,
    GroupServer,
    JIDParseError,
    LegacyUserServer,
    parse_jid,
)

# Provider prefixes this app manages credentials for.
# prefix -> (Settings attribute, environment variable pydantic-ai reads)
_MANAGED_PROVIDERS: Final[dict[str, tuple[str, str]]] = {
    "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
    "openrouter": ("openrouter_api_key", "OPENROUTER_API_KEY"),
    "deepseek": ("deepseek_api_key", "DEEPSEEK_API_KEY"),
    "kimi": ("kimi_api_key", "KIMI_API_KEY"),
    "groq": ("groq_api_key", "GROQ_API_KEY"),
    "gemini": ("gemini_api_key", "GEMINI_API_KEY"),
    "nvidia": ("nvidia_api_key", "NVIDIA_API_KEY"),
    "google-gla": ("google_api_key", "GOOGLE_API_KEY"),
}

# pydantic-ai still accepts unprefixed legacy names, mapping them to a provider by
# prefix (its private `_LEGACY_MODEL_PREFIXES`). We mirror only the entry that maps
# to a provider we manage, so `MODEL_NAME=claude-sonnet-4-6` is credential-checked
# the same as `anthropic:claude-sonnet-4-6`.
_LEGACY_NAME_PROVIDERS: Final[dict[str, str]] = {"claude": "anthropic"}


class Settings(BaseSettings):
    # API settings
    port: int = 5001
    host: str = "0.0.0.0"

    # Database settings
    db_uri: str

    # WhatsApp settings
    whatsapp_host: str
    whatsapp_basic_auth_password: str | None = None
    whatsapp_basic_auth_user: str | None = None

    # LLM provider credentials. Which one is required depends on the provider
    # `model_name` resolves to — see `validate_model_credentials`.
    anthropic_api_key: str | None = None
    openrouter_api_key: str | None = None
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    kimi_api_key: str | None = None
    kimi_base_url: str = "https://api.moonshot.ai/v1"
    kimi_model: str = "kimi-k2-turbo-preview"
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.6-flash"
    nvidia_api_key: str | None = None
    google_api_key: str | None = None

    # Voyage settings
    voyage_api_key: str
    voyage_max_retries: int = 5

    # Model settings
    model_name: str = "google-gla:gemini-2.5-flash"
    llm_provider_order: str = "deepseek,gemini,kimi,openrouter,nvidia,groq"
    ai_enabled: bool = True

    # Direct Message settings
    dm_autoreply_enabled: bool = False
    dm_autoreply_message: str = (
        "Hello, I am not designed to answer to personal messages."
    )

    # QA tester settings (user JIDs allowed to use /kb_qa command)
    qa_testers: list[str] = []

    # QA test groups (group JIDs where /kb_qa command is allowed)
    qa_test_groups: Annotated[list[str], NoDecode] = []

    # Groups where mention-triggered replies and KB ingestion are enabled.
    active_groups: Annotated[list[str], NoDecode] = []

    # AUTO_REPLY_GROUPS is a comma-separated allowlist of group JIDs.
    auto_reply_groups: Annotated[list[str], NoDecode] = []

    # Subject prefixes used to keep known test topics out of automatic context.
    kb_exclude_subject_prefixes: Annotated[list[str], NoDecode] = ["Hackathon inquiry"]

    # Optional settings
    escalation_primary_jids: Annotated[list[str], NoDecode] = []
    escalation_secondary_jids: Annotated[list[str], NoDecode] = []
    debug: bool = False
    log_level: str = "INFO"
    logfire_token: str

    @field_validator("qa_testers")
    @classmethod
    def validate_qa_testers(cls, v: list[str]) -> list[str]:
        """Validate that qa_testers contains valid user JIDs."""
        valid_user_servers = (DefaultUserServer, LegacyUserServer)
        for jid_str in v:
            try:
                jid = parse_jid(jid_str)
            except JIDParseError as e:
                raise ValueError(f"Invalid JID '{jid_str}': {e}") from e

            if jid.server not in valid_user_servers:
                raise ValueError(
                    f"Invalid user JID '{jid_str}'. Expected server to be one of "
                    f"{valid_user_servers}, got '{jid.server}'"
                )
            if not jid.user:
                raise ValueError(f"Invalid user JID '{jid_str}'. Missing user part.")
        return v

    @field_validator("qa_test_groups", "active_groups")
    @classmethod
    def validate_qa_test_groups(cls, v: list[str]) -> list[str]:
        """Validate that qa_test_groups contains valid group JIDs."""
        for jid_str in v:
            try:
                jid = parse_jid(jid_str)
            except JIDParseError as e:
                raise ValueError(f"Invalid JID '{jid_str}': {e}") from e

            if not jid.is_group():
                raise ValueError(
                    f"Invalid group JID '{jid_str}'. Expected server '{GroupServer}', "
                    f"got '{jid.server}'"
                )
            if not jid.user:
                raise ValueError(
                    f"Invalid group JID '{jid_str}'. Missing group ID part."
                )
        return v

    @field_validator(
        "qa_test_groups",
        "active_groups",
        "auto_reply_groups",
        "kb_exclude_subject_prefixes",
        "escalation_primary_jids",
        "escalation_secondary_jids",
        mode="before",
    )
    @classmethod
    def parse_group_lists(cls, v: object) -> list[str]:
        """Accept JSON arrays and comma-separated, trimmed group JIDs."""
        if v is None:
            return []
        if isinstance(v, str):
            raw = v.strip()
            if not raw:
                return []
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                values = raw.split(",")
            else:
                values = decoded if isinstance(decoded, list) else raw.split(",")
        elif isinstance(v, list):
            values = v
        else:
            values = [v]
        return [str(value).strip() for value in values if str(value).strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        arbitrary_types_allowed=True,
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_model_credentials(self) -> Self:
        """Require the API key for the provider `model_name` resolves to.

        Mirrors pydantic-ai's resolution order: an explicit `provider:` prefix first,
        then its legacy unprefixed names. Providers we don't manage are left alone —
        pydantic-ai is the authority on what it supports and raises its own error.
        """
        if not self.ai_enabled:
            return self

        model_name = self.model_name.strip()
        provider, sep, _ = model_name.partition(":")
        if not sep:
            provider = next(
                (
                    name
                    for prefix, name in _LEGACY_NAME_PROVIDERS.items()
                    if model_name.startswith(prefix)
                ),
                "",
            )

        managed = _MANAGED_PROVIDERS.get(provider)
        if managed is None:
            return self

        attr, env_var = managed
        if not getattr(self, attr):
            raise ValueError(
                f"model_name '{self.model_name}' uses the '{provider}' provider, "
                f"so {env_var} must be set."
            )
        return self

    @model_validator(mode="after")
    def apply_env(self) -> Self:
        if self.anthropic_api_key:
            environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key

        if self.openrouter_api_key:
            environ["OPENROUTER_API_KEY"] = self.openrouter_api_key

        if self.deepseek_api_key:
            environ["DEEPSEEK_API_KEY"] = self.deepseek_api_key
        environ["DEEPSEEK_BASE_URL"] = self.deepseek_base_url
        environ["DEEPSEEK_MODEL"] = self.deepseek_model

        if self.kimi_api_key:
            environ["KIMI_API_KEY"] = self.kimi_api_key
        environ["KIMI_BASE_URL"] = self.kimi_base_url
        environ["KIMI_MODEL"] = self.kimi_model

        if self.groq_api_key:
            environ["GROQ_API_KEY"] = self.groq_api_key

        if self.gemini_api_key:
            environ["GEMINI_API_KEY"] = self.gemini_api_key
        environ["GEMINI_MODEL"] = self.gemini_model

        if self.nvidia_api_key:
            environ["NVIDIA_API_KEY"] = self.nvidia_api_key

        environ["LLM_PROVIDER_ORDER"] = self.llm_provider_order
        if self.google_api_key:
            environ["GOOGLE_API_KEY"] = self.google_api_key

        if self.logfire_token:
            environ["LOGFIRE_TOKEN"] = self.logfire_token

        return self


@lru_cache
def get_settings() -> Settings:
    # Use model_validate({}) to trigger Pydantic's validation and environment variable loading
    # without passing arguments directly, which satisfies type checkers that would otherwise
    # complain about missing required fields in __init__.
    return Settings.model_validate({})
