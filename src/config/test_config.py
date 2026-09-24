from os import environ
from typing import Any

import pytest
from pydantic import ValidationError

from config import Settings

# Required fields unrelated to LLM provider selection, so each test only has to
# state the model/key combination it actually cares about.
_BASE: dict[str, Any] = {
    "db_uri": "db-test://localhost/test",
    "whatsapp_host": "http://localhost:3000",
    "voyage_api_key": "voyage-test",
    "logfire_token": "logfire-test",
}

_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "GOOGLE_API_KEY",
    "LOGFIRE_TOKEN",
    "MODEL_NAME",
    "AI_ENABLED",
    "QA_TEST_GROUPS",
    "AUTO_REPLY_GROUPS",
    "KB_EXCLUDE_SUBJECT_PREFIXES",
    "GEMINI_MODEL",
    "LOGFIRE_SEND_TO_LOGFIRE",
    "MAX_REPLY_CHARS",
)

_OPENROUTER_MODEL = "openrouter:anthropic/claude-sonnet-4.6"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from ambient credentials.

    Registering each variable with monkeypatch first is what allows the direct
    ``os.environ`` writes in ``Settings.apply_env`` to be rolled back afterwards.
    """
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def build(**overrides: Any) -> Settings:
    # `_env_file=None` keeps a developer's local .env from leaking into the test.
    # It goes inside the unpacked dict because pydantic's synthesised __init__
    # signature hides BaseSettings' underscore-prefixed parameters from pyright.
    return Settings(**{**_BASE, **overrides, "_env_file": None})


def test_anthropic_model_with_key_exports_env_var():
    settings = build(
        model_name="anthropic:claude-sonnet-4-6",
        anthropic_api_key="anthropic-test-key",
    )

    assert settings.model_name == "anthropic:claude-sonnet-4-6"
    assert environ["ANTHROPIC_API_KEY"] == "anthropic-test-key"


def test_openrouter_model_needs_no_anthropic_key():
    settings = build(
        model_name=_OPENROUTER_MODEL, openrouter_api_key="openrouter-test-key"
    )

    assert settings.anthropic_api_key is None
    assert environ["OPENROUTER_API_KEY"] == "openrouter-test-key"
    assert "ANTHROPIC_API_KEY" not in environ


def test_openrouter_model_without_key_is_rejected():
    with pytest.raises(ValidationError, match="OPENROUTER_API_KEY"):
        build(model_name=_OPENROUTER_MODEL)


def test_anthropic_model_without_key_is_rejected():
    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
        build(model_name="anthropic:claude-sonnet-4-6")


def test_openai_model_requires_its_managed_key():
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        build(model_name="openai:gpt-5")


def test_openai_responses_model_requires_its_managed_key():
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        build(model_name="openai-responses:gpt-5.6-luna")


def test_openai_responses_exact_model_name_is_accepted_with_key():
    settings = build(
        model_name="openai-responses:gpt-5.6-luna",
        openai_api_key="openai-test-key",
    )

    assert settings.model_name == "openai-responses:gpt-5.6-luna"


def test_unmanaged_provider_passes_through_without_a_key():
    # pydantic-ai owns credential resolution for providers we don't manage.
    settings = build(model_name="mistral:mistral-large-latest")

    assert settings.model_name == "mistral:mistral-large-latest"


def test_legacy_unprefixed_claude_name_still_requires_anthropic_key():
    # pydantic-ai maps the bare `claude...` name onto the anthropic provider,
    # so the credential check has to follow it there.
    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
        build(model_name="claude-sonnet-4-6")


def test_test_model_needs_no_credentials():
    assert build(model_name="test").model_name == "test"


def test_both_keys_are_exported_when_both_are_set():
    build(
        model_name=_OPENROUTER_MODEL,
        anthropic_api_key="anthropic-test-key",
        openrouter_api_key="openrouter-test-key",
    )

    assert environ["ANTHROPIC_API_KEY"] == "anthropic-test-key"
    assert environ["OPENROUTER_API_KEY"] == "openrouter-test-key"


def test_default_model_is_gemini():
    settings = build(google_api_key="google-test")
    assert settings.model_name == "google-gla:gemini-2.5-flash"
    assert environ["GOOGLE_API_KEY"] == "google-test"


def test_gemini_requires_key_when_ai_enabled():
    with pytest.raises(ValidationError, match="GOOGLE_API_KEY"):
        build()


def test_ai_disabled_allows_startup_without_provider_key():
    settings = build(ai_enabled=False)
    assert settings.google_api_key is None
    assert settings.ai_enabled is False


def test_auto_reply_groups_parse_comma_separated_values():
    settings = build(
        model_name="test",
        auto_reply_groups="  first@g.us, ,second@g.us,  ",
    )

    assert settings.auto_reply_groups == ["first@g.us", "second@g.us"]


def test_group_settings_parse_json_and_comma_environment_values(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("QA_TEST_GROUPS", '["qa@g.us", "other@g.us"]')
    monkeypatch.setenv("AUTO_REPLY_GROUPS", "auto@g.us, ,second@g.us")
    monkeypatch.setenv("KB_EXCLUDE_SUBJECT_PREFIXES", "Hackathon inquiry, Test topic")

    settings = Settings(**{**_BASE, "model_name": "test", "_env_file": None})

    assert settings.qa_test_groups == ["qa@g.us", "other@g.us"]
    assert settings.auto_reply_groups == ["auto@g.us", "second@g.us"]
    assert settings.kb_exclude_subject_prefixes == [
        "Hackathon inquiry",
        "Test topic",
    ]


def test_gemini_model_default_and_environment_override(monkeypatch: pytest.MonkeyPatch):
    default = build(model_name="test")
    assert default.gemini_model == "gemini-3.6-flash"

    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    configured = Settings(**{**_BASE, "model_name": "test", "_env_file": None})
    assert configured.gemini_model == "gemini-test-model"


def test_recent_message_context_limit_defaults_to_twenty_and_accepts_one_hundred():
    default = build(model_name="test")
    maximum = build(model_name="test", recent_message_context_limit=100)

    assert default.recent_message_context_limit == 20
    assert maximum.recent_message_context_limit == 100


def test_recent_message_context_limit_rejects_values_above_one_hundred():
    with pytest.raises(ValidationError, match="recent_message_context_limit"):
        build(model_name="test", recent_message_context_limit=101)


def test_max_reply_chars_defaults_to_five_thousand_and_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
):
    default = build(model_name="test")
    assert default.max_reply_chars == 5000

    monkeypatch.setenv("MAX_REPLY_CHARS", "6000")
    configured = build(model_name="test")
    assert configured.max_reply_chars == 6000


def test_max_reply_chars_rejects_a_small_total_budget():
    with pytest.raises(ValidationError, match="max_reply_chars"):
        build(model_name="test", max_reply_chars=4999)
