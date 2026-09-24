"""LLM execution helpers with configurable provider failover."""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from pydantic_ai.providers.openai import OpenAIProvider

from config import Settings

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-flash"
KIMI_BASE_URL = "https://api.moonshot.ai/v1"
KIMI_MODEL = "kimi-k2-turbo-preview"
GEMINI_MODEL = "gemini-3.6-flash"
NVIDIA_MODEL = "openai/gpt-oss-20b"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
OPENAI_MODEL = "gpt-4o-mini"
PROVIDER_TIMEOUT_SECONDS = 30.0
CIRCUIT_BREAKER_SECONDS = 60 * 60
RATE_LIMIT_COOLDOWN_SECONDS = 60.0
DEFAULT_PROVIDER_ORDER = "openai,deepseek,gemini,kimi,openrouter,nvidia,groq"
_provider_circuit_open_until: dict[str, float] = {}
_provider_rate_limited_at: dict[str, float] = {}


class ProviderConfigurationError(RuntimeError):
    """Raised when no configured LLM provider can serve a request."""


class ProviderFallbackError(RuntimeError):
    """Raised after every active provider fails with a retryable error."""


@dataclass(frozen=True)
class _ModelCandidate:
    provider: str
    model: Any
    name: str
    model_settings: dict[str, Any] | None = None


def _status_code(error: Exception) -> int | None:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def is_rate_limit_error(error: Exception) -> bool:
    """Return whether a provider explicitly rejected a request with HTTP 429."""

    return _status_code(error) == 429


def _is_retryable_provider_error(error: Exception) -> bool:
    status_code = _status_code(error)
    if status_code is not None and 400 <= status_code <= 599:
        return True

    return isinstance(
        error,
        (
            asyncio.TimeoutError,
            TimeoutError,
            ConnectionError,
            OSError,
            httpx.TimeoutException,
            httpx.NetworkError,
        ),
    )


def _provider_is_circuit_open(provider: str) -> bool:
    return _provider_circuit_open_until.get(provider, 0.0) > time.monotonic()


def provider_rate_limit_recently(window_seconds: float = 300.0) -> bool:
    """Return whether any provider was rate limited during the recent window."""
    now = time.monotonic()
    return any(
        now - timestamp <= window_seconds
        for timestamp in _provider_rate_limited_at.values()
    )


def _retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    value = headers.get("Retry-After") if headers is not None else None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            retry_at = datetime.strptime(value, "%a, %d %b %Y %H:%M:%S GMT")
        except (TypeError, ValueError):
            return None
        return max(
            0.0,
            (
                retry_at.replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)
            ).total_seconds(),
        )


def _open_provider_circuit(
    provider: str,
    status_code: int | None,
    error: Exception,
    settings: Settings,
) -> None:
    now = time.monotonic()
    if status_code in {401, 403, 404, 410}:
        cooldown = float(
            getattr(
                settings, "provider_permanent_cooldown_seconds", CIRCUIT_BREAKER_SECONDS
            )
        )
        already_open = _provider_circuit_open_until.get(provider, 0.0) > now
        _provider_circuit_open_until[provider] = now + cooldown
        if not already_open:
            logger.error(
                "Provider circuit opened provider=%s status=%s cooldown_seconds=%s",
                provider,
                status_code,
                int(cooldown),
            )
    elif status_code == 429:
        _provider_rate_limited_at[provider] = now
        cooldown = _retry_after_seconds(error)
        if cooldown is None:
            cooldown = float(
                getattr(
                    settings,
                    "provider_rate_limit_cooldown_seconds",
                    RATE_LIMIT_COOLDOWN_SECONDS,
                )
            )
        _provider_circuit_open_until[provider] = now + cooldown
        logger.warning(
            "Provider rate limited provider=%s cooldown_seconds=%s",
            provider,
            int(cooldown),
        )


def _openai_model(
    model_name: str,
    *,
    base_url: str,
    api_key: str,
) -> OpenAIChatModel:
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=PROVIDER_TIMEOUT_SECONDS,
        max_retries=0,
    )
    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(openai_client=client),
    )


def configured_model(
    settings: Settings,
) -> OpenAIResponsesModel | OpenAIChatModel | str:
    """Resolve direct agent calls with the same OpenAI policy as the provider chain."""
    prefix, _, name = settings.model_name.partition(":")
    if prefix not in {"openai", "openai-chat", "openai-responses"}:
        return settings.model_name
    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url="https://api.openai.com/v1",
        timeout=PROVIDER_TIMEOUT_SECONDS,
        max_retries=0,
    )
    options = OpenAIResponsesModelSettings(
        openai_store=False,
        timeout=PROVIDER_TIMEOUT_SECONDS,
        openai_reasoning_effort="medium",
    )
    model_class = (
        OpenAIResponsesModel if prefix == "openai-responses" else OpenAIChatModel
    )
    return model_class(
        name, provider=OpenAIProvider(openai_client=client), settings=options
    )


def _openrouter_model(settings: Settings) -> OpenAIChatModel:
    model_name = settings.model_name.partition(":")[2] or settings.model_name
    return _openai_model(
        model_name,
        base_url=OPENROUTER_BASE_URL,
        api_key=settings.openrouter_api_key or "",
    )


def _openai_direct_model(settings: Settings) -> OpenAIChatModel:
    model_name = getattr(settings, "openai_model", None) or os.getenv(
        "OPENAI_MODEL", OPENAI_MODEL
    )
    return _openai_model(
        model_name,
        base_url="https://api.openai.com/v1",
        api_key=settings.openai_api_key or "",
    )


def _openai_responses_model(settings: Settings) -> OpenAIResponsesModel:
    model_name = settings.model_name.partition(":")[2] or settings.model_name
    client = AsyncOpenAI(
        api_key=settings.openai_api_key or "",
        base_url="https://api.openai.com/v1",
        timeout=PROVIDER_TIMEOUT_SECONDS,
        max_retries=0,
    )
    model_settings = OpenAIResponsesModelSettings(
        openai_store=False,
        timeout=PROVIDER_TIMEOUT_SECONDS,
        openai_reasoning_effort="medium",
    )
    return OpenAIResponsesModel(
        model_name,
        provider=OpenAIProvider(openai_client=client),
        settings=model_settings,
    )


def _openai_primary_model(settings: Settings) -> OpenAIChatModel | OpenAIResponsesModel:
    if settings.model_name.partition(":")[0].lower() == "openai-responses":
        return _openai_responses_model(settings)
    return _openai_direct_model(settings)


def _openai_primary_name(settings: Settings) -> str:
    if settings.model_name.partition(":")[0].lower() == "openai-responses":
        return settings.model_name
    return f"openai:{getattr(settings, 'openai_model', None) or os.getenv('OPENAI_MODEL', OPENAI_MODEL)}"


def _deepseek_model(settings: Settings) -> OpenAIChatModel:
    return _openai_model(
        getattr(settings, "deepseek_model", DEEPSEEK_MODEL),
        base_url=getattr(settings, "deepseek_base_url", DEEPSEEK_BASE_URL),
        api_key=settings.deepseek_api_key or "",
    )


def _kimi_model(settings: Settings) -> OpenAIChatModel:
    return _openai_model(
        getattr(settings, "kimi_model", KIMI_MODEL),
        base_url=getattr(settings, "kimi_base_url", KIMI_BASE_URL),
        api_key=settings.kimi_api_key or "",
    )


def _groq_model(settings: Settings) -> OpenAIChatModel:
    return _openai_model(
        "llama-3.3-70b-versatile",
        base_url=GROQ_BASE_URL,
        api_key=settings.groq_api_key or "",
    )


def _nvidia_model(settings: Settings) -> OpenAIChatModel:
    return _openai_model(
        NVIDIA_MODEL,
        base_url=NVIDIA_BASE_URL,
        api_key=settings.nvidia_api_key or "",
    )


def _provider_order(settings: Settings) -> list[str]:
    configured = getattr(settings, "llm_provider_order", DEFAULT_PROVIDER_ORDER)
    return [name.strip().lower() for name in configured.split(",") if name.strip()]


def _model_settings(provider: str, settings: Settings) -> dict[str, Any] | None:
    """Return provider-specific reasoning settings supported by Pydantic AI."""

    model_provider = getattr(settings, "model_name", "").partition(":")[0].lower()
    if provider == "openai" and model_provider == "openai-responses":
        return {"openai_reasoning_effort": "medium"}
    if provider == "deepseek":
        # DeepSeek's OpenAI-compatible API maps the requested medium thinking
        # level to its supported high reasoning effort.
        return {"openai_reasoning_effort": "high"}
    return None


def _model_chain(settings: Settings) -> list[_ModelCandidate]:
    """Build active providers in the order configured by ``LLM_PROVIDER_ORDER``."""

    model_name = getattr(settings, "model_name", "")
    model_provider = model_name.partition(":")[0].lower()
    builders = {
        "openai": (
            "openai_api_key",
            lambda: _openai_primary_model(settings),
            lambda: _openai_primary_name(settings),
        ),
        "deepseek": (
            "deepseek_api_key",
            lambda: _deepseek_model(settings),
            lambda: getattr(settings, "deepseek_model", DEEPSEEK_MODEL),
        ),
        "gemini": (
            "gemini_api_key",
            lambda: f"google-gla:{getattr(settings, 'gemini_model', GEMINI_MODEL)}",
            lambda: f"google-gla:{getattr(settings, 'gemini_model', GEMINI_MODEL)}",
        ),
        "kimi": (
            "kimi_api_key",
            lambda: _kimi_model(settings),
            lambda: getattr(settings, "kimi_model", KIMI_MODEL),
        ),
        "openrouter": (
            "openrouter_api_key",
            lambda: _openrouter_model(settings),
            lambda: model_name or "openrouter",
        ),
        "nvidia": (
            "nvidia_api_key",
            lambda: _nvidia_model(settings),
            lambda: NVIDIA_MODEL,
        ),
        "groq": (
            "groq_api_key",
            lambda: _groq_model(settings),
            lambda: "groq:llama-3.3-70b-versatile",
        ),
        "anthropic": (
            "anthropic_api_key",
            lambda: model_name,
            lambda: model_name,
        ),
    }

    candidates: list[_ModelCandidate] = []
    seen_providers: set[str] = set()
    for provider in _provider_order(settings):
        if provider in seen_providers:
            continue
        seen_providers.add(provider)

        builder = builders.get(provider)
        if builder is None:
            logger.warning(
                "Ignoring unknown LLM provider in LLM_PROVIDER_ORDER: %s", provider
            )
            continue

        key_attribute, model_builder, name_builder = builder
        if not getattr(settings, key_attribute, None):
            logger.debug("Skipping LLM provider with no API key: %s", provider)
            continue

        if provider == "anthropic" and model_provider != "anthropic":
            logger.warning(
                "Skipping anthropic because MODEL_NAME is not anthropic-prefixed: %s",
                model_name,
            )
            continue

        candidates.append(
            _ModelCandidate(
                provider=provider,
                model=model_builder(),
                name=name_builder(),
                model_settings=_model_settings(provider, settings),
            )
        )

    if not candidates:
        raise ProviderConfigurationError(
            "No active LLM providers. Set an API key for at least one provider "
            "listed in LLM_PROVIDER_ORDER."
        )

    logger.info(
        "Active LLM providers (in order): %s",
        ", ".join(candidate.provider for candidate in candidates),
    )
    return candidates


async def run_with_provider_fallback(
    settings: Settings,
    *,
    system_prompt: str,
    prompt: str,
    output_type: Any = None,
):
    """Run an agent and fall through on provider HTTP/network failures."""

    models = _model_chain(settings)
    failures: list[tuple[str, Exception]] = []

    for candidate in models:
        if _provider_is_circuit_open(candidate.provider):
            logger.info("Skipping circuit-open provider=%s", candidate.provider)
            continue
        try:
            agent_kwargs: dict[str, Any] = {
                "model": candidate.model,
                "system_prompt": system_prompt,
            }
            if output_type is not None:
                agent_kwargs["output_type"] = output_type
            if candidate.model_settings is not None:
                agent_kwargs["model_settings"] = candidate.model_settings
            result = await asyncio.wait_for(
                Agent(**agent_kwargs).run(prompt),
                timeout=float(
                    getattr(
                        settings, "provider_timeout_seconds", PROVIDER_TIMEOUT_SECONDS
                    )
                ),
            )
        except Exception as error:
            if not _is_retryable_provider_error(error):
                raise

            failures.append((candidate.provider, error))
            status_code = _status_code(error)
            _open_provider_circuit(candidate.provider, status_code, error, settings)
            logger.warning(
                "LLM provider failed provider=%s status=%s error=%s, falling back",
                candidate.provider,
                status_code if status_code is not None else "unknown",
                type(error).__name__,
            )
            continue

        logger.info(
            "LLM provider served provider=%s model=%s",
            candidate.provider,
            candidate.name,
        )
        return result

    details = ", ".join(
        f"{provider}: {type(error).__name__}" for provider, error in failures
    )
    logger.error("All active LLM providers failed: %s", details)
    if failures:
        last_error = failures[-1][1]
        if len(models) == 1 and is_rate_limit_error(last_error):
            raise last_error
        raise ProviderFallbackError(
            f"All active LLM providers failed ({details})."
        ) from last_error
    raise ProviderFallbackError("All active LLM providers are temporarily unavailable.")
