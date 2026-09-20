from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.exceptions import ModelHTTPError

from utils.llm_provider import (
    ProviderConfigurationError,
    ProviderFallbackError,
    _provider_circuit_open_until,
    _model_chain,
    run_with_provider_fallback,
)


@pytest.fixture(autouse=True)
def reset_provider_circuits():
    _provider_circuit_open_until.clear()
    yield
    _provider_circuit_open_until.clear()


def settings(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "model_name": "openrouter:configured-model",
        "llm_provider_order": "deepseek,gemini,kimi,openrouter,nvidia,groq",
        "deepseek_api_key": "deepseek-test-key",
        "deepseek_base_url": "https://api.deepseek.com",
        "deepseek_model": "deepseek-chat",
        "gemini_api_key": "gemini-test-key",
        "gemini_model": "gemini-3.6-flash",
        "kimi_api_key": "kimi-test-key",
        "kimi_base_url": "https://api.moonshot.ai/v1",
        "kimi_model": "kimi-k2-turbo-preview",
        "openrouter_api_key": "openrouter-test-key",
        "nvidia_api_key": "nvidia-test-key",
        "groq_api_key": "groq-test-key",
        "anthropic_api_key": "anthropic-test-key",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Agent:
    def __init__(self, run: AsyncMock):
        self.run = run


@pytest.mark.asyncio
async def test_primary_succeeds():
    run = AsyncMock(return_value=AgentRunResult(output="primary response"))

    with patch("utils.llm_provider.Agent", return_value=_Agent(run)):
        result = await run_with_provider_fallback(
            settings(), system_prompt="system", prompt="question"
        )

    assert result.output == "primary response"
    run.assert_awaited_once_with("question")


@pytest.mark.asyncio
async def test_openai_provider_is_selected_and_uses_direct_model():
    run = AsyncMock(return_value=AgentRunResult(output="openai response"))

    with patch("utils.llm_provider.Agent", return_value=_Agent(run)) as agent:
        result = await run_with_provider_fallback(
            settings(
                llm_provider_order="openai",
                openai_api_key="openai-test-key",
            ),
            system_prompt="system",
            prompt="question",
        )

    assert result.output == "openai response"
    assert agent.call_args.kwargs["model"].model_name == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_primary_429_falls_back_to_second():
    first = AsyncMock(side_effect=ModelHTTPError(429, "deepseek-chat"))
    second = AsyncMock(return_value=AgentRunResult(output="gemini response"))

    with patch(
        "utils.llm_provider.Agent",
        side_effect=[_Agent(first), _Agent(second)],
    ):
        result = await run_with_provider_fallback(
            settings(), system_prompt="system", prompt="question"
        )

    assert result.output == "gemini response"
    first.assert_awaited_once_with("question")
    second.assert_awaited_once_with("question")


@pytest.mark.asyncio
async def test_primary_and_second_fail_third_succeeds():
    first = AsyncMock(side_effect=ModelHTTPError(503, "deepseek-chat"))
    second = AsyncMock(side_effect=TimeoutError("gemini timed out"))
    third = AsyncMock(return_value=AgentRunResult(output="kimi response"))

    with patch(
        "utils.llm_provider.Agent",
        side_effect=[_Agent(first), _Agent(second), _Agent(third)],
    ):
        result = await run_with_provider_fallback(
            settings(), system_prompt="system", prompt="question"
        )

    assert result.output == "kimi response"


@pytest.mark.asyncio
async def test_auth_error_falls_through():
    first = AsyncMock(side_effect=ModelHTTPError(401, "deepseek-chat"))
    second = AsyncMock(return_value=AgentRunResult(output="fallback response"))

    with patch(
        "utils.llm_provider.Agent",
        side_effect=[_Agent(first), _Agent(second)],
    ):
        result = await run_with_provider_fallback(
            settings(), system_prompt="system", prompt="question"
        )

    assert result.output == "fallback response"


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [402, 404])
async def test_provider_http_errors_fall_through(status_code: int):
    first = AsyncMock(side_effect=ModelHTTPError(status_code, "deepseek-chat"))
    second = AsyncMock(return_value=AgentRunResult(output="fallback response"))

    with patch(
        "utils.llm_provider.Agent",
        side_effect=[_Agent(first), _Agent(second)],
    ):
        result = await run_with_provider_fallback(
            settings(), system_prompt="system", prompt="question"
        )

    assert result.output == "fallback response"


@pytest.mark.asyncio
async def test_circuit_breaker_skips_provider_for_subsequent_request():
    failed = AsyncMock(side_effect=ModelHTTPError(401, "deepseek-chat"))
    first_fallback = AsyncMock(return_value=AgentRunResult(output="first fallback"))

    with patch(
        "utils.llm_provider.Agent",
        side_effect=[_Agent(failed), _Agent(first_fallback)],
    ):
        first_result = await run_with_provider_fallback(
            settings(llm_provider_order="deepseek,kimi"),
            system_prompt="system",
            prompt="question",
        )

    second_fallback = AsyncMock(return_value=AgentRunResult(output="second fallback"))
    with patch(
        "utils.llm_provider.Agent", return_value=_Agent(second_fallback)
    ) as agent:
        second_result = await run_with_provider_fallback(
            settings(llm_provider_order="deepseek,kimi"),
            system_prompt="system",
            prompt="question",
        )

    assert first_result.output == "first fallback"
    assert second_result.output == "second fallback"
    assert agent.call_count == 1
    assert second_fallback.await_count == 1


@pytest.mark.asyncio
async def test_all_providers_fail_with_graceful_error():
    first = AsyncMock(side_effect=ModelHTTPError(429, "deepseek-chat"))
    second = AsyncMock(side_effect=ModelHTTPError(500, "gemini-2.0-flash"))
    third = AsyncMock(side_effect=ConnectionError("connection refused"))

    with patch(
        "utils.llm_provider.Agent",
        side_effect=[_Agent(first), _Agent(second), _Agent(third)],
    ):
        with pytest.raises(
            ProviderFallbackError, match="All active LLM providers failed"
        ):
            await run_with_provider_fallback(
                settings(
                    llm_provider_order="deepseek,gemini,kimi",
                ),
                system_prompt="system",
                prompt="question",
            )


@pytest.mark.asyncio
async def test_empty_key_provider_is_skipped():
    run = AsyncMock(return_value=AgentRunResult(output="groq response"))

    with patch("utils.llm_provider.Agent", return_value=_Agent(run)) as agent:
        result = await run_with_provider_fallback(
            settings(
                deepseek_api_key="",
                llm_provider_order="deepseek,groq",
            ),
            system_prompt="system",
            prompt="question",
        )

    assert result.output == "groq response"
    assert agent.call_args.kwargs["model"].model_name == "llama-3.3-70b-versatile"


def test_custom_provider_order_is_respected():
    chain = _model_chain(
        settings(
            llm_provider_order="groq, kimi, unknown, deepseek",
        )
    )

    assert [candidate.provider for candidate in chain] == ["groq", "kimi", "deepseek"]


def test_no_active_provider_is_a_clear_configuration_error():
    with pytest.raises(ProviderConfigurationError, match="No active LLM providers"):
        _model_chain(
            settings(
                llm_provider_order="deepseek,gemini,kimi",
                deepseek_api_key="",
                gemini_api_key="",
                kimi_api_key="",
            )
        )


@pytest.mark.asyncio
async def test_non_provider_error_does_not_fall_through():
    error = ValueError("invalid request")
    run = AsyncMock(side_effect=error)

    with patch("utils.llm_provider.Agent", return_value=_Agent(run)):
        with pytest.raises(ValueError, match="invalid request"):
            await run_with_provider_fallback(
                settings(), system_prompt="system", prompt="question"
            )
