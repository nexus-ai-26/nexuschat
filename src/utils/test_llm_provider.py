from typing import Any
import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.openai import OpenAIResponsesModel

from config import Settings
from utils.llm_provider import (
    ProviderConfigurationError,
    ProviderFallbackError,
    _openai_direct_model,
    _model_chain,
    _provider_circuit_open_until,
    _provider_rate_limited_at,
    provider_rate_limit_recently,
    run_with_provider_fallback,
)


@pytest.fixture(autouse=True)
def reset_provider_circuits():
    _provider_circuit_open_until.clear()
    _provider_rate_limited_at.clear()
    yield
    _provider_circuit_open_until.clear()
    _provider_rate_limited_at.clear()


def settings(**overrides: object) -> Settings:
    values: dict[str, Any] = {
        "model_name": "openrouter:configured-model",
        "llm_provider_order": "deepseek,gemini,kimi,openrouter,nvidia,groq",
        "deepseek_api_key": "deepseek-test-key",
        "deepseek_base_url": "https://api.deepseek.com",
        "deepseek_model": "deepseek-flash",
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
    return Settings.model_construct(**values)


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


def test_recent_rate_limit_state_is_visible_to_catchup():
    _provider_rate_limited_at["deepseek"] = time.monotonic()
    assert provider_rate_limit_recently(300)


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


def test_openai_responses_uses_exact_configured_model_name():
    chain = _model_chain(
        settings(
            model_name="openai-responses:gpt-5.6-luna",
            llm_provider_order="openai",
            openai_api_key="openai-test-key",
        )
    )

    assert len(chain) == 1
    assert chain[0].provider == "openai"
    assert isinstance(chain[0].model, OpenAIResponsesModel)
    assert chain[0].model.model_name == "gpt-5.6-luna"
    assert chain[0].name == "openai-responses:gpt-5.6-luna"
    assert chain[0].model_settings == {"openai_reasoning_effort": "medium"}


def test_owner_provider_order_is_built_in_configured_order():
    chain = _model_chain(
        settings(
            model_name="openai-responses:gpt-5.6-luna",
            llm_provider_order="openai,deepseek,gemini,kimi,openrouter,nvidia,groq",
            openai_api_key="openai-test-key",
            deepseek_model="deepseek-flash",
        )
    )

    assert [candidate.provider for candidate in chain] == [
        "openai",
        "deepseek",
        "gemini",
        "kimi",
        "openrouter",
        "nvidia",
        "groq",
    ]
    assert chain[1].model.model_name == "deepseek-flash"
    assert chain[0].model_settings == {"openai_reasoning_effort": "medium"}
    assert chain[1].model_settings == {"openai_reasoning_effort": "high"}


@pytest.mark.asyncio
async def test_openai_retryable_failure_falls_back_to_deepseek():
    first = AsyncMock(side_effect=ModelHTTPError(503, "gpt-5.6-luna"))
    second = AsyncMock(return_value=AgentRunResult(output="deepseek response"))

    with patch(
        "utils.llm_provider.Agent",
        side_effect=[_Agent(first), _Agent(second)],
    ) as agent:
        result = await run_with_provider_fallback(
            settings(
                model_name="openai-responses:gpt-5.6-luna",
                llm_provider_order="openai,deepseek",
                openai_api_key="openai-test-key",
                deepseek_model="deepseek-flash",
            ),
            system_prompt="system",
            prompt="question",
        )

    assert result.output == "deepseek response"
    first.assert_awaited_once_with("question")
    second.assert_awaited_once_with("question")
    assert agent.call_args_list[0].kwargs["model_settings"] == {
        "openai_reasoning_effort": "medium"
    }
    assert agent.call_args_list[1].kwargs["model_settings"] == {
        "openai_reasoning_effort": "high"
    }


@pytest.mark.asyncio
async def test_openai_provider_served_log_is_emitted(caplog):
    run = AsyncMock(return_value=AgentRunResult(output="openai response"))

    with (
        patch("utils.llm_provider.Agent", return_value=_Agent(run)),
        caplog.at_level("INFO", logger="utils.llm_provider"),
    ):
        await run_with_provider_fallback(
            settings(
                llm_provider_order="openai",
                openai_api_key="openai-test-key",
            ),
            system_prompt="system",
            prompt="question",
        )

    assert "LLM provider served provider=openai" in caplog.text


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
@pytest.mark.parametrize("status_code", [403, 410])
async def test_permanent_provider_errors_open_one_hour_circuit(
    status_code: int, caplog: pytest.LogCaptureFixture
):
    first = AsyncMock(side_effect=ModelHTTPError(status_code, "deepseek-chat"))
    second = AsyncMock(return_value=AgentRunResult(output="fallback response"))

    with patch(
        "utils.llm_provider.Agent",
        side_effect=[_Agent(first), _Agent(second)],
    ):
        result = await run_with_provider_fallback(
            settings(llm_provider_order="deepseek,kimi"),
            system_prompt="system",
            prompt="question",
        )

    assert result.output == "fallback response"
    assert _provider_circuit_open_until["deepseek"] > 0
    assert (
        sum(
            "Provider circuit opened provider=deepseek" in r.message
            for r in caplog.records
        )
        == 1
    )


class _RetryAfterError(Exception):
    status_code = 429
    response = SimpleNamespace(headers={"Retry-After": "120"})


@pytest.mark.asyncio
async def test_rate_limit_uses_retry_after_for_cooldown():
    first = AsyncMock(side_effect=_RetryAfterError())
    second = AsyncMock(return_value=AgentRunResult(output="fallback response"))

    with patch(
        "utils.llm_provider.Agent",
        side_effect=[_Agent(first), _Agent(second)],
    ):
        result = await run_with_provider_fallback(
            settings(llm_provider_order="deepseek,kimi"),
            system_prompt="system",
            prompt="question",
        )

    assert result.output == "fallback response"
    assert 100 <= _provider_circuit_open_until["deepseek"] - time.monotonic() <= 121


@pytest.mark.asyncio
async def test_provider_timeout_is_a_fallback_failure():
    async def never_finishes(_prompt: str):
        await asyncio.sleep(1)

    first = AsyncMock(side_effect=never_finishes)
    second = AsyncMock(return_value=AgentRunResult(output="fallback response"))

    with patch(
        "utils.llm_provider.Agent",
        side_effect=[_Agent(first), _Agent(second)],
    ):
        result = await run_with_provider_fallback(
            settings(
                llm_provider_order="deepseek,kimi",
                provider_timeout_seconds=0.01,
            ),
            system_prompt="system",
            prompt="question",
        )

    assert result.output == "fallback response"
    first.assert_awaited_once_with("question")


def test_openai_sdk_retries_are_disabled():
    with patch("utils.llm_provider.AsyncOpenAI") as client:
        _openai_direct_model(settings(openai_api_key="openai-test-key"))

    assert client.call_args.kwargs["max_retries"] == 0


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

    with (
        patch(
            "utils.llm_provider.Agent",
            side_effect=[_Agent(first), _Agent(second), _Agent(third)],
        ),
        pytest.raises(ProviderFallbackError, match="All active LLM providers failed"),
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

    with (
        patch("utils.llm_provider.Agent", return_value=_Agent(run)),
        pytest.raises(ValueError, match="invalid request"),
    ):
        await run_with_provider_fallback(
            settings(), system_prompt="system", prompt="question"
        )


def test_openai_only_chain_does_not_use_other_configured_keys():
    chain = _model_chain(
        settings(
            model_name="openai-responses:gpt-5.6-luna",
            llm_provider_order="openai",
            openai_api_key="test-key",
        )
    )
    assert [item.provider for item in chain] == ["openai"]
    assert chain[0].model.model_name == "gpt-5.6-luna"


@pytest.mark.asyncio
async def test_openai_request_uses_responses_without_remote_conversation_state(
    httpx_mock,
):
    import json

    httpx_mock.add_response(
        json={
            "id": "resp_test",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-5.6-luna",
            "output": [
                {
                    "type": "message",
                    "id": "msg_test",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Saturday confirmed",
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
        }
    )
    result = await run_with_provider_fallback(
        settings(
            model_name="openai-responses:gpt-5.6-luna",
            llm_provider_order="openai",
            openai_api_key="fake-key",
        ),
        system_prompt="Use supplied context",
        prompt="Saturday confirmed",
    )
    assert result.output == "Saturday confirmed"
    request = httpx_mock.get_request()
    assert str(request.url) == "https://api.openai.com/v1/responses"
    body = json.loads(request.content)
    assert body["store"] is False
    assert body["reasoning"]["effort"] == "medium"
    assert "previous_response_id" not in body


@pytest.mark.asyncio
async def test_openai_quota_error_keeps_status_and_is_not_retried(httpx_mock):
    httpx_mock.add_response(
        status_code=429,
        json={
            "error": {
                "message": "quota",
                "type": "insufficient_quota",
                "code": "insufficient_quota",
            }
        },
    )
    with pytest.raises(ModelHTTPError) as exc:
        await run_with_provider_fallback(
            settings(
                model_name="openai-responses:gpt-5.6-luna",
                llm_provider_order="openai",
                openai_api_key="fake-key",
            ),
            system_prompt="system",
            prompt="test",
        )
    assert exc.value.status_code == 429
    assert len(httpx_mock.get_requests()) == 1
