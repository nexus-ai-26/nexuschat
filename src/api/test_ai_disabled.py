from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import load_new_kbtopics_api, summarize_and_send_to_group_api
from api.deps import get_db_async_session, get_text_embebedding, get_whatsapp
from config import get_settings


@pytest.mark.parametrize(
    "path", ["/load_new_kbtopics", "/summarize_and_send_to_groups"]
)
def test_ai_endpoints_are_unavailable_when_disabled(path, monkeypatch):
    app = FastAPI()
    app.include_router(load_new_kbtopics_api.router)
    app.include_router(summarize_and_send_to_group_api.router)
    app.dependency_overrides[get_settings] = lambda: SimpleNamespace(ai_enabled=False)
    app.dependency_overrides[get_db_async_session] = lambda: None
    app.dependency_overrides[get_text_embebedding] = lambda: None
    app.dependency_overrides[get_whatsapp] = lambda: None
    summarize = AsyncMock()
    load_topics = AsyncMock()
    monkeypatch.setattr(
        summarize_and_send_to_group_api, "summarize_and_send_to_groups", summarize
    )
    monkeypatch.setattr(
        load_new_kbtopics_api.topicsLoader, "load_topics_for_all_groups", load_topics
    )

    response = TestClient(app).post(path)

    assert response.status_code == 503
    assert "AI features are disabled" in response.json()["detail"]
    summarize.assert_not_awaited()
    load_topics.assert_not_awaited()
