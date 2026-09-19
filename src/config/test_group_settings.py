from typing import Any

import pytest

from config import Settings


_BASE: dict[str, Any] = {
    "db_uri": "db-test://localhost/test",
    "whatsapp_host": "http://localhost:3000",
    "voyage_api_key": "voyage-test",
    "logfire_token": "logfire-test",
    "model_name": "test",
}


def test_group_lists_accept_json_and_csv_environment_values(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("QA_TEST_GROUPS", '["qa@g.us", "other@g.us"]')
    monkeypatch.setenv("AUTO_REPLY_GROUPS", "auto@g.us, ,second@g.us")
    monkeypatch.setenv("KB_EXCLUDE_SUBJECT_PREFIXES", "Hackathon inquiry, Test topic")

    settings = Settings(**{**_BASE, "_env_file": None})

    assert settings.qa_test_groups == ["qa@g.us", "other@g.us"]
    assert settings.auto_reply_groups == ["auto@g.us", "second@g.us"]
    assert settings.kb_exclude_subject_prefixes == [
        "Hackathon inquiry",
        "Test topic",
    ]


def test_kb_exclude_subject_prefix_default():
    settings = Settings(**{**_BASE, "_env_file": None})

    assert settings.kb_exclude_subject_prefixes == ["Hackathon inquiry"]
