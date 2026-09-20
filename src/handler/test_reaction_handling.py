from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from gowa_sdk.webhooks import WebhookEnvelope
from handler.base_handler import BaseHandler
from test_utils.mock_session import AsyncSessionMock


@pytest.mark.asyncio
async def test_orphaned_reaction_is_skipped_without_foreign_key_insert():
    session = AsyncSessionMock()
    handler = BaseHandler(session, AsyncMock(), AsyncMock())
    payload = WebhookEnvelope.model_validate(
        {
            "event": "message.reaction",
            "payload": {
                "chat_id": "group@g.us",
                "from": "user@s.whatsapp.net",
                "timestamp": datetime.now(timezone.utc),
                "reaction": "👍",
                "reacted_message_id": "missing-message",
            },
        }
    )

    with patch("models.reaction.Reaction.upsert_reaction", new_callable=AsyncMock) as upsert:
        assert await handler.store_reaction(payload) is None

    upsert.assert_not_awaited()
