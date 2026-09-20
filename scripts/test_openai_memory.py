"""Live synthetic memory test: real DB and AI; rolled back, never sends WhatsApp.

Run with PYTHONPATH=src python scripts/test_openai_memory.py inside the backend.
Only synthetic test text goes to providers. No existing group is read or activated.
"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from time import perf_counter
from unittest.mock import AsyncMock
from uuid import uuid4

from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from voyageai.client_async import AsyncClient

from config import get_settings
from handler.knowledge_base_answers import KnowledgeBaseAnswers
from models import Group, KBTopic, Message, Sender
from whatsapp.jid import JID
from models.kb_topic_message import KBTopicMessage
from utils.voyage_embed_text import voyage_embed_text


async def main():
    settings = get_settings()
    assert settings.model_name.startswith("openai"), "Configure an OpenAI model first"
    engine = create_async_engine(settings.db_uri)
    embeddings = AsyncClient(api_key=settings.voyage_api_key, max_retries=0)
    suffix = str(uuid4().int)[:18]
    group_id, sender_id = f"{suffix}@g.us", f"{suffix}@s.whatsapp.net"
    now = datetime.now(UTC)
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            try:
                session.add(Sender(jid=sender_id, push_name="Synthetic tester"))
                session.add(
                    Group(
                        group_jid=group_id,
                        group_name="Synthetic OpenAI test",
                        managed=False,
                    )
                )
                await session.flush()
                old = Message(
                    message_id=f"{suffix}-old",
                    chat_jid=group_id,
                    group_jid=group_id,
                    sender_jid=sender_id,
                    timestamp=now - timedelta(minutes=2),
                    text="We proposed Friday at 10:00 for the demo.",
                )
                new = Message(
                    message_id=f"{suffix}-new",
                    chat_jid=group_id,
                    group_jid=group_id,
                    sender_jid=sender_id,
                    timestamp=now - timedelta(minutes=1),
                    text="Correction: the demo is confirmed for Saturday at 14:00. Friday is cancelled. No budget has been agreed.",
                )
                session.add_all([old, new])
                await session.flush()
                vector = (await voyage_embed_text(embeddings, [old.text or ""]))[0]
                session.add(
                    KBTopic(
                        id=f"{suffix}-topic",
                        group_jid=group_id,
                        speakers="tester",
                        subject="Demo plan",
                        summary=old.text or "",
                        embedding=vector,
                        start_time=old.timestamp,
                    )
                )
                await session.flush()
                session.add(
                    KBTopicMessage(
                        kb_topic_id=f"{suffix}-topic", message_id=old.message_id
                    )
                )
                await session.flush()
                whatsapp = AsyncMock()
                whatsapp.get_my_jid.return_value = JID(
                    user="999000", server="s.whatsapp.net"
                )
                handler = KnowledgeBaseAnswers(session, whatsapp, embeddings, settings)
                handler.send_message = AsyncMock()
                for question in [
                    "When is the demo?",
                    "What is the agreed budget for the demo?",
                ]:
                    message = Message(
                        message_id=f"{suffix}-{uuid4().hex}",
                        chat_jid=group_id,
                        group_jid=group_id,
                        sender_jid=sender_id,
                        timestamp=now,
                        text=question,
                    )
                    message.group = await session.get(Group, group_id)
                    session.add(message)
                    await session.flush()
                    start = perf_counter()
                    await handler(message)
                    reply = handler.send_message.call_args.args[1]
                    print(
                        json.dumps(
                            {
                                "question": question,
                                "reply": reply,
                                "seconds": round(perf_counter() - start, 2),
                            }
                        ),
                        flush=True,
                    )
                    if question.startswith("When"):
                        assert "saturday" in reply.lower() and (
                            "14:00" in reply or "2" in reply
                        ), "Latest correction was not used"
                    else:
                        assert any(
                            word in reply.lower()
                            for word in [
                                "not",
                                "no ",
                                "hasn’t",
                                "hasn't",
                                "isn't",
                                "missing",
                                "unknown",
                            ]
                        ), "Missing information was not acknowledged"
            finally:
                await session.rollback()
                print(
                    "Synthetic transaction rolled back; no WhatsApp messages sent.",
                    flush=True,
                )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
