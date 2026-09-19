"""Remove generated/test hackathon inquiry topics from the two target groups."""

from __future__ import annotations

import asyncio
import os

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine


TARGET_GROUP_JIDS = (
    "120363413079068449@g.us",
    "120363428762021095@g.us",
)

_MATCH_SQL = """
    subject ILIKE 'Hackathon inquiry%'
    OR subject ILIKE '%what is this hackathon about%'
"""
_TOPIC_FILTER_SQL = f"""
    ({_MATCH_SQL})
    AND group_jid IN (:group_one, :group_two)
    AND COALESCE(speakers, '') <> 'faq'
"""


async def purge_topics(connection: AsyncConnection) -> tuple[int, int]:
    """Delete links first, then matching non-FAQ topics, and return row counts."""
    params = {
        "group_one": TARGET_GROUP_JIDS[0],
        "group_two": TARGET_GROUP_JIDS[1],
    }
    linked = await connection.execute(
        text(
            f"""
            DELETE FROM public.kb_topic_message
            WHERE kb_topic_id IN (
                SELECT id FROM public.kbtopic
                WHERE {_TOPIC_FILTER_SQL}
            )
            """
        ),
        params,
    )
    topics = await connection.execute(
        text(f"DELETE FROM public.kbtopic WHERE {_TOPIC_FILTER_SQL}"),
        params,
    )
    return max(linked.rowcount or 0, 0), max(topics.rowcount or 0, 0)


async def run(uri: str) -> tuple[int, int]:
    engine = create_async_engine(uri, pool_pre_ping=True)
    try:
        async with engine.begin() as connection:
            return await purge_topics(connection)
    finally:
        await engine.dispose()


def main() -> None:
    uri = os.environ.get("DB_URI")
    if not uri:
        print("DB_URI not set")
        raise SystemExit(1)
    try:
        linked, topics = asyncio.run(run(uri))
    except Exception as error:
        print(f"purge failed: {type(error).__name__}")
        raise SystemExit(1) from None
    print(f"kb_topic_message_deleted={linked}")
    print(f"kbtopic_deleted={topics}")


if __name__ == "__main__":
    main()
