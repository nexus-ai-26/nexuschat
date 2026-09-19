from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from scripts.seed_faq import (
    DEFAULT_GROUP_JIDS,
    FAQEntry,
    build_topics,
    faq_topic_id,
    parse_faq,
    seed_topics,
)


def test_parse_faq_and_stable_ids():
    entries = parse_faq(
        "Q: What is this hackathon about?\n"
        "A: Build useful projects.\n\n"
        "Q: When is it?\n"
        "A: Replace the date.\n"
    )

    assert entries == [
        FAQEntry("What is this hackathon about?", "Build useful projects."),
        FAQEntry("When is it?", "Replace the date."),
    ]
    assert faq_topic_id(DEFAULT_GROUP_JIDS[0], 0) == faq_topic_id(
        DEFAULT_GROUP_JIDS[0], 0
    )
    assert faq_topic_id(DEFAULT_GROUP_JIDS[0], 0) != faq_topic_id(
        DEFAULT_GROUP_JIDS[0], 1
    )


@pytest.mark.asyncio
async def test_seed_topics_uses_mocked_voyage_and_upserts_both_groups():
    session = Mock()
    session.get = AsyncMock(return_value=None)
    session.add = Mock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    embedding_client = Mock()
    embedding_client.embed = AsyncMock(
        return_value=SimpleNamespace(
            embeddings=[[0.0] * 1024, [1.0] * 1024], total_tokens=8
        )
    )
    entries = [
        FAQEntry("What is this hackathon about?", "Build useful projects."),
        FAQEntry("How do we submit?", "Use the submission form."),
    ]
    bulk_upsert_mock = AsyncMock()

    with patch("scripts.seed_faq.bulk_upsert", bulk_upsert_mock):
        count = await seed_topics(session, embedding_client, entries)

    assert count == 4
    embedding_client.embed.assert_awaited_once_with(
        [
            "# What is this hackathon about?\nBuild useful projects.",
            "# How do we submit?\nUse the submission form.",
        ],
        model="voyage-3",
        input_type="document",
    )
    topics = bulk_upsert_mock.await_args.args[1]
    assert len(topics) == 4
    assert [topic.group_jid for topic in topics] == [
        DEFAULT_GROUP_JIDS[0],
        DEFAULT_GROUP_JIDS[0],
        DEFAULT_GROUP_JIDS[1],
        DEFAULT_GROUP_JIDS[1],
    ]
    assert len({topic.id for topic in topics}) == 4
    session.commit.assert_awaited_once()


def test_build_topics_reuses_ids_for_reruns():
    entries = [FAQEntry("Q", "A")]
    first = build_topics(entries, DEFAULT_GROUP_JIDS, [[0.0] * 1024])
    second = build_topics(entries, DEFAULT_GROUP_JIDS, [[0.0] * 1024])

    assert [topic.id for topic in first] == [topic.id for topic in second]
