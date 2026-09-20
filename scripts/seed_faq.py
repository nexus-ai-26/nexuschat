"""Seed deterministic, embedded FAQ topics into the NexusChat database.

The embedding document intentionally matches ``load_new_kbtopics.load_topics``:
``# {subject}\n{summary}``, sent through ``voyage_embed_text`` with Voyage's
``voyage-3`` document embedding model.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from voyageai.client_async import AsyncClient

from models import Group, KBTopic
from models.upsert import bulk_upsert
from utils.voyage_embed_text import voyage_embed_text


DEFAULT_GROUP_JIDS = (
    "120363413079068449@g.us",
    "120363428762021095@g.us",
)
FAQ_START_TIME = datetime(2000, 1, 1, tzinfo=timezone.utc)


class SeedSettings(BaseSettings):
    db_uri: str
    voyage_api_key: str
    voyage_max_retries: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@dataclass(frozen=True)
class FAQEntry:
    question: str
    answer: str


def parse_faq(text: str) -> list[FAQEntry]:
    """Parse blank-line-separated Q:/A: FAQ entries."""
    entries: list[FAQEntry] = []
    for block_number, block in enumerate(text.split("\n\n"), start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if not lines[0].startswith("Q:"):
            raise ValueError(f"FAQ block {block_number} must start with Q:")
        answer_line = next(
            (
                index
                for index, line in enumerate(lines[1:], start=1)
                if line.startswith("A:")
            ),
            None,
        )
        if answer_line is None:
            raise ValueError(f"FAQ block {block_number} is missing A:")

        question = lines[0][2:].strip()
        answer = "\n".join(
            [lines[answer_line][2:].strip(), *lines[answer_line + 1 :]]
        ).strip()
        if not question or not answer:
            raise ValueError(
                f"FAQ block {block_number} has an empty question or answer"
            )
        entries.append(FAQEntry(question=question, answer=answer))

    if not entries:
        raise ValueError("FAQ file contains no entries")
    return entries


def faq_topic_id(group_jid: str, entry_index: int) -> str:
    """Return a stable ID so rerunning the seed updates instead of duplicating."""
    seed_key = f"nexuschat-faq-v1:{group_jid}:{entry_index}".encode()
    return hashlib.sha256(seed_key).hexdigest()


def build_topics(
    entries: list[FAQEntry],
    group_jids: tuple[str, ...],
    embeddings: list[list[float]],
) -> list[KBTopic]:
    if len(entries) != len(embeddings):
        raise ValueError(
            "Voyage returned a different number of embeddings than FAQ entries"
        )

    return [
        KBTopic(
            id=faq_topic_id(group_jid, index),
            group_jid=group_jid,
            start_time=FAQ_START_TIME,
            speakers="faq",
            subject=entry.question,
            summary=entry.answer,
            embedding=embedding,
        )
        for group_jid in group_jids
        for index, (entry, embedding) in enumerate(zip(entries, embeddings))
    ]


async def seed_topics(
    session: AsyncSession,
    embedding_client: AsyncClient,
    entries: list[FAQEntry],
    group_jids: tuple[str, ...] = DEFAULT_GROUP_JIDS,
) -> int:
    """Embed FAQ documents and upsert one deterministic topic per group/entry."""
    for group_jid in group_jids:
        if await session.get(Group, group_jid) is None:
            session.add(Group(group_jid=group_jid))
    await session.flush()

    documents = [f"# {entry.question}\n{entry.answer}" for entry in entries]
    embeddings = await voyage_embed_text(embedding_client, documents)
    topics = build_topics(entries, group_jids, embeddings)
    await bulk_upsert(session, topics)
    await session.commit()
    return len(topics)


async def run_seed(faq_path: Path) -> int:
    settings = SeedSettings()
    engine = create_async_engine(settings.db_uri, pool_pre_ping=True)
    session_factory = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )
    embedding_client = AsyncClient(
        api_key=settings.voyage_api_key,
        max_retries=settings.voyage_max_retries,
    )
    try:
        entries = parse_faq(faq_path.read_text(encoding="utf-8"))
        async with session_factory() as session:
            return await seed_topics(session, embedding_client, entries)
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--faq",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "faq.md",
        help="FAQ markdown file (default: repo-root faq.md)",
    )
    args = parser.parse_args()
    count = asyncio.run(run_seed(args.faq))
    print(f"Seeded {count} FAQ topics across {len(DEFAULT_GROUP_JIDS)} groups.")


if __name__ == "__main__":
    main()
