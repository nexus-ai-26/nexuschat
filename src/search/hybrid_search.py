"""
Hybrid search module that combines vector similarity search with keyword (full-text) search.

This module provides enhanced search capabilities for the knowledge base by:
1. Performing vector similarity search on topic embeddings
2. Performing PostgreSQL full-text search on actual message content
3. Combining and ranking results from both sources
"""

import logging
import re
from typing import List, Tuple
from dataclasses import dataclass

from sqlmodel import select, text, cast, String, col
from sqlmodel.ext.asyncio.session import AsyncSession

from models import KBTopic, Message
from models.kb_topic_message import KBTopicMessage


logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A search result containing topic and its associated messages."""

    topic: KBTopic
    messages: List[Message]
    vector_distance: float
    keyword_rank: float = 0.0


async def vector_search(
    session: AsyncSession,
    query_embedding: List[float],
    group_jids: List[str] | None = None,
    limit: int = 10,
) -> List[Tuple[KBTopic, float]]:
    """
    Perform vector similarity search on topic embeddings.

    Args:
        session: Database session
        query_embedding: Embedding vector for the search query
        group_jids: Optional list of group JIDs to filter by
        limit: Maximum number of results to return

    Returns:
        List of tuples containing (KBTopic, cosine_distance)
    """
    q = (
        select(
            KBTopic,
            KBTopic.embedding.cosine_distance(query_embedding).label("cosine_distance"),
        )
        .order_by(KBTopic.embedding.cosine_distance(query_embedding))
        .limit(limit)
    )

    if group_jids:
        q = q.where(cast(KBTopic.group_jid, String).in_(group_jids))

    result = await session.exec(q)
    return [(topic, distance) for topic, distance in result]


async def keyword_search(
    session: AsyncSession,
    query: str,
    group_jids: List[str] | None = None,
    limit: int = 20,
) -> List[Tuple[Message, float]]:
    """
    Perform PostgreSQL full-text search on message content.

    Args:
        session: Database session
        query: The search query string
        group_jids: Optional list of group JIDs to filter by
        limit: Maximum number of results to return

    Returns:
        List of tuples containing (Message, ts_rank score)
    """
    # Build the full-text search query dynamically
    # We use plainto_tsquery for simple keyword matching
    # Note: We build the query conditionally to avoid asyncpg AmbiguousParameterError
    # when group_jids is None (asyncpg can't infer the type of a NULL array parameter)

    base_query = """
        SELECT m.*, ts_rank(to_tsvector('simple', COALESCE(m.text, '')), plainto_tsquery('simple', :query)) as rank
        FROM message m
        WHERE to_tsvector('simple', COALESCE(m.text, '')) @@ plainto_tsquery('simple', :query)
    """

    params: dict = {"query": query, "limit": limit}

    if group_jids:
        base_query += " AND m.group_jid = ANY(:group_jids)"
        params["group_jids"] = group_jids

    base_query += " ORDER BY rank DESC LIMIT :limit"

    search_query = text(base_query)

    result = await session.execute(search_query, params)

    rows = result.fetchall()
    messages = []
    for row in rows:
        # Construct Message from row data
        msg = Message(
            message_id=row.message_id,
            timestamp=row.timestamp,
            text=row.text,
            media_url=row.media_url,
            chat_jid=row.chat_jid,
            sender_jid=row.sender_jid,
            group_jid=row.group_jid,
            reply_to_id=row.reply_to_id,
        )
        messages.append((msg, row.rank))

    return messages


async def get_messages_for_topic(
    session: AsyncSession,
    topic_id: str,
    limit: int = 10,
) -> List[Message]:
    """
    Get the source messages that were used to create a topic.

    Args:
        session: Database session
        topic_id: The KB topic ID
        limit: Maximum number of messages to return

    Returns:
        List of Message objects linked to the topic
    """
    q = (
        select(Message)
        .join(KBTopicMessage, col(Message.message_id) == col(KBTopicMessage.message_id))
        .where(col(KBTopicMessage.kb_topic_id) == topic_id)
        .limit(limit)
    )

    result = await session.exec(q)
    return list(result.all())


async def hybrid_search(
    session: AsyncSession,
    query: str,
    query_embedding: List[float],
    group_jids: List[str] | None = None,
    vector_limit: int = 10,
    messages_per_topic: int = 5,
    max_vector_distance: float | None = None,
) -> List[SearchResult]:
    """
    Perform hybrid search combining vector similarity and keyword search.

    This function:
    1. Performs vector search to find semantically similar topics
    2. For each topic, retrieves its associated messages
    3. Optionally enriches with keyword search results

    Args:
        session: Database session
        query: The text search query
        query_embedding: Embedding vector for the search query
        group_jids: Optional list of group JIDs to filter by
        vector_limit: Maximum number of topics from vector search
        messages_per_topic: Maximum messages to retrieve per topic
        max_vector_distance: Optional cosine-distance cutoff for vector results

    Returns:
        List of SearchResult objects containing topics and their messages
    """
    # Step 1: Vector search for similar topics
    vector_results = await vector_search(
        session, query_embedding, group_jids, vector_limit
    )
    if max_vector_distance is not None:
        vector_results = [
            (topic, distance)
            for topic, distance in vector_results
            if distance <= max_vector_distance
        ]

    # Step 2: Keyword search for relevant messages
    keyword_messages = await keyword_search(session, query, group_jids, limit=20)

    # Step 3: Merge results
    # We want to prioritize vector results, but also include topics found via keywords
    # if they aren't already in the vector results.

    # Map topic_id -> SearchResult
    results_map = {}

    # Add vector results first
    for topic, distance in vector_results:
        results_map[topic.id] = SearchResult(
            topic=topic,
            messages=[],  # Will populate later
            vector_distance=distance,
        )

    # Find topics for keyword messages
    if keyword_messages:
        # Get all message IDs found by keyword search
        msg_ids = [msg.message_id for msg, _ in keyword_messages]

        # Find which topics these messages belong to
        q = (
            select(KBTopic, KBTopicMessage.message_id)
            .join(KBTopicMessage, col(KBTopic.id) == col(KBTopicMessage.kb_topic_id))
            .where(col(KBTopicMessage.message_id).in_(msg_ids))
        )

        topic_rows = await session.exec(q)

        # Map message_id -> List[KBTopic] (a message can belong to multiple topics)
        msg_to_topics = {}
        for topic, msg_id in topic_rows:
            if msg_id not in msg_to_topics:
                msg_to_topics[msg_id] = []
            msg_to_topics[msg_id].append(topic)

            # Add topic to results if not present
            if topic.id not in results_map:
                results_map[topic.id] = SearchResult(
                    topic=topic,
                    messages=[],
                    vector_distance=1.0,  # High distance (low similarity) for keyword-only matches
                    keyword_rank=1.0,  # High rank for keyword matches
                )

    # Step 4: Populate messages for all unique topics
    final_results = []
    for topic_id, result in results_map.items():
        # Get messages for this topic
        messages = await get_messages_for_topic(session, topic_id, messages_per_topic)
        result.messages = messages
        final_results.append(result)

    # Sort by vector distance (primary) and keyword rank (secondary)
    # Note: Lower vector distance is better (0 is identical)
    final_results.sort(key=lambda x: x.vector_distance)

    logger.info(
        f"Hybrid search found {len(final_results)} topics (Vector: {len(vector_results)}, "
        f"Total merged: {len(final_results)}), "
        f"total {sum(len(r.messages) for r in final_results)} messages"
    )

    return final_results


def format_search_results_for_prompt(
    results: List[SearchResult],
    opt_out_map: dict[str, str] | None = None,
) -> str:
    """
    Format search results for inclusion in an LLM prompt.

    Args:
        results: List of SearchResult objects
        opt_out_map: Optional mapping of JIDs to display names for privacy

    Returns:
        Formatted string suitable for LLM prompt inclusion
    """
    if not results:
        return "No related topics found."

    formatted_parts = []
    source_number = 0

    for result in results:
        topic = result.topic

        # Format topic header
        topic_text = f"## {topic.subject}\n{topic.summary}"

        # Format associated messages with stable citation labels.
        if result.messages:
            message_texts = []
            for msg in result.messages:
                message_content = msg.text or ""
                if msg.media_url:
                    message_content = (
                        f"{message_content}\nAttachment reference: {msg.media_url}"
                        if message_content
                        else f"Attachment reference: {msg.media_url}"
                    )
                if message_content:
                    source_number += 1
                    sender = (
                        msg.sender_jid.split("@")[0] if msg.sender_jid else "Unknown"
                    )
                    if opt_out_map:
                        sender = opt_out_map.get(sender, f"@{sender}")
                    timestamp = msg.timestamp.strftime("%Y-%m-%d %H:%M UTC")
                    message_texts.append(
                        f"- [{source_number}] {sender} ({timestamp}): {message_content[:400]}"
                    )

            if message_texts:
                topic_text += "\n\n### Related Messages:\n" + "\n".join(message_texts)

        formatted_parts.append(topic_text)

    return "\n\n---\n\n".join(formatted_parts)


def format_citations_for_response(
    results: List[SearchResult],
    response: str,
    opt_out_map: dict[str, str] | None = None,
) -> str:
    """Append grounded source references to a generated Q&A response.

    The model is asked to cite ``[n]`` labels from the prompt. If it omits
    them, include the top three retrieved sources rather than returning an
    apparently authoritative answer with no evidence trail.
    """
    sources: list[tuple[int, str]] = []
    source_number = 0

    for result in results:
        topic = result.topic
        if result.messages:
            for msg in result.messages:
                if not msg.text:
                    continue
                source_number += 1
                sender = msg.sender_jid.split("@")[0] if msg.sender_jid else "Unknown"
                if opt_out_map:
                    sender = opt_out_map.get(sender, f"@{sender}")
                timestamp = msg.timestamp.strftime("%Y-%m-%d")
                sources.append(
                    (
                        source_number,
                        f"_{topic.subject} — {timestamp} — {sender}:_ {msg.text[:240]}",
                    )
                )
        else:
            source_number += 1
            sources.append((source_number, f"_{topic.subject}:_ {topic.summary[:240]}"))

    if not sources:
        return ""

    cited_numbers = {
        int(number)
        for number in re.findall(r"\[(\d+)\]", response)
        if int(number) <= len(sources)
    }
    selected = [source for source in sources if source[0] in cited_numbers]
    if not selected:
        selected = sources[:3]

    lines = ["*Sources:*"]
    lines.extend(f"[{number}] {description}" for number, description in selected)
    return "\n\n" + "\n".join(lines)
