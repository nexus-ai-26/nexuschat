import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import List
from urllib.parse import unquote, urlparse

import httpx

from pydantic_ai.agent import AgentRunResult
from sqlmodel import col, desc, select
from sqlmodel.ext.asyncio.session import AsyncSession
from voyageai.client_async import AsyncClient

from models import Message
from whatsapp import WhatsAppClient
from whatsapp.jid import parse_jid
from utils.chat_text import chat2text
from utils.opt_out import get_opt_out_map
from utils.voyage_embed_text import voyage_embed_text
from .auto_reply import (
    auto_reply_limiter,
    auto_reply_question_rule,
    has_confident_match,
    is_no_answer,
    is_too_short_for_auto_reply,
    normalize_question,
    silent_message_reason,
)
from .base_handler import BaseHandler
from config import Settings
from config.trusted_facts import (
    format_trusted_application_facts,
    trusted_admin_display_names,
    trusted_admin_jids,
    trusted_contacts_for_query,
)
from services.prompt_manager import prompt_manager
from services.document_ingestion import (
    document_topics_for_message,
    index_document,
    is_supported_document,
)
from utils.llm_provider import ProviderFallbackError, run_with_provider_fallback
from utils.reply_text import clean_visible_reply


# Creating an object
logger = logging.getLogger(__name__)

_FILE_REQUEST_RE = re.compile(
    r"(?:\b(?:send|share|forward|resend|download)\b.*\b(?:file|document|pdf|guideline\w*|"
    r"recording\w*|transcript\w*|slide\w*|presentation\w*)\b)"
    r"|(?:\b(?:file|document|pdf|guideline\w*|recording\w*|transcript\w*|"
    r"slide\w*|presentation\w*)\b.*\b(?:send|share|forward|resend|download)\b)",
    re.IGNORECASE,
)
_MAX_FORWARD_BYTES = 25 * 1024 * 1024
_RECAP_RE = re.compile(
    r"\b(?:recap|summary|summarize|this\s+week|what\s+happened|updates?)\b",
    re.IGNORECASE,
)
_RECENT_UPDATE_RE = re.compile(
    r"\b(?:correction|corrected|update|updated|announcement|extended|changed|"
    r"what\s+changed)\b",
    re.IGNORECASE,
)
_GROUP_MEMBER_COUNT_RE = re.compile(
    r"(?:\b(?:how\s+many|number\s+of|count\s+of)\b.*\b(?:members?|participants?)\b|"
    r"\b(?:current(?:ly)?\s+)?group\s+size\b|"
    r"\b(?:current(?:ly)?\s+)?(?:members?|participants?)\b.*\b(?:count|number|size|many)\b)",
    re.IGNORECASE,
)
_NON_ANSWER_REPLY_RE = re.compile(
    r"i\s+don['’]?t\s+have.*programme\s+materials|"
    r"flag\s+this.*organizer|trouble\s+answering|please\s+ask.*programme\s+materials",
    re.IGNORECASE | re.DOTALL,
)


class KnowledgeBaseAnswers(BaseHandler):
    def __init__(
        self,
        session: AsyncSession,
        whatsapp: WhatsAppClient,
        embedding_client: AsyncClient,
        settings: Settings,
    ):
        self.settings = settings
        super().__init__(session, whatsapp, embedding_client)

    async def __call__(self, message: Message, *, auto_reply: bool = False) -> bool:
        # Ensure message.text is not None before passing to generation_agent
        if message.text is None:
            logger.warning(f"Received message with no text from {message.sender_jid}")
            if auto_reply:
                self._log_auto_reply_skip(message, "no text")
            return False

        silent_reason = silent_message_reason(message.text)
        if silent_reason:
            logger.info(
                "RAG reply skipped chat=%s message=%s reason=%s",
                message.chat_jid,
                message.message_id,
                silent_reason,
            )
            return False

        if auto_reply:
            skip = await self._auto_reply_pre_check(message)
            if skip:
                self._log_auto_reply_skip(message, skip)
                return False

        my_jid = await self.whatsapp.get_my_jid()
        bot_jid = my_jid.normalize_str()

        # Exclude test-like, repeated, and bot-authored context on every reply path.
        context_limit = self._recent_message_context_limit()
        stmt = (
            select(Message)
            .where(Message.chat_jid == message.chat_jid)
            .order_by(desc(Message.timestamp))
            .limit(context_limit)
        )
        res = await self.session.exec(stmt)
        history: list[Message] = list(reversed(list(res.all())))
        history = self._filter_auto_reply_messages(
            history,
            current_question=message.text,
            current_sender=message.sender_jid,
            group_jid=message.group_jid,
            bot_jid=bot_jid,
        )
        organizer_history = await self._recent_organizer_messages(message)

        # Dynamic membership is answered only from live WhatsApp state.  If the
        # SDK/bridge cannot provide it, stay silent rather than letting the model
        # manufacture a number.
        if self._is_group_member_count_request(message.text):
            member_count = await self._live_group_member_count(message)
            if member_count is None:
                logger.info(
                    "RAG reply skipped chat=%s message=%s reason=live member count unavailable",
                    message.chat_jid,
                    message.message_id,
                )
                return False
            await self.send_message(
                message.chat_jid,
                f"This group currently has {member_count} members.",
                in_reply_to=message.message_id,
            )
            return True

        # Get opt-out map
        all_jids = {m.sender_jid for m in history + organizer_history}
        all_jids.add(message.sender_jid)
        opt_out_map = await get_opt_out_map(self.session, list(all_jids))
        # Release any DB transaction/connection before the first provider call.
        await self.session.commit()

        from search.hybrid_search import format_search_results_for_prompt, hybrid_search

        document_results, document_error = await self._quoted_document_results(message)
        if document_error:
            logger.info(
                "RAG reply skipped chat=%s message=%s reason=unreadable document",
                message.chat_jid,
                message.message_id,
            )
            return False

        if document_results:
            # A quoted document is the authoritative source for this question.
            # Do not mix unrelated group links or conversation gossip into it.
            search_results = document_results
            weak_match_context = False
        else:
            try:
                rephrased_result = await self.rephrasing_agent(
                    my_jid.user, message, history, opt_out_map
                )
            except (ProviderFallbackError, asyncio.TimeoutError, TimeoutError) as error:
                self._log_provider_skip(message, error)
                return False
            embedded_question = (
                await voyage_embed_text(
                    self.embedding_client, [rephrased_result.output]
                )
            )[0]

            # Determine which groups to search
            group_jids = None
            if message.group:
                group_jids = [message.group.group_jid]
                if message.group.community_keys:
                    related_groups = await message.group.get_related_community_groups(
                        self.session
                    )
                    group_jids.extend([g.group_jid for g in related_groups])

            search_results = await hybrid_search(
                session=self.session,
                query=message.text,
                query_embedding=embedded_question,
                group_jids=group_jids,
                vector_limit=10,
                messages_per_topic=5,
            )
            search_results = self._filter_auto_reply_results(
                search_results,
                current_question=message.text,
                current_sender=message.sender_jid,
                group_jid=message.group_jid,
                bot_jid=bot_jid,
            )
            weak_match_context = False
            if not has_confident_match(search_results):
                weak_match_context = True
                search_results = await hybrid_search(
                    session=self.session,
                    query=message.text,
                    query_embedding=embedded_question,
                    group_jids=group_jids,
                    vector_limit=10,
                    messages_per_topic=5,
                    max_vector_distance=0.6,
                )
                search_results = self._filter_auto_reply_results(
                    search_results,
                    current_question=message.text,
                    current_sender=message.sender_jid,
                    group_jid=message.group_jid,
                    bot_jid=bot_jid,
                )
        # Search is complete; do not retain a DB connection while generating.
        await self.session.commit()

        if await self._try_forward_file(message, search_results):
            return True

        # Format results for the generation agent
        formatted_topics = format_search_results_for_prompt(search_results, opt_out_map)
        trusted_facts = format_trusted_application_facts(message.text)
        # Also prepare distances for logging
        similar_topics_distances = [
            f"topic_distance: {r.vector_distance}" for r in search_results
        ]

        sender_number = parse_jid(message.sender_jid).user
        try:
            generation_result = await self.generation_agent(
                message.text,
                formatted_topics,
                message.sender_jid,
                history,
                opt_out_map,
                auto_reply=auto_reply,
                weak_match=weak_match_context,
                organizer_history=organizer_history,
                trusted_application_facts=trusted_facts,
            )
        except (ProviderFallbackError, asyncio.TimeoutError, TimeoutError) as error:
            self._log_provider_skip(message, error)
            return False
        logger.info(
            "RAG query completed sender=%s chat=%s retrieved_topics=%s "
            "total_messages=%s similarity_scores=%s",
            sender_number,
            message.chat_jid,
            len(search_results),
            sum(len(r.messages) for r in search_results),
            similar_topics_distances,
        )

        allowed_contacts = trusted_contacts_for_query(message.text)
        response_text = self._clean_auto_reply_text(
            generation_result.output,
            allowed_phone_numbers=allowed_contacts,
        )
        if not response_text or is_no_answer(response_text):
            self._log_auto_reply_skip(message, "model returned NO_ANSWER")
            if not message.chat_jid.endswith("@g.us"):
                await self.send_message(
                    message.chat_jid,
                    "I don't have that detail yet. Please ask the organizers, and I'll gladly help with anything about sessions, deadlines, MIT, Wadhwani or links.",
                    in_reply_to=message.message_id,
                )
                return True
            return False
        if _NON_ANSWER_REPLY_RE.search(response_text):
            self._log_auto_reply_skip(message, "model returned non-answer text")
            return False
        await self.send_message(
            message.chat_jid,
            response_text,
            in_reply_to=message.message_id,
            allowed_phone_numbers=allowed_contacts,
        )

        return True

    @staticmethod
    def _log_provider_skip(message: Message, error: Exception) -> None:
        logger.warning(
            "RAG reply skipped chat=%s message=%s reason=provider failure error=%s",
            message.chat_jid,
            message.message_id,
            type(error).__name__,
        )

    @staticmethod
    def _log_auto_reply_skip(message: Message, reason: str) -> None:
        logger.info("auto-reply skipped group=%s reason=%s", message.group_jid, reason)

    async def _auto_reply_pre_check(self, message: Message) -> str | None:
        if is_too_short_for_auto_reply(message.text):
            return "too short"
        if not message.group_jid:
            return "missing group_jid"
        return auto_reply_limiter.skip_reason(message.group_jid, message)

    async def _recent_group_messages(
        self,
        group_jid: str | None,
        *,
        current_question: str,
        current_sender: str,
        bot_jid: str,
        limit: int,
    ) -> list[Message]:
        if not group_jid:
            return []
        stmt = (
            select(Message)
            .where(Message.group_jid == group_jid)
            .order_by(desc(Message.timestamp))
            .limit(limit)
        )
        result = await self.session.exec(stmt)
        messages = list(reversed(list(result.all())))
        return self._filter_auto_reply_messages(
            messages,
            current_question=current_question,
            current_sender=current_sender,
            group_jid=group_jid,
            bot_jid=bot_jid,
        )

    async def _recent_organizer_messages(self, message: Message) -> list[Message]:
        """Load recent organizer statements for recap/update questions.

        These messages remain attributed chat statements.  They are useful for
        reporting a correction or announcement, but are not promoted to permanent
        programme facts by this lookup alone.
        """
        if not message.group_jid or not (
            _RECAP_RE.search(message.text or "")
            or _RECENT_UPDATE_RE.search(message.text or "")
        ):
            return []

        organizer_jids: set[str] = trusted_admin_jids()
        if message.group and message.group.owner_jid:
            organizer_jids.add(message.group.owner_jid)
        for value in (
            *(getattr(self.settings, "escalation_primary_jids", []) or []),
            *(getattr(self.settings, "escalation_secondary_jids", []) or []),
        ):
            if str(value).strip():
                organizer_jids.add(str(value).strip())
        if not organizer_jids:
            logger.info(
                "recap organizer context empty group=%s reason=no organizer identities",
                message.group_jid,
            )
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
        stmt = (
            select(Message)
            .where(Message.group_jid == message.group_jid)
            .where(Message.timestamp >= cutoff)
            .order_by(col(Message.timestamp))
            .limit(100)
        )
        result = await self.session.exec(stmt)
        messages = [
            candidate
            for candidate in result.all()
            if candidate.sender_jid in organizer_jids and candidate.text
        ]
        messages = self._deduplicate_messages(messages)
        logger.info(
            "recap organizer context group=%s messages=%s window_hours=48",
            message.group_jid,
            len(messages),
        )
        return messages

    async def _quoted_document_results(self, message: Message):
        """Return only the quoted document's chunks, or mark an unreadable file."""
        if not message.reply_to_id:
            return [], False
        source = await self.session.get(Message, message.reply_to_id)
        await self.session.commit()
        if source is None or not is_supported_document(source):
            return [], False

        topics = await document_topics_for_message(self.session, source.message_id)
        await self.session.commit()
        extracted_marker = "\n\n" not in (source.text or "")
        if extracted_marker or not topics:
            await index_document(
                self.session,
                self.embedding_client,
                self.whatsapp,
                source,
            )
            topics = await document_topics_for_message(self.session, source.message_id)
        await self.session.commit()
        if not topics:
            return [], True

        from search.hybrid_search import SearchResult

        return [
            SearchResult(topic=topic, messages=[], vector_distance=0.0)
            for topic in topics
        ], False

    def _configured_kb_exclude_subject_prefixes(self) -> tuple[str, ...]:
        configured = getattr(
            self.settings, "kb_exclude_subject_prefixes", ["Hackathon inquiry"]
        )
        if isinstance(configured, str):
            configured = configured.split(",")
        return tuple(
            prefix.casefold().strip()
            for prefix in configured or []
            if str(prefix).strip()
        )

    @staticmethod
    def _message_timestamp(message: Message) -> datetime:
        timestamp = message.timestamp
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)

    @classmethod
    def _is_auto_reply_context_message_allowed(
        cls,
        message: Message,
        *,
        current_question: str,
        current_sender: str,
        group_jid: str | None,
        bot_jid: str,
        now: datetime,
    ) -> bool:
        if message.sender_jid == bot_jid:
            return False
        if normalize_question(message.text) == normalize_question(current_question):
            return False

        is_recent = now - cls._message_timestamp(message) <= timedelta(hours=24)
        is_other_group_member = (
            bool(group_jid)
            and message.group_jid == group_jid
            and message.sender_jid != current_sender
        )
        is_short_question = (
            bool(message.text)
            and len(message.text.split()) <= 12
            and auto_reply_question_rule(message.text) is not None
        )
        return not (
            is_recent
            and is_other_group_member
            and is_short_question
            and not _RECENT_UPDATE_RE.search(message.text or "")
        )

    def _filter_auto_reply_messages(
        self,
        messages: list[Message],
        *,
        current_question: str,
        current_sender: str,
        group_jid: str | None,
        bot_jid: str,
    ) -> list[Message]:
        now = datetime.now(timezone.utc)
        filtered = [
            message
            for message in messages
            if self._is_auto_reply_context_message_allowed(
                message,
                current_question=current_question,
                current_sender=current_sender,
                group_jid=group_jid,
                bot_jid=bot_jid,
                now=now,
            )
        ]
        return self._deduplicate_messages(filtered)

    @staticmethod
    def _deduplicate_messages(messages: list[Message]) -> list[Message]:
        """Keep the newest copy of repeated text while preserving chronology."""

        latest_by_text: dict[str, Message] = {}
        without_text: list[Message] = []
        for message in messages:
            key = normalize_question(message.text)
            if key:
                latest_by_text[key] = message
            else:
                without_text.append(message)
        return sorted(
            [*latest_by_text.values(), *without_text],
            key=KnowledgeBaseAnswers._message_timestamp,
        )

    def _recent_message_context_limit(self) -> int:
        value = getattr(self.settings, "recent_message_context_limit", 30)
        if isinstance(value, bool) or not isinstance(value, int):
            return 30
        return min(max(value, 1), 100)

    @staticmethod
    def _is_group_member_count_request(text: str | None) -> bool:
        return bool(text and _GROUP_MEMBER_COUNT_RE.search(text))

    async def _live_group_member_count(self, message: Message) -> int | None:
        if not message.group_jid:
            return None
        try:
            member_count = await self.whatsapp.get_group_member_count(message.group_jid)
        except Exception:
            logger.warning(
                "Live group member lookup failed group=%s", message.group_jid
            )
            return None
        return (
            member_count
            if isinstance(member_count, int) and member_count >= 0
            else None
        )

    def _is_excluded_auto_reply_topic(self, topic) -> bool:
        subject = (topic.subject or "").casefold().strip()
        if not any(
            subject.startswith(prefix)
            for prefix in self._configured_kb_exclude_subject_prefixes()
        ):
            return False
        # FAQ seed rows are explicitly trusted; non-FAQ rows with this test prefix
        # are the generated/test topics that must not be cited by auto-reply.
        return (topic.speakers or "").strip().casefold() != "faq"

    def _filter_auto_reply_results(
        self,
        results,
        *,
        current_question: str,
        current_sender: str,
        group_jid: str | None,
        bot_jid: str,
    ):
        filtered = []
        for result in results:
            if self._is_excluded_auto_reply_topic(result.topic):
                continue
            filtered.append(
                type(result)(
                    topic=result.topic,
                    messages=self._filter_auto_reply_messages(
                        result.messages,
                        current_question=current_question,
                        current_sender=current_sender,
                        group_jid=group_jid,
                        bot_jid=bot_jid,
                    ),
                    vector_distance=result.vector_distance,
                    keyword_rank=result.keyword_rank,
                )
            )
        return filtered

    @staticmethod
    def _is_file_request(text: str) -> bool:
        return bool(_FILE_REQUEST_RE.search(text))

    @staticmethod
    def _filename_for_attachment(message: Message, reference: str) -> str:
        attached_name = re.search(r"\[\[Attached [^\]]+\]\]\s*(.+)", message.text or "")
        candidate = attached_name.group(1).strip() if attached_name else ""
        if not candidate:
            candidate = unquote(urlparse(reference).path).rstrip("/").rsplit("/", 1)[-1]
        candidate = candidate.splitlines()[0].strip()
        candidate = re.sub(r"[^A-Za-z0-9._ -]", "_", candidate)
        return candidate[:120] or "document"

    async def _download_file_reference(self, reference: str) -> bytes | None:
        if reference.startswith(("http://", "https://")):
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                response = await client.get(reference)
        else:
            path = reference if reference.startswith("/") else f"/{reference}"
            response = await self.whatsapp._get(path)
        response.raise_for_status()
        if len(response.content) > _MAX_FORWARD_BYTES:
            return None
        return response.content

    async def _try_forward_file(self, message: Message, search_results) -> bool:
        if not self._is_file_request(message.text or ""):
            return False

        for result in search_results:
            for source_message in result.messages:
                if not source_message.media_url:
                    continue
                try:
                    file_content = await self._download_file_reference(
                        source_message.media_url
                    )
                    if not file_content:
                        logger.warning(
                            "File forwarding skipped chat=%s reason=empty_or_too_large",
                            message.chat_jid,
                        )
                        return False
                    await self.send_file(
                        message.chat_jid,
                        file_content,
                        filename=self._filename_for_attachment(
                            source_message, source_message.media_url
                        ),
                    )
                    return True
                except Exception:
                    logger.warning("File forwarding failed chat=%s", message.chat_jid)
                    return False
        return False

    @staticmethod
    def _clean_auto_reply_text(
        text: str,
        *,
        allowed_phone_numbers: tuple[str, ...] = (),
    ) -> str:
        """Remove citations, source blocks, identifiers, and excess sentences."""
        return clean_visible_reply(
            text,
            max_sentences=3,
            allowed_phone_numbers=allowed_phone_numbers,
        )

    async def generation_agent(
        self,
        query: str,
        topics: str,  # receives pre-formatted topics
        sender: str,
        history: List[Message],
        opt_out_map: dict[str, str],
        auto_reply: bool = False,
        weak_match: bool = False,
        organizer_history: list[Message] | None = None,
        trusted_application_facts: str | None = None,
        live_tool_results: str | None = None,
    ) -> AgentRunResult[str]:
        sender_user = parse_jid(sender).user
        sender_display = opt_out_map.get(sender_user, f"@{sender_user}")
        trusted_names = trusted_admin_display_names()
        recent_context = chat2text(history, opt_out_map, trusted_names)
        history_ids = {item.message_id for item in history}
        history_text = {
            normalize_question(item.text)
            for item in history
            if normalize_question(item.text)
        }
        organizer_only = [
            item
            for item in organizer_history or []
            if item.message_id not in history_ids
            and normalize_question(item.text) not in history_text
        ]
        if organizer_only:
            recent_context += (
                "\n\nOrganizer statements selected for this recent update question:\n"
                + chat2text(organizer_only, opt_out_map, trusted_names)
            )

        prompt_template = f"""
        {sender_display}: {query}

        # TRUSTED_APPLICATION_FACTS
        {trusted_application_facts or "No relevant trusted application facts are available."}

        # LIVE_TOOL_RESULTS
        {live_tool_results or "No live tool results are available."}

        # RECENT_CHAT_CONTEXT
        {recent_context or "No recent chat context is available."}

        # CURATED_KB_CONTEXT
        {topics}
        """

        return await run_with_provider_fallback(
            self.settings,
            system_prompt=prompt_manager.render(
                "rag.j2", auto_reply=auto_reply, weak_match=weak_match
            ),
            prompt=prompt_template,
        )

    async def rephrasing_agent(
        self,
        my_jid: str,
        message: Message,
        history: List[Message],
        opt_out_map: dict[str, str],
    ) -> AgentRunResult[str]:
        # We obviously need to translate the question and turn the question vebality to a title / summary text to make it closer to the questions in the rag
        return await run_with_provider_fallback(
            self.settings,
            system_prompt=prompt_manager.render("rephrase.j2", my_jid=my_jid),
            prompt=(
                f"{message.text}\n\n## Recent chat history:\n "
                f"{chat2text(history, opt_out_map, trusted_admin_display_names())}"
            ),
        )
