import logging
from datetime import datetime, timedelta, timezone
from typing import List

from pydantic_ai.agent import AgentRunResult
from sqlmodel import select, desc
from sqlmodel.ext.asyncio.session import AsyncSession
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
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
)
from .base_handler import BaseHandler
from .escalation import offer_escalation
from config import Settings
from services.prompt_manager import prompt_manager
from utils.llm_provider import is_rate_limit_error, run_with_provider_fallback
from utils.reply_text import clean_visible_reply


# Creating an object
logger = logging.getLogger(__name__)

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

        if auto_reply:
            skip = await self._auto_reply_pre_check(message)
            if skip:
                self._log_auto_reply_skip(message, skip)
                return False

        my_jid = await self.whatsapp.get_my_jid()
        bot_jid = my_jid.normalize_str()

        # Exclude test-like, repeated, and bot-authored context on every reply path.
        stmt = (
            select(Message)
            .where(Message.chat_jid == message.chat_jid)
            .order_by(desc(Message.timestamp))
            .limit(7)
        )
        res = await self.session.exec(stmt)
        history: list[Message] = list(res.all())
        history = self._filter_auto_reply_messages(
            history,
            current_question=message.text,
            current_sender=message.sender_jid,
            group_jid=message.group_jid,
            bot_jid=bot_jid,
        )

        # Get opt-out map
        all_jids = {m.sender_jid for m in history}
        all_jids.add(message.sender_jid)
        opt_out_map = await get_opt_out_map(self.session, list(all_jids))

        rephrased_result = await self.rephrasing_agent(
            my_jid.user, message, history, opt_out_map
        )
        # Get query embedding
        embedded_question = (
            await voyage_embed_text(self.embedding_client, [rephrased_result.output])
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

        # Use hybrid search to get topics with their source messages
        from search.hybrid_search import format_search_results_for_prompt, hybrid_search

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
            recent_group_messages = await self._recent_group_messages(
                message.group_jid,
                current_question=message.text,
                current_sender=message.sender_jid,
                bot_jid=bot_jid,
                limit=50,
            )
        else:
            recent_group_messages = []

        # Format results for the generation agent
        formatted_topics = format_search_results_for_prompt(search_results, opt_out_map)
        if weak_match_context and recent_group_messages:
            formatted_topics += (
                "\n\n## Last 50 group messages:\n"
                + chat2text(recent_group_messages, opt_out_map)
            )

        # Also prepare distances for logging
        similar_topics_distances = [
            f"topic_distance: {r.vector_distance}" for r in search_results
        ]

        sender_number = parse_jid(message.sender_jid).user
        generation_result = await self.generation_agent(
            message.text,
            formatted_topics,
            message.sender_jid,
            history,
            opt_out_map,
            auto_reply=auto_reply,
            weak_match=weak_match_context,
        )
        logger.info(
            "RAG query completed sender=%s chat=%s retrieved_topics=%s "
            "total_messages=%s similarity_scores=%s",
            sender_number,
            message.chat_jid,
            len(search_results),
            sum(len(r.messages) for r in search_results),
            similar_topics_distances,
        )

        response_text = self._clean_auto_reply_text(generation_result.output)
        if is_no_answer(response_text):
            if auto_reply:
                self._log_auto_reply_skip(message, "model returned NO_ANSWER")
            await offer_escalation(self, message)
            return False
        await self.send_message(
            message.chat_jid,
            response_text,
            # in_reply_to=message.message_id,
        )

        return True

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
        return not (is_recent and is_other_group_member and is_short_question)

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
        return [
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
    def _clean_auto_reply_text(text: str) -> str:
        """Remove citations, source blocks, identifiers, and excess sentences."""
        return clean_visible_reply(text, max_sentences=3)

    @retry(
        retry=retry_if_exception(lambda error: not is_rate_limit_error(error)),
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(6),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
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
    ) -> AgentRunResult[str]:
        sender_user = parse_jid(sender).user
        sender_display = opt_out_map.get(sender_user, f"@{sender_user}")

        prompt_template = f"""
        {sender_display}: {query}
        
        # Recent chat history:
        {chat2text(history, opt_out_map)}
        
        # Related Topics:
        {topics}
        """

        return await run_with_provider_fallback(
            self.settings,
            system_prompt=prompt_manager.render(
                "rag.j2", auto_reply=auto_reply, weak_match=weak_match
            ),
            prompt=prompt_template,
        )

    @retry(
        retry=retry_if_exception(lambda error: not is_rate_limit_error(error)),
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(6),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
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
                f"{chat2text(history, opt_out_map)}"
            ),
        )
