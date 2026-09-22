import asyncio
import logging
import re
from typing import Sequence
from datetime import datetime, timedelta
from enum import Enum

from pydantic import BaseModel, Field
from sqlmodel import desc, select
from sqlmodel.ext.asyncio.session import AsyncSession
from voyageai.client_async import AsyncClient

from handler.knowledge_base_answers import KnowledgeBaseAnswers
from models import Message
from whatsapp.jid import parse_jid
from utils.chat_text import chat2text
from utils.opt_out import get_opt_out_map
from whatsapp import WhatsAppClient
from config import Settings
from .base_handler import BaseHandler
from services.prompt_manager import prompt_manager
from utils.llm_provider import ProviderFallbackError, run_with_provider_fallback
from .auto_reply import is_programme_request, silent_message_reason


# Creating an object
logger = logging.getLogger(__name__)

NEXUS_INTRO = (
    "Hey, I'm Nexus! I'm here to help you catch up on this group's conversations "
    "and answer questions using the messages saved in our knowledge base.\n\n"
    "Mention this account and ask a question, or ask me to summarize recent messages. "
    "I only know the messages available to me, so I'll say when information is missing or uncertain.\n\n"
    "I can point out obvious filler such as Lorem ipsum. I can't reliably tell whether "
    "a message was AI-generated just by reading it, and I won't pretend otherwise."
)
BOT_FIXED_REPLY = "I'm Nexus, the UniPods METI AI programme assistant. Ask me about sessions, deadlines, MIT, Wadhwani or links."
_BOT_QUESTION_RE = re.compile(
    r"^\s*(?:@\S+\s+)*(?:who\s+are\s+you|are\s+you\s+(?:a\s+)?bot|"
    r"what\s+can\s+you\s+do)\s*[?!.,]*\s*$",
    re.IGNORECASE,
)


_CONTENT_QUESTION_RE = re.compile(
    r"\b(?:what|when|where|how|why|who|can|does|is|help|please|give|send|"
    r"share|forward|need|recap|summary|summarize|happened|quel|quelle|"
    r"quels|quelles|quand|comment|pourquoi|qui|quoi|pouvez|peut|aidez|"
    r"svp|inscription)\b",
    re.IGNORECASE,
)


def _looks_like_content_question(text: str) -> bool:
    """Keep multilingual content questions out of the generic intent fallback."""
    return (
        "?" in text
        or bool(_CONTENT_QUESTION_RE.search(text))
        or is_programme_request(text)
    )


class IntentEnum(str, Enum):
    summarize = "summarize"
    ask_question = "ask_question"
    about = "about"
    other = "other"


class Intent(BaseModel):
    intent: IntentEnum = Field(
        description="""The intent of the message.
- summarize: Summarize TODAY's chat messages, or catch up on the chat messages FROM TODAY ONLY. This will trigger the summarization of the chat messages. This is only relevant for queries about TODDAY chat. A query across a broader timespan is classified as ask_question
- ask_question: Ask a question or learn from the collective knowledge of the group. This will trigger the knowledge base to answer the question.
- about: Learn about me(bot) and my capabilities. This will trigger the about section.
- other:  something else. This will trigger the default response."""
    )


class Router(BaseHandler):
    def __init__(
        self,
        session: AsyncSession,
        whatsapp: WhatsAppClient,
        embedding_client: AsyncClient,
        settings: Settings,
    ):
        self.settings = settings
        self.ask_knowledge_base = KnowledgeBaseAnswers(
            session, whatsapp, embedding_client, settings
        )
        super().__init__(session, whatsapp, embedding_client)

    async def __call__(self, message: Message):
        if not message.text:
            logger.info("reply skipped chat=%s reason=no text", message.chat_jid)
            return

        if _BOT_QUESTION_RE.fullmatch(message.text):
            await self.send_message(
                message.chat_jid,
                BOT_FIXED_REPLY,
                in_reply_to=message.message_id,
            )
            return

        if not message.chat_jid.endswith("@g.us"):
            import re as _re
            import time as _time

            _text = (message.text or "").strip()
            if _re.fullmatch(
                r"(?i)(?:hi+|hello+|hey+|hola|bonjour|salut|greetings|yo|good\s+(?:morning|afternoon|evening|day))(?:\s+(?:there|all|everyone|nexus|bot))?[\W_]*",
                _text,
            ):
                await self.send_message(
                    message.chat_jid,
                    "Hi \U0001f44b I'm the UniPods METI AI programme assistant, Nexus bot. Ask me anything about the programme \u2014 sessions, deadlines, MIT, Wadhwani, links \u2014 and I'll help.",
                    in_reply_to=message.message_id,
                )
                return
            if _re.fullmatch(
                r"(?i)(?:thanks?|thank\s+you|thx|merci)(?:\s+(?:so\s+much|a\s+lot|nexus|bot))?[\W_]*",
                _text,
            ):
                await self.send_message(
                    message.chat_jid,
                    "You're welcome! Let me know if you need anything else about the programme.",
                    in_reply_to=message.message_id,
                )
                return
            if silent_message_reason(message.text):
                _seen = globals().setdefault("_DM_NUDGE_AT", {})
                _now = _time.monotonic()
                if _now - _seen.get(message.chat_jid, -1e9) > 300:
                    _seen[message.chat_jid] = _now
                    await self.send_message(
                        message.chat_jid,
                        "I'm here to help with the UniPods METI programme. Ask me about sessions, deadlines, MIT, Wadhwani, the hackathon or links, and I'll answer right away.",
                        in_reply_to=message.message_id,
                    )
                return
        reason = silent_message_reason(message.text)
        if reason:
            logger.info(
                "reply skipped chat=%s message=%s reason=%s",
                message.chat_jid,
                message.message_id,
                reason,
            )
            return

        try:
            if _looks_like_content_question(message.text):
                await self.ask_knowledge_base(message)
                return

            route = await self._route(message.text)
            match route:
                case IntentEnum.ask_question:
                    await self.ask_knowledge_base(message)
                case IntentEnum.summarize | IntentEnum.about | IntentEnum.other:
                    logger.info(
                        "reply skipped chat=%s message=%s reason=not grounded programme answer intent=%s",
                        message.chat_jid,
                        message.message_id,
                        route.value,
                    )
        except (ProviderFallbackError, asyncio.TimeoutError, TimeoutError) as error:
            logger.warning(
                "reply skipped chat=%s message=%s reason=provider failure error=%s",
                message.chat_jid,
                message.message_id,
                type(error).__name__,
            )

    async def _route(self, message: str) -> IntentEnum:
        result = await run_with_provider_fallback(
            self.settings,
            system_prompt=prompt_manager.render("intent.j2"),
            prompt=message,
            output_type=Intent,
        )
        return result.output.intent

    async def summarize(self, message: Message):
        logger.info(
            "reply skipped chat=%s message=%s reason=summarization is not a programme question",
            message.chat_jid,
            message.message_id,
        )
        return
        time_24_hours_ago = datetime.now() - timedelta(hours=24)
        stmt = (
            select(Message)
            .where(Message.chat_jid == message.chat_jid)
            .where(Message.timestamp >= time_24_hours_ago)
            .order_by(desc(Message.timestamp))
            .limit(30)
        )
        res = await self.session.exec(stmt)
        messages: Sequence[Message] = res.all()

        # Get opt-out map for all senders in the history + current sender
        all_jids = {m.sender_jid for m in messages}
        all_jids.add(message.sender_jid)
        opt_out_map = await get_opt_out_map(self.session, list(all_jids))
        # The summary provider must not run while this session owns a DB connection.
        await self.session.commit()

        sender_user = parse_jid(message.sender_jid).user
        sender_display = opt_out_map.get(sender_user, f"@{sender_user}")

        result = await run_with_provider_fallback(
            self.settings,
            system_prompt=prompt_manager.render("summarize.j2"),
            prompt=(
                f"{sender_display}: {message.text}\n\n # History:\n "
                f"{chat2text(list(messages), opt_out_map)}"
            ),
            output_type=str,
        )
        await self.send_message(
            message.chat_jid,
            result.output,
            in_reply_to=message.message_id,
        )

    async def about(self, message):
        logger.info(
            "reply skipped chat=%s message=%s reason=about request is not fixed identity question",
            message.chat_jid,
            message.message_id,
        )

    async def default_response(self, message):
        logger.info(
            "reply skipped chat=%s message=%s reason=not a programme question",
            message.chat_jid,
            message.message_id,
        )
