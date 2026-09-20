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
from utils.llm_provider import run_with_provider_fallback
from .auto_reply import is_clear_banter


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
BOT_FIXED_REPLY = (
    "I'm Nexus, the programme-materials assistant. "
    "I answer questions only from the programme materials."
)
_BOT_QUESTION_RE = re.compile(
    r"\b(?:bot|bots|nexus|who are you|what can you do|tell me about (?:yourself|you))\b",
    re.IGNORECASE,
)


_CONTENT_QUESTION_RE = re.compile(
    r"\b(?:quel|quelle|quels|quelles|quand|comment|pourquoi|qui|quoi|"
    r"pouvez|peut|aidez|svp|inscription)\b",
    re.IGNORECASE,
)


def _looks_like_content_question(text: str) -> bool:
    """Keep multilingual content questions out of the generic intent fallback."""
    return "?" in text or bool(_CONTENT_QUESTION_RE.search(text))


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
            return

        if _BOT_QUESTION_RE.search(message.text):
            await self.send_message(
                message.chat_jid,
                BOT_FIXED_REPLY,
                in_reply_to=message.message_id,
            )
            return

        if is_clear_banter(message.text):
            await self.send_message(
                message.chat_jid,
                "Please ask a question about the programme materials.",
                in_reply_to=message.message_id,
            )
            return

        if is_clear_banter(message.text):
            await self.send_message(
                message.chat_jid,
                "😄 I’m filing that under *excellent banter*. Ask me a real question when you’re ready!",
            )
            return

        if _looks_like_content_question(message.text):
            await self.ask_knowledge_base(message)
            return

        route = await self._route(message.text)
        match route:
            case IntentEnum.summarize:
                await self.summarize(message)
            case IntentEnum.ask_question:
                await self.ask_knowledge_base(message)
            case IntentEnum.about:
                await self.about(message)
            case IntentEnum.other:
                await self.default_response(message)

    async def _route(self, message: str) -> IntentEnum:
        result = await run_with_provider_fallback(
            self.settings,
            system_prompt=prompt_manager.render("intent.j2"),
            prompt=message,
            output_type=Intent,
        )
        return result.output.intent

    async def summarize(self, message: Message):
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
        await self.send_message(
            message.chat_jid,
            NEXUS_INTRO,
            in_reply_to=message.message_id,
        )

    async def default_response(self, message):
        await self.send_message(
            message.chat_jid,
            "I'm sorry, but I dont think this is something I can help with right now 😅.\n I can help catch up on the chat messages or answer questions based on the group's knowledge.",
            in_reply_to=message.message_id,
        )
