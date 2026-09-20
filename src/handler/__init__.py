import asyncio
import logging
import re
from urllib.parse import urlparse

from cachetools import TTLCache
from gowa_sdk.webhooks import WebhookEnvelope
from sqlmodel.ext.asyncio.session import AsyncSession
from voyageai.client_async import AsyncClient

from config import Settings
from handler.auto_reply import (
    auto_reply_limiter,
    auto_reply_question_rule,
    auto_reply_text_skip_reason,
)
from handler.escalation import (
    handle_pending_confirmation,
    is_escalation_status_question,
    is_human_request,
    offer_escalation,
    pending_status_message,
)
from handler.kb_qa import KBQAHandler
from handler.router import Router
from handler.whatsapp_group_link_spam import WhatsappGroupLinkSpamHandler
from models import BaseGroup, Group, Message, OptOut
from services.document_ingestion import index_document
from whatsapp import WhatsAppClient

from .base_handler import BaseHandler

logger = logging.getLogger(__name__)

# In-memory processing guard: 4 minutes TTL to prevent duplicate handling
_processing_cache = TTLCache(maxsize=1000, ttl=4 * 60)
_processing_lock = asyncio.Lock()


class MessageHandler(BaseHandler):
    def __init__(
        self,
        session: AsyncSession,
        whatsapp: WhatsAppClient,
        embedding_client: AsyncClient,
        settings: Settings,
    ):
        self.router = Router(session, whatsapp, embedding_client, settings)
        self.whatsapp_group_link_spam = WhatsappGroupLinkSpamHandler(
            session, whatsapp, embedding_client, settings
        )
        self.kb_qa_handler = KBQAHandler(session, whatsapp, embedding_client, settings)
        self.settings = settings
        super().__init__(session, whatsapp, embedding_client)

    async def __call__(self, payload: WebhookEnvelope):
        message = await self.store_message(payload)

        # Persist immediately: a failure later in the handler must not roll
        # back the stored message (it would be lost for the daily summary).
        await self.session.commit()

        # Ignore messages that don't exist.
        if not message:
            return

        auto_reply_groups = self._configured_auto_reply_groups()
        group_jid = message.group_jid
        auto_reply_group = bool(group_jid and group_jid in auto_reply_groups)

        if auto_reply_group:
            await self._ensure_active_group(message)

        if self._payload_from_me(payload) or bool(getattr(message, "from_me", False)):
            if auto_reply_group:
                self._log_auto_reply_skip(group_jid, "from_me")
            return

        # Ignore messages sent by the bot itself
        my_jid = await self.whatsapp.get_my_jid()
        if message.sender_jid == my_jid.normalize_str():
            return

        # Index documents before routing any follow-up question. This also handles
        # file-only messages, which have no text until extraction succeeds.
        if message.group_jid:
            await index_document(
                self.session,
                self.embedding_client,
                self.whatsapp,
                message,
                payload=payload,
            )

        if not message.text:
            if auto_reply_group:
                self._log_auto_reply_skip(group_jid, "no text")
            return

        if message.sender_jid.endswith("@lid"):
            logger.info(
                f"Received message from {message.sender_jid}: {payload.model_dump_json()}"
            )

        if not message.group and is_escalation_status_question(message.text):
            await self.send_message(message.chat_jid, pending_status_message(message))
            return

        if await handle_pending_confirmation(self, message):
            return

        # direct message
        if message and not message.group:
            command = message.text.strip().lower()
            if command == "opt-out":
                await self.handle_opt_out(message)
                return
            elif command == "opt-in":
                await self.handle_opt_in(message)
                return
            elif command == "status":
                await self.handle_opt_status(message)
                return
            if is_human_request(message.text):
                await offer_escalation(self, message)
                return
            await self.router(message)
            return

        if not self.settings.ai_enabled:
            return

        # In-memory dedupe: if this message is already being processed/recently processed, skip
        if message and message.message_id:
            async with _processing_lock:
                if message.message_id in _processing_cache:
                    logger.info(
                        f"Message {message.message_id} already in processing cache; skipping."
                    )
                    return
                _processing_cache[message.message_id] = True

        # Check for /kb_qa command (super admin only)
        # This does not have to be a managed group
        if message.group and message.text.startswith("/kb_qa "):
            if message.chat_jid not in self.settings.qa_test_groups:
                logger.warning(
                    f"QA command attempted from non-whitelisted group: {message.chat_jid}"
                )
                return  # Silent failure
            # Check if sender is a QA tester
            if message.sender_jid not in self.settings.qa_testers:
                logger.warning(f"Unauthorized /kb_qa attempt from {message.sender_jid}")
                return  # Silent failure

            await self.kb_qa_handler(message)
            return

        active_group = bool(group_jid and group_jid in self._configured_active_groups())
        if active_group:
            await self._ensure_active_group(message)

        # Explicitly enabled groups may be mention-triggered without a managed DB flag.
        if (
            message
            and message.group
            and not message.group.managed
            and not auto_reply_group
            and not active_group
        ):
            return

        if is_escalation_status_question(message.text):
            await self.send_message(message.chat_jid, pending_status_message(message))
            return

        mentioned = message.has_mentioned(my_jid)
        if is_human_request(message.text) and (
            mentioned or auto_reply_group or active_group
        ):
            await offer_escalation(self, message)
            return
        if mentioned:
            await self.router(message)
            return

        if (
            message.group
            and message.group.notify_on_spam
            and self._contains_whatsapp_group_link(message.text)
        ):
            await self.whatsapp_group_link_spam(message)
            return

        # Only configured groups get the unmentioned automatic-reply path.
        if auto_reply_group:
            # ``auto_reply_group`` can only be true when ``group_jid`` is set.
            assert group_jid is not None
            rule = auto_reply_question_rule(message.text)
            if rule is None:
                self._log_auto_reply_skip(
                    group_jid, auto_reply_text_skip_reason(message.text)
                )
                return

            skip_reason = auto_reply_limiter.skip_reason(group_jid, message)
            if skip_reason:
                self._log_auto_reply_skip(group_jid, skip_reason)
                return

            logger.info("auto-reply gate fired group=%s rule=%s", group_jid, rule)
            replied = await self.router.ask_knowledge_base(message, auto_reply=True)
            if replied is True:
                auto_reply_limiter.record(group_jid, message)

    def _configured_auto_reply_groups(self) -> set[str]:
        configured = getattr(self.settings, "auto_reply_groups", [])
        if isinstance(configured, str):
            configured = configured.split(",")
        return {
            normalized
            for value in configured or []
            if (normalized := str(value).strip())
        }

    def _configured_active_groups(self) -> set[str]:
        configured = getattr(self.settings, "active_groups", [])
        if isinstance(configured, str):
            configured = configured.split(",")
        return {
            normalized
            for value in configured or []
            if (normalized := str(value).strip())
        }

    async def _ensure_active_group(self, message: Message) -> None:
        """Make configured groups eligible even if group sync has not seen them."""
        if not message.group_jid:
            return
        if message.group is None:
            group = await self.session.get(Group, message.group_jid)
            if group is None:
                group = await self.upsert(
                    Group(**BaseGroup(group_jid=message.group_jid).model_dump())
                )
            message.group = group

    @staticmethod
    def _payload_from_me(payload: WebhookEnvelope) -> bool:
        value = payload.payload.get("from_me", payload.payload.get("fromMe", False))
        return (
            value is True
            or value == 1
            or (isinstance(value, str) and value.strip().casefold() == "true")
        )

    @staticmethod
    def _log_auto_reply_skip(group_jid: str | None, reason: str) -> None:
        logger.info("auto-reply skipped group=%s reason=%s", group_jid, reason)

    def _contains_whatsapp_group_link(self, text: str) -> bool:
        """
        Return True if the given text contains a WhatsApp group invite link
        hosted on chat.whatsapp.com.
        """
        if not text:
            return False

        # Simple regex to extract candidate HTTP(S) URLs from text.
        url_pattern = re.compile(r"https?://[^\s]+", re.IGNORECASE)
        for match in url_pattern.finditer(text):
            candidate = match.group(0)
            parsed = urlparse(candidate)
            if (
                parsed.scheme in ("http", "https")
                and parsed.hostname == "chat.whatsapp.com"
            ):
                return True
        return False

    async def handle_opt_out(self, message: Message):
        opt_out = await self.session.get(OptOut, message.sender_jid)
        if not opt_out:
            opt_out = OptOut(jid=message.sender_jid)
            await self.upsert(opt_out)
            await self.send_message(
                message.chat_jid,
                "You have been opted out. You will no longer be tagged in summaries and answers.",
            )
        else:
            await self.send_message(
                message.chat_jid,
                "You are already opted out.",
            )

    async def handle_opt_in(self, message: Message):
        opt_out = await self.session.get(OptOut, message.sender_jid)
        if opt_out:
            await self.session.delete(opt_out)
            await self.session.commit()
            await self.send_message(
                message.chat_jid,
                "You have been opted in. You will now be tagged in summaries and answers.",
            )
        else:
            await self.send_message(
                message.chat_jid,
                "You are already opted in.",
            )

    async def handle_opt_status(self, message: Message):
        opt_out = await self.session.get(OptOut, message.sender_jid)
        status = "opted out" if opt_out else "opted in"
        await self.send_message(
            message.chat_jid,
            f"You are currently {status}.",
        )
