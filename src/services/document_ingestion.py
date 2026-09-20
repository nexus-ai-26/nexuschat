"""Download, extract, and index WhatsApp document messages."""

from __future__ import annotations

import hashlib
import io
import logging
import re
import zipfile
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree

import httpx
from sqlalchemy.dialects.postgresql import insert
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from voyageai.client_async import AsyncClient

from models import KBTopic, Message
from models.kb_topic_message import KBTopicMessage
from models.upsert import bulk_upsert
from utils.voyage_embed_text import voyage_embed_text

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_DOCUMENT_BYTES = 25 * 1024 * 1024
MAX_DOCUMENT_TEXT = 250_000
DOCUMENT_MARKER_RE = re.compile(
    r"\[\[Attached Document\]\]\s*(?P<name>[^\n]+)", re.IGNORECASE
)


@dataclass(frozen=True)
class DocumentContext:
    filename: str
    text: str
    chunks: tuple[str, ...]


def document_filename(message: Message, payload: Any | None = None) -> str | None:
    """Get a safe source filename from a webhook payload or stored message."""
    document = getattr(payload, "document", None) if payload is not None else None
    if isinstance(document, dict):
        for key in ("file_name", "filename", "name"):
            value = document.get(key)
            if value:
                return str(value).strip()

    match = DOCUMENT_MARKER_RE.search(message.text or "")
    if match:
        return match.group("name").strip()

    reference = message.media_url or ""
    name = unquote(urlparse(reference).path).rstrip("/").rsplit("/", 1)[-1]
    return name or None


def is_supported_document(message: Message, payload: Any | None = None) -> bool:
    document = getattr(payload, "document", None) if payload is not None else None
    if document is not None:
        return True
    filename = document_filename(message, payload)
    return bool(filename and _extension(filename) in SUPPORTED_EXTENSIONS)


def _extension(filename: str) -> str:
    return "." + filename.rsplit(".", 1)[-1].casefold() if "." in filename else ""


def chunk_document_text(text: str, *, max_chars: int = 3000) -> list[str]:
    """Split extracted text at paragraph/line boundaries for KB embeddings."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = [
            paragraph[i : i + max_chars] for i in range(0, len(paragraph), max_chars)
        ]
        for piece in pieces:
            if current and len(current) + len(piece) + 2 > max_chars:
                chunks.append(current)
                current = ""
            current = f"{current}\n\n{piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def extract_document_text(data: bytes, filename: str) -> str:
    """Extract text from PDF, DOCX, or UTF text without guessing unsupported files."""
    extension = _extension(filename)
    if extension == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    elif extension == ".docx":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml)
        text_parts: list[str] = []
        for element in root.iter():
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "t" and element.text:
                text_parts.append(element.text)
            elif tag in {"p", "br", "tab"}:
                text_parts.append("\n" if tag != "tab" else "\t")
        text = "".join(text_parts)
    elif extension == ".txt":
        for encoding in ("utf-8-sig", "utf-16", "latin-1"):
            try:
                text = data.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:  # pragma: no cover - latin-1 always decodes
            text = ""
    else:
        raise ValueError("unsupported_document_type")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise ValueError("document_has_no_extractable_text")
    return text[:MAX_DOCUMENT_TEXT]


async def download_document(whatsapp, message: Message) -> bytes:
    """Use GoWA's message-media endpoint, with the stored media path as fallback."""
    try:
        return await whatsapp.download_message_media(
            message.message_id, message.chat_jid
        )
    except Exception as direct_error:
        reference = message.media_url
        if not reference:
            raise direct_error
        if reference.startswith(("http://", "https://")):
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                response = await client.get(reference)
        else:
            path = reference if reference.startswith("/") else f"/{reference}"
            response = await whatsapp._get(path)
        response.raise_for_status()
        return response.content


async def load_document_context(
    whatsapp,
    message: Message,
    *,
    payload: Any | None = None,
) -> DocumentContext:
    filename = document_filename(message, payload)
    if not filename or _extension(filename) not in SUPPORTED_EXTENSIONS:
        raise ValueError("unsupported_document_type")
    data = await download_document(whatsapp, message)
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError("document_too_large")
    text = extract_document_text(data, filename)
    return DocumentContext(filename, text, tuple(chunk_document_text(text)))


async def index_document(
    session: AsyncSession,
    embedding_client: AsyncClient,
    whatsapp,
    message: Message,
    *,
    payload: Any | None = None,
) -> bool:
    """Index one document using existing KB topic/vector tables."""
    if not is_supported_document(message, payload):
        return False
    try:
        context = await load_document_context(whatsapp, message, payload=payload)
        embeddings = await voyage_embed_text(embedding_client, list(context.chunks))
    except Exception as error:
        logger.warning(
            "Document extraction failed message=%s error=%s",
            message.message_id,
            type(error).__name__,
        )
        return False

    stored = await session.get(Message, message.message_id)
    if stored is None:
        stored = message
    stored.text = f"[[Attached Document]] {context.filename}\n\n{context.text}"
    session.add(stored)

    topics = []
    for index, (chunk, embedding) in enumerate(
        zip(context.chunks, embeddings, strict=True), 1
    ):
        topic_id = hashlib.sha256(
            f"document:{message.message_id}:{index}:{chunk}".encode()
        ).hexdigest()
        topics.append(
            KBTopic(
                id=topic_id,
                group_jid=message.group_jid,
                subject=f"{context.filename} (part {index})",
                summary=chunk,
                speakers="document",
                embedding=embedding,
            )
        )
    await bulk_upsert(session, topics)
    for topic in topics:
        statement = insert(KBTopicMessage).values(
            kb_topic_id=topic.id, message_id=message.message_id
        )
        await session.execute(statement.on_conflict_do_nothing())
    await session.commit()
    logger.info(
        "Document indexed message=%s filename=%s chunks=%s",
        message.message_id,
        context.filename,
        len(context.chunks),
    )
    return True


async def document_topics_for_message(
    session: AsyncSession, message_id: str
) -> list[KBTopic]:
    result = await session.exec(
        select(KBTopic)
        .join(KBTopicMessage, KBTopic.id == KBTopicMessage.kb_topic_id)
        .where(KBTopicMessage.message_id == message_id)
        .order_by(KBTopic.subject)
    )
    return list(result.all())
