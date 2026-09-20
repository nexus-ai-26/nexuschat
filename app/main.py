import asyncio
import os
from contextlib import asynccontextmanager
from warnings import warn
import logging

from fastapi import FastAPI
from gowa_sdk.webhooks import WebhookEnvelope
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession
import logfire

from api import load_new_kbtopics_api, status, summarize_and_send_to_group_api, webhook
import models  # noqa
from config import get_settings
from whatsapp import WhatsAppClient
from whatsapp.init_groups import gather_groups
from voyageai.client_async import AsyncClient
from load_new_kbtopics import topicsLoader
from services.webhook_processing import process_webhook_message
from services.webhook_queue import PerChatQueue


logger = logging.getLogger(__name__)
KB_TOPIC_SYNC_INTERVAL_SECONDS = 15 * 60


async def sync_kb_topics_periodically(
    async_session, embedding_client, whatsapp, background_semaphore
):
    while True:
        try:
            async with background_semaphore:
                async with async_session() as session:
                    await topicsLoader().load_topics_for_all_groups(
                        session, embedding_client, whatsapp
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Periodic KB topic sync failed")
        await asyncio.sleep(KB_TOPIC_SYNC_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Create and configure logger
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=settings.log_level,
    )

    app.state.settings = settings

    app.state.whatsapp = WhatsAppClient(
        settings.whatsapp_host,
        settings.whatsapp_basic_auth_user,
        settings.whatsapp_basic_auth_password,
    )

    if settings.db_uri.startswith("postgresql://"):
        warn("use 'postgresql+asyncpg://' instead of 'postgresql://' in db_uri")
    engine = create_async_engine(
        settings.db_uri,
        pool_size=20,
        max_overflow=40,
        pool_timeout=30,
        pool_pre_ping=True,
        pool_recycle=600,
        future=True,
    )
    logfire.instrument_sqlalchemy(engine)
    async_session = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )

    async def sync_groups_on_startup() -> None:
        async with app.state.background_semaphore:
            async with async_session() as session:
                try:
                    await gather_groups(session, app.state.whatsapp)
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

    startup_groups_task = asyncio.create_task(sync_groups_on_startup())
    app.state.startup_groups_task = startup_groups_task

    app.state.db_engine = engine
    app.state.async_session = async_session
    app.state.agent_semaphore = asyncio.Semaphore(settings.agent_concurrency_limit)
    app.state.background_semaphore = asyncio.Semaphore(
        settings.background_concurrency_limit
    )
    webhook_queue: PerChatQueue[WebhookEnvelope] = PerChatQueue(
        lambda payload: process_webhook_message(app, payload),
        maxsize=settings.webhook_queue_maxsize,
    )
    app.state.webhook_queue = webhook_queue
    app.state.embedding_client = AsyncClient(
        api_key=settings.voyage_api_key, max_retries=settings.voyage_max_retries
    )
    kb_topic_sync_task = asyncio.create_task(
        sync_kb_topics_periodically(
            async_session,
            app.state.embedding_client,
            app.state.whatsapp,
            app.state.background_semaphore,
        )
    )
    logger.info(
        "Periodic KB topic sync started interval_seconds=%s",
        KB_TOPIC_SYNC_INTERVAL_SECONDS,
    )
    try:
        yield
    finally:
        await app.state.webhook_queue.close()
        startup_groups_task.cancel()
        await asyncio.gather(startup_groups_task, return_exceptions=True)
        kb_topic_sync_task.cancel()
        await asyncio.gather(kb_topic_sync_task, return_exceptions=True)
        await app.state.whatsapp.close()
        await engine.dispose()


# Initialize FastAPI app
app = FastAPI(title="Webhook API", lifespan=lifespan)

# Never let an unset token make Logfire attempt an export. The explicit false
# also prevents a stale LOGFIRE_TOKEN environment value from enabling export.
logfire.configure(
    send_to_logfire=(
        os.getenv("LOGFIRE_SEND_TO_LOGFIRE", "false").strip().casefold() == "true"
        and bool(os.getenv("LOGFIRE_TOKEN"))
    )
)
logfire.instrument_pydantic_ai()
logfire.instrument_fastapi(app)
logfire.instrument_httpx(capture_all=True)
logfire.instrument_system_metrics()


app.include_router(webhook.router)
app.include_router(status.router)
app.include_router(summarize_and_send_to_group_api.router)
app.include_router(load_new_kbtopics_api.router)

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    print(f"Running on {settings.host}:{settings.port}")

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
