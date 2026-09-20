import logging
from fastapi import Request
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlmodel.ext.asyncio.session import AsyncSession
from voyageai.client_async import AsyncClient

from load_new_kbtopics import topicsLoader
from whatsapp import WhatsAppClient

from .deps import (
    get_db_async_session,
    get_text_embebedding,
    get_whatsapp,
    require_ai_enabled,
)

router = APIRouter()

# Configure logger for this module
logger = logging.getLogger(__name__)


@router.post("/load_new_kbtopics", dependencies=[Depends(require_ai_enabled)])
async def load_new_kbtopics_api(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db_async_session)],
    whatsapp: Annotated[WhatsAppClient, Depends(get_whatsapp)],
    embedding_client: Annotated[AsyncClient, Depends(get_text_embebedding)],
) -> dict[str, Any]:
    """
    Trigger load new kbtopics for all managed groups.
    Returns a success message upon completion.
    """
    try:
        logger.info("Starting load new kbtopics sync via API")

        topics_loader = topicsLoader()
        semaphore = getattr(request.app.state, "background_semaphore", None)
        if semaphore is None:
            await topics_loader.load_topics_for_all_groups(
                session, embedding_client, whatsapp
            )
        else:
            async with semaphore:
                await topics_loader.load_topics_for_all_groups(
                    session, embedding_client, whatsapp
                )

        logger.info("load new kbtopics sync completed successfully")

        return {
            "status": "success",
            "message": "load new kbtopics sync completed successfully",
        }

    except Exception as error:
        logger.error(
            "Error during load new kbtopics sync error=%s",
            type(error).__name__,
        )
        # Re-raise the exception to let FastAPI handle it with proper error response
        raise
