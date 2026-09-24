from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Request

from services.catchup import CatchupService, valid_admin_secret


router = APIRouter(tags=["admin"])


@router.post("/admin/catchup")
async def catchup(
    request: Request,
    secret: Annotated[str | None, Header(alias="X-Catchup-Secret")] = None,
) -> dict[str, Any]:
    configured = request.app.state.settings.catchup_admin_secret
    if not valid_admin_secret(secret, configured):
        raise HTTPException(status_code=403, detail="Invalid catch-up secret")
    lock = request.app.state.catchup_lock
    if lock.locked():
        raise HTTPException(status_code=409, detail="Catch-up is already running")
    async with lock:
        return (await CatchupService(request.app).run()).as_dict()
