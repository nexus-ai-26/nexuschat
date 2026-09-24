"""Authenticated dashboard/control-plane API foundations."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from config import Settings, get_settings
from models import Group, GroupMemberPermission, GroupSchedule
from services.group_control import group_state, set_group_paused, set_group_selection
from services.group_scheduler import next_schedule_time, validate_schedule
from whatsapp import WhatsAppClient

from .deps import get_db_async_session, get_whatsapp

router = APIRouter(prefix="/admin", tags=["control-plane"])


def require_dashboard_secret(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    configured = settings.dashboard_admin_secret
    provided = request.headers.get("x-nexus-admin-secret")
    if not configured:
        raise HTTPException(
            status_code=503, detail="Dashboard control is not configured"
        )
    if not provided or not secrets.compare_digest(provided, configured):
        raise HTTPException(status_code=403, detail="Dashboard authorization failed")


class GroupControlRequest(BaseModel):
    action: Literal["select", "unselect", "pause", "resume"]


class PermissionRequest(BaseModel):
    member_jid: str = Field(min_length=3, max_length=255)
    allowed: bool


class ScheduleRequest(BaseModel):
    schedule_kind: str = Field(min_length=1, max_length=32)
    cron_expression: str | None = Field(default=None, max_length=128)
    timezone_name: str = Field(default="UTC", max_length=64)
    enabled: bool = True


async def _group(session: AsyncSession, group_jid: str) -> Group:
    group = await session.get(Group, group_jid)
    if group is None:
        raise HTTPException(status_code=404, detail="Group not found")
    return group


@router.get("/groups", dependencies=[Depends(require_dashboard_secret)])
async def list_groups(
    session: Annotated[AsyncSession, Depends(get_db_async_session)],
) -> list[dict[str, Any]]:
    result = await session.exec(select(Group).order_by(Group.group_name))
    return [
        {
            "group_jid": group.group_jid,
            "group_name": group.group_name,
            "state": group_state(group),
            "selected": group.selected or group.managed,
            "paused": group.paused,
            "member_policy": group.member_policy,
        }
        for group in result.all()
    ]


@router.post(
    "/groups/{group_jid}/control", dependencies=[Depends(require_dashboard_secret)]
)
async def control_group(
    group_jid: str,
    payload: GroupControlRequest,
    session: Annotated[AsyncSession, Depends(get_db_async_session)],
) -> dict[str, str]:
    group = await _group(session, group_jid)
    if payload.action == "select":
        set_group_selection(group, True)
    elif payload.action == "unselect":
        set_group_selection(group, False)
    elif payload.action == "pause":
        set_group_selection(group, True)
        set_group_paused(group, paused=True, actor_jid="dashboard")
    else:
        set_group_selection(group, True)
        set_group_paused(group, paused=False, actor_jid="dashboard")
    session.add(group)
    await session.commit()
    return {"group_jid": group.group_jid, "state": group_state(group)}


@router.post(
    "/groups/{group_jid}/permissions", dependencies=[Depends(require_dashboard_secret)]
)
async def set_permission(
    group_jid: str,
    payload: PermissionRequest,
    session: Annotated[AsyncSession, Depends(get_db_async_session)],
) -> dict[str, Any]:
    group = await _group(session, group_jid)
    permission = await session.get(
        GroupMemberPermission, (group.group_jid, payload.member_jid)
    )
    if permission is None:
        permission = GroupMemberPermission(
            group_jid=group.group_jid,
            member_jid=payload.member_jid,
            updated_by="dashboard",
        )
    permission.allowed = payload.allowed
    permission.updated_by = "dashboard"
    permission.updated_at = datetime.now(timezone.utc)
    session.add(permission)
    await session.commit()
    return {
        "group_jid": group.group_jid,
        "member_jid": permission.member_jid,
        "allowed": permission.allowed,
    }


@router.post(
    "/groups/{group_jid}/schedules", dependencies=[Depends(require_dashboard_secret)]
)
async def create_schedule(
    group_jid: str,
    payload: ScheduleRequest,
    session: Annotated[AsyncSession, Depends(get_db_async_session)],
) -> dict[str, Any]:
    group = await _group(session, group_jid)
    try:
        validate_schedule(
            payload.schedule_kind,
            cron_expression=payload.cron_expression,
            timezone_name=payload.timezone_name,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    schedule = await session.exec(
        select(GroupSchedule).where(
            GroupSchedule.group_jid == group.group_jid,
            GroupSchedule.schedule_kind == payload.schedule_kind,
        )
    )
    existing = schedule.first()
    if existing is None:
        existing = GroupSchedule(
            group_jid=group.group_jid,
            schedule_kind=payload.schedule_kind,
            created_by="dashboard",
        )
    existing.cron_expression = payload.cron_expression
    existing.timezone_name = payload.timezone_name
    existing.enabled = payload.enabled
    existing.next_run_at = next_schedule_time(existing)
    existing.updated_at = datetime.now(timezone.utc)
    session.add(existing)
    await session.commit()
    return {
        "group_jid": existing.group_jid,
        "schedule_kind": existing.schedule_kind,
        "enabled": existing.enabled,
        "next_run_at": existing.next_run_at,
    }


@router.get("/whatsapp/connection", dependencies=[Depends(require_dashboard_secret)])
async def whatsapp_connection(
    whatsapp: Annotated[WhatsAppClient, Depends(get_whatsapp)],
) -> dict[str, Any]:
    devices = await whatsapp.get_devices()
    status = await whatsapp.get_status()
    result = getattr(status, "results", None)
    status_value = result.get("status") if isinstance(result, dict) else None
    devices_results = devices.results or []
    return {
        "connected": bool(devices_results),
        "device_count": len(devices_results),
        "status": status_value or ("connected" if devices_results else "disconnected"),
    }


@router.get(
    "/whatsapp/qr/{device_id}", dependencies=[Depends(require_dashboard_secret)]
)
async def whatsapp_qr(
    device_id: str,
    whatsapp: Annotated[WhatsAppClient, Depends(get_whatsapp)],
) -> dict[str, Any]:
    response = await whatsapp.device_login_qr(device_id)
    result = getattr(response, "results", None)
    if isinstance(result, dict):
        return {
            "qr_link": result.get("qr_link"),
            "qr_duration": result.get("qr_duration"),
        }
    return {
        "qr_link": getattr(result, "qr_link", None),
        "qr_duration": getattr(result, "qr_duration", None),
    }
