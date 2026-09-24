"""Persistent, group-scoped control-plane helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum

from models import Group, GroupMemberPermission
from whatsapp.jid import normalize_jid


class GroupCommand(StrEnum):
    pause = "pause"
    resume = "resume"
    select = "select"
    unselect = "unselect"
    allow = "allow"
    deny = "deny"
    allow_all = "allow-all"
    deny_all = "deny-all"


@dataclass(frozen=True)
class ParsedGroupCommand:
    command: GroupCommand
    argument: str | None = None


_COMMAND_RE = re.compile(
    r"^\s*(?:/|nexus\s+)(?P<command>pause|stop|resume|start|select|enable|"
    r"unselect|disable|allow-all|deny-all|allow|deny)(?:\s+(?P<argument>[^\s]+))?\s*$",
    re.IGNORECASE,
)


def parse_group_command(text: str | None) -> ParsedGroupCommand | None:
    if not text:
        return None
    match = _COMMAND_RE.fullmatch(text)
    if not match:
        return None
    command = match.group("command").casefold()
    aliases = {
        "stop": GroupCommand.pause,
        "start": GroupCommand.resume,
        "enable": GroupCommand.select,
        "disable": GroupCommand.unselect,
    }
    resolved = aliases.get(command)
    if resolved is None:
        resolved = GroupCommand(command)
    return ParsedGroupCommand(command=resolved, argument=match.group("argument"))


def group_is_selected(group: Group | None) -> bool:
    """Treat legacy managed rows as selected while migrations roll forward."""

    return bool(group and (group.selected or group.managed))


def group_state(group: Group | None) -> str:
    if group is None or not group_is_selected(group):
        return "UNSELECTED"
    if group.paused:
        return "PAUSED"
    if group.managed:
        return "ACTIVE"
    return "SELECTED"


def set_group_selection(group: Group, selected: bool) -> None:
    group.selected = selected
    if not selected:
        # Keep the legacy engagement flag in sync when explicitly unselected.
        group.managed = False
        group.paused = False
        group.paused_by = None
        group.paused_at = None


def set_group_paused(group: Group, *, paused: bool, actor_jid: str) -> None:
    now = datetime.now(timezone.utc)
    group.paused = paused
    if paused:
        group.paused_by = (
            actor_jid if actor_jid == "dashboard" else normalize_jid(actor_jid)
        )
        group.paused_at = now
    else:
        group.resumed_at = now


def permission_allows(
    group: Group,
    permission: GroupMemberPermission | None,
    member_jid: str,
    *,
    is_admin: bool = False,
) -> bool:
    if is_admin:
        return True
    normalized = normalize_jid(member_jid)
    if permission is not None and normalize_jid(permission.member_jid) == normalized:
        return permission.allowed
    return group.member_policy != "deny"
