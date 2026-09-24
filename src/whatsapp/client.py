from __future__ import annotations

from typing import Optional

from gowa_sdk import GoWaClient

from .jid import JID, parse_jid


class WhatsAppClient(GoWaClient):
    """Thin wrapper over GoWaClient for app-specific helpers."""

    _jid: Optional[JID] = None

    async def get_my_jid(self) -> JID:
        if self._jid:
            return self._jid

        # GoWA v9's /app/devices endpoint returns internal UUIDs. The detailed
        # /devices endpoint carries the logged-in account JID; keep the legacy
        # fallback for older bridge versions used by local deployments.
        try:
            detailed = await self.list_devices()
        except Exception:
            detailed = None

        if detailed and detailed.results:
            for result in detailed.results:
                device_jid = getattr(result, "jid", None) or getattr(
                    result, "phone", None
                )
                if device_jid and ("@" in device_jid or device_jid.isnumeric()):
                    self._jid = parse_jid(device_jid)
                    return self._jid

        info = await self.get_devices()
        if not info.results:
            raise ValueError("No devices found")
        result = info.results[0]
        device_jid = getattr(result, "jid", None) or result.device
        if not device_jid or ("@" not in device_jid and not device_jid.isnumeric()):
            raise ValueError("No primary device JID available")
        self._jid = parse_jid(device_jid)
        return self._jid

    async def get_group_member_count(self, group_jid: str) -> int | None:
        """Return the live participant count when GoWA provides it.

        GoWA's participant endpoint returns ``GroupResponse`` with the group
        records under ``results.data``.  Missing participant data is treated as
        unavailable rather than as an empty group.
        """

        response = await self.get_group_participants(group_jid)
        results = getattr(response, "results", None)
        data = getattr(results, "data", None) if results is not None else None
        if data is None:
            return None
        groups = data if isinstance(data, list) else [data]
        participant_count = 0
        has_participant_data = False
        for group in groups:
            participants = getattr(group, "participants", None)
            if isinstance(participants, list):
                has_participant_data = True
                participant_count += len(participants)
        if not has_participant_data:
            return None
        return participant_count

    async def is_group_admin(self, group_jid: str, member_jid: str) -> bool:
        """Check live WhatsApp group-admin status without exposing participant data."""

        response = await self.get_group_participants(group_jid)
        results = getattr(response, "results", None)
        data = getattr(results, "data", None) if results is not None else None
        groups = data if isinstance(data, list) else [data]
        normalized_member = member_jid.casefold()
        for group in groups:
            participants = getattr(group, "participants", None)
            if isinstance(group, dict):
                participants = group.get("participants")
            for participant in participants or []:
                if isinstance(participant, dict):
                    candidate = participant.get("jid") or participant.get("JID")
                    candidate = (
                        candidate or participant.get("lid") or participant.get("LID")
                    )
                    is_admin = participant.get("is_admin") or participant.get("IsAdmin")
                    is_super_admin = participant.get(
                        "is_super_admin"
                    ) or participant.get("IsSuperAdmin")
                else:
                    candidate = getattr(participant, "jid", None) or getattr(
                        participant, "lid", None
                    )
                    is_admin = getattr(participant, "is_admin", False)
                    is_super_admin = getattr(participant, "is_super_admin", False)
                if candidate and str(candidate).casefold() == normalized_member:
                    return bool(is_admin or is_super_admin)
        return False
