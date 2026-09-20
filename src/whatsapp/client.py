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
