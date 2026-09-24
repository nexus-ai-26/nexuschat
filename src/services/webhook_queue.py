"""Small per-chat work queues used to serialize WhatsApp replies."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar


logger = logging.getLogger(__name__)
T = TypeVar("T")


class PerChatQueue(Generic[T]):
    """Run jobs for one chat in order while allowing chats to run concurrently."""

    def __init__(
        self,
        worker: Callable[[T], Awaitable[None]],
        *,
        maxsize: int = 100,
    ) -> None:
        self._worker = worker
        self._maxsize = maxsize
        self._queues: dict[str, asyncio.Queue[T]] = {}
        self._workers: dict[str, asyncio.Task[None]] = {}

    def enqueue(self, chat_key: str, item: T) -> bool:
        queue = self._queues.setdefault(chat_key, asyncio.Queue(maxsize=self._maxsize))
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            logger.error(
                "Dropping queued WhatsApp job chat=%s reason=queue_full", chat_key
            )
            return False

        worker = self._workers.get(chat_key)
        if worker is None or worker.done():
            self._workers[chat_key] = asyncio.create_task(self._run(chat_key, queue))
        return True

    async def _run(self, chat_key: str, queue: asyncio.Queue[T]) -> None:
        while True:
            item = await queue.get()
            try:
                await self._worker(item)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error(
                    "Queued WhatsApp job failed chat=%s error=%s",
                    chat_key,
                    type(error).__name__,
                )
            finally:
                queue.task_done()

    async def close(self) -> None:
        workers = list(self._workers.values())
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        self._workers.clear()
        self._queues.clear()
