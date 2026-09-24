import asyncio

import pytest

from services.webhook_queue import PerChatQueue


@pytest.mark.asyncio
async def test_jobs_for_one_chat_run_in_order():
    events: list[str] = []

    async def worker(item: str) -> None:
        events.append(f"start-{item}")
        await asyncio.sleep(0)
        events.append(f"end-{item}")

    queue = PerChatQueue(worker)
    assert queue.enqueue("group@g.us", "one") is True
    assert queue.enqueue("group@g.us", "two") is True

    for _ in range(10):
        await asyncio.sleep(0)
        if events == ["start-one", "end-one", "start-two", "end-two"]:
            break

    assert events == ["start-one", "end-one", "start-two", "end-two"]
    await queue.close()
