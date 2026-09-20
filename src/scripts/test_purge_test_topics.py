from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.purge_test_topics import purge_topics


@pytest.mark.asyncio
async def test_purge_deletes_only_non_faq_topics_in_target_groups():
    connection = MagicMock()
    connection.execute = AsyncMock(
        side_effect=[MagicMock(rowcount=4), MagicMock(rowcount=2)]
    )

    linked, topics = await purge_topics(connection)

    assert (linked, topics) == (4, 2)
    statements = [call.args[0].text for call in connection.execute.await_args_list]
    assert all("kbtopic" in statement for statement in statements)
    assert all(
        "speakers" in statement and "faq" in statement for statement in statements
    )
    assert all(
        "group_one" in statement and "group_two" in statement
        for statement in statements
    )
    assert connection.execute.await_args_list[0].args[1] == {
        "group_one": "120363413079068449@g.us",
        "group_two": "120363428762021095@g.us",
    }
