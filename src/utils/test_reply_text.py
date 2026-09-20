from utils.reply_text import clean_visible_reply


def test_converts_markdown_bold_and_run_on_bullets():
    assert clean_visible_reply("**Key:** one - two - three") == (
        "*Key:* one\n- two\n- three"
    )


def test_puts_each_link_on_its_own_line():
    assert clean_visible_reply("Read https://example.org/rules then ask.") == (
        "Read\nhttps://example.org/rules\nthen ask."
    )
