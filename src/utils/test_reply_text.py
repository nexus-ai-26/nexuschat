from utils.reply_text import clean_visible_reply


def test_converts_markdown_bold_and_run_on_bullets():
    assert clean_visible_reply("**Key:** one - two - three") == (
        "*Key:* one\n- two\n- three"
    )


def test_puts_each_link_on_its_own_line():
    assert clean_visible_reply("Read https://example.org/rules then ask.") == (
        "Read\nhttps://example.org/rules\nthen ask."
    )


def test_formats_gypcy_links_as_labelled_lines_and_strips_url_punctuation():
    gypcy_reply = (
        "Gypcy helps with applications.\n\n"
        "Links: application form https://gypcy.example/app; guidelines "
        "https://gypcy.example/guide,.\n\n"
        "Contact the organisers if you need help."
    )

    assert clean_visible_reply(gypcy_reply) == (
        "Gypcy helps with applications.\n\n"
        "Links:\napplication form\nhttps://gypcy.example/app\n"
        "guidelines\nhttps://gypcy.example/guide\n\n"
        "Contact the organisers if you need help."
    )


def test_removes_leading_spaces_from_every_formatted_line():
    assert clean_visible_reply("  *Key:* value\n   - item\n") == (
        "*Key:* value\n- item"
    )
