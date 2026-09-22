from utils.reply_text import (
    WHATSAPP_MAX_CHUNK_CHARS,
    chunk_reply_text,
    clean_visible_reply,
    limit_reply_text,
)


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


def test_only_explicitly_trusted_phone_numbers_survive_cleanup():
    assert (
        clean_visible_reply(
            "Contact Diana at +250 783 188 655 or an unverified number +999 111 222 333.",
            allowed_phone_numbers=("+250783188655",),
        )
        == "Contact Diana at +250 783 188 655 or an unverified number ."
    )


def test_normal_reply_is_not_resized():
    text = "A concise answer with the supported detail."

    assert limit_reply_text(text) == text


def test_long_reply_uses_the_full_logical_budget_and_explains_limitation():
    text = "Sentence with supported detail. " * 300

    limited = limit_reply_text(text)

    assert len(limited) <= 5000
    assert "shortened this answer" in limited


def test_chunking_preserves_content_and_transport_limit():
    text = (
        ("Paragraph one with supported detail. " * 100)
        + "\n\n"
        + ("Paragraph two with supported detail. " * 25)
    )

    chunks = chunk_reply_text(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= WHATSAPP_MAX_CHUNK_CHARS for chunk in chunks)
    assert "".join(chunks) == text
    assert len("".join(chunks)) <= 5000
