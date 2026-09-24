from io import BytesIO
from zipfile import ZipFile

from services.document_ingestion import (
    chunk_document_text,
    extract_document_text,
)


def test_extracts_text_and_chunks_a_txt_document():
    text = extract_document_text(
        b"What to submit\n\nDeadline: 30 September", "guidelines.txt"
    )

    assert "What to submit" in text
    assert chunk_document_text(text) == ["What to submit\n\nDeadline: 30 September"]


def test_extracts_text_from_a_docx_document():
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr(
            "word/document.xml",
            "<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>"
            "<w:body><w:p><w:r><w:t>Submit a proposal.</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>Deadline: 30 September.</w:t></w:r></w:p></w:body></w:document>",
        )

    text = extract_document_text(output.getvalue(), "guidelines.docx")

    assert "Submit a proposal." in text
    assert "Deadline: 30 September." in text
