"""Parser unit tests (NFR-20)."""

from pathlib import Path

import pytest

from src.ingestion.parsers import ParserError, get_parser
from src.ingestion.parsers.docx import DocxParser
from src.ingestion.parsers.html import HtmlParser
from src.ingestion.parsers.pdf import PdfParser
from src.ingestion.parsers.txt import PlainTextParser

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


class TestPdfParser:
    def test_extracts_text_and_page_numbers(self) -> None:
        result = PdfParser().parse(FIXTURES / "sample.pdf")
        assert len(result.pages) == 2
        assert result.pages[0].page_number == 1
        assert result.pages[1].page_number == 2
        assert "Deployment Guide" in result.pages[0].text
        assert "Troubleshooting" in result.pages[1].text

    def test_scanned_pdf_raises_empty_extraction(self) -> None:
        with pytest.raises(ParserError) as exc_info:
            PdfParser().parse(FIXTURES / "scanned_only.pdf")
        assert exc_info.value.reason == "empty_extraction"

    def test_corrupt_file_raises_corrupt_file_error(self, tmp_path: Path) -> None:
        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_bytes(b"not a real pdf")
        with pytest.raises(ParserError) as exc_info:
            PdfParser().parse(bad_pdf)
        assert exc_info.value.reason == "corrupt_file"


class TestDocxParser:
    def test_extracts_paragraph_text(self) -> None:
        result = DocxParser().parse(FIXTURES / "sample.docx")
        assert result.pages[0].page_number is None
        assert "Sample Document" in result.full_text
        assert "sentence-aware chunker" in result.full_text

    def test_corrupt_file_raises_corrupt_file_error(self, tmp_path: Path) -> None:
        bad_docx = tmp_path / "bad.docx"
        bad_docx.write_bytes(b"not a real docx")
        with pytest.raises(ParserError) as exc_info:
            DocxParser().parse(bad_docx)
        assert exc_info.value.reason == "corrupt_file"


class TestHtmlParser:
    def test_extracts_text_and_strips_scripts(self) -> None:
        result = HtmlParser().parse(FIXTURES / "sample.html")
        assert "Sample HTML Document" in result.full_text
        assert "should never appear" not in result.full_text

    def test_empty_html_raises_empty_extraction(self, tmp_path: Path) -> None:
        empty_html = tmp_path / "empty.html"
        empty_html.write_text("<html><body></body></html>")
        with pytest.raises(ParserError) as exc_info:
            HtmlParser().parse(empty_html)
        assert exc_info.value.reason == "empty_extraction"


class TestPlainTextParser:
    def test_extracts_text(self) -> None:
        result = PlainTextParser().parse(FIXTURES / "sample.txt")
        assert "sample plain text document" in result.full_text

    def test_empty_file_raises_empty_extraction(self, tmp_path: Path) -> None:
        empty_txt = tmp_path / "empty.txt"
        empty_txt.write_text("")
        with pytest.raises(ParserError) as exc_info:
            PlainTextParser().parse(empty_txt)
        assert exc_info.value.reason == "empty_extraction"


class TestParserDispatch:
    @pytest.mark.parametrize(
        ("filename", "expected_cls"),
        [
            ("doc.pdf", PdfParser),
            ("doc.PDF", PdfParser),
            ("doc.docx", DocxParser),
            ("doc.html", HtmlParser),
            ("doc.htm", HtmlParser),
            ("doc.txt", PlainTextParser),
        ],
    )
    def test_dispatches_correct_parser(self, filename: str, expected_cls: type) -> None:
        assert isinstance(get_parser(filename), expected_cls)

    def test_unsupported_extension_raises_clear_error(self) -> None:
        with pytest.raises(ParserError) as exc_info:
            get_parser("doc.xlsx")
        assert exc_info.value.reason == "unsupported_format"
