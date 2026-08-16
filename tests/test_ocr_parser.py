"""Unit tests for ChandraOCRParser.

Heavy model inference (InferenceManager) is mocked out so tests run on
CPU in CI without a GPU or a downloaded VLM checkpoint. PDF rasterisation
(pymupdf) is also mocked to avoid needing a real PDF file.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.ingestion.ocr_parser import (
    ChandraAPIParser,
    ChandraOCRParser,
    DatalabConvertResponse,
    _get_inference_manager,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fitz_doc_mock(markdown: str, raw: str = "") -> tuple[MagicMock, MagicMock]:
    """Return (fitz_doc_mock, manager_mock) for a single-page PDF."""
    pixmap = MagicMock()
    pixmap.width = 10
    pixmap.height = 10
    pixmap.samples = b"\xff" * (10 * 10 * 3)

    fitz_page = MagicMock()
    fitz_page.get_pixmap.return_value = pixmap

    fitz_doc = MagicMock()
    fitz_doc.__enter__ = MagicMock(return_value=fitz_doc)
    fitz_doc.__exit__ = MagicMock(return_value=False)
    fitz_doc.page_count = 1
    fitz_doc.__getitem__ = MagicMock(return_value=fitz_page)

    chandra_result = MagicMock()
    chandra_result.markdown = markdown
    chandra_result.raw = raw

    manager_mock = MagicMock()
    manager_mock.generate.return_value = [chandra_result]

    return fitz_doc, manager_mock


# ---------------------------------------------------------------------------
# ChandraOCRParser.parse_pdf (integration with mocks)
# ---------------------------------------------------------------------------


class TestChandraOCRParserParsePDF:
    def test_single_page_pdf_produces_one_parsed_page(self, tmp_path: Path) -> None:
        # Given: a single-page PDF returning clean markdown from Chandra
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 dummy content")
        sample_markdown = (
            "# Section 1\n\nThis is the body text.\n\n| Col A | Col B |\n| --- | --- |\n| Value 1 | Value 2 |"
        )
        fitz_doc, manager_mock = _make_fitz_doc_mock(markdown=sample_markdown)

        with (
            patch("backend.ingestion.ocr_parser.fitz.open", return_value=fitz_doc),
            patch("backend.ingestion.ocr_parser._get_inference_manager", return_value=manager_mock),
        ):
            # When: parse_pdf called
            pages = ChandraOCRParser.parse_pdf(str(test_pdf), cache_dir=str(tmp_path / "cache"))

        # Then: generate called with include_headers_footers=False
        manager_mock.generate.assert_called_once()
        assert manager_mock.generate.call_args is not None
        call_kwargs = manager_mock.generate.call_args.kwargs
        assert call_kwargs.get("include_headers_footers") is False

        # Then: one page with expected content
        assert len(pages) == 1
        page = pages[0]
        assert page.page_number == 1
        assert "body text" in page.combined_content
        assert "Col A" in page.combined_content

    def test_empty_markdown_falls_back_to_raw(self, tmp_path: Path) -> None:
        # Given: Chandra returns empty markdown but non-empty raw
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 dummy content")
        fitz_doc, manager_mock = _make_fitz_doc_mock(
            markdown="",
            raw="Fallback raw text from model output.",
        )

        with (
            patch("backend.ingestion.ocr_parser.fitz.open", return_value=fitz_doc),
            patch("backend.ingestion.ocr_parser._get_inference_manager", return_value=manager_mock),
        ):
            # When: parse_pdf called
            pages = ChandraOCRParser.parse_pdf(str(test_pdf), cache_dir=str(tmp_path / "cache"))

        # Then: raw content used instead of raising
        assert len(pages) == 1
        page = pages[0]
        assert "Fallback raw text" in page.combined_content

    def test_generate_receives_include_headers_footers_false(self, tmp_path: Path) -> None:
        # Given: a minimal single-page PDF with body text
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 dummy content")
        fitz_doc, manager_mock = _make_fitz_doc_mock(markdown="Body only text, no headers or footers.")

        with (
            patch("backend.ingestion.ocr_parser.fitz.open", return_value=fitz_doc),
            patch("backend.ingestion.ocr_parser._get_inference_manager", return_value=manager_mock),
        ):
            # When: parse_pdf called
            ChandraOCRParser.parse_pdf(str(test_pdf), cache_dir=str(tmp_path / "cache"))

        # Then: Chandra's native filtering is engaged
        manager_mock.generate.assert_called_once()
        assert manager_mock.generate.call_args is not None
        call_kwargs = manager_mock.generate.call_args.kwargs
        assert call_kwargs.get("include_headers_footers") is False

    def test_custom_dpi_passed_to_rasteriser(self, tmp_path: Path) -> None:
        # Given: single-page PDF mock and custom dpi=75
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 dummy content")
        fitz_doc, manager_mock = _make_fitz_doc_mock(markdown="Sample text")

        with (
            patch("backend.ingestion.ocr_parser.fitz.open", return_value=fitz_doc),
            patch("backend.ingestion.ocr_parser._get_inference_manager", return_value=manager_mock),
            patch("backend.ingestion.ocr_parser._pdf_page_to_image") as mock_page_to_img,
        ):
            mock_page_to_img.return_value = MagicMock()

            # When: parse_pdf called with custom dpi=75
            pages = ChandraOCRParser.parse_pdf(str(test_pdf), dpi=75, cache_dir=str(tmp_path / "cache"))

        # Then: rasteriser called with dpi=75
        mock_page_to_img.assert_called_once_with(fitz_doc[0], dpi=75)
        assert len(pages) == 1


class TestChandraAPIParser:
    def test_raises_value_error_without_api_key(self, tmp_path: Path) -> None:
        # Given: existing PDF file but no API key in environment
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 dummy content")
        with patch.dict("os.environ", {}, clear=True):
            # When/Then: parse_pdf raises ValueError
            with pytest.raises(ValueError, match="Datalab API key missing"):
                ChandraAPIParser.parse_pdf(str(test_pdf))

    def test_successful_api_conversion(self, tmp_path: Path) -> None:
        # Given: valid API key and mocked requests
        upload_res = MagicMock()
        upload_res.json.return_value = {
            "upload_url": "https://upload.example.com",
            "file_id": 123,
            "reference": "datalab://file-123",
        }
        upload_res.ok = True

        convert_res = MagicMock()
        convert_res.json.return_value = {"request_check_url": "https://datalab.to/check/123"}
        convert_res.ok = True

        def post_side_effect(url: str, **kwargs: object) -> MagicMock:
            if "/files/upload" in url:
                return upload_res
            return convert_res

        put_res = MagicMock()
        put_res.ok = True

        confirm_res = MagicMock()
        confirm_res.json.return_value = {"success": True}
        confirm_res.ok = True

        poll_response = MagicMock()
        poll_response.json.return_value = {
            "status": "complete",
            "markdown": "# Heading\n\nPage 1 body text.\n\f# Page 2\n\nPage 2 body text.",
        }
        poll_response.raise_for_status.return_value = None

        def get_side_effect(url: str, **kwargs: object) -> MagicMock:
            if "/confirm" in url:
                return confirm_res
            return poll_response

        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"dummy pdf content")

        with (
            patch.dict("os.environ", {"DATALAB_API_KEY": "dummy_key"}),
            patch("requests.post", side_effect=post_side_effect) as mock_post,
            patch("requests.put", return_value=put_res),
            patch("requests.get", side_effect=get_side_effect) as mock_get,
            patch("time.sleep", return_value=None),
        ):
            # When: parse_pdf called via API
            pages = ChandraAPIParser.parse_pdf(str(test_pdf), cache_dir=str(tmp_path / "cache"))

        # Then: two pages parsed successfully
        assert mock_post.called
        assert mock_get.called
        assert len(pages) == 2
        assert pages[0].page_number == 1
        assert "Page 1 body text" in pages[0].combined_content
        assert pages[1].page_number == 2
        assert "Page 2 body text" in pages[1].combined_content


class TestOCRCaching:
    def test_cache_hit_bypasses_model_inference(self, tmp_path: Path) -> None:
        # Given: a PDF and pre-populated cache
        test_pdf = tmp_path / "sample.pdf"
        test_pdf.write_bytes(b"sample pdf data")
        cache_dir = tmp_path / "cache"

        sample_markdown = "Cached markdown content."
        fitz_doc, manager_mock = _make_fitz_doc_mock(markdown=sample_markdown)

        # First run: populates cache
        with (
            patch("backend.ingestion.ocr_parser.fitz.open", return_value=fitz_doc),
            patch("backend.ingestion.ocr_parser._get_inference_manager", return_value=manager_mock),
        ):
            pages_first = ChandraOCRParser.parse_pdf(str(test_pdf), cache_dir=str(cache_dir))
            assert manager_mock.generate.call_count == 1

        # Second run: should hit cache without calling model.generate
        manager_mock.reset_mock()
        with (
            patch("backend.ingestion.ocr_parser.fitz.open", return_value=fitz_doc),
            patch("backend.ingestion.ocr_parser._get_inference_manager", return_value=manager_mock),
        ):
            pages_second = ChandraOCRParser.parse_pdf(str(test_pdf), cache_dir=str(cache_dir))
            assert manager_mock.generate.call_count == 0
            assert len(pages_second) == len(pages_first)
            assert pages_second[0].combined_content == pages_first[0].combined_content

    def test_force_parsing_bypasses_cache(self, tmp_path: Path) -> None:
        # Given: a PDF and pre-populated cache
        test_pdf = tmp_path / "sample.pdf"
        test_pdf.write_bytes(b"sample pdf data")
        cache_dir = tmp_path / "cache"

        sample_markdown = "First run content."
        fitz_doc, manager_mock = _make_fitz_doc_mock(markdown=sample_markdown)

        # Populate cache
        with (
            patch("backend.ingestion.ocr_parser.fitz.open", return_value=fitz_doc),
            patch("backend.ingestion.ocr_parser._get_inference_manager", return_value=manager_mock),
        ):
            ChandraOCRParser.parse_pdf(str(test_pdf), cache_dir=str(cache_dir))

        # Second run with force_parsing=True: should call model.generate again
        manager_mock.reset_mock()
        with (
            patch("backend.ingestion.ocr_parser.fitz.open", return_value=fitz_doc),
            patch("backend.ingestion.ocr_parser._get_inference_manager", return_value=manager_mock),
        ):
            ChandraOCRParser.parse_pdf(str(test_pdf), force_parsing=True, cache_dir=str(cache_dir))
            assert manager_mock.generate.call_count == 1


class TestGetInferenceManager:
    def test_default_uses_local_hf_method(self) -> None:
        _get_inference_manager.cache_clear()
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("chandra.model.InferenceManager") as mock_mgr,
            patch("backend.ingestion.ocr_parser.auto_detect_device", return_value="cpu"),
        ):
            _get_inference_manager()
            mock_mgr.assert_called_once_with(method="hf")

    def test_vllm_api_base_triggers_vllm_method(self) -> None:
        _get_inference_manager.cache_clear()
        with (
            patch.dict("os.environ", {"VLLM_API_BASE": "http://192.168.1.50:8000/v1"}, clear=True),
            patch("chandra.model.InferenceManager") as mock_mgr,
        ):
            _get_inference_manager()
            mock_mgr.assert_called_once_with(method="vllm")

    def test_explicit_chandra_inference_method_override(self) -> None:
        _get_inference_manager.cache_clear()
        with (
            patch.dict("os.environ", {"CHANDRA_INFERENCE_METHOD": "vllm"}, clear=True),
            patch("chandra.model.InferenceManager") as mock_mgr,
        ):
            _get_inference_manager()
            mock_mgr.assert_called_once_with(method="vllm")


def test_chandra_api_processing_location(tmp_path: Path) -> None:
    """Test passing processing_location to Datalab API convert payload."""
    test_pdf = tmp_path / "sample_doc.pdf"
    test_pdf.write_bytes(b"dummy pdf content")

    mock_resp = DatalabConvertResponse(
        request_check_url="https://www.datalab.to/api/v1/convert/check_123", status="complete"
    )

    with (
        patch.dict("os.environ", {"DATALAB_API_KEY": "test_key", "CHANDRA_PROCESSING_LOCATION": "eu"}),
        patch.object(ChandraAPIParser, "_submit_api_job", return_value=mock_resp) as mock_submit,
        patch.object(ChandraAPIParser, "_poll_api_job", return_value="Parsed Markdown Content"),
    ):
        pages = ChandraAPIParser.parse_pdf(str(test_pdf), force_parsing=True, cache_dir=str(tmp_path / "cache"))
        assert len(pages) == 1
        assert pages[0].combined_content == "Parsed Markdown Content"
        mock_submit.assert_called_once()
        assert mock_submit.call_args is not None
        call_kwargs = mock_submit.call_args.kwargs
        assert call_kwargs.get("processing_location") == "eu"
