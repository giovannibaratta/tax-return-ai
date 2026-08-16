"""Tests for TODO refactoring and edge cases in backend/ingestion/ocr_parser.py and helpers.py."""

import os
from unittest.mock import patch

import pytest

from backend.ingestion.ocr_parser import (
    ChandraAPIParser,
    DatalabCheckResultResponse,
    DatalabConvertResponse,
    _get_ocr_cache_path,
    _split_markdown_into_pages,
)


class TestOCRParserTODOs:
    def test_cache_path_raises_file_not_found_if_pdf_missing(self) -> None:
        # Given: Non-existent file path string
        missing_file = "/tmp/non_existent_file_12345.pdf"

        # When & Then: FileNotFoundError is raised
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _get_ocr_cache_path(missing_file, "tag", cache_dir=".cache/ocr")

    def test_api_parser_raises_value_error_for_invalid_mode(self) -> None:
        # Given: Invalid Datalab mode
        # When & Then: ValueError is raised
        with pytest.raises(ValueError, match="Invalid mode 'invalid_mode'"):
            ChandraAPIParser.parse_pdf("fake.pdf", mode="invalid_mode")  # pyright: ignore[reportArgumentType]

    def test_api_parser_raises_when_datalab_key_missing(self, tmp_path) -> None:
        # Given: Existing PDF file but no DATALAB_API_KEY set
        dummy_pdf = tmp_path / "test.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 dummy content")

        with patch.dict(os.environ, {"DATALAB_API_KEY": "", "CHANDRA_API_KEY": ""}, clear=True):
            # When & Then: ValueError is raised
            with pytest.raises(ValueError, match="Datalab API key missing"):
                ChandraAPIParser.parse_pdf(str(dummy_pdf))

    def test_datalab_pydantic_response_models(self) -> None:
        # Given: JSON payload from Datalab API endpoints
        convert_json = {
            "request_check_url": "https://www.datalab.to/api/v1/check/123",
            "status": "pending",
        }
        check_json = {
            "status": "complete",
            "markdown": "# Page 1\nContent",
            "result_url": None,
        }

        # When: Validated using Pydantic models
        convert_res = DatalabConvertResponse.model_validate(convert_json)
        check_res = DatalabCheckResultResponse.model_validate(check_json)

        # Then: Attributes parsed correctly
        assert convert_res.request_check_url == "https://www.datalab.to/api/v1/check/123"
        assert convert_res.status == "pending"
        assert check_res.status == "complete"
        assert check_res.markdown == "# Page 1\nContent"

    def test_split_markdown_fallbacks(self) -> None:
        # Given: Markdown split by Form Feed \f
        form_feed_md = "Page 1 Content\fPage 2 Content"
        pages_ff = _split_markdown_into_pages(form_feed_md)
        assert pages_ff == ["Page 1 Content", "Page 2 Content"]

        # Given: Markdown split by horizontal rules
        hr_md = "Page A Content\n---\nPage B Content"
        pages_hr = _split_markdown_into_pages(hr_md)
        assert pages_hr == ["Page A Content", "Page B Content"]
