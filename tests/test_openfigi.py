"""Unit tests for OpenFIGIMapper module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.ingestion.openfigi import (
    FIGIMappingResult,
    OpenFIGIMapper,
    OpenFIGIRateLimitError,
)


def test_openfigi_invalid_isin_raises_value_error(tmp_path: Path) -> None:
    # Given: OpenFIGIMapper instance and invalid ISIN string
    mapper = OpenFIGIMapper(cache_dir=str(tmp_path))

    # When / Then: Raising ValueError on invalid ISIN lengths or empty strings
    with pytest.raises(ValueError, match="Invalid ISIN code"):
        mapper.map_isin("SHORT")

    with pytest.raises(ValueError, match="Invalid ISIN code"):
        mapper.map_isin("")


def test_openfigi_cache_hit(tmp_path: Path) -> None:
    # Given: Cached JSON file for valid ISIN
    cache_dir = str(tmp_path)
    isin = "US0378331005"
    cache_file = tmp_path / f"{isin}.json"
    cache_file.write_text(json.dumps({"ticker": "AAPL", "name": "APPLE INC"}))

    mapper = OpenFIGIMapper(cache_dir=cache_dir)

    # When: Querying map_isin for cached ISIN
    res = mapper.map_isin(isin)

    # Then: Returns FIGIMappingResult from cache without API calls
    assert isinstance(res, FIGIMappingResult)
    assert res.ticker == "AAPL"
    assert res.name == "APPLE INC"


@patch("requests.post")
def test_openfigi_api_success_and_caching(mock_post: MagicMock, tmp_path: Path) -> None:
    # Given: OpenFIGI API returning successful 200 mapping response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"data": [{"ticker": "ENI", "name": "ENI SPA"}]}]
    mock_post.return_value = mock_resp

    mapper = OpenFIGIMapper(cache_dir=str(tmp_path))
    isin = "IT0003132476"

    # When: Mapping ISIN via API
    res = mapper.map_isin(isin)

    # Then: Ticker and name resolved and saved to local disk cache
    assert res.ticker == "ENI"
    assert res.name == "ENI SPA"

    # Verify cache file created
    cache_file = tmp_path / f"{isin}.json"
    assert cache_file.exists()


@patch("requests.post")
def test_openfigi_rate_limit_raises_error(mock_post: MagicMock, tmp_path: Path) -> None:
    # Given: OpenFIGI API returning 429 rate limit repeatedly
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_post.return_value = mock_resp

    mapper = OpenFIGIMapper(cache_dir=str(tmp_path))

    # When / Then: OpenFIGIRateLimitError raised after max retries
    with patch("time.sleep"):
        with pytest.raises(OpenFIGIRateLimitError, match="rate limit exceeded"):
            mapper.map_isin("US0378331005")
