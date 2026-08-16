"""Maps ISIN codes to ticker symbols and friendly company names using OpenFIGI API with local disk caching."""

import json
import logging
import os
import time
from dataclasses import dataclass

import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class OpenFIGIError(Exception):
    """Base exception for OpenFIGI API lookup errors."""


class OpenFIGIRateLimitError(OpenFIGIError):
    """Exception raised when OpenFIGI API rate limit (HTTP 429) is exceeded."""


@dataclass
class FIGIMappingResult:
    """Result of OpenFIGI ISIN to ticker and name mapping."""

    ticker: str | None = None
    name: str | None = None


class OpenFIGIMatchItem(BaseModel):
    """Single matched security record from OpenFIGI API."""

    ticker: str | None = None
    name: str | None = None


class OpenFIGIResultItem(BaseModel):
    """Result item for a queried ISIN from OpenFIGI API."""

    data: list[OpenFIGIMatchItem] = []
    error: str | None = None


class OpenFIGIMapper:
    """Maps ISIN codes to ticker symbols and friendly company names using OpenFIGI API with local disk caching."""

    def __init__(self, cache_dir: str = "database/openfigi_cache", api_key: str | None = None) -> None:
        """Initialize OpenFIGIMapper.

        Args:
            cache_dir: Directory path for disk caching API responses.
            api_key: Optional OpenFIGI API key. Defaults to OPENFIGI_API_KEY environment variable.
        """
        self.cache_dir = cache_dir
        self.api_key = api_key or os.environ.get("OPENFIGI_API_KEY")
        os.makedirs(self.cache_dir, exist_ok=True)

    def map_isin(self, isin: str) -> FIGIMappingResult:
        """Map an ISIN to a FIGIMappingResult containing ticker symbol and company name.

        Checks local cache first before querying remote OpenFIGI API.

        Args:
            isin: 12-character ISIN identifier string.

        Returns:
            FIGIMappingResult containing resolved ticker and name (or None for unresolvable ISINs).

        Raises:
            ValueError: If ISIN string is empty or not exactly 12 characters.
            OpenFIGIRateLimitError: If OpenFIGI API rate limit is exceeded after retries.
            OpenFIGIError: If OpenFIGI API request fails.
        """
        if not isin or len(isin.strip()) != 12:
            raise ValueError(f"Invalid ISIN code '{isin}': ISIN must be a 12-character string.")

        isin_clean = isin.strip().upper()
        cache_path = os.path.join(self.cache_dir, f"{isin_clean}.json")

        # 1. Check local cache
        cached_result = self._read_cache(cache_path)
        if cached_result is not None:
            return cached_result

        # 2. Query OpenFIGI API
        return self._fetch_from_api(isin_clean, cache_path)

    def _read_cache(self, cache_path: str) -> FIGIMappingResult | None:
        """Read and validate cached mapping from disk.

        Args:
            cache_path: File path to cached JSON.

        Returns:
            FIGIMappingResult if valid cache exists, None otherwise.
        """
        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, encoding="utf-8") as f:
                data: object = json.load(f)
                if isinstance(data, dict):
                    # pyright: ignore[reportUnknownVariableType]
                    raw_dict: dict[object, object] = data
                    ticker_val = raw_dict.get("ticker")
                    name_val = raw_dict.get("name")
                    return FIGIMappingResult(
                        ticker=str(ticker_val) if isinstance(ticker_val, str) else None,
                        name=str(name_val) if isinstance(name_val, str) else None,
                    )
        except Exception as exc:
            logger.warning(f"Failed to read OpenFIGI cache at {cache_path}: {exc}")

        return None

    def _fetch_from_api(self, isin_clean: str, cache_path: str) -> FIGIMappingResult:
        """Query OpenFIGI API with retries and cache the result on success.

        Args:
            isin_clean: Normalized 12-character ISIN code.
            cache_path: Disk path to store cached JSON result.

        Returns:
            Resolved FIGIMappingResult.

        Raises:
            OpenFIGIRateLimitError: If rate limited (HTTP 429) after all retries.
            OpenFIGIError: If API request returns non-200 or network fails.
        """
        logger.info(f"Querying OpenFIGI API for ISIN: {isin_clean}...")
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-OPENFIGI-APIKEY"] = self.api_key

        payload = [{"idType": "ID_ISIN", "idValue": isin_clean}]
        max_retries = 3

        for attempt in range(max_retries):
            try:
                res = requests.post(
                    "https://api.openfigi.com/v3/mapping",
                    headers=headers,
                    json=payload,
                    timeout=10,
                )

                if res.status_code == 200:
                    raw_json: object = res.json()
                    if isinstance(raw_json, list) and raw_json:
                        # pyright: ignore[reportUnknownVariableType]
                        first_item: object = raw_json[0]
                        if isinstance(first_item, dict):
                            result_item = OpenFIGIResultItem.model_validate(first_item)
                            if result_item.data:
                                match = result_item.data[0]
                                mapping = FIGIMappingResult(ticker=match.ticker, name=match.name)

                                result_to_cache = {
                                    "ticker": mapping.ticker,
                                    "name": mapping.name,
                                }
                                with open(cache_path, "w", encoding="utf-8") as f:
                                    json.dump(result_to_cache, f, indent=2)

                                return mapping

                    # Cache empty result for unresolvable ISINs to prevent repeated API calls
                    empty_mapping = FIGIMappingResult(ticker=None, name=None)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump({"ticker": None, "name": None}, f, indent=2)
                    return empty_mapping

                elif res.status_code == 429:
                    if attempt < max_retries - 1:
                        backoff = 2**attempt
                        logger.warning(
                            f"OpenFIGI API rate limited (HTTP 429) for {isin_clean}. Retrying in {backoff}s..."
                        )
                        time.sleep(backoff)
                        continue
                    raise OpenFIGIRateLimitError(
                        f"OpenFIGI API rate limit exceeded (HTTP 429) for ISIN '{isin_clean}' after {max_retries} attempts."
                    )
                else:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    raise OpenFIGIError(f"OpenFIGI API returned status code {res.status_code}: {res.text}")

            except requests.RequestException as exc:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise OpenFIGIError(f"Failed to query OpenFIGI API for ISIN '{isin_clean}': {exc}") from exc

        raise OpenFIGIError(f"Failed to resolve ISIN '{isin_clean}' after {max_retries} retries.")
