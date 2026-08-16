"""Chandra OCR-2 based PDF parsers (Local PyTorch and Datalab Cloud API).

Provides both local (:class:`ChandraOCRParser`) and cloud API (:class:`ChandraAPIParser`)
implementations producing :class:`ParsedPage` objects with the same interface as
:class:`PDFDocumentParser`, allowing drop-in replacement in the ingestion pipeline.

Includes local disk caching of raw OCR outputs (pre-chunking) to prevent
redundant model inference or API calls.
"""

from __future__ import annotations

import glob
import json
import os
import re
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, override

import fitz  # PyMuPDF: Python binding for the MuPDF C graphics and PDF library
import requests
from chandra.model.schema import BatchInputItem
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, TypeAdapter

from backend.ingestion.helpers import calculate_sha256
from backend.ingestion.parser import BasePDFParser, ParsedPage
from backend.utils.utils import auto_detect_device

if TYPE_CHECKING:
    from chandra.model import InferenceManager

_ = load_dotenv()


# Pixels-per-inch used when rasterising PDF pages locally.
_DEFAULT_DPI: int = 300

# Chandra prompt type — uses full layout-aware OCR with image captioning.
_PROMPT_TYPE: str = "ocr_layout"

DATALAB_API_URL: str = "https://www.datalab.to/api/v1"

# Allowed mode parameters for Datalab API conversion
DatalabMode = Literal["balanced", "fast", "accurate"]
VALID_DATALAB_MODES: set[str] = {"balanced", "fast", "accurate"}


class DatalabConvertResponse(BaseModel):
    """Pydantic model validating response from Datalab POST /convert endpoint.

    Ref: https://documentation.datalab.to/api-reference/convert-document.md
    """

    request_check_url: str
    status: str | None = None


class DatalabCheckResultResponse(BaseModel):
    """Pydantic model validating response from Datalab GET check status endpoint.

    Ref: https://documentation.datalab.to/api-reference/convert-result-check.md
    """

    status: str
    markdown: str | None = None
    result_url: str | None = None
    error: str | None = None


@lru_cache(maxsize=1)
def _get_inference_manager() -> InferenceManager:
    """Return a cached singleton InferenceManager.

    Supports both local PyTorch ('hf') and remote vLLM server ('vllm') inference.
    Controlled via environment variables:
      - ``CHANDRA_INFERENCE_METHOD``: 'vllm' or 'hf' (default: 'hf')
      - ``VLLM_API_BASE``: remote CUDA server API URL (e.g. 'http://192.168.1.100:8000/v1')
      - ``VLLM_MODEL_NAME``: model name on vLLM server (default: 'datalab-to/chandra-ocr-2')

    Returns:
        Configured InferenceManager singleton.
    """
    from chandra.model import InferenceManager

    method = os.environ.get("CHANDRA_INFERENCE_METHOD", "").lower().strip()
    if not method:
        method = "vllm" if os.environ.get("VLLM_API_BASE") else "hf"

    if method == "vllm":
        api_base = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
        model_name = os.environ.get("VLLM_MODEL_NAME", "datalab-to/chandra-ocr-2")
        print(f"Initialising ChandraOCR InferenceManager (vLLM API: {api_base}, model: {model_name}) ...")
        return InferenceManager(method="vllm")

    device = auto_detect_device()
    if not os.environ.get("TORCH_DEVICE"):
        os.environ["TORCH_DEVICE"] = device
    # TORCH_ATTN="sdpa" enables PyTorch Scaled Dot-Product Attention, providing
    # optimized memory footprint and execution speed without external CUDA extension dependencies.
    if not os.environ.get("TORCH_ATTN"):
        os.environ["TORCH_ATTN"] = "sdpa"
    print(
        f"Initialising ChandraOCR InferenceManager (local PyTorch on device '{device}', attn: {os.environ.get('TORCH_ATTN')}) ..."
    )
    return InferenceManager(method="hf")


def _pdf_page_to_image(page: fitz.Page, dpi: int) -> Image.Image:
    """Rasterise a single PyMuPDF page to a PIL Image.

    PyMuPDF operates in a native coordinate resolution of 72 DPI (1 point = 1/72 inch).
    To render at a target DPI, a uniform transformation matrix scaling factor `zoom = dpi / 72.0`
    is passed to `page.get_pixmap`.

    At 200 DPI (default), an A4 page (595x842 points) renders to ~1654x2339 pixels, balancing
    high OCR recognition accuracy for dense 6-8pt tax document fonts with efficient VRAM consumption.

    Args:
        page: A PyMuPDF Page object.
        dpi: Target resolution in dots-per-inch.

    Returns:
        PIL Image in RGB mode.
    """
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def _get_ocr_cache_path(file_path: str, parser_tag: str, cache_dir: str = ".cache/ocr") -> str:
    """Generate a deterministic cache filepath for pre-chunking OCR output.

    Args:
        file_path: Path to the input PDF file (must exist).
        parser_tag: Unique tag describing parser & params (e.g. 'chandra_local_200dpi').
        cache_dir: Target cache directory.

    Returns:
        Absolute or relative path to the cache JSON file.

    Raises:
        FileNotFoundError: If input file_path does not exist on disk.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input PDF file path '{file_path}' does not exist.")

    file_sha = calculate_sha256(file_path)
    os.makedirs(cache_dir, exist_ok=True)
    exact_path = os.path.join(cache_dir, f"{file_sha}_{parser_tag}.json")
    if os.path.exists(exact_path):
        return exact_path

    # TODO: If the parser_tag is specified, we should not look for other cache files.
    # This may indicate an issue with the way the cache is being used.
    matches = glob.glob(os.path.join(cache_dir, f"{file_sha}_chandra*.json"))
    if matches:
        return sorted(matches)[0]

    return exact_path


def _load_cached_pages(cache_path: str) -> list[ParsedPage] | None:
    """Load cached ParsedPage objects from disk if present.

    Args:
        cache_path: Path to cache JSON file.

    Returns:
        List of ParsedPage objects if cache hit and valid, else None.
    """
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, encoding="utf-8") as f:
            data: list[dict[str, Any]] = json.load(f)
        pages = TypeAdapter(list[ParsedPage]).validate_python(data)
        for page in pages:
            page.combined_content = clean_ocr_markdown(page.combined_content)
        print(f"  - **Loaded cached OCR output** ({len(pages)} page(s)) from '{os.path.basename(cache_path)}'.")
        return pages
    except Exception as e:
        print(f"  - ⚠️ Warning: Failed to load OCR cache from '{cache_path}': {e}")
        return None


def _save_cached_pages(cache_path: str, pages: list[ParsedPage]) -> None:
    """Save ParsedPage objects to local disk cache as JSON.

    Args:
        cache_path: Target cache JSON filepath.
        pages: List of ParsedPage objects to serialize.
    """
    try:
        data = [page.model_dump() for page in pages]
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  - **Saved OCR output** ({len(pages)} page(s)) to cache '{os.path.basename(cache_path)}'.")
    except Exception as e:
        print(f"  - ⚠️ Warning: Failed to save OCR cache to '{cache_path}': {e}")


def clean_ocr_markdown(markdown_text: str) -> str:
    """Sanitize and format OCR markdown text to ensure proper paragraph breaks around headers.

    Args:
        markdown_text: Raw OCR markdown string.

    Returns:
        Formatted markdown string with proper newlines around section headers.
    """
    if not markdown_text:
        return markdown_text

    text = markdown_text
    if "bbox_norm" in text or "block_type" in text:
        # Convert HTML headers like <h4>Title</h4> to Markdown #### Title
        text = re.sub(r"<h([1-6])\s*>(.*?)</h\1>", lambda m: f"\n\n{'#' * int(m.group(1))} {m.group(2)}\n\n", text)
        # Convert HTML paragraph tags <p> text </p> to clean line breaks
        text = re.sub(r"</?p\s*>", "\n", text)
        # Strip JSON block layout objects and bbox_norm arrays
        text = re.sub(r'\{[^{}]*"bbox_norm"[^{}]*\}', "", text)
        text = re.sub(r'"bbox_norm"\s*:\s*\[[^\]]*\]', "", text)
        text = re.sub(r'"block_type"\s*:\s*"[^"]*"', "", text)
        text = re.sub(r'"block_id"\s*:\s*\d+', "", text)
        text = re.sub(r"</?div[^>]*>", "\n", text)

    # Ensure double newlines before inline markdown headers (#, ##, ###, ####, #####, ######)
    text = re.sub(r"(?<!\n)\s*(#{1,6}\s+)", r"\n\n\1", text)

    # Ensure double newlines after bracketed section headers like "#### Title [s.747E(6)] Text"
    text = re.sub(r"(#{1,6}\s+[^\[\n]+\s*\[[^\]]+\])\s+(?=[A-Z0-9])", r"\1\n\n", text)

    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _split_markdown_into_pages(markdown_text: str) -> list[str]:
    """Split a full document markdown string into per-page markdown chunks.

    Applies sequential fallback logic:
      1. Primary Datalab API pagination tags (`{0}------------...`).
      2. Form feed character (`\\f` or `\\x0c`).
      3. Explicit pagebreak comments (`<!-- pagebreak -->`) or markdown horizontal rules (`---`).

    Args:
        markdown_text: Full document markdown returned by Datalab API.

    Returns:
        List of page markdown strings.
    """
    # 1. Datalab API pagination pattern: {0}------------------------------------------------
    datalab_splits = re.split(r"\n*\{\d+\}-{10,}\n*", markdown_text)
    if len(datalab_splits) > 1:
        pages = [p.strip() for p in datalab_splits if p.strip()]
        if pages:
            return pages

    # 2. Form Feed character (\f or \x0c) if present in raw document
    if "\f" in markdown_text:
        pages = [p.strip() for p in markdown_text.split("\f") if p.strip()]
        if pages:
            return pages

    # 3. Split by explicit pagebreak comments or markdown horizontal rules
    page_splits = re.split(r"\n\s*(?:<!--\s*pagebreak\s*-->|[-*]{3,})\s*\n", markdown_text)
    if len(page_splits) > 1:
        pages = [p.strip() for p in page_splits if p.strip()]
        if pages:
            return pages

    return [markdown_text.strip()]


class ChandraOCRParser(BasePDFParser):
    """PDF parser backed by the local Chandra OCR-2 vision-language model.

    Produces :class:`ParsedPage` objects compatible with the rest of the
    ingestion pipeline. Sequential page-by-page processing.
    Caches output locally before chunking.
    """

    @classmethod
    @override
    def parse_pdf(
        cls,
        file_path: str,
        force_parsing: bool = False,
        dpi: int = _DEFAULT_DPI,
        cache_dir: str = ".cache/ocr",
        **kwargs: object,
    ) -> list[ParsedPage]:
        """Parse a PDF document page by page sequentially using local Chandra OCR-2.

        Args:
            file_path: Absolute or relative path to a PDF file.
            force_parsing: If True, bypasses local cache and forces full re-processing.
            dpi: Resolution in dots-per-inch for page rasterisation. Default is 200.
            cache_dir: Local directory for pre-chunking OCR cache files.

        Returns:
            List of :class:`ParsedPage` objects, one per PDF page.

        Raises:
            Exception: If a page yields no content after OCR.
        """
        cache_tag = f"chandra_local_{dpi}dpi"
        cache_path = _get_ocr_cache_path(file_path, cache_tag, cache_dir)

        if not force_parsing:
            cached_pages = _load_cached_pages(cache_path)
            if cached_pages:
                return cached_pages

        manager = _get_inference_manager()
        parsed_pages: list[ParsedPage] = []

        print(f"ChandraOCRParser (Local): opening '{file_path}' (DPI: {dpi}) ...")

        with fitz.open(file_path) as doc:
            total_pages = doc.page_count
            print(f"  Total pages: {total_pages}")

            for page_idx in range(total_pages):
                page_num = page_idx + 1
                fitz_page = doc[page_idx]

                print(f"  Processing page {page_num}/{total_pages} ...")

                t0 = time.perf_counter()
                pil_image = _pdf_page_to_image(fitz_page, dpi=dpi)
                raster_time_ms = (time.perf_counter() - t0) * 1000.0
                print(f"    Page {page_num} rasterised in {raster_time_ms:.1f}ms")

                batch = [BatchInputItem(image=pil_image, prompt_type=_PROMPT_TYPE)]
                result = manager.generate(batch, include_headers_footers=False)[0]

                clean_markdown = result.markdown or ""

                if not clean_markdown.strip() and result.raw:
                    print(f"  Warning: page {page_num} markdown empty, falling back to raw output.")
                    clean_markdown = result.raw.strip()

                if not clean_markdown:
                    raise Exception(
                        f"ChandraOCRParser: page {page_num} of '{file_path}' "
                        + "yielded no content after OCR and header/footer removal."
                    )

                combined_content = clean_markdown

                parsed_pages.append(
                    ParsedPage(
                        page_number=page_num,
                        combined_content=combined_content,
                    )
                )

        print(f"ChandraOCRParser (Local): done. Extracted {len(parsed_pages)} pages.")
        _save_cached_pages(cache_path, parsed_pages)
        return parsed_pages


class ChandraAPIParser(BasePDFParser):
    """PDF parser using the Datalab (Chandra) Cloud API.

    API Key is loaded strictly from ``DATALAB_API_KEY`` environment variable.
    Results are cached locally to disk before chunking to prevent redundant API calls.
    """

    API_URL: str = DATALAB_API_URL

    @classmethod
    def _submit_api_job(
        cls, file_path: str, mode: str, headers: dict[str, str], processing_location: str | None = None
    ) -> DatalabConvertResponse:
        """Submit PDF conversion job to Datalab API.

        Args:
            file_path: Path to PDF file.
            mode: API mode ('balanced', 'fast', 'accurate').
            headers: HTTP request headers containing API key.
            processing_location: Optional residency region override ('us', 'eu').

        Returns:
            Validated DatalabConvertResponse model.
        """
        filename = os.path.basename(file_path)
        # TODO: Why do we need two variables. One should suffice. Use the datalab processing location.
        # Also the caller alraedy has this logic. It is just duplication.
        loc = (
            processing_location
            or os.environ.get("DATALAB_PROCESSING_LOCATION")
            or os.environ.get("CHANDRA_PROCESSING_LOCATION")
        )

        data = {
            "output_format": "markdown",
            "mode": mode,
            "paginate": "true",
            "token_efficient_markdown": "true",
        }

        if loc:
            # TODO: This could be encapsulated in a private helper
            loc_clean = loc.strip().lower()
            data["processing_location"] = loc_clean

            # Step 1: Request presigned direct upload URL for regional processing
            upload_req_payload = {
                "filename": filename,
                "content_type": "application/pdf",
                "processing_location": loc_clean,
            }
            res_upload = requests.post(
                f"{cls.API_URL}/files/upload",
                json=upload_req_payload,
                headers=headers,
                timeout=30,
            )
            if not res_upload.ok:
                raise Exception(f"Datalab upload URL request failed ({res_upload.status_code}): {res_upload.text}")

            upload_info = res_upload.json()
            upload_url = upload_info.get("upload_url")
            file_id = upload_info.get("file_id")
            reference = upload_info.get("reference")

            if not upload_url or not file_id or not reference:
                raise Exception(f"Invalid upload URL response from Datalab: {upload_info}")

            # Step 2: Direct PUT to presigned upload URL
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()

            res_put = requests.put(
                upload_url,
                data=pdf_bytes,
                headers={"Content-Type": "application/pdf"},
                timeout=60,
            )
            if not res_put.ok:
                raise Exception(f"Datalab direct upload PUT failed ({res_put.status_code}): {res_put.text}")

            # Step 3: Confirm upload
            res_confirm = requests.get(
                f"{cls.API_URL}/files/{file_id}/confirm",
                headers=headers,
                timeout=30,
            )
            if not res_confirm.ok:
                raise Exception(f"Datalab upload confirmation failed ({res_confirm.status_code}): {res_confirm.text}")

            # Step 4: Submit convert job with reference
            data["file_url"] = reference
            response = requests.post(
                f"{cls.API_URL}/convert",
                data=data,
                headers=headers,
                timeout=60,
            )
        else:
            with open(file_path, "rb") as f:
                response = requests.post(
                    f"{cls.API_URL}/convert",
                    files={"file": (filename, f, "application/pdf")},
                    data=data,
                    headers=headers,
                    timeout=60,
                )

        if not response.ok:
            raise Exception(f"Datalab API request failed ({response.status_code}): {response.text}")

        return DatalabConvertResponse.model_validate(response.json())

    @classmethod
    def _poll_api_job(
        cls,
        check_url: str,
        headers: dict[str, str],
        poll_interval: float,
        max_polls: int,
    ) -> str:
        """Poll job status URL until completion or timeout.

        Args:
            check_url: Datalab status check endpoint URL.
            headers: HTTP headers with API key.
            poll_interval: Seconds between polls.
            max_polls: Maximum polling attempts.

        Returns:
            Full converted Markdown string.
        """
        for _ in range(max_polls):
            time.sleep(poll_interval)
            poll_res = requests.get(check_url, headers=headers, timeout=30)
            poll_res.raise_for_status()

            res_data = DatalabCheckResultResponse.model_validate(poll_res.json())

            if res_data.status == "failed":
                err_msg = res_data.error or "Unknown Datalab API error"
                raise Exception(f"Datalab OCR conversion failed: {err_msg}")

            if res_data.status == "complete":
                full_markdown = res_data.markdown or ""
                if not full_markdown and res_data.result_url:
                    download_res = requests.get(res_data.result_url, timeout=60)
                    download_res.raise_for_status()
                    full_markdown = download_res.text
                return full_markdown
        else:
            raise TimeoutError(f"Datalab API job timed out after {max_polls * poll_interval}s")

    @classmethod
    def _build_parsed_pages(cls, full_markdown: str) -> list[ParsedPage]:
        """Convert document Markdown into ParsedPage instances.

        Args:
            full_markdown: Document Markdown string.

        Returns:
            List of ParsedPage objects.
        """
        page_markdowns = _split_markdown_into_pages(full_markdown)
        parsed_pages: list[ParsedPage] = []

        for idx, page_md in enumerate(page_markdowns):
            page_num = idx + 1

            parsed_pages.append(
                ParsedPage(
                    page_number=page_num,
                    combined_content=clean_ocr_markdown(page_md),
                )
            )
        return parsed_pages

    @classmethod
    @override
    def parse_pdf(
        cls,
        file_path: str,
        force_parsing: bool = False,
        mode: DatalabMode = "balanced",
        processing_location: str | None = None,
        cache_dir: str = ".cache/ocr",
        poll_interval: float = 3.0,
        max_polls: int = 120,
        **kwargs: object,
    ) -> list[ParsedPage]:
        """Parse a PDF file using the remote Datalab Cloud API.

        Args:
            file_path: Path to PDF file.
            force_parsing: If True, bypasses local cache and forces API re-processing.
            mode: Datalab API mode ('balanced', 'fast', 'accurate'). Default 'balanced'.
            processing_location: Optional residency region override ('us', 'eu').
            cache_dir: Local directory for pre-chunking OCR cache files.
            poll_interval: Seconds to wait between polling job status.
            max_polls: Maximum number of status poll requests.

        Returns:
            List of :class:`ParsedPage` objects.

        Raises:
            ValueError: If DATALAB_API_KEY missing or invalid mode specified.
            Exception: If API conversion fails.
        """
        if mode not in VALID_DATALAB_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {sorted(VALID_DATALAB_MODES)}")

        cache_tag = f"chandra_api_{mode}"
        cache_path = _get_ocr_cache_path(file_path, cache_tag, cache_dir)

        if not force_parsing:
            cached_pages = _load_cached_pages(cache_path)
            if cached_pages:
                return cached_pages

        api_key = os.environ.get("DATALAB_API_KEY") or os.environ.get("CHANDRA_API_KEY")
        if not api_key:
            raise ValueError("Datalab API key missing. Set DATALAB_API_KEY environment variable.")

        headers = {"X-Api-Key": api_key}
        # TODO: Why do we need two variables. One should suffice. Use the datalab processing location
        loc = (
            processing_location
            or os.environ.get("DATALAB_PROCESSING_LOCATION")
            or os.environ.get("CHANDRA_PROCESSING_LOCATION")
        )

        print(
            f"ChandraAPIParser: submitting '{file_path}' to Datalab API ({mode} mode, location: {loc or 'default'}) ..."
        )
        convert_res = cls._submit_api_job(file_path, mode, headers, processing_location=loc)
        print(f"ChandraAPIParser: Job submitted. Polling {convert_res.request_check_url} ...")

        full_markdown = cls._poll_api_job(convert_res.request_check_url, headers, poll_interval, max_polls)

        if not full_markdown:
            raise Exception(f"Datalab API returned empty markdown for '{file_path}'")

        parsed_pages = cls._build_parsed_pages(full_markdown)

        print(f"ChandraAPIParser: completed API conversion ({len(parsed_pages)} pages extracted).")
        _save_cached_pages(cache_path, parsed_pages)
        return parsed_pages
