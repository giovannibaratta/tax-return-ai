import logging
from dataclasses import dataclass

import stanza
import torch
from stanza import DownloadMethod
from stanza import Pipeline as StanzaPipeline
from stanza.models.common.doc import Document as StanzaDocument
from transformers import (
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from backend.utils.utils import auto_detect_device, pool_and_normalize

logger = logging.getLogger(__name__)

# Silence Hugging Face Hub unauthenticated network check warnings
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Tunable chunking constants — adjust these to change retrieval quality
# ---------------------------------------------------------------------------

# Child chunk: small, precise unit used for embedding and vector retrieval.
# Increasing this reduces retrieval precision but gives the LLM more text
# per chunk when parent_text_content is absent.
CHILD_MAX_CHARS: int = 1400

# Parent chunk: large context window returned to the LLM after retrieval.
# Each parent groups multiple consecutive child chunks. The LLM reads
# parent_text_content, not child text_content, for richer context.
PARENT_MAX_CHARS: int = 3000

# Transformer context window fed to BGE-M3 for a single forward pass.
# BGE-M3 supports up to 8192 tokens (~24 000 chars). Keep comfortably below
# that limit to avoid silent truncation on dense legal text.
LATE_CHUNKING_WINDOW_MAX_CHARS: int = 8000

# Maximum lookahead (in characters) when resolving a whitespace char index
# to the nearest token boundary in the tokenizer encoding.
TOKEN_BOUNDARY_SEARCH_RADIUS: int = 30

# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SentenceSpan:
    """Character offsets and text content of a single sentence detected by Stanza.

    Attributes:
        start_char_idx: Start character index in original document text.
        end_char_idx: End character index in original document text.
        text: Sentence text content.
    """

    start_char_idx: int
    end_char_idx: int
    text: str


# Domain type aliases to replace deeply-nested tuple types:
# ChildGroup: Consecutive sentence spans grouped into a child retrieval unit (~600 chars)
ChildGroup = list[SentenceSpan]

# TransformerWindowRun: Consecutive child groups batched into a transformer forward pass window (~8000 chars)
TransformerWindowRun = list[ChildGroup]


@dataclass(frozen=True)
class ChunkWithEmbedding:
    """A child chunk paired with its late-chunking embedding and parent context.

    Attributes:
        chunk_index: Zero-based position of this child chunk in the document.
        start_char_idx: Character offset of the first character in the full
            concatenated document text.
        text_content: The child chunk text — small, precise, used for display
            and as the retrieval unit in the vector database.
        parent_text: The parent chunk text — larger window (~PARENT_MAX_CHARS)
            that groups several child chunks. Stored alongside the child row
            so the LLM receives richer context without an extra DB round-trip.
        embedding: Normalized 1024-dimensional BGE-M3 embedding computed via
            late chunking (token embeddings pooled over the child's token span
            inside the full window context).
    """

    chunk_index: int
    start_char_idx: int
    text_content: str
    parent_text: str
    embedding: list[float]


class LateChunker:
    """Late-chunking encoder with Stanza sentence boundary detection.

    Produces parent-child chunk pairs:
    - **Child chunks** (~CHILD_MAX_CHARS): embedded with BGE-M3 late chunking
      and stored in the vector index for precise semantic retrieval.
    - **Parent chunks** (~PARENT_MAX_CHARS): stored denormalized on each child
      row so the LLM always receives a full context window after retrieval.

    Sentence boundary detection uses Stanza's neural tokenizer, which handles
    Italian legal abbreviations (``art.``, ``n.``, ``comma``) correctly.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name: str = model_name
        self._device: str = auto_detect_device()

        logger.info(f"Initializing LateChunker with model '{self.model_name}' on device '{self._device}'")
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self.model_name)
        self.model: PreTrainedModel = (
            AutoModel.from_pretrained(self.model_name).to(self._device).eval()  # type: ignore[assignment]
        )

        # Stanza pipelines are loaded lazily and cached by language code.
        # Loading is slow (~1-2 s) but only happens once per language per process.
        self._stanza_pipelines: dict[str, StanzaPipeline] = {}

    # ------------------------------------------------------------------
    # Stanza sentence boundary detection
    # ------------------------------------------------------------------

    def _get_stanza_pipeline(self, language: str) -> StanzaPipeline:
        """Return a cached Stanza pipeline for the given language code.

        Loads pre-downloaded Stanza tokenizer models using DownloadMethod.REUSE_RESOURCES.

        Args:
            language: ISO 639-1 language code (e.g. ``"it"``, ``"en"``).

        Returns:
            A Stanza Pipeline instance configured for tokenisation only.
        """
        if language not in self._stanza_pipelines:
            logger.info(f"Loading Stanza pipeline for language='{language}'")
            self._stanza_pipelines[language] = stanza.Pipeline(
                lang=language,
                processors="tokenize",
                verbose=False,
                use_gpu=self._device != "cpu",
                download_method=DownloadMethod.REUSE_RESOURCES,
            )
        return self._stanza_pipelines[language]

    def _split_into_sentences(self, text: str, language: str = "en") -> list[SentenceSpan]:
        """Split text into sentences with character start/end offsets.

        Uses Stanza's neural tokenizer for language-aware sentence boundary detection.
        Performs a runtime type-guard check on Stanza output to ensure valid Document object.

        Args:
            text: Raw text to segment.
            language: ISO 639-1 language code passed to Stanza.

        Returns:
            List of SentenceSpan objects containing (start_char_idx, end_char_idx, text).

        Raises:
            TypeError: If Stanza pipeline does not return a Stanza Document instance.
        """
        nlp = self._get_stanza_pipeline(language)
        doc = nlp(text)

        # Type guard: validate Stanza returned a valid Document object
        if not isinstance(doc, StanzaDocument):
            raise TypeError(f"Expected Stanza Document from pipeline, got {type(doc).__name__}")

        sentences: list[SentenceSpan] = []
        for sent in doc.sentences:
            start = sent.tokens[0].start_char
            end = sent.tokens[-1].end_char
            sentence_text = text[start:end].strip()
            if sentence_text:
                sentences.append(
                    SentenceSpan(
                        start_char_idx=start,
                        end_char_idx=end,
                        text=sentence_text,
                    )
                )

        return sentences

    # ------------------------------------------------------------------
    # Child and parent grouping
    # ------------------------------------------------------------------

    def _group_sentences_into_children(
        self,
        sentences: list[SentenceSpan],
        max_chars: int = CHILD_MAX_CHARS,
    ) -> list[ChildGroup]:
        """Group consecutive sentences into child chunks of at most *max_chars*.

        Each child chunk is a ChildGroup (list of SentenceSpan objects), which is the
        unit embedded by BGE-M3 and stored in the vector index.

        Args:
            sentences: Sentence spans from ``_split_into_sentences``.
            max_chars: Maximum character count per child chunk (default: CHILD_MAX_CHARS = 600).

        Returns:
            A list of ChildGroup objects, each containing consecutive sentence spans.
        """
        groups: list[ChildGroup] = []  # List of completed child chunk groups
        # Accumulator for sentence spans in the current child group
        current: ChildGroup = []
        current_len = 0  # Total character count in current child group

        for span in sentences:
            sent_len = len(span.text)
            if current_len + sent_len > max_chars and current:
                groups.append(current)
                current = [span]
                current_len = sent_len
            else:
                current.append(span)
                current_len += sent_len

        if current:
            groups.append(current)

        return groups

    def _group_children_into_parents(
        self,
        child_groups: list[ChildGroup],
        full_text: str,
        max_chars: int = PARENT_MAX_CHARS,
    ) -> list[str]:
        """Build a parent text string for each child group.

        Multiple consecutive child groups are merged into a parent window up to *max_chars*
        (default: PARENT_MAX_CHARS = 3000).

        Validation & Ratio Safety:
            - `PARENT_MAX_CHARS` (3000) must be >= `CHILD_MAX_CHARS` (600). If `CHILD_MAX_CHARS`
              exceeded `PARENT_MAX_CHARS`, a single child could not fit inside any parent window.
            - Even if a single child exceeds `PARENT_MAX_CHARS` (e.g. huge single sentence),
              the child is assigned its own oversized parent window without throwing an error.

        Args:
            child_groups: List of ChildGroup objects from ``_group_sentences_into_children``.
            full_text: The original full document text (used for exact character slicing).
            max_chars: Maximum character count per parent chunk.

        Returns:
            A list of parent text strings, aligned 1:1 with *child_groups*.
        """
        if not child_groups:
            return []

        if max_chars < CHILD_MAX_CHARS:
            logger.warning(
                "PARENT_MAX_CHARS (%d) is less than CHILD_MAX_CHARS (%d). "
                "Parent windows will frequently consist of single child chunks.",
                max_chars,
                CHILD_MAX_CHARS,
            )

        parent_texts: list[str] = []
        n_children = len(child_groups)

        for i in range(n_children):
            left = i
            right = i
            current_start = child_groups[left][0].start_char_idx
            current_end = child_groups[right][-1].end_char_idx

            while True:
                expanded = False
                if left > 0:
                    cand_start = child_groups[left - 1][0].start_char_idx
                    if (current_end - cand_start) <= max_chars:
                        left -= 1
                        current_start = cand_start
                        expanded = True

                if right < n_children - 1:
                    cand_end = child_groups[right + 1][-1].end_char_idx
                    if (cand_end - current_start) <= max_chars:
                        right += 1
                        current_end = cand_end
                        expanded = True

                if not expanded:
                    break

            parent_text = full_text[current_start:current_end].strip()
            parent_texts.append(parent_text)

        return parent_texts

    # ------------------------------------------------------------------
    # Token index mapping
    # ------------------------------------------------------------------

    def _get_token_index(
        self,
        encoding: BatchEncoding,
        char_idx: int,
        full_text: str,
        direction: str = "forward",
    ) -> int:
        """Map a character index to the nearest token index.

        Stanza sentence offsets may land on whitespace or special characters
        that the tokenizer maps to ``None``. This method scans a local
        neighbourhood of size TOKEN_BOUNDARY_SEARCH_RADIUS to find the
        nearest valid token.

        Args:
            encoding: HuggingFace tokenizer BatchEncoding object with
                ``char_to_token`` support.
            char_idx: Target character index within ``full_text``.
            full_text: The text that was tokenized (used for bounds checking).
            direction: ``"forward"`` scans right; ``"backward"`` scans left.

        Returns:
            The resolved token index, or 0 if no valid token is found within
            the search radius.
        """
        num_chars = len(full_text)
        char_idx = max(0, min(char_idx, num_chars - 1))

        token_idx = encoding.char_to_token(char_idx)
        if token_idx is not None:
            return token_idx

        if direction == "forward":
            for offset in range(1, TOKEN_BOUNDARY_SEARCH_RADIUS):
                if char_idx + offset < num_chars:
                    t_idx = encoding.char_to_token(char_idx + offset)
                    if t_idx is not None:
                        return t_idx
        else:
            for offset in range(1, TOKEN_BOUNDARY_SEARCH_RADIUS):
                if char_idx - offset >= 0:
                    t_idx = encoding.char_to_token(char_idx - offset)
                    if t_idx is not None:
                        return t_idx

        return 0

    def _embed_window_run(
        self,
        window_run: TransformerWindowRun,
        full_text: str,
        parent_texts: list[str],
        start_chunk_idx: int,
        start_flat_child_idx: int,
    ) -> list[ChunkWithEmbedding]:
        """Perform forward pass and late-chunking mean-pooling for a single transformer window.

        Args:
            window_run: Child groups batched into this forward-pass window.
            full_text: Original full document text.
            parent_texts: Parallel list of parent text strings.
            start_chunk_idx: Starting zero-based chunk index.
            start_flat_child_idx: Starting index into parent_texts array.

        Returns:
            A list of ChunkWithEmbedding objects generated for this window run.
        """
        window_sentences: list[SentenceSpan] = []
        for child in window_run:
            window_sentences.extend(child)

        if not window_sentences:
            return []

        window_start_offset = window_sentences[0].start_char_idx
        window_text = full_text[window_start_offset : window_sentences[-1].end_char_idx]

        inputs = self.tokenizer(window_text, return_tensors="pt", padding=True, truncation=True)
        # Transfer tokenized input tensors (input_ids, attention_mask) from CPU to model device (GPU/MPS)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Shape: [num_tokens, hidden_dim]
            token_embeddings = outputs.last_hidden_state[0]

        encoding: BatchEncoding = self.tokenizer(window_text)  # type: ignore[assignment]
        chunks: list[ChunkWithEmbedding] = []

        chunk_idx = start_chunk_idx
        flat_child_idx = start_flat_child_idx

        for child in window_run:
            g_start_char = child[0].start_char_idx - window_start_offset
            g_end_char = child[-1].end_char_idx - window_start_offset

            start_token_idx = self._get_token_index(encoding, g_start_char, window_text, "forward")
            end_token_idx = self._get_token_index(encoding, g_end_char, window_text, "backward")

            end_token_idx = max(end_token_idx, start_token_idx)

            chunk_tokens = token_embeddings[start_token_idx : end_token_idx + 1]
            embedding_list: list[float] = pool_and_normalize(chunk_tokens, dim=0)
            child_text = " ".join(s.text for s in child)

            chunks.append(
                ChunkWithEmbedding(
                    chunk_index=chunk_idx,
                    start_char_idx=child[0].start_char_idx,
                    text_content=child_text,
                    parent_text=parent_texts[flat_child_idx],
                    embedding=embedding_list,
                )
            )
            chunk_idx += 1
            flat_child_idx += 1

        return chunks

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute_late_chunks(
        self,
        full_text: str,
        language: str = "en",
        max_chars_per_window: int = LATE_CHUNKING_WINDOW_MAX_CHARS,
    ) -> list[ChunkWithEmbedding]:
        """Perform Late Chunking on the input text with parent-child grouping.

        Pipeline:
        1. Stanza sentence boundary detection (language-aware).
        2. Child grouping: small sentence groups (CHILD_MAX_CHARS chars)
           used as the retrieval unit and embedding target.
        3. Parent grouping: each child is assigned a larger parent text
           (PARENT_MAX_CHARS chars) for LLM context.
        4. BGE-M3 forward pass over transformer windows
           (<=LATE_CHUNKING_WINDOW_MAX_CHARS chars) to produce contextual
           token-level embeddings.
        5. Mean pooling + L2 normalisation per child group.

        If the text exceeds *max_chars_per_window*, it is split into windows
        processed independently to stay within BGE-M3's 8192-token limit.

        Args:
            full_text: Raw concatenated document text.
            language: ISO 639-1 language code for Stanza SBD (``"it"`` or
                ``"en"``).
            max_chars_per_window: Maximum characters per BGE-M3 forward-pass
                window. Default: LATE_CHUNKING_WINDOW_MAX_CHARS.

        Returns:
            A list of :class:`ChunkWithEmbedding` objects, one per child group.
        """
        # 1. Sentence boundary detection
        sentences = self._split_into_sentences(full_text, language=language)
        if not sentences:
            return []

        # 2. Build child groups (small, for retrieval precision)
        child_groups = self._group_sentences_into_children(sentences, max_chars=CHILD_MAX_CHARS)

        # 3. Build parent text for each child group (large, for LLM context)
        parent_texts = self._group_children_into_parents(child_groups, full_text, max_chars=PARENT_MAX_CHARS)

        # 4. Batch child groups into BGE-M3 forward-pass windows
        all_window_runs: list[TransformerWindowRun] = []
        current_window_groups: TransformerWindowRun = []
        current_window_chars = 0

        for child in child_groups:
            child_len = sum(len(s.text) for s in child)
            if current_window_chars + child_len > max_chars_per_window and current_window_groups:
                all_window_runs.append(current_window_groups)
                current_window_groups = [child]
                current_window_chars = child_len
            else:
                current_window_groups.append(child)
                current_window_chars += child_len

        if current_window_groups:
            all_window_runs.append(current_window_groups)

        # 5. Late chunking: forward pass + mean pooling per child group
        chunks_with_embeddings: list[ChunkWithEmbedding] = []
        flat_child_idx = 0

        for window_run in all_window_runs:
            window_chunks = self._embed_window_run(
                window_run=window_run,
                full_text=full_text,
                parent_texts=parent_texts,
                start_chunk_idx=len(chunks_with_embeddings),
                start_flat_child_idx=flat_child_idx,
            )
            chunks_with_embeddings.extend(window_chunks)
            flat_child_idx += len(window_run)

        return chunks_with_embeddings

    def embed_query(self, query_text: str) -> list[float]:
        """Generate a single vector embedding for a short query string.

        Args:
            query_text: Natural language query string.

        Returns:
            Normalized 1024-dimensional BGE-M3 float vector.
        """
        inputs = self.tokenizer(query_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state[0]
            query_vector = torch.mean(token_embeddings, dim=0)
            query_vector = torch.nn.functional.normalize(query_vector, p=2, dim=0)

        return query_vector.cpu().tolist()
