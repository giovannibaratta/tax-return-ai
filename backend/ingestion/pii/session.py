class PIISession:
    """
    State container for an individual document's PII anonymization session.

    Encapsulates placeholder mappings and counters to prevent data leaks across documents.
    """

    def __init__(self) -> None:
        # Bidirectional mapping: placeholder -> original_value
        self.placeholder_map: dict[str, str] = {}
        # Key: entity type with source prefix, Value: counter integer
        self.counter: dict[str, int] = {}

    def make_placeholder(self, entity_type: str, original_val: str, source: str) -> str:
        """
        Return existing placeholder for original_val if already mapped,
        otherwise allocate a new numbered placeholder and store the mapping.
        """
        source_prefix = source.upper()
        prefix = f"[ANONYMIZED_{source_prefix}_{entity_type}_"
        for pl, orig in self.placeholder_map.items():
            if orig == original_val and pl.startswith(prefix):
                return pl

        counter_key = f"{source_prefix}_{entity_type}"
        self.counter[counter_key] = self.counter.get(counter_key, 0) + 1
        placeholder = f"{prefix}{self.counter[counter_key]}]"
        self.placeholder_map[placeholder] = original_val
        return placeholder

    def reset(self) -> None:
        """Clear the placeholder map and counters to isolate the session."""
        self.placeholder_map.clear()
        self.counter.clear()
