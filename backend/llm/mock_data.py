import json


def get_mock_voter_json(prompt: str, system_instruction: str) -> str:
    """Get static transaction extraction items JSON based on voter index and prompt broker.

    Args:
        prompt: The transaction extraction prompt.
        system_instruction: The system instructions.

    Returns:
        The JSON string representing a list of transaction extraction dicts.
    """
    # Detect voter index
    voter_index = 1
    combined_search = (system_instruction + " " + prompt).lower()
    if "#2" in combined_search:
        voter_index = 2
    elif "#3" in combined_search:
        voter_index = 3

    is_directa = "directa" in prompt.lower()
    is_ibkr = "ibkr" in prompt.lower() or "interactive" in prompt.lower()

    if is_directa:
        return json.dumps(
            [
                {
                    "event_date": "2024-07-04T10:30:00",
                    "asset_type": "stock",
                    "symbol": "ENI",
                    "isin": "IE00BFWXDV39",
                    "action": "buy",
                    "quantity": 100.0,
                    "unit_price": 14.50,
                    "currency": "EUR",
                    "fees": 5.0,
                    "total_amount": 1455.0,
                    "fx_rate": 1.0,
                    "provider": "directa",
                }
            ]
        )
    elif is_ibkr:
        qty = 50.5 if voter_index == 3 else 50.0
        return json.dumps(
            [
                {
                    "event_date": "2025-06-15T15:45:00",
                    "asset_type": "etf",
                    "symbol": "VUAA",
                    "isin": "IE00BYX5MX67",
                    "action": "buy",
                    "quantity": qty,
                    "unit_price": 82.50,
                    "currency": "EUR",
                    "fees": 2.0,
                    "total_amount": 4127.0,
                    "fx_rate": 1.0,
                    "provider": "interactive_brokers",
                }
            ]
        )
    else:
        return json.dumps(
            [
                {
                    "event_date": "2025-01-01T12:00:00",
                    "asset_type": "stock",
                    "symbol": "DUMMY",
                    "isin": "US0000000000",
                    "action": "buy",
                    "quantity": 10.0,
                    "unit_price": 10.0,
                    "currency": "EUR",
                    "fees": 0.0,
                    "total_amount": 100.0,
                    "fx_rate": 1.0,
                    "provider": "mock_provider",
                }
            ]
        )


def get_mock_pii_json() -> str:
    """Simulate PII anonymization by redacting person and location details.

    Args:
        prompt: The raw input text containing PII.
        system_instruction: The system instructions.

    Returns:
        The JSON string containing redacted text and replacements map.
    """
    return json.dumps(
        {
            "redacted_text": "This is a mock",
            "replacements": {
                "[ANONYMIZED_LLM_PERSON_1]": "MARIO ROSSI",
                "[ANONYMIZED_LLM_LOCATION_1]": "VIA SEGRETA ITALIANA 10",
            },
        }
    )
