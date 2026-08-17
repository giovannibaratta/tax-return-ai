import re
from abc import ABC, abstractmethod


class BaseLLMRunner(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        """The name of the LLM model used by this runner."""
        pass

    @abstractmethod
    def complete(self, prompt: str, system_instruction: str) -> str:
        """Execute a completions request using the configured model/environment."""
        pass

    @staticmethod
    def sanitize_json_string(s: str) -> str:
        r"""Fix invalid backslash escape sequences in LLM-generated JSON text.

        Standard JSON allows: \", \\, \/, \b, \f, \n, \r, \t, and \uXXXX.
        Any other backslash followed by a character is invalid in JSON and causes JSONDecodeError.
        This replaces invalid single backslashes with double backslashes \\.
        """
        # RFC 8259 JSON specification permits exactly 8 single-character escape sequences:
        # \" (quotation mark), \\ (reverse solidus), \/ (solidus), \b (backspace),
        # \f (formfeed), \n (linefeed), \r (carriage return), \t (horizontal tab),
        # as well as \uXXXX (4-hex-digit unicode codepoints).
        valid_escape_pattern = r'\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})'

        def replace_invalid(match: re.Match[str]) -> str:
            # LLMs frequently output raw backslashes (e.g. in regex patterns like \d,
            # file paths like C:\temp, or LaTeX symbols) that are invalid in JSON string literals.
            # If the backslash escape is valid JSON, keep it unchanged. Otherwise, prefix
            # with an extra backslash to escape it into a valid JSON string literal ("\\").
            text = match.group(0)
            if re.match(valid_escape_pattern, text):
                return text
            return "\\\\" + text[1:]

        return re.sub(r"\\.", replace_invalid, s, flags=re.DOTALL)

    @staticmethod
    def clean_json_response(raw_response: str) -> str:
        """Clean markdown code block fences, reasoning/think blocks, and invalid escapes from a JSON response.

        Args:
            raw_response: The raw response string from the LLM.

        Returns:
            The cleaned JSON string.
        """
        cleaned = raw_response.strip()

        # Remove reasoning / think block if using DeepSeek-R1 Distill models
        if "<think>" in cleaned:
            cleaned = re.sub(r"(?s)<think>.*?</think>", "", cleaned).strip()

        # Try stripping markdown wrappers
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        # Sanitize invalid backslash escape sequences
        cleaned = BaseLLMRunner.sanitize_json_string(cleaned)

        return cleaned
