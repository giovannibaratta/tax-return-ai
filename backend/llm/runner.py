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
        # TODO: Document bfnrt
        valid_escape_pattern = r'\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})'

        def replace_invalid(match: re.Match[str]) -> str:
            # TODO: Can you explain what are we doing here (or bette rwhy)
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
