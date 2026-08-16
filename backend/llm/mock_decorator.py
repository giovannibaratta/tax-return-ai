from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast
from unittest.mock import patch

from backend.llm.runners import MockRunner

F = TypeVar("F", bound=Callable[..., Any])
MockReturnValue = str | Callable[[str, str], str]


def mock_llm_runner(return_value: MockReturnValue) -> Callable[[F], F]:
    """Decorator to temporarily monkey patch MockRunner.complete for the duration of the function.

    Args:
        return_value: A static string or a callable (prompt, system_instruction) -> str.

    Returns:
        The decorator function.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            def mock_complete(self: MockRunner, prompt: str, system_instruction: str) -> str:
                if callable(return_value):
                    return return_value(prompt, system_instruction)
                return return_value

            with patch.object(MockRunner, "complete", mock_complete):
                return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator
