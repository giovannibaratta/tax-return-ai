import ast
import operator
from decimal import Decimal
from typing import Any

_SAFE_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,  # pyright: ignore[reportUnknownMemberType]
    ast.Pow: operator.pow,  # pyright: ignore[reportUnknownMemberType]
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.expr) -> Decimal:
    """Recursively evaluate a parsed AST expression using only safe arithmetic operators.

    Args:
        node: An AST expression node.

    Returns:
        The numeric result as a Decimal.

    Raises:
        ValueError: If the expression contains non-arithmetic constructs.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, str, Decimal)):
            return Decimal(str(node.value))
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return Decimal(str(_SAFE_OPERATORS[op_type](left, right)))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return Decimal(str(_SAFE_OPERATORS[op_type](_safe_eval(node.operand))))

    raise ValueError(f"Unsupported AST node type: {type(node).__name__}")


def evaluate_expression(expression: str) -> Decimal:
    """Parse and safely evaluate an arithmetic expression string.

    Only numeric literals and the operators ``+``, ``-``, ``*``, ``/``, ``**``
    are permitted. No function calls, names, or attribute access are allowed.

    Args:
        expression: An arithmetic expression string, e.g. ``"3500 * 0.26"``.

    Returns:
        The computed Decimal result.

    Raises:
        ValueError: If the expression is syntactically invalid or contains
            disallowed constructs.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}") from e

    return _safe_eval(tree.body)
