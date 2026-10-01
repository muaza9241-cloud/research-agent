"""Safe mathematical expression evaluator for the research agent."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from langchain_core.tools import tool

_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_FUNCS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}

_MAX_POW = 1_000_000
_MAX_ABS = 1e15


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(left) > 1e6 or abs(right) > 12:
                raise ValueError("Exponentiation is too large to evaluate safely.")
        result = _BIN_OPS[type(node.op)](left, right)
        if isinstance(result, (int, float)) and abs(result) > _MAX_ABS:
            raise ValueError("Result is too large to evaluate safely.")
        if isinstance(result, (int, float)) and abs(result) > _MAX_POW and isinstance(
            node.op, ast.Pow
        ):
            raise ValueError("Power result exceeds the safety limit.")
        return result

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name not in _ALLOWED_FUNCS or not callable(_ALLOWED_FUNCS[name]):
            raise ValueError(f"Function '{name}' is not allowed.")
        args = [_eval_node(arg) for arg in node.args]
        return _ALLOWED_FUNCS[name](*args)

    if isinstance(node, ast.Name):
        value = _ALLOWED_FUNCS.get(node.id)
        if value is None or callable(value):
            raise ValueError(f"Name '{node.id}' is not allowed.")
        return value

    if isinstance(node, ast.List):
        return [_eval_node(elt) for elt in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(elt) for elt in node.elts)

    raise ValueError("Unsupported or unsafe expression.")


def evaluate_expression(expression: str) -> str:
    """Evaluate a math expression using a restricted AST walker (no eval/exec)."""
    cleaned = expression.strip().replace("^", "**")
    if not cleaned:
        return "Error: empty expression."
    try:
        tree = ast.parse(cleaned, mode="eval")
        result = _eval_node(tree)
        return str(result)
    except Exception as exc:  # noqa: BLE001 — surface a clean tool error to the agent
        return f"Error evaluating expression: {exc}"


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression.

    Use this for arithmetic, budgets, unit conversions that are numeric,
    percentages, and trip cost totals. Supports +, -, *, /, //, %, **,
    parentheses, and functions: abs, round, min, max, sum, pow, sqrt,
    ceil, floor, log, log10, exp, sin, cos, tan, plus constants pi and e.
    Example: '(5000 * 2) + (1500 * 3)' or 'sqrt(144)'.
    """
    return evaluate_expression(expression)
