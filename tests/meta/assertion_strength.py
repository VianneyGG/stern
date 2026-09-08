"""AST scanner flagging test functions with no concrete expected value.

Three findings only, chosen because sampling showed everything else (bound
comparisons, torch.allclose/equal, np.testing.assert_*) is legitimate:

- NO_ASSERT: no assertion of any kind.
- BARE_TRUTHY: assertions exist but none binds a concrete expected value
  (a lone name/attribute/call, or `is not None`).
- RAISES_NO_MATCH: `pytest.raises(...)` used without `match=`.
"""

from __future__ import annotations

import ast
import enum
from dataclasses import dataclass


class Kind(enum.Enum):
    NO_ASSERT = "no_assert"
    BARE_TRUTHY = "bare_truthy"
    RAISES_NO_MATCH = "raises_no_match"


@dataclass(frozen=True)
class Finding:
    kind: Kind
    name: str
    lineno: int


MOCK_ASSERT_METHODS = (
    ".assert_called_with",
    ".assert_called_once_with",
    ".assert_called",
    ".assert_called_once",
    ".assert_any_call",
    ".assert_not_called",
    ".assert_has_calls",
)


def _is_testing_helper_call(call: ast.Call) -> bool:
    fn = ast.unparse(call.func)
    return (
        "testing.assert" in fn
        or fn.endswith("assert_close")
        or fn.endswith((".allclose", ".equal", ".isclose", "isclose"))
        or fn.endswith(MOCK_ASSERT_METHODS)
    )


def _raises_calls(node: ast.AST) -> list[ast.Call]:
    return [
        c
        for c in ast.walk(node)
        if isinstance(c, ast.Call) and "raises" in ast.unparse(c.func)
    ]


def _is_bare_is_not_none(test_expr: ast.expr) -> bool:
    return (
        isinstance(test_expr, ast.Compare)
        and len(test_expr.ops) == 1
        and isinstance(test_expr.ops[0], ast.IsNot)
    )


def _is_isinstance_call(call: ast.Call) -> bool:
    return isinstance(call.func, ast.Name) and call.func.id == "isinstance"


def _is_reduction_over_concrete(call: ast.Call) -> bool:
    if not isinstance(call.func, ast.Attribute) or call.func.attr not in {"all", "any"}:
        return False
    # method form: (expr).all() -- the receiver holds the comparison.
    if not call.args:
        return _binds_concrete_value(call.func.value)
    # function form: torch.all(expr) / np.all(expr) -- the argument does.
    return len(call.args) == 1 and _binds_concrete_value(call.args[0])


def _is_builtin_quantifier_call(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Name)
        and call.func.id in {"all", "any"}
        and len(call.args) == 1
        and isinstance(call.args[0], (ast.GeneratorExp, ast.ListComp, ast.SetComp))
    )


def _is_bool_wrapped_concrete(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Name)
        and call.func.id == "bool"
        and len(call.args) == 1
        and _binds_concrete_value(call.args[0])
    )


def _binds_concrete_value(test_expr: ast.expr) -> bool:
    if isinstance(test_expr, ast.Compare):
        return not _is_bare_is_not_none(test_expr)
    if isinstance(test_expr, ast.Call):
        return (
            _is_testing_helper_call(test_expr)
            or _is_isinstance_call(test_expr)
            or _is_reduction_over_concrete(test_expr)
            or _is_bool_wrapped_concrete(test_expr)
            or _is_builtin_quantifier_call(test_expr)
        )
    if isinstance(test_expr, ast.BoolOp):
        return all(_binds_concrete_value(v) for v in test_expr.values)
    if isinstance(test_expr, ast.UnaryOp) and isinstance(test_expr.op, ast.Not):
        return _binds_concrete_value(test_expr.operand)
    return False


def _is_bare_truthy(test_expr: ast.expr) -> bool:
    return not _binds_concrete_value(test_expr)


def scan_function(func: ast.AST) -> list[Finding]:
    findings: list[Finding] = []
    name = getattr(func, "name", "<unknown>")
    lineno = getattr(func, "lineno", 0)

    asserts = [n for n in ast.walk(func) if isinstance(n, ast.Assert)]
    helper_calls = [
        c
        for c in ast.walk(func)
        if isinstance(c, ast.Call) and _is_testing_helper_call(c)
    ]
    raises_calls = _raises_calls(func)

    if not asserts and not helper_calls and not raises_calls:
        findings.append(Finding(Kind.NO_ASSERT, name, lineno))
        return findings

    has_concrete_evidence = (
        any(_binds_concrete_value(a.test) for a in asserts)
        or bool(helper_calls)
        or any(any(kw.arg == "match" for kw in c.keywords) for c in raises_calls)
    )
    if asserts and not has_concrete_evidence:
        first_bare = next((a for a in asserts if _is_bare_truthy(a.test)), asserts[0])
        findings.append(Finding(Kind.BARE_TRUTHY, name, first_bare.lineno))

    for call in raises_calls:
        if not any(kw.arg == "match" for kw in call.keywords):
            findings.append(Finding(Kind.RAISES_NO_MATCH, name, call.lineno))

    return findings


def scan_source(source: str) -> list[Finding]:
    tree = ast.parse(source)
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name.startswith("test_"):
            findings.extend(scan_function(node))
    return findings
