from __future__ import annotations

from collections import defaultdict
from collections.abc import MutableMapping, MutableSet

from tbps.expr import App, BVar, Const, Expr, FVar, ForallE, Lam, LetE, Lit, MData, MVar, Proj, Sort


def de_bruijn_to_binder_name(expr: Expr, binder_stack: tuple[str, ...] = ()) -> Expr:
    """Apply the binder-name conversion used by the published upstream CSE pass."""
    if isinstance(expr, BVar):
        try:
            return BVar(binder_stack[expr.de_bruijn_index])
        except (IndexError, TypeError) as error:
            raise ValueError(f"unbound de Bruijn index: {expr.de_bruijn_index}") from error
    if isinstance(expr, App):
        return App(
            de_bruijn_to_binder_name(expr.fn, binder_stack),
            de_bruijn_to_binder_name(expr.arg, binder_stack),
        )
    if isinstance(expr, (Lam, ForallE)):
        converted_type = de_bruijn_to_binder_name(expr.binder_type, binder_stack)
        converted_body = de_bruijn_to_binder_name(expr.body, (expr.binder_name, *binder_stack))
        cls = Lam if isinstance(expr, Lam) else ForallE
        return cls(expr.binder_name, converted_type, converted_body, expr.binder_info)
    if isinstance(expr, LetE):
        extended = (expr.decl_name, *binder_stack)
        return LetE(
            expr.decl_name,
            de_bruijn_to_binder_name(expr.type, binder_stack),
            # This intentionally mirrors upstream, which extends the stack for value too.
            de_bruijn_to_binder_name(expr.value, extended),
            de_bruijn_to_binder_name(expr.body, extended),
            expr.non_dep,
        )
    if isinstance(expr, MData):
        return MData(expr.data, de_bruijn_to_binder_name(expr.expr, binder_stack))
    if isinstance(expr, Proj):
        return Proj(expr.type_name, expr.idx, de_bruijn_to_binder_name(expr.struct, binder_stack))
    return expr


def expression_key(expr: Expr) -> str:
    """Return the structural string key used by upstream instead of Python's hash()."""
    if isinstance(expr, BVar):
        return f"BVar-{expr.de_bruijn_index}"
    if isinstance(expr, FVar):
        return f"FVar-{expr.fvar_id}"
    if isinstance(expr, MVar):
        return f"MVar-{expr.mvar_id}"
    if isinstance(expr, Sort):
        return f"Sort-{expr.universe}"
    if isinstance(expr, Const):
        return f"Const-{expr.decl_name}-" + ",".join(expr.universes)
    if isinstance(expr, App):
        return f"App-{expression_key(expr.fn)}-{expression_key(expr.arg)}"
    if isinstance(expr, Lam):
        return (
            f"Lam-{expr.binder_name}-{expression_key(expr.binder_type)}-{expression_key(expr.body)}"
        )
    if isinstance(expr, ForallE):
        return (
            f"ForallE-{expr.binder_name}-{expression_key(expr.binder_type)}-"
            f"{expression_key(expr.body)}"
        )
    if isinstance(expr, LetE):
        return (
            f"LetE-{expr.decl_name}-{expression_key(expr.type)}-"
            f"{expression_key(expr.value)}-{expression_key(expr.body)}"
        )
    if isinstance(expr, Lit):
        return f"Lit-{expr.literal}"
    if isinstance(expr, MData):
        return f"MData-{expr.data}-{expression_key(expr.expr)}"
    if isinstance(expr, Proj):
        return f"Proj-{expr.type_name}-{expr.idx}-{expression_key(expr.struct)}"
    raise TypeError(f"unsupported Expr: {type(expr).__name__}")


def _collect(expr: Expr, counts: MutableMapping[str, int]) -> None:
    counts[expression_key(expr)] += 1
    for child in _children(expr):
        _collect(child, counts)


def _fresh_variable(existing: MutableSet[str]) -> str:
    index = 0
    while f"v{index}" in existing:
        index += 1
    name = f"v{index}"
    existing.add(name)
    return name


def _replace(
    expr: Expr,
    counts: MutableMapping[str, int],
    variables: MutableMapping[str, str],
    existing: MutableSet[str],
) -> Expr:
    key = expression_key(expr)
    if counts[key] > 1 and not isinstance(expr, Const):
        if key not in variables:
            variables[key] = _fresh_variable(existing)
        return FVar(variables[key])
    if isinstance(expr, App):
        return App(
            _replace(expr.fn, counts, variables, existing),
            _replace(expr.arg, counts, variables, existing),
        )
    if isinstance(expr, (Lam, ForallE)):
        cls = Lam if isinstance(expr, Lam) else ForallE
        return cls(
            expr.binder_name,
            _replace(expr.binder_type, counts, variables, existing),
            _replace(expr.body, counts, variables, existing),
            expr.binder_info,
        )
    if isinstance(expr, LetE):
        return LetE(
            expr.decl_name,
            _replace(expr.type, counts, variables, existing),
            _replace(expr.value, counts, variables, existing),
            _replace(expr.body, counts, variables, existing),
            expr.non_dep,
        )
    if isinstance(expr, MData):
        return MData(expr.data, _replace(expr.expr, counts, variables, existing))
    if isinstance(expr, Proj):
        return Proj(expr.type_name, expr.idx, _replace(expr.struct, counts, variables, existing))
    return expr


def common_subexpression_elimination(expr: Expr, *, convert_binders: bool = True) -> Expr:
    """Replace every repeated non-constant subtree with a deterministic free variable."""
    if convert_binders:
        expr = de_bruijn_to_binder_name(expr)
    counts: defaultdict[str, int] = defaultdict(int)
    _collect(expr, counts)
    return _replace(expr, counts, {}, set())


def _children(expr: Expr) -> tuple[Expr, ...]:
    if isinstance(expr, App):
        return (expr.fn, expr.arg)
    if isinstance(expr, (Lam, ForallE)):
        return (expr.binder_type, expr.body)
    if isinstance(expr, LetE):
        return (expr.type, expr.value, expr.body)
    if isinstance(expr, MData):
        return (expr.expr,)
    if isinstance(expr, Proj):
        return (expr.struct,)
    return ()
