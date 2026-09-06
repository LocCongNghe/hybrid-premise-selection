from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class BVar:
    # CSE intentionally replaces indices with binder names, matching upstream's dynamic model.
    de_bruijn_index: int | str


@dataclass(frozen=True)
class FVar:
    fvar_id: str


@dataclass(frozen=True)
class MVar:
    mvar_id: str


@dataclass(frozen=True)
class Sort:
    universe: str


@dataclass(frozen=True)
class Const:
    decl_name: str
    universes: tuple[str, ...]


@dataclass(frozen=True)
class App:
    fn: Expr
    arg: Expr


@dataclass(frozen=True)
class Lam:
    binder_name: str
    binder_type: Expr
    body: Expr
    binder_info: str


@dataclass(frozen=True)
class ForallE:
    binder_name: str
    binder_type: Expr
    body: Expr
    binder_info: str


@dataclass(frozen=True)
class LetE:
    decl_name: str
    type: Expr
    value: Expr
    body: Expr
    non_dep: bool


@dataclass(frozen=True)
class Lit:
    literal: str


@dataclass(frozen=True)
class MData:
    data: object
    expr: Expr


@dataclass(frozen=True)
class Proj:
    type_name: str
    idx: int
    struct: Expr


Expr: TypeAlias = (
    BVar | FVar | MVar | Sort | Const | App | Lam | ForallE | LetE | Lit | MData | Proj
)


def deserialize_expr(data: object) -> Expr:
    """Deserialize the JSON representation emitted by the upstream Lean extractor."""
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError("an Expr must be an object with exactly one constructor")

    expr_type, value = next(iter(data.items()))
    if expr_type == "bvar":
        fields = _mapping(value, expr_type)
        index = fields.get("deBruijnIndex")
        if not isinstance(index, (int, str)) or isinstance(index, bool):
            raise ValueError("deBruijnIndex must be an integer or binder name")
        return BVar(de_bruijn_index=index)
    if expr_type == "fvar":
        if isinstance(value, dict):
            return FVar(fvar_id=_string(value, "fvarId"))
        return FVar(fvar_id=_string_value(value, expr_type))
    if expr_type == "mvar":
        if isinstance(value, dict):
            return MVar(mvar_id=_string(value, "mvarId"))
        return MVar(mvar_id=_string_value(value, expr_type))
    if expr_type == "sort":
        if isinstance(value, dict):
            return Sort(universe=_string(value, "u"))
        return Sort(universe=_string_value(value, expr_type))
    if expr_type == "const":
        fields = _mapping(value, expr_type)
        universes = fields.get("us")
        if not isinstance(universes, list) or not all(isinstance(item, str) for item in universes):
            raise ValueError("const.us must be a list of strings")
        return Const(decl_name=_string(fields, "declName"), universes=tuple(universes))
    if expr_type == "app":
        fields = _mapping(value, expr_type)
        return App(fn=deserialize_expr(fields.get("fn")), arg=deserialize_expr(fields.get("arg")))
    if expr_type in {"lam", "forallE"}:
        fields = _mapping(value, expr_type)
        common = {
            "binder_name": _string(fields, "binderName"),
            "binder_type": deserialize_expr(fields.get("binderType")),
            "body": deserialize_expr(fields.get("body")),
            "binder_info": _string(fields, "binderInfo"),
        }
        return Lam(**common) if expr_type == "lam" else ForallE(**common)
    if expr_type == "letE":
        fields = _mapping(value, expr_type)
        non_dep = fields.get("nonDep")
        if not isinstance(non_dep, bool):
            raise ValueError("letE.nonDep must be a boolean")
        return LetE(
            decl_name=_string(fields, "declName"),
            type=deserialize_expr(fields.get("type")),
            value=deserialize_expr(fields.get("value")),
            body=deserialize_expr(fields.get("body")),
            non_dep=non_dep,
        )
    if expr_type == "lit":
        return Lit(literal=_string(_mapping(value, expr_type), "literal"))
    if expr_type == "mdata":
        fields = _mapping(value, expr_type)
        if "data" not in fields:
            raise ValueError("mdata.data is missing")
        return MData(data=fields["data"], expr=deserialize_expr(fields.get("expr")))
    if expr_type == "proj":
        fields = _mapping(value, expr_type)
        return Proj(
            type_name=_string(fields, "typeName"),
            idx=_integer(fields, "idx"),
            struct=deserialize_expr(fields.get("struct")),
        )
    raise ValueError(f"unknown Expr constructor: {expr_type}")


def count_nodes(expr: Expr) -> int:
    if isinstance(expr, App):
        return 1 + count_nodes(expr.fn) + count_nodes(expr.arg)
    if isinstance(expr, (Lam, ForallE)):
        return 1 + count_nodes(expr.binder_type) + count_nodes(expr.body)
    if isinstance(expr, LetE):
        return 1 + count_nodes(expr.type) + count_nodes(expr.value) + count_nodes(expr.body)
    if isinstance(expr, MData):
        return 1 + count_nodes(expr.expr)
    if isinstance(expr, Proj):
        return 1 + count_nodes(expr.struct)
    return 1


def _mapping(value: object, constructor: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{constructor} payload must be an object")
    return value


def _string(fields: dict[str, object], field: str) -> str:
    return _string_value(fields.get(field), field)


def _string_value(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _integer(fields: dict[str, object], field: str) -> int:
    value = fields.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value
