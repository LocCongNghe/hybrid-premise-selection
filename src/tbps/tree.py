from __future__ import annotations

import re
from dataclasses import dataclass

from zss.compare import AnnotatedTree

from tbps.expr import App, BVar, Const, Expr, FVar, ForallE, Lam, LetE, Lit, MData, MVar, Proj, Sort


@dataclass(frozen=True)
class TreeNode:
    label: str
    children: tuple[TreeNode, ...] = ()

    def get_children(self) -> tuple[TreeNode, ...]:
        return self.children


def simplify_forall(expr: Expr) -> Expr:
    """Repeat upstream's forall simplification until reaching a fixed point."""
    while True:
        simplified = _simplify_forall_once(expr)
        if simplified == expr:
            return expr
        expr = simplified


def _simplify_forall_once(expr: Expr) -> Expr:
    """Apply one recursive pass of upstream's forall simplification."""
    if isinstance(expr, ForallE):
        if isinstance(expr.binder_type, (BVar, FVar, MVar, Sort, Const)):
            return _simplify_forall_once(expr.body)
        return ForallE(
            expr.binder_name,
            _simplify_forall_once(expr.binder_type),
            _simplify_forall_once(expr.body),
            expr.binder_info,
        )
    if isinstance(expr, App):
        return App(_simplify_forall_once(expr.fn), _simplify_forall_once(expr.arg))
    if isinstance(expr, Lam):
        return Lam(
            expr.binder_name,
            _simplify_forall_once(expr.binder_type),
            _simplify_forall_once(expr.body),
            expr.binder_info,
        )
    if isinstance(expr, LetE):
        return LetE(
            expr.decl_name,
            _simplify_forall_once(expr.type),
            _simplify_forall_once(expr.value),
            _simplify_forall_once(expr.body),
            expr.non_dep,
        )
    if isinstance(expr, MData):
        return MData(expr.data, _simplify_forall_once(expr.expr))
    if isinstance(expr, Proj):
        return Proj(expr.type_name, expr.idx, _simplify_forall_once(expr.struct))
    return expr


def expr_to_tree(expr: Expr) -> TreeNode:
    name = type(expr).__name__
    if isinstance(expr, BVar):
        return TreeNode(f"{name}({expr.de_bruijn_index})")
    if isinstance(expr, FVar):
        return TreeNode(f"{name}({expr.fvar_id})")
    if isinstance(expr, MVar):
        return TreeNode(f"{name}({expr.mvar_id})")
    if isinstance(expr, Sort):
        return TreeNode(f"{name}({expr.universe})")
    if isinstance(expr, Const):
        return TreeNode(f"{name}({expr.decl_name}, {list(expr.universes)!r})")
    if isinstance(expr, App):
        return TreeNode(name, (expr_to_tree(expr.fn), expr_to_tree(expr.arg)))
    if isinstance(expr, (Lam, ForallE)):
        return TreeNode(name, (expr_to_tree(expr.binder_type), expr_to_tree(expr.body)))
    if isinstance(expr, LetE):
        return TreeNode(
            name,
            (expr_to_tree(expr.type), expr_to_tree(expr.value), expr_to_tree(expr.body)),
        )
    if isinstance(expr, Lit):
        return TreeNode(f"{name}({expr.literal})")
    if isinstance(expr, MData):
        return TreeNode(name, (expr_to_tree(expr.expr),))
    if isinstance(expr, Proj):
        return TreeNode(name, (expr_to_tree(expr.struct),))
    raise TypeError(f"unsupported Expr: {type(expr).__name__}")


def tree_node_count(tree: TreeNode) -> int:
    return 1 + sum(tree_node_count(child) for child in tree.children)


def tree_depth(tree: TreeNode) -> int:
    return 0 if not tree.children else 1 + max(tree_depth(child) for child in tree.children)


def _ted_build_side(
    root: TreeNode,
    prefixes: tuple[str, ...],
    base_cost: float,
    scale: float,
) -> tuple[AnnotatedTree, list, list[float], list[bool], list[object]]:
    """Precompute the per-node structures Zhang-Shasha needs for one tree.

    Returns (annotated, nodes, node_cost, is_simple, structural_sig) where:
      - node_cost[i] = base_cost * scale if node i is simple else base_cost
      - is_simple[i] = label starts with one of ``prefixes``
      - structural_sig[i] = (label, tuple(child sigs)) — two nodes are structurally
        equal (TreeNode ``==``) iff their sigs are equal, but compared in O(1)
        instead of the O(n) structural ``==`` the original cost function did per cell.
    Precomputing these out of the inner DP loop is the main speedup: the hot loop
    becomes array lookups + comparisons instead of lambda calls + ``str.startswith``
    + recursive ``__eq__`` per cell.
    """
    annotated = AnnotatedTree(root, TreeNode.get_children)
    nodes = annotated.nodes
    n = len(nodes)
    node_cost = [0.0] * n
    is_simple = [False] * n
    for i, node in enumerate(nodes):
        s = node.label.startswith(prefixes)
        is_simple[i] = s
        node_cost[i] = base_cost * scale if s else base_cost
    # Map each node to its children's post-order indices (AnnotatedTree.nodes is post-order,
    # so children always appear before their parent).
    node_to_idx = {id(node): i for i, node in enumerate(nodes)}
    child_map = [
        [node_to_idx[id(child)] for child in TreeNode.get_children(node)] for node in nodes
    ]
    sig: list[object] = [None] * n  # type: ignore[list-item]
    for i in range(n):
        node = nodes[i]
        sig[i] = (node.label, tuple(sig[c] for c in child_map[i]))
    return annotated, nodes, node_cost, is_simple, sig


def fast_ted_distance(
    left: TreeNode,
    right: TreeNode,
    *,
    simple_prefixes: tuple[str, ...] = ("BVar", "FVar", "MVar", "Sort", "Const"),
    simple_node_scale: float = 0.2,
    insert_cost: float = 1.0,
    delete_cost: float = 1.0,
    replace_cost: float = 0.4,
) -> float:
    """Zhang-Shasha tree-edit distance, optimized but exact vs ``zss.distance``.

    Three changes from ``zss.distance``, all preserving the exact distance:
      1. Drops zss's ``operations`` / ``partial_operations`` bookkeeping (only the scalar
         distance is ever consumed by ``normalized_teds``).
      2. Precomputes per-node insert/remove costs, simplicity flags, and structural
         signatures once per tree, so the DP inner loop uses array lookups instead of
         lambda calls + ``str.startswith`` + recursive ``TreeNode.__eq__`` per cell.
      3. Uses plain Python lists for the small (~40x40) forest-distance matrices instead
         of ``numpy.zeros`` (the per-call numpy overhead exceeds any vectorization gain).
    Verified byte-identical to ``zss.distance`` on 1500 candidates (~5-6x faster than the
    bookkeeping-free variant, ~13x faster than stock zss).
    """
    prefixes = tuple(f"{prefix}(" for prefix in simple_prefixes)
    a, an, rem_a, simp_a, sig_a = _ted_build_side(left, prefixes, delete_cost, simple_node_scale)
    b, bn, ins_b, simp_b, sig_b = _ted_build_side(right, prefixes, insert_cost, simple_node_scale)
    size_a, size_b = len(an), len(bn)
    treedists = [[0.0] * size_b for _ in range(size_a)]
    al, bl = a.lmds, b.lmds

    def treedist(i: int, j: int) -> None:
        m = i - al[i] + 2
        n = j - bl[j] + 2
        fd = [[0.0] * n for _ in range(m)]
        ioff = al[i] - 1
        joff = bl[j] - 1
        for x in range(1, m):
            fd[x][0] = fd[x - 1][0] + rem_a[x + ioff]
        for y in range(1, n):
            fd[0][y] = fd[0][y - 1] + ins_b[y + joff]
        for x in range(1, m):
            fx = fd[x]
            fxm1 = fd[x - 1]
            for y in range(1, n):
                ia = x + ioff
                jb = y + joff
                if al[i] == al[ia] and bl[j] == bl[jb]:
                    # update_cost(a,b) = 0 iff structurally-equal or either is simple
                    upd = (
                        0.0
                        if (sig_a[ia] == sig_b[jb] or simp_a[ia] or simp_b[jb])
                        else replace_cost
                    )
                    fx[y] = min(
                        fxm1[y] + rem_a[ia],
                        fx[y - 1] + ins_b[jb],
                        fxm1[y - 1] + upd,
                    )
                    treedists[ia][jb] = fx[y]
                else:
                    p = al[ia] - 1 - ioff
                    q = bl[jb] - 1 - joff
                    fx[y] = min(
                        fxm1[y] + rem_a[ia],
                        fx[y - 1] + ins_b[jb],
                        fd[p][q] + treedists[ia][jb],
                    )

    for i in a.keyroots:
        for j in b.keyroots:
            treedist(i, j)
    return float(treedists[-1][-1])


def normalized_teds(
    left: TreeNode,
    right: TreeNode,
    *,
    left_size: int | None = None,
    right_size: int | None = None,
    simple_prefixes: tuple[str, ...] = ("BVar", "FVar", "MVar", "Sort", "Const"),
    simple_node_scale: float = 0.2,
    insert_cost: float = 1.0,
    delete_cost: float = 1.0,
    replace_cost: float = 0.4,
) -> float:
    """Compute upstream's normalized, cost-sensitive tree-edit similarity."""
    distance = fast_ted_distance(
        left,
        right,
        simple_prefixes=simple_prefixes,
        simple_node_scale=simple_node_scale,
        insert_cost=insert_cost,
        delete_cost=delete_cost,
        replace_cost=replace_cost,
    )
    denominator = max(
        tree_node_count(left) if left_size is None else left_size,
        tree_node_count(right) if right_size is None else right_size,
    )
    return 1.0 - float(distance) / denominator if denominator else 0.0


def constant_names(tree: TreeNode) -> set[str]:
    """Extract constants with the same first-segment regex behavior as upstream."""
    result: set[str] = set()

    def visit(node: TreeNode) -> None:
        if node.label.startswith("Const("):
            match = re.match(r"Const\((\w+)", node.label)
            if match:
                name = match.group(1)
                if "inst" not in name:
                    result.add(name)
            else:
                start = node.label.find("(") + 1
                end = node.label.find(",", start)
                if start > 0 and end != -1:
                    name = node.label[start:end].strip()
                    if name:
                        result.add(name)
        for child in node.children:
            visit(child)

    visit(tree)
    return result


def constant_jaccard(left: TreeNode, right: TreeNode) -> float:
    left_names = constant_names(left)
    right_names = constant_names(right)
    if not left_names and not right_names:
        return 1.0
    union = left_names | right_names
    return len(left_names & right_names) / len(union) if union else 0.0


def collapse_match(target: TreeNode, candidate: TreeNode) -> float:
    """Score the directional soft-collapse match; candidate leaves are wildcards."""

    def score(left: TreeNode, right: TreeNode) -> float:
        if not right.children:
            return 1.0
        if left.label != right.label or len(left.children) != len(right.children):
            return 0.0
        return 1.0 + sum(score(a, b) for a, b in zip(left.children, right.children, strict=True))

    return score(target, candidate) / tree_node_count(candidate)
