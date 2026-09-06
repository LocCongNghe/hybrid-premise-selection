from __future__ import annotations

import hashlib
import math
from collections import Counter

from tbps.tree import TreeNode, tree_depth


def initialize_labels(
    tree: TreeNode,
    simple_prefixes: tuple[str, ...] = ("BVar", "FVar", "MVar", "Sort", "Const"),
) -> dict[int, str]:
    labels: dict[int, str] = {}

    def visit(node: TreeNode, node_id: int = 0, depth: int = 0) -> int:
        base = node.label
        for prefix in simple_prefixes:
            if node.label.startswith(prefix):
                base = prefix
                break
        labels[node_id] = f"{base}_d{depth}"
        next_id = node_id + 1
        for child in node.children:
            next_id = visit(child, next_id, depth + 1)
        return next_id

    visit(tree)
    return labels


def wl_iteration(tree: TreeNode, labels: dict[int, str]) -> dict[int, str]:
    new_labels: dict[int, str] = {}

    def visit(node: TreeNode, node_id: int = 0) -> int:
        child_labels: list[str] = []
        next_id = node_id + 1
        for child in node.children:
            child_id = next_id
            next_id = visit(child, next_id)
            child_labels.append(labels[child_id])
        value = labels[node_id]
        if child_labels:
            value += "(" + ",".join(sorted(child_labels)) + ")"
        new_labels[node_id] = hashlib.md5(value.encode()).hexdigest()
        return next_id

    visit(tree)
    return new_labels


def wl_encoding(tree: TreeNode, iterations: int = 3) -> dict[str, int]:
    labels = initialize_labels(tree)
    combined: dict[str, int] = {}
    for iteration in range(min(tree_depth(tree), iterations)):
        labels = wl_iteration(tree, labels)
        for label, count in Counter(labels.values()).items():
            combined[f"{iteration}_{label}"] = count
    return combined


def wl_kernel(left: dict[str, int], right: dict[str, int]) -> float:
    if not left or not right:
        return 0.0
    common = left.keys() & right.keys()
    if not common:
        return 0.0
    dot = sum(left[key] * right[key] for key in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


# ---------------------------------------------------------------------------
# Vectorized WL kernel (option B): same cosine, compact int[] storage.
#
# The dict kernel above is *the* correctness reference. The vec path produces a
# BYTE-IDENTICAL float for every (target, candidate) pair. The guarantee rests on
# one fact: every operation before the final sqrt/divide is INTEGER arithmetic,
# which is exact and order-independent. Concretely, for a target T and candidate C
# sharing hash keys S = T.keys() & C.keys():
#
#   dot        = sum(T[k] * C[k] for k in S)        # int * int summed -> exact int
#   target_norm = sqrt(sum(v*v for v in T.values()))  # sqrt(exact int)
#   cand_norm   = sqrt(sum(v*v for v in C.values()))  # sqrt(exact int)
#   result      = max(0.0, min(1.0, dot / (target_norm * cand_norm)))
#
# Because dot and the norm radicands are exact integers, their values do not
# depend on iteration order, PYTHONHASHSEED, or container type. Replacing the
# string-keyed dicts with sorted int-id arrays changes only HOW we find the shared
# keys (a sorted two-pointer merge instead of a set intersection); it cannot change
# the integer dot, the integer radicands, or the resulting IEEE-754 double.
#
# Two invariants MUST hold for the byte-identity to be exact (enforced below and in
# the build script):
#   1. target_norm/cand_norm use sqrt(A) and sqrt(B) SEPARATELY then multiply —
#      never sqrt(A*B), which can differ by 1 ULP.
#   2. the target norm sums over ALL target hashes (including target-only ones that
#      have no corpus id); only the DOT restricts to shared/mapped ids. Dropping
#      target-only hashes from the norm would shrink it and change the cosine.
# A global hash->int id map (sorted unique hashes, corpus-fixed) makes the bijection
# deterministic and reproducible.
# ---------------------------------------------------------------------------


def wl_hash_to_id_map(*encodings: dict[str, int]) -> dict[str, int]:
    """Deterministic hash -> int id map over every hash in the given encodings.

    Sorting the unique hash strings makes the map reproducible across runs and
    independent of dict insertion / set iteration order. The corpus is fixed
    (Mathlib v4.18.0), so this map is stable for the lifetime of the DB.
    """
    hashes: set[str] = set()
    for enc in encodings:
        if enc:
            hashes.update(enc.keys())
    return {h: i for i, h in enumerate(sorted(hashes))}


def wl_encoding_to_vec(
    encoding: dict[str, int], hash_to_id: dict[str, int]
) -> tuple[tuple[int, ...], tuple[int, ...], float]:
    """Convert a WL encoding dict to (ids, counts, norm), sorted by id.

    ids/counts hold only the hashes present in ``hash_to_id`` (mapped hashes);
    the norm sums over ALL counts in ``encoding`` (mapped AND target/cand-only),
    matching wl_kernel's left_norm/right_norm exactly. ids and counts are sorted
    by id so wl_kernel_vec can do an O(n+m) two-pointer merge with no hashing.
    """
    if not encoding:
        return ((), (), 0.0)
    items = [(hash_to_id[h], c) for h, c in encoding.items() if h in hash_to_id]
    items.sort()
    ids = tuple(i for i, _ in items)
    counts = tuple(c for _, c in items)
    # norm over ALL values — byte-identical to wl_kernel's per-side norm.
    norm = math.sqrt(sum(value * value for value in encoding.values()))
    return (ids, counts, norm)


def wl_kernel_vec(
    target_ids: tuple[int, ...],
    target_counts: tuple[int, ...],
    target_norm: float,
    cand_ids: tuple[int, ...],
    cand_counts: tuple[int, ...],
    cand_norm: float,
) -> float:
    """Byte-identical to wl_kernel, but operates on sorted int-id arrays.

    Precompute (target_ids, target_counts, target_norm) ONCE per query via
    wl_encoding_to_vec(target_wl, hash_to_id); each candidate supplies its own
    precomputed (cand_ids, cand_counts, cand_norm) from the DB. The two-pointer
    merge replaces wl_kernel's set intersection; since the dot is an integer sum,
    the traversal order cannot affect the result.
    """
    if not target_ids or not cand_ids:
        return 0.0
    if target_norm == 0 or cand_norm == 0:
        return 0.0
    dot = 0
    i = 0
    j = 0
    n = len(target_ids)
    m = len(cand_ids)
    while i < n and j < m:
        a = target_ids[i]
        b = cand_ids[j]
        if a == b:
            dot += target_counts[i] * cand_counts[j]
            i += 1
            j += 1
        elif a < b:
            i += 1
        else:
            j += 1
    if dot == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (target_norm * cand_norm)))
