from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from tbps.tree import TreeNode, collapse_match, constant_jaccard, normalized_teds, tree_node_count


@dataclass(frozen=True)
class FusionWeights:
    wl: float
    teds: float
    jaccard: float
    collapse_match: float
    # I3 dense-fusion: weight on the ByT5 dense cosine carried by a candidate. Default 0.0
    # (profiles that don't opt in are byte-identical to the pre-I3 fusion). Kept last so
    # positional 4-arg constructions (paper upstream_head etc.) remain valid.
    dense: float = 0.0
    # I3 bm25-fusion: weight on the normalized BM25 lexical score. Default 0.0 (opt-in only).
    bm25: float = 0.0


@dataclass(frozen=True)
class ScoreSettings:
    weights: FusionWeights
    large_tree_weights: FusionWeights | None = None
    tree_score_cutoff: int = 50
    simple_node_scale: float = 0.2
    insert_cost: float = 1.0
    delete_cost: float = 1.0
    replace_cost: float = 0.4
    simple_node_prefixes: tuple[str, ...] = ("BVar", "FVar", "MVar", "Sort", "Const")
    # I3 margin-gated dense fusion (idea D). Default None = byte-identical to pre-margin-gate.
    margin_gate: MarginGateConfig | None = None


@dataclass(frozen=True)
class MarginGateConfig:
    """I3 margin-gated dense fusion (idea D, refined).

    Per-query adaptive dense weight: scale the dense fusion weight DOWN when the
    structural leader has a large margin over #2 (structural is confident -> trust
    it, protect R@1 hits), and UP to full ``dense_weight`` when the margin is near
    zero (structural is unconfident -> dense likely holds the answer, rescue the
    R@10 misses). Uses only pool-wide structural properties (no target info) -> no
    leak, deterministic.

    The effective dense weight per query is::

        eff = dense_weight * max(ramp_floor, 1.0 - margin / margin_thresh)

    where ``margin`` is (top structural score - 2nd structural score) computed with
    the dense weight set to 0. ``margin_thresh`` is the margin above which dense is
    ramped all the way down to ``ramp_floor``. Default disabled (byte-identical to
    the pre-margin-gate fusion when ``enabled=False``).
    """

    enabled: bool = False
    dense_weight: float = 0.6
    margin_thresh: float = 0.05
    ramp_floor: float = 0.2


@dataclass(frozen=True)
class CandidateScore:
    name: str
    wl: float
    teds: float | None
    jaccard: float
    collapse_match: float
    final: float
    rank: int | None = None
    # I3 dense-fusion: the ByT5 dense cosine used as an extra fusion feature (0.0 when not
    # opted in). Kept last (after rank) so legacy positional 6-arg constructions stay valid.
    dense: float = 0.0
    # I3 bm25-fusion: the normalized BM25 lexical score (0.0 when not opted in).
    bm25: float = 0.0


def fuse_scores(
    *,
    wl: float,
    teds: float | None,
    jaccard: float,
    collapse: float,
    weights: FusionWeights,
    dense: float = 0.0,
    bm25: float = 0.0,
) -> float:
    return (
        weights.wl * wl
        + weights.teds * (0.0 if teds is None else teds)
        + weights.collapse_match * collapse
        + weights.jaccard * jaccard
        + weights.dense * dense
        + weights.bm25 * bm25
    )


def _structural_score(score: CandidateScore, weights: FusionWeights) -> float:
    """Structural-only final (dense/bm25 weight forced to 0). Used for margin gating."""
    teds = 0.0 if score.teds is None else score.teds
    return (
        weights.wl * score.wl
        + weights.teds * teds
        + weights.collapse_match * score.collapse_match
        + weights.jaccard * score.jaccard
    )


def apply_margin_gate(
    scores: list[CandidateScore],
    *,
    settings: ScoreSettings,
    large: bool,
) -> list[CandidateScore]:
    """Re-fuse finals with a per-query adaptive dense weight gated on the
    structural leader's margin (see MarginGateConfig). Returns NEW CandidateScore
    objects (rank unset); caller re-ranks via competition_rank.

    No-op (returns ``scores`` unchanged) when ``settings.margin_gate`` is None or
    disabled, preserving byte-identical behaviour for non-opt-in profiles.
    """
    gate = settings.margin_gate
    if gate is None or not gate.enabled or not scores:
        return scores
    weights = settings.large_tree_weights if large else settings.weights
    # structural margin: top - 2nd (dense/bm25 weight 0), name is the secondary key.
    struct = sorted(
        ((_structural_score(s, weights), s.name) for s in scores),
        key=lambda x: (-x[0], x[1]),
    )
    leader = struct[0][0]
    second = struct[1][0] if len(struct) > 1 else 0.0
    margin = leader - second
    mt = gate.margin_thresh
    if mt > 0:
        eff = gate.dense_weight * max(gate.ramp_floor, 1.0 - margin / mt)
    else:
        eff = gate.dense_weight
    # re-fuse: structural + effective dense weight * dense + bm25 (unchanged).
    rebuilt: list[CandidateScore] = []
    for s in scores:
        teds = 0.0 if s.teds is None else s.teds
        final = (
            weights.wl * s.wl
            + weights.teds * teds
            + weights.collapse_match * s.collapse_match
            + weights.jaccard * s.jaccard
            + eff * s.dense
            + weights.bm25 * s.bm25
        )
        rebuilt.append(
            CandidateScore(
                name=s.name,
                wl=s.wl,
                teds=s.teds,
                jaccard=s.jaccard,
                collapse_match=s.collapse_match,
                final=final,
                rank=None,
                dense=s.dense,
                bm25=s.bm25,
            )
        )
    return rebuilt


def score_candidate(
    name: str,
    candidate_tree: TreeNode,
    wl_score: float,
    target_tree: TreeNode,
    raw_target_size: int,
    settings: ScoreSettings,
) -> CandidateScore:
    return score_candidate_profiled(
        name, candidate_tree, wl_score, target_tree, raw_target_size, settings
    )[0]


def score_candidate_profiled(
    name: str,
    candidate_tree: TreeNode,
    wl_score: float,
    target_tree: TreeNode,
    raw_target_size: int,
    settings: ScoreSettings,
    dense: float = 0.0,
    bm25: float = 0.0,
) -> tuple[CandidateScore, dict[str, float]]:
    candidate_size = tree_node_count(candidate_tree)
    large = raw_target_size > settings.tree_score_cutoff
    teds = None
    started = time.perf_counter()
    if not large:
        teds = normalized_teds(
            target_tree,
            candidate_tree,
            left_size=raw_target_size,
            right_size=candidate_size,
            simple_prefixes=settings.simple_node_prefixes,
            simple_node_scale=settings.simple_node_scale,
            insert_cost=settings.insert_cost,
            delete_cost=settings.delete_cost,
            replace_cost=settings.replace_cost,
        )
    teds_seconds = time.perf_counter() - started
    started = time.perf_counter()
    jaccard = constant_jaccard(target_tree, candidate_tree)
    jaccard_seconds = time.perf_counter() - started
    started = time.perf_counter()
    collapse = collapse_match(target_tree, candidate_tree)
    collapse_seconds = time.perf_counter() - started
    weights = (
        settings.large_tree_weights if large and settings.large_tree_weights else settings.weights
    )
    return (
        CandidateScore(
            name=name,
            wl=wl_score,
            teds=teds,
            jaccard=jaccard,
            collapse_match=collapse,
            final=fuse_scores(
                wl=wl_score,
                teds=teds,
                jaccard=jaccard,
                collapse=collapse,
                weights=weights,
                dense=dense,
                bm25=bm25,
            ),
            dense=dense,
            bm25=bm25,
        ),
        {
            "teds": teds_seconds,
            "jaccard": jaccard_seconds,
            "collapse_match": collapse_seconds,
        },
    )


def competition_rank(scores: list[CandidateScore]) -> list[CandidateScore]:
    """Sort deterministically and assign exact-float competition ranks (1, 2, 2, 4)."""
    ordered = sorted(scores, key=lambda score: (-score.final, score.name))
    ranked: list[CandidateScore] = []
    previous: float | None = None
    rank = 1
    for index, score in enumerate(ordered):
        if score.final != previous:
            rank = index + 1
        ranked.append(
            CandidateScore(
                name=score.name,
                wl=score.wl,
                teds=score.teds,
                jaccard=score.jaccard,
                collapse_match=score.collapse_match,
                final=score.final,
                rank=rank,
                dense=score.dense,
                bm25=score.bm25,
            )
        )
        previous = score.final
    return ranked


def aggregate_metrics(records: list[dict[str, Any]], ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    """Aggregate single-label metrics over every record, including failures/misses."""
    total = len(records)
    ranks = [record.get("final_rank") for record in records]
    result: dict[str, Any] = {"queries": total, "completed": sum(r is not None for r in ranks)}
    result["mrr"] = (
        sum(1.0 / rank for rank in ranks if isinstance(rank, int)) / total if total else 0.0
    )
    for k in ks:
        hits = sum(isinstance(rank, int) and rank <= k for rank in ranks)
        recall = hits / total if total else 0.0
        precision = sum(1.0 / k for rank in ranks if isinstance(rank, int) and rank <= k)
        ndcg = sum(
            1.0 / math.log2(rank + 1) for rank in ranks if isinstance(rank, int) and rank <= k
        )
        result[f"recall@{k}"] = recall
        result[f"precision@{k}"] = precision / total if total else 0.0
        result[f"f1@{k}"] = (2.0 * hits / (k + 1)) / total if total else 0.0
        result[f"ndcg@{k}"] = ndcg / total if total else 0.0
    return result
