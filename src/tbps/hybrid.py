"""Hybrid retrieval: fuse WL + BM25 + Dense via weighted Reciprocal Rank Fusion.

``HybridRetriever`` wraps the existing ``PostgresRetriever`` (WL, structural) and adds
two complementary candidate generators — ``BM25Index`` (lexical) and ``DenseIndex``
(semantic, the LeanDojo ByT5 retriever) — fusing their top-K lists with weighted RRF
into a single ranked candidate set. It exposes the same ``retrieve() -> RetrievalResult``
interface as ``PostgresRetriever`` so the runner's scoring stage is unchanged.

RRF: ``rrf_score(name) = sum_r w_r / (k_rrf + rank_r(name))`` over the enabled
retrievers, where rank is 1-indexed and a name absent from a retriever's list
contributes 0 from that term. The fused list is sorted by ``(-rrf_score, name)``.

``wl_score`` carries the WL cosine into ``fuse_scores``: the real WL score for names
in the WL top-K, 0.0 for names that enter the pool only via BM25/Dense (they compete
via TEDS/jaccard/collapse alone). ``expression_json`` for every fused candidate is
resolved in a phase-2 DB fetch.

Determinism: all retriever lists are sorted by their own ``(-score, name)``; RRF
preserves ``name`` as the final tie-break. When ``hybrid.enabled = false`` the runner
uses ``PostgresRetriever`` directly; this module is never imported without torch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from tbps.config import HybridConfig
from tbps.database import PostgresRetriever, RetrievalResult, RetrievedCandidate
from tbps.retrieval.bm25 import BM25Index, tokenize_expr
from tbps.tree import TreeNode


@dataclass(frozen=True)
class _RankedName:
    """A name with its rank in one retriever's list (1-indexed)."""

    name: str
    rank: int


class HybridRetriever:
    """WL + BM25 + Dense fused by weighted RRF. Drop-in for ``PostgresRetriever.retrieve``.

    The BM25 and Dense indices are loaded once per worker process; the WL retriever is the
    existing ``PostgresRetriever``. The dense encoder is held only if a dense retriever is
    enabled (lazy torch import keeps the baseline venv clean).
    """

    def __init__(
        self,
        wl_retriever: PostgresRetriever,
        bm25_index: BM25Index | None,
        dense_index: object | None = None,
        dense_encoder: object | None = None,
        config: HybridConfig | None = None,
    ) -> None:
        self.wl = wl_retriever
        self.bm25 = bm25_index
        self.dense_index = dense_index
        self.dense_encoder = dense_encoder
        self.config = config or HybridConfig()
        # Resolve the per-retriever weights, normalizing only for the enabled retrievers.
        enabled = self.config.retrievers
        weights = {
            name: w
            for name, w in {
                "wl": self.config.w_wl,
                "bm25": self.config.w_bm25,
                "dense": self.config.w_dense,
            }.items()
            if name in enabled
        }
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("hybrid weights sum to 0; at least one retriever must have weight > 0")
        # Weights are deliberately NOT renormalized (raw values are used as-is).
        self._weights = weights

    def retrieve(
        self,
        target_name: str,
        target_tree: TreeNode,
        *,
        ratio_node_count: int | None = None,
        candidate_limit_override: int | None = None,
        query_text: str | None = None,
        expression_json: object | None = None,
    ) -> RetrievalResult:
        limit = candidate_limit_override or self.wl.retrieval.candidate_limit
        k_rrf = self.config.rrf_k
        stages: dict[str, float] = {}

        # --- WL retrieval (structural, the paper's core method) ---
        t0 = time.perf_counter()
        wl_result = self.wl.retrieve(
            target_name,
            target_tree,
            ratio_node_count=ratio_node_count,
            candidate_limit_override=candidate_limit_override,
        )
        stages["wl_retrieval"] = time.perf_counter() - t0
        stages.update(wl_result.duration_seconds)
        wl_candidates = wl_result.candidates
        # name -> real WL cosine score (for the scoring stage).
        wl_scores: dict[str, float] = {c.name: c.wl_score for c in wl_candidates}
        wl_ranked: list[str] = [c.name for c in wl_candidates]

        # --- BM25 retrieval (lexical) ---
        # Tokenize the SAME way the corpus was indexed (Const decl names from the Expr tree),
        # so query tokens match the inverted index. expression_json is the pre-CSE elaborated
        # Expr the runner already has; tokenize_expr deserializes + walks it.
        bm25_ranked: list[str] = []
        # name -> normalized BM25 score (divided by the query's max BM25 score so the strongest
        # lexical hit is 1.0). Carried into the scoring-stage bm25-fusion term; WL/dense-only
        # candidates get 0.0. Raw BM25 is unbounded, so normalization is required for the weight
        # to mean the same thing across queries.
        bm25_scores: dict[str, float] = {}
        if "bm25" in self._weights and self.bm25 is not None:
            t0 = time.perf_counter()
            q_tokens = tokenize_expr(expression_json) if expression_json is not None else []
            if q_tokens:
                hits = self.bm25.query(q_tokens, limit=limit)
                bm25_ranked = [name for name, _ in hits]
                max_bm25 = max((s for _, s in hits), default=0.0)
                if max_bm25 > 0.0:
                    bm25_scores = {name: s / max_bm25 for name, s in hits}
            stages["bm25_retrieval"] = time.perf_counter() - t0

        # --- Dense retrieval (semantic, LeanDojo ByT5) ---
        dense_ranked: list[str] = []
        # name -> ByT5 cosine, carried into the scoring-stage dense-fusion term. WL/BM25-only
        # candidates get 0.0 (the dense index didn't surface them); a dense-only candidate
        # (WL-blind) carries its real cosine so it can compete with WL hits in fusion.
        dense_scores: dict[str, float] = {}
        if (
            "dense" in self._weights
            and self.dense_index is not None
            and self.dense_encoder is not None
        ):
            t0 = time.perf_counter()
            if not query_text:
                raise ValueError(
                    "hybrid dense retrieval requires query_text (the Lean goal/expression string); "
                    "the runner must pass case.query_text"
                )
            goal_emb = self.dense_encoder.encode_one(query_text)
            hits = self.dense_index.query(goal_emb, limit=limit)
            dense_ranked = [name for name, _ in hits]
            dense_scores = {name: cosine for name, cosine in hits}
            stages["dense_retrieval"] = time.perf_counter() - t0

        # --- Weighted RRF fusion ---
        t0 = time.perf_counter()
        fused = _rrf_fuse(
            {"wl": wl_ranked, "bm25": bm25_ranked, "dense": dense_ranked},
            weights=self._weights,
            k=k_rrf,
            limit=limit,
        )
        stages["rrf_fusion"] = time.perf_counter() - t0

        # --- Resolve expression_json for the fused candidates (phase-2 DB fetch) ---
        # Names already in the WL top-K have their expression_json; the rest need a DB lookup.
        t0 = time.perf_counter()
        expr_map: dict[str, Any] = {c.name: c.expression_json for c in wl_candidates}
        missing = [n for n in fused if n not in expr_map]
        if missing:
            with self.wl._connect() as connection, connection.cursor() as cursor:  # noqa: SLF001
                fetched = self.wl._fetch_expr_for_names(cursor, missing)  # noqa: SLF001
                expr_map.update(fetched)
        stages["hybrid_phase2_expr_fetch"] = time.perf_counter() - t0

        # Build the final candidate list: wl_score = real WL score if in WL top-K, else 0.0;
        # dense_score = ByT5 cosine if surfaced by the dense index, else 0.0; bm25_score =
        # normalized BM25 if surfaced by the lexical index, else 0.0. Rounding to 6 dp keeps
        # the recorded provenance compact without changing ranks materially.
        candidates = tuple(
            RetrievedCandidate(
                name=name,
                expression_json=expr_map.get(name),
                wl_score=wl_scores.get(name, 0.0),
                dense_score=round(dense_scores.get(name, 0.0), 6),
                bm25_score=round(bm25_scores.get(name, 0.0), 6),
            )
            for name in fused
        )

        return RetrievalResult(
            candidates=candidates,
            corpus_count=wl_result.corpus_count,
            node_filtered_count=wl_result.node_filtered_count,
            joined_count=wl_result.joined_count,
            minimum_nodes=wl_result.minimum_nodes,
            maximum_nodes=wl_result.maximum_nodes,
            target_in_corpus=wl_result.target_in_corpus,
            target_has_wl=wl_result.target_has_wl,
            duration_seconds=stages,
            limit=limit,
        )


def _rrf_fuse(
    ranked_lists: dict[str, list[str]],
    *,
    weights: dict[str, float],
    k: int,
    limit: int,
) -> list[str]:
    """Weighted Reciprocal Rank Fusion. Returns ``limit`` names by ``(-score, name)``.

    A name absent from a list contributes 0 from that list's term. The ``name`` secondary sort
    is the deterministic tie-break (CLAUDE.md).
    """
    scores: dict[str, float] = {}
    for retriever, ranked in ranked_lists.items():
        w = weights.get(retriever, 0.0)
        if w <= 0.0 or not ranked:
            continue
        for rank, name in enumerate(ranked, 1):
            scores[name] = scores.get(name, 0.0) + w / (k + rank)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [name for name, _ in ordered[:limit]]
