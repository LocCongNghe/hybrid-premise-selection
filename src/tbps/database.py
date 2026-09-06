from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import psycopg2
from psycopg2.extensions import connection as PgConnection

from tbps.config import DatabaseConfig, RetrievalConfig
from tbps.expr import deserialize_expr
from tbps.tree import TreeNode, collapse_match, expr_to_tree, simplify_forall, tree_node_count
from tbps.wl import (
    wl_encoding,
    wl_encoding_to_vec,
    wl_kernel,
    wl_kernel_vec,
)


@dataclass(frozen=True)
class RetrievedCandidate:
    name: str
    expression_json: dict[str, Any]
    wl_score: float
    # Populated only on the paper-faithful coarse path (coarse_alpha < 1.0); None on
    # the legacy pure-WL path so existing records keep their shape.
    coarse_score: float | None = None
    collapse_match: float | None = None
    # I3 dense-fusion: the ByT5 dense cosine for this candidate (0.0 if absent or when the
    # hybrid dense path isn't enabled). None on the plain WL path keeps existing records as-is.
    dense_score: float | None = None
    # I3 bm25-fusion: the normalized BM25 lexical score for this candidate (0.0 if absent or
    # when the hybrid bm25 path isn't enabled). None on the plain WL path keeps records as-is.
    bm25_score: float | None = None


@dataclass(frozen=True)
class RetrievalResult:
    candidates: tuple[RetrievedCandidate, ...]
    corpus_count: int
    node_filtered_count: int
    joined_count: int
    minimum_nodes: float
    maximum_nodes: float
    target_in_corpus: bool
    target_has_wl: bool
    duration_seconds: dict[str, float]
    # The candidate limit this retrieval actually used (fixed candidate_limit, the
    # paper §4.1 adaptive-budget k, or an I2 cascade override). Logged for provenance.
    limit: int | None = None


class PostgresRetriever:
    def __init__(self, database: DatabaseConfig, retrieval: RetrievalConfig):
        self.database = database
        self.retrieval = retrieval
        # corpus_count is constant across queries (size of the non-null corpus), so it is
        # computed once and cached for the lifetime of the retriever (one per worker process).
        self._corpus_count: int | None = None
        # A persistent connection reused across queries. The retriever is single-threaded
        # (run_batch drives it sequentially in the main process), so no locking is needed.
        self._connection: PgConnection | None = None
        # Option B: the global hash->int id map is constant for the DB's lifetime, so it
        # is loaded once and cached. None on the legacy dict/sql backends.
        self._hash_to_id: dict[str, int] | None = None

    def retrieve(
        self,
        target_name: str,
        target_tree: TreeNode,
        *,
        ratio_node_count: int | None = None,
        candidate_limit_override: int | None = None,
    ) -> RetrievalResult:
        target_nodes = tree_node_count(target_tree)
        ratio_nodes = target_nodes if ratio_node_count is None else ratio_node_count
        ratio = (
            self.retrieval.large_tree_node_ratio
            if ratio_nodes >= self.retrieval.large_tree_node_threshold
            else self.retrieval.node_ratio
        )
        difference = self.retrieval.absolute_node_difference
        minimum = max(0.0, min(target_nodes / ratio, target_nodes - difference))
        maximum = max(target_nodes * ratio, target_nodes + difference)
        target_wl = wl_encoding(target_tree, self.retrieval.wl_iterations)
        # Paper §4.1 (p.6): k = min(k_max, β·|T_query|). When adaptive_budget is off
        # (the default), k falls back to the fixed candidate_limit, matching the legacy
        # path. An I2 cascade override (candidate_limit_override) takes precedence over
        # both — it is the per-query K chosen by the query-only policy.
        if candidate_limit_override is not None:
            limit = candidate_limit_override
        elif self.retrieval.adaptive_budget:
            limit = min(
                self.retrieval.k_max,
                int(self.retrieval.budget_beta * target_nodes),
            )
        else:
            limit = self.retrieval.candidate_limit
        use_coarse_cm = self.retrieval.coarse_alpha < 1.0

        database_started = time.perf_counter()
        # The connection context manager commits on success / rolls back on exception but
        # does NOT close the connection — so a persistent connection is reused across queries
        # while still recovering cleanly from an aborted transaction mid-query.
        with self._connect() as connection, connection.cursor() as cursor:
            corpus_count = self._cached_corpus_count(cursor)
            node_filtered_count = self._node_filtered_count(cursor, minimum, maximum)
            # On the legacy pure-WL path (coarse_alpha >= 1.0, the default) the expression
            # JSON is only needed for the final top-k winners, not for ranking — ranking is
            # driven solely by the WL kernel. Fetching it for the whole node-filtered set
            # (often 50k-100k rows, ~1.3KB each) dominated DB time. So we split into two
            # phases: phase 1 fetches only name + WL encoding, ranks, and trims to top-k;
            # phase 2 fetches expr_cse_json for just those k names. The coarse-CM path
            # needs the expression tree of *every* candidate to compute collapse_match, so
            # it keeps the single-phase fetch.
            if use_coarse_cm:
                rows = self._fetch_full(cursor, minimum, maximum)
                target_in_corpus, target_has_wl = self._coverage(cursor, target_name)
                database_seconds = time.perf_counter() - database_started
                scored = self._score_full(target_tree, target_wl, rows)
                wl_seconds = time.perf_counter() - database_started - database_seconds
                joined_count = len(rows)
            else:
                backend = self.retrieval.wl_backend
                if backend == "sql":
                    # Option C: cosine pushed down to SQL; DB returns the top-k already
                    # ranked, so there is no Python WL pass and no 50k-row fetch.
                    hash_to_id = self._cached_hash_to_id(cursor)
                    target_in_corpus, target_has_wl = self._coverage(cursor, target_name)
                    scored, joined_count = self._retrieve_sql(
                        cursor, target_wl, hash_to_id, minimum, maximum, limit
                    )
                    database_seconds = time.perf_counter() - database_started
                    wl_seconds = 0.0
                    # Phase 2 still fetches expr_cse_json for the surviving top-k names.
                    phase2_started = time.perf_counter()
                    if scored:
                        expr_map = self._fetch_expr_for_names(cursor, [c.name for c in scored])
                        scored = [
                            RetrievedCandidate(
                                name=c.name,
                                expression_json=expr_map.get(c.name),
                                wl_score=c.wl_score,
                            )
                            for c in scored
                        ]
                    phase2_seconds = time.perf_counter() - phase2_started
                    database_seconds = database_seconds + phase2_seconds
                elif backend == "vec":
                    # Option B: fetch compact int[]+norm (not JSONB dicts); kernel runs in
                    # Python over the precomputed vectors via wl_kernel_vec.
                    vec_rows = self._fetch_name_wl_vec(cursor, minimum, maximum)
                    hash_to_id = self._cached_hash_to_id(cursor)
                    target_in_corpus, target_has_wl = self._coverage(cursor, target_name)
                    phase1_seconds = time.perf_counter() - database_started
                    wl_started = time.perf_counter()
                    scored = self._score_name_wl_vec(target_wl, hash_to_id, vec_rows)
                    wl_seconds = time.perf_counter() - wl_started
                    phase2_started = time.perf_counter()
                    if scored:
                        expr_map = self._fetch_expr_for_names(
                            cursor, [c.name for c in scored[:limit]]
                        )
                        scored = [
                            RetrievedCandidate(
                                name=c.name,
                                expression_json=expr_map.get(c.name),
                                wl_score=c.wl_score,
                            )
                            for c in scored[:limit]
                        ]
                    phase2_seconds = time.perf_counter() - phase2_started
                    database_seconds = phase1_seconds + phase2_seconds
                    joined_count = len(vec_rows)
                else:
                    name_wl_rows = self._fetch_name_wl(cursor, minimum, maximum)
                    target_in_corpus, target_has_wl = self._coverage(cursor, target_name)
                    phase1_seconds = time.perf_counter() - database_started
                    # Pure-WL ranking over the full node-filtered set (Python-side).
                    wl_started = time.perf_counter()
                    scored = self._score_name_wl(target_wl, name_wl_rows)
                    wl_seconds = time.perf_counter() - wl_started
                    # Phase 2: fetch the expression JSON only for the surviving top-k names.
                    phase2_started = time.perf_counter()
                    if scored:
                        expr_map = self._fetch_expr_for_names(
                            cursor, [c.name for c in scored[:limit]]
                        )
                        scored = [
                            RetrievedCandidate(
                                name=c.name,
                                expression_json=expr_map.get(c.name),
                                wl_score=c.wl_score,
                            )
                            for c in scored[:limit]
                        ]
                    phase2_seconds = time.perf_counter() - phase2_started
                    database_seconds = phase1_seconds + phase2_seconds
                    joined_count = len(name_wl_rows)
        return RetrievalResult(
            candidates=tuple(scored),
            corpus_count=corpus_count,
            node_filtered_count=node_filtered_count,
            joined_count=joined_count,
            minimum_nodes=minimum,
            maximum_nodes=maximum,
            target_in_corpus=target_in_corpus,
            target_has_wl=target_has_wl,
            duration_seconds={"database": database_seconds, "wl": wl_seconds},
            limit=limit,
        )

    def _connect(self) -> PgConnection:
        if self._connection is None or self._connection.closed:
            self._connection = psycopg2.connect(**self.database.connection_kwargs())
        return self._connection

    def close(self) -> None:
        """Close the persistent connection (used when a retriever is discarded)."""
        if self._connection is not None and not self._connection.closed:
            self._connection.close()
        self._connection = None

    def _cached_corpus_count(self, cursor) -> int:
        if self._corpus_count is None:
            cursor.execute(
                """
                SELECT count(*)
                FROM mathlib_filtered
                WHERE expr_cse_json IS NOT NULL AND expr_cse_json <> 'null'::jsonb
                """
            )
            self._corpus_count = int(cursor.fetchone()[0])
        return self._corpus_count

    @staticmethod
    def _node_filtered_count(cursor, minimum: float, maximum: float) -> int:
        cursor.execute(
            """
            SELECT count(*)
            FROM mathlib_filtered
            WHERE expr_cse_json IS NOT NULL AND expr_cse_json <> 'null'::jsonb
              AND simp_node_count BETWEEN %s AND %s
            """,
            (minimum, maximum),
        )
        return int(cursor.fetchone()[0])

    @staticmethod
    def _fetch_name_wl(cursor, minimum: float, maximum: float) -> list[tuple]:
        cursor.execute(
            """
            SELECT d.name, w.simp_wl_encode_3
            FROM mathlib_filtered AS d
            JOIN wl_encodings_new AS w ON d.name = w.theorem_name
            WHERE d.expr_cse_json IS NOT NULL AND d.expr_cse_json <> 'null'::jsonb
              AND d.simp_node_count BETWEEN %s AND %s
            ORDER BY d.name
            """,
            (minimum, maximum),
        )
        return cursor.fetchall()

    @staticmethod
    def _fetch_full(cursor, minimum: float, maximum: float) -> list[tuple]:
        cursor.execute(
            """
            SELECT d.name, d.expr_cse_json, w.simp_wl_encode_3
            FROM mathlib_filtered AS d
            JOIN wl_encodings_new AS w ON d.name = w.theorem_name
            WHERE d.expr_cse_json IS NOT NULL AND d.expr_cse_json <> 'null'::jsonb
              AND d.simp_node_count BETWEEN %s AND %s
            ORDER BY d.name
            """,
            (minimum, maximum),
        )
        return cursor.fetchall()

    @staticmethod
    def _fetch_expr_for_names(cursor, names: list[str]) -> dict[str, object]:
        if not names:
            return {}
        cursor.execute(
            "SELECT name, expr_cse_json FROM mathlib_filtered WHERE name = ANY(%s)",
            (names,),
        )
        return {name: expr for name, expr in cursor.fetchall()}

    def _score_name_wl(self, target_wl: dict, rows: list[tuple]) -> list[RetrievedCandidate]:
        # Pure-WL coarse path: rank by WL kernel only. expr_cse_json is fetched later in
        # phase 2 for the surviving top-k, so it is left as None here.
        scored = [
            RetrievedCandidate(
                name=name, expression_json=None, wl_score=wl_kernel(target_wl, encoding or {})
            )
            for name, encoding in rows
        ]
        # Sort by WL score descending, candidate name ascending as the deterministic
        # secondary key (CLAUDE.md). Negated primary key keeps the name tie-break ascending,
        # as upstream's SQL `ORDER BY d.name` + stable WL sort does.
        scored.sort(key=lambda c: (-c.wl_score, c.name))
        return scored

    def _score_full(
        self, target_tree: TreeNode, target_wl: dict, rows: list[tuple]
    ) -> list[RetrievedCandidate]:
        # Paper §4.2 coarse path: rank by alpha*WL + (1-alpha)*collapse_match. Needs the
        # expression tree of every candidate, hence the single-phase full fetch.
        coarse_alpha = self.retrieval.coarse_alpha
        scored: list[RetrievedCandidate] = []
        for name, expression_json, encoding in rows:
            wl_score = wl_kernel(target_wl, encoding or {})
            candidate_tree = expr_to_tree(simplify_forall(deserialize_expr(expression_json)))
            cm = collapse_match(target_tree, candidate_tree)
            coarse_score = coarse_alpha * wl_score + (1.0 - coarse_alpha) * cm
            scored.append(
                RetrievedCandidate(
                    name=name,
                    expression_json=expression_json,
                    wl_score=wl_score,
                    coarse_score=coarse_score,
                    collapse_match=cm,
                )
            )
        scored.sort(key=lambda c: (-c.coarse_score, c.name))
        return scored

    # ------------------------------------------------------------------
    # Option B (vec) and Option C (sql) backends — both byte-identical to the
    # dict path. See wl.wl_kernel_vec for the exactness argument.
    # ------------------------------------------------------------------

    def _cached_hash_to_id(self, cursor) -> dict[str, int]:
        """Load the global hash->int id map once per retriever (constant for the DB)."""
        if self._hash_to_id is None:
            cursor.execute("SELECT hash, id FROM wl_hash_map")
            self._hash_to_id = dict(cursor.fetchall())
        return self._hash_to_id

    @staticmethod
    def _fetch_name_wl_vec(cursor, minimum: float, maximum: float) -> list[tuple]:
        cursor.execute(
            """
            SELECT d.name, v.wl_ids, v.wl_counts, v.wl_norm
            FROM mathlib_filtered AS d
            JOIN wl_encodings_vec AS v ON d.name = v.theorem_name
            WHERE d.expr_cse_json IS NOT NULL AND d.expr_cse_json <> 'null'::jsonb
              AND d.simp_node_count BETWEEN %s AND %s
            ORDER BY d.name
            """,
            (minimum, maximum),
        )
        return cursor.fetchall()

    def _score_name_wl_vec(
        self, target_wl: dict, hash_to_id: dict[str, int], rows: list[tuple]
    ) -> list[RetrievedCandidate]:
        # Precompute the target's sorted (ids, counts, norm) ONCE — the dict path
        # recomputes the target norm for every candidate; this is the main saving.
        t_ids, t_counts, t_norm = wl_encoding_to_vec(target_wl, hash_to_id)
        scored = [
            RetrievedCandidate(
                name=name,
                expression_json=None,
                wl_score=wl_kernel_vec(
                    t_ids, t_counts, t_norm, tuple(ids), tuple(counts), float(norm)
                ),
            )
            for name, ids, counts, norm in rows
        ]
        # Same deterministic sort as the dict path: (-wl_score, name).
        scored.sort(key=lambda c: (-c.wl_score, c.name))
        return scored

    def _retrieve_sql(
        self,
        cursor,
        target_wl: dict,
        hash_to_id: dict[str, int],
        minimum: float,
        maximum: float,
        limit: int,
    ) -> tuple[list[RetrievedCandidate], int]:
        """Option C: push the WL cosine into SQL; return the top-`limit` ranked.

        The target's hash ids/counts/norm are computed in Python (wl_encoding_to_vec)
        and passed to SQL as parallel arrays. SQL unnests each candidate's precomputed
        (wl_ids, wl_counts, wl_norm), left-joins to the target ids, sums the shared
        count products into an exact-integer dot, then computes the SAME cosine as
        wl_kernel_vec: dot / (target_norm * cand_norm), clamped to [0,1]. Because the
        dot is an integer and the norms are precomputed doubles, the result is
        byte-identical to the Python path. ORDER BY (-cosine, name) preserves the
        deterministic name tie-break; LIMIT returns only the top-k.
        """
        t_ids, t_counts, t_norm = wl_encoding_to_vec(target_wl, hash_to_id)
        if not t_ids or t_norm == 0.0:
            # No mapped target hashes: every cosine is 0.0 (matches wl_kernel returning
            # 0.0 for empty/no-common). Still return the alphabetically-first limit names
            # with wl_score=0.0, mirroring the dict path's sort of all-zero scores by name.
            cursor.execute(
                """
                SELECT d.name, count(*) OVER ()
                FROM mathlib_filtered AS d
                JOIN wl_encodings_vec AS v ON d.name = v.theorem_name
                WHERE d.expr_cse_json IS NOT NULL AND d.expr_cse_json <> 'null'::jsonb
                  AND d.simp_node_count BETWEEN %s AND %s
                ORDER BY d.name
                LIMIT %s
                """,
                (minimum, maximum, limit),
            )
            r = cursor.fetchall()
            joined = r[0][1] if r else 0
            return [
                RetrievedCandidate(name=n, expression_json=None, wl_score=0.0) for n, _ in r
            ], joined

        # Pass target ids/counts as parallel arrays; target_norm as a scalar.
        # The cosine is computed in an inner SELECT then ORDER BY (-cosine), name is
        # applied on its output column (referencing an alias inside an expression like
        # (-cosine) only resolves if cosine is a real column, hence the outer wrap).
        cursor.execute(
            """
            WITH target AS (
                SELECT id, cnt FROM unnest(%s::int[], %s::int[]) AS t(id, cnt)
            )
            SELECT name, cosine, total FROM (
                SELECT
                    d.name,
                    CASE
                        WHEN v.wl_norm = 0 OR v.wl_norm IS NULL THEN 0.0
                        ELSE greatest(0.0, least(1.0,
                            (COALESCE((
                                SELECT SUM(t.cnt::bigint * c.cnt::bigint)
                                FROM unnest(v.wl_ids, v.wl_counts) AS c(id, cnt)
                                JOIN target t ON t.id = c.id
                            ), 0)::float8) / (%s::float8 * v.wl_norm)))
                    END AS cosine,
                    count(*) OVER () AS total
                FROM mathlib_filtered AS d
                JOIN wl_encodings_vec AS v ON d.name = v.theorem_name
                WHERE d.expr_cse_json IS NOT NULL AND d.expr_cse_json <> 'null'::jsonb
                  AND d.simp_node_count BETWEEN %s AND %s
            ) s
            ORDER BY (-s.cosine), s.name
            LIMIT %s
            """,
            (list(t_ids), list(t_counts), float(t_norm), minimum, maximum, limit),
        )
        r = cursor.fetchall()
        joined = r[0][2] if r else 0
        return [
            RetrievedCandidate(name=name, expression_json=None, wl_score=float(cosine))
            for name, cosine, _ in r
        ], int(joined)

    @staticmethod
    def _coverage(cursor, target_name: str) -> tuple[bool, bool]:
        cursor.execute(
            """
            SELECT
              EXISTS(SELECT 1 FROM mathlib_filtered WHERE name = %s),
              EXISTS(SELECT 1 FROM wl_encodings_new WHERE theorem_name = %s)
            """,
            (target_name, target_name),
        )
        corpus, wl = cursor.fetchone()
        return bool(corpus), bool(wl)
