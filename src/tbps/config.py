from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from tbps.scoring import FusionWeights, MarginGateConfig, ScoreSettings


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    dbname: str
    user: str
    password_env: str

    def connection_kwargs(self) -> dict[str, str | int]:
        password = os.environ.get(self.password_env)
        if password is None:
            raise RuntimeError(
                f"database password environment variable is unset: {self.password_env}"
            )
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "password": password,
        }


@dataclass(frozen=True)
class RetrievalConfig:
    wl_iterations: int
    node_ratio: float
    large_tree_node_threshold: int
    large_tree_node_ratio: float
    absolute_node_difference: int
    candidate_limit: int
    coarse_alpha: float = 1.0
    adaptive_budget: bool = False
    k_max: int = 1500
    budget_beta: float = 0.0
    # WL retrieval backend: "dict" (JSONB dict), "vec" (precomputed int[] + norm in
    # wl_encodings_vec), or "sql" (cosine pushed down to SQL). All three compute the
    # same cosine; they only change how it is computed/transferred.
    wl_backend: str = "dict"


@dataclass(frozen=True)
class CascadeConfig:
    """Cost-aware adaptive cascade: per-query candidate limit K chosen before
    retrieval from query-side features only (``raw_target_size``).

    ``k_bins`` are the allowed candidate limits; ``policy`` names the rule
    (``"raw_size"`` maps ``raw_target_size`` to a bin via ``size_thresholds``,
    strict upper bounds with the largest bin unbounded).
    """

    enabled: bool
    k_bins: tuple[int, ...] = (100, 300, 750, 1500)
    policy: str = "raw_size"
    size_thresholds: tuple[int, ...] = (20, 50, 100)


@dataclass(frozen=True)
class HybridConfig:
    """Hybrid retrieval: WL + BM25 + Dense fused by weighted RRF.

    ``enabled`` controls whether the runner uses ``HybridRetriever`` instead of the
    plain WL-only ``PostgresRetriever``. ``retrievers`` is the set of enabled retriever
    names; the per-retriever RRF weights are ``w_*`` and ``rrf_k`` is the smoothing
    constant. Paths used when the corresponding retriever is enabled: ``bm25_index``,
    ``dense_index_dir``, ``dense_model`` (HF hub name), ``dense_device`` (auto-fallback
    to CPU if CUDA absent).

    ``dense_max_length`` is the byte-level truncation length for BOTH index build and
    query encoding — they must match so query and document embeddings live in the same
    space.
    """

    enabled: bool = False
    retrievers: tuple[str, ...] = ("wl", "bm25", "dense")
    w_wl: float = 0.50
    w_bm25: float = 0.15
    w_dense: float = 0.35
    rrf_k: float = 60.0
    bm25_index: Path | None = None
    dense_index_dir: Path | None = None
    dense_model: str = "kaiyuy/leandojo-lean4-retriever-byt5-small"
    dense_device: str = "cuda"
    dense_max_length: int = 512
    # Optional JSONL cache mapping query_id → clean Lean statement text (Lean default
    # pretty-printer). When set, the dense retriever embeds this text instead of the raw
    # query string so query and document embeddings share the same space.
    clean_query_cache: Path | None = None


@dataclass(frozen=True)
class KernelConfig:
    """Lean kernel applicability reranker (Stage 3).

    When ``enabled``, after Stage 1+2 the runner probes whether each of the top-N
    candidates' generalized conclusion is definitionally equal (``isDefEq``) to the
    query's generalized body. Applicable non-leaders get a constant ``bonus`` added to
    their structural score; the structural leader set is pinned at rank 1.

    ``include_target`` (default False) probes exactly the natural top-N; True appends
    the gold target to the probe set (measurement-only mode for applicability
    statistics; never use it for reported ranking metrics). ``mode`` selects the check
    strictness (``full``/``head``/``whnf``). The kernel runs as ONE Lean process over
    the whole batch; ``per_proc_timeout`` is the wall-clock backstop per process and
    ``batch`` caps queries per process (0 = no cap).

    Enabling kernel requires the active fusion profile to have dense=0 and bm25=0
    (leader-protect assumes final == struct); otherwise config load raises.
    """

    enabled: bool = False
    bonus: float = 0.5
    top_n: int = 50
    mode: str = "full"
    include_target: bool = False
    per_proc_timeout: int = 600
    batch: int = 0


@dataclass(frozen=True)
class BenchmarkConfig:
    test_a_query_file: Path
    test_a_label_file: Path
    test_b_manifest: Path


@dataclass(frozen=True)
class RunnerConfig:
    path: Path
    name: str
    seed: int
    workers: int
    output_top_k: int
    fusion_profile: str
    tie_method: str
    mathlib_sha: str
    database: DatabaseConfig
    retrieval: RetrievalConfig
    scoring: ScoreSettings
    benchmark: BenchmarkConfig
    cascade: CascadeConfig | None = None
    hybrid: HybridConfig | None = None
    kernel: KernelConfig | None = None


def load_config(path: Path, *, fusion_profile: str | None = None) -> RunnerConfig:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    run = data["run"]
    retrieval = data["retrieval"]
    teds = data["teds"]
    selected_profile = fusion_profile or run.get("fusion_profile", "paper")
    tie_method = retrieval["tie_method"]
    if tie_method != "competition":
        raise ValueError(f"unsupported tie method: {tie_method}")
    fusion = data["fusion"]
    if selected_profile not in fusion:
        raise ValueError(f"unknown fusion profile: {selected_profile}")
    large_profile = f"{selected_profile}_large_tree"
    large_weights = _weights(fusion[large_profile]) if large_profile in fusion else None
    database = data["database"]
    test_a = data["benchmark"]["test_a"]
    test_b = data["benchmark"]["test_b"]
    return RunnerConfig(
        path=path,
        name=run["name"],
        seed=int(run["seed"]),
        workers=int(run["workers"]),
        output_top_k=int(run.get("output_top_k", 10)),
        fusion_profile=selected_profile,
        tie_method=tie_method,
        mathlib_sha=data["versions"]["mathlib_sha"],
        database=DatabaseConfig(
            host=database["host"],
            port=int(database["port"]),
            dbname=database["dbname"],
            user=database["user"],
            password_env=database["password_env"],
        ),
        retrieval=RetrievalConfig(
            wl_iterations=int(retrieval["wl_iterations"]),
            node_ratio=float(retrieval["node_ratio"]),
            large_tree_node_threshold=int(retrieval["large_tree_node_threshold"]),
            large_tree_node_ratio=float(retrieval["large_tree_node_ratio"]),
            absolute_node_difference=int(retrieval["absolute_node_difference"]),
            candidate_limit=int(run["candidate_limit"]),
            coarse_alpha=float(retrieval.get("coarse_alpha", 1.0)),
            adaptive_budget=bool(retrieval.get("adaptive_budget", False)),
            k_max=int(retrieval.get("k_max", run["candidate_limit"])),
            budget_beta=float(retrieval.get("budget_beta", 0.0)),
            wl_backend=str(retrieval.get("wl_backend", "dict")),
        ),
        scoring=ScoreSettings(
            weights=_weights(fusion[selected_profile]),
            large_tree_weights=large_weights,
            tree_score_cutoff=int(retrieval["tree_score_cutoff"]),
            simple_node_scale=float(teds["simple_node_scale"]),
            insert_cost=float(teds["insert_cost"]),
            delete_cost=float(teds["delete_cost"]),
            replace_cost=float(teds["replace_cost"]),
            simple_node_prefixes=tuple(teds["simple_node_prefixes"]),
            margin_gate=_load_margin_gate(data.get("margin_gate")),
        ),
        benchmark=BenchmarkConfig(
            test_a_query_file=Path(test_a["query_file"]),
            test_a_label_file=Path(test_a["label_file"]),
            test_b_manifest=Path(test_b["manifest"]),
        ),
        cascade=_load_cascade(data.get("cascade")),
        hybrid=_load_hybrid(data.get("hybrid")),
        kernel=_load_kernel(data.get("kernel"), fusion[selected_profile]),
    )


def _load_cascade(data: dict | None) -> CascadeConfig | None:
    if not data:
        return None
    k_bins = tuple(int(k) for k in data.get("k_bins", (100, 300, 750, 1500)))
    thresholds = tuple(int(t) for t in data.get("size_thresholds", (20, 50, 100)))
    if len(k_bins) != len(thresholds) + 1:
        raise ValueError(
            f"cascade k_bins ({len(k_bins)}) must have one more entry than "
            f"size_thresholds ({len(thresholds)})"
        )
    return CascadeConfig(
        enabled=bool(data.get("enabled", False)),
        k_bins=k_bins,
        policy=str(data.get("policy", "raw_size")),
        size_thresholds=thresholds,
    )


def _load_margin_gate(data: dict | None) -> MarginGateConfig | None:
    if not data or not bool(data.get("enabled", False)):
        return None
    return MarginGateConfig(
        enabled=True,
        dense_weight=float(data.get("dense_weight", 0.6)),
        margin_thresh=float(data.get("margin_thresh", 0.05)),
        ramp_floor=float(data.get("ramp_floor", 0.2)),
    )


def _load_hybrid(data: dict | None) -> HybridConfig | None:
    if not data or not bool(data.get("enabled", False)):
        return None
    return HybridConfig(
        enabled=True,
        retrievers=tuple(str(r) for r in data.get("retrievers", ("wl", "bm25", "dense"))),
        w_wl=float(data.get("w_wl", 0.50)),
        w_bm25=float(data.get("w_bm25", 0.15)),
        w_dense=float(data.get("w_dense", 0.35)),
        rrf_k=float(data.get("rrf_k", 60.0)),
        bm25_index=Path(data["bm25_index"]) if data.get("bm25_index") else None,
        dense_index_dir=Path(data["dense_index_dir"]) if data.get("dense_index_dir") else None,
        dense_model=str(data.get("dense_model", "kaiyuy/leandojo-lean4-retriever-byt5-small")),
        dense_device=str(data.get("dense_device", "cuda")),
        dense_max_length=int(data.get("dense_max_length", 512)),
        clean_query_cache=Path(data["clean_query_cache"])
        if data.get("clean_query_cache")
        else None,
    )


def _weights(data: dict) -> FusionWeights:
    return FusionWeights(
        wl=float(data["wl"]),
        teds=float(data["teds"]),
        jaccard=float(data["jaccard"]),
        collapse_match=float(data["collapse_match"]),
        dense=float(data.get("dense", 0.0)),
        bm25=float(data.get("bm25", 0.0)),
    )


def _load_kernel(data: dict | None, fusion_profile: dict) -> KernelConfig | None:
    """Parse the opt-in ``[kernel]`` section. Returns None when absent/disabled so the
    baseline stays byte-identical. Validates that the active fusion profile has dense=0
    and bm25=0 (the leader-protect assumes ``final == struct``)."""
    if not data or not bool(data.get("enabled", False)):
        return None
    if float(fusion_profile.get("dense", 0.0)) != 0.0:
        raise ValueError(
            "[kernel] enabled requires the active fusion profile to have dense=0 "
            "(leader-protect assumes final == struct); "
            f"got fusion.dense={fusion_profile.get('dense', 0.0)}"
        )
    if float(fusion_profile.get("bm25", 0.0)) != 0.0:
        raise ValueError(
            "[kernel] enabled requires the active fusion profile to have bm25=0 "
            "(leader-protect assumes final == struct); "
            f"got fusion.bm25={fusion_profile.get('bm25', 0.0)}"
        )
    mode = str(data.get("mode", "full"))
    if mode not in ("full", "head", "whnf"):
        raise ValueError(f"[kernel] mode must be one of full/head/whnf, got {mode!r}")
    return KernelConfig(
        enabled=True,
        bonus=float(data.get("bonus", 0.5)),
        top_n=int(data.get("top_n", 50)),
        mode=mode,
        include_target=bool(data.get("include_target", False)),
        per_proc_timeout=int(data.get("per_proc_timeout", 600)),
        batch=int(data.get("batch", 0)),
    )
