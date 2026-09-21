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
    # Paper-faithful Stage-1 options. Defaults preserve the legacy coarse path
    # (pure WL, fixed candidate_limit) so existing configs and the golden check
    # are unaffected unless a TOML opts in. Paper §4.1 (p.6): k = min(k_max, β·|T_query|);
    # §4.2 (p.8): Stage 1 combines node count + collapse-match + WL.
    coarse_alpha: float = 1.0
    adaptive_budget: bool = False
    k_max: int = 1500
    budget_beta: float = 0.0
    # WL retrieval backend. "dict" (default) = legacy JSONB dict path, byte-identical
    # to the paper baseline. "vec" = option B: precomputed int[] + norm in
    # wl_encodings_vec, kernel in Python via wl_kernel_vec (byte-identical, faster).
    # "sql" = option C: cosine pushed down to SQL returning the top-k directly.
    # Both non-default backends are byte-identical to "dict" (verified); they only
    # change HOW the same cosine is computed/transferred, never the values.
    wl_backend: str = "dict"


@dataclass(frozen=True)
class CascadeConfig:
    """I2 cost-aware adaptive cascade.

    When ``enabled``, the runner selects a per-query candidate limit K *before*
    retrieval, using only query-side features (``raw_target_size``), then asks the
    retriever for exactly that many candidates and scores only them. This bounds the
    expensive scoring/deserialization work (and the SQL LIMIT bounds the DB work) for
    easy queries. When ``enabled`` is False (the default), behavior is unchanged:
    a single fixed ``candidate_limit``.

    ``k_bins`` are the allowed candidate limits. ``policy`` names the query-only rule;
    currently ``"raw_size"`` picks the bin whose upper ``size_thresholds`` segment
    contains ``raw_target_size`` (thresholds are strict upper bounds per bin, with the
    largest bin unbounded).

    To get the DB-side LIMIT saving, the experiment config must also set
    ``[retrieval] wl_backend = "sql"``; the cascade does not switch the backend itself.
    """

    enabled: bool
    k_bins: tuple[int, ...] = (100, 300, 750, 1500)
    policy: str = "raw_size"
    size_thresholds: tuple[int, ...] = (20, 50, 100)


@dataclass(frozen=True)
class HybridConfig:
    """I3 — hybrid retrieval (WL + BM25 + Dense fused by weighted RRF).

    ``enabled`` controls whether ``BaselineRunner`` uses ``HybridRetriever`` instead of the
    plain ``PostgresRetriever`` (WL-only). When False (the default), behavior is byte-identical
    to the locked baseline.

    ``retrievers`` is the ordered set of enabled retriever names (subset of ``{"wl", "bm25",
    "dense"}``); it determines which combination participates in RRF. The per-retriever weights
    (``w_*``) are a-priori principled values (chosen before evaluation, following the user's
    decision), NOT tuned on Test A/B. ``rrf_k`` is the RRF smoothing constant (the paper's
    authors use 60).

    Paths used when the corresponding retriever is enabled: ``bm25_index`` (JSON+DJSONL
    serialized ``BM25Index``), ``dense_index_dir`` (``embeddings.npy`` + ``names.json``), and
    ``dense_model`` (HF hub name of the LeanDojo ByT5 retriever). ``dense_device`` selects the
    encode device (auto-fallback to CPU if CUDA absent).

    ``dense_max_length`` is the byte-level truncation length for BOTH index build and query
    encoding — they MUST match so query and document embeddings live in the same space. If the
    index is built elsewhere (e.g. Kaggle) with a different ``max_length``, set this to match
    the build's value (recorded in the index provenance.json). Default 512 (the quality/speed
    balance benchmarked on a 4GB GPU: ~6 docs/s; keeps theorem name + type signature + leading
    binders).
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
    # I3 dense space-mismatch fix: path to a JSONL cache mapping query_id → clean Lean statement
    # text (Lean default pretty-printer, pp.all=false — the space LeanDojo ByT5 was trained on).
    # When set, the dense retriever embeds this CLEAN text instead of the raw `q(...)` query_text,
    # so query and document embeddings share the same clean space. Built by lean/TBPS/CleanPP.lean
    # (queries mode) → scripts/repro/export_clean_corpus.py. When None, the raw query_text is used
    # (the legacy mismatched behaviour — kept only for backward-compat / ablation). Test B
    # `state_text` is already clean notation, so it passes through unchanged.
    clean_query_cache: Path | None = None


@dataclass(frozen=True)
class KernelConfig:
    """I3 Idea 1 — Lean kernel applicability reranker (Stage 3).

    When ``enabled``, after Stage 1+2 (retrieval + structural fusion) the runner runs a
    Stage-3 post-pass: for each query it asks the Lean kernel whether each of the top-N
    candidates' *generalized conclusion* (universal binders → metavars) is definitionally
    equal (``isDefEq``) to the query's generalized body. Applicable non-leaders get a
    constant ``bonus`` added to their structural score; the structural leader set is pinned
    at rank 1 (never demoted) → R@1 == baseline by construction. This is the only mechanism
    that gains R@5/R@10 on full Test B without R@1 regression.

    ``bonus`` is an a-priori constant (NOT tuned): the sweep 0.05–5.0 leaves R@1 flat and
    R@5/10 saturate at ≥0.5, so any value in that range gives the same ranking. ``top_n``
    is the candidates-per-query checked by the kernel. ``include_target`` (default
    ``False``, the deployed leak-free setting) probes exactly the natural top-N and never
    consults the label; setting it ``True`` appends the gold target to the probe set —
    a measurement-only mode for applicability statistics that must never be used for
    reported ranking metrics. ``mode`` selects the kernel check strictness
    (``full`` = whole-conclusion isDefEq, strictest and best-performing).

    The kernel runs as ONE Lean process over the whole batch (amortizing the ~120 s
    ``import Mathlib``), mirroring the verified offline probe. ``per_proc_timeout`` is the
    wall-clock backstop per Lean process (a pathological ``isDefEq`` that does not check
    heartbeats is killed; the single hanging query is emitted as unknown and the rest
    resume in a fresh process). ``batch`` caps queries per process (0 = no cap).

    Default disabled (``enabled=False`` / section absent) → Stage 1+2 is byte-identical to
    the locked baseline and the golden check passes. The leader-protect assumes
    ``final == struct`` (dense=0/bm25=0 fusion, the deployed C1 config); enabling kernel
    with a dense/bm25 fusion weight > 0 raises at config load.
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
