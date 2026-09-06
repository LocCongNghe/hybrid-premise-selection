from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace as dataclasses_replace
from pathlib import Path
from collections.abc import Callable
from typing import Any, Iterator

from tbps.benchmark import load_test_a
from tbps.cascade import choose_K
from tbps.config import HybridConfig, RunnerConfig, load_config
from tbps.cse import common_subexpression_elimination
from tbps.database import PostgresRetriever, RetrievedCandidate
from tbps.expr import deserialize_expr
from tbps.lean_extractor import extract_lean_expression
from tbps.scoring import (
    CandidateScore,
    ScoreSettings,
    aggregate_metrics,
    apply_margin_gate,
    competition_rank,
    score_candidate_profiled,
)
from tbps.tree import TreeNode, expr_to_tree, simplify_forall, tree_node_count


@dataclass(frozen=True)
class QueryCase:
    query_id: str
    benchmark: str
    target: str
    expression_json: object | None = None
    expression_text: str | None = None
    query_text: str | None = None
    source_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class PreparedCandidate:
    name: str
    tree: TreeNode
    wl_score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0


_WORKER_TARGET_TREE: TreeNode | None = None
_WORKER_TARGET_SIZE = 0
_WORKER_SCORE_SETTINGS: ScoreSettings | None = None


def _initialize_score_worker(
    target_tree: TreeNode, target_size: int, settings: ScoreSettings
) -> None:
    global _WORKER_TARGET_TREE, _WORKER_TARGET_SIZE, _WORKER_SCORE_SETTINGS
    _WORKER_TARGET_TREE = target_tree
    _WORKER_TARGET_SIZE = target_size
    _WORKER_SCORE_SETTINGS = settings


def _score_worker(candidate: PreparedCandidate) -> tuple[CandidateScore, dict[str, float]]:
    if _WORKER_TARGET_TREE is None or _WORKER_SCORE_SETTINGS is None:
        raise RuntimeError("score worker was not initialized")
    return score_candidate_profiled(
        candidate.name,
        candidate.tree,
        candidate.wl_score,
        _WORKER_TARGET_TREE,
        _WORKER_TARGET_SIZE,
        _WORKER_SCORE_SETTINGS,
        dense=candidate.dense_score,
        bm25=candidate.bm25_score,
    )


def _prepare_candidate(candidate: RetrievedCandidate) -> PreparedCandidate:
    expression = deserialize_expr(candidate.expression_json)
    tree = expr_to_tree(simplify_forall(expression))
    return PreparedCandidate(
        candidate.name,
        tree,
        candidate.wl_score,
        candidate.dense_score or 0.0,
        candidate.bm25_score or 0.0,
    )


def _build_hybrid_retriever(wl_retriever: PostgresRetriever, config: HybridConfig) -> "object":
    """Build a ``HybridRetriever`` when the I3 hybrid hybrid config is enabled.

    Loads the BM25 index and (if enabled) the dense index + encoder. The dense encoder
    imports torch lazily inside ``DenseEncoder``, so enabling hybrid without the ``dense``
    extra fails loudly here rather than at import time.
    """
    from tbps.hybrid import HybridRetriever  # lazy import; guards torch-free baseline

    bm25_index = None
    dense_index = None
    dense_encoder = None
    if "bm25" in config.retrievers:
        if config.bm25_index is None:
            raise ValueError("hybrid enabled with bm25 retriever but no bm25_index path")
        from tbps.retrieval.bm25 import BM25Index

        bm25_index = BM25Index.load(config.bm25_index)
    if "dense" in config.retrievers:
        if config.dense_index_dir is None:
            raise ValueError("hybrid enabled with dense retriever but no dense_index_dir")
        from tbps.retrieval.dense import DenseEncoder, DenseIndex

        dense_index = DenseIndex.load(config.dense_index_dir)
        dense_encoder = DenseEncoder(
            model_name=config.dense_model,
            device=config.dense_device,
            max_length=config.dense_max_length,
        )
    return HybridRetriever(
        wl_retriever=wl_retriever,
        bm25_index=bm25_index,
        dense_index=dense_index,
        dense_encoder=dense_encoder,
        config=config,
    )


class BaselineRunner:
    def __init__(self, config: RunnerConfig, *, workers: int | None = None):
        self.config = config
        self.workers = config.workers if workers is None else workers
        if self.workers < 1:
            raise ValueError("workers must be at least one")
        self.retriever = PostgresRetriever(config.database, config.retrieval)
        hybrid = config.hybrid
        if hybrid is not None and hybrid.enabled:
            self.retriever = _build_hybrid_retriever(self.retriever, hybrid)
            # I3 dense space-mismatch fix: load the clean-query cache (query_id → clean Lean
            # statement text) so dense embeds the CLEAN text, matching the clean doc space. Test B
            # `state_text` is already clean, so it passes through unchanged (cache miss → raw text).
            self._clean_query_map: dict[str, str] = {}
            if hybrid.clean_query_cache is not None:
                cache_path = hybrid.clean_query_cache
                if cache_path.exists():
                    import json as _json

                    with cache_path.open(encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            rec = _json.loads(line)
                            qid = rec.get("query_id")
                            clean = rec.get("clean_text")
                            if qid and clean:
                                self._clean_query_map[qid] = clean
                    print(
                        f"loaded {len(self._clean_query_map)} clean query texts from {cache_path}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"warning: clean_query_cache {cache_path} does not exist; "
                        f"dense will use the raw query_text (mismatched space)",
                        file=sys.stderr,
                    )

    def run_query(self, case: QueryCase) -> dict[str, Any]:
        started = time.perf_counter()
        rss_before = _rss_mb()
        stages: dict[str, float] = {}
        stage_started = started
        expression_json = case.expression_json
        if expression_json is None:
            if case.expression_text is None:
                raise ValueError("query has neither expression JSON nor expression text")
            expression_json = extract_lean_expression(case.expression_text)
        expression = deserialize_expr(expression_json)
        cse_expression = common_subexpression_elimination(expression)
        raw_target_tree = expr_to_tree(cse_expression)
        raw_target_size = tree_node_count(raw_target_tree)
        target_tree = expr_to_tree(simplify_forall(cse_expression))
        stages["deserialize_cse_tree"] = time.perf_counter() - stage_started

        cascade = self.config.cascade
        cascade_payload: dict[str, object] | None = None
        candidate_limit_override: int | None = None
        if cascade is not None and cascade.enabled:
            k = choose_K(raw_target_size, cascade)
            candidate_limit_override = k
            cascade_payload = {
                "k": k,
                "policy": cascade.policy,
                "raw_target_size": raw_target_size,
            }

        stage_started = time.perf_counter()
        retrieval_kwargs: dict[str, Any] = {
            "target_name": case.target,
            "target_tree": target_tree,
            "ratio_node_count": raw_target_size,
            "candidate_limit_override": candidate_limit_override,
        }
        # I3 hybrid: the HybridRetriever additionally needs the query text (dense) and the
        # elaborated expr JSON (BM25 lexical tokens). The plain PostgresRetriever ignores
        # these extra kwargs, so passing them unconditionally keeps both paths uniform.
        # Dense space-mismatch fix: prefer the CLEAN query text (from the clean-query cache, in
        # the same Lean-pp space as the clean doc index) over the raw `q(...)` query_text. Test B
        # `state_text` is already clean → cache miss falls through to it unchanged.
        if self.config.hybrid is not None and self.config.hybrid.enabled:
            retrieval_kwargs["query_text"] = self._clean_query_map.get(
                case.query_id, case.query_text
            )
            retrieval_kwargs["expression_json"] = expression_json
        retrieval = self.retriever.retrieve(**retrieval_kwargs)
        stages["node_filter_wl_retrieval"] = time.perf_counter() - stage_started
        stages.update(retrieval.duration_seconds)

        stage_started = time.perf_counter()
        prepared: list[PreparedCandidate] = []
        candidate_exceptions: list[dict[str, str]] = []
        for candidate in retrieval.candidates:
            try:
                prepared.append(_prepare_candidate(candidate))
            except (
                Exception
            ) as error:  # A bad corpus row is reported and skipped, as upstream does.
                candidate_exceptions.append({"name": candidate.name, "exception": repr(error)})
        stages["candidate_deserialize_tree"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        if self.workers == 1 or len(prepared) < 2:
            profiled_scores = [
                score_candidate_profiled(
                    candidate.name,
                    candidate.tree,
                    candidate.wl_score,
                    target_tree,
                    raw_target_size,
                    self.config.scoring,
                    dense=candidate.dense_score,
                    bm25=candidate.bm25_score,
                )
                for candidate in prepared
            ]
        else:
            with ProcessPoolExecutor(
                max_workers=self.workers,
                initializer=_initialize_score_worker,
                initargs=(target_tree, raw_target_size, self.config.scoring),
            ) as executor:
                profiled_scores = list(executor.map(_score_worker, prepared, chunksize=8))
        scores = [score for score, _ in profiled_scores]
        component_cpu_seconds = {
            component: sum(timing[component] for _, timing in profiled_scores)
            for component in ("teds", "jaccard", "collapse_match")
        }
        # I3 margin-gated dense fusion: per-query adaptive dense weight gated on the
        # structural leader's margin. No-op when margin_gate is None/disabled.
        large_tree = raw_target_size > self.config.scoring.tree_score_cutoff
        scores = apply_margin_gate(scores, settings=self.config.scoring, large=large_tree)
        ranked = competition_rank(scores)
        stages["component_scores_fusion_rank"] = time.perf_counter() - stage_started

        target_score = next((score for score in ranked if score.name == case.target), None)
        top = ranked[: self.config.output_top_k]
        target_in_candidates = any(candidate.name == case.target for candidate in prepared)
        rss_after = _rss_mb()
        return {
            "query_id": case.query_id,
            "benchmark": case.benchmark,
            "target": case.target,
            "candidate_count_before_filter": retrieval.corpus_count,
            "candidate_count_after_node_filter": retrieval.node_filtered_count,
            "candidate_count_with_wl": retrieval.joined_count,
            "candidate_count_after_top_k": len(retrieval.candidates),
            "candidate_count_scored": len(ranked),
            "cascade": cascade_payload,
            "target_coverage": {
                "mathlib_filtered": retrieval.target_in_corpus,
                "wl_encodings_new": retrieval.target_has_wl,
                "top_k_candidates": target_in_candidates,
            },
            "target_component_scores": _score_payload(target_score),
            "final_rank": None if target_score is None else target_score.rank,
            "top_k": [_score_payload(score) for score in top],
            "node_filter": {
                "query_nodes_before_simplify": raw_target_size,
                "query_nodes_after_simplify": tree_node_count(target_tree),
                "minimum": retrieval.minimum_nodes,
                "maximum": retrieval.maximum_nodes,
            },
            "duration_seconds": {**stages, "total": time.perf_counter() - started},
            "component_cpu_seconds": component_cpu_seconds,
            "memory_rss_mb": {
                "before": rss_before,
                "after": rss_after,
                "delta": _optional_difference(rss_after, rss_before),
            },
            "candidate_exceptions": candidate_exceptions,
            "exception": None,
            "source_metadata": case.source_metadata,
        }


def _score_payload(score: CandidateScore | None) -> dict[str, Any] | None:
    return asdict(score) if score is not None else None


def load_test_b(path: Path) -> Iterator[QueryCase]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            yield QueryCase(
                query_id=record["query_id"],
                benchmark="test-b",
                target=record["theorem"],
                expression_json=record["state"],
                query_text=record.get("state_text"),
                source_metadata={
                    "manifest_line": line_number,
                    "source_row_id": record.get("source_row_id"),
                    "availability_reason": record.get("availability_reason"),
                },
            )


def load_test_a_cases(config: RunnerConfig) -> Iterator[QueryCase]:
    benchmark_dir = config.benchmark.test_a_query_file.parent
    for index, case in enumerate(load_test_a(benchmark_dir), 1):
        yield QueryCase(
            query_id=f"test-a-{index:03d}",
            benchmark="test-a",
            target=case.premise_name,
            expression_text=case.expression,
            query_text=case.expression,
            source_metadata={"benchmark_line": index},
        )


def run_batch(
    cases: Iterator[QueryCase],
    runner: BaselineRunner,
    output: Path,
    *,
    start: int = 0,
    limit: int | None = None,
    resume: bool = False,
    progress_callback: Callable[[dict[str, Any], int], None] | None = None,
) -> tuple[int, int]:
    if start < 0:
        raise ValueError("start must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    completed = _completed_query_ids(output) if resume else set()
    if output.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = (case for index, case in enumerate(cases) if index >= start)
    written = 0
    skipped = 0
    considered = 0
    metadata = _provenance(runner.config, runner.workers)
    with output.open("a", encoding="utf-8") as stream:
        for case in selected:
            if limit is not None and considered >= limit:
                break
            considered += 1
            if case.query_id in completed:
                skipped += 1
                continue
            started = time.perf_counter()
            try:
                record = runner.run_query(case)
            except Exception as error:
                record = {
                    "query_id": case.query_id,
                    "benchmark": case.benchmark,
                    "target": case.target,
                    "candidate_count_before_filter": None,
                    "candidate_count_after_node_filter": None,
                    "candidate_count_with_wl": None,
                    "candidate_count_after_top_k": None,
                    "candidate_count_scored": 0,
                    "cascade": None,
                    "target_coverage": None,
                    "target_component_scores": None,
                    "final_rank": None,
                    "top_k": [],
                    "duration_seconds": {"total": time.perf_counter() - started},
                    "component_cpu_seconds": None,
                    "memory_rss_mb": None,
                    "candidate_exceptions": [],
                    "exception": repr(error),
                    "source_metadata": case.source_metadata,
                }
            record["provenance"] = metadata
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            written += 1
            if progress_callback is not None:
                progress_callback(record, written)
    return written, skipped


def _completed_query_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                query_id = json.loads(line)["query_id"]
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"invalid resume output at line {line_number}: {path}") from error
            if query_id in completed:
                raise ValueError(f"duplicate query_id in resume output: {query_id}")
            completed.add(query_id)
    return completed


def _provenance(config: RunnerConfig, workers: int) -> dict[str, Any]:
    config_bytes = config.path.read_bytes()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "run_name": config.name,
        "git_commit": commit,
        "mathlib_sha": config.mathlib_sha,
        "config": str(config.path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "fusion_profile": config.fusion_profile,
        "tie_method": config.tie_method,
        "seed": config.seed,
        "workers": workers,
    }


def summarize(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return aggregate_metrics(records)


def _rss_mb() -> float | None:
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return None


def _optional_difference(value: float | None, baseline: float | None) -> float | None:
    return None if value is None or baseline is None else value - baseline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auditable Tree-Based Premise Selection runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for benchmark in ("test-a", "test-b"):
        command = subparsers.add_parser(benchmark)
        command.add_argument("--config", type=Path, default=Path("configs/baseline-paper.toml"))
        command.add_argument("--start", type=int, default=0)
        command.add_argument("--limit", type=int)
        command.add_argument("--resume", action="store_true")
        command.add_argument("--workers", type=int)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--fusion-profile")
        command.add_argument(
            "--output-top-k",
            type=int,
            help=(
                "Override the config's output_top_k (how many ranked candidates are saved "
                "per query in top_k). Set this large (e.g. 1000) to produce a WIDE-SAVE, "
                "which the Stage-3 kernel reranker needs as input (it re-ranks the full "
                "saved pool). Without it the config default (10) is used."
            ),
        )
        if benchmark == "test-a":
            command.add_argument(
                "--expression-cache",
                type=Path,
                help=(
                    "Path to a Test A expression-extraction cache (JSONL). If the file "
                    "exists, expressions are loaded from it (skipping ~120 s/query of "
                    "Lean extraction); if not, it is built once via batched extraction "
                    "and reused by later runs. Combine with --resume to incrementally "
                    "fill a partially-built cache."
                ),
            )
    # Stage-3 kernel rerank: re-rank a wide-save JSONL with the Lean kernel (leader-protect
    # + boost applicable non-leaders). Requires [kernel] enabled in the config.
    kernel = subparsers.add_parser(
        "kernel-rerank",
        help="Re-rank a wide-save JSONL with the Lean kernel reranker (Stage 3).",
    )
    kernel.add_argument("--config", type=Path, default=Path("configs/baseline-paper.toml"))
    kernel.add_argument("--input", type=Path, required=True, help="wide-save JSONL to re-rank")
    kernel.add_argument(
        "--output",
        type=Path,
        help="output kernel-reranked JSONL (default: <input>.kernel.jsonl)",
    )
    kernel.add_argument("--fusion-profile")
    kernel.add_argument("--resume", action="store_true")
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--input", type=Path, required=True)
    summary.add_argument("--output", type=Path)
    return parser


def _load_test_a_cases(config: RunnerConfig, args: argparse.Namespace):
    """Build the Test A case list, optionally via a shared extraction cache.

    Without ``--expression-cache`` this falls back to the legacy per-query extraction
    (each query pays the ~120 s ``import Mathlib`` startup inside ``run_query``). With
    a cache path: load it if it already exists (fast path, no extraction), otherwise
    build it once via batched extraction and reuse it on every later run.
    """
    cache_path: Path | None = getattr(args, "expression_cache", None)
    if cache_path is None:
        return load_test_a_cases(config)
    # Lazy import so `tbps.runner` does not depend on `tbps.extraction_cache` at
    # module load time (extraction_cache imports back from runner).
    from tbps.extraction_cache import extract_test_a_cache, load_test_a_cache

    if cache_path.exists():
        print(f"loading Test A expression cache: {cache_path}", file=sys.stderr)
        return load_test_a_cache(config, cache_path)
    print(f"building Test A expression cache: {cache_path}", file=sys.stderr)
    return extract_test_a_cache(config, cache_path, resume=args.resume)


def main() -> None:
    args = _parser().parse_args()
    if args.command == "summarize":
        result = summarize(args.input)
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            if args.output.exists():
                raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        return

    if args.command == "kernel-rerank":
        # Stage 3: re-rank a wide-save JSONL with the Lean kernel (leader-protect + boost).
        # Lazy import so the baseline venv (no Lean tooling at runtime) is unaffected.
        from tbps.kernel_rerank import run_kernel_rerank

        config = load_config(args.config, fusion_profile=args.fusion_profile)
        if config.kernel is None or not config.kernel.enabled:
            raise SystemExit(
                f"[kernel] is not enabled in {args.config}; cannot run kernel-rerank. "
                f"Add a [kernel] section with enabled=true (see "
                f"configs/baseline-paper-hybrid-kernel.toml)."
            )
        run_kernel_rerank(args.input, args.output, config, resume=args.resume)
        return

    config = load_config(args.config, fusion_profile=args.fusion_profile)
    # --output-top-k override: produce a wide-save (large top_k) for Stage-3 input.
    if getattr(args, "output_top_k", None) is not None:
        config = dataclasses_replace(config, output_top_k=args.output_top_k)
    runner = BaselineRunner(config, workers=args.workers)
    if args.command == "test-a":
        cases = _load_test_a_cases(config, args)
    else:
        cases = load_test_b(config.benchmark.test_b_manifest)
    written, skipped = run_batch(
        cases,
        runner,
        args.output,
        start=args.start,
        limit=args.limit,
        resume=args.resume,
    )
    print(json.dumps({"written": written, "resume_skipped": skipped, "output": str(args.output)}))

    # Auto-chain Stage 3 (kernel rerank) when [kernel] is enabled. Stage 1+2 (the file just
    # written) is byte-identical to the baseline; Stage 3 reads it and writes
    # <output>.kernel.jsonl. Requires the Stage-1+2 output to be a wide-save
    # (output_top_k >= kernel.top_n), validated inside run_kernel_rerank.
    if config.kernel is not None and config.kernel.enabled:
        from tbps.kernel_rerank import run_kernel_rerank

        kernel_output = args.output.parent / (args.output.stem + ".kernel.jsonl")
        print(
            f"[kernel] enabled — auto-chaining Stage 3 → {kernel_output}",
            file=sys.stderr,
        )
        run_kernel_rerank(args.output, kernel_output, config, resume=args.resume)


if __name__ == "__main__":
    main()
