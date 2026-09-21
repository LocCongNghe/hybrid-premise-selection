"""Lean kernel applicability reranker (Stage 3).

For each query, ask Lean whether each candidate theorem's generalized conclusion
(universal binders turned into metavariables) is definitionally equal (``isDefEq``)
to the query's generalized body. Applicable non-leaders get a constant ``bonus``
added to their structural score; the structural leader set is pinned at rank 1.

Requires Lean+Mathlib (WSL Ubuntu). Reads a Stage-1+2 JSONL output, probes the
top-N candidates per query in ONE Lean process, then re-ranks each record and
writes a new kernel-reranked JSONL.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path

from tbps.config import RunnerConfig
from tbps.scoring import FusionWeights, aggregate_metrics

# Lean/Mathlib run in WSL Ubuntu. When this module is invoked from Windows (PowerShell),
# the Lean process must be started via ``wsl.exe``; when running inside WSL, ``wsl.exe`` is
# unavailable and we call ``lake`` directly.
IN_WSL = platform.system() == "Linux"

ROOT = Path(__file__).resolve().parents[2]
LEAN_DIR = ROOT / "lean"


# --------------------------------------------------------------------------- #
# Requests / results TSV
# --------------------------------------------------------------------------- #
def write_requests(queries: Sequence[dict], req_path: Path) -> int:
    """Write TSV: ``qidx \\t term \\t cand1,cand2,...`` (one line per query).

    ``term`` may contain spaces (Lean terms do) but not tabs/newlines (sanitized). The
    Lean side auto-detects a Test B ``state`` JSON payload (field starts with ``{``) vs a
    Test A term string.
    """
    lines: list[str] = []
    for q in queries:
        names = ",".join(c["name"] for c in q["cands"])
        term = q["term"].replace("\t", " ").replace("\n", " ")
        lines.append(f"{q['qidx']}\t{term}\t{names}")
    req_path.parent.mkdir(parents=True, exist_ok=True)
    req_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def parse_results(res_path: Path) -> dict[str, dict[str, bool | None]]:
    """Return ``{qidx: {cand_name: applicable_bool_or_None}}``.

    ``applicable`` is ``"1"``→True, ``"0"``→False, empty→None (unknown / error / not-found).
    """
    out: dict[str, dict[str, bool | None]] = {}
    if not res_path.exists():
        return out
    for line in res_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        qidx, cand, appl = parts[0], parts[1], parts[2]
        val: bool | None
        if appl == "1":
            val = True
        elif appl == "0":
            val = False
        else:
            val = None
        out.setdefault(qidx, {})[cand] = val
    return out


# --------------------------------------------------------------------------- #
# Lean invocation
# --------------------------------------------------------------------------- #
def _to_wsl(p: Path) -> str:
    """Map a Path to its WSL-visible POSIX form (``D:/foo`` → ``/mnt/d/foo``)."""
    s = str(p).replace("\\", "/")
    if s.startswith("/"):
        return s  # already POSIX (script running in WSL)
    return "/mnt/" + s.split(":", 1)[1].lstrip("/")


def _completed_qidxs(res_path: Path) -> set[str]:
    """Query IDs that already have ≥1 result line in ``res_path`` (for resume)."""
    done: set[str] = set()
    if not res_path.exists():
        return done
    with res_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                done.add(parts[0])
    return done


def run_lean(mode: str, req_path: Path, res_path: Path, timeout: int = 7200) -> bool:
    """Invoke the Lean kernel probe in WSL Ubuntu.

    Writes a scratch driver ``.lean`` that ``import TBPS.KernelProbe``s and invokes the
    ``tbps_kernel_probe`` command, then compiles it with ``lake env lean``. Returns True
    on clean completion, False on timeout / hard failure (partial results are preserved
    by the driver's periodic flush).
    """
    lean_dir = LEAN_DIR
    req_wsl = _to_wsl(req_path)
    res_wsl = _to_wsl(res_path)
    token = uuid.uuid4().hex[:8]
    scratch_name = f".kp-run-{token}.lean"
    scratch_path = lean_dir / scratch_name
    scratch_content = (
        "import TBPS.KernelProbe\n"
        # Global heartbeat ceiling: bounds each isDefEq to ~8s; a hang throws
        # runtime.maxHeartbeats, caught by the probe's try/catch (→ unknown verdict).
        "set_option maxHeartbeats 8000000\n"
        f"tbps_kernel_probe {json.dumps(mode)} {json.dumps(req_wsl)} {json.dumps(res_wsl)}\n"
    )
    scratch_path.write_text(scratch_content, encoding="utf-8")
    try:
        cmd = f"cd {_to_wsl(LEAN_DIR)} && lake env lean {scratch_name}"
        print(f"  [kernel] running: {cmd}", flush=True)
        if IN_WSL:
            argv = ["bash", "-lc", cmd]
        else:
            argv = ["wsl.exe", "-d", "Ubuntu", "-e", "bash", "-lc", cmd]
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            # The Lean process hung; the driver's periodic flush preserved partial
            # results. Return False so the resumable loop skips the hanging query.
            print(
                f"  [kernel] TIMEOUT after {timeout}s (pathological reduction); "
                f"partial results preserved",
                flush=True,
            )
            # Kill any leftover `lake`/`lean` child so the next iteration starts clean.
            kill_argv = (
                [
                    "bash",
                    "-lc",
                    "pkill -9 -f 'kp-run' 2>/dev/null; "
                    "pkill -9 -f 'lake env lean' 2>/dev/null; true",
                ]
                if IN_WSL
                else [
                    "wsl.exe",
                    "-d",
                    "Ubuntu",
                    "-e",
                    "bash",
                    "-lc",
                    "pkill -9 -f 'kp-run' 2>/dev/null; "
                    "pkill -9 -f 'lake env lean' 2>/dev/null; true",
                ]
            )
            subprocess.run(kill_argv, capture_output=True, timeout=30, check=False)
            return False
        if proc.returncode != 0:
            print(f"  [kernel] FAILED rc={proc.returncode}", flush=True)
            print(proc.stderr[-3000:] if proc.stderr else "(no stderr)", flush=True)
            print(proc.stdout[-3000:] if proc.stdout else "(no stdout)", flush=True)
            return False
        if proc.stdout:
            tail = proc.stdout.strip().splitlines()[-6:]
            for ln in tail:
                print(f"  [kernel] {ln}", flush=True)
        return True
    finally:
        # remove scratch driver + the .olean/.ilean lake produces beside it
        stem = scratch_path.stem  # .kp-run-{token}
        for ext in (".lean", ".olean", ".ilean", ".trace", ".c"):
            (lean_dir / f"{stem}{ext}").unlink(missing_ok=True)


def run_lean_resumable(
    queries: Sequence[dict],
    mode: str,
    req_path: Path,
    res_path: Path,
    timeout: int = 600,
    batch: int = 0,
) -> bool:
    """Run the Lean probe with wall-clock timeout + resume.

    Each Lean process is bounded by wall-clock ``timeout``; the driver flushes results
    periodically, so on a timeout the partial results are preserved and the single
    hanging query is skipped while the rest resume in a fresh process. Resume is
    durable: ``res_path`` accumulates across invocations. ``batch`` (>0) caps queries
    per Lean process; 0 = no cap.
    """
    qidx_to_query = {q["qidx"]: q for q in queries}
    done = _completed_qidxs(res_path)
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.touch(exist_ok=True)
    todo = [q for q in queries if q["qidx"] not in done]
    print(
        f"  [kernel-resume] {len(done)} done, {len(todo)} todo (timeout={timeout}s/batch)",
        flush=True,
    )
    iteration = 0
    while todo:
        iteration += 1
        chunk = todo if not batch else todo[:batch]
        chunk_req = req_path.with_suffix(f".chunk{iteration}.tsv")
        chunk_res = res_path.with_suffix(f".chunk{iteration}.tsv")
        write_requests(chunk, chunk_req)
        ok = run_lean(mode, chunk_req, chunk_res, timeout=timeout)
        done_before = len(done)
        # merge whatever chunk_res produced into the main res_path (append)
        if chunk_res.exists() and chunk_res.stat().st_size > 0:
            with (
                chunk_res.open(encoding="utf-8") as src,
                res_path.open("a", encoding="utf-8") as dst,
            ):
                for line in src:
                    dst.write(line)
        chunk_req.unlink(missing_ok=True)
        chunk_res.unlink(missing_ok=True)
        done = _completed_qidxs(res_path)
        progressed = len(done) > done_before
        if ok and progressed:
            todo = [q for q in queries if q["qidx"] not in done]
            print(
                f"  [kernel-resume] iter {iteration}: process completed cleanly; "
                f"{len(done)} done, {len(todo)} todo",
                flush=True,
            )
            continue
        # TIMEOUT, hard failure, or rc==0 with zero progress: the first chunk member
        # not yet done is the culprit — emit empty (unknown) results for it so the run
        # advances instead of looping on the same chunk forever.
        hanging = next((q["qidx"] for q in chunk if q["qidx"] not in done), None)
        if hanging is not None:
            with res_path.open("a", encoding="utf-8") as dst:
                for c in qidx_to_query[hanging]["cands"]:
                    dst.write(f"{hanging}\t{c['name']}\t\t{mode}\n")
            why = (
                "rc=0/0-progress (heartbeat exception aborted loop)"
                if (ok and not progressed)
                else ("TIMEOUT" if not ok else "FAIL")
            )
            print(
                f"  [kernel-resume] iter {iteration}: {why} — skipping query "
                f"{hanging} (emitted unknown)",
                flush=True,
            )
        done = _completed_qidxs(res_path)
        todo = [q for q in queries if q["qidx"] not in done]
        print(f"  [kernel-resume] {len(done)} done, {len(todo)} todo", flush=True)
    return True


# --------------------------------------------------------------------------- #
# Query loading (from Stage-1+2 records + benchmark sources)
# --------------------------------------------------------------------------- #
def _cands_with_target(record: dict, top_n: int, include_target: bool = False) -> list[dict]:
    """Top-N candidates by saved final rank; optionally append the target if absent.

    ``include_target=False`` (the default) probes exactly the natural top-N.
    ``include_target=True`` appends the gold target when it is absent (measurement-only
    mode for applicability statistics; never use it for reported ranking metrics).
    """
    cands = sorted(record.get("top_k", []), key=lambda c: c.get("rank", 9999))
    top = cands[:top_n]
    if not include_target:
        return top
    names = {c["name"] for c in top}
    target = record["target"]
    if target not in names:
        tgt_cand = next((c for c in cands if c["name"] == target), None)
        if tgt_cand is None:
            tcs = record.get("target_component_scores")
            if tcs is not None and tcs.get("name") == target:
                tgt_cand = tcs
        if tgt_cand is not None:
            top = top + [tgt_cand]
    return top


def load_kernel_queries(
    records: Sequence[dict], config: RunnerConfig, top_n: int, include_target: bool = False
) -> list[dict]:
    """Build ``{qidx, term, target, cands, raw_target_size}`` for each record.

    Test A: ``term`` = the raw term string from ``expressions.txt``, indexed by
    ``source_metadata.benchmark_line`` (1-based). Test B: ``term`` = the compact
    ``state`` JSON from the manifest, keyed by ``query_id``. Records whose expression
    can't be resolved are skipped (their ``final_rank`` is left unchanged).
    """
    out: list[dict] = []
    for record in records:
        benchmark = record.get("benchmark")
        qidx = record["query_id"]
        target = record["target"]
        term = ""
        if benchmark == "test-b":
            states = _test_b_states(config)
            term = states.get(qidx, "")
        else:
            # test-a: index expressions.txt by the 1-based benchmark_line
            line = (record.get("source_metadata") or {}).get("benchmark_line")
            terms = _test_a_terms(config)
            if isinstance(line, int) and 1 <= line <= len(terms):
                term = terms[line - 1]
        if not term:
            continue
        cands = _cands_with_target(record, top_n, include_target=include_target)
        if not cands:
            continue
        node_filter = record.get("node_filter") or {}
        out.append(
            {
                "qidx": qidx,
                "term": term,
                "target": target,
                "cands": cands,
                "raw_target_size": node_filter.get("query_nodes_before_simplify", 0),
            }
        )
    return out


_TEST_A_TERMS_CACHE: list[str] | None = None
_TEST_A_TERMS_CONFIG_PATH: Path | None = None
_TEST_B_STATES_CACHE: dict[str, str] | None = None
_TEST_B_STATES_CONFIG_PATH: Path | None = None


def _test_a_terms(config: RunnerConfig) -> list[str]:
    """Load (and cache) Test A term strings from the configured expressions file."""
    global _TEST_A_TERMS_CACHE, _TEST_A_TERMS_CONFIG_PATH
    path = config.benchmark.test_a_query_file
    if _TEST_A_TERMS_CACHE is None or _TEST_A_TERMS_CONFIG_PATH != path:
        _TEST_A_TERMS_CACHE = path.read_text(encoding="utf-8").splitlines()
        _TEST_A_TERMS_CONFIG_PATH = path
    return _TEST_A_TERMS_CACHE


def _test_b_states(config: RunnerConfig) -> dict[str, str]:
    """Load (and cache) Test B ``query_id`` → compact ``state`` JSON from the manifest."""
    global _TEST_B_STATES_CACHE, _TEST_B_STATES_CONFIG_PATH
    path = config.benchmark.test_b_manifest
    if _TEST_B_STATES_CACHE is None or _TEST_B_STATES_CONFIG_PATH != path:
        states: dict[str, str] = {}
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = json.loads(line)
                states[m["query_id"]] = json.dumps(
                    m["state"], ensure_ascii=False, separators=(",", ":")
                )
        _TEST_B_STATES_CACHE = states
        _TEST_B_STATES_CONFIG_PATH = path
    return _TEST_B_STATES_CACHE


# --------------------------------------------------------------------------- #
# Rerank (the testable, Lean-free core)
# --------------------------------------------------------------------------- #
def _struct_score(cand: dict, weights: FusionWeights) -> float:
    """Structural-only score (dense/bm25 weight forced to 0); ``teds`` None → 0.0."""
    teds = 0.0 if cand.get("teds") is None else cand.get("teds", 0.0)
    return (
        weights.wl * cand.get("wl", 0.0)
        + weights.teds * teds
        + weights.collapse_match * cand.get("collapse_match", 0.0)
        + weights.jaccard * cand.get("jaccard", 0.0)
    )


def _appl_to_float(appl: bool | None) -> float | None:
    """Serialize applicability for the output ``kernel`` field: True→1.0, False→0.0,
    None→null (unknown). Preserves the three-valued signal."""
    if appl is True:
        return 1.0
    if appl is False:
        return 0.0
    return None


def rerank_record(
    record: dict,
    kernel_map: dict[str, bool | None],
    weights: FusionWeights,
    bonus: float,
    save_top_k: int | None = None,
) -> dict:
    """Leader-protect kernel rerank of ONE Stage-1+2 record (full-pool).

    Operates on the FULL saved pool (``record["top_k"]``) plus the target recovered from
    ``target_component_scores`` when it sits below the saved pool (rank bookkeeping only —
    the appended target gets no boost unless it was inside the probed top-N). Only
    candidates present in ``kernel_map`` can receive a boost; every other candidate keeps
    its structural score.

    Leaders (max struct) are pinned at the top in name order; non-leaders are sorted by
    ``struct + (bonus if applicable)`` with ``name`` as tie-break; exact-float competition
    ranks are assigned over the full pool. When ``kernel_map`` is empty/all-None the
    rerank is a no-op (byte-identical ranking to Stage 1+2).
    """
    target = record["target"]
    # Full saved pool + target (recovered from target_component_scores if absent).
    pool = _cands_with_target(record, len(record.get("top_k", [])), include_target=True)

    if not pool:
        # nothing to rerank (empty top_k + no recoverable target); stamp provenance + leave
        # final_rank as-is.
        return _stamp_kernel_meta(record, bonus, applied=False)

    structs = [(_struct_score(c, weights), c) for c in pool]
    smax = max(s for s, _ in structs)

    leaders: list[tuple[float, dict]] = []
    nonleaders: list[tuple[float, dict]] = []
    for s, c in structs:
        appl = kernel_map.get(c["name"])
        if s == smax:
            leaders.append((s, _with_kernel(c, appl)))
        else:
            boost = bonus if appl is True else 0.0
            nonleaders.append((s + boost, _with_kernel(c, appl, final_override=s + boost)))

    leaders.sort(key=lambda sc: (-sc[0], sc[1]["name"]))
    nonleaders.sort(key=lambda sc: (-sc[0], sc[1]["name"]))
    ordered = leaders + nonleaders

    # assign exact-float competition ranks (1, 2, 2, 4 …) over the FULL pool.
    ranked: list[dict] = []
    previous: float | None = None
    rank = 1
    for index, (sort_key, c) in enumerate(ordered):
        if sort_key != previous:
            rank = index + 1
        ranked.append(_with_rank(c, rank))
        previous = sort_key

    # target's new rank in the full pool (None if the target is a pool-miss).
    target_rank: int | None = next((c["rank"] for c in ranked if c["name"] == target), None)

    new_top = ranked if save_top_k is None else ranked[:save_top_k]
    # stamp the target_component_scores with the kernel signal + its new full-pool rank.
    tcs = record.get("target_component_scores")
    if tcs is not None and tcs.get("name") == target:
        tcs = {
            **tcs,
            "kernel": _appl_to_float(kernel_map.get(target)),
            "rank": target_rank,
        }

    return {
        **record,
        "top_k": new_top,
        "target_component_scores": tcs,
        "final_rank": target_rank,
        "kernel": {"bonus": bonus, "applied": True},
    }


def _with_kernel(cand: dict, appl: bool | None, final_override: float | None = None) -> dict:
    """Return a copy of ``cand`` with the ``kernel`` field stamped and optionally an
    overridden ``final`` (the sort key used for ranking). For leaders ``final`` is left
    unchanged (== struct); for boosted non-leaders ``final`` = struct + bonus."""
    out = {**cand, "kernel": _appl_to_float(appl)}
    if final_override is not None:
        out["final"] = final_override
    return out


def _with_rank(cand: dict, rank: int) -> dict:
    return {**cand, "rank": rank}


def _stamp_kernel_meta(record: dict, bonus: float, *, applied: bool) -> dict:
    return {**record, "kernel": {"bonus": bonus, "applied": applied}}


# --------------------------------------------------------------------------- #
# Stage-3 driver
# --------------------------------------------------------------------------- #
def _kernel_output_path(input_path: Path) -> Path:
    """``<stem>.kernel.jsonl`` beside the input (refuse-to-overwrite is enforced by the
    caller)."""
    return input_path.parent / (input_path.stem + ".kernel.jsonl")


def run_kernel_rerank(
    input_path: Path,
    output_path: Path | None,
    config: RunnerConfig,
    *,
    resume: bool = True,
) -> dict:
    """Full Stage-3 driver: read Stage-1+2 JSONL → run Lean kernel → rerank → write JSONL.

    Runs the Lean kernel probe over the batch (resumable), re-ranks each record,
    recomputes ``final_rank``, and writes ``output_path`` (or ``<input>.kernel.jsonl``
    when None). Refuses to overwrite an existing output unless ``resume``. Returns
    aggregate metrics over the reranked records.
    """
    kernel = config.kernel
    if kernel is None or not kernel.enabled:
        raise ValueError("[kernel] is not enabled in the config; cannot run Stage 3")

    if output_path is None:
        output_path = _kernel_output_path(input_path)

    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"no records in input: {input_path}")

    # Validate the input is a WIDE-SAVE: the rerank needs the full saved pool (not just
    # the final output_top_k) to compute full-pool final_rank.
    max_pool = max((len(r.get("top_k", [])) for r in records), default=0)
    if max_pool < kernel.top_n:
        raise ValueError(
            f"[kernel] input {input_path} is not a wide-save (max top_k={max_pool} < "
            f"kernel.top_n={kernel.top_n}). Stage 3 needs a wide-save (run the benchmark "
            f"with output_top_k >= kernel.top_n, e.g. 1000) so it can re-rank the full pool."
        )
    print(
        f"[kernel] input wide-save: {len(records)} records, max pool={max_pool}",
        file=sys.stderr,
    )

    queries = load_kernel_queries(records, config, kernel.top_n, kernel.include_target)
    print(
        f"[kernel] {len(queries)}/{len(records)} records have resolvable expressions "
        f"(checking top-{kernel.top_n} candidates each, mode={kernel.mode}, "
        f"bonus={kernel.bonus}, include_target={kernel.include_target})",
        file=sys.stderr,
    )

    # Run the Lean kernel probe (resumable; results TSV accumulates across runs).
    artifact_dir = ROOT / "artifacts" / "lean_kernel"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Normalize the benchmark tag for the TSV filenames (hyphen → underscore).
    bench_tag = records[0].get("benchmark", "run").replace("-", "_")
    req_path = artifact_dir / f"req_{bench_tag}_{kernel.mode}_top{kernel.top_n}.tsv"
    res_path = artifact_dir / f"res_{bench_tag}_{kernel.mode}_top{kernel.top_n}.tsv"
    run_lean_resumable(
        queries,
        kernel.mode,
        req_path,
        res_path,
        timeout=kernel.per_proc_timeout,
        batch=kernel.batch,
    )
    results = parse_results(res_path)
    print(f"[kernel] parsed {len(results)} query results → {res_path}", file=sys.stderr)

    # Rerank each record over its FULL pool. Records without a resolvable expression are
    # passed through unchanged (kernel applied=False). save_top_k=None keeps the full
    # reranked pool in the output.
    qidx_set = {q["qidx"] for q in queries}
    weights = config.scoring.weights
    reranked: list[dict] = []
    for record in records:
        if record["query_id"] in qidx_set:
            kmap = results.get(record["query_id"], {})
            reranked.append(rerank_record(record, kmap, weights, kernel.bonus))
        else:
            reranked.append(_stamp_kernel_meta(record, kernel.bonus, applied=False))

    # Write the reranked JSONL (refuse-to-overwrite unless resume).
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = _completed_query_ids(output_path) if resume else set()
    if output_path.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    mode = "a" if resume and output_path.exists() else "w"
    written = 0
    skipped = 0
    with output_path.open(mode, encoding="utf-8") as stream:
        for record in reranked:
            if record["query_id"] in completed_ids:
                skipped += 1
                continue
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            written += 1
    print(
        f"[kernel] wrote {written} reranked records ({skipped} resume-skipped) → {output_path}",
        file=sys.stderr,
    )

    metrics = aggregate_metrics(reranked)
    print(json.dumps({"kernel_stage3": metrics, "output": str(output_path)}))
    return metrics


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
