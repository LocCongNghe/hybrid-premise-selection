"""I3 Idea 1 — Lean kernel applicability reranker (Stage 3, end-to-end).

For each query, ask the Lean kernel whether each candidate theorem's *generalized
conclusion* (universal binders turned into metavariables) is definitionally equal
(``isDefEq``) to the query's generalized body. This is a SEMANTIC signal from the kernel,
orthogonal to WL/TEDS (structure), dense (cosine), and BM25 (lexical). It is the only
mechanism that gains R@5/R@10 on full Test B without R@1 regression.

This module houses the verified Lean-invocation + rerank logic that powers the integrated
Stage 3 (``tbps-run`` auto-chains it when ``[kernel]`` is enabled in the config).

Pipeline (GPU-free; needs Lean+Mathlib, run in WSL Ubuntu):
  1. Read a Stage-1+2 JSONL output (``run_batch`` product — byte-identical baseline).
  2. For each record, build the query expression + top-N candidate names (exactly the
     natural top-N by saved Stage-2 rank — the gold target is NEVER appended; a target
     outside the window is unprobed and unboosted). Test A term = raw term string from
     ``expressions.txt`` (Lean strips the ``q(...)`` wrapper + elaborates); Test B term =
     compact ``state`` JSON from the manifest (Lean deserializes with fvars/forall-binders
     → metavars).
  3. Write a requests TSV and invoke ONE Lean process (``lake env lean`` on a scratch
     driver that runs ``tbps_kernel_probe``), amortizing the ~120 s ``import Mathlib`` over
     all queries. Resumable + wall-clock-robust to pathological ``isDefEq``.
  4. Parse results; re-rank each record: applicable non-leaders get +``bonus``, the
     structural leader set is pinned at rank 1 (never demoted) → R@1 == baseline by
     construction. Recompute ``final_rank``; write a NEW kernel-reranked JSONL.

When ``[kernel]`` is disabled (the default) this module is not invoked and Stage 1+2 is
byte-identical to the locked baseline (no ``kernel`` field anywhere in the output).
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
# Lean invocation (moved verbatim from the verified offline probe)
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

    ``tbps_kernel_probe`` is a command-level ``elab`` (runs at *compile* time, not via
    ``main``), so we mirror ``lean_extractor``: write a scratch driver ``.lean`` that
    ``import TBPS.KernelProbe``s and invokes the command, then compile that scratch file
    with ``lake env lean <scratch>`` (NO ``--run``). Compiling executes the command.

    Returns True on clean completion, False on timeout / hard failure. On timeout the
    driver's every-5-query flush preserves partial results; the caller skips the single
    hanging query and resumes the rest in a fresh process.
    """
    lean_dir = LEAN_DIR
    req_wsl = _to_wsl(req_path)
    res_wsl = _to_wsl(res_path)
    token = uuid.uuid4().hex[:8]
    scratch_name = f".kp-run-{token}.lean"
    scratch_path = lean_dir / scratch_name
    scratch_content = (
        "import TBPS.KernelProbe\n"
        # Global heartbeat ceiling. `isDefEq` checks heartbeats against
        # `(← read).maxHeartbeats` captured at Core.Context creation; `withTheReader`/
        # `withOptions` inside TermElabM do NOT reliably propagate to that captured field.
        # Setting the GLOBAL `maxHeartbeats` to 8M here makes isDefEq read 8M directly, and
        # `withCurrHeartbeats` (per-isDefEq, inside checkApplicable) resets `initHeartbeats`
        # so each pair gets a fresh ~8M-heartbeat (~8s) delta — a hang throws
        # `runtime.maxHeartbeats`, caught by checkApplicable's try/catch (→ none), and the
        # process CONTINUES. 8M is generous for legitimate isDefEq (<1M) but bounds a
        # pathological pair to ~8s. The wall-clock `--timeout` is the hard backstop.
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
            # The Lean process hung (pathological reduction past the heartbeat catch, or a
            # non-isDefEq hang). The driver flushed results every 5 queries, so the partial
            # res file is preserved. Return False so the resumable loop skips the one
            # hanging query and resumes the rest in a fresh process.
            print(
                f"  [kernel] TIMEOUT after {timeout}s (pathological reduction); "
                f"partial results preserved",
                flush=True,
            )
            # The killed subprocess may leave a `lake`/`lean` child alive inside WSL — kill
            # it explicitly so the next iteration starts clean.
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
    """Run the Lean probe with wall-clock timeout + resume, robust to pathological isDefEq.

    ``isDefEq`` does NOT check heartbeats during reduction, so a single pathological pair
    can hang a Lean process indefinitely; the heartbeat mechanism cannot bound it. Instead
    we bound each Lean PROCESS by wall-clock ``timeout``: the driver flushes results every 5
    queries, so on a timeout the partial results are preserved; we then skip the single
    hanging query (the first un-flushed one) and resume the rest in a fresh Lean process
    (Mathlib re-imported). Each hang costs one re-import (~120 s) + the timeout, and loses
    only the hanging query's candidates.

    Resume is durable: ``res_path`` accumulates across invocations. ``batch`` (>0) caps
    queries per Lean process to bound memory/state growth; 0 = no cap.
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
        # Three failure modes collapse here:
        #   (a) wall-clock TIMEOUT (run_lean returned False): the Lean process hung on a
        #       pathological isDefEq; the driver flushed every 5 queries so partial results
        #       were preserved.
        #   (b) rc!=0 hard failure (run_lean returned False).
        #   (c) rc==0 but ZERO new queries completed (`progressed` is False): the `elab`
        #       command-level heartbeat exception (runtime.maxHeartbeats escaping
        #       tryCatchRuntimeEx → re-thrown by `ofExcept` at the command level → aborts
        #       the driver's `for` loop before the first 5-query flush) makes Lean exit
        #       rc=0 with an empty res file. Without this branch the loop would re-run the
        #       SAME chunk forever (ok=True, 0 progress) — the "hung with 0 new queries"
        #       stall. In all three, the first chunk member not yet `done` is the culprit:
        #       emit empty (unknown) results for it so the run ADVANCES instead of looping.
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

    ``include_target=False`` (the default, deployed) probes exactly the natural top-N and
    never consults the label — a target outside the window is unprobed and unboosted.
    ``include_target=True`` is a measurement-only mode (used to compute target
    applicability statistics): the target may sit outside the saved ``top_k`` (when
    ``output_top_k`` was small and the target ranked below it) but still be in the
    retrieval pool — in that case its component scores are in ``target_component_scores``,
    so we reconstruct a candidate dict from there. A pool-miss target
    (``target_component_scores`` is None) cannot be appended.
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

    Test A: ``term`` = the raw term string from ``expressions.txt`` (Lean strips the
    ``q(...)`` wrapper + elaborates), indexed by ``source_metadata.benchmark_line`` (1-based).
    Test B: ``term`` = the compact ``state`` JSON from the manifest (single line), keyed by
    ``query_id``. Records whose expression can't be resolved (e.g. a Test B target absent
    from the manifest) are skipped (the run ADVANCES without them; their ``final_rank`` is
    left unchanged by the rerank).

    ``include_target`` (default False) probes exactly the natural top-N. Setting it True
    appends the gold target to the probe set — measurement-only mode for applicability
    statistics; it must never be used for reported ranking metrics.
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
    """Structural-only score (dense/bm25 weight forced to 0). Matches ``_structural_score``
    in scoring.py. The large-tree branch is handled by ``teds`` being None → 0.0 (the
    deployed C1 config has no ``paper_large_tree`` profile, so the same weights apply)."""
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

    The rerank operates on the FULL saved pool (``record["top_k"]`` — e.g. 1000 candidates
    for a wide-save) PLUS the target (recovered from ``target_component_scores`` when it
    sits below the saved pool). This yields full-pool metrics comparable to the deployed C1
    baseline, NOT a truncated top-N+target sub-pool.

    Only candidates present in ``kernel_map`` (the top-N the Lean probe actually checked)
    can receive a boost; every other candidate gets bonus=0 (never checked → unknown → no
    boost). So the rerank re-orders the full pool using the kernel signal where available
    and leaves the rest on its structural score.

    Steps:
    1. ``struct_score`` per candidate (config weights, teds None→0).
    2. Leader set = candidates with max struct (pinned at rank 1; never demoted, even if a
       boosted non-leader's sort_key exceeds the leader's — R@1 == baseline by construction).
    3. Non-leaders: ``sort_key = struct + (bonus if kernel says True else 0)``.
    4. Two-tier competition rank: leaders (sorted by ``(-struct, name)`` → name order),
       then non-leaders (sorted by ``(-sort_key, name)``).
    5. ``final_rank`` = target's new competition rank in the FULL pool (None if the target
       is not in the pool — a pool-miss). ``top_k`` = the reranked pool (truncated to
       ``save_top_k`` if given; else the full pool). Each candidate dict gets a ``kernel``
       field (1.0/0.0/null); ``target_component_scores`` gets ``kernel`` + updated ``rank``.

    When ``kernel_map`` is empty/all-None, every bonus is 0 → the rerank is a no-op (same
    order as input, ranks re-stamped) → byte-identical ranking to Stage 1+2.
    """
    target = record["target"]
    # Full saved pool + target (recovered from target_component_scores if absent). This is
    # the FULL pool the baseline ranked; reranking it gives full-pool final_rank. NOTE: the
    # appended target only receives a boost if it is in kernel_map (i.e. it was inside the
    # probed top-N) — appending here is for rank bookkeeping, not for probing.
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
            # leader: final unchanged (never boosted, never demoted)
            leaders.append((s, _with_kernel(c, appl)))
        else:
            boost = bonus if appl is True else 0.0
            nonleaders.append((s + boost, _with_kernel(c, appl, final_override=s + boost)))

    # leaders: name order (all smax → (-smax, name) = name order), matching the baseline
    # competition_rank tie-break (deterministic).
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

    # target's new rank in the full pool (None only if the target is a pool-miss: not in
    # top_k AND not recoverable from target_component_scores).
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

    Reads ``input_path`` (the ``run_batch`` product, byte-identical baseline). Runs the Lean
    kernel probe over the batch (resumable), then re-ranks each record with leader-protect +
    boost, recomputes ``final_rank``, and writes ``output_path`` (or
    ``<input>.kernel.jsonl`` when None). Refuses to overwrite an existing output unless
    ``resume`` (resume validates prior records and rejects duplicate ``query_id``s, mirroring
    ``run_batch``). Returns aggregate metrics over the reranked records.
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

    # Validate the input is a WIDE-SAVE: the rerank needs the full saved pool (not just the
    # final output_top_k) to compute full-pool final_rank comparable to the deployed C1. A
    # 10-candidate top_k would make every in-pool target rank ≤ 10 → meaningless R@k.
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
    artifact_dir = ROOT / "artifacts" / "i3" / "lean_kernel"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize the benchmark tag for the filename: records carry ``"test-b"`` / ``"test-a"``
    # (hyphen), but the probe script names its TSVs with the ``--source`` flag value
    # (``test_b`` / ``test_a``, underscore). Normalizing to underscore lets the integrated
    # Stage 3 REUSE probe-generated result TSVs when they already exist (resume), instead of
    # silently re-running Lean.
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

    # Rerank each record over its FULL pool. Records without a resolvable expression (not in
    # `queries`) are passed through unchanged with kernel applied=False (final_rank untouched).
    # save_top_k=None keeps the full reranked pool in the output (a wide-save → wide-save),
    # so the output is itself re-rankable / inspectable.
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
