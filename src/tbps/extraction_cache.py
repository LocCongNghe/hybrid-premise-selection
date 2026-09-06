"""Shared Test A expression-extraction cache.

Extracting each Test A term via `lake env lean` costs ~120 s/query because of the
fixed `import Mathlib` startup. Batching many terms into one Lean process amortizes
that startup, and caching the resulting `Expr` JSON to disk lets every later run
(different fusion profile / config) skip extraction entirely.

The cache is a JSONL file: one record per query with `query_id`, `target`,
`expression_sha256` (sha256 of the raw expression text, for integrity check) and
`expression_json` (the elaborated Expr). On load each record's hash is re-checked
against the current benchmark text so a stale cache (wrong Mathlib / edited file)
is rejected rather than silently reused.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from tbps.config import RunnerConfig
from tbps.lean_extractor import extract_lean_expressions
from tbps.runner import QueryCase, load_test_a_cases

EXPECTED_QUERIES = 100
# Batch size: how many terms to elaborate in one `lake env lean` process.
# Larger = more startup amortization, but a single failure loses more work.
EXTRACTION_CHUNK_SIZE = 25


def extract_test_a_cache(
    config: RunnerConfig, cache_path: Path, *, resume: bool
) -> list[QueryCase]:
    """Return Test A cases with `expression_json` populated, using/building a cache.

    If `cache_path` exists and `resume` is False, it is an error (refuse to overwrite).
    If `resume` is True, existing cached queries are reused and only the missing ones
    are extracted. The cache is appended to incrementally and fsync'd per chunk so a
    killed process can be resumed.
    """
    base_cases = list(load_test_a_cases(config))
    if len(base_cases) != EXPECTED_QUERIES:
        raise ValueError(f"expected {EXPECTED_QUERIES} Test A cases, found {len(base_cases)}")
    cached = _read_extraction_cache(cache_path) if resume else {}
    if cache_path.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite extraction cache: {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    remaining = [case for case in base_cases if case.query_id not in cached]
    with cache_path.open("a", encoding="utf-8") as stream:
        for offset in range(0, len(remaining), EXTRACTION_CHUNK_SIZE):
            chunk = remaining[offset : offset + EXTRACTION_CHUNK_SIZE]
            expressions = [case.expression_text or "" for case in chunk]
            extracted = extract_lean_expressions(expressions, timeout_seconds=1800)
            for case, expression_json in zip(chunk, extracted, strict=True):
                record = {
                    "query_id": case.query_id,
                    "target": case.target,
                    "expression_sha256": _hash_expression(case.expression_text or ""),
                    "expression_json": expression_json,
                }
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                cached[case.query_id] = record
            stream.flush()
            os.fsync(stream.fileno())
            print(
                json.dumps(
                    {
                        "stage": "extract_test_a",
                        "cached": len(cached),
                        "expected": EXPECTED_QUERIES,
                    }
                ),
                flush=True,
            )
    if len(cached) != EXPECTED_QUERIES:
        raise ValueError(f"extraction cache has {len(cached)} unique queries")
    for case in base_cases:
        record = cached[case.query_id]
        expected_hash = _hash_expression(case.expression_text or "")
        if record["target"] != case.target or record["expression_sha256"] != expected_hash:
            raise ValueError(f"stale extraction cache record: {case.query_id}")
    return [
        replace(
            case,
            expression_json=cached[case.query_id]["expression_json"],
            expression_text=None,
        )
        for case in base_cases
    ]


def load_test_a_cache(config: RunnerConfig, cache_path: Path) -> list[QueryCase]:
    """Load a fully-built cache without extracting. Raises if incomplete or stale.

    Use this for the fast path: every later run points at the same cache file and
    skips extraction entirely. Integrity (hash + target) is verified per record.
    """
    if not cache_path.exists():
        raise FileNotFoundError(
            f"expression cache not found: {cache_path} "
            "(build it once with --expression-cache on a fresh path)"
        )
    cached = _read_extraction_cache(cache_path)
    if len(cached) != EXPECTED_QUERIES:
        raise ValueError(f"extraction cache has {len(cached)} queries, expected {EXPECTED_QUERIES}")
    base_cases = list(load_test_a_cases(config))
    if len(base_cases) != EXPECTED_QUERIES:
        raise ValueError(f"expected {EXPECTED_QUERIES} Test A cases, found {len(base_cases)}")
    for case in base_cases:
        record = cached.get(case.query_id)
        if record is None:
            raise ValueError(f"missing extraction cache record: {case.query_id}")
        expected_hash = _hash_expression(case.expression_text or "")
        if record["target"] != case.target or record["expression_sha256"] != expected_hash:
            raise ValueError(f"stale extraction cache record: {case.query_id}")
    return [
        replace(
            case,
            expression_json=cached[case.query_id]["expression_json"],
            expression_text=None,
        )
        for case in base_cases
    ]


def _read_extraction_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(line)
        query_id = record["query_id"]
        if query_id in records:
            raise ValueError(f"duplicate extraction cache query at line {line_number}: {query_id}")
        records[query_id] = record
    return records


def _hash_expression(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
