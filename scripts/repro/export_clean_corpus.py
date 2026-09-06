#!/usr/bin/env python3
"""I3 — export the CLEAN Lean corpus for dense-index building (space-mismatch fix).

The DB column ``mathlib_filtered.statement_str`` is the fully elaborated pretty-print
(``forall (n m : Nat), Eq.{1} Nat (HAdd.hAdd... instAddNat n m) ...``). That put docs in a
different space from the raw ``q(...)`` query → dense cosine was noise.

This script packages the **clean** statement text produced by ``TBPS.CleanPP`` (default Lean
pretty-printer: ``∀ (n m : ℕ), n + m = m + n``) — the space LeanDojo ByT5 was trained on, and
the same notation family as the raw query / Test B ``state_text``.

Outputs (under ``artifacts/i3/dense_clean/``):
  corpus_clean.jsonl.gz  — 217,555 (name, statement_str=clean_text) pairs, ordered by name.
  clean_query_map.jsonl  — query_id → clean_text for Test A (for the runner's clean-query cache).
  provenance.json        — counts, sha256, source TSV path, build note.

Requires the clean TSVs already built by ``lean/TBPS/CleanPP.lean`` (see its docstring;
run from the ``lean/`` dir via ``lake env lean --run TBPS/CleanPP.lean docs/queries ...``):
  artifacts/i3/dense_clean/clean_statements.tsv      (name \t clean_text)
  artifacts/i3/dense_clean/clean_queries_test_a.tsv  (idx \t clean_text)

Usage (inside WSL, with the DB running for the name list + Test A labels):
    .venv/bin/python scripts/repro/export_clean_corpus.py
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT / "artifacts/i3/dense_clean"
CLEAN_STMTS_TSV = CLEAN_DIR / "clean_statements.tsv"
CLEAN_QUERIES_TSV = CLEAN_DIR / "clean_queries_test_a.tsv"
TEST_A_EXPRS = ROOT / "benchmarks/tree_based/test_a/expressions.txt"
TEST_A_NAMES = ROOT / "benchmarks/tree_based/test_a/premise_names.txt"

OUT_CORPUS = CLEAN_DIR / "corpus_clean.jsonl.gz"
OUT_QUERY_MAP = CLEAN_DIR / "clean_query_map.jsonl"
OUT_PROVENANCE = CLEAN_DIR / "provenance.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_clean_stmts(tsv: Path) -> dict[str, str]:
    """name → clean_text. Names with empty clean text (pp failed / not found) are omitted;
    downstream falls back to the DB statement_str for those (logged in provenance)."""
    out: dict[str, str] = {}
    missing = 0
    with tsv.open(encoding="utf-8") as f:
        for line in f:
            # TSV: name \t clean_text  (clean_text may be empty on failure)
            name, _, clean = line.rstrip("\n").partition("\t")
            if clean:
                out[name] = clean
            else:
                missing += 1
    print(f"  loaded {len(out)} clean statements ({missing} had empty clean text)", flush=True)
    return out


def export_corpus(clean_map: dict[str, str]) -> tuple[int, str]:
    """Write corpus_clean.jsonl.gz with field name ``statement_str`` (the field name the
    index builder reads). Ordered by name (stable, reproducible). Returns (count, sha256)."""
    count = 0
    with gzip.open(OUT_CORPUS, "wt", encoding="utf-8") as f:
        for name in sorted(clean_map):
            f.write(json.dumps({"name": name, "statement_str": clean_map[name]}, ensure_ascii=False) + "\n")
            count += 1
    return count, sha256_file(OUT_CORPUS)


def export_query_map() -> int:
    """Write clean_query_map.jsonl: query_id (test-a-001..) → clean_text, joined with the
    premise label so the runner can key by query_id. The idx in the TSV is the 0-based line
    index into expressions.txt, matching the order premise_names.txt is read."""
    with TEST_A_EXPRS.open(encoding="utf-8") as f:
        exprs = [line.strip() for line in f if line.strip()]
    with TEST_A_NAMES.open(encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    # TSV idx → clean text
    idx_to_clean: dict[int, str] = {}
    with CLEAN_QUERIES_TSV.open(encoding="utf-8") as f:
        for line in f:
            idx_s, _, clean = line.rstrip("\n").partition("\t")
            idx_to_clean[int(idx_s)] = clean
    count = 0
    with OUT_QUERY_MAP.open("w", encoding="utf-8") as f:
        for i, (expr, name) in enumerate(zip(exprs, names, strict=True)):
            qid = f"test-a-{i+1:03d}"
            clean = idx_to_clean.get(i, expr)  # fall back to raw expr if clean missing
            f.write(json.dumps({"query_id": qid, "target": name, "clean_text": clean}, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    print("loading clean statements TSV...", flush=True)
    clean_map = load_clean_stmts(CLEAN_STMTS_TSV)
    print(f"writing {OUT_CORPUS} ...", flush=True)
    count, corpus_sha = export_corpus(clean_map)
    print(f"  {count} rows, sha256={corpus_sha[:16]}...", flush=True)

    print(f"writing {OUT_QUERY_MAP} ...", flush=True)
    qcount = export_query_map()
    print(f"  {qcount} query mappings", flush=True)

    prov = {
        "description": "CLEAN corpus for I3 dense index (space-mismatch fix). statement_str field = Lean default pretty-print of info.type (pp.all=false).",
        "corpus_count": count,
        "corpus_sha256": corpus_sha,
        "query_count": qcount,
        "source_stmts_tsv": str(CLEAN_STMTS_TSV.relative_to(ROOT)),
        "source_queries_tsv": str(CLEAN_QUERIES_TSV.relative_to(ROOT)),
        "clean_pp_tool": "lean/TBPS/CleanPP.lean (env.find? name → info.type → PrettyPrinter.ppExpr, pp.all=false)",
        "note": "field name 'statement_str' kept for compatibility with the index builder; value is the CLEAN text, not the verbose elaborated form.",
    }
    OUT_PROVENANCE.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PROVENANCE}", flush=True)
    print(json.dumps(prov, indent=2), flush=True)


if __name__ == "__main__":
    main()
