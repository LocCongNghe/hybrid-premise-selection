#!/usr/bin/env python3
"""Build the BM25 lexical index over the full ``mathlib_filtered`` corpus.

Reads ``expr_cse_json`` for every non-null corpus row, tokenizes each via
``tokenize_expr`` (Const declaration names + namespace segments), and builds a
``BM25Index`` cached to ``artifacts/i3/bm25_index.json``.

The build is deterministic: rows are fetched ``ORDER BY name`` so the serialized
index is reproducible across runs. Prints corpus/token statistics and writes a
SHA-256 of the index file alongside provenance.

Usage (inside WSL):
    .venv/bin/python scripts/repro/build_bm25_index.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import psycopg2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tbps.retrieval.bm25 import BM25Index, tokenize_expr  # noqa: E402


def main() -> None:
    out_dir = ROOT / "artifacts" / "i3"
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "bm25_index.json"

    password = os.environ.get("TBPS_DB_PASSWORD", "tbps-local-only")
    conn = psycopg2.connect(
        host="127.0.0.1", port=8923, user="tbps", dbname="tbps_baseline", password=password
    )

    started = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT name, expr_cse_json
            FROM mathlib_filtered
            WHERE expr_cse_json IS NOT NULL AND expr_cse_json <> 'null'::jsonb
            ORDER BY name
            """
        )
        rows = cur.fetchall()
    conn.close()
    fetch_seconds = time.perf_counter() - started
    print(f"fetched {len(rows)} corpus rows in {fetch_seconds:.1f}s")

    # Tokenize each document.
    tok_started = time.perf_counter()
    docs: list[tuple[str, list[str]]] = []
    empty = 0
    total_tokens = 0
    for name, expr_json in rows:
        tokens = tokenize_expr(expr_json)
        if not tokens:
            empty += 1
        else:
            total_tokens += len(tokens)
        docs.append((name, tokens))
    tok_seconds = time.perf_counter() - tok_started
    print(f"tokenized in {tok_seconds:.1f}s; empty-token docs: {empty}; "
          f"total tokens: {total_tokens}; avgdl: {total_tokens / max(1, len(docs)):.1f}")

    build_started = time.perf_counter()
    index = BM25Index.build(docs)
    build_seconds = time.perf_counter() - build_started
    print(f"built index in {build_seconds:.1f}s; vocab (distinct tokens): {len(index.postings)}")

    index.save(index_path)
    size_mb = index_path.stat().st_size / (1024 * 1024)
    sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    print(f"wrote {index_path} ({size_mb:.1f} MB) sha256={sha}")

    provenance = {
        "corpus_rows": len(rows),
        "empty_token_docs": empty,
        "total_tokens": total_tokens,
        "avgdl": index.avgdl,
        "vocab": len(index.postings),
        "k1": index.k1,
        "b": index.b,
        "index_sha256": sha,
        "index_size_mb": round(size_mb, 2),
        "fetch_seconds": round(fetch_seconds, 2),
        "tokenize_seconds": round(tok_seconds, 2),
        "build_seconds": round(build_seconds, 2),
    }
    (out_dir / "bm25_index_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print("wrote provenance:", out_dir / "bm25_index_provenance.json")


if __name__ == "__main__":
    main()
