#!/usr/bin/env python3
"""I3 — build the CLEAN dense embedding index (space-mismatch fix).

Encodes the **clean** statement text (Lean default pretty-printer, ``pp.all=false`` —
produced by ``lean/TBPS/CleanPP.lean``) instead of the elaborated ``statement_str``.
Reads from ``artifacts/i3/dense_clean/clean_statements.tsv``; names without clean text
fall back to the DB ``statement_str`` (so coverage stays at 217,555).

Usage (inside WSL, with the dense extra installed):
    .venv/bin/python scripts/repro/build_dense_index_clean.py --device cuda --batch-size 16
    .venv/bin/python scripts/repro/build_dense_index_clean.py --device cpu --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import psycopg2  # noqa: E402

from tbps.retrieval.dense import DenseEncoder, DenseIndex  # noqa: E402

DB = dict(host="127.0.0.1", port=8923, user="tbps", password="tbps-local-only", dbname="tbps_baseline")
CLEAN_TSV = ROOT / "artifacts/i3/dense_clean/clean_statements.tsv"
OUT_DIR = ROOT / "artifacts/i3/dense_index_clean"


def load_clean_map() -> dict[str, str]:
    """name → clean_text from the TSV built by TBPS.CleanPP."""
    out: dict[str, str] = {}
    with CLEAN_TSV.open(encoding="utf-8") as f:
        for line in f:
            name, _, clean = line.rstrip("\n").partition("\t")
            if clean:
                out[name] = clean
    return out


def stream_corpus(clean_map: dict[str, str]) -> list[tuple[str, str]]:
    """Return all (name, text) ordered by name. Uses clean text when available, else falls back
    to the DB statement_str (so every corpus row is present)."""
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute(
        "SELECT name, statement_str FROM mathlib_filtered "
        "WHERE statement_str IS NOT NULL AND statement_str != '' ORDER BY name"
    )
    rows = cur.fetchall()
    conn.close()
    out = []
    fallback = 0
    for name, stmt in rows:
        if name in clean_map:
            out.append((name, clean_map[name]))
        else:
            out.append((name, stmt))
            fallback += 1
    print(f"  {len(out)} rows ({fallback} fell back to elaborated statement_str)", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print(f"loading clean statement map from {CLEAN_TSV} ...", flush=True)
    clean_map = load_clean_map()
    print(f"  {len(clean_map)} clean texts", flush=True)

    print("streaming corpus from DB (ordered by name)...", flush=True)
    rows = stream_corpus(clean_map)
    if args.limit:
        rows = rows[: args.limit]
    print(f"  {len(rows)} rows to encode", flush=True)

    print(f"loading encoder (device={args.device}, batch={args.batch_size}, "
          f"max_length={args.max_length})...", flush=True)
    t0 = time.perf_counter()
    enc = DenseEncoder(device=args.device, max_length=args.max_length, batch_size=args.batch_size)
    print(f"  encoder loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    import numpy as np

    all_embs: list[object] = []
    names = [n for n, _ in rows]
    texts = [t for _, t in rows]
    chunk = args.batch_size * 50
    t0 = time.perf_counter()
    for start in range(0, len(texts), chunk):
        end = min(start + chunk, len(texts))
        emb = enc.encode(texts[start:end])
        all_embs.append(emb)
        if (start // chunk) % 5 == 0 or end == len(texts):
            dt = time.perf_counter() - t0
            rate = end / dt if dt else 0.0
            eta = (len(texts) - end) / rate if rate else 0.0
            print(f"  {end}/{len(texts)} ({100*end/len(texts):.1f}%) "
                  f"trunc={enc.truncation_rate:.3f} {rate:.0f}/s ETA {eta:.0f}s",
                  file=sys.stderr, flush=True)

    embeddings = np.concatenate(all_embs, axis=0) if all_embs else np.zeros((0, 1472), dtype=np.float32)
    print(f"encoded {len(rows)} rows in {time.perf_counter()-t0:.1f}s; "
          f"final truncation_rate={enc.truncation_rate:.3f}; shape={embeddings.shape}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = DenseIndex(doc_names=names, embeddings=embeddings)
    index.save(OUT_DIR)
    provenance = {
        "model": "kaiyuy/leandojo-lean4-retriever-byt5-small",
        "embedding_dim": 1472,
        "count": len(names),
        "truncation_rate": enc.truncation_rate,
        "max_length": args.max_length,
        "device": args.device,
        "build_seconds": time.perf_counter() - t0,
        "text_source": "CLEAN (Lean default pp, pp.all=false) from artifacts/i3/dense_clean/clean_statements.tsv; fallback to DB statement_str",
        "clean_texts_used": sum(1 for n, _ in rows if n in clean_map),
        "elaborated_fallback_used": sum(1 for n, _ in rows if n not in clean_map),
    }
    (OUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"wrote CLEAN index to {OUT_DIR} ({embeddings.nbytes / 1e6:.1f} MB)", flush=True)
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
