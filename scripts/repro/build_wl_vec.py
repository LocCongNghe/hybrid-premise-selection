#!/usr/bin/env python3
"""Build the vectorized WL tables for option B/C (no accuracy change).

Reads simp_wl_encode_3 (JSONB dict[str,int]) from wl_encodings_new, builds a single
global hash->int id map (sorted unique hashes), and writes:

  wl_hash_map(hash text PRIMARY KEY, id int)
  wl_encodings_vec(
    theorem_name text PRIMARY KEY,
    wl_ids   int[]   sorted ascending,
    wl_counts int[]   parallel to wl_ids,
    wl_norm  double precision  = sqrt(sum(count*count)) over ALL counts
  )

wl_norm is computed with the EXACT same formula as wl_kernel's per-side norm
(sqrt(sum(v*v)) over all values), so wl_kernel_vec is byte-identical to wl_kernel.

Run: TBPS_DB_PASSWORD=tbps-local-only .venv/bin/python scripts/repro/build_wl_vec.py
"""
from __future__ import annotations

import math
import os
import time

import psycopg2
from psycopg2.extras import execute_values

from tbps.wl import wl_encoding_to_vec, wl_hash_to_id_map


DSN = dict(host="127.0.0.1", port=8923, dbname="tbps_baseline", user="tbps")


def main() -> int:
    dsn = dict(DSN)
    dsn["password"] = os.environ["TBPS_DB_PASSWORD"]
    conn = psycopg2.connect(**dsn)
    conn.autocommit = False

    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT theorem_name, simp_wl_encode_3 FROM wl_encodings_new "
            "WHERE simp_wl_encode_3 IS NOT NULL ORDER BY theorem_name"
        )
        rows = cur.fetchall()
    print(f"fetched {len(rows)} encodings in {time.perf_counter()-t0:.2f}s")

    # Global hash -> id map over ALL encodings (deterministic, sorted).
    mt0 = time.perf_counter()
    encs = [r[1] or {} for r in rows]
    hash_to_id = wl_hash_to_id_map(*encs)
    print(f"built hash->id map: {len(hash_to_id)} unique hashes in {time.perf_counter()-mt0:.2f}s")
    del encs

    # Vectorize every candidate.
    vt0 = time.perf_counter()
    vec_rows = []
    for name, enc in rows:
        ids, counts, norm = wl_encoding_to_vec(enc or {}, hash_to_id)
        vec_rows.append((name, list(ids), list(counts), float(norm)))
    print(f"vectorized {len(vec_rows)} candidates in {time.perf_counter()-vt0:.2f}s")

    # Write tables.
    wt0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS wl_encodings_vec")
        cur.execute("DROP TABLE IF EXISTS wl_hash_map")
        cur.execute(
            "CREATE TABLE wl_hash_map (hash text PRIMARY KEY, id int NOT NULL)"
        )
        cur.execute(
            """
            CREATE TABLE wl_encodings_vec (
                theorem_name text PRIMARY KEY,
                wl_ids   integer[],
                wl_counts integer[],
                wl_norm  double precision
            )
            """
        )
        # hash map
        execute_values(
            cur,
            "INSERT INTO wl_hash_map (hash, id) VALUES %s ON CONFLICT DO NOTHING",
            list(hash_to_id.items()),
            page_size=10000,
        )
        # vectors
        execute_values(
            cur,
            "INSERT INTO wl_encodings_vec (theorem_name, wl_ids, wl_counts, wl_norm) VALUES %s "
            "ON CONFLICT (theorem_name) DO NOTHING",
            vec_rows,
            page_size=5000,
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS wl_encodings_vec_theorem_name_idx "
            "ON wl_encodings_vec (theorem_name)"
        )
    conn.commit()
    print(f"wrote tables in {time.perf_counter()-wt0:.2f}s")

    # Verify counts match source.
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM wl_encodings_vec")
        n_vec = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM wl_hash_map")
        n_map = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM wl_encodings_new WHERE simp_wl_encode_3 IS NOT NULL")
        n_src = cur.fetchone()[0]
    conn.close()
    print(f"wl_encodings_vec rows: {n_vec}  (source: {n_src})  match={n_vec==n_src}")
    print(f"wl_hash_map rows: {n_map}")
    print(f"TOTAL {time.perf_counter()-t0:.2f}s")
    return 0 if n_vec == n_src else 1


if __name__ == "__main__":
    raise SystemExit(main())
