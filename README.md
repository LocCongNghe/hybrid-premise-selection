# Hybrid Retrieval and Kernel-Guarded Reranking for Premise Selection in Lean 4

End-to-end premise-selection pipeline for Mathlib/Lean `v4.18.0`: hybrid
candidate retrieval (WL structural + BM25 lexical + LeanDojo ByT5 dense)
followed by an optional Lean kernel-based reranking stage, with metric
aggregation for every run. The baseline configuration reproduces *Tree-Based
Premise Selection for Lean4* (Wang et al., NeurIPS 2025) on the same corpus.

The repository covers everything from raw data import to per-run metrics.

## Repository layout

```
lean/                 Lean project pinned to Mathlib v4.18.0
  TBPS/ExtractExpr.lean   elaborated-Expr JSON extractor (Test A queries)
  TBPS/CleanPP.lean       clean printer for the dense-embedding space
  TBPS/KernelProbe.lean   streaming isDefEq applicability probe (Stage 3)
src/tbps/             Python: retrieval, scoring, fusion, reranking, metrics
  retrieval/bm25.py       BM25 lexical index
  retrieval/dense.py      LeanDojo ByT5 dense index (checkpoint from HuggingFace)
  kernel_rerank.py        Stage-3 reranking driver (resumable batches)
configs/              run profiles (baseline, hybrid, hybrid+kernel)
scripts/repro/        environment bootstrap, data import, index builders
data/                 download guide for the upstream SQL archives and benchmark inputs
```

## Requirements

- **Ubuntu WSL2 / Linux** for the shell scripts (they check `uname`).
- Docker (PostgreSQL corpus container, bound to `127.0.0.1:8923`).
- [elan](https://github.com/leanprover/elan) with the toolchain pinned in
  `lean/lean-toolchain`.
- Python ≥ 3.11.

## Quickstart

```sh
# 1. Environment: build Lean/Mathlib, create .venv, start PostgreSQL
bash scripts/repro/bootstrap.sh

# 2. Data: download the SQL archives + Test A inputs from the original authors'
#    repository, import the archives into PostgreSQL, and build the canonical
#    Test-B manifest (see data/README.md for the download links)
bash scripts/repro/import-data.sh
.venv/bin/tbps-build-test-b

# 3. Build the WL vector table read by the primary config (~140 s; the table
#    is not created by import-data.sh)
python scripts/repro/build_wl_vec.py
```

### Runs

Every run writes one JSONL record per query plus aggregate metrics, over the
benchmark inputs prepared in Quickstart step 2 (see `data/README.md`).

```sh
# Baseline reproduction (WL-only)
.venv/bin/tbps-run test-a --output artifacts/runs/test-a.jsonl --workers 4
.venv/bin/tbps-run test-b --resume --output artifacts/runs/test-b.jsonl --workers 4

# Hybrid (WL + BM25 + dense pool)
.venv/bin/tbps-run test-b --resume --output artifacts/runs/hybrid.jsonl \
    --config configs/baseline-paper-hybrid.toml --workers 4

# Hybrid + kernel rerank (Stage 3 auto-chained)
.venv/bin/tbps-run test-b --config configs/baseline-paper-hybrid-kernel.toml \
    --output artifacts/runs/test-b-kernel.jsonl --workers 4

# Metrics for any run output
.venv/bin/tbps-run summarize --input artifacts/runs/test-b-kernel.kernel.jsonl
```

### Hybrid retrieval (dense + BM25)

The dense retriever needs the `dense` Python extra (torch + transformers); the
checkpoint (`kaiyuy/leandojo-lean4-retriever-byt5-small`) is downloaded
automatically from HuggingFace:

```sh
.venv/bin/pip install -c requirements.lock -e ".[dense]"
```

Build the clean corpus export and the dense index. First produce the clean-text
TSVs with the CleanPP Lean pass (run from `lean/`; see `lean/TBPS/CleanPP.lean`):

```sh
mkdir -p ../artifacts/i3/dense_clean
# corpus names from the imported database:
psql "postgresql://tbps:tbps-local-only@127.0.0.1:8923/tbps_baseline" \
    -Atc 'SELECT name FROM mathlib_filtered ORDER BY name' > ../artifacts/corpus_names.txt
lake env lean --run TBPS/CleanPP.lean docs ../artifacts/corpus_names.txt \
    ../artifacts/i3/dense_clean/clean_statements.tsv
lake env lean --run TBPS/CleanPP.lean queries \
    ../benchmarks/tree_based/test_a/expressions.txt \
    ../artifacts/i3/dense_clean/clean_queries_test_a.tsv
# (the expressions file is the Test A input downloaded in Quickstart step 2)
```

Then build the indexes:

```sh
python scripts/repro/export_clean_corpus.py      # corpus_clean.jsonl.gz + query map
python scripts/repro/build_dense_index_clean.py  # embeds the clean statements
python scripts/repro/build_bm25_index.py         # BM25 over constant tokens
```

With the indexes in place, run the hybrid and kernel configurations from
"Runs" above.

### Kernel reranking (Stage 3)

One command runs Stage 1+2 and auto-chains Stage 3 over the wide-save pool
(`configs/baseline-paper-hybrid-kernel.toml`):

```sh
.venv/bin/tbps-run test-b --config configs/baseline-paper-hybrid-kernel.toml \
    --output artifacts/runs/test-b-kernel.jsonl --workers 4
# writes test-b-kernel.jsonl (Stage 1+2) and test-b-kernel.kernel.jsonl (Stage 3)
.venv/bin/tbps-run summarize --input artifacts/runs/test-b-kernel.kernel.jsonl
```

The same command with `test-a` reproduces the Test A kernel result. An existing
wide-save can be re-ranked alone via `tbps-run kernel-rerank --input <wide-save>.jsonl
--config configs/baseline-paper-hybrid-kernel.toml`.

## Determinism and provenance

- Ranking is fully deterministic: ties are broken by candidate name everywhere
  (SQL `ORDER BY`, WL sort, RRF, competition ranks). Repeated full runs are
  byte-identical.
- Every run record carries provenance (git commit, Mathlib SHA, config path +
  hash, fusion profile, workers).
- Outputs are never overwritten: the runner refuses existing files unless
  `--resume`, and resume rejects duplicate query ids.

## License

TBD
