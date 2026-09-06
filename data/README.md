# Data archives

The files below are published by the original Tree-Based Premise Selection
authors and are downloaded directly from their repository:

**https://github.com/imathwy/tbps/tree/main/data**

Download the three SQL archives into this directory (`data/`):

| Archive | Contents | Size | SHA-256 |
|---|---|---|---|
| `mathlib_filtered_backup0515.sql.xz` | `mathlib_filtered` corpus (~217k theorems) | 152 MB | `22664bd0…` |
| `wl_encodings_new_backup0515.sql.xz` | `wl_encodings_new` precomputed WL encodings | 169 MB | `e80cd583…` |
| `test_sets_B_C_tactic_steps.sql.gz` | `tactic_step` Test B+C proof-state rows | 186 MB | `a2617070…` |

Full SHA-256 hashes are recorded in the upstream repository's Git LFS objects.
After downloading, verify with:

```sh
sha256sum data/*.xz data/*.gz
```

Also download (used by the Test-B manifest builder, not by the DB import):

| File | Contents |
|---|---|
| `test_set_B_tactic_step.sql.tar.gz` | legacy Test-B-only archive — the authoritative source of the 6,119 canonical query ids |

Place it under `upstream/imathwy-tbps/data/` (the manifest builder's default
`--legacy-archive` path), or pass `--legacy-archive data/test_set_B_tactic_step.sql.tar.gz`
to `tbps-build-test-b` explicitly.

## Test A inputs

The two Test A input files are also published upstream (byte-identical to the
ones used here). Download them into `benchmarks/tree_based/test_a/`, renaming
`Prop_name.txt` to the label-file name the configs expect:

```sh
mkdir -p benchmarks/tree_based/test_a
curl -L -o benchmarks/tree_based/test_a/expressions.txt \
    https://raw.githubusercontent.com/imathwy/tbps/main/data/expressions.txt
curl -L -o benchmarks/tree_based/test_a/premise_names.txt \
    https://raw.githubusercontent.com/imathwy/tbps/main/data/Prop_name.txt
```

## Import into PostgreSQL

```sh
bash scripts/repro/import-data.sh
```

Then build the canonical Test-B manifest from the imported tables:

```sh
.venv/bin/tbps-build-test-b
```

This writes `benchmarks/tree_based/test_b/manifest.jsonl` (6,119 queries) and
`exclusions.jsonl`, and fails unless the exact expected row counts are found.
