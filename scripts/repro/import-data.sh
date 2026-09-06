#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

# Archives downloaded manually from the upstream authors' repository
# (https://github.com/imathwy/tbps/tree/main/data) into data/.
# Falls back to a pristine upstream checkout if one is present instead.
data_dir="${REPO_ROOT}/data"
if [[ ! -s "${data_dir}/mathlib_filtered_backup0515.sql.xz" ]]; then
    data_dir="${REPO_ROOT}/upstream/imathwy-tbps/data"
fi
log_dir="${REPO_ROOT}/artifacts/import"
mkdir -p "${log_dir}"
log_file="${log_dir}/import-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee "${log_file}") 2>&1

for archive in \
    mathlib_filtered_backup0515.sql.xz \
    wl_encodings_new_backup0515.sql.xz \
    test_sets_B_C_tactic_steps.sql.gz; do
    test -s "${data_dir}/${archive}" || {
        echo "Missing archive: ${archive}" >&2
        echo "Download it from https://github.com/imathwy/tbps/tree/main/data into data/" >&2
        exit 2
    }
done

compose up -d --wait postgres

echo "Resetting only the three tables in the dedicated tbps_baseline database."
compose exec -T postgres psql -v ON_ERROR_STOP=1 -U tbps -d tbps_baseline <<'SQL'
DROP TABLE IF EXISTS public.tactic_step CASCADE;
DROP TABLE IF EXISTS public.tactic_step_2 CASCADE;
DROP TABLE IF EXISTS public.wl_encodings_new CASCADE;
DROP TABLE IF EXISTS public.mathlib_filtered CASCADE;
SQL

restore_xz() {
    local archive="$1"
    echo "Restoring ${archive}"
    xz -dc "${data_dir}/${archive}" \
        | sed -E '/ OWNER TO princhern;/d' \
        | compose exec -T postgres psql -v ON_ERROR_STOP=1 -U tbps -d tbps_baseline
}

restore_xz mathlib_filtered_backup0515.sql.xz
restore_xz wl_encodings_new_backup0515.sql.xz

echo "Restoring Test B from the recommended combined B+C archive by stream"
gzip -dc "${data_dir}/test_sets_B_C_tactic_steps.sql.gz" \
    | grep -v '^\\restrict ' \
    | grep -v '^\\unrestrict ' \
    | compose exec -T postgres psql -v ON_ERROR_STOP=1 -U tbps -d tbps_baseline

test_c_rows="$(compose exec -T postgres psql -U tbps -d tbps_baseline -Atc \
    'SELECT count(*) FROM public.tactic_step_2;')"
if [[ "${test_c_rows}" != "12494" ]]; then
    echo "Expected 12494 Test C rows, found ${test_c_rows}" >&2
    exit 1
fi
echo "Verified ${test_c_rows} out-of-scope Test C rows; dropping tactic_step_2."
compose exec -T postgres psql -v ON_ERROR_STOP=1 -U tbps -d tbps_baseline \
    -c 'DROP TABLE public.tactic_step_2 CASCADE;'

compose exec -T postgres psql -v ON_ERROR_STOP=1 -U tbps -d tbps_baseline <<'SQL'
CREATE INDEX IF NOT EXISTS mathlib_filtered_name_idx ON public.mathlib_filtered (name);
CREATE INDEX IF NOT EXISTS wl_encodings_new_theorem_name_idx ON public.wl_encodings_new (theorem_name);
CREATE INDEX IF NOT EXISTS tactic_step_command_kind_idx
    ON public.tactic_step ((data ->> 'commandSyntaxKind'));
ANALYZE public.mathlib_filtered;
ANALYZE public.wl_encodings_new;
ANALYZE public.tactic_step;
SQL

schema_dir="${REPO_ROOT}/artifacts/data"
mkdir -p "${schema_dir}"
compose exec -T postgres pg_dump --schema-only --no-owner -U tbps -d tbps_baseline \
    > "${schema_dir}/schema.sql"

echo "import_log=${log_file}"
