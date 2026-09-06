#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/docker-compose.repro.yml"
export TBPS_DB_PASSWORD="${TBPS_DB_PASSWORD:-tbps-local-only}"
export TBPS_DATABASE_URL="${TBPS_DATABASE_URL:-postgresql://tbps:${TBPS_DB_PASSWORD}@127.0.0.1:8923/tbps_baseline}"

compose() {
    docker compose -f "${COMPOSE_FILE}" "$@"
}
