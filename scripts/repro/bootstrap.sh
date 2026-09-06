#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Run this script inside Ubuntu WSL2 or Linux." >&2
    exit 2
fi

cd "${REPO_ROOT}/lean"
lake update
lake exe cache get
lake build

cd "${REPO_ROOT}"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade 'pip==25.1.1'
.venv/bin/pip install -c requirements.lock -e .

compose up -d --wait postgres
compose exec -T postgres pg_isready -U tbps -d tbps_baseline
