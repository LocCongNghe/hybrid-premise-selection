from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path


def extract_lean_expression(expression: str, *, project_dir: Path = Path("lean")) -> object:
    """Elaborate one Test A term with pinned Mathlib and return its Expr JSON."""
    project_dir = project_dir.resolve()
    token = uuid.uuid4().hex
    input_path = project_dir / f".tbps-input-{token}.txt"
    output_path = project_dir / f".tbps-output-{token}.json"
    command_path = project_dir / f".tbps-extract-{token}.lean"
    input_path.write_text(normalize_test_a_expression(expression), encoding="utf-8")
    command_path.write_text(
        "import TBPS.ExtractExpr\n"
        "open scoped Qq\n"
        "set_option maxRecDepth 100000\n"
        f"tbps_parse_and_write {json.dumps(input_path.as_posix())} "
        f"{json.dumps(output_path.as_posix())}\n",
        encoding="utf-8",
    )
    try:
        completed = subprocess.run(
            ["lake", "env", "lean", command_path.name],
            cwd=project_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=300,
            check=False,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Lean extraction failed: {message}")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return payload["your_expr"]
    finally:
        for path in (input_path, output_path, command_path):
            path.unlink(missing_ok=True)


def extract_lean_expressions(
    expressions: list[str], *, project_dir: Path = Path("lean"), timeout_seconds: int = 600
) -> list[object]:
    """Elaborate several Test A terms in one Lean process to amortize Mathlib startup."""
    if not expressions:
        return []
    project_dir = project_dir.resolve()
    token = uuid.uuid4().hex
    input_paths = [
        project_dir / f".tbps-input-{token}-{index}.txt" for index in range(len(expressions))
    ]
    output_paths = [
        project_dir / f".tbps-output-{token}-{index}.json" for index in range(len(expressions))
    ]
    command_path = project_dir / f".tbps-extract-{token}.lean"
    for path, expression in zip(input_paths, expressions, strict=True):
        path.write_text(normalize_test_a_expression(expression), encoding="utf-8")
    commands = [
        f"tbps_parse_and_write {json.dumps(input_path.as_posix())} "
        f"{json.dumps(output_path.as_posix())}"
        for input_path, output_path in zip(input_paths, output_paths, strict=True)
    ]
    command_path.write_text(
        "import TBPS.ExtractExpr\nopen scoped Qq\nset_option maxRecDepth 100000\n"
        + "\n".join(commands)
        + "\n",
        encoding="utf-8",
    )
    try:
        completed = subprocess.run(
            ["lake", "env", "lean", command_path.name],
            cwd=project_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Lean batch extraction failed: {message}")
        return [json.loads(path.read_text(encoding="utf-8"))["your_expr"] for path in output_paths]
    finally:
        for path in (*input_paths, *output_paths, command_path):
            path.unlink(missing_ok=True)


def normalize_test_a_expression(expression: str) -> str:
    """Remove the Qq quotation wrapper stored in the official Test A text file."""
    normalized = expression.strip()
    if normalized.startswith("q(") and normalized.endswith(")"):
        return normalized[2:-1].strip()
    return normalized
