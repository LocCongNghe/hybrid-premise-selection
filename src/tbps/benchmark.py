from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TestACase:
    expression: str
    premise_name: str


def load_test_a(benchmark_dir: Path) -> list[TestACase]:
    """Load Test A while enforcing the one-query/one-label invariant."""
    expressions = _read_nonempty_lines(benchmark_dir / "expressions.txt")
    premise_names = _read_nonempty_lines(benchmark_dir / "premise_names.txt")
    if len(expressions) != len(premise_names):
        raise ValueError(
            f"Test A has {len(expressions)} expressions but {len(premise_names)} labels"
        )
    return [
        TestACase(expression=expression, premise_name=premise_name)
        for expression, premise_name in zip(expressions, premise_names, strict=True)
    ]


def _read_nonempty_lines(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not line for line in lines):
        raise ValueError(f"Blank benchmark entry in {path}")
    return lines
