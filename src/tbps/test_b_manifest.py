from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO

import psycopg2

from tbps.expr import count_nodes, deserialize_expr

THEOREM_COMMAND = "Lean.Parser.Command.theorem"
SOURCE_ARCHIVE_SHA256 = "a26170707fbf63cf32c4597e9f63d2b050b1c9807eb0bf1c9e88839e7b64e036"
SPLIT_ARCHIVE_SHA256 = "aa043a96dd494097407c0469bf39acbf84bae4e2499abad6640b58cf119987be"
THEOREM_FIELD = b'"commandSyntaxKind": "Lean.Parser.Command.theorem"'
COPY_HEADER = b"COPY public.tactic_step (id, data) FROM stdin;"
ROW_START = re.compile(rb'^(\d+)\t\{"uri": "')
ROW_KEY = re.compile(
    rb'^\{"uri": "([^"]+)", "range": '
    rb'\{"endPosition": \{"line": (\d+), "column": (\d+)\}, '
    rb'"startPosition": \{"line": (\d+), "column": (\d+)\}\}'
)
LegacyKey = tuple[str, int, int, int, int]
LegacyOccurrenceKey = tuple[LegacyKey, int]


def _json_object(value: object) -> dict[str, object]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("mainGoalTypeJson must decode to a JSON object")
    return value


def build_manifest(
    database_url: str,
    manifest_path: Path,
    exclusions_path: Path,
    legacy_archive: Path,
) -> dict[str, Any]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    exclusions_tmp = exclusions_path.with_suffix(exclusions_path.suffix + ".tmp")

    counters: Counter[str] = Counter()
    pair_hashes: set[str] = set()
    duplicate_pairs = 0
    eligible_keys = legacy_theorem_keys(legacy_archive)
    counters["legacy_theorem_ids"] = len(eligible_keys)
    matched_legacy_keys: set[LegacyOccurrenceKey] = set()
    combined_occurrences: Counter[LegacyKey] = Counter()

    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT name FROM public.mathlib_filtered WHERE name IS NOT NULL")
            mathlib_names = {row[0] for row in cursor}
            cursor.execute(
                "SELECT theorem_name FROM public.wl_encodings_new WHERE theorem_name IS NOT NULL"
            )
            wl_names = {row[0] for row in cursor}

        with (
            manifest_tmp.open("w", encoding="utf-8", newline="\n") as manifest_file,
            exclusions_tmp.open("w", encoding="utf-8", newline="\n") as exclusions_file,
            conn.cursor(name="test_b_manifest_rows") as rows,
        ):
            rows.itersize = 50
            rows.execute("SELECT id, data FROM public.tactic_step ORDER BY id")
            for source_id, data in rows:
                counters["raw_rows"] += 1
                reasons: list[str] = []
                if not isinstance(data, dict):
                    reasons.append("data_not_object")
                    data = {}

                command_kind = data.get("commandSyntaxKind")
                row_key = _database_row_key(data)
                occurrence_key = (row_key, combined_occurrences[row_key])
                combined_occurrences[row_key] += 1
                legacy_source_id = eligible_keys.get(occurrence_key)
                if legacy_source_id is None:
                    reasons.append("not_legacy_theorem_declaration")
                else:
                    matched_legacy_keys.add(occurrence_key)
                    counters[f"canonical_current_kind:{command_kind}"] += 1

                label_value = data.get("appFnName")
                label = label_value.strip() if isinstance(label_value, str) else ""
                if not label:
                    reasons.append("missing_premise_label")

                state: dict[str, object] | None = None
                node_count: int | None = None
                try:
                    state = _json_object(data.get("mainGoalTypeJson"))
                    node_count = count_nodes(deserialize_expr(state))
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                    RecursionError,
                ) as error:
                    reasons.append(f"invalid_expr_json:{type(error).__name__}")

                in_mathlib = bool(label) and label in mathlib_names
                in_wl = bool(label) and label in wl_names

                if reasons:
                    for reason in reasons:
                        counters[f"excluded:{reason}"] += 1
                    exclusions_file.write(
                        json.dumps(
                            {
                                "source_row_id": source_id,
                                "label": label or None,
                                "reasons": reasons,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    continue

                if label and not in_mathlib:
                    counters["retained:label_missing_mathlib_filtered"] += 1
                if label and not in_wl:
                    counters["retained:label_missing_wl_encodings_new"] += 1

                assert state is not None and node_count is not None
                canonical_state = json.dumps(
                    state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                pair_digest = hashlib.sha256((label + "\0" + canonical_state).encode()).hexdigest()
                if pair_digest in pair_hashes:
                    duplicate_pairs += 1
                pair_hashes.add(pair_digest)

                record = {
                    "query_id": f"test-b-{legacy_source_id}",
                    "source_row_id": source_id,
                    "legacy_source_row_id": legacy_source_id,
                    "theorem": label,
                    "state": state,
                    "state_text": data.get("mainGoalTypeStr"),
                    "exact_syntax": data.get("exactStxReprint"),
                    "command_syntax_kind": command_kind,
                    "node_count": node_count,
                    "availability": {"mathlib_filtered": in_mathlib, "wl_encodings_new": in_wl},
                    "availability_reason": (
                        None if in_mathlib and in_wl else "target_not_in_retrieval_corpus"
                    ),
                    "payload_source": "combined_repaired_dump",
                    "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                    "split_membership_archive_sha256": SPLIT_ARCHIVE_SHA256,
                }
                manifest_file.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                counters["canonical_rows"] += 1

            missing_keys = set(eligible_keys) - matched_legacy_keys
            legacy_payloads = legacy_theorem_payloads(legacy_archive, missing_keys)
            for occurrence_key in sorted(missing_keys, key=lambda key: eligible_keys[key]):
                legacy_source_id = eligible_keys[occurrence_key]
                data = legacy_payloads[occurrence_key]
                label_value = data.get("appFnName")
                label = label_value.strip() if isinstance(label_value, str) else ""
                reasons: list[str] = []
                if not label:
                    reasons.append("missing_premise_label")
                state: dict[str, object] | None = None
                node_count: int | None = None
                try:
                    state = _json_object(data.get("mainGoalTypeJson"))
                    node_count = count_nodes(deserialize_expr(state))
                except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as error:
                    reasons.append(f"invalid_expr_json:{type(error).__name__}")

                in_mathlib = bool(label) and label in mathlib_names
                in_wl = bool(label) and label in wl_names
                if label and not in_mathlib:
                    counters["retained:label_missing_mathlib_filtered"] += 1
                if label and not in_wl:
                    counters["retained:label_missing_wl_encodings_new"] += 1
                if reasons:
                    for reason in reasons:
                        counters[f"excluded:{reason}"] += 1
                    exclusions_file.write(
                        json.dumps(
                            {
                                "source_row_id": None,
                                "legacy_source_row_id": legacy_source_id,
                                "label": label or None,
                                "reasons": reasons,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    continue

                assert state is not None and node_count is not None
                canonical_state = json.dumps(
                    state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                pair_digest = hashlib.sha256((label + "\0" + canonical_state).encode()).hexdigest()
                if pair_digest in pair_hashes:
                    duplicate_pairs += 1
                pair_hashes.add(pair_digest)
                record = {
                    "query_id": f"test-b-{legacy_source_id}",
                    "source_row_id": None,
                    "legacy_source_row_id": legacy_source_id,
                    "theorem": label,
                    "state": state,
                    "state_text": data.get("mainGoalTypeStr"),
                    "exact_syntax": data.get("exactStxReprint"),
                    "command_syntax_kind": THEOREM_COMMAND,
                    "node_count": node_count,
                    "availability": {"mathlib_filtered": in_mathlib, "wl_encodings_new": in_wl},
                    "availability_reason": (
                        None if in_mathlib and in_wl else "target_not_in_retrieval_corpus"
                    ),
                    "payload_source": "legacy_copy_repaired",
                    "source_archive_sha256": SPLIT_ARCHIVE_SHA256,
                    "split_membership_archive_sha256": SPLIT_ARCHIVE_SHA256,
                }
                manifest_file.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                counters["canonical_current_kind:legacy_only"] += 1
                counters["canonical_rows"] += 1

    manifest_tmp.replace(manifest_path)
    exclusions_tmp.replace(exclusions_path)
    manifest_hash = _sha256(manifest_path)
    exclusions_hash = _sha256(exclusions_path)
    manifest_path.with_suffix(manifest_path.suffix + ".sha256").write_text(
        f"{manifest_hash}  {manifest_path.name}\n", encoding="ascii"
    )
    exclusions_path.with_suffix(exclusions_path.suffix + ".sha256").write_text(
        f"{exclusions_hash}  {exclusions_path.name}\n", encoding="ascii"
    )
    return {
        "counts": dict(sorted(counters.items())),
        "duplicate_state_theorem_pairs_retained": duplicate_pairs,
        "manifest_sha256": manifest_hash,
        "exclusions_sha256": exclusions_hash,
    }


def legacy_theorem_ids(archive_path: Path) -> set[int]:
    """Read paper-split IDs without parsing the legacy archive's malformed JSON COPY field."""
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if len(members) != 1:
            raise ValueError(f"expected one SQL member in {archive_path}, found {len(members)}")
        stream = archive.extractfile(members[0])
        if stream is None:
            raise ValueError(f"could not open SQL member in {archive_path}")
        return set(_theorem_keys_from_sql(stream).values())


def legacy_theorem_keys(archive_path: Path) -> dict[LegacyOccurrenceKey, int]:
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if len(members) != 1:
            raise ValueError(f"expected one SQL member in {archive_path}, found {len(members)}")
        stream = archive.extractfile(members[0])
        if stream is None:
            raise ValueError(f"could not open SQL member in {archive_path}")
        return _theorem_keys_from_sql(stream)


def _theorem_ids_from_sql(stream: BinaryIO) -> set[int]:
    return set(_theorem_keys_from_sql(stream).values())


def _theorem_keys_from_sql(stream: BinaryIO) -> dict[LegacyOccurrenceKey, int]:
    records: dict[LegacyOccurrenceKey, int] = {}
    occurrences: Counter[LegacyKey] = Counter()
    in_copy = False
    current_id: int | None = None
    current_key: LegacyKey | None = None
    current_is_theorem = False

    def finish_record() -> None:
        if current_id is not None and current_is_theorem:
            if current_key is None:
                raise ValueError(f"legacy tactic_step {current_id} has no stable source key")
            occurrence_key = (current_key, occurrences[current_key])
            records[occurrence_key] = current_id
        if current_id is not None and current_key is not None:
            occurrences[current_key] += 1

    for line in stream:
        if not in_copy:
            if line.rstrip(b"\r\n") == COPY_HEADER:
                in_copy = True
            continue
        if line.rstrip(b"\r\n") == b"\\.":
            finish_record()
            return records
        match = ROW_START.match(line)
        if match:
            finish_record()
            current_id = int(match.group(1))
            payload = line.split(b"\t", 1)[1]
            key_match = ROW_KEY.match(payload)
            current_key = _legacy_match_key(key_match) if key_match else None
            current_is_theorem = THEOREM_FIELD in line
        elif current_id is not None and THEOREM_FIELD in line:
            current_is_theorem = True
    raise ValueError("legacy tactic_step COPY terminator not found")


def legacy_theorem_payloads(
    archive_path: Path, wanted: set[LegacyOccurrenceKey]
) -> dict[LegacyOccurrenceKey, dict[str, object]]:
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        stream = archive.extractfile(members[0])
        if stream is None:
            raise ValueError(f"could not open SQL member in {archive_path}")
        payloads = _legacy_payloads_from_sql(stream, wanted)
    missing = wanted - set(payloads)
    if missing:
        raise ValueError(f"could not recover {len(missing)} legacy theorem payloads")
    return payloads


def _legacy_payloads_from_sql(
    stream: BinaryIO, wanted: set[LegacyOccurrenceKey]
) -> dict[LegacyOccurrenceKey, dict[str, object]]:
    payloads: dict[LegacyOccurrenceKey, dict[str, object]] = {}
    occurrences: Counter[LegacyKey] = Counter()
    in_copy = False
    current_key: LegacyKey | None = None
    chunks: list[bytes] = []

    def finish_record() -> None:
        if current_key is None:
            return
        occurrence_key = (current_key, occurrences[current_key])
        occurrences[current_key] += 1
        if occurrence_key in wanted:
            payloads[occurrence_key] = _repair_legacy_json(b"\n".join(chunks), current_key)

    for line in stream:
        if not in_copy:
            if line.rstrip(b"\r\n") == COPY_HEADER:
                in_copy = True
            continue
        if line.rstrip(b"\r\n") == b"\\.":
            finish_record()
            return payloads
        match = ROW_START.match(line)
        if match:
            finish_record()
            field = line.split(b"\t", 1)[1].rstrip(b"\r\n")
            key_match = ROW_KEY.match(field)
            current_key = _legacy_match_key(key_match) if key_match else None
            chunks = [field]
        else:
            chunks.append(line.rstrip(b"\r\n"))
    raise ValueError("legacy tactic_step COPY terminator not found")


def _repair_legacy_json(copy_field: bytes, key: LegacyKey) -> dict[str, object]:
    decoded = _copy_text_unescape(copy_field)
    try:
        value = json.loads(_escape_json_controls(decoded))
        if isinstance(value, dict):
            return value
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass

    result: dict[str, object] = {
        "uri": key[0],
        "range": {
            "endPosition": {"line": key[1], "column": key[2]},
            "startPosition": {"line": key[3], "column": key[4]},
        },
        "commandSyntaxKind": THEOREM_COMMAND,
    }
    result["appFnName"] = _extract_json_string_field(decoded, "appFnName")
    result["mainGoalTypeJson"] = _extract_bounded_main_goal(decoded)
    for field in ("mainGoalTypeStr", "exactStxReprint"):
        try:
            result[field] = _extract_json_string_field(decoded, field)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            result[field] = None
    return result


def _copy_text_unescape(value: bytes) -> bytes:
    simple = {ord("b"): 8, ord("f"): 12, ord("n"): 10, ord("r"): 13, ord("t"): 9, ord("v"): 11}
    output = bytearray()
    index = 0
    while index < len(value):
        if value[index] != ord("\\") or index + 1 == len(value):
            output.append(value[index])
            index += 1
            continue
        escaped = value[index + 1]
        output.append(simple.get(escaped, escaped))
        index += 2
    return bytes(output)


def _escape_json_controls(value: bytes) -> str:
    replacements = {8: b"\\b", 9: b"\\t", 10: b"\\n", 12: b"\\f", 13: b"\\r"}
    output = bytearray()
    in_string = False
    escaped = False
    for byte in value:
        if in_string and byte < 32:
            output.extend(replacements.get(byte, f"\\u{byte:04x}".encode("ascii")))
            escaped = False
            continue
        output.append(byte)
        if not in_string:
            if byte == ord('"'):
                in_string = True
        elif escaped:
            escaped = False
        elif byte == ord("\\"):
            escaped = True
        elif byte == ord('"'):
            in_string = False
    return output.decode("utf-8")


def _extract_json_string_field(value: bytes, field: str) -> str:
    marker = f'"{field}": '.encode("ascii")
    start = value.find(marker)
    if start < 0:
        raise ValueError(f"legacy payload is missing {field}")
    fragment = _escape_json_controls(value[start + len(marker) :])
    decoded, _ = json.JSONDecoder().raw_decode(fragment)
    if not isinstance(decoded, str):
        raise ValueError(f"legacy field {field} is not a string")
    return decoded


def _extract_bounded_main_goal(value: bytes) -> str:
    marker = b'"mainGoalTypeJson": "'
    start = value.find(marker)
    if start < 0:
        raise ValueError("legacy payload is missing mainGoalTypeJson")
    start += len(marker)
    end = value.find(b'", "commandSyntaxKind":', start)
    if end < 0:
        raise ValueError("legacy payload has no mainGoalTypeJson field boundary")
    return _loose_json_string_unescape(value[start:end]).decode("utf-8")


def _loose_json_string_unescape(value: bytes) -> bytes:
    simple = {
        ord('"'): ord('"'),
        ord("\\"): ord("\\"),
        ord("/"): ord("/"),
        ord("b"): 8,
        ord("f"): 12,
        ord("n"): 10,
        ord("r"): 13,
        ord("t"): 9,
    }
    output = bytearray()
    index = 0
    while index < len(value):
        if value[index] != ord("\\") or index + 1 == len(value):
            output.append(value[index])
            index += 1
            continue
        escaped = value[index + 1]
        if escaped == ord("u") and index + 5 < len(value):
            codepoint = int(value[index + 2 : index + 6], 16)
            output.extend(chr(codepoint).encode("utf-8"))
            index += 6
            continue
        output.append(simple.get(escaped, escaped))
        index += 2
    return bytes(output)


def _legacy_match_key(match: re.Match[bytes]) -> LegacyKey:
    return (
        match.group(1).decode("utf-8"),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
        int(match.group(5)),
    )


def _database_row_key(data: dict[str, object]) -> LegacyKey:
    uri = data.get("uri")
    range_value = data.get("range")
    if not isinstance(uri, str) or not isinstance(range_value, dict):
        raise ValueError("tactic_step is missing uri/range source key")
    end = range_value.get("endPosition")
    start = range_value.get("startPosition")
    if not isinstance(end, dict) or not isinstance(start, dict):
        raise ValueError("tactic_step range is missing start/end positions")
    values = (end.get("line"), end.get("column"), start.get("line"), start.get("column"))
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        raise ValueError("tactic_step source positions must be integers")
    return (uri, *values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical Test B JSONL manifest")
    parser.add_argument(
        "--database-url",
        default=os.environ.get(
            "TBPS_DATABASE_URL",
            "postgresql://tbps:tbps-local-only@127.0.0.1:8923/tbps_baseline",
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/tree_based/test_b/manifest.jsonl"),
    )
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=Path("benchmarks/tree_based/test_b/exclusions.jsonl"),
    )
    parser.add_argument("--expected-raw", type=int, default=8288)
    parser.add_argument("--expected-canonical", type=int, default=6119)
    parser.add_argument(
        "--legacy-archive",
        type=Path,
        default=Path("upstream/imathwy-tbps/data/test_set_B_tactic_step.sql.tar.gz"),
    )
    parser.add_argument("--report", type=Path, default=Path("artifacts/data/manifest-build.json"))
    args = parser.parse_args()

    report = build_manifest(args.database_url, args.manifest, args.exclusions, args.legacy_archive)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    counts = report["counts"]
    if counts.get("raw_rows") != args.expected_raw:
        raise SystemExit(f"expected {args.expected_raw} raw rows, got {counts.get('raw_rows')}")
    if counts.get("canonical_rows") != args.expected_canonical:
        raise SystemExit(
            f"expected {args.expected_canonical} canonical rows, got {counts.get('canonical_rows')}"
        )


if __name__ == "__main__":
    main()
