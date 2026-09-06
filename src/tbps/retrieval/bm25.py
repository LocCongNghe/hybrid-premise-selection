"""BM25 lexical retrieval over the Lean `Expr` corpus (I3 — hybrid retrieval).

Tokenization
------------
A "document" is a theorem's elaborated ``Expr`` (the same ``expr_cse_json`` the WL
pipeline uses). The lexical tokens are the declaration names of every ``Const``
node in the tree, plus namespace segments so a query mentioning ``Set.union`` also
matches theorems built around ``Set``. This is the structural-lexical signal that
complements the WL kernel: WL captures neighborhood shape, BM25 captures which
named constants the goal and a candidate share.

Determinism
-----------
The query result is sorted by ``(-bm25_score, name)`` so the candidate ``name`` is
the deterministic secondary tie-break, matching the WL path and CLAUDE.md. BM25 is
otherwise fully deterministic (no randomness, no IDF smoothing that varies by run).

This module is pure-Python and adds no new dependency (keeps ``requirements.lock``
intact). The implementation is the standard Okapi BM25 with ``k1`` and ``b``.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from tbps.expr import (
    App,
    Const,
    Expr,
    ForallE,
    Lam,
    LetE,
    MData,
    Proj,
    deserialize_expr,
)


def collect_const_names(expr: Expr, out: list[str]) -> None:
    """Walk an ``Expr`` tree, appending every ``Const.decl_name`` to ``out``.

    Mirrors the recursion shape of ``tree.count_nodes`` so every Const is visited
    exactly once (an App visits fn then arg, a binder visits type then body, etc.).
    """
    if isinstance(expr, Const):
        out.append(expr.decl_name)
        return
    if isinstance(expr, App):
        collect_const_names(expr.fn, out)
        collect_const_names(expr.arg, out)
        return
    if isinstance(expr, (Lam, ForallE)):
        collect_const_names(expr.binder_type, out)
        collect_const_names(expr.body, out)
        return
    if isinstance(expr, LetE):
        collect_const_names(expr.type, out)
        collect_const_names(expr.value, out)
        collect_const_names(expr.body, out)
        return
    if isinstance(expr, MData):
        collect_const_names(expr.expr, out)
        return
    if isinstance(expr, Proj):
        out.append(expr.type_name)
        collect_const_names(expr.struct, out)
        return
    # BVar, FVar, MVar, Sort, Lit are leaves with no Const underneath.
    return


def tokenize_name(name: str) -> list[str]:
    """Split a fully-qualified declaration name into lexical tokens.

    ``Set.PairwiseDisjoint`` -> ``["Set", "PairwiseDisjoint", "Set.PairwiseDisjoint"]``.
    The full name is kept as a token so exact-name matches score strongly; the
    segments give partial-match coverage across a namespace.
    """
    name = name.strip()
    if not name:
        return []
    tokens = [name]
    # Split on namespace separators; ignore empty segments (leading/trailing dots).
    parts = [p for p in name.replace("'", ".").split(".") if p]
    # Deduplicate while preserving order so ``Foo.Foo`` does not double-count.
    seen: set[str] = set()
    for p in parts:
        if p not in seen:
            seen.add(p)
            tokens.append(p)
    return tokens


def tokenize_expr(expression_json: object) -> list[str]:
    """Deserialize + walk an ``Expr`` JSON, returning the BM25 token list.

    Each Const contributes its name tokens; order is the tree's natural traversal
    order (so repeated constants appear with the right term frequency). Returns an
    empty list for ``None``/null payloads (the corpus filter excludes those, but a
    query might receive one).
    """
    if expression_json is None:
        return []
    try:
        expr = deserialize_expr(expression_json)
    except (ValueError, TypeError):
        return []
    consts: list[str] = []
    collect_const_names(expr, consts)
    tokens: list[str] = []
    for name in consts:
        tokens.extend(tokenize_name(name))
    return tokens


@dataclass
class BM25Index:
    """Okapi BM25 over a fixed corpus of tokenized documents.

    The index is built once from the 217k-row corpus and cached to disk. A query
    scores every document that shares at least one token with the query (via the
    inverted index); documents with no shared token score 0.0 and are returned
    only to fill the top-k when fewer than k documents share a token.
    """

    # Document metadata.
    doc_names: list[str] = field(default_factory=list)
    doc_token_counts: list[Counter] = field(default_factory=list)
    doc_lengths: list[int] = field(default_factory=list)
    avgdl: float = 0.0
    # Inverted index: token -> list of (doc_index, term_frequency).
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    # BM25 parameters (standard defaults; NOT tuned on Test A/B).
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(
        cls,
        docs: list[tuple[str, list[str]]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        """Build an index from ``(name, tokens)`` pairs.

        ``docs`` must be in a stable order (the caller sorts by name) so the
        serialized index is reproducible across runs.
        """
        index = cls(k1=k1, b=b)
        postings: dict[str, list[tuple[int, int]]] = {}
        total_len = 0
        for doc_idx, (name, tokens) in enumerate(docs):
            tf = Counter(tokens)
            index.doc_names.append(name)
            index.doc_token_counts.append(tf)
            index.doc_lengths.append(len(tokens))
            total_len += len(tokens)
            for token, freq in tf.items():
                postings.setdefault(token, []).append((doc_idx, freq))
        index.avgdl = (total_len / len(docs)) if docs else 0.0
        index.postings = postings
        return index

    def idf(self, token: str) -> float:
        """Okapi BM25 IDF with +1 smoothing (always positive, never negative)."""
        n = len(self.doc_names)
        df = len(self.postings.get(token, ()))
        # idf = ln((n - df + 0.5) / (df + 0.5) + 1)  — the +1 inside keeps it > 0.
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def query(self, query_tokens: list[str], *, limit: int = 1500) -> list[tuple[str, float]]:
        """Score the corpus against ``query_tokens``; return top-``limit`` by score.

        Returns ``(name, score)`` pairs sorted by ``(-score, name)``. Documents
        sharing no query token are not returned unless fewer than ``limit``
        documents share a token (then zero-score documents fill by name).
        """
        q_tf = Counter(query_tokens)
        scores: dict[int, float] = {}
        for token, q_freq in q_tf.items():
            postings = self.postings.get(token)
            if not postings:
                continue
            idf = self.idf(token)
            if idf <= 0.0:
                continue
            for doc_idx, tf in postings:
                dl = self.doc_lengths[doc_idx]
                denom = (
                    tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl) if self.avgdl else 1.0
                )
                # q_freq is absorbed into the standard BM25 query-term weight; we use
                # the per-term contribution repeated q_freq times (matching rank_bm25).
                scores[doc_idx] = (
                    scores.get(doc_idx, 0.0) + idf * (tf * (self.k1 + 1.0)) / denom * q_freq
                )
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.doc_names[kv[0]]))
        if len(ranked) < limit:
            # Fill with zero-score docs by name to reach `limit` (matches WL behavior
            # of returning a full top-k even when many candidates tie at 0).
            have = {idx for idx, _ in ranked}
            fill = sorted(
                (
                    (idx, self.doc_names[idx])
                    for idx in range(len(self.doc_names))
                    if idx not in have
                ),
                key=lambda x: x[1],
            )
            ranked.extend((idx, 0.0) for idx, _ in fill[: limit - len(ranked)])
        else:
            ranked = ranked[:limit]
        return [(self.doc_names[idx], score) for idx, score in ranked]

    def doc_index(self, name: str) -> int | None:
        """Return the internal index of a document by name, or None if absent."""
        # Linear scan is too slow for 217k; callers should build a name->idx map.
        # This helper is for small ad-hoc lookups only.
        try:
            return self.doc_names.index(name)
        except ValueError:
            return None

    def save(self, path: Path) -> None:
        """Serialize the index to JSON (deterministic: sorted keys, stable order)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k1": self.k1,
            "b": self.b,
            "avgdl": self.avgdl,
            "doc_names": self.doc_names,
            "doc_lengths": self.doc_lengths,
            # Drop doc_token_counts (reconstructable from postings); keeps the file
            # much smaller.
            "postings": {
                token: [[doc_idx, freq] for doc_idx, freq in pairs]
                for token, pairs in sorted(self.postings.items())
            },
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        data = json.loads(path.read_text(encoding="utf-8"))
        index = cls(k1=data["k1"], b=data["b"])
        index.doc_names = data["doc_names"]
        index.doc_lengths = data["doc_lengths"]
        index.avgdl = data["avgdl"]
        index.postings = {
            token: [(int(d), int(f)) for d, f in pairs] for token, pairs in data["postings"].items()
        }
        # Rebuild per-doc token Counters from postings (needed? query doesn't use
        # them — only postings + doc_lengths — but keep for completeness/inspection).
        index.doc_token_counts = [Counter() for _ in index.doc_names]
        for token, pairs in index.postings.items():
            for doc_idx, freq in pairs:
                index.doc_token_counts[doc_idx][token] = freq
        return index
