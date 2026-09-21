"""Retrieval backends for the hybrid premise-selection pipeline.

This package holds the candidate generators that complement the WL kernel:
BM25 lexical retrieval (``bm25``) and dense retrieval. Each backend exposes
the same shape — a ranked ``(name, score)`` list with ``name`` as the
deterministic secondary sort key — so they can be unioned and fused uniformly.
"""

from tbps.retrieval.bm25 import BM25Index, tokenize_expr, tokenize_name

__all__ = ["BM25Index", "tokenize_expr", "tokenize_name"]
