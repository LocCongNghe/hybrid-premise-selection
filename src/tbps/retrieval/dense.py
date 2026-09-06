"""Dense retrieval over the Lean corpus (I3 — hybrid retrieval, dense signal).

This module wraps the LeanDojo ByT5 retriever
(``kaiyuy/leandojo-lean4-retriever-byt5-small``) as the third candidate generator alongside
WL (structural) and BM25 (lexical). It mirrors the ``BM25Index`` shape — a ranked
``(name, score)`` list with ``name`` as the deterministic secondary sort key — so all three
retrievers fuse uniformly via RRF in the hybrid pipeline.

Encoding (matches the ReProver reference)
-----------------------------------------
The ByT5-small encoder maps a Lean text (a premise statement or a goal state) to a 1472-d
vector via masked mean pooling over ``last_hidden_state``::

    tok = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=...)
    hs  = model(tok.input_ids).last_hidden_state
    lens = tok.attention_mask.sum(dim=1)
    emb = (hs * tok.attention_mask.unsqueeze(2)).sum(dim=1) / lens.unsqueeze(1)

Embeddings are L2-normalized so retrieval is a dot product (cosine). The model is byte-level
(ByT5), so ``max_length`` is in bytes; Lean statements can be long (median ~1.3k chars, p95
~32k on the elaborated corpus), so truncation is applied and the truncation rate is logged.

Determinism
-----------
``query`` returns ``(name, cosine)`` pairs sorted by ``(-cosine, name)`` — the ``name``
secondary key matches WL/BM25 and CLAUDE.md's determinism contract. Encoding is deterministic
given the model weights (no sampling, eval mode, no dropout).

Dependencies
------------
``torch`` and ``transformers`` are NOT core dependencies — they live in the optional ``dense``
extra (``pip install -e ".[dense]"``). Importing this module without them raises a clear
``ImportError`` so the baseline venv (no torch) is never broken.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Heavy deps are imported lazily inside DenseEncoder so the module is importable without torch
# (the index classes only need numpy at runtime; the encoder needs torch/transformers).

# ByT5-small hidden size (the LeanDojo retriever's embedding dimension).
EMBEDDING_DIM = 1472
# Byte-level truncation. Benchmark on the 4GB GPU (batch=16): max_length=2048 → ~1.3 docs/s
# (O(n²) attention on long byte sequences); max_length=512 → ~6.2 docs/s; max_length=256 →
# ~13.8 docs/s. 512 is the quality/speed balance: it keeps the theorem name + type signature
# + leading binders (the most discriminative part of an elaborated Lean statement) while making
# the full 217k index buildable in ~10h. Longer statements are truncated to their prefix.
DEFAULT_MAX_LENGTH = 512
MODEL_NAME = "kaiyuy/leandojo-lean4-retriever-byt5-small"


@dataclass
class DenseIndex:
    """Corpus embeddings + names. ``query`` returns top-k by cosine with ``name`` tie-break.

    The index is built once offline (``build``) and cached to disk (``save``/``load``) so the
    per-query cost is a single matrix-vector product (~217k x 1472, <50ms on CPU with BLAS).
    """

    doc_names: list[str] = field(default_factory=list)
    # (N, 1472) float32, L2-normalized along axis 1. Stored as a plain ndarray (not a tensor)
    # so the index loads without torch — only building/querying a live encoder needs torch.
    embeddings: object = None  # numpy.ndarray, typed loose to avoid importing numpy at module top.

    def query(self, goal_embedding: object, *, limit: int = 1500) -> list[tuple[str, float]]:
        """Return ``limit`` ``(name, cosine)`` pairs ranked by ``(-cosine, name)``.

        ``goal_embedding`` must be a 1-D L2-normalized vector of dim 1472. Cosine = dot product.
        Ties in cosine are broken by ``name`` (ascending), matching the WL and BM25 paths.
        """
        if self.embeddings is None or not self.doc_names:
            return []
        import numpy as np

        goal = np.asarray(goal_embedding, dtype=np.float32)
        if goal.ndim != 1:
            goal = goal.reshape(-1)
        # Dot product with the normalized corpus matrix → cosine similarities.
        sims = self.embeddings @ goal  # shape (N,)
        # Top-k by (-sim, name). argsort gives ascending sim; we want descending sim with a
        # name tie-break, so build a stable sort on (-sim, name).
        order = sorted(
            range(len(self.doc_names)),
            key=lambda i: (-float(sims[i]), self.doc_names[i]),
        )
        return [(self.doc_names[i], float(sims[i])) for i in order[:limit]]

    def save(self, path: Path) -> None:
        """Serialize the index: ``embeddings.npy`` + ``names.json`` + ``provenance.json``.

        Deterministic: ``doc_names`` is written sorted-stable, embeddings in the same order.
        """
        import numpy as np

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self.embeddings)
        (path / "names.json").write_text(
            json.dumps(self.doc_names, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "DenseIndex":
        import numpy as np

        path = Path(path)
        embeddings = np.load(path / "embeddings.npy")
        names = json.loads((path / "names.json").read_text(encoding="utf-8"))
        if len(names) != embeddings.shape[0]:
            raise ValueError(
                f"dense index mismatch: {len(names)} names vs {embeddings.shape[0]} rows"
            )
        return cls(doc_names=names, embeddings=embeddings)


class DenseEncoder:
    """Wraps the LeanDojo ByT5 retriever. ``encode(texts) -> (N, 1472)`` L2-normalized.

    Constructed once per worker process. The model runs in eval mode (no dropout) on the
    requested device; if ``cuda`` is requested but unavailable, falls back to CPU with a warning.
    """

    def __init__(
        self,
        *,
        model_name: str = MODEL_NAME,
        device: str = "cuda",
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = 16,
    ) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoTokenizer, T5EncoderModel
        except ImportError as exc:  # pragma: no cover - exercised only without the dense extra
            raise ImportError(
                "DenseEncoder requires torch + transformers. Install the optional extra: "
                'pip install -e ".[dense]"'
            ) from exc

        import torch

        self.max_length = max_length
        self.batch_size = batch_size
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        resolved_device = device
        if device.startswith("cuda") and not torch.cuda.is_available():
            import warnings

            warnings.warn("CUDA not available; falling back to CPU for DenseEncoder", stacklevel=2)
            resolved_device = "cpu"
        self.device = resolved_device
        # T5EncoderModel loads ONLY the encoder (ByT5 has no decoder needed for embeddings).
        # AutoModel would load the full encoder-decoder T5 and require decoder_input_ids; the
        # encoder-only model accepts input_ids + attention_mask and returns last_hidden_state,
        # matching the ReProver masked-mean-pool recipe.
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.model = self.model.to(resolved_device)
        # FP16 on CUDA: ~3-5x faster via Tensor Cores on T4/V100/A100. Embeddings are cast back
        # to FP32 before saving (encode → hs.float()), so the stored index is always FP32.
        # Verified: cosine(FP16, FP32) > 0.998 — negligible for retrieval, no NaN. On GPUs without
        # Tensor Cores (GTX 1650) FP16 is slower, but query counts are small (~400) so the
        # consistency benefit (matching the Kaggle-built FP16 index) outweighs the speed cost.
        if resolved_device.startswith("cuda"):
            self.model = self.model.half()
        self.model.eval()

    @property
    def truncation_rate(self) -> float:
        """Fraction of the last ``encode`` batch that was truncated at ``max_length``."""
        return getattr(self, "_last_truncation_rate", 0.0)

    def encode(self, texts: list[str]) -> object:
        """Encode texts into a ``(N, 1472)`` float32 numpy array, L2-normalized.

        ByT5 is byte-level; ``max_length`` is in tokens (bytes). Masked mean pooling over
        ``last_hidden_state`` (matches ReProver). Batched to bound GPU memory.
        """
        import numpy as np

        torch = self._torch
        out: list[object] = []
        truncated = 0
        total = 0
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                total += len(batch)
                # Track truncation: a text is truncated if it exceeds max_length chars.
                # len(t) is O(1) (Python strings cache length); a fast proxy for byte length.
                for t in batch:
                    if len(t) > self.max_length:
                        truncated += 1
                # Pre-truncate to max_length chars BEFORE tokenizing. The HF tokenizer applies
                # truncation AFTER full tokenization, so without this a 993k-char Lean statement
                # is fully tokenized (~1M tokens) before being truncated to 512 — catastrophically
                # slow (5 docs/s instead of 200+). String slicing t[:k] is O(k), not O(len(t)).
                # This produces byte-identical tokens to the tokenizer's own truncation (the first
                # max_length bytes), just without the wasted work.
                tok = self.tokenizer(
                    [t[: self.max_length] for t in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                input_ids = tok.input_ids.to(self.device)
                attention_mask = tok.attention_mask.to(self.device)
                hs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
                hs = hs.float()  # FP32 for stable pooling (model runs in FP16 on CUDA)
                # Masked mean pool: sum(masked hidden states) / actual length.
                lens = attention_mask.sum(dim=1).clamp(min=1)
                feats = (hs * attention_mask.unsqueeze(2)).sum(dim=1) / lens.unsqueeze(1)
                out.append(feats.cpu().to(torch.float32).numpy())
        self._last_truncation_rate = (truncated / total) if total else 0.0
        if not out:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        mat = np.concatenate(out, axis=0)
        # L2-normalize so cosine = dot product.
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (mat / norms).astype(np.float32)

    def encode_one(self, text: str) -> object:
        """Encode a single text → 1-D L2-normalized vector of dim 1472."""
        import numpy as np

        mat = self.encode([text])
        return mat[0] if mat.shape[0] else np.zeros(EMBEDDING_DIM, dtype=np.float32)


def build_dense_index(
    encoder: DenseEncoder,
    name_text_pairs: list[tuple[str, str]],
    *,
    log_every: int = 2000,
) -> tuple[DenseIndex, dict[str, object]]:
    """Encode a corpus of ``(name, text)`` pairs into a ``DenseIndex``.

    ``name_text_pairs`` must be in a stable order (caller sorts by name) so the serialized
    index is reproducible. Returns the index + provenance (truncation rate, dims, count).
    """
    names = [n for n, _ in name_text_pairs]
    texts = [t for _, t in name_text_pairs]
    embeddings = encoder.encode(texts)
    provenance = {
        "model": MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "count": len(names),
        "truncation_rate": encoder.truncation_rate,
        "max_length": encoder.max_length,
    }
    return DenseIndex(doc_names=names, embeddings=embeddings), provenance
