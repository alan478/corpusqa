"""Vectors in memory: the only module that holds embeddings as a numpy matrix.

It lives in `store` (design.md §10) because ingest builds the matrix from the stored
embeddings at start-up and retrieval searches it; both may import `store`.

Vectors are stored as float32 little-endian bytes (`to_blob`). `VectorIndex` loads them
into one N x D matrix plus the chunk id of each row; search is a dot product (stored
vectors are unit length, so it is the cosine), then the top k rows.
"""

from collections.abc import Sequence, Set

import numpy as np

from corpusqa.config import UNIT_NORM_TOLERANCE

_FLOAT32_LE = "<f4"


def to_blob(vector: Sequence[float]) -> bytes:
    """A vector as stored: float32 little-endian bytes."""
    return np.asarray(vector, dtype=_FLOAT32_LE).tobytes()


def from_blob(blob: bytes) -> list[float]:
    """A stored vector back as floats."""
    return np.frombuffer(blob, dtype=_FLOAT32_LE).astype(float).tolist()


class BadVector(ValueError):
    """A vector that must not be stored; `reason` is a short code for counts."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


def unit_blob(values: Sequence[float], dims: int) -> bytes:
    """`values` checked (dims values, all finite, not all zero), L2-normalised, as stored.

    Raises BadVector otherwise. Every vector is checked here before it is stored, whoever
    made it, so a bad answer can never reach the index.
    """
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (dims,):
        raise BadVector("size", f"has {vector.size} values, expected {dims}")
    if not np.isfinite(vector).all():
        raise BadVector("not_finite", "has a value that is not finite")
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise BadVector("zero", "is all zeros")
    return (vector / norm).astype(_FLOAT32_LE).tobytes()


def blob_fits(blob: bytes, dims: int) -> bool:
    """A stored vector as `unit_blob` makes it: `dims` finite values of unit length (checked
    before a cached vector is reused, so a zero or unnormalised entry is a cache miss)."""
    if len(blob) != dims * np.dtype(_FLOAT32_LE).itemsize:
        return False
    vector = np.frombuffer(blob, dtype=_FLOAT32_LE).astype(np.float64)
    if not np.isfinite(vector).all():
        return False
    return abs(float(np.linalg.norm(vector)) - 1.0) <= UNIT_NORM_TOLERANCE


class VectorIndex:
    """Exact cosine search over unit vectors held in memory."""

    def __init__(self, chunk_ids: Sequence[int], vectors: Sequence[bytes], dims: int) -> None:
        if len(chunk_ids) != len(vectors):
            raise ValueError(f"{len(chunk_ids)} ids for {len(vectors)} vectors")
        self.dims = dims
        self._ids = np.asarray(chunk_ids, dtype=np.int64)
        matrix = np.frombuffer(b"".join(vectors), dtype=_FLOAT32_LE)
        if matrix.size != len(vectors) * dims:
            raise ValueError(f"vectors do not all have {dims} values")
        self._matrix = matrix.reshape(len(vectors), dims)

    @classmethod
    def from_rows(cls, rows: Sequence[tuple[int, bytes]], dims: int) -> "VectorIndex":
        """From (chunk_id, vector bytes) rows, as the store returns them."""
        return cls([chunk_id for chunk_id, _ in rows], [blob for _, blob in rows], dims)

    def __len__(self) -> int:
        return len(self._ids)

    def search(
        self, query: Sequence[float], k: int, allowed: Set[int] | None = None
    ) -> list[tuple[int, float]]:
        """The k rows most similar to `query` as (chunk_id, cosine), best first; equal
        scores go to the lower chunk id, so the best k is always the start of the best 2k.
        With `allowed`, only those chunks are scored (the best k among them, not a cut of
        the best k overall)."""
        ids, matrix = self._ids, self._matrix
        if allowed is not None:
            wanted = np.fromiter(allowed, dtype=np.int64, count=len(allowed))
            rows = np.flatnonzero(np.isin(ids, wanted))
            ids, matrix = ids[rows], matrix[rows]
        k = min(k, len(ids))
        if k <= 0:
            return []
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.dims,):
            raise ValueError(f"query has {q.size} values, the index {self.dims}")
        norm = float(np.linalg.norm(q))
        if norm == 0:
            return []
        scores = matrix @ (q / norm)
        if k < len(scores):
            # Every row at least as good as the k-th best (ties at the cut included), then
            # ordered by score and chunk id and cut to k.
            kth = scores[np.argpartition(-scores, k - 1)[k - 1]]
            top = np.flatnonzero(scores >= kth)
        else:
            top = np.arange(len(scores))
        best = top[np.lexsort((ids[top], -scores[top]))][:k]
        return [(int(ids[i]), float(scores[i])) for i in best]
