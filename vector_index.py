import numpy as np
from typing import Dict, List, Tuple


class VectorIndex:
    """
    Dynamic vector index (Section A).

    Rules:
    - Dot-product similarity on L2-normalized vectors.
    - insert: succeeds iff ID does not exist; duplicate IDs in one batch must not occur in data.
    - delete: succeeds iff ID exists; non-existing IDs must not crash.
    - search: return shape (num_queries, min(k, n_active)); IDs sorted by descending dot product.
    - No line-count limit is enforced; keep the implementation clean and modular.
    """
    _SEARCH_BLOCK_SIZE = 131072
    _INITIAL_CAPACITY_FACTOR = 1.35

    def __init__(self, dim: int):
        self.dim = int(dim)
        self._vectors = np.empty((0, self.dim), dtype=np.float32)
        self._ids = np.empty(0, dtype=np.int64)
        self._active = np.empty(0, dtype=bool)
        self._row_by_id: Dict[int, int] = {}
        self._size = 0
        self._active_count = 0

    def insert(self, batch: Dict[int, np.ndarray]) -> Dict[str, List[int]]:
        """Return {"succeeded": [...], "failed": [...]} preserving input order per list."""
        succeeded: List[int] = []
        failed: List[int] = []
        new_ids: List[int] = []
        new_vectors: List[np.ndarray] = []
        for vid, vec in batch.items():
            vid = int(vid)
            if vid in self._row_by_id:
                failed.append(vid)
            else:
                succeeded.append(vid)
                new_ids.append(vid)
                new_vectors.append(np.asarray(vec, dtype=np.float32))
        if new_ids:
            self._append(np.asarray(new_ids, dtype=np.int64), np.vstack(new_vectors))
        return {"succeeded": succeeded, "failed": failed}

    def delete(self, ids: np.ndarray) -> Dict[str, List[int]]:
        """Return {"succeeded": [...], "failed": [...]} preserving input order per list."""
        succeeded: List[int] = []
        failed: List[int] = []
        for raw_id in np.asarray(ids, dtype=np.int64):
            vid = int(raw_id)
            row = self._row_by_id.pop(vid, None)
            if row is None:
                failed.append(vid)
            else:
                self._active[row] = False
                self._active_count -= 1
                succeeded.append(vid)
        return {"succeeded": succeeded, "failed": failed}

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Return (num_queries, min(k, n_active)) int64 array of vector IDs."""
        queries = np.asarray(queries, dtype=np.float32)
        k_eff = min(int(k), self._active_count)
        if k_eff <= 0:
            return np.empty((queries.shape[0], 0), dtype=np.int64)
        self._compact_if_needed()
        return self._exact_search(queries, k_eff)

    def _append(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        start = self._size
        rows = np.arange(start, start + len(ids), dtype=np.int64)
        self._ensure_capacity(self._size + len(ids))
        self._vectors[rows] = vectors
        self._ids[rows] = ids
        self._active[rows] = True
        self._size += len(ids)
        self._active_count += len(ids)
        self._row_by_id.update((int(vid), int(row)) for vid, row in zip(ids, rows))

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= len(self._ids):
            return
        capacity = max(
            1024,
            int(np.ceil(needed * self._INITIAL_CAPACITY_FACTOR)),
            int(max(1, len(self._ids)) * 1.5),
        )
        vectors = np.empty((capacity, self.dim), dtype=np.float32)
        ids = np.empty(capacity, dtype=np.int64)
        active = np.zeros(capacity, dtype=bool)
        vectors[: self._size] = self._vectors[: self._size]
        ids[: self._size] = self._ids[: self._size]
        active[: self._size] = self._active[: self._size]
        self._vectors, self._ids, self._active = vectors, ids, active

    def _compact_if_needed(self) -> None:
        inactive_count = self._size - self._active_count
        if inactive_count < max(10000, self._size // 20):
            return
        rows = np.flatnonzero(self._active[: self._size])
        n_rows = len(rows)
        self._vectors[:n_rows] = self._vectors[rows]
        self._ids[:n_rows] = self._ids[rows]
        self._active[:n_rows] = True
        self._active[n_rows : self._size] = False
        self._size = n_rows
        self._row_by_id = {int(vid): row for row, vid in enumerate(self._ids[:n_rows])}

    def _exact_search(self, queries: np.ndarray, k_eff: int) -> np.ndarray:
        if self._active_count == self._size:
            vectors = self._vectors[: self._size]
            ids = self._ids[: self._size]
        else:
            rows = np.flatnonzero(self._active[: self._size])
            vectors = self._vectors[rows]
            ids = self._ids[rows]
        best_scores = np.full((len(queries), k_eff), -np.inf, dtype=np.float32)
        best_ids = np.empty((len(queries), k_eff), dtype=np.int64)
        for start in range(0, len(vectors), self._SEARCH_BLOCK_SIZE):
            stop = min(start + self._SEARCH_BLOCK_SIZE, len(vectors))
            best_scores, best_ids = self._merge_block(
                queries, vectors[start:stop], ids[start:stop], best_scores, best_ids, k_eff
            )
        order = np.argsort(-best_scores, axis=1)
        return np.take_along_axis(best_ids, order, axis=1)

    def _merge_block(
        self,
        queries: np.ndarray,
        vectors: np.ndarray,
        ids: np.ndarray,
        best_scores: np.ndarray,
        best_ids: np.ndarray,
        k_eff: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores = queries @ vectors.T
        if len(ids) > k_eff:
            top = np.argpartition(-scores, k_eff - 1, axis=1)[:, :k_eff]
            scores = np.take_along_axis(scores, top, axis=1)
            ids = ids[top]
        else:
            ids = np.broadcast_to(ids, scores.shape)
        scores = np.concatenate((best_scores, scores), axis=1)
        ids = np.concatenate((best_ids, ids), axis=1)
        top = np.argpartition(-scores, k_eff - 1, axis=1)[:, :k_eff]
        return np.take_along_axis(scores, top, axis=1), np.take_along_axis(ids, top, axis=1)
