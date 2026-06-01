import numpy as np
from typing import Dict, List


class VectorIndex:
    """Naive baseline: dict storage + full scan search (do not modify for grading)."""

    def __init__(self, dim: int):
        self.dim = int(dim)
        self._store: Dict[int, np.ndarray] = {}

    def insert(self, batch: Dict[int, np.ndarray]) -> Dict[str, List[int]]:
        succeeded: List[int] = []
        failed: List[int] = []
        for vid, vec in batch.items():
            vid = int(vid)
            vec = np.asarray(vec, dtype=np.float32)
            if vid in self._store:
                failed.append(vid)
            else:
                self._store[vid] = vec
                succeeded.append(vid)
        return {"succeeded": succeeded, "failed": failed}

    def delete(self, ids: np.ndarray) -> Dict[str, List[int]]:
        succeeded: List[int] = []
        failed: List[int] = []
        for vid in np.asarray(ids, dtype=np.int64):
            vid = int(vid)
            if vid in self._store:
                del self._store[vid]
                succeeded.append(vid)
            else:
                failed.append(vid)
        return {"succeeded": succeeded, "failed": failed}

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        queries = np.asarray(queries, dtype=np.float32)
        n_q = queries.shape[0]
        if not self._store:
            return np.empty((n_q, 0), dtype=np.int64)
        ids = np.fromiter(self._store.keys(), dtype=np.int64)
        vectors = np.stack([self._store[int(i)] for i in ids])
        scores = queries @ vectors.T
        n_active = len(ids)
        k_eff = min(int(k), n_active)
        topk_unsorted = np.argpartition(-scores, kth=k_eff - 1, axis=1)[:, :k_eff]
        topk_scores = np.take_along_axis(scores, topk_unsorted, axis=1)
        order = np.argsort(-topk_scores, axis=1)
        topk_pos = np.take_along_axis(topk_unsorted, order, axis=1)
        return ids[topk_pos]
