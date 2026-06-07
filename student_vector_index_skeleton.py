import numpy as np
from typing import Dict, List


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

    def __init__(self, dim: int):
        self.dim = int(dim)
        # TODO: initialize your index data structures

    def insert(self, batch: Dict[int, np.ndarray]) -> Dict[str, List[int]]:
        """Return {"succeeded": [...], "failed": [...]} preserving input order per list."""
        raise NotImplementedError

    def delete(self, ids: np.ndarray) -> Dict[str, List[int]]:
        """Return {"succeeded": [...], "failed": [...]} preserving input order per list."""
        raise NotImplementedError

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Return (num_queries, min(k, n_active)) int64 array of vector IDs."""
        raise NotImplementedError
