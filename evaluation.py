"""Shared evaluation utilities for Section A scenarios."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, TypedDict

import numpy as np

K_EVAL = 10
BASELINE_CACHE_NAME = ".baseline_local.json"
DEFAULT_BASELINE_RUNS = 3
BASELINE_CACHE_VERSION = 2
SCENARIO_WALL_LIMIT_SEC = 50.0


class BaselineStats(TypedDict):
    baseline_initial: float
    baseline_dynamic: float
    baseline_total: float


def jaccard(a: List[int], b: List[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def recall_at_k(results: np.ndarray, ground_truth: np.ndarray, k: int = K_EVAL) -> float:
    results = np.asarray(results, dtype=np.int64)
    ground_truth = np.asarray(ground_truth, dtype=np.int64)
    scores: List[float] = []
    for res_row, gt_row in zip(results, ground_truth):
        gt_top = gt_row
        k_eff = min(k, len(gt_top))
        if k_eff == 0:
            scores.append(1.0)
            continue
        res_top = res_row[:k_eff]
        gt_eval = gt_top[:k_eff]
        scores.append(len(set(res_top.tolist()) & set(gt_eval.tolist())) / k_eff)
    return float(np.mean(scores))


def runtime_multiplier(ratio: float) -> float:
    if ratio <= 0.5:
        return 1.0
    if ratio <= 1.0:
        return 0.9
    if ratio <= 2.0:
        return 0.75
    return 0.5


def load_batch_dict(path: Path) -> Dict[int, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {int(item[0]): np.asarray(item[1], dtype=np.float32) for item in arr}


def run_scenario(
    scenario_dir: Path,
    index_factory: Callable[[int], object],
    *,
    baseline_time: Optional[float] = None,
    wall_limit_sec: Optional[float] = SCENARIO_WALL_LIMIT_SEC,
) -> Dict[str, float]:
    """Replay a scenario and score correctness + runtime.

    Initial bulk insert (first manifest insert, batch0) is timed for reporting only.
    Runtime penalty uses the dynamic phase: later inserts, deletes, and searches.
    ``baseline_time`` must be the naive dynamic-phase average from ``get_local_baseline``.

    When ``wall_limit_sec`` is set (default 50s for graded runs), the entire scenario
    must finish within that wall clock; exceeding it forces the minimum runtime multiplier.
    Pass ``wall_limit_sec=None`` for baseline calibration replays.
    """
    manifest = json.loads((scenario_dir / "manifest.json").read_text())
    cfg = manifest["config"]
    dim = int(manifest["dim"])
    total_insert = max(int(cfg["total_insert"]), 1)
    total_delete = max(int(cfg["total_delete"]), 1)
    total_search = max(int(cfg["total_search"]), 1)

    if baseline_time is None:
        raise ValueError(
            "baseline_time is required. Use get_local_baseline() before scoring, "
            "or pass baseline_time=1.0 when measuring naive replay time only."
        )

    index = index_factory(dim)
    insert_score = 0.0
    delete_score = 0.0
    search_scores: List[float] = []
    t_initial = 0.0
    t_insert_dynamic = t_delete = t_search = 0.0
    seen_initial_insert = False
    wall_timeout = False
    t_scenario_start = time.perf_counter()

    for op in manifest["operations"]:
        if (
            wall_limit_sec is not None
            and time.perf_counter() - t_scenario_start > wall_limit_sec
        ):
            wall_timeout = True
            break
        if op["type"] == "insert":
            batch = load_batch_dict(scenario_dir / op["batch"])
            gt = json.loads((scenario_dir / op["gt"]).read_text())
            bs = len(batch)
            t0 = time.perf_counter()
            out = index.insert(batch)
            elapsed = time.perf_counter() - t0
            if not seen_initial_insert:
                t_initial += elapsed
                seen_initial_insert = True
            else:
                t_insert_dynamic += elapsed
            inc = (bs / total_insert) * 0.5 * (
                jaccard(gt["succeeded"], out["succeeded"])
                + jaccard(gt["failed"], out["failed"])
            )
            insert_score += inc
        elif op["type"] == "delete":
            ids = np.load(scenario_dir / op["ids"])
            gt = json.loads((scenario_dir / op["gt"]).read_text())
            bs = len(ids)
            t0 = time.perf_counter()
            out = index.delete(ids)
            t_delete += time.perf_counter() - t0
            inc = (bs / total_delete) * 0.5 * (
                jaccard(gt["succeeded"], out["succeeded"])
                + jaccard(gt["failed"], out["failed"])
            )
            delete_score += inc
        else:
            queries = np.load(scenario_dir / op["queries"])
            gt = np.load(scenario_dir / op["gt"])
            nq = queries.shape[0]
            t0 = time.perf_counter()
            res = index.search(queries, K_EVAL)
            t_search += time.perf_counter() - t0
            r = recall_at_k(res, gt[:, :K_EVAL], K_EVAL)
            search_scores.append((nq / total_search) * r)
        if (
            wall_limit_sec is not None
            and time.perf_counter() - t_scenario_start > wall_limit_sec
        ):
            wall_timeout = True
            break

    scenario_wall_time = time.perf_counter() - t_scenario_start
    if wall_limit_sec is not None and scenario_wall_time > wall_limit_sec:
        wall_timeout = True

    functional = 0.5 * sum(search_scores) + 0.3 * insert_score + 0.2 * delete_score
    dynamic_time = t_insert_dynamic + t_delete + t_search
    total_time = t_initial + dynamic_time
    ratio = dynamic_time / max(baseline_time, 1e-9)
    mult = runtime_multiplier(ratio)
    if wall_timeout:
        mult = 0.5
    final = functional * mult
    return {
        "insert_score": insert_score,
        "delete_score": delete_score,
        "search_score": sum(search_scores),
        "functional_score": functional,
        "initial_time": t_initial,
        "dynamic_time": dynamic_time,
        "total_time": total_time,
        "scenario_wall_time": scenario_wall_time,
        "wall_timeout": float(wall_timeout),
        "runtime_multiplier": mult,
        "final_score": final,
        "baseline_time": baseline_time,
        "speed_ratio": ratio,
    }


def measure_local_baseline(
    scenario_dir: Path,
    naive_factory: Callable[[int], object],
    *,
    n_runs: int = DEFAULT_BASELINE_RUNS,
) -> BaselineStats:
    """Run naive replay ``n_runs`` times; return mean phase times (machine-local)."""
    initial_times: List[float] = []
    dynamic_times: List[float] = []
    total_times: List[float] = []
    for _ in range(n_runs):
        stats = run_scenario(
            scenario_dir,
            naive_factory,
            baseline_time=1.0,
            wall_limit_sec=None,
        )
        initial_times.append(stats["initial_time"])
        dynamic_times.append(stats["dynamic_time"])
        total_times.append(stats["total_time"])
    n = float(len(initial_times))
    return {
        "baseline_initial": float(sum(initial_times) / n),
        "baseline_dynamic": float(sum(dynamic_times) / n),
        "baseline_total": float(sum(total_times) / n),
    }


def _read_baseline_cache(cache_path: Path, n_runs: int) -> Optional[BaselineStats]:
    if not cache_path.is_file():
        return None
    data = json.loads(cache_path.read_text())
    if int(data.get("n_runs", 0)) != n_runs:
        return None
    if int(data.get("cache_version", 0)) >= BASELINE_CACHE_VERSION:
        return {
            "baseline_initial": float(data["baseline_initial"]),
            "baseline_dynamic": float(data["baseline_dynamic"]),
            "baseline_total": float(data["baseline_total"]),
        }
    return None


def get_local_baseline(
    scenario_dir: Path,
    naive_factory: Callable[[int], object],
    *,
    n_runs: int = DEFAULT_BASELINE_RUNS,
    recalibrate: bool = False,
) -> float:
    """Return cached naive dynamic-phase baseline, or measure and cache it.

    The returned float is ``baseline_dynamic`` (used for runtime scoring).
    Full phase stats are stored in ``.baseline_local.json``.
    """
    cache_path = scenario_dir / BASELINE_CACHE_NAME
    if not recalibrate:
        cached = _read_baseline_cache(cache_path, n_runs)
        if cached is not None:
            return cached["baseline_dynamic"]

    stats = measure_local_baseline(scenario_dir, naive_factory, n_runs=n_runs)
    cache_path.write_text(
        json.dumps(
            {
                "cache_version": BASELINE_CACHE_VERSION,
                "n_runs": n_runs,
                "baseline_initial": stats["baseline_initial"],
                "baseline_dynamic": stats["baseline_dynamic"],
                "baseline_total": stats["baseline_total"],
                "baseline_time": stats["baseline_dynamic"],
            },
            indent=2,
        )
    )
    return stats["baseline_dynamic"]


def get_local_baseline_stats(
    scenario_dir: Path,
    naive_factory: Callable[[int], object],
    *,
    n_runs: int = DEFAULT_BASELINE_RUNS,
    recalibrate: bool = False,
) -> BaselineStats:
    """Like ``get_local_baseline`` but returns all cached phase averages."""
    get_local_baseline(
        scenario_dir,
        naive_factory,
        n_runs=n_runs,
        recalibrate=recalibrate,
    )
    cached = _read_baseline_cache(scenario_dir / BASELINE_CACHE_NAME, n_runs)
    if cached is None:
        raise RuntimeError("baseline cache missing after measurement")
    return cached
