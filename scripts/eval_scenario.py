"""Evaluate a public scenario (student self-test)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from evaluation import (
    DEFAULT_BASELINE_RUNS,
    SCENARIO_WALL_LIMIT_SEC,
    get_local_baseline,
    get_local_baseline_stats,
    run_scenario,
)
from naive_vector_index import VectorIndex as NaiveVectorIndex
from vector_index import VectorIndex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, required=True)
    parser.add_argument(
        "--recalibrate-baseline",
        action="store_true",
        help=f"Re-run naive baseline ({DEFAULT_BASELINE_RUNS} runs) instead of using cache",
    )
    args = parser.parse_args()
    scenario_dir = STUDENT_ROOT / "data" / "public" / f"scenario_{args.scenario:02d}"
    if not scenario_dir.is_dir():
        raise FileNotFoundError(scenario_dir)

    print(f"Measuring local naive baseline ({DEFAULT_BASELINE_RUNS} runs)...")
    baseline_stats = get_local_baseline_stats(
        scenario_dir,
        NaiveVectorIndex,
        recalibrate=args.recalibrate_baseline,
    )
    baseline_dynamic = baseline_stats["baseline_dynamic"]
    print(
        f"baseline_dynamic={baseline_dynamic:.4f}s "
        f"(initial={baseline_stats['baseline_initial']:.4f}s, "
        f"total={baseline_stats['baseline_total']:.4f}s, naive average)"
    )

    stats = run_scenario(scenario_dir, VectorIndex, baseline_time=baseline_dynamic)
    print(f"scenario={args.scenario:02d} (public)")
    print(f"insert_score={stats['insert_score']:.4f}")
    print(f"delete_score={stats['delete_score']:.4f}")
    print(f"search_score={stats['search_score']:.4f}")
    print(f"functional_score={stats['functional_score']:.4f}")
    print(f"runtime_multiplier={stats['runtime_multiplier']:.4f}")
    print(f"final_score={stats['final_score']:.4f}")
    print(f"your_initial_time={stats['initial_time']:.4f}s (untimed)")
    print(f"your_dynamic_time={stats['dynamic_time']:.4f}s (graded)")
    print(f"your_total_time={stats['total_time']:.4f}s")
    print(f"scenario_wall_time={stats['scenario_wall_time']:.4f}s")
    print(
        f"wall_timeout={bool(stats['wall_timeout'])} "
        f"(limit={SCENARIO_WALL_LIMIT_SEC:.0f}s)"
    )
    print(f"speed_ratio={stats['speed_ratio']:.4f}")


if __name__ == "__main__":
    main()
