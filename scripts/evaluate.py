#!/usr/bin/env python
"""Score a DORA run (answers + trajectories).

    python scripts/evaluate.py --run outputs/gemini-3-flash_AP            # -> outputs/gemini-3-flash_AP/eval.json
    python scripts/evaluate.py --sanity                                   # gold vs gold: every metric must be 1.0

``--run`` points to the directory holding the five ``<task>/benchmark.jsonl`` files.
"""
import argparse
import json
import sys
from pathlib import Path

from dora import evaluation
from dora.paths import TASKS_DIR


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=Path, help="run directory produced by scripts/run_benchmark.py")
    group.add_argument("--sanity", action="store_true", help="evaluate the gold trajectories against themselves")
    p.add_argument("--tasks-dir", type=Path, default=TASKS_DIR, help="gold task files (default: <data>/tasks)")
    p.add_argument("--output", type=Path, default=None, help="result JSON (default: <run>/eval.json)")
    p.add_argument("--tau-r", type=float, default=evaluation.TAU_R, help="relative tolerance for scalar fields")
    p.add_argument("--tau-a", type=float, default=evaluation.TAU_A, help="absolute tolerance for scalar fields")
    args = p.parse_args()

    if args.run is not None and not args.run.is_dir():
        sys.exit(f"run directory not found: {args.run}")
    evaluation._update_tolerance(args.tau_r, args.tau_a)
    results = evaluation.run_evaluation(args.tasks_dir, None if args.sanity else args.run)
    table = evaluation.format_results_table(results)
    print(table)

    output = args.output or (Path("outputs/eval_sanity.json") if args.sanity else args.run / "eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary_table": table, **results}, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".txt").write_text(table + "\n", encoding="utf-8")
    print(f"\nresults: {output}")


if __name__ == "__main__":
    main()
