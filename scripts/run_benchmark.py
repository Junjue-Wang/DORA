#!/usr/bin/env python
"""Run an LLM agent on DORA.

Examples:
    # validate data, checkpoints and the 108 MCP tools (no API key needed)
    python scripts/run_benchmark.py --model configs/models/gemini-3-flash.json --dry-run

    # all five task dimensions, autonomous planning (AP)
    python scripts/run_benchmark.py --model configs/models/gemini-3-flash.json

    # instruction following (gold tool order given), two workers on two GPUs, resumable
    python scripts/run_benchmark.py --model configs/models/gemini-3-flash.json --mode if --workers 2 --gpus 0,1 --resume
"""
import argparse
import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path

from dora.agent.mcp_tools import close_client, create_tool_pool, server_specs
from dora.agent.runner import AgentConfig, run_task
from dora.benchmark import TASKS, load_questions, missing_input_files, task_name
from dora.paths import CHECKPOINTS_DIR, DATA_ROOT
from dora.tools.segmentation import MODEL_ZOO

log = logging.getLogger("dora")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="model config JSON (see configs/models/)")
    p.add_argument("--tasks", nargs="+", default=list(TASKS), help="task dimensions, e.g. t1 t3 (default: all)")
    p.add_argument("--mode", choices=["ap", "if"], default="ap",
                   help="ap: autonomous planning; if: gold tool order in the prompt + gold tools only")
    p.add_argument("--workers", type=int, default=1, help="parallel agents, each with its own tool servers")
    p.add_argument("--gpus", default=None,
                   help="comma-separated GPU ids assigned round-robin to workers' perception servers; "
                        "'none' for CPU; default: inherit CUDA_VISIBLE_DEVICES")
    p.add_argument("--output-dir", default="outputs", help="root of run directories")
    p.add_argument("--run-name", default=None, help="run directory name (default: <model>_<MODE>)")
    p.add_argument("--resume", action="store_true", help="skip questions already answered in the run directory")
    p.add_argument("--question-ids", nargs="+", default=None, help="only these question ids")
    p.add_argument("--limit", type=int, default=None, help="only the first N questions of each task")
    p.add_argument("--rate-limit", type=float, default=1.0, help="seconds to wait after each question (per worker)")
    p.add_argument("--dry-run", action="store_true", help="check data, checkpoints and tool servers, then exit")
    return p.parse_args()


async def dry_run(questions_by_task):
    missing = {q["task_scoped_question_id"]: m for qs in questions_by_task.values() for q in qs if (m := missing_input_files(q))}
    for qid, files in list(missing.items())[:10]:
        log.error("missing inputs for %s: %s", qid, files[:3])
    ckpts = [f for _, f in MODEL_ZOO.values() if not (CHECKPOINTS_DIR / f).is_file()]
    for f in ckpts:
        log.error("missing checkpoint: %s", CHECKPOINTS_DIR / f)
    with tempfile.TemporaryDirectory() as tmp:
        tools, client = await create_tool_pool(server_specs(Path(tmp)))
        await close_client(client)
    n_questions = sum(len(q) for q in questions_by_task.values())
    log.info("data root: %s", DATA_ROOT)
    log.info("%d questions, %d with missing inputs, %d/%d checkpoints present, %d tools loaded",
             n_questions, len(missing), len(MODEL_ZOO) - len(ckpts), len(MODEL_ZOO), len(tools))
    return 0 if not missing and not ckpts and len(tools) == 108 else 1


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    for noisy in ("httpx", "mcp", "langchain_mcp_adapters"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    model_cfg = json.loads(Path(args.model).read_text(encoding="utf-8"))
    cfg = AgentConfig(autoplanning=args.mode == "ap", rate_limit_s=args.rate_limit, **model_cfg.get("agent", {}))

    questions_by_task = {}
    for task in args.tasks:
        questions = load_questions(task)
        if args.question_ids:
            questions = [q for q in questions if q["question_id"] in args.question_ids]
        questions_by_task[task_name(task)] = questions[: args.limit]

    if args.dry_run:
        sys.exit(asyncio.run(dry_run(questions_by_task)))

    run_dir = Path(args.output_dir) / (args.run_name or f"{model_cfg['name']}_{args.mode.upper()}")
    if run_dir.exists() and not args.resume:
        sys.exit(f"{run_dir} exists: pass --resume to continue it or --run-name for a new run")
    gpus = None if args.gpus is None else ([] if args.gpus == "none" else args.gpus.split(","))

    for task, questions in questions_by_task.items():
        log.info("== %s: %d questions -> %s", task, len(questions), run_dir / task)
        stats = asyncio.run(run_task(questions, run_dir / task, model_cfg, cfg,
                                     workers=args.workers, gpus=gpus, resume=args.resume))
        log.info("== %s done: %s", task, stats)
    log.info("evaluate with: python scripts/evaluate.py --run %s", run_dir)


if __name__ == "__main__":
    main()
