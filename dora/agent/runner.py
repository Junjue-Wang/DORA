"""Run an LLM agent (LangGraph ReAct) over DORA tasks and record answers + tool-call trajectories.

Output layout (one directory per task)::

    <run_dir>/<task>/benchmark.jsonl          one record per question (appended as it finishes)
    <run_dir>/<task>/trajectories/*.json      full per-question trajectory
    <run_dir>/<task>/run_stats.json
    <run_dir>/<task>/workers/w<i>/tmp/<question_id>/   tool artifacts
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from .llm import build_chat_model
from .mcp_tools import close_client, create_tool_pool, filter_tools_for_question, server_specs
from .prompts import build_prompt

log = logging.getLogger("dora")


@dataclass
class AgentConfig:
    autoplanning: bool = True          # AP; False = IF (gold tool order in the prompt, gold tools only)
    recursion_limit: int = 50          # LangGraph super-steps per question
    question_timeout_s: float = 900    # hard wall-clock limit per question
    request_timeout_s: float = 1200    # per LLM request
    rate_limit_s: float = 1.0          # pause after each question (per worker)
    filter_tools_by_trajectory: bool = True


@dataclass
class Worker:
    index: int
    root: Path
    gpu: Optional[str] = None
    llm: Any = None
    tools: List[Any] = field(default_factory=list)
    client: Any = None
    agent: Any = None

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

    async def start(self, model_cfg: Dict[str, Any], cfg: AgentConfig, gpu_slot: int = 0):
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.llm = build_chat_model(model_cfg, request_timeout=cfg.request_timeout_s)
        self.tools, self.client = await create_tool_pool(server_specs(self.tmp, self.gpu, gpu_slot))
        self.agent = create_react_agent(self.llm, self.tools)

    async def stop(self):
        await close_client(self.client)


# ---------------------------------------------------------------- message parsing


def message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in ("text", "content", "value"):
                    if key in item:
                        parts.append(str(item[key]))
                        break
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)


def _parse_json_object(text: str):
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1] if "{" in text and "}" in text else None):
        if candidate:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, TypeError):
                pass
    return None


def extract_answer(messages: List[Any]):
    """Final answer: the JSON inside ``<Answer>...</Answer>`` of the latest AI message that has one;
    otherwise the latest non-empty AI message (parsed as JSON when possible)."""
    fallback = None
    for message in reversed(messages):
        if getattr(message, "type", None) != "ai":
            continue
        content = message_text(message.content)
        if "<Answer>" in content and "</Answer>" in content:
            answer = content[content.find("<Answer>") + len("<Answer>"): content.find("</Answer>")].strip()
            parsed = _parse_json_object(answer)
            return parsed if parsed is not None else answer
        if content.strip() and fallback is None:
            fallback = content
    if fallback is not None:
        parsed = _parse_json_object(fallback[fallback.find("{"): fallback.rfind("}") + 1]) if "{" in fallback and "}" in fallback else None
        return parsed if parsed is not None else fallback
    return "No answer found"


def extract_tool_calls(message) -> List[Dict[str, Any]]:
    """Tool calls of an AI message (LangChain ``tool_calls`` and raw OpenAI ``additional_kwargs``)."""
    calls, seen = [], set()

    def add(call):
        func = call.get("function", {}) if isinstance(call, dict) else getattr(call, "function", {})
        call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
        name = func.get("name") if isinstance(func, dict) else getattr(func, "name", None)
        args = func.get("arguments") if isinstance(func, dict) else getattr(func, "arguments", None)
        if name is None:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        if args is None:
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        key = call_id or f"{name}_{len(calls)}"
        if key not in seen:
            seen.add(key)
            calls.append({"id": call_id, "name": name, "input": args})

    for call in getattr(message, "tool_calls", None) or []:
        add(call)
    for call in (getattr(message, "additional_kwargs", None) or {}).get("tool_calls", []) or []:
        add(call)
    return calls


def normalize_observation(content: Any) -> Any:
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw_output": content}
    if isinstance(content, list) and len(content) == 1:
        item = content[0]
        if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
            try:
                return json.loads(item["text"])
            except json.JSONDecodeError:
                return content
    return content


def extract_trajectory(messages: List[Any]) -> List[Dict[str, Any]]:
    """``[{call: "name(arg, ...)", args, obs}, ...]`` in call order, observations matched by call id."""
    trajectory, pending = [], {}
    for message in messages:
        kind = getattr(message, "type", None)
        if kind == "ai":
            for tc in extract_tool_calls(message):
                arg_names = ", ".join(tc["input"].keys()) if isinstance(tc["input"], dict) and tc["input"] else ""
                step = {"call": f"{tc['name']}({arg_names})", "args": tc["input"], "obs": None}
                pending[tc.get("id") or f"{tc['name']}_{len(pending)}"] = step
                trajectory.append(step)
        elif kind == "tool":
            obs = normalize_observation(message.content)
            call_id = getattr(message, "tool_call_id", None)
            if call_id and call_id in pending:
                pending[call_id]["obs"] = obs
            else:
                name = getattr(message, "name", "unknown")
                for step in reversed(trajectory):
                    if step["obs"] is None and name in step["call"]:
                        step["obs"] = obs
                        break
    return trajectory


# ---------------------------------------------------------------- one question


async def answer_question(worker: Worker, question: Dict[str, Any], cfg: AgentConfig) -> Dict[str, Any]:
    qid = question["question_id"]
    result = {
        "question_id": qid,
        "task_scoped_question_id": question["task_scoped_question_id"],
        "task_category": question["task_category"],
        "task_category_code": question["task_category_code"],
        "sample_id": question["sample_id"],
        "worker_id": worker.index,
        "question": question["question"],
        "predicted_answer": None,
        "error": None,
        "trajectory": [],
        "elapsed_time": 0,
    }
    agent = worker.agent
    if not cfg.autoplanning and cfg.filter_tools_by_trajectory:
        agent = create_react_agent(worker.llm, filter_tools_for_question(worker.tools, question))

    # Tool servers write this question's artifacts to tmp/<question_id>/ while the session file exists.
    session_file = worker.tmp / ".session"
    session_file.write_text(qid.replace("/", "_"))
    started = time.perf_counter()
    try:
        prompt = build_prompt(question, cfg.autoplanning)
        try:
            response = await asyncio.wait_for(
                agent.ainvoke({"messages": [HumanMessage(content=prompt)]},
                              config={"recursion_limit": cfg.recursion_limit}),
                timeout=cfg.question_timeout_s,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"question_timeout after {cfg.question_timeout_s}s (likely stuck MCP tool call)")
        messages = response.get("messages", [])
        result["trajectory"] = extract_trajectory(messages)
        result["predicted_answer"] = extract_answer(messages)
        log.info("done %s: %d tool calls", qid, len(result["trajectory"]))
    except Exception as e:
        log.error("error %s: %s", qid, e)
        result["error"] = str(e)
        result["predicted_answer"] = f"Error: {e}"
    finally:
        result["elapsed_time"] = time.perf_counter() - started
        session_file.unlink(missing_ok=True)
    return result


# ---------------------------------------------------------------- one task


def completed_ids(jsonl_path: Path) -> set:
    """Questions already answered without error (for --resume)."""
    done = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("predicted_answer") and not record.get("error"):
                done.add(record.get("task_scoped_question_id") or record.get("question_id"))
    return done


def _write_result(out_dir: Path, result: Dict[str, Any]) -> None:
    record = {k: result[k] for k in ("question_id", "task_scoped_question_id", "task_category",
                                     "task_category_code", "sample_id", "worker_id", "question",
                                     "predicted_answer", "error", "trajectory", "elapsed_time")}
    record["timestamp"] = datetime.now().isoformat()
    with open(out_dir / "benchmark.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    stem = result["task_scoped_question_id"].replace("::", "__")
    trajectory = {k: v for k, v in record.items() if k != "predicted_answer"}
    trajectory["final_answer"] = result["predicted_answer"]
    with open(traj_dir / f"{stem}_trajectory.json", "w", encoding="utf-8") as f:
        json.dump(trajectory, f, ensure_ascii=False, indent=2)


def _write_stats(out_dir: Path) -> Dict[str, Any]:
    records = []
    jsonl = out_dir / "benchmark.jsonl"
    if jsonl.exists():
        records = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    stats = {
        "total": len(records),
        "success": sum(1 for r in records if r.get("predicted_answer") and not r.get("error")),
        "failed": sum(1 for r in records if r.get("error") is not None),
        "avg_time": sum(r.get("elapsed_time", 0) for r in records) / len(records) if records else 0,
        "timestamp": datetime.now().isoformat(),
    }
    (out_dir / "run_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


async def run_task(questions: List[Dict[str, Any]], out_dir: Path, model_cfg: Dict[str, Any], cfg: AgentConfig,
                   workers: int = 1, gpus: Optional[List[str]] = None, resume: bool = False) -> Dict[str, Any]:
    """Answer ``questions`` with ``workers`` parallel agents, each owning its own tool servers.

    Worker ``i`` pins its Perception server to ``gpus[i % len(gpus)]`` (``gpus`` empty -> CPU,
    ``None`` -> inherit ``CUDA_VISIBLE_DEVICES``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        done = completed_ids(out_dir / "benchmark.jsonl")
        questions = [q for q in questions if q["task_scoped_question_id"] not in done]
        log.info("resume: %d already answered, %d remaining", len(done), len(questions))
    if not questions:
        return _write_stats(out_dir)

    workers = max(1, min(workers, len(questions)))
    pool = []
    for i in range(workers):
        gpu = None if gpus is None else (gpus[i % len(gpus)] if gpus else "")
        pool.append(Worker(index=i, root=out_dir / "workers" / f"w{i}", gpu=gpu))
    queues = [questions[i::workers] for i in range(workers)]
    progress = {"done": 0}

    async def work(worker: Worker, queue: List[Dict[str, Any]]):
        await worker.start(model_cfg, cfg, gpu_slot=worker.index % max(1, len(gpus or [0])))
        try:
            for question in queue:
                result = await answer_question(worker, question, cfg)
                _write_result(out_dir, result)
                progress["done"] += 1
                log.info("[%s] %d/%d", out_dir.name, progress["done"], len(questions))
                if cfg.rate_limit_s > 0:
                    await asyncio.sleep(cfg.rate_limit_s)
        finally:
            await worker.stop()

    await asyncio.gather(*(work(w, q) for w, q in zip(pool, queues)))
    return _write_stats(out_dir)
