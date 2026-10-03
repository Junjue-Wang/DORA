"""Benchmark task files -> agent questions.

Each ``tasks/<task>.json`` holds samples; a sample bundles an ``input_data`` manifest
(paths relative to ``images/``) and one or more tasks with question, gold tool plan
(``tools``), executed gold trajectory, final answer and ``eval_spec``.
"""
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

from .paths import IMAGES_DIR, TASKS_DIR

TASKS = {
    "t1": "t1_perception_and_assessment",
    "t2": "t2_spatial_relation_analysis",
    "t3": "t3_operational_planning",
    "t4": "t4_multi_temporal_reasoning",
    "t5": "t5_multi_modal_report_synthesis",
}
PATH_SUFFIXES = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".geojson", ".json", ".shp")


def task_name(task: str) -> str:
    """Accept ``t1`` or ``t1_perception_and_assessment``."""
    return TASKS.get(task, task)


def load_samples(task: str, tasks_dir: Path = TASKS_DIR) -> List[Dict[str, Any]]:
    with open(Path(tasks_dir) / f"{task_name(task)}.json", encoding="utf-8") as f:
        return json.load(f)


def resolve_paths(value: Any, images_dir: Path = IMAGES_DIR) -> Any:
    """Turn every relative file path inside an ``input_data`` manifest into an absolute path."""
    if isinstance(value, dict):
        return {k: resolve_paths(v, images_dir) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_paths(v, images_dir) for v in value]
    if isinstance(value, str) and value.lower().endswith(PATH_SUFFIXES):
        return str(Path(images_dir) / value)
    return value


def normalize_classes_for_tools(classes: Dict[str, Any]):
    """Map dataset road labels (10/11/12) to the road-damage tool's output classes (1/2/3)."""
    mapped = copy.deepcopy(classes)
    note = ""
    roads = mapped.get("roads")
    if isinstance(roads, dict) and {str(k) for k in roads} == {"10", "11", "12"}:
        mapped["roads"] = {"1": "intact", "2": "flooded", "3": "blocked"}
        note = "Road segmentation tool outputs classes 1/2/3 (intact/flooded/blocked)."
    return mapped, note


def sample_context(sample: Dict[str, Any], images_dir: Path = IMAGES_DIR):
    """Metadata block shown to the agent (everything but the tasks, with absolute input paths)."""
    classes, road_note = normalize_classes_for_tools(sample.get("classes", {}))
    metadata = {k: copy.deepcopy(v) for k, v in sample.items() if k != "tasks"}
    if metadata.get("input_data"):
        metadata["input_data"] = resolve_paths(metadata["input_data"], images_dir)
    if classes:
        metadata["classes"] = classes
    metadata_str = json.dumps(metadata, indent=2, ensure_ascii=False)
    context = f"""
            Sample Metadata:
            {metadata_str}
            {road_note}
            """
    return context, metadata


def load_questions(task: str, tasks_dir: Path = TASKS_DIR, images_dir: Path = IMAGES_DIR) -> List[Dict[str, Any]]:
    """One question per (sample, task); ``question_id = <sample_id>_task<i>``."""
    questions = []
    for sample in load_samples(task, tasks_dir):
        context, metadata = sample_context(sample, images_dir)
        for idx, t in enumerate(sample["tasks"]):
            question_id = f"{sample['sample_id']}_task{idx}"
            questions.append({
                "question_id": question_id,
                "task_scoped_question_id": f"{t['task_category']}::{question_id}",
                "task_category": t["task_category"],
                "task_category_code": t["task_category_code"],
                "sample_id": sample["sample_id"],
                "question": t["question"],
                "context": context,
                "input_data": metadata.get("input_data", {}),
                "eval_spec": t.get("eval_spec", {}),
                "tools": t.get("tools", []),
                "expected_trajectory": t.get("trajectory", []),
            })
    return questions


def missing_input_files(question: Dict[str, Any]) -> List[str]:
    """Absolute input paths of a question that do not exist on disk."""
    missing = []

    def walk(value):
        if isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        elif isinstance(value, str) and value.lower().endswith(PATH_SUFFIXES) and not Path(value).exists():
            missing.append(value)

    walk(question.get("input_data"))
    return missing
