"""
Model MCP server: LLM-backed evidence extraction and report summarisation (Gemini / OpenAI).
"""
import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._common import session_temp_dir
from ._types import ListDict, OptListInt, OptListStr, OptDictStrAny, OptDictAnyStr

import numpy as np
from PIL import Image
from fastmcp import FastMCP

mcp = FastMCP()
TEMP_DIR = session_temp_dir()

# Gemini API availability
try:
    from google import genai
    GENAI_OK = True
except Exception:
    genai = None
    GENAI_OK = False

try:
    from openai import AzureOpenAI, OpenAI
    OPENAI_OK = True
except Exception:
    AzureOpenAI = None
    OpenAI = None
    OPENAI_OK = False


GEMINI_MODEL = "gemini-3-flash-preview"

GPT_MODEL = "gpt-5.4-nano"


def _normalize_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return np.zeros(arr.shape, dtype=np.uint8)
    valid = arr[finite_mask]
    lo = float(valid.min())
    hi = float(valid.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - lo) / (hi - lo)
    scaled = np.clip(scaled * 255.0, 0, 255)
    return scaled.astype(np.uint8)


def _image_to_part(image_path: str):
    from google.genai import types

    img = Image.open(image_path)
    arr = np.array(img)

    if arr.ndim == 2:
        arr = np.stack([_normalize_uint8(arr)] * 3, axis=-1)
    elif arr.ndim == 3:
        if arr.shape[-1] > 4:
            arr = arr[..., :3]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.shape[-1] == 2:
            arr = np.concatenate([arr, arr[..., :1]], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        arr = _normalize_uint8(arr)
    else:
        raise ValueError(f"Unsupported image rank: {arr.ndim}")

    display_img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    display_img.save(buf, format="PNG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")


def _image_to_data_url(image_path: str) -> str:
    img = Image.open(image_path)
    arr = np.array(img)

    if arr.ndim == 2:
        arr = np.stack([_normalize_uint8(arr)] * 3, axis=-1)
    elif arr.ndim == 3:
        if arr.shape[-1] > 4:
            arr = arr[..., :3]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.shape[-1] == 2:
            arr = np.concatenate([arr, arr[..., :1]], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        arr = _normalize_uint8(arr)
    else:
        raise ValueError(f"Unsupported image rank: {arr.ndim}")

    display_img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    display_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _call_gemini(
    prompt: str,
    image_paths: OptListStr = None,
    model: str = GEMINI_MODEL,
    temperature: float = 0.2,
) -> str:
    """Call Gemini API for text generation."""
    if not GENAI_OK:
        raise ImportError("google-genai is not installed. Install with: pip install google-genai")

    vertexai = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"

    if vertexai:
        try:
            client = genai.Client(
                vertexai=True,
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
                project=os.environ.get("GOOGLE_CLOUD_PROJECT", "default-project"),
            )
        except Exception:
            vertexai = False

    if not vertexai:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

    contents: Any
    image_paths = [p for p in (image_paths or []) if isinstance(p, str) and os.path.exists(p)]
    if image_paths:
        from google.genai import types

        parts = [types.Part.from_text(text=prompt)]
        for image_path in image_paths:
            try:
                parts.append(_image_to_part(image_path))
            except Exception:
                continue
        contents = [{"role": "user", "parts": parts}]
    else:
        contents = [{"role": "user", "parts": [{"text": prompt}]}]

    try:
        if vertexai:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config={
                    "temperature": temperature,
                    "response_mime_type": "application/json",
                },
            )
        else:
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config={
                    "temperature": temperature,
                    "response_mime_type": "application/json",
                },
            )

        text = getattr(resp, "text", None)
        if text:
            return text

        candidates = getattr(resp, "candidates", None)
        if candidates:
            for cand in candidates:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if parts:
                    for part in parts:
                        t = getattr(part, "text", None)
                        if t:
                            return t

        raise ValueError("Unable to extract text from Gemini response")
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}")


def _call_gpt(
    prompt: str,
    image_paths: OptListStr = None,
    model: str = GPT_MODEL,
    temperature: float = 0.2,
) -> str:
    """Call OpenAI GPT API for JSON text generation."""
    if not OPENAI_OK:
        raise ImportError("openai is not installed. Install with: pip install openai")

    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()

    if azure_endpoint and azure_deployment:
        api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable not set")
        if AzureOpenAI is None:
            raise ImportError("openai Azure client is unavailable. Upgrade openai package.")
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
        )
        resolved_model = azure_deployment
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        client = OpenAI(api_key=api_key)
        resolved_model = model

    image_paths = [p for p in (image_paths or []) if isinstance(p, str) and os.path.exists(p)]
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        try:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_to_data_url(image_path)},
                }
            )
        except Exception:
            continue

    try:
        resp = client.chat.completions.create(
            model=resolved_model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Return a single valid JSON object and no extra prose.",
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
        )
        text = resp.choices[0].message.content if resp.choices else None
        if isinstance(text, str) and text.strip():
            return text
        raise ValueError("Unable to extract text from GPT response")
    except Exception as e:
        raise RuntimeError(f"OpenAI GPT API call failed: {e}")


def _is_gpt_model(model: str) -> bool:
    model_l = (model or "").lower()
    prefixes = ("gpt", "o1", "o3", "o4", "gpt-5.4-nano")
    return model_l.startswith(prefixes)


def _guess_image_role(image_path: str) -> str:
    name = Path(image_path).stem.lower()
    if "before_after" in name:
        return "before_after_panel"
    if "board" in name:
        return "situation_board"
    if "chart" in name or "bar" in name or "pie" in name or "line" in name:
        return "chart"
    if "report" in name:
        return "report_page"
    if "damage" in name or "overlay" in name or "map" in name:
        return "damage_or_hazard_map"
    if "pre" in name and "post" not in name:
        return "pre_disaster_image"
    if "post" in name:
        return "post_disaster_image"
    return "context_image"


def _compact_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key, value in (obs or {}).items():
        if key.endswith("_path") and isinstance(value, str):
            compact[key] = Path(value).name
            continue

        if isinstance(value, (int, float, bool)):
            compact[key] = value
            continue

        if isinstance(value, str):
            compact[key] = value[:200] if len(value) > 200 else value
            continue

        if isinstance(value, list):
            compact[f"{key}_count"] = len(value)
            continue

        if isinstance(value, dict):
            sub: Dict[str, Any] = {}
            for sub_key, sub_val in list(value.items())[:12]:
                if isinstance(sub_val, (int, float, bool)):
                    sub[sub_key] = sub_val
                elif isinstance(sub_val, str) and len(sub_val) <= 80:
                    sub[sub_key] = sub_val
                elif isinstance(sub_val, dict):
                    inner = {
                        inner_key: inner_val
                        for inner_key, inner_val in list(sub_val.items())[:6]
                        if isinstance(inner_val, (int, float, bool)) or (isinstance(inner_val, str) and len(inner_val) <= 60)
                    }
                    if inner:
                        sub[sub_key] = inner
            if sub:
                compact[key] = sub
    return compact


def _compact_trajectory(trajectory: ListDict, max_steps: int = 40) -> List[Dict[str, Any]]:
    compact_steps: List[Dict[str, Any]] = []
    for idx, step in enumerate((trajectory or [])[:max_steps]):
        if not isinstance(step, dict):
            continue
        item: Dict[str, Any] = {"step_index": idx, "call": step.get("call") or step.get("tool")}
        obs = step.get("obs")
        if isinstance(obs, dict) and obs:
            item["obs"] = _compact_obs(obs)
        if "error" in step:
            item["error"] = str(step["error"])
        compact_steps.append(item)
    return compact_steps


def _extract_prominent_statistics(compact_trajectory: ListDict, max_items: int = 20) -> Dict[str, Any]:
    hints = ("count", "area", "length", "distance", "ratio", "total", "mean", "damaged", "damage", "destroyed", "flood", "road", "building", "confidence")
    skip = ("row", "col", "bbox", "centroid", "shape")
    stats: Dict[str, Any] = {}

    for step in compact_trajectory:
        call = step.get("call") or "step"
        obs = step.get("obs", {})
        if not isinstance(obs, dict):
            continue
        for key, value in obs.items():
            full_key = f"{call}.{key}"
            key_str = str(key)
            key_l = key_str.lower()
            if any(s in key_l for s in skip):
                continue
            if isinstance(value, (int, float, bool)) and (any(h in key_l for h in hints) or len(stats) < max_items // 2):
                stats[full_key] = value
            elif isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    sub_full = f"{full_key}.{sub_key}"
                    sub_key_l = f"{key_l}.{str(sub_key).lower()}"
                    if any(s in sub_key_l for s in skip):
                        continue
                    if isinstance(sub_val, (int, float, bool)) and any(h in sub_key_l for h in hints):
                        stats[sub_full] = sub_val
                    elif isinstance(sub_val, str) and any(h in sub_key_l for h in ("trend", "region", "status")):
                        stats[sub_full] = sub_val
            elif isinstance(value, str) and any(h in key_l for h in ("trend", "region", "status")):
                stats[full_key] = value
            if len(stats) >= max_items:
                return stats
    return stats


def _infer_task_mode(task_question: str) -> Dict[str, bool]:
    q = (task_question or "").lower()
    return {
        "needs_recommendations": any(k in q for k in ("recommend", "recovery", "restore", "restoration", "repair", "rebuild", "reconstruction")),
        "report_style": any(k in q for k in ("report", "briefing", "summary", "situation", "overview", "board")),
        "image_focused": any(k in q for k in ("image", "scene", "caption", "visual", "remote sensing")),
    }


def _build_summary_prompt(
    task_question: str,
    compact_trajectory: ListDict,
    prominent_statistics: Dict[str, Any],
    final_answer_template: OptDictStrAny = None,
    image_paths: OptListStr = None,
) -> str:
    mode = _infer_task_mode(task_question)
    has_images = bool(image_paths)
    image_descriptions = [
        {"index": idx + 1, "role": _guess_image_role(path), "filename": Path(path).name}
        for idx, path in enumerate(image_paths or [])
    ]

    instructions = ["You are a disaster response and remote sensing analyst."]
    if has_images:
        instructions.append(
            "Your TASK is to analyze the provided disaster remote sensing images. "
            "You will act as a remote sensing analyst to identify the type of disaster and assess its impact on both built and natural environments."
        )
        instructions.append(
            "The image input count is variable. Use all provided images in order, including raw pre/post scenes and any derived boards, maps, or charts."
        )
        instructions.append(
            "Combine visual observations from the images with quantitative evidence from the trajectory. If the visual impression and the trajectory disagree, say so explicitly."
        )
    else:
        instructions.append("No image input is provided for this task. Base the answer on trajectory evidence and quantitative statistics only.")

    if mode["needs_recommendations"]:
        instructions.append(
            "Your TASK is to generate concise and integrated recovery recommendations for the affected area based on the provided disaster remote sensing images and trajectory statistics. "
            "Aspects to focus on include infrastructure restoration, housing reconstruction, and ecological and geological environment restoration."
        )
    elif mode["report_style"]:
        instructions.append("Produce a concise report-style summary suitable for a one-page disaster briefing.")
    else:
        instructions.append("Answer the task question directly and concisely.")

    instructions.append(
        "Prioritize concrete, decision-useful statements. Mention the most important quantitative findings and visible impact patterns. "
        "Keep the summary compact enough to fit inside a report page."
    )

    schema = {
        "summary": "3-5 sentence integrated summary",
        "image_observations": ["short observation"],
        "key_findings": ["finding1", "finding2", "finding3"],
        "statistics": {"metric_name": 0},
        "recommendations": ["action1", "action2"],
        "confidence": "high/medium/low",
    }

    return "\n\n".join([
        "\n".join(instructions),
        f"## Task Question\n{task_question}",
        f"## Image Inputs\n{json.dumps(image_descriptions, ensure_ascii=False, indent=2)}",
        f"## Compact Trajectory\n{json.dumps(compact_trajectory, ensure_ascii=False, indent=2)}",
        f"## Prominent Statistics\n{json.dumps(prominent_statistics, ensure_ascii=False, indent=2)}",
        f"## Final Answer Template\n{json.dumps(final_answer_template or {}, ensure_ascii=False, indent=2)}",
        "Return a single JSON object with this schema and no extra text:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}",
    ])


def _build_fallback_summary(
    task_question: str,
    prominent_statistics: Dict[str, Any],
    image_paths: OptListStr = None,
) -> Dict[str, Any]:
    mode = _infer_task_mode(task_question)
    stats_items = list(prominent_statistics.items())[:4]
    stats_text = ", ".join(f"{k}={v}" for k, v in stats_items) if stats_items else "no prominent quantitative statistics were extracted"
    prefix = "Image-aware LLM summary unavailable; using trajectory statistics only." if image_paths else "LLM summary unavailable; using trajectory statistics only."
    summary = f"{prefix} Key outputs include {stats_text}."
    recommendations: List[str] = []
    if mode["needs_recommendations"]:
        recommendations = [
            "Prioritize the highest-damage infrastructure and access corridors first.",
            "Stage housing reconstruction after basic access and safety conditions are restored.",
            "Address ecological or geotechnical stabilization where hazard signals remain active.",
        ]
        summary += " Recovery should prioritize critical infrastructure, housing, and environmental stabilization in that order where feasible."
    return {
        "summary": summary,
        "image_observations": [f"{len(image_paths or [])} image inputs were supplied."] if image_paths else [],
        "key_findings": [summary],
        "statistics": prominent_statistics,
        "recommendations": recommendations,
        "confidence": "low",
    }


def _safe_load_mask(mask_path: str) -> np.ndarray:
    """Load a raster/mask file into a numpy array."""
    arr = np.array(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _connected_components_count(binary: np.ndarray) -> int:
    """Simple 4-neighborhood connected component count without extra deps."""
    binary = np.asarray(binary).astype(bool)
    if binary.ndim != 2:
        raise ValueError("binary mask must be 2D")

    h, w = binary.shape
    visited = np.zeros((h, w), dtype=bool)
    count = 0

    for i in range(h):
        for j in range(w):
            if not binary[i, j] or visited[i, j]:
                continue
            count += 1
            stack = [(i, j)]
            visited[i, j] = True

            while stack:
                x, y = stack.pop()
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < h and 0 <= ny < w and binary[nx, ny] and not visited[nx, ny]:
                        visited[nx, ny] = True
                        stack.append((nx, ny))

    return count


def _quadrant_from_center(row: float, col: float, h: int, w: int) -> str:
    vertical = "north" if row < h / 2 else "south"
    horizontal = "west" if col < w / 2 else "east"
    return f"{vertical}{horizontal}"


def _normalize_classes(classes: Optional[Dict[Any, Any]]) -> Dict[str, str]:
    if not isinstance(classes, dict):
        return {}
    out = {}
    for k, v in classes.items():
        out[str(k)] = str(v)
    return out


def _normalize_target_classes(target_classes: Optional[List[Any]]) -> Optional[List[int]]:
    if not isinstance(target_classes, (list, tuple, set)):
        return None
    values: List[int] = []
    for value in target_classes:
        try:
            int_value = int(value)
        except Exception:
            continue
        if int_value == 0:
            continue
        values.append(int_value)
    if not values:
        return None
    return sorted(set(values))


def _extract_timestamp_from_item(item: Any) -> Optional[str]:
    if not isinstance(item, dict):
        return None

    timestamp = item.get("timestamp")
    if isinstance(timestamp, str) and len(timestamp) >= 2 and timestamp[0] == "t" and timestamp[1:].isdigit():
        return timestamp

    for key in ("post_image_path", "image_path", "mask_path"):
        value = item.get(key)
        if not isinstance(value, str):
            continue
        match = re.search(r"_(t\d+)(?:_|$|\.)", Path(value).stem)
        if match:
            return match.group(1)
    return None


def _describe_mask(
    mask_path: str,
    classes: OptDictAnyStr = None,
    pixel_size_m: Optional[float] = None,
    target_classes: OptListInt = None,
) -> Dict[str, Any]:
    arr = _safe_load_mask(mask_path)
    classes = _normalize_classes(classes)
    target_values = _normalize_target_classes(target_classes)

    if target_values is not None:
        selected = np.isin(arr, target_values)
        unique, counts = np.unique(arr[selected], return_counts=True)
        nonzero_pairs = [(int(v), int(c)) for v, c in zip(unique.tolist(), counts.tolist()) if int(v) in target_values]
        mask_nonzero = selected
    else:
        unique, counts = np.unique(arr, return_counts=True)
        nonzero_pairs = [(int(v), int(c)) for v, c in zip(unique.tolist(), counts.tolist()) if int(v) != 0]
        mask_nonzero = arr != 0
    total_nonzero = int(sum(c for _, c in nonzero_pairs))

    by_class: Dict[str, Dict[str, Any]] = {}
    for value, count in nonzero_pairs:
        item: Dict[str, Any] = {
            "pixel_count": count,
            "label": classes.get(str(value), str(value))
        }
        if pixel_size_m is not None:
            item["area_m2"] = float(count * (float(pixel_size_m) ** 2))
        by_class[str(value)] = item

    bbox = None
    centroid = None
    dominant_region = None
    connected_components = 0

    if np.any(mask_nonzero):
        rows, cols = np.where(mask_nonzero)

        bbox = {
            "row_min": int(rows.min()),
            "row_max": int(rows.max()),
            "col_min": int(cols.min()),
            "col_max": int(cols.max()),
        }

        centroid_row = float(rows.mean())
        centroid_col = float(cols.mean())
        centroid = {
            "row": centroid_row,
            "col": centroid_col
        }

        dominant_region = _quadrant_from_center(
            centroid_row, centroid_col, arr.shape[0], arr.shape[1]
        )
        connected_components = _connected_components_count(mask_nonzero)

    return {
        "mask_path": mask_path,
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "nonzero_pixel_count": total_nonzero,
        "nonzero_area_m2": float(total_nonzero * (float(pixel_size_m) ** 2)) if pixel_size_m is not None else None,
        "connected_components": connected_components,
        "bbox": bbox,
        "centroid": centroid,
        "dominant_region": dominant_region,
        "classes": classes,
        "target_classes": target_values,
        "class_statistics": by_class,
    }


def _collect_paths(obj: Any, parent_key: str = "") -> List[Tuple[str, str]]:
    """
    Recursively collect possible mask paths from obs / nested dicts.
    """
    out: List[Tuple[str, str]] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{parent_key}.{k}" if parent_key else k

            if k in {"mask_path", "output_mask_path"} and isinstance(v, str):
                out.append((key, v))

            elif k in {"mask_paths"} and isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, str):
                        out.append((f"{key}.{kk}", vv))

            else:
                out.extend(_collect_paths(v, key))

    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            out.extend(_collect_paths(item, f"{parent_key}[{idx}]"))

    return out


def _extract_timestamp_from_key(key: str) -> Optional[str]:
    """
    Try to recover t1/t2/t3/... from source keys like:
    obs.mask_paths.t1
    obs.mask_paths.t2
    """
    parts = key.replace("[", ".").replace("]", "").split(".")
    for p in reversed(parts):
        if len(p) >= 2 and p[0] == "t" and p[1:].isdigit():
            return p
    return None


def report_extract_evidence(
    trajectory: ListDict,
    pixel_size_m: Optional[float] = None,
    include_empty_masks: bool = False,
    target_classes: OptListInt = None,
) -> Dict[str, Any]:
    """
    Extract structured evidence from intermediate mask outputs in a trajectory.

    Parameters:
    - trajectory: List of trajectory steps
    - pixel_size_m: Pixel ground resolution in meters
    - include_empty_masks: Whether to keep masks with zero nonzero pixels

    Returns:
    - JSON-like dict of extracted evidence
    """
    evidence_items: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    tool_rollup: Dict[str, Dict[str, Any]] = {}
    temporal_series: Dict[str, Dict[str, Any]] = {}

    for step_idx, step in enumerate(trajectory):
        if not isinstance(step, dict):
            continue

        step_timestamp = None
        obs = step.get("obs", {})
        call_name = step.get("call") or step.get("tool") or f"step_{step_idx}"

        if not isinstance(obs, dict) and isinstance(step.get("last_obs"), dict):
            obs = step.get("last_obs", {})
            call_name = obs.get("tool", call_name)
            step_timestamp = _extract_timestamp_from_item(step.get("item"))
        elif isinstance(step.get("last_obs"), dict) and not obs:
            obs = step.get("last_obs", {})
            call_name = obs.get("tool", call_name)
            step_timestamp = _extract_timestamp_from_item(step.get("item"))

        classes = obs.get("classes") if isinstance(obs, dict) else None

        for key, path in _collect_paths(obs):
            if not isinstance(path, str):
                continue

            suffix = Path(path).suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
                continue

            key_l = str(key).lower()
            if "mask" not in key_l and "mask" not in Path(path).stem.lower():
                continue

            if not os.path.exists(path):
                errors.append({
                    "step_index": str(step_idx),
                    "source_key": key,
                    "path": path,
                    "error": "file_not_found"
                })
                continue

            try:
                item = _describe_mask(
                    path,
                    classes=classes,
                    pixel_size_m=pixel_size_m,
                    target_classes=target_classes,
                )

                if (not include_empty_masks) and item["nonzero_pixel_count"] == 0:
                    continue

                item["step_index"] = step_idx
                item["source_tool"] = obs.get("tool", call_name) if isinstance(obs, dict) else call_name
                item["source_key"] = key
                evidence_items.append(item)

                roll = tool_rollup.setdefault(
                    item["source_tool"],
                    {
                        "mask_count": 0,
                        "total_nonzero_pixels": 0,
                        "total_nonzero_area_m2": 0.0 if pixel_size_m is not None else None,
                        "dominant_regions": [],
                    },
                )
                roll["mask_count"] += 1
                roll["total_nonzero_pixels"] += item["nonzero_pixel_count"]
                if pixel_size_m is not None:
                    roll["total_nonzero_area_m2"] += item["nonzero_area_m2"] or 0.0
                if item["dominant_region"] is not None:
                    roll["dominant_regions"].append(item["dominant_region"])

                ts = _extract_timestamp_from_key(key) or step_timestamp
                if ts is not None:
                    temporal_series[ts] = {
                        "nonzero_pixel_count": item["nonzero_pixel_count"],
                        "nonzero_area_m2": item["nonzero_area_m2"],
                        "source_tool": item["source_tool"],
                        "dominant_region": item["dominant_region"],
                        "connected_components": item["connected_components"],
                    }

            except Exception as e:
                errors.append({
                    "step_index": str(step_idx),
                    "source_key": key,
                    "path": path,
                    "error": str(e)
                })

    timeline = None
    peak_timestamp = None
    trend = None

    if temporal_series:
        def _sort_key(ts_name: str) -> int:
            try:
                return int(ts_name[1:])
            except Exception:
                return 10**9

        ordered_ts = sorted(temporal_series.keys(), key=_sort_key)
        timeline = [{"timestamp": ts, **temporal_series[ts]} for ts in ordered_ts]

        peak_timestamp = max(
            ordered_ts,
            key=lambda ts: temporal_series[ts]["nonzero_pixel_count"]
        )

        counts = [temporal_series[ts]["nonzero_pixel_count"] for ts in ordered_ts]
        if len(counts) >= 2:
            if all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1)):
                trend = "expanding_or_non_decreasing"
            elif all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)):
                trend = "shrinking_or_non_increasing"
            else:
                peak_idx = ordered_ts.index(peak_timestamp)
                if 0 < peak_idx < len(ordered_ts) - 1:
                    trend = "expand_then_recede"
                else:
                    trend = "mixed"
        else:
            trend = "single_timestamp"

    return {
        "tool": "report.extract_evidence",
        "total_evidence_items": len(evidence_items),
        "evidence_items": evidence_items,
        "tool_rollup": tool_rollup,
        "timeline": timeline,
        "peak_timestamp": peak_timestamp,
        "trend": trend,
        "errors": errors,
    }


def model_summarize(
    task_question: str,
    trajectory: ListDict,
    image_paths: OptListStr = None,
    final_answer_template: OptDictStrAny = None,
    model: str = GPT_MODEL,
) -> Dict[str, Any]:
    """
    Use LLM to generate a summary of disaster analysis results.
    """
    compact_trajectory = _compact_trajectory(trajectory)
    prominent_statistics = _extract_prominent_statistics(compact_trajectory)
    valid_image_paths = [p for p in (image_paths or []) if isinstance(p, str) and os.path.exists(p)]

    prompt = _build_summary_prompt(
        task_question=task_question,
        compact_trajectory=compact_trajectory,
        prominent_statistics=prominent_statistics,
        final_answer_template=final_answer_template,
        image_paths=valid_image_paths,
    )

    try:
        if _is_gpt_model(model):
            response = _call_gpt(prompt, image_paths=valid_image_paths, model=model)
        else:
            if not GENAI_OK:
                raise ImportError("google-genai not installed")
            response = _call_gemini(prompt, image_paths=valid_image_paths, model=model)
        result = json.loads(response)
        if not isinstance(result, dict):
            raise ValueError("model_summarize expected a JSON object response")
        result.setdefault("summary", _build_fallback_summary(task_question, prominent_statistics, valid_image_paths)["summary"])
        result.setdefault("image_observations", [])
        result.setdefault("key_findings", [])
        result.setdefault("statistics", prominent_statistics)
        result.setdefault("recommendations", [])
        result.setdefault("confidence", "medium")
        result["tool"] = "model.summarize"
        result["model"] = model
        result["image_input_count"] = len(valid_image_paths)
        return result
    except json.JSONDecodeError:
        fallback = _build_fallback_summary(task_question, prominent_statistics, valid_image_paths)
        fallback.update({
            "tool": "model.summarize",
            "status": "parsed",
            "raw_response": response,
            "model": model,
            "image_input_count": len(valid_image_paths),
        })
        return fallback
    except Exception as e:
        fallback = _build_fallback_summary(task_question, prominent_statistics, valid_image_paths)
        fallback.update({
            "tool": "model.summarize",
            "status": "stub" if (not _is_gpt_model(model) and not GENAI_OK) or (_is_gpt_model(model) and not OPENAI_OK) else "error",
            "error": str(e),
            "model": model,
            "image_input_count": len(valid_image_paths),
        })
        return fallback


@mcp.tool(name="report.extract_evidence", description='''
Extract structured evidence from intermediate mask outputs in a trajectory.

Parameters:
- trajectory (list[dict]): List of trajectory steps with obs results
- pixel_size_m (float, optional): Pixel size in meters for area computation
- include_empty_masks (bool, optional): Whether to keep masks with zero nonzero pixels

Returns:
- total_evidence_items
- evidence_items
- tool_rollup
- timeline
- peak_timestamp
- trend
- errors
''')
def report_extract_evidence_wrapper(
    trajectory: ListDict,
    pixel_size_m: Optional[float] = None,
    include_empty_masks: bool = False,
    target_classes: OptListInt = None,
) -> str:
    result = report_extract_evidence(
        trajectory=trajectory,
        pixel_size_m=pixel_size_m,
        include_empty_masks=include_empty_masks,
        target_classes=target_classes,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool(name="model.summarize", description='''
Use LLM to generate a summary of disaster analysis results.

Parameters:
- task_question (str): The original task question
- trajectory (list[dict]): List of trajectory steps with obs results from previous tool calls
- image_paths (list[str], optional): Variable-length list of raw or derived images to summarize jointly with the trajectory
- final_answer_template (dict, optional): Template for the final answer
- model (str, optional): Gemini model to use (default: gemini-3-pro-preview)

Returns:
- summary (str): Natural language summary
- image_observations (list): Short visual observations
- key_findings (list): List of key findings
- statistics (dict): Key statistics
- recommendations (list): Optional recovery or response actions
- confidence (str): Confidence level
''')
def model_summarize_wrapper(
    task_question: str,
    trajectory: ListDict,
    image_paths: OptListStr = None,
    final_answer_template: OptDictStrAny = None,
    model: str = GEMINI_MODEL,
) -> str:
    result = model_summarize(task_question, trajectory, image_paths, final_answer_template, model)
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run(show_banner=False)
