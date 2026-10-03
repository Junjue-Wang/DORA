"""Agent prompts. The text is part of the benchmark protocol: keep it byte-identical across releases.

AP (autonomous planning) and IF (instruction following) share one template; IF only adds
the gold tool-name sequence. The whole prompt is sent as a single user message.
"""
from typing import Any, Dict, List

_GUIDELINES_AND_FORMAT = '''
GUIDELINES:

1. **Think in three stages**: Perceive → Analyze → Answer.
   - **Perceive**: Run the appropriate perception tools to extract information from imagery.
   - **Analyze**: Compose analysis tools to answer the question (spatial operations, arithmetic, routing, aggregation, etc.).
   - **Answer**: Extract the numeric/structured result and format it as JSON.

2. **Tool output chaining**: When a tool returns a file path (e.g., `mask_path`, `vector_path`, `buffer_path`), use that EXACT path as input to the next tool. Never fabricate or guess file paths.

3. **Error recovery**: If a tool returns an error, check file paths and parameters. Retry ONCE with corrected inputs. If it still fails, proceed with available intermediate results.

ANSWER FORMAT:
Your final answer MUST be a JSON object with the exact field names from ANSWER_FIELDS.
Format: <Answer>{"field_name_1": value1, "field_name_2": value2}</Answer>

Rules:
- Use the EXACT field names listed in ANSWER_FIELDS
- Numeric values should be numbers (integers for counts, floats for measurements)
- String values should be quoted
- If a field expects a collection, use a JSON array or object as appropriate
- File path values should be the actual output file path as a string
- If you cannot compute a field, use null
- ALWAYS provide the <Answer> tag even if some fields are null
'''

_HEADER = '''You are a disaster-response geospatial analyst. You orchestrate MCP tools to answer quantitative questions about disaster imagery.

TASK: Answer the question about disaster imagery analysis by composing tools into a pipeline.
'''

# eval_spec field type -> data type shown to the agent (metric names are never shown)
_TYPE_DISPLAY = {
    "scalar_count": "integer",
    "scalar_continuous": "float",
    "scalar_numeric": "float",
    "exact_match": "string",
    "scalar_dict": "dict {string: number}",
    "set_f1": "list of strings",
    "ranking_kendall": "list of strings (ranked)",
    "polygon_iou": "file path (GeoJSON)",
    "point_pixel": "integer (pixel coordinate)",
    "point_haversine": "float (decimal degrees)",
    "text_bleu": "string",
}


def system_prompt(trajectory_hint: str = "") -> str:
    """AP prompt; with ``trajectory_hint`` it becomes the IF prompt."""
    sequence_block = f"\n{trajectory_hint}\n" if trajectory_hint else ""
    return _HEADER + sequence_block + _GUIDELINES_AND_FORMAT


def to_model_tool_name(name: str) -> str:
    """MCP ids are dotted (``seg.flood``); OpenAI-style APIs need ``seg_flood``."""
    return name.replace(".", "_") if isinstance(name, str) else name


def format_trajectory_hint(trajectory: List[Dict[str, Any]]) -> str:
    """IF mode: the gold tool names in order, without arguments."""
    names = []
    for step in trajectory or []:
        call = step.get("call", "")
        tool_name = call.split("(", 1)[0].strip() if isinstance(call, str) and call else ""
        if tool_name:
            names.append(to_model_tool_name(tool_name))
    if not names:
        return ""
    lines = ["RECOMMENDED TOOL SEQUENCE (a plan to guide you; you may expand, split, or adapt steps as needed — bind arguments yourself):"]
    lines += [f"  Step {i}: {name}" for i, name in enumerate(names, 1)]
    return "\n".join(lines)


def answer_fields_hint(eval_spec: Dict[str, str]) -> str:
    fields = []
    for field, field_type in (eval_spec or {}).items():
        if field_type == "ignore":
            continue
        if field_type == "ratio_pct":
            # Disambiguate the scale by the field-name suffix (e.g. 0.55 vs 55.0).
            lower = field.lower()
            if lower.endswith(("_pct", "_percent", "_percentage")):
                dtype = "float in [0, 100] (e.g., 55.0 means 55%)"
            elif lower.endswith(("_ratio", "_share", "_fraction")):
                dtype = "float in [0, 1] (e.g., 0.55 means 55%)"
            else:
                dtype = "float (match the field-name suffix: _ratio=[0,1], _pct=[0,100])"
        else:
            dtype = _TYPE_DISPLAY.get(field_type, field_type)
        fields.append(f'  - "{field}": {dtype}')
    if not fields:
        return ""
    return "ANSWER_FIELDS (your <Answer> must be a JSON object with exactly these keys):\n" + "\n".join(fields)


def build_prompt(question: Dict[str, Any], autoplanning: bool = True) -> str:
    """Full user message for one question."""
    parts = [question["question"], "", question["context"]]
    hint = answer_fields_hint(question.get("eval_spec", {}))
    if hint:
        parts += ["", hint]
    trajectory_hint = ""
    if not autoplanning:
        trajectory_hint = format_trajectory_hint(question.get("expected_trajectory") or question.get("tools", []))
    return f"{system_prompt(trajectory_hint)}\n\nQuestion: " + "\n".join(parts)
