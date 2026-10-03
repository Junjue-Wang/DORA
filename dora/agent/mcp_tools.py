"""Spawn the six DORA MCP tool servers and expose their tools to LangChain."""
import functools
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient

from ..paths import DATA_ROOT
from .prompts import to_model_tool_name

# MCP server name -> module. The order fixes the order of tools shown to the model.
SERVERS = {
    "Analysis": "dora.tools.analysis",
    "Calculate": "dora.tools.calculate",
    "Perception": "dora.tools.perception",
    "Vis": "dora.tools.vis",
    "Model": "dora.tools.model",
    "POI": "dora.tools.poi",
}


def server_specs(temp_dir: Path, gpu: Optional[str] = None, gpu_slot: int = 0) -> Dict[str, Dict[str, Any]]:
    """stdio launch specs; ``gpu`` pins the Perception server to one CUDA device ("" = CPU)."""
    env = {k: v for k, v in os.environ.items()}
    env.update({
        "DORA_DATA": str(DATA_ROOT),
        "PYTHONIOENCODING": "utf-8",
        # Servers are spawned per tool call: no banner, no PyPI update check.
        "FASTMCP_SHOW_SERVER_BANNER": "false",
        "FASTMCP_CHECK_FOR_UPDATES": "off",
        "FASTMCP_LOG_LEVEL": "WARNING",
    })
    specs = {}
    for name, module in SERVERS.items():
        server_env = dict(env)
        if name == "Perception" and gpu is not None:
            server_env.update({"CUDA_VISIBLE_DEVICES": gpu, "DORA_GPU_SLOT": str(gpu_slot)})
        specs[name] = {
            "command": sys.executable,
            "args": ["-m", module, "--temp_dir", str(temp_dir)],
            "transport": "stdio",
            "env": server_env,
        }
    return specs


def normalize_tool_arguments(value: Any) -> Any:
    """Recursively turn JSON-stringified list/dict arguments back into native values."""
    if isinstance(value, str):
        stripped = value.strip()
        if (stripped.startswith("[") and stripped.endswith("]")) or (stripped.startswith("{") and stripped.endswith("}")):
            try:
                return normalize_tool_arguments(json.loads(stripped))
            except (TypeError, ValueError):
                return value
        return value
    if isinstance(value, list):
        return [normalize_tool_arguments(v) for v in value]
    if isinstance(value, tuple):
        return tuple(normalize_tool_arguments(v) for v in value)
    if isinstance(value, dict):
        return {k: normalize_tool_arguments(v) for k, v in value.items()}
    return value


def _with_normalized_args(tool):
    original_invoke, original_ainvoke = tool.invoke, tool.ainvoke

    @functools.wraps(original_invoke)
    def invoke(input, *args, **kwargs):
        return original_invoke(normalize_tool_arguments(input) if isinstance(input, dict) else input, *args, **kwargs)

    @functools.wraps(original_ainvoke)
    async def ainvoke(input, *args, **kwargs):
        return await original_ainvoke(normalize_tool_arguments(input) if isinstance(input, dict) else input,
                                      *args, **kwargs)

    object.__setattr__(tool, "invoke", invoke)
    object.__setattr__(tool, "ainvoke", ainvoke)
    return tool


def _alias_for_model(tools: List[Any]) -> List[Any]:
    """Rename dotted MCP tools (``seg.flood``) to model-safe names (``seg_flood``)."""
    aliased: Dict[str, Any] = {}
    for tool in tools:
        alias = to_model_tool_name(tool.name)
        if alias in aliased:
            raise ValueError(f"Tool name alias collision: {tool.name} -> {alias}")
        aliased[alias] = tool if tool.name == alias else tool.model_copy(update={"name": alias})
    return list(aliased.values())


async def create_tool_pool(specs: Dict[str, Dict[str, Any]]):
    """Start the servers, load and wrap their tools. Returns ``(tools, client)``."""
    client = MultiServerMCPClient(specs)
    try:
        tools = []
        for name in specs:
            try:
                tools.extend(await client.get_tools(server_name=name))
            except Exception as e:
                raise RuntimeError(f"Failed to load MCP tools from server '{name}': {e}") from e
        return [_with_normalized_args(t) for t in _alias_for_model(tools)], client
    except Exception:
        await close_client(client)
        raise


async def close_client(client) -> None:
    close = getattr(client, "close", None)
    if close is not None:
        try:
            await close()
        except Exception:
            pass


def required_tool_names(question: Dict[str, Any]) -> set:
    """Dotted tool names of the gold trajectory/plan, including calls nested in tool.loop / tool.reduce."""
    names = set()

    def walk(step):
        if not isinstance(step, dict):
            return
        call = step.get("call") or step.get("name") or ""
        if isinstance(call, str) and call.split("(", 1)[0].strip():
            names.add(call.split("(", 1)[0].strip())
        params = step.get("args") if isinstance(step.get("args"), dict) else step.get("inputs")
        if isinstance(params, dict):
            for nested in params.get("calls") or []:
                walk(nested)

    for step in (question.get("expected_trajectory") or []) + (question.get("tools") or []):
        walk(step)
    return names


def filter_tools_for_question(tools: List[Any], question: Dict[str, Any]) -> List[Any]:
    """IF mode: expose only the gold tools (falls back to the full pool if nothing matches)."""
    wanted = {to_model_tool_name(n) for n in required_tool_names(question)}
    subset = [t for t in tools if t.name in wanted]
    return subset or tools
