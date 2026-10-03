"""
Analysis MCP server: raster (ras.*), vector (vec.*, mask.*) and logical (tool.*) operations.
"""
import functools
import inspect
import json
import os.path
from pathlib import Path
import re
import warnings
from typing import Dict, Any, List, Optional, Tuple, Union, Annotated

from ._common import session_temp_dir, short_name as _short_name
from ._types import ListInt, ListStr, ListFloat, ListDict, ListDictStrInt, DictStrStr

from pydantic import BeforeValidator
try:
    from pydantic import PydanticDeprecatedSince20
except Exception:
    PydanticDeprecatedSince20 = None

from skimage.io import imread
import numpy as np
from PIL import Image
from fastmcp import FastMCP
import networkx as nx

# Geo/vector deps
try:
    import geopandas as gpd
    from shapely.geometry import shape, LineString
    GEO_DEPS_OK = True
except Exception:
    gpd = None
    shape = None
    LineString = None
    GEO_DEPS_OK = False

try:
    from rasterio import features
    from affine import Affine
    RASTERIO_OK = True
except Exception:
    features = None
    Affine = None
    RASTERIO_OK = False

# skeleton utils
try:
    from skimage.morphology import skeletonize, remove_small_objects, binary_closing
    SKIMAGE_OK = True
except Exception:
    SKIMAGE_OK = False

from networkx.readwrite import write_graphml, read_graphml

if PydanticDeprecatedSince20 is not None:
    warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)


mcp = FastMCP()
TEMP_DIR = session_temp_dir()

RESCUENET_CLASSES = {
    "0": "background",
    "1": "building_no_damage",
    "2": "building_damage",
    "3": "building_total_damage",
    "4": "pool",
    "5": "road_no_damage",
    "6": "road_sand_covered",
    "7": "tree_fallen",
    "8": "tree_not_fallen",
    "9": "vehicle_not_trapped",
    "10": "vehicle_trapped",
    "11": "water",
    "255": "ignore",
}


import hashlib
import logging

logger = logging.getLogger(__name__)


def _coerce_loop_item(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return {
            "x1": int(value[0]),
            "x2": int(value[1]),
            "y1": int(value[2]),
            "y2": int(value[3]),
        }
    return value


LoopItem = Annotated[Dict[str, Any], BeforeValidator(_coerce_loop_item)]


def _coerce_json_list(v):
    if isinstance(v, str):
        import json as _j
        try:
            return _j.loads(v.strip())
        except Exception:
            return v
    return v


ListLoopItem = Annotated[List[LoopItem], BeforeValidator(_coerce_json_list)]


def _connected_components(binary: np.ndarray) -> Tuple[np.ndarray, int]:
    """4-connected component labelling."""
    h, w = binary.shape
    labels = np.zeros_like(binary, dtype=np.int32)
    label = 0
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for i in range(h):
        for j in range(w):
            if binary[i, j] == 0 or labels[i, j] != 0:
                continue
            label += 1
            stack = [(i, j)]
            labels[i, j] = label
            while stack:
                x, y = stack.pop()
                for dx, dy in directions:
                    nx_, ny = x + dx, y + dy
                    if 0 <= nx_ < h and 0 <= ny < w and binary[nx_, ny] == 1 and labels[nx_, ny] == 0:
                        labels[nx_, ny] = label
                        stack.append((nx_, ny))
    return labels, label


def _stub_result(tool: str, **inputs: Any) -> Dict[str, Any]:
    """Structured error result returned when a tool fails."""
    result = {
        "tool": tool,
        "status": "stub",
        "note": "Replace this stub with a real model backend.",
        "inputs": inputs,
    }
    if "error" in inputs:
        result["error"] = inputs["error"]
    return result


def _tool_guard(tool_name: str):
    def decorator(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                result = fn(*args, **kwargs)
            except FileNotFoundError as e:
                # Missing input file → return graceful empty result instead of stub
                try:
                    bound = sig.bind_partial(*args, **kwargs)
                    inputs = dict(bound.arguments)
                except Exception:
                    inputs = {}
                empty = {"tool": tool_name, "note": "No input data available; returning empty result."}
                # Add tool-specific empty fields
                if "clip" in tool_name:
                    empty.update({"clip_path": None, "input_count": 0, "output_count": 0})
                elif "intersect" in tool_name:
                    empty.update({"intersection_count": 0, "intersection_area_m2": 0.0, "intersection_path": None})
                elif "nearest" in tool_name:
                    empty.update({"pair_count": 0, "table_path": None, "links_geojson": None})
                elif "filter" in tool_name:
                    empty.update({"original_count": 0, "filtered_count": 0, "removed_count": 0})
                elif "connected" in tool_name:
                    empty.update({"count": 0, "vector_path": None})
                else:
                    empty.update({"count": 0})
                empty.update(inputs)
                return json.dumps(empty, ensure_ascii=False)
            except Exception as e:
                try:
                    bound = sig.bind_partial(*args, **kwargs)
                    inputs = dict(bound.arguments)
                except Exception:
                    inputs = {}
                return json.dumps(_stub_result(tool_name, error=str(e), **inputs), ensure_ascii=False)

            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False)

        return wrapper

    return decorator


def _require_geo_deps(tool: str):
    """Raise a clear error if the vector/raster stack is not installed."""
    if not GEO_DEPS_OK:
        raise RuntimeError(
            f"{tool}: geopandas is not installed (pip install geopandas)"
        )
    if not RASTERIO_OK:
        raise RuntimeError(
            f"{tool}: rasterio is not installed (pip install rasterio)"
        )


def _require_skimage(tool: str):
    if not SKIMAGE_OK:
        raise RuntimeError(
            f"{tool}: scikit-image is not installed (pip install scikit-image)"
        )


def _vectorize_mask_to_gdf(mask: np.ndarray, classes: ListInt, pixel_size_m: float = None, transform=None, crs=None):
    """
    Vectorize a mask into a GeoDataFrame with rasterio.features.shapes.

    Parameters
    ----------
    transform : affine.Affine, optional
        Geo-transform read from a GeoTIFF.  When *None* an identity
        transform is used (pixel-coordinate space).
    crs : rasterio.crs.CRS | pyproj.CRS | None, optional
        Coordinate reference system.  Passed through to the resulting
        GeoDataFrame.
    """
    _require_geo_deps("ras.vectorize")

    target = np.isin(mask, classes).astype(np.uint8)
    if target.sum() == 0:
        return gpd.GeoDataFrame(
            columns=["id", "class", "pixel_count", "area_m2", "geometry"], geometry="geometry", crs=crs
        )

    if transform is None:
        transform = Affine.translation(0, 0) * Affine.scale(1, 1)

    records = []
    for idx, (geom, value) in enumerate(features.shapes(mask.astype(np.int32), mask=target, transform=transform)):
        geom_obj = shape(geom)
        crs_area = geom_obj.area
        if transform is not None and transform != Affine.identity():
            # Area is already in CRS units (e.g., m² for UTM projections)
            area_m2 = round(crs_area, 2)
            pixel_count = int(round(crs_area / abs(transform.a * transform.e)))
        else:
            pixel_count = int(round(crs_area))
            area_m2 = round(pixel_count * (pixel_size_m**2), 2) if pixel_size_m else None
        records.append(
            {
                "id": idx + 1,
                "class": int(value),
                "pixel_count": pixel_count,
                "area_m2": float(area_m2) if area_m2 is not None else 0,
                "pixel_size_m": float(pixel_size_m) if pixel_size_m is not None else None,
                "geometry": geom_obj,
            }
        )

    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


def _save_gdf(gdf, filename: str, *key_parts: str) -> str:
    """Save a GeoDataFrame with a unique filename.

    If *key_parts* are supplied they are hashed into the filename;
    otherwise the gdf length + column signature is used.
    """
    p = Path(filename)
    stem, ext = p.stem, p.suffix or ".geojson"
    if key_parts:
        raw = "|".join(str(k) for k in key_parts)
    else:
        raw = f"{len(gdf)}|{'|'.join(gdf.columns.tolist())}"
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:6]
    out_path = TEMP_DIR / f"{stem}_{h}{ext}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If CRS is unknown, write GeoJSON without a CRS field.
    if getattr(gdf, "crs", None) is None:
        data = json.loads(gdf.to_json())
        data.pop("crs", None)
        out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return str(out_path)

    # 1) prefer pyogrio
    try:
        gdf.to_file(out_path, driver="GeoJSON", engine="pyogrio")
        return str(out_path)
    except Exception:
        pass

    # 2) fallback: fiona (force the GeoJSON driver to read-write)
    try:
        import fiona
        fiona.supported_drivers["GeoJSON"] = "rw"
        gdf.to_file(out_path, driver="GeoJSON", engine="fiona")
        return str(out_path)
    except Exception:
        # 3) last resort: write the JSON text directly (no GDAL/Fiona)
        out_path.write_text(gdf.to_json(), encoding="utf-8")
        return str(out_path)

def _safe_raster_profile(src,
                         *,
                         driver: str = "GTiff",
                         dtype: Optional[str] = None,
                         count: Optional[int] = None,
                         nodata: Optional[float] = None,
                         compress: Optional[str] = None) -> Dict[str, Any]:
    """
    Build a write-safe raster profile from an input dataset.

    Key rule: never blindly reuse the input driver when the output suffix/format
    is different. This avoids cases like writing PNG bytes into a `.tif` path.
    """
    profile = src.profile.copy()
    profile["driver"] = driver

    if dtype is not None:
        profile["dtype"] = dtype
    if count is not None:
        profile["count"] = int(count)

    if nodata is not None:
        profile["nodata"] = nodata
    elif profile.get("nodata") is None:
        profile.pop("nodata", None)

    if compress is not None and driver == "GTiff":
        profile["compress"] = compress
    else:
        profile.pop("compress", None)

    if profile.get("crs") is None:
        profile.pop("crs", None)
    if profile.get("transform") is None:
        profile.pop("transform", None)

    return profile




def _read_vector_no_default_crs(vector_path: str):
    if vector_path is None or not os.path.exists(str(vector_path)):
        return gpd.GeoDataFrame(geometry=[])
    gdf = gpd.read_file(vector_path)
    if vector_path.lower().endswith(".geojson"):
        try:
            with open(vector_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "crs" not in data:
                gdf = gdf.set_crs(None, allow_override=True)
        except Exception:
            pass
    return gdf


def get_obs_value(obs, key):
    if isinstance(obs, list):
        if key == "results":
            return obs
        return None
    if isinstance(obs, dict):
        return obs.get(key)
    return None

@mcp.tool(name="tool.loop", description='''
Description:
Loop over a list of items and execute a sequence of tool calls for each item.
Supports per-item binding and simple step-result references.

Parameters:
- calls (list[dict]): [{tool: str, args: dict}, ...]
- loop_over (list[dict]): items to iterate over (dicts with x1/x2/y1/y2). List windows are accepted and coerced.
- bind (dict, optional): map arg_name -> item_key to inject into args
- grid_n (int, optional): if loop_over items are windows, provide n to compute row/col
- return_all (bool, optional): return per-iteration results

Returns:
- results (list, optional): per-iteration outputs if return_all true
''')
def tool_loop(calls: ListDict,
              loop_over: ListLoopItem,
              bind: DictStrStr = None,
              grid_n: int = None,
              return_all: bool = True,
              show_progress: bool = True,
              trajectory: ListDict = None) -> str:
    try:
        import json
        import os
        # Define regex patterns first
        traj_ref = re.compile(r"<([\w_]+)>\s*from\s*trajectory\[(\d+)\]\.obs", re.IGNORECASE)
        # If loop_over is a trajectory reference or file path, resolve it
        if isinstance(loop_over, str):
            # First check if it's a trajectory reference
            traj_match = traj_ref.fullmatch(loop_over.strip())
            if traj_match:
                key = traj_match.group(1)
                idx = int(traj_match.group(2))
                if trajectory is not None and 0 <= idx < len(trajectory):
                    obs = trajectory[idx].get("obs", {})
                    value = get_obs_value(obs, key)
                    # If value is a file path, read features from file
                    if isinstance(value, str) and value.endswith((".geojson", ".json")):
                        if os.path.exists(value):
                            with open(value, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, dict):
                                    loop_over = data.get("features", [])
                                elif isinstance(data, list):
                                    loop_over = data
                            loop_over = loop_over or []
                        else:
                            loop_over = []
                    else:
                        loop_over = value if isinstance(value, list) else []
            # If it's already a file path, read from file
            elif loop_over.endswith((".geojson", ".json")):
                if os.path.exists(loop_over):
                    with open(loop_over, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            loop_over = data.get("features", [])
                        elif isinstance(data, list):
                            loop_over = data
                    loop_over = loop_over or []
                else:
                    loop_over = []

        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None
        def call_tool_obj(func, **kwargs):
            if callable(func):
                return func(**kwargs)
            for attr in ("fn", "func", "run", "call", "__call__"):
                target = getattr(func, attr, None)
                if callable(target):
                    return target(**kwargs)
            raise TypeError(f"tool.loop: tool object not callable: {type(func)}")
        bind = bind or {}
        trajectory = trajectory or []
        step_ref = re.compile(r"<([\w_]+)>\s*from\s*step\[(\d+)\]\.obs", re.IGNORECASE)
        item_ref = re.compile(r"<item\.([\w_.]+)>", re.IGNORECASE)

        def resolve_nested_key(obj, keys):
            """Resolve nested keys like 'properties.pixel_row' from an object."""
            result = obj
            for k in keys:
                if isinstance(result, dict):
                    result = result.get(k)
                else:
                    return None
            return result

        def resolve_arg(val, steps, item, trajectory=None):
            if isinstance(val, list):
                return [resolve_arg(v, steps, item, trajectory) for v in val]
            if isinstance(val, dict):
                return {k: resolve_arg(v, steps, item, trajectory) for k, v in val.items()}
            if isinstance(val, str):
                m = step_ref.fullmatch(val.strip())
                if m:
                    key = m.group(1)
                    idx = int(m.group(2))
                    if 0 <= idx < len(steps):
                        return steps[idx].get("obs", {}).get(key, val)
                # Also support trajectory[N].obs references
                m = traj_ref.fullmatch(val.strip())
                if m:
                    key = m.group(1)
                    idx = int(m.group(2))
                    if trajectory is not None and 0 <= idx < len(trajectory):
                        obs = trajectory[idx].get("obs", {})
                        resolved = get_obs_value(obs, key)
                        return resolved if resolved is not None else val
                    return val
                m = item_ref.fullmatch(val.strip())
                if m:
                    keys = m.group(1).split(".")
                    if isinstance(item, dict):
                        result = resolve_nested_key(item, keys)
                        return result if result is not None else val
                    return val
            return val

        def coerce_item(it, idx):
            # Handle nested tool.loop results: {"item": Feature, "last_obs": ...}
            # Keep the inner item as-is for property access via <item.properties.X>
            if isinstance(it, dict) and "item" in it and "last_obs" in it:
                # This is already a tool.loop result format, keep the inner item
                it = it["item"]
            if isinstance(it, dict):
                if grid_n and ("row_idx" not in it or "col_idx" not in it):
                    it = dict(it)
                    it["row_idx"] = int(idx // grid_n)
                    it["col_idx"] = int(idx % grid_n)
                return it
            if isinstance(it, (list, tuple)) and len(it) == 4:
                row_idx = int(idx // grid_n) if grid_n else None
                col_idx = int(idx % grid_n) if grid_n else None
                return {
                    "x1": int(it[0]),
                    "x2": int(it[1]),
                    "y1": int(it[2]),
                    "y2": int(it[3]),
                    "row_idx": row_idx,
                    "col_idx": col_idx,
                }
            return {"value": it}

        def resolve_tool(tool_name: str):
            import sys
            internal_aliases = {
                "vec.build_graph": "build_road_graph",
                "vec.nearest": "vec_polygon_nearest",
            }
            # Strip the 'functions.' namespace prefix that OpenAI-style models
            # prepend when nesting tool calls inside tool.loop's `calls` arg
            # (e.g. "functions.seg_flood"). Without this strip the subsequent
            # module lookup ends up searching for "functions_seg_flood" and
            # always fails, forcing the model to abandon tool.loop.
            if tool_name.startswith("functions."):
                tool_name = tool_name[len("functions.") :]
            func_name = tool_name.replace(".", "_")
            # Get module globals for current module
            module_globals = sys._getframe(1).f_globals
            alias_name = internal_aliases.get(tool_name)
            if alias_name and alias_name in module_globals:
                return module_globals[alias_name]
            if func_name in module_globals:
                return module_globals[func_name]
            try:
                from dora.tools import calculate as Calc
                if hasattr(Calc, func_name):
                    return getattr(Calc, func_name)
            except Exception:
                pass
            try:
                from dora.tools import perception as Perc
                if hasattr(Perc, func_name):
                    return getattr(Perc, func_name)
            except Exception:
                pass
            # Also try without prefix (e.g., "vec.graph_shortest_path" -> "graph_shortest_path")
            if "." in tool_name:
                short_name = tool_name.split(".", 1)[1]
                short_func_name = short_name.replace(".", "_")
                if short_func_name in module_globals:
                    return module_globals[short_func_name]
                try:
                    from dora.tools import analysis as Analysis
                    if hasattr(Analysis, short_func_name):
                        return getattr(Analysis, short_func_name)
                except Exception:
                    pass
            return None

        results = []

        iterable = list(loop_over or [])
        if show_progress and tqdm is not None:
            iterable = tqdm(iterable, desc="tool.loop")

        for idx, raw_item in enumerate(iterable):
            item = coerce_item(raw_item, idx)
            steps = []
            for call in calls:
                tool_name = call.get("call", call.get("tool"))
                args = dict(call.get("args") or {})
                for arg_name, item_key in bind.items():
                    if arg_name not in args:
                        continue
                    # Check if item_key is a trajectory reference
                    traj_match = traj_ref.fullmatch(item_key.strip())
                    if traj_match:
                        key = traj_match.group(1)
                        idx = int(traj_match.group(2))
                        if trajectory is not None and 0 <= idx < len(trajectory):
                            obs = trajectory[idx].get("obs", {})
                            resolved = get_obs_value(obs, key)
                            args[arg_name] = resolved if resolved is not None else item_key
                    elif isinstance(item, dict) and item_key in item:
                        args[arg_name] = item[item_key]

                args = resolve_arg(args, steps, item, trajectory)
                func = resolve_tool(tool_name)
                if func is None:
                    raise KeyError(f"tool.loop: tool not found: {tool_name}")
                raw = call_tool_obj(func, **args)
                obs = json.loads(raw) if isinstance(raw, str) else raw
                steps.append({"tool": tool_name, "obs": obs})

            if return_all:
                results.append({
                    "item": item,
                    "last_obs": steps[-1]["obs"] if steps else None,
                })

        result = {
            "tool": "tool.loop",
            "iteration_count": int(len(loop_over or [])),
        }
        if return_all:
            # Return the bare results list so downstream tools (e.g. poi.group_by) can consume it
            return json.dumps(results, ensure_ascii=False)
        result["results"] = results
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("tool.loop", error=str(e),
                                       calls=calls, loop_over=loop_over, bind=bind, grid_n=grid_n))


@mcp.tool(name="tool.reduce", description='''
Description:
Reduce a list of dicts by a numeric field.

Parameters:
- values (list[dict]): items to reduce
- field (str): dotted path, e.g. "a.b.c"
- mode (str): "max", "min", or "mean"

Returns:
- best_value (float|None): for max/min
- best_index (int|None): for max/min
- best_item (dict|None): for max/min
- mean_value (float|None): for mean
''')
def tool_reduce(values: ListDict, field: str, mode: str = "max") -> str:
    try:
        def get_field(item, path):
            cur = item
            for key in path.split("."):
                if not isinstance(cur, dict) or key not in cur:
                    return None
                cur = cur[key]
            return cur

        nums = []
        for i, item in enumerate(values or []):
            val = get_field(item, field)
            if isinstance(val, (int, float)):
                nums.append((i, float(val)))

        if not nums:
            return json.dumps({
                "tool": "tool.reduce",
                "mode": mode,
                "best_value": None,
                "best_index": None,
                "best_item": None,
                "mean_value": None
            }, ensure_ascii=False)

        if mode == "mean":
            mean_val = sum(v for _, v in nums) / len(nums)
            return json.dumps({
                "tool": "tool.reduce",
                "mode": mode,
                "mean_value": float(mean_val)
            }, ensure_ascii=False)

        best_idx, best_val = max(nums, key=lambda x: x[1]) if mode != "min" else min(nums, key=lambda x: x[1])
        best_item = values[best_idx] if values and best_idx < len(values) else None
        result = {
            "tool": "tool.reduce",
            "mode": mode,
            "best_value": float(best_val),
            "best_index": int(best_idx),
            "best_item": best_item
        }
        if isinstance(best_item, dict):
            item = best_item.get("item", best_item)
            if isinstance(item, dict):
                if "region" in item:
                    result["region"] = item["region"]
                if "row_idx" in item:
                    result["row_idx"] = item["row_idx"]
                if "col_idx" in item:
                    result["col_idx"] = item["col_idx"]
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("tool.reduce", error=str(e),
                                       values=values, field=field, mode=mode))

# ====== Road graph helpers (from demo) ======

OFFS8 = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dy == 0 and dx == 0)]


def _to_skeleton(road_all: np.ndarray) -> np.ndarray:
    _require_skimage("vec.skeleton")
    road = road_all.astype(bool)
    road = remove_small_objects(road, min_size=32)
    road = binary_closing(road, np.ones((3, 3), dtype=np.uint8))
    skel = skeletonize(road)
    return skel.astype(bool)


def _neighbor_count(skel: np.ndarray) -> np.ndarray:
    H, W = skel.shape
    pad = np.pad(skel.astype(np.uint8), 1)
    cnt = np.zeros_like(pad, dtype=np.uint8)
    for dy, dx in OFFS8:
        cnt[1:1 + H, 1:1 + W] += pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
    return cnt[1:-1, 1:-1]


def _get_neighbors(p: tuple, skel: np.ndarray):
    y, x = p
    H, W = skel.shape
    ans = []
    for dy, dx in OFFS8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
            ans.append((ny, nx))
    return ans


def _step_len(a: tuple, b: tuple) -> float:
    dy = abs(a[0] - b[0])
    dx = abs(a[1] - b[1])
    return 1.41421356237 if (dy == 1 and dx == 1) else 1.0


def _count_damaged_pixels(damaged_mask: np.ndarray, pixels: List[tuple]) -> int:
    """Count damaged pixels along a polyline (pixels are (row,col))."""
    if not pixels:
        return 0
    arr = np.array(pixels, dtype=np.int32).T  # (2, N)
    return int(np.count_nonzero(damaged_mask[tuple(arr)]))


def _postprocess_intersection_damage(
    G: nx.Graph,
    damaged_mask: np.ndarray,
    damaged_ratio_edge: float,
    min_damaged_pixels: int,
    *,
    junction_deg: int = 3,
    junction_trim_px: int = 3,
    connector_max_len_px: float = 12.0,
) -> None:
    """Mitigate 'damage bleeding' at junctions."""
    if G.number_of_edges() == 0:
        return

    deg = dict(G.degree())

    for u, v, data in G.edges(data=True):
        pixels = data.get("pixels", [])
        if not pixels:
            continue
        ltrim = junction_trim_px if deg.get(u, 0) >= junction_deg else 0
        rtrim = junction_trim_px if deg.get(v, 0) >= junction_deg else 0
        if ltrim + rtrim >= len(pixels):
            trimmed = []
        else:
            trimmed = pixels[ltrim: len(pixels) - rtrim]
        if not trimmed:
            continue
        dmg_count = _count_damaged_pixels(damaged_mask, trimmed)
        dmg_ratio = dmg_count / max(len(trimmed), 1)
        data["damaged"] = bool((dmg_ratio >= damaged_ratio_edge) and (dmg_count >= min_damaged_pixels))

    junction_nodes = [n for n, d in deg.items() if d >= junction_deg]
    if not junction_nodes:
        return

    connector_edges = []
    for u, v, data in G.edges(data=True):
        if deg.get(u, 0) >= junction_deg and deg.get(v, 0) >= junction_deg:
            try:
                elen = float(data.get("length", 0.0))
            except Exception:
                elen = 0.0
            if elen <= connector_max_len_px:
                connector_edges.append((u, v))

    if not connector_edges:
        return

    C = nx.Graph()
    C.add_nodes_from(junction_nodes)
    C.add_edges_from(connector_edges)

    for comp in nx.connected_components(C):
        comp = set(comp)
        intact_arm_count = 0
        seen_arm = set()
        for u in comp:
            for v in G.neighbors(u):
                if v in comp:
                    continue
                key = (min(u, v), max(u, v))
                if key in seen_arm:
                    continue
                seen_arm.add(key)
                if not bool(G[u][v].get("damaged", False)):
                    intact_arm_count += 1

        if intact_arm_count >= 2:
            for u, v in G.subgraph(comp).edges():
                try:
                    elen = float(G[u][v].get("length", 0.0))
                except Exception:
                    elen = 0.0
                if elen <= connector_max_len_px:
                    G[u][v]["damaged"] = False


def _save_graph_graphml(G: nx.Graph, path) -> None:
    """
    Serialize graph attributes into GraphML-friendly types.
    Converts tuples/lists to JSON strings because GraphML writer
    does not support them directly.
    """
    H = nx.Graph()
    H.graph.update(G.graph)  # preserve metadata such as image size
    for nid, data in G.nodes(data=True):
        attr = dict(data)
        xy = attr.pop("xy", None)
        if xy is not None:
            attr["xy_json"] = json.dumps(list(xy))
        H.add_node(str(nid), **attr)

    for u, v, data in G.edges(data=True):
        attr = dict(data)
        pixels = attr.pop("pixels", None)
        if pixels is not None:
            attr["pixels_json"] = json.dumps([list(p) for p in pixels])
        H.add_edge(str(u), str(v), **attr)

    write_graphml(H, str(path))


def _load_graph_graphml(path) -> nx.Graph:
    """Inverse of _save_graph_graphml: restore tuples/lists from JSON strings."""
    H = read_graphml(str(path))
    G = nx.Graph()
    G.graph.update(H.graph)

    for nid, data in H.nodes(data=True):
        xy_json = data.pop("xy_json", None)
        xy = tuple(json.loads(xy_json)) if xy_json is not None else None
        G.add_node(int(nid), xy=xy)

    for u, v, data in H.edges(data=True):
        attr = dict(data)
        pix_json = attr.pop("pixels_json", None)
        pixels = [tuple(p) for p in json.loads(pix_json)] if pix_json is not None else []

        length_val = attr.get("length")
        try:
            length = float(length_val) if length_val is not None else float(len(pixels) - 1)
        except Exception:
            length = float(len(pixels) - 1)

        dam_val = attr.get("damaged", False)
        if isinstance(dam_val, str):
            damaged = dam_val.lower() == "true"
        else:
            damaged = bool(dam_val)

        G.add_edge(int(u), int(v), pixels=pixels, length=length, damaged=damaged)

    return G


def _build_road_graph(road_mask: np.ndarray, damaged_mask: np.ndarray,
                      include_turn_nodes: bool = True, turn_angle_min_deg: float = 5.0,
                      min_segment_pixels: int = 4,
                      damaged_ratio_edge: float = 0.5,
                      min_damaged_pixels: int = 1) -> nx.Graph:
    skel = _to_skeleton(road_mask)
    deg = _neighbor_count(skel)
    node_mask = skel & (deg != 2)
    node_coords = list(map(tuple, np.argwhere(node_mask)))
    node_id_map = {c: i for i, c in enumerate(node_coords)}
    next_id = len(node_coords)

    G = nx.Graph()
    for nid, xy in enumerate(node_coords):
        G.add_node(nid, xy=xy)

    seen_edges = set()

    def add_edge(u: int, v: int, pixels: List[tuple]):
        if u == v:
            return
        key = (min(u, v), max(u, v))
        if key in seen_edges:
            return
        seen_edges.add(key)
        length = float(sum(_step_len(pixels[i - 1], pixels[i]) for i in range(1, len(pixels))))
        dmg_count = int(np.count_nonzero(damaged_mask[tuple(np.array(pixels).T)]))
        dmg_ratio = dmg_count / max(len(pixels), 1)
        damaged = (dmg_ratio >= damaged_ratio_edge) and (dmg_count >= min_damaged_pixels)
        G.add_edge(u, v, pixels=pixels, length=length, damaged=bool(damaged))

    visited_dir = set()

    G.graph["img_h"] = int(road_mask.shape[0])
    G.graph["img_w"] = int(road_mask.shape[1])

    for n_xy in node_coords:
        nbrs = _get_neighbors(n_xy, skel)
        for nb in nbrs:
            if (n_xy, nb) in visited_dir:
                continue
            path_pixels = [n_xy]
            last_node_id = node_id_map[n_xy]
            prev = n_xy
            cur = nb
            visited_dir.add((prev, cur))

            def vec(a, b):
                return (b[0] - a[0], b[1] - a[1])

            last_vec = vec(prev, cur)
            since_last_split = 0

            while True:
                path_pixels.append(cur)
                since_last_split += 1
                if node_mask[cur] and cur != n_xy:
                    v_id = node_id_map[cur]
                    add_edge(last_node_id, v_id, path_pixels[:])
                    break
                nbr2 = [p for p in _get_neighbors(cur, skel) if p != prev]
                if not nbr2:
                    break
                nxt = nbr2[0]

                # Cycle guard: if this directed step has already been
                # walked (either by this walk or an earlier one), stop —
                # otherwise a closed skeleton ring (common in road graphs)
                # spins this loop forever.
                if (cur, nxt) in visited_dir:
                    add_edge(last_node_id, node_id_map.get(cur, last_node_id),
                             path_pixels[:])
                    break
                visited_dir.add((cur, nxt))

                if include_turn_nodes:
                    v_now = vec(cur, nxt)
                    dot = last_vec[0] * v_now[0] + last_vec[1] * v_now[1]
                    norm = (np.hypot(*last_vec) * np.hypot(*v_now) + 1e-6)
                    ang = np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))
                    if ang >= turn_angle_min_deg and since_last_split >= min_segment_pixels:
                        nid = next_id
                        next_id += 1
                        node_id_map[cur] = nid
                        G.add_node(nid, xy=cur)
                        add_edge(last_node_id, nid, path_pixels[:])
                        last_node_id = nid
                        path_pixels = [cur]
                        since_last_split = 0

                prev, cur = cur, nxt
                last_vec = vec(prev, cur)
    _postprocess_intersection_damage(G, damaged_mask.astype(bool), damaged_ratio_edge, min_damaged_pixels)
    G.graph["road_mask"] = road_mask.astype(bool)
    G.graph["damaged_mask"] = damaged_mask.astype(bool)
    return G


def _all_edge_samples(G: nx.Graph):
    coords = []
    meta = []
    for u, v, data in G.edges(data=True):
        dam = bool(data.get("damaged", False))
        pix = data["pixels"]
        for idx, p in enumerate(pix):
            coords.append(p)
            meta.append((u, v, idx, dam))
    if not coords:
        return np.empty((0, 2), dtype=np.int32), []
    arr = np.array(coords, dtype=np.int32)
    return arr, meta


def _split_edge_at_pixel(H: nx.Graph, u: int, v: int, target_xy: tuple) -> int:
    nid = None
    for node_id, attr in H.nodes(data=True):
        if tuple(attr.get("xy")) == tuple(target_xy):
            nid = node_id
            break
    if nid is not None:
        return nid

    def find_idx_on_edge(data):
        pix = data["pixels"]
        for i, p in enumerate(pix):
            if tuple(p) == tuple(target_xy):
                return i
        arr = np.array(pix)
        d2 = (arr[:, 0] - target_xy[0]) ** 2 + (arr[:, 1] - target_xy[1]) ** 2
        return int(np.argmin(d2))

    def split_at_index(u, v, idx):
        data = H.get_edge_data(u, v)
        pix = data["pixels"]
        if idx <= 0:
            return u
        if idx >= len(pix) - 1:
            return v
        damaged = data["damaged"]
        left = pix[:idx + 1]
        right = pix[idx:]

        def seg_len(ps):
            return float(sum(_step_len(ps[i - 1], ps[i]) for i in range(1, len(ps))))

        new_id = max(H.nodes) + 1
        H.add_node(new_id, xy=tuple(pix[idx]))
        H.remove_edge(u, v)
        H.add_edge(u, new_id, pixels=left, length=seg_len(left), damaged=damaged)
        H.add_edge(new_id, v, pixels=right, length=seg_len(right), damaged=damaged)
        return new_id

    data = H.get_edge_data(u, v)
    if data is not None:
        idx = find_idx_on_edge(data)
        return split_at_index(u, v, idx)

    for a, b, d in H.edges(data=True):
        if tuple(target_xy) in set(map(tuple, d["pixels"])):
            idx = find_idx_on_edge(d)
            return split_at_index(a, b, idx)

    raise RuntimeError(f"Cannot find edge containing pixel {target_xy}")


def _edge_pixels_in_direction(H: nx.Graph, a: int, b: int, data: Dict) -> List[tuple]:
    pix = data["pixels"]
    if not pix:
        return pix
    a_xy = tuple(H.nodes[a]["xy"])
    b_xy = tuple(H.nodes[b]["xy"])

    if tuple(pix[0]) == a_xy and tuple(pix[-1]) == b_xy:
        return pix
    if tuple(pix[0]) == b_xy and tuple(pix[-1]) == a_xy:
        return list(reversed(pix))

    d0 = (pix[0][0] - a_xy[0]) ** 2 + (pix[0][1] - a_xy[1]) ** 2
    d1 = (pix[-1][0] - a_xy[0]) ** 2 + (pix[-1][1] - a_xy[1]) ** 2
    return pix if d0 <= d1 else list(reversed(pix))


def _graph_edges_to_centerline_gdf(G: nx.Graph, transform=None, crs=None):
    img_h = G.graph.get("img_h")
    img_w = G.graph.get("img_w")

    def _pix_to_xy(p):
        r, c = p
        if transform is not None:
            # Convert pixel (row, col) to world coordinates via affine transform
            x, y = transform * (c + 0.5, r + 0.5)
            return (x, y)
        if img_h is not None:
            return (c, img_h - 1 - r)
        return (c, r)

    rows = []
    for u, v, data in G.edges(data=True):
        pix = data.get("pixels", [])
        if not pix:
            continue
        line = LineString([_pix_to_xy(p) for p in pix])
        length_val = float(data.get("length", line.length))
        rows.append({
            "u": u,
            "v": v,
            "length_px": length_val,
            "length": length_val,
            "damaged": bool(data.get("damaged", False)),
            "img_h": img_h,
            "img_w": img_w,
            "geometry": line
        })
    if not rows:
        return gpd.GeoDataFrame(
            columns=["u", "v", "length_px", "length", "damaged", "img_h", "img_w", "geometry"],
            geometry="geometry",
            crs=crs,
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


# =================================== Raster tools ================================
@mcp.tool(name="ras.count_pixels", description='''
Description:
Count pixels for specified classes in a mask image.

Parameters:
- mask_path (str): Path to the mask image file
- classes (list[int]): List of class values to count

Returns:
- counts (dict): Pixel counts per class
- total (int): Total pixel count for all specified classes
''')
@_tool_guard("ras.count_pixels")
def ras_count_pixels(mask_path: str, classes: ListInt) -> str:
    """
    Count pixels per class.
    """
    mask = imread(mask_path)

    counts = {}
    for cls in classes:
        counts[cls] = int(np.sum(mask == cls))

    total = sum(counts.values())

    result = {
        "tool": "ras.count_pixels",
        "counts": counts,
        "total": total,
        "classes": classes
    }
    return result


def ras_sample_point(image_path: str, row: int, col: int) -> str:
    """
    Sample pixel value at a specific location.
    """
    _require_geo_deps("ras.sample_point")
    import rasterio

    with rasterio.open(image_path) as src:
        if row < 0 or row >= src.height or col < 0 or col >= src.width:
            return _stub_result("ras.sample_point",
                                note="Coordinates out of bounds",
                                row=row, col=col,
                                image_bounds=f"0-{src.height}, 0-{src.width}")

        values = src.read()
        pixel_values = {}
        for band_idx in range(src.count):
            pixel_values[band_idx + 1] = int(values[band_idx, row, col])

    return {
        "tool": "ras.sample_point",
        "row": row,
        "col": col,
        "pixel_values": pixel_values,
        "is_inside": True
    }


def ras_sample_points(image_path: str, points: ListDictStrInt) -> str:
    """
    Sample pixel values at multiple locations.

    Parameters:
    - image_path: Path to raster image
    - points: List of {row, col} objects

    Returns:
    - results: List of {row, col, pixel_values} for each point
    """
    _require_geo_deps("ras.sample_points")
    import rasterio

    with rasterio.open(image_path) as src:
        values = src.read()
        results = []

        for point in points:
            row = point.get("row")
            col = point.get("col")

            if row < 0 or row >= src.height or col < 0 or col >= src.width:
                results.append({
                    "row": row,
                    "col": col,
                    "pixel_values": None,
                    "is_inside": False
                })
            else:
                pixel_values = {}
                for band_idx in range(src.count):
                    pixel_values[band_idx + 1] = int(values[band_idx, row, col])
                results.append({
                    "row": row,
                    "col": col,
                    "pixel_values": pixel_values,
                    "is_inside": True
                })

    return {
        "tool": "ras.sample_points",
        "image_path": image_path,
        "total_points": len(points),
        "results": results
    }


@mcp.tool(name="ras.sample_point", description='''
Description:
Sample pixel value at a specific (row, col) location in a raster image.

Parameters:
- image_path (str): Path to raster image file
- row (int): Pixel row (y coordinate)
- col (int): Pixel column (x coordinate)

Returns:
- row (int): Input row coordinate
- col (int): Input column coordinate
- pixel_values (dict): Pixel value for each band (1-indexed)
- is_inside (bool): Whether the coordinates are within image bounds
''')
@_tool_guard("ras.sample_point")
def ras_sample_point_tool(image_path: str, row: int, col: int) -> str:
    result = ras_sample_point(image_path, row, col)
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(name="ras.sample_points", description='''
Description:
Sample pixel values at multiple locations in a raster image.

Parameters:
- image_path (str): Path to raster image file
- points (list[dict]): List of {row, col} objects

Returns:
- total_points (int): Number of input points
- results (list): List of {row, col, pixel_values, is_inside} for each point
''')
@_tool_guard("ras.sample_points")
def ras_sample_points_tool(image_path: str, points: ListDictStrInt) -> str:
    result = ras_sample_points(image_path, points)
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(name="ras.stats", description='''
Description:
Compute summary statistics for one raster band while ignoring nodata values.

Parameters:
- image_path (str): Path to raster image
- band (int, optional): 1-indexed band number. Default is 1
- ignore_nodata (bool, optional): Ignore nodata and NaN values. Default is true

Returns:
- min (float|None): Minimum valid value
- max (float|None): Maximum valid value
- mean (float|None): Mean valid value
- std (float|None): Standard deviation of valid values
- valid_pixels (int): Number of valid pixels used
- band (int): Band index used
- nodata (float|None): Raster nodata value
''')
@_tool_guard("ras.stats")
def ras_stats(image_path: str, band: int = 1, ignore_nodata: bool = True) -> str:
    _require_geo_deps("ras.stats")
    import rasterio

    with rasterio.open(image_path) as src:
        band = int(band)
        if band < 1 or band > src.count:
            raise ValueError(f"ras.stats: band={band} out of range for raster with {src.count} band(s).")

        arr = src.read(band)
        nodata = src.nodata
        valid = np.isfinite(arr)
        if ignore_nodata and nodata is not None:
            valid &= arr != nodata

        vals = arr[valid]
        if vals.size == 0:
            return {
                "tool": "ras.stats",
                "image_path": image_path,
                "band": band,
                "nodata": float(nodata) if nodata is not None else None,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "valid_pixels": 0,
            }

        return {
            "tool": "ras.stats",
            "image_path": image_path,
            "band": band,
            "nodata": float(nodata) if nodata is not None else None,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "valid_pixels": int(vals.size),
        }


@mcp.tool(name="ras.threshold", description='''
Description:
Create a binary mask by thresholding one raster band.

Parameters:
- image_path (str): Path to raster image
- threshold (float): Threshold value
- op (str, optional): One of >=, >, <=, <, ==. Default is >=
- band (int, optional): 1-indexed band number. Default is 1
- true_value (int, optional): Pixel value for selected pixels. Default is 255
- false_value (int, optional): Pixel value for non-selected pixels. Default is 0

Returns:
- mask_path (str): Path to threshold mask in TEMP_DIR
- selected_pixels (int): Number of pixels passing the threshold
- threshold (float): Input threshold
- op (str): Comparison operator used
''')
@_tool_guard("ras.threshold")
def ras_threshold(image_path: str,
                  threshold: float,
                  op: str = ">=",
                  band: int = 1,
                  true_value: int = 255,
                  false_value: int = 0) -> str:
    _require_geo_deps("ras.threshold")
    import rasterio

    comparators = {
        ">=": lambda a, b: a >= b,
        ">": lambda a, b: a > b,
        "<=": lambda a, b: a <= b,
        "<": lambda a, b: a < b,
        "==": lambda a, b: a == b,
    }
    op_tokens = {
        ">=": "ge",
        ">": "gt",
        "<=": "le",
        "<": "lt",
        "==": "eq",
    }
    if op not in comparators:
        raise ValueError("ras.threshold: op must be one of >=, >, <=, <, ==")

    with rasterio.open(image_path) as src:
        band = int(band)
        if band < 1 or band > src.count:
            raise ValueError(f"ras.threshold: band={band} out of range for raster with {src.count} band(s).")

        arr = src.read(band)
        nodata = src.nodata
        valid = np.isfinite(arr)
        if nodata is not None:
            valid &= arr != nodata

        selected = np.zeros(arr.shape, dtype=bool)
        selected[valid] = comparators[op](arr[valid], float(threshold))
        out = np.where(selected, int(true_value), int(false_value)).astype(np.uint8)

        out_name = _short_name(Path(image_path).stem, f"thr_{op_tokens[op]}_{threshold}", suffix=".tif")
        out_path = TEMP_DIR / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        profile = _safe_raster_profile(
            src,
            driver="GTiff",
            dtype="uint8",
            count=1,
            nodata=int(false_value),
        )
        with rasterio.open(out_path.as_posix(), "w", **profile) as dst:
            dst.write(out, 1)

    return {
        "tool": "ras.threshold",
        "image_path": image_path,
        "mask_path": str(out_path),
        "threshold": float(threshold),
        "op": op,
        "band": int(band),
        "selected_pixels": int(selected.sum()),
        "true_value": int(true_value),
        "false_value": int(false_value),
    }


@mcp.tool(name="ras.diff", description='''
Description:
Subtract raster B from raster A band by band and save the difference raster.
Useful for deriving relative height such as DSM - DEM.

Parameters:
- image_a_path (str): Path to raster A
- image_b_path (str): Path to raster B
- band_a (int, optional): Band index for raster A. Default is 1
- band_b (int, optional): Band index for raster B. Default is 1

Returns:
- output_path (str): Path to the difference raster in TEMP_DIR
- min (float|None): Minimum valid difference
- max (float|None): Maximum valid difference
- mean (float|None): Mean valid difference
- valid_pixels (int): Number of valid pixels used
''')
@_tool_guard("ras.diff")
def ras_diff(image_a_path: str,
             image_b_path: str,
             band_a: int = 1,
             band_b: int = 1) -> str:
    _require_geo_deps("ras.diff")
    import rasterio

    with rasterio.open(image_a_path) as src_a, rasterio.open(image_b_path) as src_b:
        if src_a.width != src_b.width or src_a.height != src_b.height:
            raise ValueError("ras.diff: input rasters must share the same height and width.")
        if src_a.crs != src_b.crs:
            raise ValueError("ras.diff: input rasters must share the same CRS.")
        if src_a.transform != src_b.transform:
            raise ValueError("ras.diff: input rasters must share the same transform.")

        band_a = int(band_a)
        band_b = int(band_b)
        if band_a < 1 or band_a > src_a.count:
            raise ValueError(f"ras.diff: band_a={band_a} out of range for raster A with {src_a.count} band(s).")
        if band_b < 1 or band_b > src_b.count:
            raise ValueError(f"ras.diff: band_b={band_b} out of range for raster B with {src_b.count} band(s).")

        arr_a = src_a.read(band_a).astype(np.float32)
        arr_b = src_b.read(band_b).astype(np.float32)

        valid = np.isfinite(arr_a) & np.isfinite(arr_b)
        if src_a.nodata is not None:
            valid &= arr_a != src_a.nodata
        if src_b.nodata is not None:
            valid &= arr_b != src_b.nodata

        nodata_out = -9999.0
        diff = np.full(arr_a.shape, nodata_out, dtype=np.float32)
        diff[valid] = arr_a[valid] - arr_b[valid]

        out_name = _short_name(Path(image_a_path).stem, "minus", Path(image_b_path).stem, suffix=".tif")
        out_path = TEMP_DIR / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        profile = _safe_raster_profile(
            src_a,
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=nodata_out,
            compress="lzw",
        )
        with rasterio.open(out_path.as_posix(), "w", **profile) as dst:
            dst.write(diff, 1)

    vals = diff[valid]
    return {
        "tool": "ras.diff",
        "image_a_path": image_a_path,
        "image_b_path": image_b_path,
        "output_path": str(out_path),
        "band_a": int(band_a),
        "band_b": int(band_b),
        "nodata": float(nodata_out),
        "min": float(vals.min()) if vals.size else None,
        "max": float(vals.max()) if vals.size else None,
        "mean": float(vals.mean()) if vals.size else None,
        "valid_pixels": int(vals.size),
    }


def mask_compose_multiclass(
    mask_paths: ListStr,
    class_values: ListInt,
    output_name: str = "composed_mask.tif",
) -> Dict[str, Any]:
    import rasterio

    if not mask_paths:
        raise ValueError("mask.compose_multiclass requires at least one mask.")
    if len(mask_paths) != len(class_values):
        raise ValueError("mask_paths and class_values must have the same length.")

    output_path = Path(output_name)
    if not output_path.is_absolute():
        output_path = TEMP_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(mask_paths[0]) as src0:
        composed = np.zeros((src0.height, src0.width), dtype="uint8")
        profile = src0.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0)

    # Force driver to match the output file extension
    ext = output_path.suffix.lower()
    if ext in (".tif", ".tiff"):
        profile.update(driver="GTiff")
    elif ext == ".png":
        profile.update(driver="PNG")

    for mask_path, class_value in zip(mask_paths, class_values):
        with rasterio.open(mask_path) as src:
            arr = src.read(1)
            composed[arr > 0] = int(class_value)

    with rasterio.open(output_path.as_posix(), "w", **profile) as dst:
        dst.write(composed, 1)

    class_labels = {"0": "background"}
    for class_value in class_values:
        class_labels[str(class_value)] = RESCUENET_CLASSES.get(str(class_value), f"class_{class_value}")

    return {
        "tool": "mask.compose_multiclass",
        "mask_path": str(output_path),
        "class_values": [int(v) for v in class_values],
        "classes": class_labels,
    }


@mcp.tool(name="mask.compose_multiclass", description='''
Description:
Compose multiple binary masks into a single multiclass raster mask.
Each input mask contributes one class value, and any positive pixel in a mask
is assigned to that class in the output raster.

Parameters:
- mask_paths (list[str]): Paths to binary mask rasters
- class_values (list[int]): Class ids aligned with mask_paths

Output file is written to a managed location; its path is returned as mask_path.

Returns:
- mask_path (str): Path to the composed multiclass raster
- class_values (list[int]): Class ids written into the output
- classes (dict): Class label mapping for the output mask
''')
@_tool_guard("mask.compose_multiclass")
def mask_compose_multiclass_tool(mask_paths: ListStr, class_values: ListInt) -> str:
    output_name = _short_name(
        "compose_multiclass",
        *[Path(p).stem for p in mask_paths],
        "_".join(str(v) for v in class_values),
        suffix=".tif",
    )
    result = mask_compose_multiclass(mask_paths, class_values, output_name)
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(name="ras.vectorize", description='''
Description:
Vectorize a raster mask to extract polygons for specified classes.
Converts pixel-based mask to vector polygons with area statistics and saves GeoJSON.
This tool facilitates the later vector-based processing.

Parameters:
- mask_path (str): Path to the mask image file
- classes (list[int]): List of class values to vectorize
- pixel_size_m (float, optional): Pixel size (meters) to compute area_m2

Returns:
- vector_path (str): Path to the saved vector GeoJSON
- polygons (list): List of polygon properties (id/class/pixel_count/area_m2)
- count (int): Number of polygons extracted
- total_pixels (int): Total pixel count for the classes
- total_area_m2 (float|None): Total area in m2 if pixel_size_m provided
''')
@_tool_guard("ras.vectorize")
def ras_vectorize(mask_path: str, classes: ListInt, pixel_size_m: float = None) -> str:
    if mask_path is None or not os.path.exists(str(mask_path)):
        return {
            "tool": "ras.vectorize",
            "vector_path": None,
            "count": 0,
            "total_pixels": 0,
            "total_area_m2": None,
            "note": "No raster data available; zero features extracted."
        }

    class_str = "_".join([str(cls) for cls in classes])

    # Try rasterio first to preserve CRS / geo-transform from GeoTIFF;
    # fall back to skimage.imread for plain images (PNG, JPEG, etc.).
    geo_transform = None
    geo_crs = None
    mask_data = None
    try:
        import rasterio as rio

        with rio.open(mask_path) as src:
            if src.crs is not None:
                geo_crs = src.crs
                geo_transform = src.transform
            mask_data = src.read(1)  # first band as 2-D array
    except Exception:
        pass

    if mask_data is None:
        try:
            mask_data = imread(mask_path)
        except Exception:
            try:
                from PIL import Image
                mask_data = np.array(Image.open(mask_path))
            except Exception:
                import tifffile
                mask_data = tifffile.imread(mask_path)

    gdf = _vectorize_mask_to_gdf(mask_data, classes, pixel_size_m, transform=geo_transform, crs=geo_crs)
    # Ensure geometry column even when gdf is empty, so downstream tools don't crash
    if len(gdf) == 0:
        import geopandas as gpd
        gdf = gpd.GeoDataFrame({'geometry': [], 'class': [], 'pixel_count': [], 'area_m2': []},
                               geometry='geometry')

    # Handle both .png and .tif extensions
    base_name = os.path.basename(mask_path)
    for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg"):
        if base_name.lower().endswith(ext):
            base_name = base_name[:-len(ext)]
            break
    vector_path = _save_gdf(gdf, f"{base_name}_class_{class_str}.geojson")

    total_pixels = int(gdf["pixel_count"].sum()) if len(gdf) else 0
    total_area = float(gdf["area_m2"].sum()) if "area_m2" in gdf and gdf["area_m2"].notna().any() else None

    result = {
        "tool": "ras.vectorize",
        "vector_path": vector_path,
        "count": int(len(gdf)),
        "total_pixels": total_pixels,
        "total_area_m2": total_area
    }
    return result

@mcp.tool(name="ras.clip", description='''
Description:
Clip a raster by a pixel window [x1:x2, y1:y2], where x is row (top→bottom) and y is column (left→right).
Preserves georeferencing if the input has it (GeoTIFF/COG/etc.).

Parameters:
- image_path (str): Path to raster (GeoTIFF/COG/PNG/JPG…)
- x1 (int), x2 (int): row range (Python slicing semantics, x2 exclusive)
- y1 (int), y2 (int): col range (y2 exclusive)

Returns:
- output_path (str): Path to clipped raster in TEMP_DIR
- window (dict): {x1, x2, y1, y2} actually used
- out_shape (list[int,int]): [height, width]
- bands (int), dtype (str)
- crs (str|null), transform (list[float]|null)
- clamped (bool): whether input window was clamped to image bounds
''')
def ras_clip(image_path: str, x1: int, x2: int, y1: int, y2: int) -> str:
    try:
        # --- sanitize & order ---
        x1, x2 = int(min(x1, x2)), int(max(x1, x2))
        y1, y2 = int(min(y1, y2)), int(max(y1, y2))
        if x1 == x2 or y1 == y2:
            raise ValueError("ras.clip: empty window (x1==x2 or y1==y2).")

        # Try rasterio first to keep georeferencing; fall back to a PIL crop
        try:
            import rasterio
            from rasterio.windows import Window
            from rasterio.transform import Affine
            use_rasterio = True
        except Exception:
            use_rasterio = False

        _h = hashlib.md5(f"{image_path}|{x1}|{y1}|{x2}|{y2}".encode()).hexdigest()[:6]
        out_name = f"clip_{Path(image_path).stem}_{_h}.tif"
        out_path = TEMP_DIR / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if use_rasterio:
            with rasterio.open(image_path) as src:
                H, W = src.height, src.width
                cx1, cx2 = max(0, x1), min(H, x2)
                cy1, cy2 = max(0, y1), min(W, y2)
                clamped = (cx1 != x1) or (cx2 != x2) or (cy1 != y1) or (cy2 != y2)
                if cx1 >= cx2 or cy1 >= cy2:
                    raise ValueError("ras.clip: window outside image.")

                win = Window.from_slices((cx1, cx2), (cy1, cy2))
                data = src.read(window=win)  # (bands, h, w)
                new_transform = rasterio.windows.transform(win, src.transform)
                profile = _safe_raster_profile(
                    src,
                    driver="GTiff",
                    dtype=src.dtypes[0] if src.count > 0 else None,
                    count=src.count,
                )
                profile.update(height=data.shape[1], width=data.shape[2], transform=new_transform)

                with rasterio.open(out_path.as_posix(), "w", **profile) as dst:
                    dst.write(data)

                result = {
                    "tool": "ras.clip",
                    "output_path": str(out_path),
                    "window": {"x1": int(cx1), "x2": int(cx2), "y1": int(cy1), "y2": int(cy2)},
                    "out_shape": [int(profile["height"]), int(profile["width"])],
                    "bands": int(src.count),
                    "dtype": str(src.dtypes[0]) if src.count > 0 else None,
                    "crs": str(src.crs) if src.crs else None,
                    "transform": list(profile["transform"]) if isinstance(profile["transform"], Affine) else None,
                    "clamped": bool(clamped),
                }
                return json.dumps(result, ensure_ascii=False)

        # --- Fallback: PIL/NumPy (no georeferencing) ---
        img = Image.open(image_path)
        arr = np.asarray(img)
        H, W = arr.shape[0], arr.shape[1]
        cx1, cx2 = max(0, x1), min(H, x2)
        cy1, cy2 = max(0, y1), min(W, y2)
        clamped = (cx1 != x1) or (cx2 != x2) or (cy1 != y1) or (cy2 != y2)
        if cx1 >= cx2 or cy1 >= cy2:
            raise ValueError("ras.clip: window outside image.")

        clipped = arr[cx1:cx2, cy1:cy2, ...]
        Image.fromarray(clipped).save(out_path.as_posix())

        bands = 1 if clipped.ndim == 2 else clipped.shape[2]
        result = {
            "tool": "ras.clip",
            "output_path": str(out_path),
            "window": {"x1": int(cx1), "x2": int(cx2), "y1": int(cy1), "y2": int(cy2)},
            "out_shape": [int(clipped.shape[0]), int(clipped.shape[1])],
            "bands": int(bands),
            "dtype": str(clipped.dtype),
            "crs": None,
            "transform": None,
            "clamped": bool(clamped),
        }
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps(_stub_result("ras.clip", error=str(e), image_path=image_path,
                                       x1=x1, x2=x2, y1=y1, y2=y2))


@mcp.tool(name="ras.concat", description='''
Description:
Concatenate multiple rasters along the band/channel dimension into a single GeoTIFF.
All inputs must have identical height/width and, if present, the same CRS and transform.

Parameters:
- image_path_list (list[str]): list of raster paths (GeoTIFF/COG/…)
- dim (str): only "bands" or "channels" is supported for now

Returns:
- output_path (str): Path to the concatenated GeoTIFF in TEMP_DIR
- height (int), width (int)
- bands (int): total bands in the output
- dtype (str)
- crs (str|null), transform (list[float]|null)
- band_map (list[dict]): [{path, in_bands, out_start_band, out_end_band}]
- note (str)
''')
def ras_concat(image_path_list: ListStr, dim: str = "bands") -> str:
    try:
        _require_geo_deps("ras.concat")
        import rasterio
        from rasterio.transform import Affine
        import numpy as np

        if not image_path_list or len(image_path_list) == 0:
            raise ValueError("ras.concat: image_path_list is empty.")
        if dim.lower() not in ("bands", "channels"):
            raise ValueError('ras.concat: dim must be "bands" (or "channels").')

        # --- open first file as reference ---
        with rasterio.open(image_path_list[0]) as ref:
            H, W = ref.height, ref.width
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_dtype = ref.dtypes[0] if ref.count > 0 else "uint8"
            total_bands = ref.count
            arrays = [ref.read()]  # shape: (count, H, W)
            dtypes = [np.dtype(ref_dtype)]
            band_map = [{
                "path": image_path_list[0],
                "in_bands": int(ref.count),
                "out_start_band": 1,
                "out_end_band": int(ref.count)
            }]

        # --- read others & validate ---
        cur_out_band = total_bands
        for p in image_path_list[1:]:
            with rasterio.open(p) as src:
                if src.height != H or src.width != W:
                    raise ValueError(f"ras.concat: size mismatch: {p} has {src.height}x{src.width}, expected {H}x{W}.")
                # CRS/transform check (allow all-None)
                if (ref_crs is not None and src.crs is not None and ref_crs != src.crs):
                    raise ValueError(f"ras.concat: CRS mismatch between {image_path_list[0]} and {p}.")
                if (ref_transform is not None and src.transform is not None and ref_transform != src.transform):
                    raise ValueError(f"ras.concat: transform mismatch between {image_path_list[0]} and {p}.")

                arr = src.read()  # (count,H,W)
                arrays.append(arr)
                dtypes.append(np.dtype(src.dtypes[0] if src.count > 0 else ref_dtype))

                band_map.append({
                    "path": p,
                    "in_bands": int(src.count),
                    "out_start_band": int(cur_out_band + 1),
                    "out_end_band": int(cur_out_band + src.count)
                })
                cur_out_band += src.count
                total_bands += 0  # kept for readability

        total_bands = band_map[-1]["out_end_band"]

        # --- unify dtype & stack ---
        out_dtype = np.result_type(*dtypes)
        stacked = []
        for arr in arrays:
            if arr.dtype != out_dtype:
                arr = arr.astype(out_dtype, copy=False)
            stacked.append(arr)
        data = np.concatenate(stacked, axis=0)  # (sum_bands, H, W)

        # --- write output ---
        out_profile = {
            "driver": "GTiff",
            "height": H,
            "width": W,
            "count": int(data.shape[0]),
            "dtype": str(out_dtype),
            "compress": "lzw"
        }

        # Keep the georeferencing of the reference raster, if any
        with rasterio.open(image_path_list[0]) as ref0:
            if ref0.crs is not None:
                out_profile["crs"] = ref0.crs
            if ref0.transform is not None:
                out_profile["transform"] = ref0.transform

        base_stem = Path(image_path_list[0]).stem
        _h = hashlib.md5("|".join(image_path_list).encode()).hexdigest()[:6]
        out_path = TEMP_DIR / f"{base_stem}_concat_{_h}.tif"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path.as_posix(), "w", **out_profile) as dst:
            dst.write(data)

        result = {
            "tool": "ras.concat",
            "output_path": str(out_path),
            "height": int(H),
            "width": int(W),
            "bands": int(data.shape[0]),
            "dtype": str(out_dtype),
            "crs": str(out_profile.get("crs")) if out_profile.get("crs") else None,
            "transform": list(out_profile["transform"]) if isinstance(out_profile.get("transform"), Affine) else None,
            "band_map": band_map,
            "note": "Concatenated along bands; inputs must share size & georeferencing."
        }
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps(_stub_result("ras.concat", error=str(e),
                                       image_path_list=image_path_list, dim=dim))


################################# Vector tools ##################################


@mcp.tool(name="vec.connected_components", description='''
Description:
Summarize the connected components based on vector GeoJSON or a raster mask.

Parameters:
- vector_path (str): Path to the vector GeoJSON

Returns:
- count (int): Number of polygons/components
- vector_path or mask_path (str): Echo of the input path
''')
@_tool_guard("vec.connected_components")
def vec_connected_components(vector_path: str) -> str:
    if vector_path is None or not os.path.exists(str(vector_path)):
        return {
            "tool": "vec.connected_components",
            "count": 0,
            "vector_path": vector_path,
            "mode": "vector",
            "note": "No input vector available; zero components."
        }
    vector_error = None
    if GEO_DEPS_OK:
        try:
            gdf = gpd.read_file(vector_path)
            return {
                "tool": "vec.connected_components",
                "count": int(len(gdf)),
                "vector_path": vector_path,
                "mode": "vector",
            }
        except Exception as exc:
            vector_error = str(exc)
    else:
        vector_error = "geopandas/rasterio not available"

    mask = imread(vector_path)
    if mask.ndim == 3:
        mask = np.any(mask != 0, axis=2).astype(np.uint8)
    else:
        mask = (mask != 0).astype(np.uint8)
    _, count = _connected_components(mask)
    result = {
        "tool": "vec.connected_components",
        "count": int(count),
        "mask_path": vector_path,
        "mode": "raster",
    }
    if vector_error:
        result["vector_error"] = vector_error
    return result





@mcp.tool(name="vec.intersect", description='''
Description:
Find intersection between two vector datasets (GeoJSON) using vector overlay.

Parameters:
- vector_a_path (str): Path to first vector GeoJSON
- vector_b_path (str): Path to second vector GeoJSON
- pixel_size_m (float, optional): Pixel size to compute area_m2 if geometry area is in pixel units

Returns:
- intersection_count (int): Number of intersecting regions
- intersection_area_m2 (float): Total intersection area in square meters
- intersection_path (str): Path to intersection GeoJSON
''')
@_tool_guard("vec.intersect")
def vec_intersect(vector_a_path: str, vector_b_path: str, pixel_size_m: float = None) -> str:
    """
    Vector intersection (pixel space first): reports area_px2 and area_m2 (when the pixel size is known).
    """
    _require_geo_deps("vec.intersect")
    gdf_a = _read_vector_no_default_crs(vector_a_path)
    gdf_b = _read_vector_no_default_crs(vector_b_path)

    # Warn when one input has CRS and the other does not -- this typically
    # means coordinates live in incompatible spaces and the intersection
    # will be empty.
    if (gdf_a.crs is None) != (gdf_b.crs is None):
        logger.warning(
            "vec.intersect: CRS mismatch -- gdf_a.crs=%s, gdf_b.crs=%s. "
            "Results may be empty due to coordinate space incompatibility.",
            gdf_a.crs,
            gdf_b.crs,
        )

    # Normalize mixed geometry types (MultiPolygon/Polygon -> explode; etc)
    # to avoid "df1 contains mixed geometry types" error from gpd.overlay
    def _normalize_geom(gdf):
        if len(gdf) == 0 or 'geometry' not in gdf.columns:
            return gdf
        types = gdf.geometry.geom_type.unique()
        if len(types) > 1:
            try:
                gdf = gdf.explode(ignore_index=True)
            except Exception:
                pass
        return gdf

    gdf_a = _normalize_geom(gdf_a)
    gdf_b = _normalize_geom(gdf_b)

    # CRS may be unreliable (pixel-space data can be read as EPSG:4326), so do not rely on it
    # If both layers carry different CRSs, still reproject to a common one
    if gdf_a.empty or gdf_b.empty or 'geometry' not in gdf_a.columns or 'geometry' not in gdf_b.columns:
        out_name = _short_name(Path(vector_a_path or "a").stem, Path(vector_b_path or "b").stem, "intersection")
        out_path = _save_gdf(gpd.GeoDataFrame(geometry=[]), out_name)
        return {
            "tool": "vec.intersect",
            "intersection_count": 0,
            "intersection_area_px2": 0.0,
            "intersection_area_m2": 0.0 if pixel_size_m is not None else None,
            "pixel_size_m": float(pixel_size_m) if pixel_size_m is not None else None,
            "intersection_path": out_path,
            "note": "One or both inputs are empty or have no geometry."
        }

    # If both layers carry different CRSs, reproject to a common one
    if gdf_a.crs and gdf_b.crs and gdf_a.crs != gdf_b.crs:
        gdf_b = gdf_b.to_crs(gdf_a.crs)

    inter = gpd.overlay(gdf_a, gdf_b, how="intersection", keep_geom_type=True)
    if inter.empty:
        out_name = _short_name(Path(vector_a_path).stem, Path(vector_b_path).stem, "intersection")
        out_path = _save_gdf(inter, out_name)
        return {
            "tool": "vec.intersect",
            "intersection_count": 0,
            "intersection_area_px2": 0.0,
            "intersection_area_m2": 0.0 if pixel_size_m is not None else None,
            "pixel_size_m": float(pixel_size_m) if pixel_size_m is not None else None,
            "intersection_path": out_path
        }

    px = None
    if pixel_size_m is not None:
        px = float(pixel_size_m)
    else:
        px_cols = [c for c in inter.columns if c.startswith("pixel_size_m")]
        for c in px_cols:
            s = inter[c].dropna()
            if len(s) > 0:
                try:
                    px = float(s.iloc[0])
                    break
                except Exception:
                    pass

    inter["area_px2"] = inter.geometry.area.astype(float)
    inter["pixel_size_m"] = px
    inter["area_m2"] = (inter["area_px2"] * (px ** 2)).astype(float) if px is not None else None

    out_name = _short_name(Path(vector_a_path).stem, Path(vector_b_path).stem, "intersection")
    inter_path = _save_gdf(inter, out_name)

    intersection_count = int(len(inter))
    area_px2_sum = float(inter["area_px2"].sum()) if intersection_count > 0 else 0.0
    area_m2_sum = float(inter["area_m2"].sum()) if (px is not None and intersection_count > 0) else None

    result = {
        "tool": "vec.intersect",
        "intersection_count": intersection_count,
        "intersection_area_px2": area_px2_sum,
        "intersection_area_m2": area_m2_sum,
        "pixel_size_m": px,
        "intersection_path": inter_path,
    }
    return result





@mcp.tool(name="vec.max_polygon", description='''
Description:
Find the largest polygon(s) from existing vector data based on area.

Parameters:
- vector_path (str): Path to the vector GeoJSON
- pixel_size_m (float, optional): Pixel size to compute area_m2 if geometry units are pixel
- topk (int, optional): Number of top polygons to return. Default is 1.

Returns:
- max_area_m2 (float): Area of the largest polygon in square meters
- max_pixel_count (int): Pixel count of the largest polygon (if available)
- topk_polygons (list): Top-k polygons by area
- total_polygons (int): Total polygons found
- vector_path (str): Echo of input
''')
@_tool_guard("vec.max_polygon")
def vec_max_polygon(vector_path: str, pixel_size_m: float = None, topk: int = 1) -> str:
    """
    Largest / top-K polygons of an existing vector layer.
    """
    _require_geo_deps("vec.max_polygon")
    gdf = _read_vector_no_default_crs(vector_path)

    if len(gdf) == 0:
        top_areas = []
        max_area = 0.0
        max_pixels = 0
    else:
        if "area_m2" not in gdf or gdf["area_m2"].isna().all():
            gdf["area_m2"] = gdf.geometry.area * (pixel_size_m ** 2 if pixel_size_m else 1)

        if "pixel_count" not in gdf:
            gdf["pixel_count"] = None

        sorted_gdf = gdf.sort_values(by="area_m2", ascending=False)
        top_slice = sorted_gdf.head(topk)
        # Only select columns that actually exist to avoid KeyError on non-standard inputs
        wanted_cols = ["id", "class", "pixel_count", "area_m2"]
        avail_cols = [c for c in wanted_cols if c in top_slice.columns]
        top_areas = top_slice[avail_cols].to_dict(orient="records")
        max_area = float(top_slice.iloc[0]["area_m2"])
        pc = top_slice.iloc[0].get("pixel_count") if "pixel_count" in top_slice.columns else None
        max_pixels = int(pc) if pc is not None and pc == pc else None  # pc==pc filters NaN

    result = {
        "tool": "vec.max_polygon",
        "max_area_m2": float(max_area),
        "max_pixel_count": max_pixels,
        "topk_polygons": top_areas,
        "total_polygons": int(len(gdf)),
        "vector_path": vector_path
    }
    return result


def vec_centroid(vector_path: str, feature_idx: int = 0) -> str:
    """
    Calculate the centroid of a polygon/point feature.

    Parameters:
    - vector_path: Path to vector GeoJSON
    - feature_idx: Index of feature to get centroid for (default: 0, the largest)

    Returns:
    - centroid: {lat, lon} or {pixel_row, pixel_col} depending on whether the vector has CRS
    """
    _require_geo_deps("vec.centroid")
    gdf = _read_vector_no_default_crs(vector_path)

    if gdf.empty:
        return {"tool": "vec.centroid", "centroid_x": None, "centroid_y": None,
                "vector_path": vector_path, "note": "Empty input; no features to compute centroid."}

    # If feature_idx is -1, find the largest polygon
    if feature_idx == -1:
        if "area_m2" in gdf.columns and gdf["area_m2"].notna().any():
            feature_idx = gdf["area_m2"].idxmax()
        else:
            # Calculate areas based on geometry
            gdf["temp_area"] = gdf.geometry.area
            feature_idx = gdf["temp_area"].idxmax()

    geom = gdf.loc[feature_idx].geometry
    if geom is None or geom.is_empty:
        return _stub_result("vec.centroid", note="Empty geometry", feature_idx=feature_idx)

    centroid = geom.centroid

    # Check if the vector has a CRS (geographic/projected coordinate system)
    has_crs = gdf.crs is not None

    result = {
        "tool": "vec.centroid",
        "feature_idx": int(feature_idx),
        "geometry_type": geom.geom_type
    }

    if has_crs:
        # Return latitude/longitude for geographic CRS
        result["latitude"] = float(centroid.y)
        result["longitude"] = float(centroid.x)
    else:
        # Return pixel_row/pixel_col for pixel coordinates (no CRS)
        result["pixel_row"] = float(centroid.y)
        result["pixel_col"] = float(centroid.x)

    return result




@mcp.tool(name="vec.centroid", description='''
Description:
Calculate the centroid of a polygon or point feature in a vector file.
Useful for finding the center point of the largest damaged area (explosion center).

Parameters:
- vector_path (str): Path to input vector GeoJSON
- feature_idx (int): Index of feature to get centroid for (default: 0, use -1 for largest)

Returns:
- feature_idx (int): The feature index used
- geometry_type (str): Type of the original geometry
- If vector has CRS: latitude (float), longitude (float) in WGS84
- If vector has no CRS: pixel_row (float), pixel_col (float) in pixel coordinates
''')
@_tool_guard("vec.centroid")
def vec_centroid_tool(vector_path: str, feature_idx: int = 0) -> str:
    result = vec_centroid(vector_path, feature_idx)
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(name="vec.pixel_to_world", description='''
Description:
Convert pixel coordinates to world coordinates (WGS84) using a reference GeoTIFF.

Parameters:
- raster_path (str): Path to the reference GeoTIFF (e.g., post-disaster image)
- pixel_row (int): Pixel row coordinate (y)
- pixel_col (int): Pixel column coordinate (x)

Returns:
- latitude (float): Y coordinate in WGS84
- longitude (float): X coordinate in WGS84
- pixel_row (int): Input pixel row
- pixel_col (int): Input pixel col
''')
@_tool_guard("vec.pixel_to_world")
def vec_pixel_to_world(raster_path: str, pixel_row: int, pixel_col: int) -> str:
    _require_geo_deps("vec.pixel_to_world")
    import rasterio
    if not RASTERIO_OK:
        return _stub_result("vec.pixel_to_world", note="rasterio not available")

    try:
        with rasterio.open(raster_path) as src:
            # Get the affine transform
            transform = src.transform

            # Convert pixel to world coordinates
            # rasterio uses (col, row) order for pixel_inverse
            world_x = transform[2] + transform[0] * pixel_col + transform[1] * pixel_row
            world_y = transform[5] + transform[4] * pixel_col + transform[3] * pixel_row

            # If CRS is not WGS84, convert
            if src.crs is not None and src.crs.to_epsg() != 4326:
                from pyproj import Transformer
                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                lon, lat = transformer.transform(world_x, world_y)
            else:
                lon, lat = world_x, world_y

            return {
                "tool": "vec.pixel_to_world",
                "pixel_row": pixel_row,
                "pixel_col": pixel_col,
                "latitude": float(lat),
                "longitude": float(lon),
                "crs": str(src.crs) if src.crs else "unknown"
            }
    except Exception as e:
        return _stub_result("vec.pixel_to_world", error=str(e))


@mcp.tool(name="vec.buffer", description='''
Description:
Create a buffer around all geometries in a vector file.
The distance is interpreted in the CRS units (meters if CRS is metric).

Parameters:
- vector_path (str): Path to input vector GeoJSON
- meter (float): Buffer distance

Returns:
- buffer_path (str): Path to buffered GeoJSON
''')
@_tool_guard("vec.buffer")
def vec_buffer(vector_path: str, meter: float) -> str:
    _require_geo_deps("vec.buffer")
    gdf = _read_vector_no_default_crs(vector_path)
    if gdf.empty:
        return {"tool": "vec.buffer", "buffer_path": None, "vector_path": vector_path,
                "note": "Empty input; no features to buffer."}

    px_size = None
    if "pixel_size_m" in gdf.columns:
        non_null_px = gdf["pixel_size_m"].dropna()
        if len(non_null_px) > 0:
            try:
                px_size = float(non_null_px.iloc[0])
            except Exception:
                px_size = None

    buffer_dist = meter

    if px_size and gdf.crs is None:
        buffer_dist = meter / px_size  # convert meters to pixel units

    buffered = gdf.copy()
    buffered["geometry"] = gdf.geometry.buffer(buffer_dist)
    buffered["pixel_size_m"] = buffered.get("pixel_size_m", px_size)
    buffered["buffer_input_m"] = meter
    buffered["buffer_used_units"] = buffer_dist
    fname = os.path.basename(vector_path).replace(".geojson", f"_buffer_{meter}m.geojson")
    buffer_path = _save_gdf(buffered, fname)

    result = {
        "tool": "vec.buffer",
        "buffer_path": buffer_path,
        "distance_input_m": meter,
        "distance_used_units": buffer_dist,
        "input_vector": vector_path
    }
    return result

@mcp.tool(name="vec.kbuffer", description='''
Description:
Create multiple buffers for given distances around geometries in a vector file.
The distances are interpreted in CRS units (meters if CRS is metric).

Parameters:
- vector_path (str): Path to input vector GeoJSON
- meters (list[float]): Distances for buffering

Returns:
- buffers (list): List of {distance, buffer_path}
''')
@_tool_guard("vec.kbuffer")
def vec_kbuffer(vector_path: str, meters: ListFloat) -> str:
    _require_geo_deps("vec.kbuffer")
    gdf = _read_vector_no_default_crs(vector_path)
    if gdf.empty:
        return {"tool": "vec.kbuffer", "buffer_path": None, "vector_path": vector_path,
                "meters": meters, "note": "Empty input; no features to buffer."}

    px_size = None
    if "pixel_size_m" in gdf.columns:
        non_null_px = gdf["pixel_size_m"].dropna()
        if len(non_null_px) > 0:
            try:
                px_size = float(non_null_px.iloc[0])
            except Exception:
                px_size = None

    buffers = []
    for dist in meters:
        buffer_dist = dist
        if px_size and gdf.crs is None:
            buffer_dist = dist / px_size

        buffered = gdf.copy()
        buffered["geometry"] = gdf.geometry.buffer(buffer_dist)
        buffered["pixel_size_m"] = buffered.get("pixel_size_m", px_size)
        buffered["buffer_input_m"] = dist
        buffered["buffer_used_units"] = buffer_dist
        buffer_path = _save_gdf(buffered, f"buffer_{dist}.geojson")
        buffers.append({"distance_input_m": dist, "distance_used_units": buffer_dist, "buffer_path": buffer_path})

    result = {
        "tool": "vec.kbuffer",
        "buffers": buffers,
        "input_vector": vector_path
    }
    return result



def color2labelidx(mask: np.ndarray) -> np.ndarray:
    mapping_mat = np.array([3, 4, 5]).reshape(3, 1)
    labels = [0, 1, 2, 3]
    features = np.array([0, 0, 0, 255, 255, 255, 248, 179, 101, 255, 0, 0]).reshape(-1, 3)
    keys = np.matmul(features, mapping_mat).squeeze()
    q = np.matmul(mask, mapping_mat).squeeze()

    out = np.zeros_like(q)
    for label, k in zip(labels, keys):
        out = np.where(q == k, label, out)
    return out



@mcp.tool(name="vec.road_centerline_extraction", description='''
Description:
Extract road centerlines directly from a road mask by skeletonizing and outputting
centerlines as GeoJSON LineStrings. The caller must supply the class-id lists
corresponding to intact vs damaged road pixels (look them up in the upstream
segmentation tool's returned `classes` dict).

Parameters:
- mask_path (str): Path to road mask
- intact_classes (list[int], optional): Class ids treated as intact road
- damaged_classes (list[int], optional): Class ids treated as damaged road

Returns:
- centerline_path (str): Path to centerline GeoJSON
''')
@_tool_guard("vec.road_centerline_extraction")
def road_centerline_extraction(mask_path: str,
                               intact_classes: ListInt = None,
                               damaged_classes: ListInt = None) -> str:
    _require_geo_deps("vec.road_centerline_extraction")
    _require_skimage("vec.road_centerline_extraction")
    if intact_classes is None:
        intact_classes = [1]
    if damaged_classes is None:
        damaged_classes = [2, 3]

    # Try rasterio first to preserve CRS / geo-transform from GeoTIFF;
    # fall back to skimage.imread for plain images (PNG, JPEG, etc.).
    geo_transform = None
    geo_crs = None
    mask = None
    try:
        import rasterio as rio

        with rio.open(mask_path) as src:
            if src.crs is not None:
                geo_crs = src.crs
                geo_transform = src.transform
            mask = src.read(1)
    except Exception:
        pass

    if mask is None:
        mask = imread(mask_path)
    if mask.ndim == 3:
        mask = color2labelidx(mask)

    road_all = np.isin(mask, intact_classes + damaged_classes)
    damage_mask = np.isin(mask, damaged_classes)

    G = _build_road_graph(road_all, damage_mask)
    center_gdf = _graph_edges_to_centerline_gdf(G, transform=geo_transform, crs=geo_crs)
    center_path = _save_gdf(center_gdf, "road_centerline.geojson")

    result = {
        "tool": "vec.road_centerline_extraction",
        "centerline_path": center_path,
        "input_mask": mask_path,
        "edge_count": int(len(center_gdf))
    }
    return result






@mcp.tool(name="vec.build_graph", description='''
Description:
Build a road graph from centerline GeoJSON. Nodes are endpoints of each line, edges carry pixel coordinates and length.

Parameters:
- centerline_path (str): Path to centerline GeoJSON (LineString geometries, pixel space)

Returns:
- graph_path (str): Path to saved graph pickle (networkx graphml)
- node_count (int)
- edge_count (int)
''')
def build_road_graph(centerline_path: str) -> str:
    try:
        _require_geo_deps("vec.build_graph")
        gdf = _read_vector_no_default_crs(centerline_path)
        if gdf.empty:
            return json.dumps(_stub_result("vec.build_graph", note="Empty centerline", centerline_path=centerline_path))

        G = nx.Graph()
        node_id_map = {}
        img_h = None
        img_w = None
        if "img_h" in gdf.columns:
            try:
                first_h = gdf["img_h"].dropna()
                if len(first_h):
                    img_h = int(first_h.iloc[0])
            except Exception:
                img_h = None
        if "img_w" in gdf.columns:
            try:
                first_w = gdf["img_w"].dropna()
                if len(first_w):
                    img_w = int(first_w.iloc[0])
            except Exception:
                img_w = None

        def get_node_id(xy):
            if xy not in node_id_map:
                node_id_map[xy] = len(node_id_map)
                G.add_node(node_id_map[xy], xy=xy)
            return node_id_map[xy]

        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom.is_empty:
                continue
            coords = list(geom.coords)
            if img_h is not None:
                pixels = [(int(round(img_h - 1 - y)), int(round(x))) for x, y in coords]
            else:
                pixels = [(int(round(y)), int(round(x))) for x, y in coords]  # store as (row, col)
            if len(pixels) < 2:
                continue
            u_xy = pixels[0]
            v_xy = pixels[-1]
            u = get_node_id(u_xy)
            v = get_node_id(v_xy)
            # length = float(len(pixels) - 1)
            length = float(sum(_step_len(pixels[i - 1], pixels[i]) for i in range(1, len(pixels))))
            dam_val = row.get("damaged", False)
            if isinstance(dam_val, str):
                damaged_flag = dam_val.lower() == "true"
            else:
                damaged_flag = bool(dam_val)
            G.add_edge(u, v, pixels=pixels, length=length, damaged=damaged_flag)

        if img_h is not None:
            G.graph["img_h"] = img_h
        if img_w is not None:
            G.graph["img_w"] = img_w

        _h = hashlib.md5(centerline_path.encode("utf-8")).hexdigest()[:6]
        graph_path = TEMP_DIR / f"road_graph_{_h}.graphml"
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        _save_graph_graphml(G, graph_path)

        result = {
            "tool": "vec.build_graph",
            "graph_path": str(graph_path),
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "input_centerline": centerline_path
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("vec.build_graph", error=str(e), centerline_path=centerline_path))

SNAP_BUFFER_PX = 10


def shortest_path_on_intact_roads(G: nx.Graph, start_xy: tuple, end_xy: tuple,
                                  snap_buffer_px: int = SNAP_BUFFER_PX):
    H = G.copy()

    samples_all, meta_all = _all_edge_samples(G)
    if len(samples_all) == 0:
        return None, None, None, None, None, False

    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(samples_all)
    except Exception:
        tree = None

    def _snap(p: tuple):
        if tree is not None:
            k = min(64, len(samples_all))
            dists, idxs = tree.query(np.asarray(p, dtype=np.int32), k=k)
            dists = np.atleast_1d(dists)
            idxs = np.atleast_1d(idxs)
        else:
            dif = samples_all - np.array(p, dtype=np.int32)
            d2 = dif[:, 0] * dif[:, 0] + dif[:, 1] * dif[:, 1]
            order = np.argsort(d2)[:64]
            dists = np.sqrt(d2[order])
            idxs = order

        within = [(dist, idx) for dist, idx in zip(dists, idxs) if dist <= snap_buffer_px]
        if not within:
            return None  # nothing within buffer

        intact_pick = None
        damaged_pick = None
        for dist, idx in within:
            nearest_xy = tuple(map(int, samples_all[idx]))
            tie_idx = np.where((samples_all[:, 0] == nearest_xy[0]) & (samples_all[:, 1] == nearest_xy[1]))[0]
            for j in tie_idx:
                u, v, px_idx, dam = meta_all[j]
                if not dam and intact_pick is None:
                    intact_pick = (u, v, px_idx, nearest_xy, dam)
                if damaged_pick is None:
                    damaged_pick = (u, v, px_idx, nearest_xy, dam)
            if intact_pick is not None:
                break
        return intact_pick if intact_pick is not None else damaged_pick

    start_pick = _snap(start_xy)
    end_pick = _snap(end_xy)
    snap_failed = False
    if start_pick is None or end_pick is None:
        snap_failed = True
        return None, None, start_xy, end_xy, True, snap_failed

    su, sv, sidx, sxy, sdam = start_pick
    tu, tv, tidx, txy, tdam = end_pick

    if sdam or tdam:
        return None, None, sxy, txy, True, snap_failed

    s_node = _split_edge_at_pixel(H, su, sv, sxy)
    t_node = _split_edge_at_pixel(H, tu, tv, txy)

    H.remove_edges_from([(u, v) for u, v, d in H.edges(data=True) if d.get("damaged", False)])

    try:
        node_path = nx.shortest_path(H, s_node, t_node, weight="length")
    except nx.NetworkXNoPath:
        return None, None, sxy, txy, False, snap_failed

    path_pixels: List[tuple] = []
    total_len = 0.0
    for a, b in zip(node_path[:-1], node_path[1:]):
        data = H.get_edge_data(a, b)
        if data is None:
            continue
        pix = _edge_pixels_in_direction(H, a, b, data)
        total_len += float(data["length"])
        if path_pixels:
            path_pixels.extend(pix[1:])
        else:
            path_pixels.extend(pix[:])

    return total_len, path_pixels, sxy, txy, False, snap_failed

@mcp.tool(name="vec.graph_shortest_path", description='''
Description:
Find shortest path on an intact-road graph.

Parameters:
- graph_path (str): Path to graph graphml produced by vec.build_graph
- start (list[int]): [row, col] start point (pixel coordinates)
- end (list[int]): [row, col] end point (pixel coordinates)
- pixel_size_m (float, optional): Pixel size in meters for distance conversion

Returns:
- distance_px (float|null): Shortest path length in pixels (null if no path)
- distance_m (float|null): Shortest path length in meters (null if no path)
- path_pixels (list): List of [row, col] along path
- path_geojson (str|null): Path to saved path GeoJSON (null if no path)
- snapped_start/end (list|None): Snapped coordinates on centerline
''')
@_tool_guard("vec.graph_shortest_path")
def graph_shortest_path(graph_path: str,
                        start: ListInt,
                        end: ListInt,
                        pixel_size_m: float = 1.0,
                        snap_buffer_px: int = SNAP_BUFFER_PX) -> str:
    _require_geo_deps("vec.graph_shortest_path")
    G = _load_graph_graphml(graph_path)

    img_h = G.graph.get("img_h")

    total_len_px, path_pixels, sxy, txy, snapped_on_damaged, snap_failed = shortest_path_on_intact_roads(
        G, tuple(start), tuple(end), snap_buffer_px=snap_buffer_px
    )

    def _pix_to_xy(p):
        if img_h is not None:
            return (p[1], img_h - 1 - p[0])
        return (p[1], p[0])

    if total_len_px is None or snapped_on_damaged:
        start_xy = tuple(sxy) if sxy is not None else tuple(start)
        end_xy = tuple(txy) if txy is not None else tuple(end)
        line = LineString([_pix_to_xy(start_xy), _pix_to_xy(end_xy)])
        gdf = gpd.GeoDataFrame([{
            "distance_px": None,
            "distance_m": None,
            "snapped_on_damaged": snapped_on_damaged,
            "snap_failed": snap_failed
        }], geometry=[line], crs=None)
        path_geojson = _save_gdf(gdf, "road_path.geojson")
        return {
            "tool": "vec.graph_shortest_path",
            "distance_px": None,
            "distance_m": None,
            "path_pixels": [],
            "path_geojson": path_geojson,
            "snapped_start": start_xy,
            "snapped_end": end_xy,
            "snapped_on_damaged": snapped_on_damaged,
            "snap_failed": snap_failed
        }

    distance_m = float(total_len_px * pixel_size_m)
    line = LineString([_pix_to_xy(c) for c in path_pixels])
    gdf = gpd.GeoDataFrame([{
        "distance_px": total_len_px,
        "distance_m": distance_m
    }], geometry=[line], crs=None)
    path_geojson = _save_gdf(gdf, "road_path.geojson")

    result = {
        "tool": "vec.graph_shortest_path",
        "distance_px": float(total_len_px),
        "distance_m": distance_m,
        "path_pixels": [list(p) for p in path_pixels],
        "path_geojson": path_geojson,
        "snapped_start": list(sxy) if sxy else None,
        "snapped_end": list(txy) if txy else None,
        "snapped_on_damaged": snapped_on_damaged,
        "snap_failed": snap_failed
    }
    return result



@mcp.tool(name="vec.length", description='''
Description:
Compute total length (m) for LineString features.

Parameters:
- vector_path (str)
- pixel_size_m (float): meters per pixel for pixel-space vectors

Returns:
- total_length_m (float)
- table_path (str): CSV with {id, length_m}
''')
@_tool_guard("vec.length")
def vec_length(vector_path: str, pixel_size_m: float = None) -> str:
    _require_geo_deps("vec.length")
    gdf = _read_vector_no_default_crs(vector_path)
    if gdf.empty:
        return {"tool": "vec.length", "total_length_m": 0.0, "count": 0,
                "vector_path": vector_path, "note": "Empty input; no features to measure."}
    if "id" not in gdf.columns:
        gdf["id"] = list(range(1, len(gdf) + 1))

    if pixel_size_m is not None:
        gdf["length_m"] = gdf.geometry.length * pixel_size_m
    else:
        if gdf.crs is None:
            raise ValueError("Need pixel_size_m for pixel-space vectors.")
        gdf["length_m"] = gdf.geometry.length

    total_len = float(gdf["length_m"].sum())
    _h = hashlib.md5(vector_path.encode("utf-8")).hexdigest()[:6]
    table_path = TEMP_DIR / f"vec_length_table_{_h}.csv"
    gdf.drop(columns=["geometry"]).to_csv(table_path, index=False)

    return {
        "tool": "vec.length",
        "total_length_m": total_len,
        "table_path": str(table_path),
        "input_vector": vector_path
    }




@mcp.tool(name="vec.perimeter", description='''
Description:
Compute perimeters (m) for polygon features.

Parameters:
- vector_path (str)
- pixel_size_m (float): meters per pixel for pixel-space vectors

Returns:
- total_perimeter_m (float)
- table_path (str)
''')
@_tool_guard("vec.perimeter")
def vec_perimeter(vector_path: str, pixel_size_m: float = None) -> str:
    _require_geo_deps("vec.perimeter")
    gdf = _read_vector_no_default_crs(vector_path)
    if gdf.empty:
        return {"tool": "vec.perimeter", "total_perimeter_m": 0.0, "count": 0,
                "vector_path": vector_path, "note": "Empty input; no features to measure."}
    if "id" not in gdf.columns:
        gdf["id"] = list(range(1, len(gdf) + 1))

    if pixel_size_m is None:
        raise ValueError("vec.perimeter: pixel-space vectors require pixel_size_m.")
    gdf["perimeter_m"] = gdf.geometry.length * pixel_size_m

    total_p = float(gdf["perimeter_m"].sum())
    _h = hashlib.md5(vector_path.encode("utf-8")).hexdigest()[:6]
    table_path = TEMP_DIR / f"vec_perimeter_table_{_h}.csv"
    gdf.drop(columns=["geometry"]).to_csv(table_path, index=False)

    return {
        "tool": "vec.perimeter",
        "total_perimeter_m": total_p,
        "table_path": str(table_path),
        "input_vector": vector_path
    }

@mcp.tool(name="vec.raster_stats", description='''
Description:
Sample raster values under vector geometries and compute summary statistics.
Supports point, line, and polygon geometries. For points, the raster value at the
point pixel is used. For non-point geometries, the raster is sampled over covered pixels.

Parameters:
- vector_path (str): Path to vector GeoJSON/Shapefile
- raster_path (str): Path to raster image
- stats (list[str], optional): Any of mean, min, max, std. Default is [mean, min, max]
- band (int, optional): Raster band to use. Default is 1
- return_features (bool, optional): Whether to include per-feature statistics. Default is true
- min_pixels (int, optional): Minimum covered pixels required to keep a feature. Default is 1
- all_touched (bool, optional): Whether to rasterize vectors with all_touched=True. Default is false

Returns:
- feature_count (int): Number of features retained after min_pixels filtering
- feature_stats (list[dict], optional): Per-feature statistics
- output_vector_path (str): GeoJSON with attached statistics for retained features
- mean/min/max/std (float|None): Aggregate stats over the union of retained features
''')
@_tool_guard("vec.raster_stats")
def vec_raster_stats(vector_path: str,
                     raster_path: str,
                     stats: ListStr = None,
                     band: int = 1,
                     return_features: bool = True,
                     min_pixels: int = 1,
                     all_touched: bool = False) -> str:
    _require_geo_deps("vec.raster_stats")
    import rasterio

    stats = stats or ["mean", "min", "max"]
    allowed_stats = {"mean", "min", "max", "std"}
    bad_stats = [s for s in stats if s not in allowed_stats]
    if bad_stats:
        raise ValueError(f"vec.raster_stats: unsupported stats={bad_stats}")

    gdf = _read_vector_no_default_crs(vector_path)
    if gdf.empty:
        return {"tool": "vec.raster_stats", "feature_stats": [], "count": 0,
                "vector_path": vector_path, "raster_path": raster_path,
                "note": "Empty input; no features for raster stats."}

    with rasterio.open(raster_path) as src:
        band = int(band)
        if band < 1 or band > src.count:
            raise ValueError(f"vec.raster_stats: band={band} out of range for raster with {src.count} band(s).")

        if gdf.crs and src.crs and gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        arr = src.read(band)
        nodata = src.nodata
        valid = np.isfinite(arr)
        if nodata is not None:
            valid &= arr != nodata

        pixel_area_m2 = None
        try:
            pixel_area_m2 = abs(float(src.transform.a) * float(src.transform.e))
        except Exception:
            pixel_area_m2 = None
        if pixel_area_m2 is not None:
            pixel_area_m2 = abs(pixel_area_m2)

        union_mask = np.zeros(arr.shape, dtype=bool)
        kept_rows = []
        feature_stats = []

        def _scalarize(value: Any) -> Any:
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, float) and np.isnan(value):
                return None
            return value

        for feature_idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            sample_mask = np.zeros(arr.shape, dtype=bool)
            pixel_row = None
            pixel_col = None

            if geom.geom_type == "Point":
                pixel_row, pixel_col = src.index(geom.x, geom.y)
                if 0 <= pixel_row < src.height and 0 <= pixel_col < src.width:
                    sample_mask[pixel_row, pixel_col] = True
            else:
                sample_mask = features.geometry_mask(
                    [geom.__geo_interface__],
                    out_shape=arr.shape,
                    transform=src.transform,
                    invert=True,
                    all_touched=bool(all_touched),
                )

            sample_mask &= valid
            pixel_count = int(sample_mask.sum())
            if pixel_count < int(min_pixels):
                continue

            vals = arr[sample_mask]
            item = {
                "feature_idx": int(feature_idx),
                "geometry_type": geom.geom_type,
                "pixel_count": pixel_count,
            }
            if pixel_area_m2 is not None:
                item["sampled_area_m2"] = float(pixel_count * pixel_area_m2)
            if pixel_row is not None and pixel_col is not None:
                item["pixel_row"] = int(pixel_row)
                item["pixel_col"] = int(pixel_col)

            for col in ("osm_id", "name", "type", "ref"):
                if col in row.index:
                    item[col] = _scalarize(row[col])

            if "mean" in stats:
                item["mean"] = float(vals.mean())
            if "min" in stats:
                item["min"] = float(vals.min())
            if "max" in stats:
                item["max"] = float(vals.max())
            if "std" in stats:
                item["std"] = float(vals.std())

            kept_rows.append((feature_idx, item))
            feature_stats.append(item)
            union_mask |= sample_mask

        if kept_rows:
            kept_gdf = gdf.loc[[idx for idx, _ in kept_rows]].copy()
            for idx, item in kept_rows:
                for key, value in item.items():
                    if key == "feature_idx":
                        continue
                    kept_gdf.loc[idx, key] = value
        else:
            kept_gdf = gdf.iloc[0:0].copy()

        out_name = _short_name(Path(vector_path).stem, Path(raster_path).stem, "stats", suffix=".geojson")
        output_vector_path = _save_gdf(kept_gdf, out_name)

        result = {
            "tool": "vec.raster_stats",
            "vector_path": vector_path,
            "raster_path": raster_path,
            "output_vector_path": output_vector_path,
            "band": int(band),
            "stats": list(stats),
            "feature_count": int(len(feature_stats)),
            "min_pixels": int(min_pixels),
            "pixel_area_m2": float(pixel_area_m2) if pixel_area_m2 is not None else None,
        }
        if return_features:
            result["feature_stats"] = feature_stats

        union_vals = arr[union_mask]
        if union_vals.size == 0:
            for stat_name in allowed_stats:
                if stat_name in stats:
                    result[stat_name] = None
        else:
            if "mean" in stats:
                result["mean"] = float(union_vals.mean())
            if "min" in stats:
                result["min"] = float(union_vals.min())
            if "max" in stats:
                result["max"] = float(union_vals.max())
            if "std" in stats:
                result["std"] = float(union_vals.std())

        return result



@mcp.tool(name="vec.filter_by_area", description='''
Description:
Remove polygons smaller than threshold; useful for cleaning noise.

Parameters:
- vector_path (str)
- min_area_m2 (float)
- pixel_size_m (float): meters per pixel for pixel-space vectors

Returns:
- vector_path (str)
''')
@_tool_guard("vec.filter_by_area")
def vec_filter_by_area(vector_path: str, min_area_m2: float, pixel_size_m: float) -> str:
    _require_geo_deps("vec.filter_by_area")
    gdf = _read_vector_no_default_crs(vector_path)
    if gdf.empty:
        return {"tool": "vec.filter_by_area", "vector_path": vector_path,
                "original_count": 0, "filtered_count": 0, "removed_count": 0,
                "min_area_m2": float(min_area_m2)}

    if pixel_size_m is None:
        raise ValueError("vec.filter_by_area: pixel_size_m is required for pixel-space vectors.")

    if "pixel_size_m" in gdf.columns:
        px = gdf["pixel_size_m"].fillna(pixel_size_m).astype(float)
    else:
        px = float(pixel_size_m)

    area_m2 = gdf.geometry.area * (px ** 2)
    out = gdf[area_m2 >= float(min_area_m2)].copy()

    out["area_m2"] = area_m2.loc[out.index].astype(float)
    out["pixel_size_m"] = px if "pixel_size_m" in out.columns else float(pixel_size_m)

    out_path = _save_gdf(out, "filtered_by_area.geojson")
    return {"tool": "vec.filter_by_area", "vector_path": out_path, "min_area_m2": float(min_area_m2)}


@mcp.tool(name="vec.filter_by_attribute", description='''
Description:
Filter vector features by attribute values. This is useful when a vector file
contains mixed feature types and the task only wants a semantic subset, such as
hospitals or commercial POIs based on the `types` attribute.

Parameters:
- vector_path (str)
- attribute_name (str): attribute/column to inspect, such as `types`
- allowed_values (list[str]): accepted token values, such as `["hospital"]`

Returns:
- vector_path (str)
- total_input (int)
- total_filtered (int)
''')
@_tool_guard("vec.filter_by_attribute")
def vec_filter_by_attribute(vector_path: str, attribute_name: str, allowed_values: ListStr) -> str:
    _require_geo_deps("vec.filter_by_attribute")
    gdf = _read_vector_no_default_crs(vector_path)
    if gdf.empty:
        return {"tool": "vec.filter_by_attribute", "vector_path": vector_path,
                "total_input": 0, "total_filtered": 0,
                "note": "Empty input; no features to filter."}

    attr = attribute_name
    if attr not in gdf.columns and "." in attr:
        fallback = attr.split(".")[-1]
        if fallback in gdf.columns:
            attr = fallback
    if attr not in gdf.columns:
        avail = [c for c in gdf.columns if c != 'geometry']
        return _stub_result(
            "vec.filter_by_attribute",
            error=f"Attribute '{attribute_name}' not found. Available attributes: {avail}",
            vector_path=vector_path,
            available_attributes=avail,
        )

    allowed = {str(v).strip().lower() for v in allowed_values if str(v).strip()}

    def _match(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, list):
            tokens = [str(v).strip().lower() for v in value]
        else:
            tokens = [tok.strip().lower() for tok in str(value).split(",")]
        return any(tok in allowed for tok in tokens if tok)

    out = gdf[gdf[attr].apply(_match)].copy()
    out_path = _save_gdf(out, "filtered_by_attribute.geojson")
    return {
        "tool": "vec.filter_by_attribute",
        "vector_path": out_path,
        "attribute_name": attribute_name,
        "allowed_values": list(allowed_values),
        "total_input": int(len(gdf)),
        "total_filtered": int(len(out)),
    }


def vec_filter_by_distance(vector_path: str, center_lat: float, center_lon: float,
                           max_distance_m: float) -> str:
    """
    Filter vector features by distance from a center point.

    Parameters:
    - vector_path: Path to vector GeoJSON/Shapefile
    - center_lat: Center point latitude (WGS84)
    - center_lon: Center point longitude (WGS84)
    - max_distance_m: Maximum distance in meters

    Returns:
    - filtered vector with distance info
    """
    _require_geo_deps("vec.filter_by_distance")
    from math import radians, sin, cos, sqrt, atan2

    gdf = _read_vector_no_default_crs(vector_path)
    if gdf.empty:
        return {"tool": "vec.filter_by_distance", "vector_path": vector_path,
                "total_input": 0, "total_filtered": 0,
                "note": "Empty input; no features to filter by distance."}

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # Earth radius (m)
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    # Distance of every feature to the reference point
    distances = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            distances.append(float('inf'))
        else:
            if geom.geom_type == 'Point':
                lat, lon = geom.y, geom.x
            else:
                centroid = geom.centroid
                lat, lon = centroid.y, centroid.x
            dist = haversine(center_lat, center_lon, lat, lon)
            distances.append(dist)

    gdf['distance_m'] = distances

    # Filter
    out = gdf[gdf['distance_m'] <= float(max_distance_m)].copy()

    out_path = _save_gdf(out, "filtered_by_distance.geojson")

    return {
        "tool": "vec.filter_by_distance",
        "vector_path": out_path,
        "center_lat": float(center_lat),
        "center_lon": float(center_lon),
        "max_distance_m": float(max_distance_m),
        "total_input": len(gdf),
        "total_filtered": len(out),
        "nearest_distance_m": float(out['distance_m'].min()) if len(out) > 0 else None,
        "farthest_distance_m": float(out['distance_m'].max()) if len(out) > 0 else None
    }


@mcp.tool(name="vec.filter_by_distance", description="""
    Filter vector features by distance from a center point (WGS84).

    Parameters:
    - vector_path (str): Path to vector GeoJSON/Shapefile
    - center_lat (float): Center point latitude (WGS84)
    - center_lon (float): Center point longitude (WGS84)
    - max_distance_m (float): Maximum distance in meters

    Returns:
    - vector_path (str): Path to filtered vector
    - total_input (int): Number of input features
    - total_filtered (int): Number of features within distance
    - nearest_distance_m (float): Distance to nearest feature
    - farthest_distance_m (float): Distance to farthest feature
    """)
@_tool_guard("vec.filter_by_distance")
def vec_filter_by_distance_tool(vector_path: str, center_lat: float, center_lon: float,
                                 max_distance_m: float) -> str:
    result = vec_filter_by_distance(vector_path, center_lat, center_lon, max_distance_m)
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(name="vec.nearest", description="""
    For each feature in vector A, find the nearest feature(s) in vector B
    using Euclidean (straight-line) distance in pixel space.

    Parameters:
    - vector_a_path (str): Path to vector A GeoJSON
    - vector_b_path (str): Path to vector B GeoJSON
    - pixel_size_m (float, optional): meters per pixel
    - topk (int, optional): number of nearest neighbors per A (default 1)

    Returns:
    - table_path (str): CSV with nearest pairs and distances
    - links_geojson (str|None): GeoJSON of A–B nearest connectors
    - pair_count (int)
    - pixel_size_m (float|None)
    """)
@_tool_guard("vec.nearest")
def vec_polygon_nearest(vector_a_path: str,
                vector_b_path: str,
                pixel_size_m: float = None,
                topk: int = 1) -> str:
    _require_geo_deps("vec.nearest")
    import pandas as pd
    from shapely.ops import nearest_points

    gdf_a = _read_vector_no_default_crs(vector_a_path)
    gdf_b = _read_vector_no_default_crs(vector_b_path)

    if gdf_a.empty or gdf_b.empty:
        return {"tool": "vec.nearest", "pair_count": 0, "table_path": None,
                "links_geojson": None, "pixel_size_m": pixel_size_m,
                "vector_a_path": vector_a_path, "vector_b_path": vector_b_path,
                "note": "Empty input A or B; no nearest pairs."}

    if gdf_a.crs and gdf_b.crs and gdf_a.crs != gdf_b.crs:
        gdf_b = gdf_b.to_crs(gdf_a.crs)

    px = None
    if pixel_size_m is not None:
        px = float(pixel_size_m)
    else:
        for gdf in (gdf_a, gdf_b):
            if "pixel_size_m" in gdf.columns:
                s = gdf["pixel_size_m"].dropna()
                if len(s) > 0:
                    try:
                        px = float(s.iloc[0])
                        break
                    except Exception:
                        pass

    if "id" not in gdf_a.columns:
        gdf_a["id"] = list(range(1, len(gdf_a) + 1))
    if "id" not in gdf_b.columns:
        gdf_b["id"] = list(range(1, len(gdf_b) + 1))

    sindex_b = gdf_b.sindex

    rows = []
    link_geoms = []

    for _, a_row in gdf_a.iterrows():
        a_geom = a_row.geometry
        if a_geom is None or a_geom.is_empty:
            continue

        cand_idx = list(sindex_b.intersection(a_geom.bounds))
        cand = gdf_b.iloc[cand_idx] if cand_idx else gdf_b

        dists = []
        for _, b_row in cand.iterrows():
            b_geom = b_row.geometry
            if b_geom is None or b_geom.is_empty:
                continue
            d = float(a_geom.distance(b_geom))
            dists.append((d, b_row))

        if not dists:
            continue

        dists.sort(key=lambda x: x[0])
        for rank, (dist_px, b_row) in enumerate(dists[:topk], start=1):
            p_a, p_b = nearest_points(a_geom, b_row.geometry)

            dist_m = float(dist_px * px) if px is not None else None

            rows.append({
                "a_id": int(a_row["id"]),
                "b_id": int(b_row["id"]),
                "rank": rank,
                "distance_px": dist_px,
                "distance_m": dist_m,
                "pixel_size_m": px,
                "a_nearest_x": float(p_a.x),
                "a_nearest_y": float(p_a.y),
                "b_nearest_x": float(p_b.x),
                "b_nearest_y": float(p_b.y),
            })

            link_geoms.append(
                LineString([(p_a.x, p_a.y), (p_b.x, p_b.y)])
            )

    if not rows:
        return {"tool": "vec.nearest", "pair_count": 0, "table_path": None,
                "links_geojson": None, "pixel_size_m": px,
                "vector_a_path": vector_a_path, "vector_b_path": vector_b_path,
                "note": "No valid nearest pairs found."}

    df = pd.DataFrame(rows)
    _h = hashlib.md5(f"{vector_a_path}|{vector_b_path}".encode("utf-8")).hexdigest()[:6]
    table_path = TEMP_DIR / f"vec_nearest_table_{_h}.csv"
    df.to_csv(table_path, index=False)

    link_path = None
    if link_geoms:
        link_gdf = gpd.GeoDataFrame(df, geometry=link_geoms, crs=None)
        link_path = _save_gdf(link_gdf, "vec_nearest_links.geojson")

    return {
        "distance_px": dist_px,
        "distance_m": dist_m,
        "tool": "vec.nearest",
        "mode": "euclidean",
        "topk": topk,
        "pair_count": int(len(rows)),
        "pixel_size_m": px,
        "table_path": str(table_path),
        "links_geojson": link_path,
        "vector_a_path": vector_a_path,
        "vector_b_path": vector_b_path
    }


@mcp.tool(name="vec.clip", description="""
    Clip input vectors by ROI vectors.

    Parameters:
    - vector_path (str): Path to input vector GeoJSON (any geometry type)
    - roi_vector_path (str): Path to ROI vector GeoJSON (Polygon/MultiPolygon recommended)

    Returns:
    - clip_path (str): Path to clipped GeoJSON
    - input_count (int): number of input features
    - output_count (int): number of output features
    """)
@_tool_guard("vec.clip")
def vec_clip(vector_path: str, roi_vector_path: str) -> str:
    _require_geo_deps("vec.clip")
    gdf = _read_vector_no_default_crs(vector_path)
    roi = _read_vector_no_default_crs(roi_vector_path)

    if gdf.empty:
        return {"tool": "vec.clip", "clip_path": None, "input_count": 0,
                "output_count": 0, "input_vector": vector_path,
                "roi_vector": roi_vector_path, "note": "Empty input; 0 features to clip."}
    if roi.empty:
        return {"tool": "vec.clip", "clip_path": None, "input_count": int(len(gdf)),
                "output_count": 0, "input_vector": vector_path,
                "roi_vector": roi_vector_path, "note": "Empty ROI; no clip region."}

    if (gdf.crs is None) != (roi.crs is None):
        logger.warning(
            "vec.clip: CRS mismatch -- gdf.crs=%s, roi.crs=%s. "
            "Results may be empty due to coordinate space incompatibility.",
            gdf.crs,
            roi.crs,
        )

    if gdf.crs and roi.crs and gdf.crs != roi.crs:
        roi = roi.to_crs(gdf.crs)

    roi_geom = roi.geometry.unary_union
    if roi_geom is None or roi_geom.is_empty:
        return {"tool": "vec.clip", "clip_path": None, "input_count": int(len(gdf)),
                "output_count": 0, "input_vector": vector_path,
                "roi_vector": roi_vector_path, "note": "ROI geometry empty after union."}

    clipped = gpd.clip(gdf, roi_geom)
    clipped = clipped[~clipped.geometry.is_empty].copy()

    if "pixel_size_m" in gdf.columns and "pixel_size_m" not in clipped.columns:
        clipped["pixel_size_m"] = gdf["pixel_size_m"].iloc[0]

    out_name = _short_name(f"{Path(vector_path).stem}_CLIP_{Path(roi_vector_path).stem}.geojson")
    clip_path = _save_gdf(clipped, out_name)

    return {
        "tool": "vec.clip",
        "clip_path": clip_path,
        "input_count": int(len(gdf)),
        "output_count": int(len(clipped)),
        "input_vector": vector_path,
        "roi_vector": roi_vector_path
    }


# ============================================================
#  RASTER EXTENSIONS
# ============================================================

@mcp.tool(name="ras.hillshade", description='''
Description:
Compute a hillshade image from a Digital Elevation Model (DEM) raster,
revealing terrain relief. Uses the standard Horn (1981) formulation with
configurable sun azimuth and altitude angles. Useful for landslide and
debris-flow susceptibility mapping.

Parameters:
- dem_path (str): Path to DEM GeoTIFF (single-band elevation raster)
- azimuth (float, optional): Sun azimuth in degrees, clockwise from north. Default 315.
- altitude (float, optional): Sun altitude in degrees above horizon. Default 45.
- z_factor (float, optional): Vertical exaggeration factor. Default 1.0.
- out_path (str, optional): Output GeoTIFF path. Leave empty/omit this parameter to let the tool choose a managed location automatically. Only pass a value when the user explicitly provides an absolute path.

Returns:
- hillshade_path (str): Path to the generated hillshade image
- shape (list[int]): [height, width] of the output
- stats (dict): min, max, mean of hillshade values (0-255)
''')
def ras_hillshade(
    dem_path: str,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    out_path: Optional[str] = None,
) -> str:
    try:
        try:
            import rasterio
            with rasterio.open(dem_path) as src:
                dem = src.read(1).astype(np.float32)
                profile = src.profile.copy()
                xres = abs(src.transform.a)
                yres = abs(src.transform.e)
        except Exception:
            from PIL import Image
            dem = np.asarray(Image.open(dem_path), dtype=np.float32)
            profile = None
            xres = yres = 1.0

        dem = dem * z_factor
        dy, dx = np.gradient(dem, yres, xres)
        slope = np.arctan(np.hypot(dx, dy))
        aspect = np.arctan2(-dx, dy)

        az_rad = np.deg2rad(360.0 - azimuth + 90.0)
        alt_rad = np.deg2rad(altitude)

        hillshade = (
            np.sin(alt_rad) * np.cos(slope)
            + np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect)
        )
        hillshade = np.clip(hillshade * 255.0, 0, 255).astype(np.uint8)

        out_path = out_path or str(TEMP_DIR / "hillshade.tif")
        try:
            import rasterio
            if profile is not None:
                profile.update(dtype="uint8", count=1)
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(hillshade, 1)
            else:
                raise RuntimeError("no profile")
        except Exception:
            from PIL import Image
            Image.fromarray(hillshade).save(out_path)

        return json.dumps({
            "tool": "ras.hillshade",
            "hillshade_path": out_path,
            "shape": [int(hillshade.shape[0]), int(hillshade.shape[1])],
            "stats": {
                "min": int(hillshade.min()),
                "max": int(hillshade.max()),
                "mean": float(hillshade.mean()),
            },
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("ras.hillshade", error=str(e), dem_path=dem_path))


@mcp.tool(name="ras.slope", description='''
Description:
Compute a slope map in degrees from a DEM raster. Slope is derived via
the Horn (1981) gradient formulation and converted from radians to degrees.
Commonly used as input for landslide susceptibility and terrain stability.

Parameters:
- dem_path (str): Path to DEM GeoTIFF
- units (str, optional): "degrees" (default) or "percent"
- out_path (str, optional): Output GeoTIFF path. Leave empty/omit this parameter to let the tool choose a managed location automatically. Only pass a value when the user explicitly provides an absolute path.

Returns:
- slope_path (str): Path to the generated slope raster
- units (str): Units of the output values
- stats (dict): min, max, mean slope
''')
def ras_slope(
    dem_path: str,
    units: str = "degrees",
    out_path: Optional[str] = None,
) -> str:
    try:
        try:
            import rasterio
            with rasterio.open(dem_path) as src:
                dem = src.read(1).astype(np.float32)
                profile = src.profile.copy()
                xres = abs(src.transform.a)
                yres = abs(src.transform.e)
        except Exception:
            from PIL import Image
            dem = np.asarray(Image.open(dem_path), dtype=np.float32)
            profile = None
            xres = yres = 1.0

        dy, dx = np.gradient(dem, yres, xres)
        slope_rad = np.arctan(np.hypot(dx, dy))
        if units == "percent":
            slope = np.tan(slope_rad) * 100.0
        else:
            slope = np.rad2deg(slope_rad)
        slope = slope.astype(np.float32)

        out_path = out_path or str(TEMP_DIR / "slope.tif")
        try:
            import rasterio
            if profile is not None:
                profile.update(dtype="float32", count=1)
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(slope, 1)
            else:
                raise RuntimeError("no profile")
        except Exception:
            np.save(out_path + ".npy", slope)

        return json.dumps({
            "tool": "ras.slope",
            "slope_path": out_path,
            "units": units,
            "stats": {
                "min": float(slope.min()),
                "max": float(slope.max()),
                "mean": float(slope.mean()),
            },
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("ras.slope", error=str(e), dem_path=dem_path))


@mcp.tool(name="ras.reproject", description='''
Description:
Reproject a raster from its current CRS to a target CRS. Useful for aligning
multi-source data (e.g., optical + SAR, DEM + damage mask) so downstream
operations can be performed in a common coordinate system.

Parameters:
- raster_path (str): Path to the input raster
- dst_crs (str): Target CRS as an EPSG code (e.g., "EPSG:32636")
- resampling (str, optional): "nearest" | "bilinear" | "cubic". Default "bilinear".
- out_path (str, optional): Output GeoTIFF path. Leave empty/omit this parameter to let the tool choose a managed location automatically. Only pass a value when the user explicitly provides an absolute path.

Returns:
- output_path (str): Path to the reprojected raster
- src_crs (str): Original CRS
- dst_crs (str): New CRS
- shape (list[int]): [height, width]
''')
def ras_reproject(
    raster_path: str,
    dst_crs: str,
    resampling: str = "bilinear",
    out_path: Optional[str] = None,
) -> str:
    try:
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject, Resampling

        resampling_map = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
        }
        resample_method = resampling_map.get(resampling, Resampling.bilinear)

        with rasterio.open(raster_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
            })
            src_crs = str(src.crs)

            out_path = out_path or str(TEMP_DIR / "reprojected.tif")
            with rasterio.open(out_path, "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=resample_method,
                    )

        return json.dumps({
            "tool": "ras.reproject",
            "output_path": out_path,
            "src_crs": src_crs,
            "dst_crs": dst_crs,
            "shape": [int(height), int(width)],
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result(
            "ras.reproject", error=str(e),
            raster_path=raster_path, dst_crs=dst_crs
        ))


@mcp.tool(name="ras.resample", description='''
Description:
Resample a raster to a new pixel size. Useful for harmonizing resolution
between multi-source data (e.g., resampling a coarser grid to a finer target resolution).

Parameters:
- raster_path (str): Path to input raster
- target_pixel_size_m (float): Desired pixel size in meters
- resampling (str, optional): "nearest" | "bilinear" | "cubic". Default "bilinear".
- out_path (str, optional): Output GeoTIFF path. Leave empty/omit this parameter to let the tool choose a managed location automatically. Only pass a value when the user explicitly provides an absolute path.

Returns:
- output_path (str): Path to the resampled raster
- src_pixel_size_m (float): Original pixel size
- dst_pixel_size_m (float): New pixel size
- shape (list[int]): [height, width]
''')
def ras_resample(
    raster_path: str,
    target_pixel_size_m: float,
    resampling: str = "bilinear",
    out_path: Optional[str] = None,
) -> str:
    try:
        import rasterio
        from rasterio.enums import Resampling

        resampling_map = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
        }
        resample_method = resampling_map.get(resampling, Resampling.bilinear)

        if not target_pixel_size_m or float(target_pixel_size_m) <= 0:
            raise ValueError(f"ras.resample: target_pixel_size_m must be > 0, got {target_pixel_size_m}")

        with rasterio.open(raster_path) as src:
            src_res = abs(src.transform.a)
            if src_res <= 0:
                src_res = 1.0  # fallback: treat as pixel units
            scale = src_res / float(target_pixel_size_m)
            new_width = max(1, int(src.width * scale))
            new_height = max(1, int(src.height * scale))

            data = src.read(
                out_shape=(src.count, new_height, new_width),
                resampling=resample_method,
            )
            new_transform = src.transform * src.transform.scale(
                src.width / new_width, src.height / new_height
            )
            profile = src.profile.copy()
            profile.update({
                "height": new_height,
                "width": new_width,
                "transform": new_transform,
            })

            out_path = out_path or str(TEMP_DIR / "resampled.tif")
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data)

        return json.dumps({
            "tool": "ras.resample",
            "output_path": out_path,
            "src_pixel_size_m": float(src_res),
            "dst_pixel_size_m": float(target_pixel_size_m),
            "shape": [int(new_height), int(new_width)],
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result(
            "ras.resample", error=str(e),
            raster_path=raster_path, target_pixel_size_m=target_pixel_size_m
        ))


@mcp.tool(name="ras.focal_stats", description='''
Description:
Compute focal (moving-window) statistics over a raster. For each pixel,
aggregate the values inside a square window of side `window_size` centred
on the pixel (odd window sizes recommended). Typical uses: smooth a damage
density map, detect local hotspots, extract local roughness from a DEM,
or pre-process a change map before thresholding. This is the local
counterpart of global zonal statistics and is orthogonal to
`vec.raster_stats` (which aggregates by vector zones).

Parameters:
- raster_path (str): Path to the input raster (band 1 is used).
- statistic (str, optional): One of "mean", "sum", "min", "max", "std".
  Default "mean".
- window_size (int, optional): Odd integer window side in pixels.
  Default 5 (i.e. a 5x5 window).
- nodata (float, optional): Value to treat as nodata. Defaults to the
  raster's own nodata if present; otherwise pixels are all valid.

Returns (JSON string with fields):
- tool: "ras.focal_stats"
- output_path (str): Path to the resulting focal-stat raster.
- statistic (str): Which statistic was applied.
- window_size (int): Window side used.
- shape (list[int]): Output raster shape (rows, cols).
''')
def ras_focal_stats(
    raster_path: str,
    statistic: str = "mean",
    window_size: int = 5,
    nodata: Optional[float] = None,
    out_path: Optional[str] = None,
) -> str:
    try:
        import numpy as np
        import rasterio
        from scipy import ndimage

        statistic = (statistic or "mean").lower()
        valid_stats = {"mean", "sum", "min", "max", "std"}
        if statistic not in valid_stats:
            raise ValueError(
                f"ras.focal_stats: unsupported statistic={statistic}; "
                f"expected one of {sorted(valid_stats)}"
            )
        if window_size < 1:
            raise ValueError("ras.focal_stats: window_size must be >= 1")
        # Force odd window size for symmetric neighbourhoods.
        if window_size % 2 == 0:
            window_size += 1

        with rasterio.open(raster_path) as src:
            arr = src.read(1).astype("float32")
            profile = src.profile.copy()
            src_nodata = src.nodata

        nd = nodata if nodata is not None else src_nodata
        if nd is not None:
            valid = arr != nd
        else:
            valid = np.ones_like(arr, dtype=bool)

        work = np.where(valid, arr, 0.0)

        # Compute the statistic using scipy.ndimage.
        if statistic == "mean":
            num = ndimage.uniform_filter(work, size=window_size, mode="nearest") \
                * (window_size * window_size)
            cnt = ndimage.uniform_filter(valid.astype("float32"), size=window_size,
                                         mode="nearest") * (window_size * window_size)
            out = np.where(cnt > 0, num / np.maximum(cnt, 1e-6), 0.0)
        elif statistic == "sum":
            out = ndimage.uniform_filter(work, size=window_size, mode="nearest") \
                * (window_size * window_size)
        elif statistic == "min":
            out = ndimage.minimum_filter(work, size=window_size, mode="nearest")
        elif statistic == "max":
            out = ndimage.maximum_filter(work, size=window_size, mode="nearest")
        elif statistic == "std":
            mean = ndimage.uniform_filter(work, size=window_size, mode="nearest")
            mean_sq = ndimage.uniform_filter(work * work, size=window_size,
                                             mode="nearest")
            out = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))

        out = out.astype("float32")
        if nd is not None:
            out = np.where(valid, out, nd)

        stem = Path(raster_path).stem
        out_path = out_path or str(
            TEMP_DIR / f"{stem}_focal_{statistic}_{window_size}.tif"
        )

        profile.update(dtype="float32", count=1)
        if nd is not None:
            profile["nodata"] = nd

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out, 1)

        return json.dumps({
            "tool": "ras.focal_stats",
            "output_path": out_path,
            "statistic": statistic,
            "window_size": window_size,
            "shape": list(out.shape),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result(
            "ras.focal_stats", error=str(e),
            raster_path=raster_path,
            statistic=statistic, window_size=window_size
        ))


# ============================================================
#  VECTOR EXTENSIONS
# ============================================================

@mcp.tool(name="vec.convex_hull", description='''
Description:
Compute the convex hull of a set of polygons or points in a GeoJSON file.
Useful for estimating the spatial envelope of a damage cluster or hazard
spread footprint.

Parameters:
- vector_path (str): Path to input GeoJSON
- out_path (str, optional): Output GeoJSON path. Leave empty/omit this parameter to let the tool choose a managed location automatically. Only pass a value when the user explicitly provides an absolute path.

Returns:
- output_path (str): Path to convex hull GeoJSON
- area_m2 (float): Area of the hull (if CRS is projected)
- vertex_count (int): Number of vertices in the hull
''')
def vec_convex_hull(
    vector_path: str,
    out_path: Optional[str] = None,
) -> str:
    try:
        import geopandas as gpd
        from shapely.ops import unary_union

        gdf = gpd.read_file(vector_path)
        if gdf.empty:
            raise ValueError("vec.convex_hull: empty input")

        merged = unary_union(list(gdf.geometry))
        hull = merged.convex_hull

        out_path = out_path or str(TEMP_DIR / "convex_hull.geojson")
        out_gdf = gpd.GeoDataFrame(geometry=[hull], crs=gdf.crs)
        out_gdf.to_file(out_path, driver="GeoJSON")

        area = float(hull.area) if gdf.crs and gdf.crs.is_projected else 0.0
        vertex_count = len(list(hull.exterior.coords)) if hasattr(hull, "exterior") else 0

        return json.dumps({
            "tool": "vec.convex_hull",
            "output_path": out_path,
            "area_m2": area,
            "vertex_count": int(vertex_count),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("vec.convex_hull", error=str(e), vector_path=vector_path))


@mcp.tool(name="vec.voronoi", description='''
Description:
Compute a Voronoi diagram (service-area partition) from a set of input points.
Each resulting polygon contains all locations closer to its seed point than to
any other. Useful for assigning hospital/firestation service areas.

Parameters:
- points_path (str): Path to input points GeoJSON
- boundary_path (str, optional): Optional boundary polygon to clip the diagram
- out_path (str, optional): Output GeoJSON path. Leave empty/omit this parameter to let the tool choose a managed location automatically. Only pass a value when the user explicitly provides an absolute path.

Returns:
- output_path (str): Path to Voronoi polygons GeoJSON
- cell_count (int): Number of Voronoi cells generated
''')
def vec_voronoi(
    points_path: str,
    boundary_path: Optional[str] = None,
    out_path: Optional[str] = None,
) -> str:
    try:
        import geopandas as gpd
        from shapely.geometry import MultiPoint
        from shapely.ops import voronoi_diagram

        gdf = gpd.read_file(points_path)
        if gdf.empty:
            raise ValueError("vec.voronoi: empty input")
        points = MultiPoint([(geom.x, geom.y) for geom in gdf.geometry if geom.geom_type == "Point"])
        if len(points.geoms) < 2:
            raise ValueError("vec.voronoi: need at least 2 points")

        envelope = points.envelope.buffer(max(points.envelope.length, 1.0))
        vor = voronoi_diagram(points, envelope=envelope)

        if boundary_path:
            boundary = gpd.read_file(boundary_path).unary_union
            cells = [g.intersection(boundary) for g in vor.geoms if not g.is_empty]
            cells = [c for c in cells if not c.is_empty]
        else:
            cells = [g for g in vor.geoms if not g.is_empty]

        out_gdf = gpd.GeoDataFrame(
            {"cell_id": list(range(len(cells)))}, geometry=cells, crs=gdf.crs
        )
        out_path = out_path or str(TEMP_DIR / "voronoi.geojson")
        out_gdf.to_file(out_path, driver="GeoJSON")

        return json.dumps({
            "tool": "vec.voronoi",
            "output_path": out_path,
            "cell_count": int(len(cells)),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("vec.voronoi", error=str(e), points_path=points_path))


@mcp.tool(name="vec.dissolve", description='''
Description:
Dissolve (merge) overlapping or adjacent polygons into larger ones, optionally
grouping by an attribute field. Useful for consolidating fragmented damage
patches into contiguous impact zones.

Parameters:
- vector_path (str): Path to input polygon GeoJSON
- by (str, optional): Attribute field to group by before dissolving
- out_path (str, optional): Output GeoJSON path. Leave empty/omit this parameter to let the tool choose a managed location automatically. Only pass a value when the user explicitly provides an absolute path.

Returns:
- output_path (str): Path to dissolved GeoJSON
- input_count (int): Number of input features
- output_count (int): Number of features after dissolve
''')
def vec_dissolve(
    vector_path: str,
    by: Optional[str] = None,
    out_path: Optional[str] = None,
) -> str:
    try:
        import geopandas as gpd

        gdf = gpd.read_file(vector_path)
        input_count = len(gdf)
        if by and by in gdf.columns:
            out_gdf = gdf.dissolve(by=by).reset_index()
        else:
            out_gdf = gdf.dissolve()

        out_path = out_path or str(TEMP_DIR / "dissolved.geojson")
        out_gdf.to_file(out_path, driver="GeoJSON")

        return json.dumps({
            "tool": "vec.dissolve",
            "output_path": out_path,
            "input_count": int(input_count),
            "output_count": int(len(out_gdf)),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("vec.dissolve", error=str(e), vector_path=vector_path))


@mcp.tool(name="vec.simplify", description='''
Description:
Simplify polygon or line geometries using the Douglas-Peucker algorithm.
Reduces the number of vertices while preserving overall shape, producing
lighter outputs suitable for visualization or downstream computation.

Parameters:
- vector_path (str): Path to input GeoJSON
- tolerance (float): Simplification tolerance in CRS units (meters for projected)
- preserve_topology (bool, optional): Default True
- out_path (str, optional): Output GeoJSON path. Leave empty/omit this parameter to let the tool choose a managed location automatically. Only pass a value when the user explicitly provides an absolute path.

Returns:
- output_path (str): Path to simplified GeoJSON
- input_vertex_count (int): Total vertices in input
- output_vertex_count (int): Total vertices in output
''')
def vec_simplify(
    vector_path: str,
    tolerance: float,
    preserve_topology: bool = True,
    out_path: Optional[str] = None,
) -> str:
    try:
        import geopandas as gpd

        def _count_vertices(geom) -> int:
            if geom is None or geom.is_empty:
                return 0
            if geom.geom_type == "Polygon":
                return len(list(geom.exterior.coords)) + sum(
                    len(list(r.coords)) for r in geom.interiors
                )
            if geom.geom_type == "LineString":
                return len(list(geom.coords))
            if geom.geom_type in ("MultiPolygon", "MultiLineString", "GeometryCollection"):
                return sum(_count_vertices(g) for g in geom.geoms)
            return 0

        gdf = gpd.read_file(vector_path)
        input_vertices = sum(_count_vertices(g) for g in gdf.geometry)
        gdf["geometry"] = gdf.geometry.simplify(
            tolerance=tolerance, preserve_topology=preserve_topology
        )
        output_vertices = sum(_count_vertices(g) for g in gdf.geometry)

        out_path = out_path or str(TEMP_DIR / "simplified.geojson")
        gdf.to_file(out_path, driver="GeoJSON")

        return json.dumps({
            "tool": "vec.simplify",
            "output_path": out_path,
            "input_vertex_count": int(input_vertices),
            "output_vertex_count": int(output_vertices),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result(
            "vec.simplify", error=str(e),
            vector_path=vector_path, tolerance=tolerance
        ))


@mcp.tool(name="vec.point_in_polygon", description='''
Description:
For each input point, test whether it lies inside any polygon of a polygon
layer. Returns the matching polygon id(s) if any. Useful for filtering POIs
to those located within damaged zones.

Parameters:
- points_path (str): Path to point GeoJSON
- polygons_path (str): Path to polygon GeoJSON
- polygon_id_field (str, optional): Attribute field used as polygon identifier

Returns:
- matches (list[dict]): List of {point_index, polygon_ids} for inside points
- inside_count (int): Number of points located inside at least one polygon
''')
def vec_point_in_polygon(
    points_path: str,
    polygons_path: str,
    polygon_id_field: Optional[str] = None,
) -> str:
    try:
        import geopandas as gpd

        pts = gpd.read_file(points_path)
        polys = gpd.read_file(polygons_path)
        if pts.crs != polys.crs and polys.crs is not None:
            pts = pts.to_crs(polys.crs)

        matches: List[Dict[str, Any]] = []
        for pidx, pt in pts.iterrows():
            if pt.geometry is None:
                continue
            hits = polys[polys.contains(pt.geometry)]
            if len(hits) > 0:
                if polygon_id_field and polygon_id_field in hits.columns:
                    ids = hits[polygon_id_field].tolist()
                else:
                    ids = hits.index.tolist()
                matches.append({"point_index": int(pidx), "polygon_ids": ids})

        return json.dumps({
            "tool": "vec.point_in_polygon",
            "matches": matches,
            "inside_count": len(matches),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result(
            "vec.point_in_polygon", error=str(e),
            points_path=points_path, polygons_path=polygons_path
        ))


@mcp.tool(name="vec.repair_min_edges", description='''
Description:
Given a road graph with some edges marked as damaged/blocked, find the minimum
set of edges that must be repaired so that a shortest path between source and
target nodes becomes possible. The solution minimises the number of repairs
first and breaks ties by total path length.

The graph may be provided as a NetworkX .graphml file (output of vec.build_graph)
or as a JSON file with {"nodes": [...], "edges": [...]}. If source/target are not
given or do not exist in the graph, the function automatically selects two nodes
from the largest connected component: a node near the centroid of the component
as source and its farthest reachable node as target.

Parameters:
- graph_path (str): Path to a graph file (.graphml or .json/.geojson)
- source (str | int, optional): Source node id (e.g., a rescue base). If omitted
  or not found, an auto-selection is performed.
- target (str | int, optional): Target node id (e.g., a hospital). Same rule.
- blocked_attr (str, optional): Edge attribute marking a damaged/blocked edge.
  Defaults to "damaged" (native attribute from vec.build_graph); "blocked" is
  also accepted.

Returns:
- repair_edges (list): Edge endpoints that must be repaired, e.g. [[u, v], ...]
- repair_count (int): Minimum number of repairs needed on the optimal path
- path_length_m (float): Total length of the post-repair path (metres)
- source (int): Source node id actually used
- target (int): Target node id actually used
- auto_selected (bool): Whether source/target were auto-selected
''')
def vec_repair_min_edges(
    graph_path: str,
    source: Optional[Union[str, int]] = None,
    target: Optional[Union[str, int]] = None,
    blocked_attr: str = "damaged",
) -> str:
    try:
        import json as _json
        import networkx as nx
        from pathlib import Path as _Path

        # ---- 1. Load graph (.graphml or .json) ----
        G = nx.Graph()
        suffix = _Path(graph_path).suffix.lower()
        if suffix == ".graphml":
            H = nx.read_graphml(graph_path)
            for nid, data in H.nodes(data=True):
                try:
                    new_id = int(nid)
                except Exception:
                    new_id = nid
                G.add_node(new_id, **dict(data))
            for u, v, data in H.edges(data=True):
                try:
                    uu, vv = int(u), int(v)
                except Exception:
                    uu, vv = u, v
                attr = dict(data)
                # Normalise common attribute name variants
                length_val = attr.get("length_m", attr.get("length", attr.get("weight", 1.0)))
                try:
                    length = float(length_val)
                except Exception:
                    length = 1.0
                # Accept either `damaged` (vec.build_graph output) or caller override
                dmg_val = attr.get(blocked_attr, attr.get("damaged", False))
                if isinstance(dmg_val, str):
                    blocked = dmg_val.strip().lower() in ("true", "1", "yes")
                else:
                    blocked = bool(dmg_val)
                G.add_edge(uu, vv, length_m=length, blocked=blocked)
        else:
            with open(graph_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            for node in data.get("nodes", []):
                G.add_node(node["id"])
            for edge in data.get("edges", []):
                u, v = edge["source"], edge["target"]
                length = float(edge.get("length_m", edge.get("length", edge.get("weight", 1.0))))
                raw = edge.get(blocked_attr, edge.get("damaged", False))
                if isinstance(raw, str):
                    blocked = raw.strip().lower() in ("true", "1", "yes")
                else:
                    blocked = bool(raw)
                G.add_edge(u, v, length_m=length, blocked=blocked)

        if G.number_of_nodes() == 0:
            raise ValueError("vec.repair_min_edges: graph has no nodes")

        # ---- 2. Determine source/target ----
        nodes_set = set(G.nodes)
        auto_selected = False

        def _coerce(x):
            try:
                return int(x) if x is not None else None
            except Exception:
                return x

        src = _coerce(source)
        tgt = _coerce(target)

        if src is None or src not in nodes_set or tgt is None or tgt not in nodes_set:
            # Auto-select from the largest connected component.
            largest = max(nx.connected_components(G), key=len)
            sub = G.subgraph(largest).copy()
            if sub.number_of_nodes() < 2:
                raise ValueError("vec.repair_min_edges: largest component has <2 nodes")
            # Use node with smallest id as source, and compute the farthest node (in terms
            # of pure hop count) as target. This guarantees they are distinct and reachable.
            comp_nodes = sorted(sub.nodes)
            src = comp_nodes[0]
            # Farthest by unweighted BFS for determinism.
            lengths = nx.single_source_shortest_path_length(sub, src)
            tgt = max(lengths, key=lengths.get)
            auto_selected = True

        # ---- 3. Shortest path with very large penalty for blocked edges ----
        BLOCK_PENALTY = 1e12

        def weight_fn(u, v, d):
            return d.get("length_m", 1.0) + (BLOCK_PENALTY if d.get("blocked", False) else 0.0)

        try:
            path = nx.shortest_path(G, source=src, target=tgt, weight=weight_fn)
        except nx.NetworkXNoPath:
            raise ValueError(
                f"vec.repair_min_edges: no path exists between {src} and {tgt} "
                "even with repairs allowed (disconnected components)."
            )

        repair_edges: List[List[Any]] = []
        path_length = 0.0
        for u, v in zip(path[:-1], path[1:]):
            d = G[u][v]
            path_length += float(d.get("length_m", 1.0))
            if d.get("blocked", False):
                repair_edges.append([u, v])

        return json.dumps({
            "tool": "vec.repair_min_edges",
            "repair_edges": repair_edges,
            "repair_count": int(len(repair_edges)),
            "path_length_m": float(path_length),
            "path_node_count": int(len(path)),
            "source": src if isinstance(src, (int, float, str)) else str(src),
            "target": tgt if isinstance(tgt, (int, float, str)) else str(tgt),
            "auto_selected": bool(auto_selected),
            "blocked_attr": blocked_attr,
            "graph_nodes": int(G.number_of_nodes()),
            "graph_edges": int(G.number_of_edges()),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result(
            "vec.repair_min_edges", error=str(e),
            graph_path=graph_path, source=source, target=target,
        ))


# ============== MCP entry point ==============

if __name__ == "__main__":
    mcp.run(show_banner=False)
