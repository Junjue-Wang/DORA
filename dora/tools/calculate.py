"""
Calculate MCP server: geometry (area, volume, distance), arithmetic and logistics (logi.*) tools.
"""
import json
import math
import re
from typing import Dict, Any, List, Optional

from ._common import session_temp_dir
from ._types import ListFloat, ListDict, DictStrFloat, ListOrDictStrAny, OptListInt

import numpy as np
from PIL import Image
from fastmcp import FastMCP
from skimage.io import imread

mcp = FastMCP()
TEMP_DIR = session_temp_dir()




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


def _nested_get(value: Any, field_path: Optional[str]) -> Any:
    if field_path in (None, ""):
        return value

    current = value
    for part in str(field_path).split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _normalize_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _render_label_template(template: str, item: Dict[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(1)
        value = _nested_get(item, token)
        if value is None:
            return ""
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return str(value)

    return re.sub(r"\{([\w.]+)\}", repl, template)


# ============== Geometry tools ==============

@mcp.tool(name="ras.area", description='''
Description:
Calculate the area of regions in a mask.
Only supports pixel-based masks.

Parameters:
- mask_path (str): Path to the mask image file
- pixel_size_m (float): Size of each pixel in meters
- classes (list[int], optional): List of class values to include. If not provided, counts all non-zero pixels.

Returns:
- area_m2 (float): Total area in square meters
- pixel_count (int): Number of pixels counted
''')
def ras_area(mask_path: str, pixel_size_m: float, classes: OptListInt = None) -> str:
    """
    Total area of the given classes in a mask.
    """
    try:
        try:
            mask = imread(mask_path)
        except Exception:
            mask = np.array(Image.open(mask_path))

        if classes:
            target = np.isin(mask, classes)
        else:
            target = mask > 0
        
        pixel_count = int(np.sum(target))
        area_m2 = pixel_count * (pixel_size_m ** 2)
        
        result = {
            "tool": "ras.area",
            "area_m2": float(area_m2),
            "pixel_count": pixel_count,
            "pixel_size_m": pixel_size_m
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("ras.area", error=str(e), mask_path=mask_path, pixel_size_m=pixel_size_m))



@mcp.tool(name="ras.grid_windows", description='''
Description:
Split a raster image into a uniform n×n grid and return pixel windows.
Indexing uses image[x1:x2, y1:y2] where x is row (top→bottom), y is column (left→right).
If H or W is not divisible by n, the last row/col expands to the image boundary.

Parameters:
- image_path (str): Path to raster (GeoTIFF/COG/PNG/JPG…)
- n (int): number of splits per axis (n >= 1)

Returns:
- windows (list[list[int]]): [[x1,x2,y1,y2], ...] in row-major order
- image_shape (list[int]): [height, width]
- base_tile_px (list[int]): [floor(H/n), floor(W/n)]
- count (int): n*n
- note (str)
''')
def ras_grid_windows(image_path: str, n: int) -> str:
    try:
        if n is None or int(n) < 1:
            raise ValueError("ras.grid_windows: n must be >= 1.")
        n = int(n)

        # Read size via rasterio if available; fallback to PIL
        H = W = None
        try:
            import rasterio
            with rasterio.open(image_path) as src:
                H, W = src.height, src.width
        except Exception:
            from PIL import Image
            import numpy as np
            img = Image.open(image_path)
            arr = np.asarray(img)
            H, W = int(arr.shape[0]), int(arr.shape[1])

        if H < n or W < n:
            raise ValueError(f"ras.grid_windows: n={n} too large for image {H}x{W}.")

        # Base tile size (last row/col expands to cover remainder)
        bh = H // n
        bw = W // n
        if bh == 0 or bw == 0:
            raise ValueError("ras.grid_windows: tile size computed as 0; decrease n.")

        windows: list[list[int]] = []
        for i in range(n):
            x1 = i * bh
            x2 = (i + 1) * bh if i < n - 1 else H
            for j in range(n):
                y1 = j * bw
                y2 = (j + 1) * bw if j < n - 1 else W
                windows.append([int(x1), int(x2), int(y1), int(y2)])

        return json.dumps({
            "tool": "ras.grid_windows",
            "windows": windows,
            "image_shape": [int(H), int(W)],
            "base_tile_px": [int(bh), int(bw)],
            "count": int(len(windows)),
            "note": "Row-major; last row/col expanded to image boundary."
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps(_stub_result("ras.grid_windows", error=str(e),
                                       image_path=image_path, n=n if isinstance(n, int) else n))


# ============== Volume and logistics tools ==============

@mcp.tool(name="calc.volume", description='''
Description:
Calculate volume from area and depth.
Volume = Area × Depth

Parameters:
- area_m2 (float): Area in square meters
- depth_m (float): Depth/height in meters

Returns:
- volume_m3 (float): Volume in cubic meters
''')
def calc_volume(area_m2: float, depth_m: float) -> str:
    """
    Volume from area and depth.
    """
    try:
        volume_m3 = float(area_m2) * depth_m
        
        result = {
            "tool": "calc.volume",
            "volume_m3": float(volume_m3),
            "area_m2": float(area_m2),
            "depth_m": float(depth_m)
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("calc.volume", error=str(e), 
                                       area_m2=area_m2, depth_m=depth_m))


@mcp.tool(name="calc.add", description='''
Description:
Add two numbers.

Parameters:
- a (float): first value
- b (float): second value

Returns:
- sum (float): a + b
''')
def calc_add(a: float, b: float) -> str:
    try:
        result = {
            "tool": "calc.add",
            "sum": float(a) + float(b),
            "a": float(a),
            "b": float(b)
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("calc.add", error=str(e), a=a, b=b))


@mcp.tool(name="calc.mul", description='''
Description:
Multiply two numbers.

Parameters:
- a (float): first value
- b (float): second value

Returns:
- product (float): a * b
''')
def calc_mul(a: float, b: float) -> str:
    try:
        result = {
            "tool": "calc.mul",
            "product": float(a) * float(b),
            "a": float(a),
            "b": float(b)
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("calc.mul", error=str(e), a=a, b=b))


@mcp.tool(name="calc.point_distance", description='''
Description:
Compute straight-line distance between two pixel coordinates.

Parameters:
- point_a (list[float]): [row, col]
- point_b (list[float]): [row, col]
- pixel_size_m (float): pixel size in meters

Returns:
- distance_px (float): Euclidean distance in pixels
- distance_m (float): Euclidean distance in meters
''')
def calc_point_distance(point_a: ListFloat, point_b: ListFloat, pixel_size_m: float = 1.0) -> str:
    try:
        if not isinstance(point_a, (list, tuple)) or not isinstance(point_b, (list, tuple)):
            raise ValueError("calc.point_distance: point_a and point_b must be [row, col].")
        if len(point_a) != 2 or len(point_b) != 2:
            raise ValueError("calc.point_distance: point_a and point_b must have length 2.")

        ax, ay = float(point_a[0]), float(point_a[1])
        bx, by = float(point_b[0]), float(point_b[1])
        distance_px = math.hypot(ax - bx, ay - by)
        distance_m = distance_px * float(pixel_size_m)

        return json.dumps({
            "tool": "calc.point_distance",
            "point_a": [ax, ay],
            "point_b": [bx, by],
            "pixel_size_m": float(pixel_size_m),
            "distance_px": float(distance_px),
            "distance_m": float(distance_m),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result(
            "calc.point_distance",
            error=str(e),
            point_a=point_a,
            point_b=point_b,
            pixel_size_m=pixel_size_m,
        ))


@mcp.tool(name="calc.sub", description='''
Description:
Subtract two numbers.

Parameters:
- a (float): minuend
- b (float): subtrahend

Returns:
- difference (float): a - b
''')
def calc_sub(a: float, b: float) -> str:
    try:
        result = {
            "tool": "calc.sub",
            "difference": float(a) - float(b),
            "a": float(a),
            "b": float(b)
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("calc.sub", error=str(e), a=a, b=b))


@mcp.tool(name="calc.diff", description='''
Description:
Compute absolute difference between two numbers.

Parameters:
- a (float): first value
- b (float): second value

Returns:
- diff (float): |a - b|
''')
def calc_diff(a: float, b: float) -> str:
    try:
        diff = abs(float(a) - float(b))
        result = {
            "tool": "calc.diff",
            "diff": float(diff),
            "a": float(a),
            "b": float(b)
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("calc.diff", error=str(e), a=a, b=b))


@mcp.tool(name="calc.div", description='''
Description:
Divide two numbers.

Parameters:
- a (float): numerator
- b (float): denominator

Returns:
- ratio (float): a / b
''')
def calc_div(a: float, b: float) -> str:
    try:
        denom = float(b)
        ratio = float(a) / denom if denom != 0 else 0.0
        result = {
            "tool": "calc.div",
            "ratio": float(ratio),
            "a": float(a),
            "b": float(b)
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("calc.div", error=str(e), a=a, b=b))


@mcp.tool(name="calc.greater_than", description='''
Description:
Compare two numbers and return whether a > b.

Parameters:
- a (float): left-hand value
- b (float): right-hand value

Returns:
- value (int): 1 if a > b else 0
- result (bool): True if a > b else False
''')
def calc_greater_than(a: float, b: float) -> str:
    try:
        result = float(a) > float(b)
        payload = {
            "tool": "calc.greater_than",
            "value": int(result),
            "result": bool(result),
            "a": float(a),
            "b": float(b)
        }
        return json.dumps(payload, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("calc.greater_than", error=str(e), a=a, b=b))


@mcp.tool(name="tool.chart_data", description='''
Description:
Build chart-ready key/value data from a list of records or a dict of records.
Useful for converting loop outputs or feature statistics into dynamic inputs for vis.bar_chart / vis.line_chart.

Parameters:
- values (list[dict] | dict): source records
- value_field (str): nested field path containing the numeric value
- label_field (str, optional): nested field path for the primary label
- fallback_label_field (str, optional): fallback nested field path when label_field is empty
- fallback_label_prefix (str, optional): prefix for fallback label
- label_template (str, optional): template such as "r{item.row_idx}c{item.col_idx}"
- sort_by (str, optional): nested field path to sort by; if omitted, preserves input order
- descending (bool, optional): sort order, default True
- top_k (int, optional): keep only top k records
- round_digits (int, optional): round numeric values for chart display

Returns:
- data (dict): ordered chart-ready mapping
- records (list[dict]): ordered label/value records
- source_count (int)
- kept_count (int)
''')
def tool_chart_data(
    values: ListOrDictStrAny,
    value_field: str,
    label_field: Optional[str] = None,
    fallback_label_field: Optional[str] = None,
    fallback_label_prefix: str = "",
    label_template: Optional[str] = None,
    sort_by: Optional[str] = None,
    descending: bool = True,
    top_k: Optional[int] = None,
    round_digits: Optional[int] = None,
) -> str:
    try:
        if not value_field:
            raise ValueError("tool.chart_data: value_field is required.")

        source_records: List[Dict[str, Any]] = []
        if isinstance(values, dict):
            for key, item in values.items():
                if isinstance(item, dict):
                    normalized = dict(item)
                    normalized.setdefault("__key__", key)
                    source_records.append(normalized)
                else:
                    source_records.append({"__key__": key, value_field: item})
        elif isinstance(values, list):
            source_records = [item for item in values if isinstance(item, dict)]
        else:
            raise ValueError("tool.chart_data: values must be a list of dict records or a dict mapping labels to records.")

        extracted = []
        for idx, item in enumerate(source_records):
            raw_value = _nested_get(item, value_field)
            if raw_value is None:
                continue
            numeric_value = float(raw_value)

            label = None
            if label_template:
                label = _render_label_template(label_template, item).strip()
            if not label and label_field:
                label = _normalize_label(_nested_get(item, label_field))
            if not label and fallback_label_field:
                fallback = _normalize_label(_nested_get(item, fallback_label_field))
                if fallback is not None:
                    label = f"{fallback_label_prefix}{fallback}"
            if not label and "__key__" in item:
                label = _normalize_label(item.get("__key__"))
            if not label:
                label = f"item_{idx}"

            sort_value = idx
            if sort_by:
                raw_sort = _nested_get(item, sort_by)
                try:
                    sort_value = float(raw_sort)
                except Exception:
                    sort_value = numeric_value

            extracted.append({
                "label": label,
                "value": numeric_value,
                "sort_value": sort_value,
                "source_index": idx,
            })

        if sort_by:
            extracted.sort(key=lambda rec: rec["sort_value"], reverse=bool(descending))
        if top_k is not None:
            extracted = extracted[: max(int(top_k), 0)]

        data: Dict[str, float] = {}
        records: List[Dict[str, Any]] = []
        seen_labels: Dict[str, int] = {}
        for rank, rec in enumerate(extracted, start=1):
            label = rec["label"]
            if label in seen_labels:
                seen_labels[label] += 1
                label = f"{label} ({seen_labels[label]})"
            else:
                seen_labels[label] = 1

            value = rec["value"]
            if round_digits is not None:
                value = round(value, int(round_digits))

            data[label] = value
            records.append({
                "rank": rank,
                "label": label,
                "value": value,
                "source_index": rec["source_index"],
            })

        return json.dumps({
            "tool": "tool.chart_data",
            "data": data,
            "records": records,
            "source_count": len(source_records),
            "kept_count": len(records),
            "value_field": value_field,
            "sort_by": sort_by,
            "descending": bool(descending),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result(
            "tool.chart_data",
            error=str(e),
            value_field=value_field,
            label_field=label_field,
            fallback_label_field=fallback_label_field,
            sort_by=sort_by,
            top_k=top_k,
            round_digits=round_digits,
        ))


@mcp.tool(name="logi.truck_trips", description='''
Description:
Calculate the number of truck trips needed to transport a given volume.
Trips = ceil(Volume / Capacity)

Parameters:
- volume_m3 (float): Total volume to transport in cubic meters
- capacity_m3 (float): Truck capacity in cubic meters

Returns:
- trips_required (int): Number of truck trips needed
''')
def logi_truck_trips(volume_m3: float, capacity_m3: float) -> str:
    """
    Number of truck trips needed to haul a volume.
    """
    try:
        if capacity_m3 <= 0:
            trips = 0
        else:
            trips = math.ceil(float(volume_m3) / float(capacity_m3))
        
        result = {
            "tool": "logi.truck_trips",
            "trips_required": int(trips),
            "volume_m3": float(volume_m3),
            "capacity_m3": float(capacity_m3)
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("logi.truck_trips", error=str(e), 
                                       volume_m3=volume_m3, capacity_m3=capacity_m3))


# ============================================================
#  LOGICAL / DECISION EXTENSIONS
# ============================================================

@mcp.tool(name="calc.weighted_sum", description='''
Description:
Compute a weighted sum over a list of numeric values (multi-criteria
aggregation). Commonly used for composite severity indices, e.g.,
combining building damage, flood depth, and road disruption into one score.

Parameters:
- values (list[float]): Input values, same length as weights
- weights (list[float]): Weights (need not sum to 1)
- normalize (bool, optional): If True, normalize weights to sum to 1. Default False.

Returns:
- score (float): Weighted sum
- normalized (bool): Whether weights were normalized
''')
def calc_weighted_sum(
    values: ListFloat,
    weights: ListFloat,
    normalize: bool = False,
) -> str:
    try:
        if len(values) != len(weights):
            raise ValueError("calc.weighted_sum: values and weights length mismatch")
        w = np.asarray(weights, dtype=np.float64)
        v = np.asarray(values, dtype=np.float64)
        if normalize and w.sum() != 0:
            w = w / w.sum()
        score = float(np.dot(v, w))
        return json.dumps({
            "tool": "calc.weighted_sum",
            "score": score,
            "normalized": bool(normalize),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result(
            "calc.weighted_sum", error=str(e),
            values=values, weights=weights
        ))


@mcp.tool(name="logi.priority_score", description='''
Description:
Rank a list of items (e.g., affected districts, damaged facilities) by a
priority score computed as a weighted combination of criteria. Returns the
items sorted from highest to lowest priority. Used to support emergency
dispatch decisions.

Parameters:
- items (list[dict]): Candidate items, each a dict of numeric fields
- criteria (dict[str, float]): Mapping from field name to weight
- top_k (int, optional): If set, return only the top_k items

Returns:
- ranked (list[dict]): Items sorted by priority score (descending), each
  augmented with a "priority_score" field
- count (int): Number of items returned
''')
def logi_priority_score(
    items: ListDict,
    criteria: DictStrFloat,
    top_k: Optional[int] = None,
) -> str:
    try:
        scored: List[Dict[str, Any]] = []
        for it in items:
            score = 0.0
            for field, weight in criteria.items():
                val = it.get(field, 0.0)
                try:
                    score += float(val) * float(weight)
                except Exception:
                    continue
            entry = dict(it)
            entry["priority_score"] = float(score)
            scored.append(entry)
        scored.sort(key=lambda x: x["priority_score"], reverse=True)
        if top_k is not None:
            scored = scored[: int(top_k)]
        return json.dumps({
            "tool": "logi.priority_score",
            "ranked": scored,
            "count": len(scored),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result(
            "logi.priority_score", error=str(e),
            criteria=criteria
        ))


@mcp.tool(name="logi.resource_allocation", description='''
Description:
Allocate a limited resource (e.g., rescue teams, medical supplies) across
multiple demand sites in proportion to their priority/need. Uses a greedy
largest-share-first scheme with per-site capacity caps.

Parameters:
- demands (list[dict]): Each dict has {"id": ..., "need": float, "cap": float (optional)}
- total_supply (float): Total units available
- min_per_site (float, optional): Minimum allocation per site. Default 0.

Returns:
- allocations (list[dict]): Per-site allocation {"id", "need", "allocated"}
- allocated_total (float): Total units actually allocated
- unallocated (float): Units left over
''')
def logi_resource_allocation(
    demands: ListDict,
    total_supply: float,
    min_per_site: float = 0.0,
) -> str:
    try:
        remaining = float(total_supply)
        allocations: List[Dict[str, Any]] = []
        # Minimum guarantee
        for d in demands:
            base = min(float(min_per_site), remaining)
            allocations.append({"id": d.get("id"), "need": float(d.get("need", 0.0)), "allocated": base})
            remaining -= base
        # Proportional top-up
        total_need = sum(max(float(d.get("need", 0.0)) - float(min_per_site), 0.0) for d in demands)
        if total_need > 0 and remaining > 0:
            for d, a in zip(demands, allocations):
                share = max(float(d.get("need", 0.0)) - float(min_per_site), 0.0) / total_need
                extra = min(share * remaining, float(d.get("cap", float("inf"))) - a["allocated"])
                extra = max(extra, 0.0)
                a["allocated"] += extra
        allocated_total = sum(a["allocated"] for a in allocations)
        return json.dumps({
            "tool": "logi.resource_allocation",
            "allocations": allocations,
            "allocated_total": float(allocated_total),
            "unallocated": float(max(total_supply - allocated_total, 0.0)),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result(
            "logi.resource_allocation", error=str(e),
            total_supply=total_supply
        ))


# ============== MCP entry point ==============

if __name__ == "__main__":
    mcp.run(show_banner=False)
