"""DORA evaluation: final-answer accuracy and tool-trajectory metrics.

Final answers are scored per field according to the task's ``eval_spec`` (scalar
closeness, exact match, set F1, dict of scalars, ranking Kendall, point distance,
polygon IoU); a task scores the mean of its fields, a dimension (T1-T5) the mean of its
tasks, and the DORA score the macro-average of the five dimensions. Trajectories are
scored with Tool-Any-Order, Tool-In-Order, Tool-Exact-Match (prefix), Parameter
Accuracy and Efficiency against the gold trajectory.
"""
import copy
import json
import math
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .benchmark import TASKS, load_samples

# ═══════════════════════════════════════════════════════════════════════════
# Tolerances
# ═══════════════════════════════════════════════════════════════════════════

# Scalar closeness: |y_p - y_g| <= TAU_A + TAU_R * |y_g|  (numpy.isclose style)
TAU_R = 0.2
TAU_A = 1.0

def _update_tolerance(tau_r: float, tau_a: float):
    global TAU_R, TAU_A
    TAU_R = tau_r
    TAU_A = tau_a

# Polygon IoU threshold
TAU_IOU = 0.5

# Point thresholds
TAU_PX = 20.0        # pixel Euclidean distance
TAU_GEO_PX = 20      # geographic: threshold = TAU_GEO_PX * GSD  (meters)
TAU_GEO_FALLBACK = 20.0  # fallback if GSD unavailable (meters)


# ═══════════════════════════════════════════════════════════════════════════
# String helpers
# ═══════════════════════════════════════════════════════════════════════════

def normalize_string(s: Any) -> str:
    """Unicode NFKC + lowercase + strip whitespace/punctuation."""
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_tool_name(name: Optional[str]) -> str:
    if not name:
        return ""
    name = name.split("(")[0].strip()
    if "." not in name and "_" in name:
        parts = name.split("_", 1)
        return f"{parts[0]}.{parts[1]}"
    return name


# ═══════════════════════════════════════════════════════════════════════════
# Field-level scoring functions
# ═══════════════════════════════════════════════════════════════════════════

def score_scalar(y_pred: Any, y_gold: Any,
                 tau_r: Optional[float] = None,
                 tau_a: Optional[float] = None) -> float:
    """Scalar closeness: 1[|y_p - y_g| <= tau_a + tau_r * |y_g|]."""
    if tau_r is None:
        tau_r = TAU_R
    if tau_a is None:
        tau_a = TAU_A
    # Exact identity shortcut (handles non-numeric values in sanity mode)
    if y_pred == y_gold:
        return 1.0
    try:
        yp, yg = float(y_pred), float(y_gold)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(yp) and math.isnan(yg):
        return 1.0
    return 1.0 if abs(yp - yg) <= tau_a + tau_r * abs(yg) else 0.0


def _score_form_tolerant(p_f: float, g_f: float, tau_r: Optional[float] = None, tau_a: float = 0.0) -> float:
    """Score a ratio that may be given as a fraction (0.55) or a percentage (55).

    The raw value and the value rescaled to the gold's form (x100 or /100) are both tried and the
    better score is kept, so a genuine ratio above 1.5 (e.g. a detour factor of 1.6) is not taken
    for a percentage."""
    best = score_scalar(p_f, g_f, tau_r, tau_a)
    gold_in_fraction, pred_in_fraction = abs(g_f) <= 1.5, abs(p_f) <= 1.5
    if gold_in_fraction != pred_in_fraction:
        best = max(best, score_scalar(p_f / 100.0 if gold_in_fraction else p_f * 100.0, g_f, tau_r, tau_a))
    return best


_BOOL_YES = {"true", "yes"}
_BOOL_NO = {"false", "no"}


def _canonicalize_exact(v: Any) -> str:
    """Canonicalize exact_match value to handle alias variations:
      - Phase keys: 'phase 3', 'phase_3', 'Phase3' → 'p3'
      - Phase with t: 'phase t1', 'Phase T1' → 't1'
      - Numerics: '5.0' → '5'  (integer-valued floats)
      - Booleans: 'True', 'yes', 'y' → 'true'; 'False', 'no' → 'false'
      - Otherwise: normalize_string (NFKC, lowercase, punct→space, trim)
    """
    if v is None:
        return ""
    # Boolean python value
    if isinstance(v, bool):
        return "true" if v else "false"
    # Integer-valued float: 5.0 → '5'
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, int):
        return str(v)
    s = normalize_string(v)
    if not s:
        return s
    # Boolean aliases
    if s in _BOOL_YES:
        return "true"
    if s in _BOOL_NO:
        return "false"
    # "phase tN" or "phase_tN" → "tN"
    m = re.match(r'^phase[\s_]*(t\d+)$', s)
    if m:
        return m.group(1)
    # "phase N" or "phase_N" → "pN"
    m = re.match(r'^phase[\s_]*(\d+)$', s)
    if m:
        return f'p{m.group(1)}'
    # "t_N" / "t N" → "tN"
    m = re.match(r'^t[\s_]+(\d+)$', s)
    if m:
        return f't{m.group(1)}'
    # "p_N" / "p N" → "pN"
    m = re.match(r'^p[\s_]+(\d+)$', s)
    if m:
        return f'p{m.group(1)}'
    # Numeric string: '5.0' → '5'
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except (ValueError, TypeError):
        pass
    return s


def score_exact(y_pred: Any, y_gold: Any) -> float:
    """Normalized exact string match with alias canonicalization."""
    return 1.0 if _canonicalize_exact(y_pred) == _canonicalize_exact(y_gold) else 0.0


def score_set_f1(y_pred: Any, y_gold: Any) -> float:
    """F1 over normalized string sets."""
    def to_set(v):
        if isinstance(v, (list, tuple, set)):
            return {normalize_string(x) for x in v}
        if isinstance(v, str):
            return {normalize_string(x) for x in v.split(",") if x.strip()}
        return set()

    pred_s, gold_s = to_set(y_pred), to_set(y_gold)
    if not gold_s and not pred_s:
        return 1.0
    if not gold_s or not pred_s:
        return 0.0
    tp = len(pred_s & gold_s)
    p = tp / len(pred_s)
    r = tp / len(gold_s)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _normalize_dict_key(k: Any) -> str:
    """Normalize dict keys for matching across format variations.

    Examples:
      'phase_2', 'phase2', 'phase 2', 'Phase 2'  -> 'p2'
      'phase_t1', 'phase t1', 'Phase T1'         -> 't1'
      't_1', 'T1', 't1'                          -> 't1'
      'p_2', 'P2'                                -> 'p2'
      'Flood', 'FLOOD', 'flood'                  -> 'flood'
    """
    if not isinstance(k, str):
        return str(k).lower().strip()
    s = k.lower().strip()
    # "phase tN" or "phase_tN" → "tN"
    m = re.match(r'^phase[\s_]*(t\d+)$', s)
    if m:
        return m.group(1)
    # "phase N" or "phase_N" or "phaseN" → "pN"
    m = re.match(r'^phase[\s_]*(\d+)$', s)
    if m:
        return f'p{m.group(1)}'
    # "t_N" / "tN" / "t N" → "tN"
    m = re.match(r'^t[\s_]*(\d+)$', s)
    if m:
        return f't{m.group(1)}'
    # "p_N" / "pN" / "p N" → "pN"
    m = re.match(r'^p[\s_]*(\d+)$', s)
    if m:
        return f'p{m.group(1)}'
    # "rank_N" / "rankN" / "rank N" / bare "N" → "N" (for top-K rank keys)
    m = re.match(r'^rank[\s_]*(\d+)$', s)
    if m:
        return m.group(1)
    # Default: lowercase, strip, normalize underscore/space/hyphen
    s = re.sub(r'[\s\-_]+', '_', s).strip('_')
    # Strip trailing plural 's' on last token (buildings→building, roads→road)
    tokens = s.split('_')
    if tokens and len(tokens[-1]) > 3 and tokens[-1].endswith('s') and not tokens[-1].endswith('ss'):
        tokens[-1] = tokens[-1][:-1]
    s = '_'.join(tokens)
    return s


def score_scalar_dict(y_pred: Any, y_gold: Any,
                      tau_r: Optional[float] = None,
                      tau_a: Optional[float] = None,
                      gsd_m: Optional[float] = None) -> float:
    """Per-key scalar scoring averaged over the gold keys, with key normalization.

    Identifier keys (``*_id``) must match exactly; latitude/longitude keys use the point tolerance
    (TAU_GEO_PX pixels at the scene GSD, in degrees) instead of a relative one."""
    if tau_r is None:
        tau_r = TAU_R
    if tau_a is None:
        tau_a = TAU_A
    if not isinstance(y_gold, dict):
        return 0.0
    if not isinstance(y_pred, dict):
        return 0.0
    # Normalize keys to handle format variations (phase_2 vs p2 vs phase2)
    y_gold_norm = {_normalize_dict_key(k): v for k, v in y_gold.items()}
    y_pred_norm = {_normalize_dict_key(k): v for k, v in y_pred.items()}
    y_gold = y_gold_norm
    y_pred = y_pred_norm
    # Score only over GOLD keys — extra pred keys don't penalize correctness.
    # This accommodates models that emit helpful supplementary statistics
    # (e.g., total_points, valid_points alongside the required fields).
    keys = set(y_gold)
    if not keys:
        return 1.0
    scores = []
    for k in keys:
        if k in y_gold and k in y_pred:
            # Per-value adaptive tolerance: if gold value is ratio-like
            # (|gold|<=1.5), use pure relative tolerance and form normalization
            # — same logic as ratio_pct. This prevents tau_a=1 from trivially
            # passing all predictions against small ratio values (e.g. 0.1944).
            g_val = y_gold[k]
            p_val = y_pred[k]
            if k == "id" or k.endswith("_id") or k.endswith("osmid"):
                scores.append(score_exact(p_val, g_val))
                continue
            try:
                g_f = float(g_val)
                p_f = float(p_val)
                if re.search(r"(^|_)(lat|latitude|lon|lng|longitude)$", k):
                    tol_m = TAU_GEO_PX * gsd_m if gsd_m else TAU_GEO_FALLBACK
                    scores.append(1.0 if abs(p_f - g_f) <= tol_m / 111_320.0 else 0.0)
                    continue
                effective_tau_a = 0.0 if abs(g_f) <= 1.5 else tau_a
                scores.append(_score_form_tolerant(p_f, g_f, tau_r, effective_tau_a))
            except (TypeError, ValueError):
                scores.append(score_scalar(p_val, g_val, tau_r, tau_a))
        else:
            scores.append(0.0)
    return sum(scores) / len(scores)


def _extract_point_coords(v: Any, want: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract (row, col) or (lat, lon) from various container shapes.

    Supports:
      - Dict with keys: {pixel_row, pixel_col}, {row, col}, {r, c}
      - Dict with keys: {latitude, longitude}, {lat, lon}, {lat, lng}
      - List/tuple of 2: [row, col] or [lat, lon]
      - Separate fields (caller handles pairing)
    Returns (first, second) or (None, None) if can't extract.
    """
    if v is None:
        return None, None
    if isinstance(v, (list, tuple)) and len(v) == 2:
        try:
            return float(v[0]), float(v[1])
        except (TypeError, ValueError):
            return None, None
    if isinstance(v, dict):
        if want == "pixel":
            key_pairs = [
                ("pixel_row", "pixel_col"), ("row", "col"),
                ("r", "c"), ("y", "x"),
            ]
        else:  # geo
            key_pairs = [
                ("latitude", "longitude"), ("lat", "lon"),
                ("lat", "lng"), ("y", "x"),
            ]
        for k1, k2 in key_pairs:
            if k1 in v and k2 in v:
                try:
                    return float(v[k1]), float(v[k2])
                except (TypeError, ValueError):
                    pass
    return None, None


def _score_point_field(
    y_pred: Any, y_gold: Any, etype: str, gsd_m: Optional[float]
) -> float:
    """Score a paired point field that holds both coordinates in one value.

    Handles dict-vs-list/dict format mismatches (e.g., gold is
    {"pixel_row": 461, "pixel_col": 856} but pred is [444, 878]).
    """
    want = "pixel" if etype == "point_pixel" else "geo"
    g1, g2 = _extract_point_coords(y_gold, want)
    p1, p2 = _extract_point_coords(y_pred, want)
    if g1 is None or g2 is None or p1 is None or p2 is None:
        # Fallback to exact/scalar compare
        if y_pred == y_gold:
            return 1.0
        return score_scalar(y_pred, y_gold)
    if want == "pixel":
        dist = math.sqrt((p1 - g1) ** 2 + (p2 - g2) ** 2)
        return 1.0 if dist <= TAU_PX else 0.0
    else:
        dist = _haversine_m(p1, p2, g1, g2)
        threshold = TAU_GEO_PX * gsd_m if gsd_m else TAU_GEO_FALLBACK
        return 1.0 if dist <= threshold else 0.0


def score_ranking_kendall(y_pred: Any, y_gold: Any) -> float:
    """Kendall rank correlation (mapped to [0, 1])."""
    if not isinstance(y_gold, list) or not isinstance(y_pred, list):
        return 0.0
    gold_rank = {v: i for i, v in enumerate(y_gold)}
    pred_rank = {v: i for i, v in enumerate(y_pred)}
    common = [v for v in y_gold if v in pred_rank]
    if len(common) < 2:
        return 1.0 if set(y_pred) == set(y_gold) else 0.0
    n = len(common)
    concordant = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            gi = gold_rank[common[i]] - gold_rank[common[j]]
            pi = pred_rank[common[i]] - pred_rank[common[j]]
            total += 1
            if gi * pi > 0:
                concordant += 1
    tau = (2 * concordant - total) / total if total > 0 else 0.0
    return (tau + 1.0) / 2.0  # map [-1, 1] → [0, 1]


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in meters."""
    la1, lo1 = math.radians(lat1), math.radians(lon1)
    la2, lo2 = math.radians(lat2), math.radians(lon2)
    dlat, dlon = la2 - la1, lo2 - lo1
    a = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return 6_371_000 * 2 * math.asin(math.sqrt(min(a, 1.0)))


def resolve_polygon_path(p: str, roots: Tuple[Path, ...] = ()) -> Optional[Path]:
    """Locate a polygon file: as given, relative to one of ``roots``, or -- for a run copied
    from another machine -- the longest trailing part of the path that exists under a root."""
    if not p:
        return None
    if Path(p).is_file():
        return Path(p)
    parts = [x for x in str(p).replace("\\", "/").split("/") if x and not x.endswith(":")]
    for i in range(len(parts)):
        for root in roots:
            candidate = Path(root).joinpath(*parts[i:])
            if candidate.is_file():
                return candidate
    return None


_POLYGON_IOU_WARNED: set = set()
_EMPTY_ANSWERS = {"", "none", "null", "n/a", "na", "nan", "[]", "{}"}


def _warn_missing(kind: str, path: str) -> None:
    if (kind, path) not in _POLYGON_IOU_WARNED:
        _POLYGON_IOU_WARNED.add((kind, path))
        warnings.warn(f"polygon_iou: cannot locate {kind} polygon file: {path}", stacklevel=3)


def _largest_polygon(geojson: Dict[str, Any], grid: Optional[Dict[str, Any]] = None):
    """Largest polygon of a GeoJSON object, expressed in the world frame of ``grid`` if given.

    Files without a ``crs`` member are in pixel coordinates (column, row) and are georeferenced
    with the grid's affine transform; files in another CRS are reprojected to the grid's CRS.
    """
    from shapely.affinity import affine_transform
    from shapely.geometry import shape

    if geojson.get("type") == "FeatureCollection":
        geometries = [f.get("geometry") for f in geojson.get("features", [])]
    else:
        geometries = [geojson.get("geometry") if geojson.get("type") == "Feature" else geojson]
    polygons = []
    for g in filter(None, geometries):
        geom = shape(g)
        polygons += [p for p in getattr(geom, "geoms", [geom]) if p.geom_type == "Polygon" and not p.is_empty]
    if not polygons:
        return None
    largest = max(polygons, key=lambda p: p.area)
    if grid is None:
        return largest
    crs = (geojson.get("crs") or {}).get("properties", {}).get("name")
    if crs is None:
        a, b, c, d, e, f = grid["transform"]
        return affine_transform(largest, [a, b, d, e, c, f])
    from pyproj import CRS, Transformer
    from shapely.ops import transform

    if CRS(crs) != CRS(grid["crs"]):
        largest = transform(Transformer.from_crs(CRS(crs), CRS(grid["crs"]), always_xy=True).transform, largest)
    return largest


def score_polygon_iou(pred_path: Any, gold_path: Any, tau_iou: float = TAU_IOU,
                      roots: Tuple[Path, ...] = ()) -> float:
    """1 if the largest predicted polygon overlaps the largest gold polygon with IoU >= ``tau_iou``.

    Gold files (``tasks/gold_polygons``) hold the largest gold polygon. For georeferenced scenes
    they are in world coordinates and carry the image's pixel grid (``dora_grid``), so predictions
    vectorized from PNG masks (pixel coordinates) and from GeoTIFF masks are compared in one frame.
    An empty gold file means there is no such polygon: an empty prediction then scores 1.
    """
    pred_path, gold_path = str(pred_path), str(gold_path)
    if pred_path == gold_path:
        return 1.0
    gold_file = resolve_polygon_path(gold_path, roots)
    if gold_file is None:
        _warn_missing("gold", gold_path)
        return 0.0
    empty_answer = pred_path.strip().lower() in _EMPTY_ANSWERS
    try:
        with open(gold_file, encoding="utf-8") as f:
            gold_json = json.load(f)
        gold = _largest_polygon(gold_json)
        pred_file = None if empty_answer else resolve_polygon_path(pred_path, roots)
        if pred_file is None and not empty_answer:
            _warn_missing("pred", pred_path)
        if pred_file is None:
            return 1.0 if gold is None and empty_answer else 0.0
        with open(pred_file, encoding="utf-8") as f:
            pred = _largest_polygon(json.load(f), gold_json.get("dora_grid"))
        if gold is None or pred is None:
            return 1.0 if gold is None and pred is None else 0.0
        gold, pred = gold.buffer(0), pred.buffer(0)  # repair self-intersecting rings
        union = gold.union(pred).area
        return 1.0 if union > 0 and gold.intersection(pred).area / union >= tau_iou else 0.0
    except Exception as e:
        warnings.warn(f"polygon_iou error for {pred_path}: {e}", stacklevel=2)
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Point pairing logic
# ═══════════════════════════════════════════════════════════════════════════

_GEO_SUFFIXES = {"_latitude": "lat", "_longitude": "lon",
                 "_lat": "lat", "_lon": "lon"}
_PX_SUFFIXES = {"_row": "row", "_col": "col"}


def _detect_point_pairs(eval_spec: Dict[str, str]) -> Dict[str, Dict]:
    """Group point_haversine / point_pixel fields into (base, {lat/row, lon/col}) pairs.

    Returns: {base_name: {"type": "geo"|"pixel", "lat"|"row": field, "lon"|"col": field}}
    """
    pairs: Dict[str, Dict] = {}

    for field, etype in eval_spec.items():
        if etype == "point_haversine":
            for suffix, role in _GEO_SUFFIXES.items():
                if field.endswith(suffix):
                    base = field[: -len(suffix)]
                    pairs.setdefault(base, {"type": "geo"})
                    pairs[base][role] = field
                    break
        elif etype == "point_pixel":
            for suffix, role in _PX_SUFFIXES.items():
                if field.endswith(suffix):
                    base = field[: -len(suffix)]
                    pairs.setdefault(base, {"type": "pixel"})
                    pairs[base][role] = field
                    break

    return pairs


# ═══════════════════════════════════════════════════════════════════════════
# Task-level answer scoring
# ═══════════════════════════════════════════════════════════════════════════

def score_task_answer(
    pred_answer: Dict[str, Any],
    gold_answer: Dict[str, Any],
    eval_spec: Dict[str, str],
    gsd_m: Optional[float] = None,
    polygon_roots: Tuple[Path, ...] = (),
) -> Tuple[float, Dict[str, float]]:
    """Score all fields of a single task.

    Returns (task_score, {field_name: score}).
    """
    field_scores: Dict[str, float] = {}

    # Detect point pairs so we score them as single units
    point_pairs = _detect_point_pairs(eval_spec)
    paired_fields: set = set()
    for base, info in point_pairs.items():
        paired_fields.update(v for k, v in info.items() if k != "type")

    for field, etype in eval_spec.items():
        # Skip ignored fields
        if etype == "ignore":
            continue

        # Skip fields that are part of a point pair (scored together below)
        if field in paired_fields:
            continue

        y_pred = pred_answer.get(field)
        y_gold = gold_answer.get(field)

        if etype in ("scalar_continuous", "scalar_count"):
            # For percentage-named fields, treat as ratio_pct: strip '%',
            # auto-normalize fraction (0-1) vs percent (>1) form mismatches.
            # Matches fields like 'persistent_percentage_of_final_extent',
            # 'percentage_of_pre_disaster_building_footprint', '*_pct', '*_ratio'.
            fl = field.lower()
            is_pct_like = (
                etype == "scalar_continuous"
                and (
                    "percentage" in fl or "_pct" in fl or fl.endswith("pct")
                    or "_ratio" in fl or fl.endswith("ratio")
                    or "_share" in fl or fl.endswith("share")
                )
            )
            if is_pct_like:
                y_p, y_g = y_pred, y_gold
                try:
                    def _parse_ratio(v):
                        if isinstance(v, str):
                            v = v.strip().rstrip('%').strip()
                        return float(v)
                    if y_p is not None and y_g is not None:
                        y_p, y_g = _parse_ratio(y_p), _parse_ratio(y_g)
                        field_scores[field] = _score_form_tolerant(y_p, y_g)
                except (TypeError, ValueError):
                    pass
                if field not in field_scores:
                    field_scores[field] = score_scalar(y_p, y_g, tau_a=0.0)
            else:
                field_scores[field] = score_scalar(y_pred, y_gold)

        elif etype == "ratio_pct":
            # Two-step ratio handling:
            # (1) Strip trailing '%' sign and normalize pred to match gold's form
            #     (fraction 0-1 vs percent >1), since models may return 0.65,
            #     65, or "65%" for the same 65% ratio.
            # (2) Apply pure relative tolerance (tau_a=0) since many ratio gold
            #     values are <1, where the default tau_a=1 would make any
            #     prediction in [-1, 1+gold] pass (trivially true).
            y_p, y_g = y_pred, y_gold
            try:
                def _parse_ratio(v):
                    if isinstance(v, str):
                        v = v.strip().rstrip('%').strip()
                    return float(v)
                if y_p is not None and y_g is not None:
                    # Fraction (0.65) or percent (65): both forms are tried against the gold.
                    y_p, y_g = _parse_ratio(y_p), _parse_ratio(y_g)
                    field_scores[field] = _score_form_tolerant(y_p, y_g)
            except (TypeError, ValueError):
                pass
            if field not in field_scores:
                field_scores[field] = score_scalar(y_p, y_g, tau_a=0.0)

        elif etype == "exact_match":
            field_scores[field] = score_exact(y_pred, y_gold)

        elif etype == "scalar_dict":
            field_scores[field] = score_scalar_dict(y_pred, y_gold, gsd_m=gsd_m)

        elif etype == "set_f1":
            field_scores[field] = score_set_f1(y_pred, y_gold)

        elif etype == "polygon_iou":
            field_scores[field] = score_polygon_iou(y_pred, y_gold, roots=polygon_roots)

        elif etype == "ranking_kendall":
            field_scores[field] = score_ranking_kendall(y_pred, y_gold)

        elif etype in ("point_haversine", "point_pixel"):
            field_scores[field] = _score_point_field(
                y_pred, y_gold, etype, gsd_m
            )

        else:
            # Unknown type → treat as exact match
            field_scores[field] = score_exact(y_pred, y_gold)

    # Score point pairs
    for base, info in point_pairs.items():
        ptype = info["type"]
        if ptype == "geo" and "lat" in info and "lon" in info:
            try:
                pred_lat = float(pred_answer.get(info["lat"], 0))
                pred_lon = float(pred_answer.get(info["lon"], 0))
                gold_lat = float(gold_answer.get(info["lat"], 0))
                gold_lon = float(gold_answer.get(info["lon"], 0))
                dist = _haversine_m(pred_lat, pred_lon, gold_lat, gold_lon)
                threshold = TAU_GEO_PX * gsd_m if gsd_m else TAU_GEO_FALLBACK
                s = 1.0 if dist <= threshold else 0.0
            except (TypeError, ValueError):
                s = 0.0
            field_scores[f"{base}__point_geo"] = s

        elif ptype == "pixel" and "row" in info and "col" in info:
            try:
                pr = float(pred_answer.get(info["row"], 0))
                pc = float(pred_answer.get(info["col"], 0))
                gr = float(gold_answer.get(info["row"], 0))
                gc = float(gold_answer.get(info["col"], 0))
                dist = math.sqrt((pr - gr) ** 2 + (pc - gc) ** 2)
                s = 1.0 if dist <= TAU_PX else 0.0
            except (TypeError, ValueError):
                s = 0.0
            field_scores[f"{base}__point_px"] = s

    # Task score = mean of scored fields (NaN if no scored fields)
    if not field_scores:
        return float("nan"), field_scores
    task_score = sum(field_scores.values()) / len(field_scores)
    return task_score, field_scores


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory metrics
# ═══════════════════════════════════════════════════════════════════════════

def _flatten_trajectory(traj: List[Dict]) -> List[Dict]:
    """Flatten trajectory, skipping tool.loop wrappers.

    The gold trajectory stores tool.loop as a wrapper step followed by the
    actual expanded tool calls.  We skip the wrapper entirely to avoid
    double-counting the inner calls.

    Models often emit the wrapper as `tool_loop(bind, calls, loop_over, ...)`
    with args in parentheses — we strip the parens via normalize_tool_name
    before the identity check, so these are also recognised as wrappers.
    """
    flat = []
    for step in traj:
        if not isinstance(step, dict):
            continue
        call = step.get("call", "")
        # Normalize to strip parenthesized arg list and unify tool.X vs tool_X
        normalized = normalize_tool_name(call)
        if normalized in ("tool.loop", "tool.reduce"):
            # Skip the wrapper — the expanded calls follow as subsequent steps
            continue
        elif call:
            flat.append(step)
    return flat


def extract_tool_seq(traj: List[Dict]) -> List[str]:
    flat = _flatten_trajectory(traj)
    return [normalize_tool_name(s.get("call", "")) for s in flat
            if normalize_tool_name(s.get("call", ""))]


def metric_tool_any_order(pred_seq: List[str], gold_seq: List[str]) -> float:
    """Tool-set recall (order-agnostic)."""
    if not gold_seq:
        return 1.0 if not pred_seq else 0.0
    gold_set = set(gold_seq)
    return len(gold_set & set(pred_seq)) / len(gold_set)


def metric_tool_in_order(pred_seq: List[str], gold_seq: List[str]) -> float:
    """Longest common subsequence / |gold|."""
    if not gold_seq:
        return 1.0 if not pred_seq else 0.0
    m, n = len(gold_seq), len(pred_seq)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gold_seq[i - 1] == pred_seq[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n] / m


def metric_tool_exact_match(pred_seq: List[str], gold_seq: List[str]) -> float:
    """Longest matching PREFIX / |gold|."""
    if not gold_seq:
        return 1.0 if not pred_seq else 0.0
    prefix_len = 0
    for g, p in zip(gold_seq, pred_seq):
        if g == p:
            prefix_len += 1
        else:
            break
    return prefix_len / len(gold_seq)


def _is_file_path(s: str) -> bool:
    """Heuristic: is this string a file/directory path?"""
    if not isinstance(s, str) or len(s) < 4:
        return False
    # Contains path separators and a file extension
    has_sep = "/" in s or "\\" in s
    has_ext = bool(re.search(r"\.\w{1,6}$", s))
    # Or starts with common path prefixes
    starts_like_path = s.startswith(("/", "D:", "C:", "./", "../", "~"))
    return (has_sep and has_ext) or starts_like_path


def _build_trajectory_output_map(flat_traj: List[Dict]) -> dict:
    """Build a map from output file paths to (step_index, output_key) for a
    trajectory, so we can check if a later step's input references an earlier
    step's output (internal consistency).

    Returns: {path_string: (step_idx, key_name), ...}
    """
    import json as _json
    out_map = {}
    for idx, step in enumerate(flat_traj):
        obs_list = step.get("obs", [])
        if not isinstance(obs_list, list):
            obs_list = [obs_list]
        for obs_item in obs_list:
            d = None
            if isinstance(obs_item, dict):
                # Gold trajectories store obs as a direct dict
                # (e.g., {"tool": "seg.X", "mask_path": "/path/..."})
                # Pred trajectories store obs as {"text": "{JSON string}"}
                text = obs_item.get("text", "")
                if text:
                    try:
                        d = _json.loads(text)
                    except (ValueError, TypeError):
                        pass
                else:
                    # Direct dict obs (gold format)
                    d = obs_item
            elif isinstance(obs_item, str):
                try:
                    d = _json.loads(obs_item)
                except (ValueError, TypeError):
                    pass
            if isinstance(d, dict):
                for k, v in d.items():
                    if _is_file_path(str(v)):
                        out_map[str(v)] = (idx, k)
        # Also check args that look like outputs (some tools store output in args)
        args = step.get("args", {})
        if isinstance(args, dict):
            for k, v in args.items():
                if k.startswith("output") and _is_file_path(str(v)):
                    out_map[str(v)] = (idx, k)
    return out_map


def _match_path_arg(
    gv: str, pv: str,
    g_step_idx: int, p_step_idx: int,
    gold_out_map: dict, pred_out_map: dict,
    gold_flat: List[Dict], pred_flat: List[Dict],
) -> bool:
    """File-path argument comparison based on structural role consistency.

    Core principle: file paths are intermediate artifacts with run-specific
    names. What matters is that pred and gold reference the **same structural
    role** — i.e., they originate from the same pipeline step or serve as the
    same raw input. We do NOT require string similarity between paths.

    Strategy (in priority order):
      1. Exact string match → True  (trivial case)
      2. Both are raw inputs (not produced by any trajectory step) → True
         (both reference dataset files provided in the question context)
      3. Both are outputs of the same upstream step index → True
         (internal consistency: pred step j produced it, gold step j produced it)
      4. One is tracked as output, the other is not → check if they come from
         a step with the same tool name (handles partial output-map coverage)
      5. Fallback basename / canonical-stem match for edge cases
    """
    import os.path as _osp

    # 1. Exact normalized match
    if normalize_string(pv) == normalize_string(gv):
        return True

    g_is_output = gv in gold_out_map
    p_is_output = pv in pred_out_map

    # 2. Both are raw inputs (not produced by any trajectory step)
    #    → they must be dataset input files; accept as match
    if not g_is_output and not p_is_output:
        return True

    # 3. Both are outputs of trajectory steps → same upstream step index
    if g_is_output and p_is_output:
        g_src_step, g_src_key = gold_out_map[gv]
        p_src_step, p_src_key = pred_out_map[pv]
        if g_src_step == p_src_step:
            return True

    # 4. One tracked, the other not → check if a step at the same index
    #    with the same tool name produced a similar output key
    if g_is_output and not p_is_output:
        g_src_step, g_src_key = gold_out_map[gv]
        # Check if pred has an untracked path from the same step's tool
        if g_src_step < len(pred_flat):
            g_tool = normalize_tool_name(gold_flat[g_src_step].get("call", ""))
            p_tool = normalize_tool_name(pred_flat[g_src_step].get("call", ""))
            if g_tool == p_tool:
                return True
    if p_is_output and not g_is_output:
        p_src_step, p_src_key = pred_out_map[pv]
        if p_src_step < len(gold_flat):
            p_tool = normalize_tool_name(pred_flat[p_src_step].get("call", ""))
            g_tool = normalize_tool_name(gold_flat[p_src_step].get("call", ""))
            if g_tool == p_tool:
                return True

    # 5. Fallback: basename / canonical-stem match
    gv_n = gv.replace("\\", "/")
    pv_n = pv.replace("\\", "/")
    g_base = _osp.basename(gv_n)
    p_base = _osp.basename(pv_n)

    if g_base and g_base == p_base:
        return True

    def _canonical_stem(basename: str) -> str:
        """Remove interleaved _<hex> hash segments before extension."""
        name, ext = _osp.splitext(basename)
        prev = None
        while prev != name:
            prev = name
            name = re.sub(r"_[0-9a-f]{6,10}(?=_|$)", "", name)
        return name + ext

    g_canon = _canonical_stem(g_base)
    p_canon = _canonical_stem(p_base)
    if g_canon and g_canon == p_canon:
        return True

    return False


def metric_parameter_accuracy(
    pred_traj: List[Dict], gold_traj: List[Dict]
) -> float:
    """Per-step argument matching, conditioned on correct tool name.

    For each gold step i, if pred step i has the same tool name, compare args
    using type-aware matching:
      - Scalars: closeness within TAU_A + TAU_R * |gold|
      - File paths: structural-role & internal-consistency matching
        (same upstream tool output, same canonical stem, or same basename)
      - Other strings: normalized exact match
      - Booleans, lists, dicts: equality
    Score = matched_steps / |gold|.
    """
    pred_flat = _flatten_trajectory(pred_traj)
    gold_flat = _flatten_trajectory(gold_traj)

    if not gold_flat:
        return 1.0 if not pred_flat else 0.0

    # Pre-build output maps for internal-consistency checking
    gold_out_map = _build_trajectory_output_map(gold_flat)
    pred_out_map = _build_trajectory_output_map(pred_flat)

    matched = 0
    for i, g_step in enumerate(gold_flat):
        g_name = normalize_tool_name(g_step.get("call", ""))
        g_args = g_step.get("args", {})
        if not isinstance(g_args, dict):
            g_args = {}

        if i >= len(pred_flat):
            continue

        p_name = normalize_tool_name(pred_flat[i].get("call", ""))
        p_args = pred_flat[i].get("args", {})
        if not isinstance(p_args, dict):
            p_args = {}

        if g_name != p_name:
            continue

        # Compare args: all gold arg keys must match
        if not g_args:
            matched += 1
            continue

        # Pattern for gold trajectory placeholder references like
        # "<results> from trajectory[1].obs" or "<area_m2> from trajectory[3].obs"
        _TRAJ_REF_RE = re.compile(
            r"^<\w+>\s+from\s+trajectory\[\d+\]\.obs$"
        )

        def _is_traj_ref(v) -> bool:
            """Check if a value is a gold trajectory placeholder reference."""
            return isinstance(v, str) and bool(_TRAJ_REF_RE.match(v))

        def _match_value(gv, pv):
            """Recursively compare arg values with path-aware matching."""
            # Gold placeholder reference: "<xxx> from trajectory[N].obs"
            # Accept any non-None pred value (the agent resolved it at runtime)
            if _is_traj_ref(gv):
                return pv is not None
            if isinstance(gv, (int, float)) and not isinstance(gv, bool):
                try:
                    pv_f = float(pv)
                    gv_f = float(gv)
                    if math.isnan(pv_f) and math.isnan(gv_f):
                        return True
                    return abs(pv_f - gv_f) <= TAU_A + TAU_R * abs(gv_f)
                except (TypeError, ValueError):
                    return False
            elif isinstance(gv, str):
                if _is_file_path(gv) or _is_file_path(str(pv)):
                    return _match_path_arg(
                        gv, str(pv), i, i,
                        gold_out_map, pred_out_map,
                        gold_flat, pred_flat,
                    )
                else:
                    return normalize_string(pv) == normalize_string(gv)
            elif isinstance(gv, bool):
                return pv == gv
            elif isinstance(gv, list):
                if not isinstance(pv, list) or len(pv) != len(gv):
                    return False
                if not gv:
                    return True
                return all(_match_value(g_el, p_el) for g_el, p_el in zip(gv, pv))
            elif isinstance(gv, dict):
                if not isinstance(pv, dict):
                    return False
                if set(gv.keys()) != set(pv.keys()):
                    return False
                if not gv:
                    return True
                return all(_match_value(gv[k2], pv[k2]) for k2 in gv)
            else:
                return str(pv) == str(gv)

        arg_matches = 0
        for k, gv in g_args.items():
            if k not in p_args:
                continue
            if _match_value(gv, p_args[k]):
                arg_matches += 1

        matched += arg_matches / len(g_args)

    return matched / len(gold_flat)


def metric_efficiency(pred_seq: List[str], gold_seq: List[str]) -> float:
    """|gold| / max(|pred|, |gold|)."""
    g, p = len(gold_seq), len(pred_seq)
    if g == 0:
        return 1.0 if p == 0 else 0.0
    return g / max(p, g)


def compute_trajectory_metrics(
    pred_traj: List[Dict], gold_traj: List[Dict]
) -> Dict[str, float]:
    pred_seq = extract_tool_seq(pred_traj)
    gold_seq = extract_tool_seq(gold_traj)
    return {
        "t_any":   metric_tool_any_order(pred_seq, gold_seq),
        "t_ord":   metric_tool_in_order(pred_seq, gold_seq),
        "t_em":    metric_tool_exact_match(pred_seq, gold_seq),
        "par_acc": metric_parameter_accuracy(pred_traj, gold_traj),
        "eff":     metric_efficiency(pred_seq, gold_seq),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_gold(tasks_dir: Path) -> Dict[str, List[Dict]]:
    """Gold samples per dimension: ``{"t1": [sample, ...], ...}``."""
    return {dim: load_samples(task, tasks_dir) for dim, task in TASKS.items() if (Path(tasks_dir) / f"{task}.json").exists()}


def get_gsd(sample: Dict) -> Optional[float]:
    """Extract pixel_size_m (GSD) from sample input_data."""
    inp = sample.get("input_data", {})
    for key in ("pre_image", "post_image"):
        img = inp.get(key, {})
        if isinstance(img, dict) and "pixel_size_m" in img:
            return float(img["pixel_size_m"])
    # T4: temporal images
    temp = inp.get("temporal_images", {})
    if isinstance(temp, dict):
        for ts_data in temp.values():
            if isinstance(ts_data, dict) and "pixel_size_m" in ts_data:
                return float(ts_data["pixel_size_m"])
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation engine
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_dimension(
    gold_samples: List[Dict],
    pred_samples: Optional[List[Dict]] = None,  # None → sanity mode
    polygon_roots: Tuple[Path, ...] = (),
) -> Dict[str, Any]:
    """Evaluate one dimension (T1–T5).

    In sanity mode (pred_samples is None), gold is used as both gold and pred.
    """
    task_results = []

    # Build sample_id → pred_sample lookup for robust matching
    pred_lookup = {}
    if pred_samples is not None:
        for ps in pred_samples:
            sid = ps.get("sample_id", "")
            if sid:
                pred_lookup[sid] = ps

    for s_idx, gold_sample in enumerate(gold_samples):
        gold_sid = gold_sample.get("sample_id", f"sample_{s_idx}")
        if pred_samples is None:
            pred_sample = gold_sample
        elif gold_sid in pred_lookup:
            pred_sample = pred_lookup[gold_sid]
        elif s_idx < len(pred_samples):
            # Fallback to positional matching
            pred_sample = pred_samples[s_idx]
        else:
            # Prediction missing — use empty stub so tasks score 0
            pred_sample = {"sample_id": gold_sid, "tasks": []}
        gsd = get_gsd(gold_sample)
        gold_tasks = gold_sample.get("tasks", [])
        pred_tasks = pred_sample.get("tasks", [])

        for t_idx, gold_task in enumerate(gold_tasks):
            pred_task = gold_task if pred_samples is None else (
                pred_tasks[t_idx] if t_idx < len(pred_tasks) else {}
            )

            gold_answer = gold_task.get("final_answer", {})
            pred_answer = pred_task.get("final_answer", {})
            eval_spec = gold_task.get("eval_spec", {})
            gold_traj = gold_task.get("trajectory", [])
            pred_traj = pred_task.get("trajectory", [])

            # Answer metrics
            task_score, field_scores = score_task_answer(
                pred_answer, gold_answer, eval_spec, gsd_m=gsd, polygon_roots=polygon_roots
            )

            # Trajectory metrics
            traj_metrics = compute_trajectory_metrics(pred_traj, gold_traj)

            task_results.append({
                "sample_id": gold_sample.get("sample_id", f"sample_{s_idx}"),
                "task_idx": t_idx,
                "task_score": round(task_score, 6) if not math.isnan(task_score) else None,
                "field_scores": {k: round(v, 6) for k, v in field_scores.items()},
                "n_scored_fields": len(field_scores),
                **{k: round(v, 6) for k, v in traj_metrics.items()},
            })

    # Dimension-level aggregation (skip NaN task scores, i.e. all-ignore tasks)
    n = len(task_results)
    if n == 0:
        return {"tasks": [], "n_tasks": 0, "n_scored_tasks": 0}

    scored = [r for r in task_results if r["task_score"] is not None]
    n_scored = len(scored)
    dim_score = sum(r["task_score"] for r in scored) / n_scored if n_scored else 0.0
    dim_traj = {}
    for m in ("t_any", "t_ord", "t_em", "par_acc", "eff"):
        dim_traj[m] = sum(r[m] for r in task_results) / n

    return {
        "dim_score": round(dim_score, 6),
        "n_tasks": n,
        "n_scored_tasks": n_scored,
        **{k: round(v, 6) for k, v in dim_traj.items()},
        "tasks": task_results,
    }


def _parse_string_prediction(text: str) -> Any:
    """Parse a string predicted_answer into a dict when possible.

    Handles:
      - Bare JSON dict:           '{"field": 1, ...}'
      - Markdown-fenced JSON:     '```json\\n{...}\\n```' or '```\\n{...}\\n```'
      - JSON embedded in prose:   'The answer is {"field": 1}'
      - <Answer>{...}</Answer>:   agent answer tag wrappers
    Returns the parsed dict, or the original string if no dict found.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    # Strip <Answer>...</Answer> tags
    m = re.search(r'<Answer>\s*(.+?)\s*</Answer>', text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)
    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    m = re.search(r'```(?:json|JSON)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if m:
        text = m.group(1)
    text = text.strip()
    # Try direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, TypeError):
        pass
    # Find balanced {...} block
    depth = 0
    start = None
    for i, c in enumerate(text):
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    parsed = json.loads(text[start:i+1])
                    if isinstance(parsed, dict):
                        return parsed
                except (ValueError, TypeError):
                    start = None
    return text


def load_predictions(run_dir: Path, gold_data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """Merge ``<run_dir>/<task>/benchmark.jsonl`` records into copies of the gold samples.

    Each record's ``question_id`` (``<sample_id>_task<i>``) selects the task; its
    ``predicted_answer`` becomes ``final_answer`` (strings are parsed for JSON) and its
    ``trajectory`` replaces the gold one. Unanswered tasks keep an empty answer/trajectory.
    """
    pred_data = {}
    for dim, task in TASKS.items():
        jsonl_path = Path(run_dir) / task / "benchmark.jsonl"
        if dim not in gold_data or not jsonl_path.exists():
            if dim in gold_data:
                print(f"  WARNING: {jsonl_path} not found, skipping {dim}")
            continue
        pred_samples = copy.deepcopy(gold_data[dim])
        lookup = {}
        for s in pred_samples:
            for ti, t in enumerate(s.get("tasks", [])):
                t["final_answer"], t["trajectory"] = {}, []
                lookup[(s.get("sample_id", ""), ti)] = t

        n_matched = 0
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            qid, sid = record.get("question_id", ""), record.get("sample_id", "")
            parts = qid.rsplit("_task", 1)
            if len(parts) == 2:
                try:
                    ti = int(parts[1])
                except ValueError:
                    ti = 0
                sid = sid or parts[0]
            else:
                ti, sid = 0, sid or qid
            target = lookup.get((sid, ti))
            if target is None:
                continue
            answer = record.get("predicted_answer", record.get("final_answer"))
            if isinstance(answer, dict):
                target["final_answer"] = answer
            elif answer is not None:
                parsed = _parse_string_prediction(str(answer))
                target["final_answer"] = parsed if isinstance(parsed, dict) else {"_raw": str(answer)}
            if record.get("trajectory"):
                target["trajectory"] = record["trajectory"]
            n_matched += 1
        pred_data[dim] = pred_samples
        print(f"  Loaded {dim}: {n_matched} predictions matched to {len(lookup)} tasks")
    return pred_data


def run_evaluation(tasks_dir: Path, run_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Evaluate a run directory (or the gold trajectories themselves when ``run_dir`` is None)."""
    gold_data = load_gold(tasks_dir)
    pred_data = None if run_dir is None else load_predictions(run_dir, gold_data)
    roots = (Path(tasks_dir),) + (() if run_dir is None else (Path(run_dir),))

    results, dim_scores = {}, []
    for dim in TASKS:
        # Dimensions without predictions are skipped (never scored gold-vs-gold).
        if dim not in gold_data or (pred_data is not None and dim not in pred_data):
            continue
        pred = None if pred_data is None else pred_data[dim]
        results[dim] = evaluate_dimension(gold_data[dim], pred, polygon_roots=roots)
        if results[dim]["n_tasks"] > 0:
            dim_scores.append(results[dim]["dim_score"])
    if len(results) < len(TASKS):
        warnings.warn(f"only {sorted(results)} evaluated: the DORA score averages these dimensions only")

    dora_score = sum(dim_scores) / len(dim_scores) if dim_scores else 0.0
    return {
        "dora_score": round(dora_score, 6),
        "dimensions": {d: {k: v for k, v in r.items() if k != "tasks"} for d, r in results.items()},
        "full": results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def format_results_table(results: Dict[str, Any]) -> str:
    """Render the 5-dim summary as a plain-text pretty table (same format
    used for stdout). Returned string is suitable for embedding in JSON or
    writing to a sibling .txt file. Uses 6-decimal precision."""
    dims = results["dimensions"]

    # 6-decimal fields need width 9 (e.g. "0.376812"), Answer gets a bit more.
    header = f"{'Dim':<6} {'Tasks':>6} {'Answer':>10} {'T-Any':>9} {'T-Ord':>9} {'T-EM':>9} {'ParAcc':>9} {'Eff':>9}"
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for dim in TASKS:
        if dim not in dims:
            continue
        d = dims[dim]
        lines.append(
            f"{dim:<6} {d['n_tasks']:>6} {d['dim_score']:>10.6f} "
            f"{d['t_any']:>9.6f} {d['t_ord']:>9.6f} {d['t_em']:>9.6f} "
            f"{d['par_acc']:>9.6f} {d['eff']:>9.6f}"
        )

    lines.append(sep)

    # Aggregate trajectory metrics (macro-average over dims with tasks > 0)
    active_dims = [dims[d] for d in TASKS if d in dims and dims[d]["n_tasks"] > 0]
    n_d = len(active_dims)
    if n_d > 0:
        agg = {m: sum(d[m] for d in active_dims) / n_d
               for m in ("t_any", "t_ord", "t_em", "par_acc", "eff")}
        total_tasks = sum(d["n_tasks"] for d in active_dims)
        lines.append(
            f"{'DORA':<6} {total_tasks:>6} {results['dora_score']:>10.6f} "
            f"{agg['t_any']:>9.6f} {agg['t_ord']:>9.6f} {agg['t_em']:>9.6f} "
            f"{agg['par_acc']:>9.6f} {agg['eff']:>9.6f}"
        )
    lines.append(sep)
    lines.append("")
    lines.append(f"  DORA Score: {results['dora_score']:.6f}")
    return "\n".join(lines)
