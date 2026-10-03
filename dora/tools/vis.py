"""Vis MCP server: damage maps, charts, temporal panels and one-page report composition (``vis.*``)."""

import os
import json
import math
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._common import session_temp_dir
from ._types import (
    ListStr, ListListInt, DictStrFloat, OptListInt, OptListStr, OptListDict, OptListDictStrAny,
    OptDictStrStr, OptDictStrAny,
)

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fastmcp import FastMCP

mcp = FastMCP()
TEMP_DIR = session_temp_dir()


def _stub_result(tool: str, **inputs: Any) -> Dict[str, Any]:
    """Structured error result returned when a tool fails."""
    return {
        "tool": tool,
        "status": "stub",
        "note": "Replace this stub with a real model backend.",
        "inputs": inputs,
    }


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _hash_inputs(*args: Any) -> str:
    """Short content hash for building unique filenames per tool call."""
    payload = json.dumps(args, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]


def _auto_out(name: str, *inputs: Any) -> str:
    """Deterministic per-call output path under TEMP_DIR."""
    return str(TEMP_DIR / f"{name}_{_hash_inputs(*inputs)}.png")


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_mask(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _save_image(img: Image.Image, output_path: str) -> str:
    _ensure_parent(output_path)
    img.save(output_path)
    return output_path


def _default_font(size: int = 18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _get_text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        return draw.textsize(text, font=font)


def _blend_color(base_rgb: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out = (1 - alpha) * base_rgb.astype(np.float32) + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def _legend_display_color(
    color: Tuple[int, int, int],
    alpha: Optional[float] = None,
    background: Tuple[int, int, int] = (240, 240, 240),
) -> Tuple[int, int, int]:
    if alpha is None:
        return tuple(int(c) for c in color)
    base = np.array(background, dtype=np.uint8).reshape(1, 1, 3)
    blended = _blend_color(base, color, float(alpha))[0, 0]
    return tuple(int(v) for v in blended.tolist())


def _class_color(value: int) -> Tuple[int, int, int]:
    """
    Default palette for disaster classes.
    """
    palette = {
        1: (255, 215, 0),    # yellow
        2: (255, 140, 0),    # orange
        3: (220, 20, 60),    # crimson
        4: (128, 0, 128),    # purple
        5: (30, 144, 255),   # dodger blue
        6: (0, 191, 255),    # deep sky blue
    }
    return palette.get(int(value), (255, 0, 0))


def _draw_north_arrow(draw: ImageDraw.ImageDraw, x: int, y: int, size: int = 60) -> None:
    font = _default_font(max(14, size // 4))
    draw.line((x, y + size, x, y), fill=(0, 0, 0), width=3)
    draw.polygon(
        [(x, y), (x - size // 6, y + size // 4), (x + size // 6, y + size // 4)],
        fill=(0, 0, 0)
    )
    tw, th = _get_text_size(draw, "N", font)
    draw.text((x - tw // 2, y + size + 4), "N", fill=(0, 0, 0), font=font)


def _draw_scale_bar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    pixel_size_m: float,
    target_m: Optional[float] = None,
    width_px: int = 120
) -> None:
    """
    Draw a simple scale bar.
    If target_m is None, infer from width_px and pixel_size_m.
    """
    if pixel_size_m <= 0:
        return

    font = _default_font(14)

    if target_m is None:
        target_m = pixel_size_m * width_px

    if target_m >= 1000:
        label = f"{target_m / 1000:.1f} km"
    else:
        label = f"{int(round(target_m))} m"

    draw.rectangle((x, y, x + width_px, y + 10), fill=(0, 0, 0))
    draw.rectangle((x, y, x + width_px // 2, y + 10), fill=(255, 255, 255), outline=(0, 0, 0))
    draw.rectangle((x, y, x + width_px, y + 10), outline=(0, 0, 0))
    tw, th = _get_text_size(draw, label, font)
    draw.text((x + width_px // 2 - tw // 2, y + 14), label, fill=(0, 0, 0), font=font)


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    items: List[Tuple[str, Tuple[int, int, int]]],
    x: int,
    y: int,
    box_size: int = 16,
    padding: int = 8
) -> Tuple[int, int]:
    font = _default_font(16)
    current_y = y
    max_w = 0
    for label, color in items:
        draw.rectangle((x, current_y, x + box_size, current_y + box_size), fill=color, outline=(0, 0, 0))
        draw.text((x + box_size + 8, current_y - 1), label, fill=(0, 0, 0), font=font)
        tw, th = _get_text_size(draw, label, font)
        max_w = max(max_w, box_size + 8 + tw)
        current_y += box_size + padding
    return max_w, current_y - y


def _draw_point_markers(
    draw: ImageDraw.ImageDraw,
    markers: List[Dict[str, Any]],
    offset_x: int = 0,
    offset_y: int = 0,
    canvas_size: Optional[Tuple[int, int]] = None,
) -> List[str]:
    font = _default_font(16)
    labels: List[str] = []
    canvas_w, canvas_h = canvas_size or (0, 0)

    for marker in markers:
        col = marker.get("pixel_col", marker.get("col", marker.get("x")))
        row = marker.get("pixel_row", marker.get("row", marker.get("y")))
        if col is None or row is None:
            continue

        x = int(round(float(col))) + offset_x
        y = int(round(float(row))) + offset_y
        radius = int(marker.get("radius", 7))
        color = tuple(marker.get("color", [0, 255, 255]))
        label = str(marker.get("label", "")).strip()

        draw.line((x - radius - 4, y, x + radius + 4, y), fill=(0, 0, 0), width=2)
        draw.line((x, y - radius - 4, x, y + radius + 4), fill=(0, 0, 0), width=2)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 0, 0), width=2)

        if label:
            labels.append(label)
            tw, th = _get_text_size(draw, label, font)
            label_x = x + radius + 10
            label_y = y - th - 6
            if canvas_w:
                label_x = min(label_x, max(4, canvas_w - tw - 8))
            if canvas_h:
                label_y = min(max(4, label_y), max(4, canvas_h - th - 8))
            draw.rectangle(
                (label_x - 4, label_y - 2, label_x + tw + 4, label_y + th + 2),
                fill=(255, 255, 255),
                outline=(0, 0, 0),
            )
            draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font)

    return labels


def _fit_image_with_padding(
    img: Image.Image,
    target_w: int,
    target_h: int,
    bg: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    canvas = Image.new("RGB", (target_w, target_h), bg)
    src_w, src_h = img.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = img.resize((new_w, new_h))
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas.paste(resized, (x, y))
    return canvas


def _normalize_pixel_point(point: Optional[List[int]]) -> Optional[Tuple[int, int]]:
    if not point or len(point) < 2:
        return None
    return int(round(float(point[0]))), int(round(float(point[1])))


def _normalize_path_pixels(path_pixels: Optional[List[List[int]]]) -> List[Tuple[int, int]]:
    normalized: List[Tuple[int, int]] = []
    for point in path_pixels or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        normalized.append((int(round(float(point[0]))), int(round(float(point[1])))))
    return normalized


# ---------------------------------------------------------------------
# Core visualization functions
# ---------------------------------------------------------------------

def vis_overlay_mask(
    image_path: str,
    mask_path: str,
    output_path: str,
    alpha: float = 0.45,
    class_labels: OptDictStrStr = None,
) -> Dict[str, Any]:
    base = _load_image(image_path)
    base_arr = np.array(base)
    mask = _load_mask(mask_path)

    # Some generated masks and source images are not pixel-aligned in size.
    # Resize the mask with nearest-neighbor so overlay rendering remains usable.
    if mask.shape[:2] != base_arr.shape[:2]:
        mask_img = Image.fromarray(mask.astype(np.uint8))
        mask = np.array(mask_img.resize((base_arr.shape[1], base_arr.shape[0]), Image.NEAREST))

    out = base_arr.copy()
    unique_vals = sorted([int(v) for v in np.unique(mask).tolist() if int(v) != 0])

    for v in unique_vals:
        region = mask == v
        if not np.any(region):
            continue
        color = np.array(_class_color(v), dtype=np.uint8)
        out[region] = ((1 - alpha) * out[region] + alpha * color).astype(np.uint8)

    out_img = Image.fromarray(out)
    _save_image(out_img, output_path)

    return {
        "tool": "vis.overlay_mask",
        "output_path": output_path,
        "classes_present": unique_vals,
        "class_labels": class_labels or {},
    }


def vis_add_map_elements(
    image_path: str,
    output_path: str,
    title: Optional[str] = None,
    legend_items: OptListDictStrAny = None,
    legend_alpha: Optional[float] = None,
    point_markers: OptListDict = None,
    north_arrow: bool = True,
    scale_bar: bool = True,
    pixel_size_m: Optional[float] = None,
) -> Dict[str, Any]:
    img = _load_image(image_path)
    w, h = img.size

    top_pad = 60 if title else 20
    right_pad = 220 if legend_items else 20
    bottom_pad = 60 if scale_bar else 20
    left_pad = 20

    canvas = Image.new("RGB", (w + left_pad + right_pad, h + top_pad + bottom_pad), (255, 255, 255))
    canvas.paste(img, (left_pad, top_pad))

    draw = ImageDraw.Draw(canvas)

    if title:
        font = _default_font(24)
        tw, th = _get_text_size(draw, title, font)
        draw.text(((canvas.size[0] - tw) // 2, 15), title, fill=(0, 0, 0), font=font)

    if north_arrow:
        _draw_north_arrow(draw, canvas.size[0] - right_pad + 60, top_pad + 20, size=60)

    if scale_bar and pixel_size_m is not None and pixel_size_m > 0:
        _draw_scale_bar(draw, left_pad + 20, top_pad + h + 15, pixel_size_m=pixel_size_m)

    if legend_items:
        items = []
        for it in legend_items:
            label = str(it.get("label", "item"))
            raw_color = tuple(it.get("color", [255, 0, 0]))
            color = _legend_display_color(raw_color, legend_alpha)
            items.append((label, color))
        draw.text((w + left_pad + 20, top_pad + 110), "Legend", fill=(0, 0, 0), font=_default_font(18))
        _draw_legend(draw, items, w + left_pad + 20, top_pad + 140)

    marker_labels: List[str] = []
    if point_markers:
        marker_labels = _draw_point_markers(
            draw,
            point_markers,
            offset_x=left_pad,
            offset_y=top_pad,
            canvas_size=canvas.size,
        )

    _save_image(canvas, output_path)
    return {
        "tool": "vis.add_map_elements",
        "output_path": output_path,
        "title": title,
        "has_legend": bool(legend_items),
        "legend_alpha": legend_alpha,
        "point_marker_count": len(point_markers or []),
        "point_marker_labels": marker_labels,
        "north_arrow": north_arrow,
        "scale_bar": scale_bar and (pixel_size_m is not None),
    }


def vis_damage_map(
    base_image_path: str,
    overlay_mask_path: Optional[str],
    output_path: str,
    title: str = "Damage Assessment Map",
    pixel_size_m: Optional[float] = None,
    class_labels: OptDictStrStr = None,
    alpha: float = 0.45,
) -> Dict[str, Any]:
    temp_overlay = str(TEMP_DIR / f"overlay_{Path(output_path).stem}.png")
    if overlay_mask_path:
        overlay_info = vis_overlay_mask(
            image_path=base_image_path,
            mask_path=overlay_mask_path,
            output_path=temp_overlay,
            alpha=alpha,
            class_labels=class_labels,
        )
        image_for_layout = overlay_info["output_path"]
        classes_present = overlay_info["classes_present"]
    else:
        image_for_layout = base_image_path
        classes_present = []

    legend_items = []
    for v in classes_present:
        label = (class_labels or {}).get(str(v), f"class_{v}")
        legend_items.append({
            "label": label,
            "color": list(_class_color(v))
        })

    result = vis_add_map_elements(
        image_path=image_for_layout,
        output_path=output_path,
        title=title,
        legend_items=legend_items,
        legend_alpha=alpha if overlay_mask_path else None,
        north_arrow=True,
        scale_bar=True,
        pixel_size_m=pixel_size_m,
    )
    result["tool"] = "vis.damage_map"
    result["layers"] = ["base_image"] + (["overlay_mask"] if overlay_mask_path else [])
    return result


def vis_route_map(
    base_image_path: str,
    path_pixels: ListListInt,
    output_path: str,
    title: str = "Intact Route Map",
    pixel_size_m: Optional[float] = None,
    start_point: OptListInt = None,
    end_point: OptListInt = None,
    overlay_mask_path: Optional[str] = None,
    class_labels: OptDictStrStr = None,
    alpha: float = 0.35,
    route_color: OptListInt = None,
    route_width: int = 5,
) -> Dict[str, Any]:
    normalized_path = _normalize_path_pixels(path_pixels)
    route_rgb = tuple(int(v) for v in (route_color or [0, 191, 255]))

    temp_overlay = str(TEMP_DIR / f"route_overlay_{Path(output_path).stem}.png")
    classes_present: List[int] = []
    if overlay_mask_path:
        overlay_info = vis_overlay_mask(
            image_path=base_image_path,
            mask_path=overlay_mask_path,
            output_path=temp_overlay,
            alpha=alpha,
            class_labels=class_labels,
        )
        image_for_layout = overlay_info["output_path"]
        classes_present = overlay_info["classes_present"]
    else:
        image_for_layout = base_image_path

    img = _load_image(image_for_layout)
    w, h = img.size

    top_pad = 60 if title else 20
    right_pad = 260
    bottom_pad = 60 if pixel_size_m else 20
    left_pad = 20

    canvas = Image.new("RGB", (w + left_pad + right_pad, h + top_pad + bottom_pad), (255, 255, 255))
    canvas.paste(img, (left_pad, top_pad))
    draw = ImageDraw.Draw(canvas)

    if title:
        title_font = _default_font(24)
        tw, th = _get_text_size(draw, title, title_font)
        draw.text(((canvas.size[0] - tw) // 2, 15), title, fill=(0, 0, 0), font=title_font)

    _draw_north_arrow(draw, canvas.size[0] - right_pad + 60, top_pad + 20, size=60)
    if pixel_size_m is not None and pixel_size_m > 0:
        _draw_scale_bar(draw, left_pad + 20, top_pad + h + 15, pixel_size_m=pixel_size_m)

    if len(normalized_path) >= 2:
        route_xy = [(left_pad + col, top_pad + row) for row, col in normalized_path]
        draw.line(route_xy, fill=(0, 0, 0), width=max(1, route_width + 2))
        draw.line(route_xy, fill=route_rgb, width=max(1, route_width))

    route_start = _normalize_pixel_point(start_point) or (normalized_path[0] if normalized_path else None)
    route_end = _normalize_pixel_point(end_point) or (normalized_path[-1] if normalized_path else None)
    point_markers: List[Dict[str, Any]] = []
    if route_start:
        point_markers.append({
            "pixel_row": route_start[0],
            "pixel_col": route_start[1],
            "label": "Route start",
            "radius": 8,
            "color": [46, 204, 113],
        })
    if route_end:
        point_markers.append({
            "pixel_row": route_end[0],
            "pixel_col": route_end[1],
            "label": "Route end",
            "radius": 8,
            "color": [231, 76, 60],
        })
    marker_labels = _draw_point_markers(
        draw,
        point_markers,
        offset_x=left_pad,
        offset_y=top_pad,
        canvas_size=canvas.size,
    )

    legend_items: List[Tuple[str, Tuple[int, int, int]]] = []
    for v in classes_present:
        label = (class_labels or {}).get(str(v), f"class_{v}")
        legend_items.append((label, _legend_display_color(_class_color(v), alpha)))
    legend_items.extend([
        ("intact route", route_rgb),
        ("route start", (46, 204, 113)),
        ("route end", (231, 76, 60)),
    ])
    draw.text((w + left_pad + 20, top_pad + 110), "Legend", fill=(0, 0, 0), font=_default_font(18))
    _draw_legend(draw, legend_items, w + left_pad + 20, top_pad + 140)

    _save_image(canvas, output_path)
    return {
        "tool": "vis.route_map",
        "output_path": output_path,
        "path_point_count": len(normalized_path),
        "start_point": list(route_start) if route_start else None,
        "end_point": list(route_end) if route_end else None,
        "point_marker_labels": marker_labels,
        "has_overlay": bool(overlay_mask_path),
        "legend_items": [label for label, _ in legend_items],
    }


def vis_pie_chart(
    data: DictStrFloat,
    output_path: str,
    title: str = "Pie Chart",
    size: Tuple[int, int] = (800, 600),
) -> Dict[str, Any]:
    w, h = size
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_font = _default_font(28)
    font = _default_font(18)

    tw, th = _get_text_size(draw, title, title_font)
    draw.text(((w - tw) // 2, 20), title, fill=(0, 0, 0), font=title_font)

    values = [float(v) for v in data.values()]
    labels = list(data.keys())
    total = sum(values) if sum(values) > 0 else 1.0

    cx, cy = 280, 330
    radius = 180
    bbox = (cx - radius, cy - radius, cx + radius, cy + radius)

    start = 0.0
    legend_items = []
    for idx, (label, value) in enumerate(zip(labels, values)):
        frac = value / total
        end = start + frac * 360.0
        color = _class_color(idx + 1)
        draw.pieslice(bbox, start=start, end=end, fill=color, outline=(255, 255, 255))
        legend_items.append((f"{label}: {value}", color))
        start = end

    draw.text((540, 140), "Legend", fill=(0, 0, 0), font=_default_font(20))
    _draw_legend(draw, legend_items, 540, 180, box_size=18, padding=12)

    _save_image(img, output_path)
    return {
        "tool": "vis.pie_chart",
        "output_path": output_path,
        "chart_type": "pie",
        "statistics": data,
        "title": title,
    }


def vis_bar_chart(
    data: DictStrFloat,
    output_path: str,
    title: str = "Bar Chart",
    size: Tuple[int, int] = (900, 600),
) -> Dict[str, Any]:
    w, h = size
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_font = _default_font(28)
    font = _default_font(16)

    tw, th = _get_text_size(draw, title, title_font)
    draw.text(((w - tw) // 2, 20), title, fill=(0, 0, 0), font=title_font)

    left, top, right, bottom = 90, 100, w - 40, h - 100
    draw.line((left, bottom, right, bottom), fill=(0, 0, 0), width=2)
    draw.line((left, bottom, left, top), fill=(0, 0, 0), width=2)

    labels = list(data.keys())
    values = [float(v) for v in data.values()]
    max_val = max(values) if values else 1.0
    if max_val <= 0:
        max_val = 1.0

    n = max(1, len(labels))
    bar_space = (right - left) / n
    bar_w = int(bar_space * 0.55)

    for i, (label, value) in enumerate(zip(labels, values)):
        x0 = int(left + i * bar_space + (bar_space - bar_w) / 2)
        x1 = x0 + bar_w
        bh = int((value / max_val) * (bottom - top - 20))
        y0 = bottom - bh
        color = _class_color(i + 1)
        draw.rectangle((x0, y0, x1, bottom), fill=color, outline=(0, 0, 0))

        val_text = str(value)
        vtw, vth = _get_text_size(draw, val_text, font)
        draw.text((x0 + (bar_w - vtw) // 2, y0 - vth - 4), val_text, fill=(0, 0, 0), font=font)

        short_label = label[:16]
        ltw, lth = _get_text_size(draw, short_label, font)
        draw.text((x0 + (bar_w - ltw) // 2, bottom + 8), short_label, fill=(0, 0, 0), font=font)

    _save_image(img, output_path)
    return {
        "tool": "vis.bar_chart",
        "output_path": output_path,
        "chart_type": "bar",
        "statistics": data,
        "title": title,
    }


def vis_line_chart(
    data: DictStrFloat,
    output_path: str,
    title: str = "Line Chart",
    size: Tuple[int, int] = (900, 600),
) -> Dict[str, Any]:
    w, h = size
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_font = _default_font(28)
    font = _default_font(16)

    tw, th = _get_text_size(draw, title, title_font)
    draw.text(((w - tw) // 2, 20), title, fill=(0, 0, 0), font=title_font)

    left, top, right, bottom = 90, 100, w - 40, h - 100
    draw.line((left, bottom, right, bottom), fill=(0, 0, 0), width=2)
    draw.line((left, bottom, left, top), fill=(0, 0, 0), width=2)

    xs = list(data.keys())
    ys = [float(v) for v in data.values()]
    max_y = max(ys) if ys else 1.0
    if max_y <= 0:
        max_y = 1.0

    n = max(1, len(xs))
    points = []
    for i, (x_label, y_val) in enumerate(zip(xs, ys)):
        px = int(left + i * (right - left) / max(1, n - 1))
        py = int(bottom - (y_val / max_y) * (bottom - top - 10))
        points.append((px, py))

    if len(points) >= 2:
        draw.line(points, fill=(30, 144, 255), width=3)

    for i, ((px, py), x_label, y_val) in enumerate(zip(points, xs, ys)):
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=(220, 20, 60), outline=(0, 0, 0))
        vt = str(y_val)
        draw.text((px - 10, py - 24), vt, fill=(0, 0, 0), font=font)
        xtw, xth = _get_text_size(draw, x_label, font)
        draw.text((px - xtw // 2, bottom + 8), x_label, fill=(0, 0, 0), font=font)

    _save_image(img, output_path)
    return {
        "tool": "vis.line_chart",
        "output_path": output_path,
        "chart_type": "line",
        "statistics": data,
        "title": title,
    }


def vis_before_after_panel(
    pre_image_path: str,
    post_image_path: str,
    output_path: str,
    title: str = "Before / After Comparison",
    panel_size: Tuple[int, int] = (700, 500),
) -> Dict[str, Any]:
    panel_w, panel_h = panel_size
    canvas = Image.new("RGB", (panel_w * 2 + 60, panel_h + 120), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    title_font = _default_font(28)
    font = _default_font(20)

    tw, th = _get_text_size(draw, title, title_font)
    draw.text(((canvas.size[0] - tw) // 2, 20), title, fill=(0, 0, 0), font=title_font)

    pre = _fit_image_with_padding(_load_image(pre_image_path), panel_w, panel_h)
    post = _fit_image_with_padding(_load_image(post_image_path), panel_w, panel_h)

    canvas.paste(pre, (20, 80))
    canvas.paste(post, (40 + panel_w, 80))

    draw.text((20 + panel_w // 2 - 20, 50), "Pre", fill=(0, 0, 0), font=font)
    draw.text((40 + panel_w + panel_w // 2 - 25, 50), "Post", fill=(0, 0, 0), font=font)

    _save_image(canvas, output_path)
    return {
        "tool": "vis.before_after_panel",
        "output_path": output_path,
        "title": title,
    }


def vis_temporal_panel(
    image_paths: ListStr,
    output_path: str,
    labels: OptListStr = None,
    title: str = "Temporal Panel",
    columns: int = 2,
    panel_size: Tuple[int, int] = (420, 320),
) -> Dict[str, Any]:
    labels = labels or [f"t{i + 1}" for i in range(len(image_paths))]
    panel_w, panel_h = panel_size
    n = len(image_paths)
    cols = max(1, columns)
    rows = int(math.ceil(n / cols))

    canvas_w = cols * panel_w + (cols + 1) * 20
    canvas_h = rows * panel_h + (rows + 1) * 40 + 60

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    title_font = _default_font(28)
    font = _default_font(18)

    tw, th = _get_text_size(draw, title, title_font)
    draw.text(((canvas_w - tw) // 2, 20), title, fill=(0, 0, 0), font=title_font)

    for idx, (path, label) in enumerate(zip(image_paths, labels)):
        r = idx // cols
        c = idx % cols
        x = 20 + c * panel_w + c * 20
        y = 70 + r * panel_h + r * 40

        panel = _fit_image_with_padding(_load_image(path), panel_w, panel_h)
        canvas.paste(panel, (x, y))
        draw.rectangle((x, y, x + panel_w, y + panel_h), outline=(0, 0, 0), width=1)
        draw.text((x + 8, y + 8), label, fill=(0, 0, 0), font=font)

    _save_image(canvas, output_path)
    return {
        "tool": "vis.temporal_panel",
        "output_path": output_path,
        "labels": labels,
        "title": title,
    }


def vis_report_page(
    map_image_path: Optional[str],
    chart_image_path: Optional[str],
    output_path: str,
    summary_text: str = "",
    key_statistics: OptDictStrAny = None,
    title: str = "Disaster Situation Report",
    page_size: Tuple[int, int] = (1400, 1000),
) -> Dict[str, Any]:
    page_w, page_h = page_size
    canvas = Image.new("RGB", (page_w, page_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    title_font = _default_font(34)
    section_font = _default_font(22)
    body_font = _default_font(18)

    tw, th = _get_text_size(draw, title, title_font)
    draw.text(((page_w - tw) // 2, 20), title, fill=(0, 0, 0), font=title_font)

    # Layout
    map_box = (40, 90, 840, 620)
    chart_box = (880, 90, 1360, 440)
    summary_box = (880, 470, 1360, 760)
    stats_box = (40, 660, 1360, 940)

    draw.rectangle(map_box, outline=(0, 0, 0), width=2)
    draw.rectangle(chart_box, outline=(0, 0, 0), width=2)
    draw.rectangle(summary_box, outline=(0, 0, 0), width=2)
    draw.rectangle(stats_box, outline=(0, 0, 0), width=2)

    draw.text((map_box[0] + 10, map_box[1] + 8), "Map", fill=(0, 0, 0), font=section_font)
    draw.text((chart_box[0] + 10, chart_box[1] + 8), "Chart", fill=(0, 0, 0), font=section_font)
    draw.text((summary_box[0] + 10, summary_box[1] + 8), "Summary", fill=(0, 0, 0), font=section_font)
    draw.text((stats_box[0] + 10, stats_box[1] + 8), "Key Statistics", fill=(0, 0, 0), font=section_font)

    if map_image_path and os.path.exists(map_image_path):
        map_img = _fit_image_with_padding(
            _load_image(map_image_path),
            map_box[2] - map_box[0] - 20,
            map_box[3] - map_box[1] - 50
        )
        canvas.paste(map_img, (map_box[0] + 10, map_box[1] + 40))

    if chart_image_path and os.path.exists(chart_image_path):
        chart_img = _fit_image_with_padding(
            _load_image(chart_image_path),
            chart_box[2] - chart_box[0] - 20,
            chart_box[3] - chart_box[1] - 50
        )
        canvas.paste(chart_img, (chart_box[0] + 10, chart_box[1] + 40))

    # Summary text
    def wrap_text(text: str, max_chars: int = 48) -> List[str]:
        if not text:
            return []
        words = text.split()
        lines = []
        current = []
        current_len = 0
        for w in words:
            add_len = len(w) + (1 if current else 0)
            if current_len + add_len <= max_chars:
                current.append(w)
                current_len += add_len
            else:
                lines.append(" ".join(current))
                current = [w]
                current_len = len(w)
        if current:
            lines.append(" ".join(current))
        return lines

    y = summary_box[1] + 45
    for line in wrap_text(summary_text, max_chars=48):
        draw.text((summary_box[0] + 12, y), line, fill=(0, 0, 0), font=body_font)
        y += 28

    # Stats
    stats = key_statistics or {}
    sy = stats_box[1] + 45
    sx = stats_box[0] + 14
    for k, v in stats.items():
        line = f"{k}: {v}"
        draw.text((sx, sy), line, fill=(0, 0, 0), font=body_font)
        sy += 28
        if sy > stats_box[3] - 30:
            sy = stats_box[1] + 45
            sx += 420

    _save_image(canvas, output_path)
    return {
        "tool": "vis.report_page",
        "output_path": output_path,
        "title": title,
        "components": {
            "map": bool(map_image_path),
            "chart": bool(chart_image_path),
            "summary_text": bool(summary_text),
            "key_statistics": bool(key_statistics),
        }
    }


# ---------------------------------------------------------------------
# MCP wrappers
# ---------------------------------------------------------------------

@mcp.tool(name="vis.overlay_mask", description="""
Overlay a segmentation mask on top of a base image.
Inputs:
- image_path: base raster image
- mask_path: segmentation mask path
- alpha: overlay opacity
- class_labels: optional class label dict
Output file is written to a managed location; its path is returned in the result dict as output_path.
""")
def overlay_mask_tool(
    image_path: str,
    mask_path: str,
    alpha: float = 0.45,
    class_labels: OptDictStrStr = None,
) -> str:
    output_path = _auto_out("overlay_mask", image_path, mask_path, alpha, class_labels)
    return json.dumps(
        vis_overlay_mask(image_path, mask_path, output_path, alpha, class_labels),
        ensure_ascii=False, indent=2
    )


@mcp.tool(name="vis.add_map_elements", description="""
Add title, legend, north arrow, and scale bar to an image.
Output file is written to a managed location; its path is returned in the result dict as output_path.
""")
def add_map_elements_tool(
    image_path: str,
    title: Optional[str] = None,
    legend_items: OptListDictStrAny = None,
    legend_alpha: Optional[float] = None,
    point_markers: OptListDict = None,
    north_arrow: bool = True,
    scale_bar: bool = True,
    pixel_size_m: Optional[float] = None,
) -> str:
    output_path = _auto_out(
        "map_with_elements",
        image_path, title, legend_items, legend_alpha,
        point_markers, north_arrow, scale_bar, pixel_size_m,
    )
    return json.dumps(
        vis_add_map_elements(
            image_path,
            output_path,
            title,
            legend_items,
            legend_alpha,
            point_markers,
            north_arrow,
            scale_bar,
            pixel_size_m,
        ),
        ensure_ascii=False, indent=2
    )


@mcp.tool(name="vis.damage_map", description="""
Generate a standard damage assessment map.
Output file is written to a managed location; its path is returned in the result dict as output_path.
""")
def damage_map_tool(
    base_image_path: str,
    overlay_mask_path: Optional[str],
    title: str = "Damage Assessment Map",
    pixel_size_m: Optional[float] = None,
    class_labels: OptDictStrStr = None,
    alpha: float = 0.45,
) -> str:
    output_path = _auto_out(
        "damage_map",
        base_image_path, overlay_mask_path, title, pixel_size_m, class_labels, alpha,
    )
    return json.dumps(
        vis_damage_map(base_image_path, overlay_mask_path, output_path, title, pixel_size_m, class_labels, alpha),
        ensure_ascii=False, indent=2
    )


@mcp.tool(name="vis.pie_chart", description="""
Generate a pie chart from category statistics.
Output file is written to a managed location; its path is returned in the result dict as output_path.
""")
def pie_chart_tool(
    data: DictStrFloat,
    title: str = "Pie Chart",
) -> str:
    output_path = _auto_out("pie_chart", data, title)
    return json.dumps(
        vis_pie_chart(data, output_path, title),
        ensure_ascii=False, indent=2
    )


@mcp.tool(name="vis.route_map", description="""
Render a shortest-path route on a base image, optionally with a damage-mask overlay.
Output file is written to a managed location; its path is returned in the result dict as output_path.
""")
def route_map_tool(
    base_image_path: str,
    path_pixels: ListListInt,
    title: str = "Intact Route Map",
    pixel_size_m: Optional[float] = None,
    start_point: OptListInt = None,
    end_point: OptListInt = None,
    overlay_mask_path: Optional[str] = None,
    class_labels: OptDictStrStr = None,
    alpha: float = 0.35,
    route_color: OptListInt = None,
    route_width: int = 5,
) -> str:
    output_path = _auto_out(
        "route_map",
        base_image_path, path_pixels, title, pixel_size_m,
        start_point, end_point, overlay_mask_path, class_labels,
        alpha, route_color, route_width,
    )
    return json.dumps(
        vis_route_map(
            base_image_path,
            path_pixels,
            output_path,
            title,
            pixel_size_m,
            start_point,
            end_point,
            overlay_mask_path,
            class_labels,
            alpha,
            route_color,
            route_width,
        ),
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool(name="vis.bar_chart", description="""
Generate a bar chart from category statistics.
Output file is written to a managed location; its path is returned in the result dict as output_path.
""")
def bar_chart_tool(
    data: DictStrFloat,
    title: str = "Bar Chart",
) -> str:
    output_path = _auto_out("bar_chart", data, title)
    return json.dumps(
        vis_bar_chart(data, output_path, title),
        ensure_ascii=False, indent=2
    )


@mcp.tool(name="vis.line_chart", description="""
Generate a line chart from ordered statistics.
Output file is written to a managed location; its path is returned in the result dict as output_path.
""")
def line_chart_tool(
    data: DictStrFloat,
    title: str = "Line Chart",
) -> str:
    output_path = _auto_out("line_chart", data, title)
    return json.dumps(
        vis_line_chart(data, output_path, title),
        ensure_ascii=False, indent=2
    )


@mcp.tool(name="vis.before_after_panel", description="""
Generate a before/after comparison panel.
Output file is written to a managed location; its path is returned in the result dict as output_path.
""")
def before_after_panel_tool(
    pre_image_path: str,
    post_image_path: str,
    title: str = "Before / After Comparison",
) -> str:
    output_path = _auto_out("before_after_panel", pre_image_path, post_image_path, title)
    return json.dumps(
        vis_before_after_panel(pre_image_path, post_image_path, output_path, title),
        ensure_ascii=False, indent=2
    )


@mcp.tool(name="vis.temporal_panel", description="""
Generate a multi-temporal panel from multiple image paths.
Output file is written to a managed location; its path is returned in the result dict as output_path.
""")
def temporal_panel_tool(
    image_paths: ListStr,
    labels: OptListStr = None,
    title: str = "Temporal Panel",
    columns: int = 2,
) -> str:
    output_path = _auto_out("temporal_panel", image_paths, labels, title, columns)
    return json.dumps(
        vis_temporal_panel(image_paths, output_path, labels, title, columns),
        ensure_ascii=False, indent=2
    )


@mcp.tool(name="vis.report_page", description="""
Compose a one-page disaster report from map, chart, summary, and statistics.
Output file is written to a managed location; its path is returned in the result dict as output_path.
""")
def report_page_tool(
    map_image_path: Optional[str],
    chart_image_path: Optional[str],
    summary_text: str = "",
    key_statistics: OptDictStrAny = None,
    title: str = "Disaster Situation Report",
) -> str:
    output_path = _auto_out(
        "report_page",
        map_image_path, chart_image_path, summary_text, key_statistics, title,
    )
    return json.dumps(
        vis_report_page(map_image_path, chart_image_path, output_path, summary_text, key_statistics, title),
        ensure_ascii=False, indent=2
    )


# ============================================================
#  VISUALIZATION EXTENSIONS
# ============================================================

@mcp.tool(name="vis.heatmap", description='''
Description:
Render a 2D heatmap visualizing the spatial density of input points or polygon
centroids. Uses Gaussian kernel density estimation and a color-mapped overlay.
If the input vector contains polygons (e.g., building damage outlines), their
centroids are computed automatically.

Parameters:
- points_path (str): Path to GeoJSON containing points, polygons, or mixed geometries
- base_image_path (str, optional): Background image to overlay on
- bandwidth (float, optional): Gaussian kernel bandwidth. Default auto.
- colormap (str, optional): Matplotlib colormap. Default "hot".

Output file is written to a managed location and returned as heatmap_path.

Returns:
- heatmap_path (str): Path to rendered heatmap image
- point_count (int): Number of points used (polygon centroids count as points)
''')
def vis_heatmap(
    points_path: str,
    base_image_path: Optional[str] = None,
    bandwidth: Optional[float] = None,
    colormap: str = "hot",
) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        gdf = gpd.read_file(points_path)
        if gdf.empty:
            raise ValueError("vis.heatmap: empty input")

        # Accept both points and polygons (compute centroids for polygons).
        coords: List[Tuple[float, float]] = []
        for g in gdf.geometry:
            if g is None or g.is_empty:
                continue
            if g.geom_type == "Point":
                coords.append((g.x, g.y))
            elif g.geom_type in ("Polygon", "MultiPolygon", "LineString", "MultiLineString"):
                c = g.centroid
                coords.append((c.x, c.y))
            elif g.geom_type == "MultiPoint":
                for p in g.geoms:
                    coords.append((p.x, p.y))
        pts = np.asarray(coords, dtype=float)
        if len(pts) < 2:
            raise ValueError("vis.heatmap: need at least 2 points/polygons")

        # Handle degenerate (all-identical) coordinates to avoid singular KDE.
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        if xmax - xmin < 1e-6:
            xmax = xmin + 1.0
        if ymax - ymin < 1e-6:
            ymax = ymin + 1.0

        kde = gaussian_kde(pts.T, bw_method=bandwidth)
        xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
        if base_image_path:
            try:
                from PIL import Image
                img = np.asarray(Image.open(base_image_path))
                ax.imshow(img, extent=[xmin, xmax, ymin, ymax], alpha=0.6)
            except Exception:
                pass
        ax.imshow(
            np.rot90(zz), extent=[xmin, xmax, ymin, ymax],
            cmap=colormap, alpha=0.7,
        )
        ax.scatter(pts[:, 0], pts[:, 1], s=5, c="white", alpha=0.4)
        ax.set_title("Damage Density Heatmap")
        ax.set_axis_off()

        out_path = _auto_out("heatmap", points_path, base_image_path, bandwidth, colormap)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        stats = {
            "kde_max": float(zz.max()),
            "kde_mean": float(zz.mean()),
            "extent": [float(xmin), float(ymin), float(xmax), float(ymax)],
        }
        return json.dumps({
            "tool": "vis.heatmap",
            "heatmap_path": out_path,
            "point_count": int(len(pts)),
            "stats": stats,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result(
            "vis.heatmap", error=str(e), points_path=points_path,
        ))


if __name__ == "__main__":
    mcp.run(show_banner=False)
