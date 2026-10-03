"""Perception MCP server: segmentation models (``seg.*`` tools).

The server uses one device for its whole lifetime: ``cuda`` (the first GPU visible
through ``CUDA_VISIBLE_DEVICES``) when available, otherwise the CPU. The benchmark
runner pins each worker's server to its own GPU.
"""
import gc
import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from albumentations import Compose, Normalize
from fastmcp import FastMCP
from PIL import Image
from skimage.io import imread

from ..paths import CHECKPOINTS_DIR, IMAGES_DIR
from ._common import session_temp_dir, temp_dir_arg
from .segmentation import LARGE_IMAGE_PIXELS, load_model, predict_labels_out_of_core, predict_probs

mcp = FastMCP()
TEMP_DIR = session_temp_dir()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)
DPT_PATCH = 518
_MODEL_CACHE: "OrderedDict[str, torch.nn.Module]" = OrderedDict()


# ============== Helpers ==============

def _tool_meta(forward_ms=None) -> Dict[str, Any]:
    """Execution metadata attached to model-backed tool outputs."""
    question_id = None
    session_file = Path(temp_dir_arg()) / ".session"
    if session_file.is_file():
        try:
            question_id = session_file.read_text().strip() or None
        except Exception:
            pass
    worker = re.search(r"(?:^|/|\\)workers(?:/|\\)w(\d+)(?:/|\\|$)", temp_dir_arg())
    meta = {
        "question_id": question_id,
        "worker_id": int(worker.group(1)) if worker else 0,
        "pid": os.getpid(),
        "gpu_slot": int(os.environ.get("DORA_GPU_SLOT", "0")) if DEVICE.type == "cuda" else "cpu",
        "lock_wait_ms": 0.0,
    }
    if forward_ms is not None:
        meta["forward_ms"] = round(float(forward_ms), 2)
    return meta


def _attach_tool_meta(result, forward_ms=None):
    result["_meta"] = _tool_meta(forward_ms)
    return result


def _get_model(name: str) -> torch.nn.Module:
    """Load a zoo model (kept on CPU between calls); only one model is cached at a time."""
    model = _MODEL_CACHE.get(name)
    if model is not None:
        _MODEL_CACHE.move_to_end(name)
        return model
    while _MODEL_CACHE:
        _, old_model = _MODEL_CACHE.popitem(last=False)
        del old_model
        gc.collect()
        torch.cuda.empty_cache()
    model = load_model(name, CHECKPOINTS_DIR)
    _MODEL_CACHE[name] = model
    return model


def _run_on_device(model, fn):
    """Move ``model`` to the server device for ``fn(model)``, then back to CPU; returns (output, ms)."""
    started = time.perf_counter()
    model.to(DEVICE)
    try:
        with torch.inference_mode():
            out = fn(model)
    finally:
        model.cpu()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return out, (time.perf_counter() - started) * 1000.0


def _save_mask(mask: np.ndarray, full_path: Path, reference_raster: str = None) -> Path:
    """Save a mask as GeoTIFF when ``reference_raster`` carries a CRS, otherwise as PNG."""
    full_path.parent.mkdir(parents=True, exist_ok=True)
    if reference_raster:
        try:
            import rasterio as rio

            with rio.open(reference_raster) as src:
                if src.crs is not None:
                    tif_path = full_path.with_suffix(".tif")
                    with rio.open(tif_path, "w", driver="GTiff", height=mask.shape[0], width=mask.shape[1],
                                  count=1, dtype="uint8", crs=src.crs, transform=src.transform) as dst:
                        dst.write(mask.astype(np.uint8), 1)
                    return tif_path
        except Exception:
            pass
    Image.fromarray(mask.astype(np.uint8)).save(full_path)
    return full_path


def _unique_name(base: str, *key_parts: str, ext: str = ".png") -> str:
    """Collision-free filename from ``base`` and the tool inputs."""
    raw = "|".join(str(p) for p in key_parts)
    return f"{base}_{hashlib.md5(raw.encode('utf-8')).hexdigest()[:6]}{ext}"


def _stub_result(tool: str, **inputs: Any) -> Dict[str, Any]:
    result = {
        "tool": tool,
        "status": "stub",
        "note": "Replace this stub with a real model backend.",
        "inputs": inputs,
    }
    if "error" in inputs:
        result["error"] = inputs["error"]
    return result


def _resolve_local_data_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path.resolve()
    for candidate in (IMAGES_DIR / path_str, IMAGES_DIR / "multi_temp" / path_str):
        if candidate.exists():
            return candidate.resolve()
    return path


def _rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.ndim < 2:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    return image[:, :, :3]


def _segment(model, image: torch.Tensor, kernel, stride, preprocess=None):
    """Label map for an ``H x W x C`` float image (out-of-core merge for very large scenes)."""
    kwargs = {"device": DEVICE} if preprocess is None else {"device": DEVICE, "preprocess": preprocess}
    if image.shape[0] * image.shape[1] > LARGE_IMAGE_PIXELS:
        return predict_labels_out_of_core(model, image, kernel, stride, **kwargs)
    return predict_probs(model, image, kernel, stride, **kwargs).argmax(dim=0).numpy()


def model_sliding_infer(model, pre_image_path, post_image_path):
    """Bi-temporal inference: pre/post RGB stacked on channels, 518 px windows."""
    bi_image = np.concatenate([_rgb(imread(pre_image_path)), _rgb(imread(post_image_path))], axis=2)
    normalize = Compose([Normalize(mean=IMAGENET_MEAN * 2, std=IMAGENET_STD * 2, max_pixel_value=1)])
    image = torch.from_numpy(normalize(image=bi_image)["image"]).float()
    del bi_image
    return _run_on_device(model, lambda m: _segment(m, image, DPT_PATCH, DPT_PATCH))


def single_temporal_model_sliding_infer(model, image_path):
    """Single-image inference; pads to a square multiple of the 518 px window, crops back."""
    image = _rgb(imread(image_path))
    h_orig, w_orig = image.shape[:2]
    pad_h = max(0, DPT_PATCH - h_orig) + (DPT_PATCH - h_orig % DPT_PATCH) % DPT_PATCH
    pad_w = max(0, DPT_PATCH - w_orig) + (DPT_PATCH - w_orig % DPT_PATCH) % DPT_PATCH
    h, w = h_orig + pad_h, w_orig + pad_w
    if h != w:
        side = max(h, w)
        pad_h, pad_w = side - h_orig, side - w_orig
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    normalize = Compose([Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1)])
    tensor = torch.from_numpy(normalize(image=image)["image"]).float()
    del image
    out, forward_ms = _run_on_device(model, lambda m: _segment(m, tensor, DPT_PATCH, DPT_PATCH))
    return out[:h_orig, :w_orig], forward_ms


def landslide4sense_sliding_infer(model, image_path):
    """14-band Landslide4Sense inference (Sentinel-2 + DEM + slope), 512 px windows, stride 256."""
    image = imread(image_path)
    if image.ndim != 3 or image.shape[2] != 14:
        raise ValueError(
            f"This tool requires 14-band input [H,W,14], "
            f"got shape {image.shape}. "
            f"Use ras.concat to merge Sentinel-2 (12 bands) + DEM (1) + Slope (1) first."
        )
    band_mean = [1111.81236406, 824.63171476, 663.41636217, 445.17289745, 645.8582926, 1547.73508126, 1960.44401001,
                 1941.32229668, 674.07572865, 9.04787384, 1113.98338755, 519.90397929, 20.29228266, 772.83144788]
    norm_mean = [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516,
                 0.3338, 0.7819]
    norm_std = [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018,
                1.2913]
    normalize = Compose([Normalize(mean=norm_mean, std=norm_std, max_pixel_value=1)])
    tensor = torch.from_numpy(normalize(image=image / band_mean)["image"]).float()
    return _run_on_device(
        model, lambda m: predict_probs(m, tensor, 512, 256, device=DEVICE).argmax(dim=0).numpy() * 255)


def _binary_mask(model_name: str, image: str, target_ids: set, out_filename: str,
                 classes_map: Dict[int, str] = None, note: str = None) -> Dict[str, Any]:
    """Run a single-image model and merge ``target_ids`` into one binary mask (value 255)."""
    out, forward_ms = single_temporal_model_sliding_infer(_get_model(model_name), image)
    mask = np.isin(out, list(target_ids)).astype(np.uint8) * 255
    full_path = TEMP_DIR / _unique_name(Path(out_filename).stem, image, str(sorted(target_ids)))
    _save_mask(mask, full_path)
    result = {"mask_path": str(full_path)}
    if classes_map is not None:
        result["classes"] = classes_map
    result["mask_value"] = 255
    if note:
        result["note"] = note
    return _attach_tool_meta(result, forward_ms)


def _water_mask(image: str, out_filename: str) -> Dict[str, Any]:
    out, forward_ms = single_temporal_model_sliding_infer(_get_model("water"), image)
    full_path = TEMP_DIR / _unique_name(Path(out_filename).stem, image)
    _save_mask(out.astype(np.uint8) * 255, full_path)
    return _attach_tool_meta({"mask_path": str(full_path), "mask_value": 255}, forward_ms)


# ============== Segmentation Tools ==============

@mcp.tool(name="seg.building_damage", description='''
Description:
Segment building damage from pre/post disaster optical satellite images.
Uses difference-based detection to classify buildings into damage levels.
Spatial resolution: 0.8m.
The input pre and post image requires 3-bands.

Parameters:
- pre_image_path (str): Path to pre-disaster image
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to the saved damage mask image
- classes (dict): Mapping of class values to damage levels
  * 1: intact
  * 2: partially damaged
  * 3: totally destroyed
''')
def seg_building_damage(pre_image_path: str, post_image_path: str) -> str:
    try:
        out, forward_ms = model_sliding_infer(_get_model("building_damage"), pre_image_path, post_image_path)
        full_path = TEMP_DIR / _unique_name("seg_building_damage_mask", pre_image_path, post_image_path)
        full_path = _save_mask(out, full_path, reference_raster=pre_image_path)
        result = {
            "tool": "seg.building_damage",
            "mask_path": str(full_path),
            "classes": {1: "intact", 2: "partially damaged", 3: "totally destroyed"},
        }
        return json.dumps(_attach_tool_meta(result, forward_ms), ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("seg.building_damage", error=str(e),
                                       pre_image_path=pre_image_path, post_image_path=post_image_path))


@mcp.tool(name="seg.building_damage_opt_sar", description='''
Description:
Segment building damage from pre-disaster optical and post-disaster SAR satellite images.
Uses difference-based detection to classify buildings into damage levels.
Spatial resolution: 0.8m.
The input pre and post image requires 3-bands.

Parameters:
- pre_image_path (str): Path to pre-disaster image
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to the saved damage mask image
- classes (dict): Mapping of class values to damage levels
  * 1: intact
  * 2: partially damaged
  * 3: totally destroyed
''')
def seg_building_damage_opt_sar(pre_image_path: str, post_image_path: str) -> str:
    try:
        out, forward_ms = model_sliding_infer(_get_model("building_damage_opt_sar"), pre_image_path, post_image_path)
        full_path = TEMP_DIR / _unique_name("seg_building_damage_optsar_mask", pre_image_path, post_image_path)
        _save_mask(out, full_path)
        result = {
            "tool": "seg.building_damage_opt_sar",
            "mask_path": str(full_path),
            "classes": {1: "intact", 2: "partially damaged", 3: "totally destroyed"},
        }
        return json.dumps(_attach_tool_meta(result, forward_ms), ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("seg.building_damage_opt_sar", error=str(e),
                                       pre_image_path=pre_image_path, post_image_path=post_image_path))


@mcp.tool(name="seg.lava", description='''
Description:
Segment volcano lava from pre/post disaster optical satellite images.
Spatial resolution: 0.8m.
The input pre and post image requires 3-bands.

Parameters:
- pre_image_path (str): Path to pre-disaster image
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to the saved lava mask image
- classes (dict): Mapping of class values
  * 0: non-lava
  * 255: lava
''')
def seg_lava(pre_image_path: str, post_image_path: str) -> str:
    try:
        out, forward_ms = model_sliding_infer(_get_model("lava"), pre_image_path, post_image_path)
        full_path = TEMP_DIR / _unique_name("seg_lava_mask", pre_image_path, post_image_path)
        full_path = _save_mask((out > 0).astype(np.uint8) * 255, full_path, reference_raster=pre_image_path)
        result = {
            "tool": "seg.lava",
            "mask_path": str(full_path),
            "classes": {0: "non-lava", 255: "lava"},
        }
        return json.dumps(_attach_tool_meta(result, forward_ms), ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("seg.lava", error=str(e), pre_image=pre_image_path, post_image=post_image_path))


@mcp.tool(name="seg.reconstruction_building", description='''
Description:
Segment buildings under reconstruction from a pre-disaster optical image and a
post-disaster recovery-phase optical image (bi-temporal change detection).
Spatial resolution: 0.8m.
The input pre and post image requires 3-bands.

Parameters:
- pre_image_path (str): Path to pre-disaster image
- post_image_path (str): Path to the recovery-phase image, e.g. *_phase2.png.

Returns:
- mask_path (str): Path to the saved building-under-reconstruction mask image
- classes (dict): Mapping of class values
  * 0: not_under_reconstruction
  * 255: building_under_reconstruction
''')
def seg_reconstruction_building(pre_image_path: str, post_image_path: str) -> str:
    try:
        pre_path = _resolve_local_data_path(pre_image_path)
        post_path = _resolve_local_data_path(post_image_path)
        out, forward_ms = model_sliding_infer(_get_model("reconstruction_building"), str(pre_path), str(post_path))
        full_path = TEMP_DIR / f"{post_path.stem}_reconstruction_building_mask.png"
        full_path = _save_mask((out > 0).astype(np.uint8) * 255, full_path, reference_raster=str(pre_path))
        result = {
            "tool": "seg.reconstruction_building",
            "mask_path": str(full_path),
            "classes": {0: "not_under_reconstruction", 255: "building_under_reconstruction"},
        }
        return json.dumps(_attach_tool_meta(result, forward_ms), ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("seg.reconstruction_building", error=str(e),
                                       pre_image=pre_image_path, post_image=post_image_path))


@mcp.tool(name="seg.landslide_ms", description='''
Description:
Segment landslide from post-disaster multi-modal data: multi-spectral optical + DEM + slope.
Spatial resolution: 10m.
IMPORTANT: The input image MUST have exactly 14 bands (12-band multi-spectral + DEM: 1 band + Slope: 1 band).
If your data has separate multi-spectral, DEM, and slope files, use ras.concat to merge them first.

Parameters:
- post_image_path (str): Path to 14-band concatenated image (multi-spectral + dem + slope)

Returns:
- mask_path (str): Path to the saved landslide mask image
- classes (dict): Mapping of class values
  * 0: non-landslide
  * 255: landslide
''')
def seg_landslide_ms(post_image_path: str) -> str:
    try:
        full_path = TEMP_DIR / _unique_name("seg_landslide_ms_mask", post_image_path)
        out, forward_ms = landslide4sense_sliding_infer(load_model("landslide_ms", CHECKPOINTS_DIR), post_image_path)
        full_path = _save_mask(out, full_path, reference_raster=post_image_path)
        result = {
            "tool": "seg.landslide_ms",
            "mask_path": str(full_path),
            "classes": {0: "non-landslide", 255: "landslide"},
        }
        return json.dumps(_attach_tool_meta(result, forward_ms), ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("seg.landslide_ms", error=str(e), pre_image=post_image_path, post_image=post_image_path))


@mcp.tool(name="seg.landslide_bitemporal", description='''
Description:
Segment landslide from pre/post disaster optical satellite images (bi-temporal change detection).
Spatial resolution: 0.59m.

Parameters:
- pre_image_path (str): Path to pre-disaster image
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to the saved landslide mask image
- classes (dict): Mapping of class values
  * 0: non-landslide
  * 255: landslide
''')
def seg_landslide_bitemporal(pre_image_path: str, post_image_path: str) -> str:
    try:
        full_path = TEMP_DIR / _unique_name("seg_landslide_bitemporal_mask", pre_image_path, post_image_path)
        model = load_model("landslide_bitemporal", CHECKPOINTS_DIR)
        out, forward_ms = model_sliding_infer(model, pre_image_path, post_image_path)
        full_path = _save_mask(out * 255, full_path, reference_raster=pre_image_path)
        result = {
            "tool": "seg.landslide_bitemporal",
            "mask_path": str(full_path),
            "classes": {0: "non-landslide", 255: "landslide"},
        }
        return json.dumps(_attach_tool_meta(result, forward_ms), ensure_ascii=False)
    except Exception as e:
        return json.dumps(
            _stub_result("seg.landslide_bitemporal", error=str(e), pre_image=post_image_path, post_image=post_image_path))


@mcp.tool(name="seg.reservoir", description='''
Description:
Segment reservoirs from multi-modal data: multi-spectral optical + DEM + slope.
Spatial resolution: 10m.
IMPORTANT: The input image MUST have exactly 14 bands (12-band multi-spectral + DEM: 1 band + Slope: 1 band).
If your data has separate multi-spectral, DEM, and slope files, use ras.concat to merge them first.

Parameters:
- post_image_path (str): Path to 14-band concatenated image (multi-spectral + dem + slope)

Returns:
- mask_path (str): Path to the saved binary reservoir mask
- classes (dict): Mapping of class values to semantic labels
  * 0: background
  * 255: reservoir
''')
def seg_reservoir(post_image_path: str) -> str:
    try:
        full_path = TEMP_DIR / _unique_name("seg_reservoir_mask", post_image_path)
        out, forward_ms = landslide4sense_sliding_infer(load_model("reservoir", CHECKPOINTS_DIR), post_image_path)
        full_path = _save_mask(out, full_path, reference_raster=post_image_path)
        result = {
            "tool": "seg.reservoir",
            "mask_path": str(full_path),
            "classes": {0: "background", 255: "reservoir"},
        }
        return json.dumps(_attach_tool_meta(result, forward_ms), ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("seg.reservoir", error=str(e), post_image=post_image_path))


@mcp.tool(name="seg.road_damage", description='''
Description:
Segment road damage from pre/post disaster optical satellite images.
Detects flooded and blocked road segments.
Spatial resolution: 0.5m-1.0m.

Parameters:
- pre_image_path (str): Path to pre-disaster image
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to the saved road damage mask
- classes (dict): Mapping of class values to road conditions
  * 1: intact
  * 2: flooded
  * 3: debris blocked
''')
def seg_road_damage(pre_image_path: str, post_image_path: str) -> str:
    try:
        out, forward_ms = model_sliding_infer(_get_model("road_damage"), pre_image_path, post_image_path)
        full_path = TEMP_DIR / _unique_name("seg_road_damage_mask", pre_image_path, post_image_path)
        full_path = _save_mask(out, full_path, reference_raster=pre_image_path)
        result = {
            "tool": "seg.road_damage",
            "mask_path": str(full_path),
            "classes": {1: "intact", 2: "flooded", 3: "debris blocked"},
        }
        return json.dumps(_attach_tool_meta(result, forward_ms), ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("seg.road_damage", error=str(e), pre_image=pre_image_path, post_image=post_image_path))


# iSAID single-temporal derived segmentations. The model outputs the official iSAID class ids,
# except that 2 is roundabout and 12 is storage tank (their names are swapped in the colour table
# the model was trained with): 1 ship, 2 roundabout, 3 baseball_diamond, 4 tennis_court,
# 5 basketball_court, 6 ground_track_field, 7 bridge, 8 large_vehicle, 9 small_vehicle,
# 10 helicopter, 11 swimming_pool, 12 storage_tank, 13 soccer_ball_field, 14 plane, 15 harbor.

def _isaid_tool(tool: str, post_image_path: str, ids: set, out_filename: str,
                classes: Dict[int, str], note: str = None) -> str:
    try:
        res = _binary_mask("isaid", post_image_path, ids, out_filename, note=note)
        res["tool"] = tool
        return json.dumps(res, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result(tool, error=str(e), post_image=post_image_path))


@mcp.tool(name="seg.vehicle", description='''
Segment vehicles (large and small) from post-disaster optical satellite image.
Spatial resolution: 0.5m-1.0m.

Parameters:
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to binary vehicle mask (255 = vehicle)
- classes (dict): {8: large_vehicle, 9: small_vehicle}
''')
def seg_vehicle(post_image_path: str) -> str:
    return _isaid_tool("seg.vehicle", post_image_path, {8, 9}, "seg_vehicle_mask.png",
                       {8: "large_vehicle", 9: "small_vehicle"},
                       note="Binary mask: selected classes merged into 255.")


@mcp.tool(name="seg.playground", description='''
Segment playground/sports fields from post-disaster optical image.
Classes: 3 baseball_diamond, 4 tennis_court, 5 basketball_court, 6 ground_track_field, 13 soccer_ball_field.
Spatial resolution: 0.5m-1.0m.

Parameters:
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to binary playground mask (255 = playground)
- classes (dict): Mapping of class ids to names
''')
def seg_playground(post_image_path: str) -> str:
    return _isaid_tool("seg.playground", post_image_path, {3, 4, 5, 6, 13}, "seg_playground_mask.png",
                       {3: "baseball_diamond", 4: "tennis_court", 5: "basketball_court",
                        6: "ground_track_field", 13: "soccer_ball_field"},
                       note="Binary mask of playground/sports field classes merged into 255.")


@mcp.tool(name="seg.swimming_pool", description='''
Segment swimming pools from post-disaster optical satellite image.
Spatial resolution: 0.5m-1.0m.

Parameters:
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to binary pool mask (255 = swimming_pool)
- classes (dict): {11: swimming_pool}
''')
def seg_swimming_pool(post_image_path: str) -> str:
    return _isaid_tool("seg.swimming_pool", post_image_path, {11}, "seg_swimming_pool_mask.png", {11: "swimming_pool"})


@mcp.tool(name="seg.storage_tank", description='''
Segment storage tanks from post-disaster optical satellite image.
Spatial resolution: 0.5m-1.0m.

Parameters:
- post_image (str): Path to post-disaster image

Returns:
- mask_path (str): Path to binary storage tank mask (255 = storage_tank)
- classes (dict): {12: storage_tank}
''')
def seg_storage_tank(post_image_path: str) -> str:
    return _isaid_tool("seg.storage_tank", post_image_path, {12}, "seg_storage_tank_mask.png", {12: "storage_tank"})


@mcp.tool(name="seg.bridge", description='''
Segment bridges from post-disaster optical satellite image.
Spatial resolution: 0.5m-1.0m.

Parameters:
- post_image (str): Path to post-disaster image

Returns:
- mask_path (str): Path to binary bridge mask (255 = bridge)
- classes (dict): {7: bridge}
''')
def seg_bridge(post_image_path: str) -> str:
    return _isaid_tool("seg.bridge", post_image_path, {7}, "seg_bridge_mask.png", {7: "bridge"})


@mcp.tool(name="seg.harbor", description='''
Segment harbors from post-disaster optical satellite image.
Spatial resolution: 0.5m-1.0m.

Parameters:
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to binary harbor mask (255 = harbor)
- classes (dict): {15: harbor}
''')
def seg_harbor(post_image_path: str) -> str:
    return _isaid_tool("seg.harbor", post_image_path, {15}, "seg_harbor_mask.png", {15: "harbor"})


@mcp.tool(name="seg.roundabout", description='''
Segment roundabouts from post-disaster optical satellite image.
Spatial resolution: 0.5m-1.0m.

Parameters:
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to binary roundabout mask (255 = roundabout)
- classes (dict): {2: roundabout}
''')
def seg_roundabout(post_image_path: str) -> str:
    return _isaid_tool("seg.roundabout", post_image_path, {2}, "seg_roundabout_mask.png", {2: "roundabout"})


@mcp.tool(name="seg.ship", description='''
Segment ships from post-disaster optical satellite image.
Spatial resolution: 0.5m-1.0m.

Parameters:
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to binary ship mask (255 = ship)
- classes (dict): {1: ship}
''')
def seg_ship(post_image_path: str) -> str:
    return _isaid_tool("seg.ship", post_image_path, {1}, "seg_ship_mask.png", {1: "ship"})


@mcp.tool(name="seg.plane", description='''
Segment planes from post-disaster optical satellite image.
Spatial resolution: 0.5m-1.0m.

Parameters:
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to binary plane mask (255 = plane)
''')
def seg_plane(post_image_path: str) -> str:
    return _isaid_tool("seg.plane", post_image_path, {14}, "seg_plane_mask.png", {14: "plane"})


@mcp.tool(name="seg.flood", description='''
Description:
Segment flood areas from pre/post disaster optical satellite images.
Detects standing water and flooded regions.
Spatial resolution: 0.5m-1.0m.

Parameters:
- pre_image_path (str): Path to pre-disaster image
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to binary flooding mask (255 = flooding)
''')
def seg_flood(pre_image_path: str, post_image_path: str) -> str:
    try:
        pre_water_res = _water_mask(pre_image_path, _unique_name("pre_water_mask", pre_image_path))
        post_water_res = _water_mask(post_image_path, _unique_name("post_water_mask", post_image_path))
        pre_water_mask = imread(pre_water_res["mask_path"]).astype(np.int16)
        post_water_mask = imread(post_water_res["mask_path"]).astype(np.int16)
        diff = post_water_mask - pre_water_mask
        flood_mask = np.where(diff > 128, np.ones_like(pre_water_mask) * 255, np.zeros_like(pre_water_mask))
        full_path = TEMP_DIR / _unique_name("flooding_map", pre_image_path, post_image_path)
        full_path = _save_mask(flood_mask, full_path, reference_raster=pre_image_path)
        result = {
            "tool": "seg.flood",
            "mask_path": str(full_path),
            "mask_value": 255,
            "_meta": post_water_res["_meta"],
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("seg.flood", error=str(e), pre_image=pre_image_path, post_image=post_image_path))


@mcp.tool(name="seg.vegetation", description='''
Description:
Segment vegetation areas from single temporal optical satellite image.
Detects trees, shrub and agriculture land.
Spatial resolution: 0.5m-1.0m.

Parameters:
- image_path (str): Path to single temporal satellite image

Returns:
- mask_path (str): Path to binary vegetation mask (255 = vegetation)
''')
def seg_vegetation(image_path: str) -> str:
    try:
        classes = {5: "forest", 6: "agriculture"}
        vegetation_res = _binary_mask("landcover", image_path, {5, 6},
                                      _unique_name("vegetation_mask", image_path), classes)
        vegetation_mask = imread(vegetation_res["mask_path"]).astype(np.int16)
        full_path = TEMP_DIR / _unique_name("vegetation_map", image_path)
        full_path = _save_mask(vegetation_mask, full_path, reference_raster=image_path)
        result = {
            "tool": "seg.vegetation",
            "mask_path": str(full_path),
            "classes": classes,
            "mask_value": 255,
            "_meta": vegetation_res["_meta"],
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("seg.vegetation", error=str(e), image=image_path))


@mcp.tool(name="seg.building_lowres", description='''
Description:
Segment building areas from single temporal low-resolution optical satellite image.
Spatial resolution: 3.46m.

Parameters:
- image_path (str): Path to single temporal satellite image

Returns:
- mask_path (str): Path to binary building mask (255 = building)
''')
def seg_building_lowres(image_path: str) -> str:
    # The SpaceNet-7 model was trained on 3x-upsampled PlanetScope (~1 m GSD): 170 px tiles of the
    # native image are upsampled to the 1440 px model input and the logits resized back to 170 px.
    scale, patch = 3, 1440
    patch_orig = patch // scale
    stride = patch_orig // 2
    try:
        image = imread(image_path)[:, :, :3]
        h, w = image.shape[:2]
        pad_h, pad_w = max(patch_orig - h, 0), max(patch_orig - w, 0)
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
        normalize = Compose([Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255)])
        tensor = torch.from_numpy(normalize(image=image)["image"]).float()
        model = load_model("building_lowres", CHECKPOINTS_DIR)

        def infer(m):
            def wrapped(x):
                return F.interpolate(m(x), size=(patch_orig, patch_orig), mode="bilinear", align_corners=False)

            def upsample(x):
                return F.interpolate(x.permute(0, 3, 1, 2).float(), size=(patch, patch),
                                     mode="bilinear", align_corners=False)

            probs = predict_probs(wrapped, tensor, patch_orig, stride, preprocess=upsample, device=DEVICE)
            return ((1 - probs.argmax(dim=0).numpy()) * 255)[:h, :w]

        out, forward_ms = _run_on_device(model, infer)
        full_path = TEMP_DIR / _unique_name("building_planet", image_path)
        _save_mask(out, full_path)
        result = {
            "tool": "seg.building_lowres",
            "mask_path": str(full_path),
            "classes": {0: "non-building", 255: "building"},
            "mask_value": 255,
        }
        return json.dumps(_attach_tool_meta(result, forward_ms), ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result("seg.building_lowres", error=str(e), image=image_path))


# ============== RescueNet-style aerial (UAV) tools ==============

def _aerial_tool(tool: str, post_image_path: str, class_id: int, class_name: str, out_filename: str) -> str:
    try:
        res = _binary_mask("aerial", post_image_path, {class_id}, out_filename, {class_id: class_name})
        res["tool"] = tool
        return json.dumps(res, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_stub_result(tool, error=str(e), post_image=post_image_path))


@mcp.tool(name="seg.building_intact_aerial", description='''
Segment building-no-damage areas from post-disaster optical airborne image.
Spatial resolution: 0.03m.

Parameters:
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to building_no_damage mask (255 = building_no_damage)
- classes (dict): {1: "building_no_damage"}
''')
def seg_building_intact_aerial(post_image_path: str) -> str:
    return _aerial_tool("seg.building_intact_aerial", post_image_path, 1, "building_no_damage",
                        "seg_building_intact_aerial_mask.png")


@mcp.tool(name="seg.building_damage_aerial", description='''
Segment building-major-damage areas from post-disaster optical airborne image.
Spatial resolution: 0.03m.

Parameters:
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to building_damage mask (255 = building_damage)
- classes (dict): {2: "building_damage"}
''')
def seg_building_damage_aerial(post_image_path: str) -> str:
    return _aerial_tool("seg.building_damage_aerial", post_image_path, 2, "building_damage",
                        "seg_building_damage_aerial_mask.png")


@mcp.tool(name="seg.building_destroyed_aerial", description='''
Segment totally damaged building areas from post-disaster optical airborne image.
Spatial resolution: 0.03m.

Parameters:
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to building_total_damage mask (255 = building_total_damage)
- classes (dict): {3: "building_total_damage"}
''')
def seg_building_destroyed_aerial(post_image_path: str) -> str:
    return _aerial_tool("seg.building_destroyed_aerial", post_image_path, 3, "building_total_damage",
                        "seg_building_destroyed_aerial_mask.png")


@mcp.tool(name="seg.pool_aerial", description='''
Segment pool areas from post-disaster optical airborne image.
Spatial resolution: 0.03m.

Parameters:
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to pool mask (255 = pool)
- classes (dict): {4: "pool"}
''')
def seg_pool_aerial(post_image_path: str) -> str:
    return _aerial_tool("seg.pool_aerial", post_image_path, 4, "pool", "seg_pool_aerial_mask.png")


@mcp.tool(name="seg.road_intact_aerial", description='''
Segment intact road areas from post-disaster optical airborne image.
Spatial resolution: 0.03m.

Parameters:
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to road_no_damage mask (255 = road_no_damage)
- classes (dict): {5: "road_no_damage"}
''')
def seg_road_intact_aerial(post_image_path: str) -> str:
    return _aerial_tool("seg.road_intact_aerial", post_image_path, 5, "road_no_damage",
                        "seg_road_intact_aerial_mask.png")


@mcp.tool(name="seg.road_debris_aerial", description='''
Segment debris-covered road areas from post-disaster optical airborne image.
Spatial resolution: 0.03m.

Parameters:
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to road_debris_covered mask (255 = road_debris_covered)
- classes (dict): {6: "road_debris_covered"}
''')
def seg_road_debris_aerial(post_image_path: str) -> str:
    return _aerial_tool("seg.road_debris_aerial", post_image_path, 6, "road_debris_covered",
                        "seg_road_debris_aerial_mask.png")


@mcp.tool(name="seg.tree_fallen_aerial", description='''
Segment fallen-tree areas from post-disaster optical airborne image.
Spatial resolution: 0.03m.

Parameters:
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to tree_fallen mask (255 = tree_fallen)
- classes (dict): {7: "tree_fallen"}
''')
def seg_tree_fallen_aerial(post_image_path: str) -> str:
    return _aerial_tool("seg.tree_fallen_aerial", post_image_path, 7, "tree_fallen", "seg_tree_fallen_aerial_mask.png")


@mcp.tool(name="seg.tree_standing_aerial", description='''
Segment standing-tree areas from post-disaster optical airborne image.
Spatial resolution: 0.03m.

Parameters:
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to tree_not_fallen mask (255 = tree_not_fallen)
- classes (dict): {8: "tree_not_fallen"}
''')
def seg_tree_standing_aerial(post_image_path: str) -> str:
    return _aerial_tool("seg.tree_standing_aerial", post_image_path, 8, "tree_not_fallen",
                        "seg_tree_standing_aerial_mask.png")


@mcp.tool(name="seg.vehicle_free_aerial", description='''
Segment non-trapped vehicle areas from post-disaster optical airborne image.
Spatial resolution: 0.03m.

Parameters:
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to vehicle_not_trapped mask (255 = vehicle_not_trapped)
- classes (dict): {9: "vehicle_not_trapped"}
''')
def seg_vehicle_free_aerial(post_image_path: str) -> str:
    return _aerial_tool("seg.vehicle_free_aerial", post_image_path, 9, "vehicle_not_trapped",
                        "seg_vehicle_free_aerial_mask.png")


@mcp.tool(name="seg.vehicle_trapped_aerial", description='''
Segment trapped vehicle areas from post-disaster optical airborne image.
Spatial resolution: 0.03m.

Parameters:
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to vehicle_trapped mask (255 = vehicle_trapped)
- classes (dict): {10: "vehicle_trapped"}
''')
def seg_vehicle_trapped_aerial(post_image_path: str) -> str:
    return _aerial_tool("seg.vehicle_trapped_aerial", post_image_path, 10, "vehicle_trapped",
                        "seg_vehicle_trapped_aerial_mask.png")


@mcp.tool(name="seg.water_aerial", description='''
Segment water areas from post-disaster optical airborne image.
Spatial resolution: 0.03m.

Parameters:
- post_image_path (str): Path to post-disaster image

Returns:
- mask_path (str): Path to water mask (255 = water)
- classes (dict): {11: "water"}
''')
def seg_water_aerial(post_image_path: str) -> str:
    return _aerial_tool("seg.water_aerial", post_image_path, 11, "water", "seg_water_aerial_mask.png")


if __name__ == "__main__":
    mcp.run(show_banner=False)
