"""
POI MCP server: search, filter and project point-of-interest layers onto imagery.
"""
import functools
import inspect
import json
from pathlib import Path
from typing import Dict, Any, Tuple

from ._common import session_temp_dir, short_name as _short_name
from ._types import ListStr, ListDict, OptListStr

from fastmcp import FastMCP

# Geo/vector deps
try:
    import geopandas as gpd
    GEO_DEPS_OK = True
except Exception:
    gpd = None
    GEO_DEPS_OK = False

try:
    import rasterio
    RASTERIO_OK = True
except Exception:
    rasterio = None
    RASTERIO_OK = False

try:
    from pyproj import Transformer
    PYPROJ_OK = True
except Exception:
    PYPROJ_OK = False


mcp = FastMCP()
TEMP_DIR = session_temp_dir()


def _stub_result(tool: str, **inputs: Any) -> Dict[str, Any]:
    result = {
        "tool": tool,
        "status": "stub",
        "note": "Replace this stub with a real implementation.",
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


def _read_geojson(path: str) -> gpd.GeoDataFrame:
    """Read a GeoJSON or Shapefile."""
    path_lower = path.lower()
    if path_lower.endswith(".geojson") or path_lower.endswith(".shp"):
        gdf = gpd.read_file(path)
        if gdf.crs is not None:
            gdf = gdf.set_crs(gdf.crs, allow_override=True)
        return gdf
    else:
        raise ValueError(f"Unsupported file format: {path}")


def _world_to_pixel(transform, x: float, y: float) -> Tuple[int, int]:
    col = int((x - transform[2]) / transform[0])
    row = int((transform[5] - y) / -transform[4])
    return row, col


def poi_search_by_name(geojson_path: str, name: str, exact: bool = False) -> Dict[str, Any]:
    """
    Search POIs by name.

    Parameters:
    - geojson_path: GeoJSON/Shapefile path
    - name: name to search (fuzzy by default)
    - exact: exact match only

    Returns:
    - results: matching POIs with name, coordinates and properties
    """
    if not GEO_DEPS_OK:
        return _stub_result("poi.search_by_name", error="geopandas/shapely is not installed")

    gdf = _read_geojson(geojson_path)

    if len(gdf) == 0:
        return _stub_result("poi.search_by_name", note="Empty GeoJSON input", geojson_path=geojson_path)

    results = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        poi_name = row.get('name', '') or ''

        if exact:
            match = poi_name == name
        else:
            match = name.lower() in poi_name.lower()

        if match:
            props = {k: v for k, v in row.items() if k != 'geometry'}
            props["geometry_type"] = geom.geom_type

            results.append({
                "name": poi_name,
                "latitude": float(geom.y),
                "longitude": float(geom.x),
                "properties": props
            })

    # Save the matches as a vector file
    output_path = None
    if results:
        # Build a GeoJSON FeatureCollection
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        for r in results:
            geojson["features"].append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [r["longitude"], r["latitude"]]
                },
                "properties": {k: v for k, v in r.items() if k not in ["latitude", "longitude"]}
            })

        # Save
        output_name = _short_name(f"poi_search_{Path(geojson_path).stem}_{name}.geojson")
        output_path = TEMP_DIR / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

    return {
        "tool": "poi.search_by_name",
        "search_name": name,
        "exact_match": exact,
        "total_found": len(results),
        "results": results,
        "output_path": str(output_path)
    }


def poi_to_vector(geojson_path: str, name: str = None, exact: bool = False) -> Dict[str, Any]:
    """
    Convert POIs to a GeoJSON vector layer.

    Parameters:
    - geojson_path: GeoJSON/Shapefile path
    - name: optional name filter
    - exact: exact name match only

    Returns:
    - vector_path: output vector path
    - count: number of features
    """
    if not GEO_DEPS_OK:
        return _stub_result("poi.to_vector", error="geopandas/shapely is not installed")

    gdf = _read_geojson(geojson_path)

    if len(gdf) == 0:
        return _stub_result("poi.to_vector", note="Empty GeoJSON input", geojson_path=geojson_path)

    # Optional name filter
    if name:
        if exact:
            mask = gdf['name'] == name
        else:
            mask = gdf['name'].str.lower().str.contains(name.lower(), na=False)
        gdf = gdf[mask].copy()

    # Drop the CRS
    if gdf.crs is not None:
        gdf = gdf.set_crs(gdf.crs, allow_override=True)

    # Save
    output_name = _short_name(f"poi_vector_{Path(geojson_path).stem}.geojson")
    output_path = TEMP_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(output_path), driver="GeoJSON", encoding="utf-8")

    return {
        "tool": "poi.to_vector",
        "vector_path": str(output_path),
        "count": len(gdf),
        "name_filter": name,
        "exact_match": exact
    }


def poi_group_by(data: ListDict, group_field: str, count_field: str = None, return_item_details: bool = False, return_sum: bool = False) -> Dict[str, Any]:
    """
    Group a list of dictionaries by a field and count.
    Supports nested field paths like "item.properties.types".

    Parameters:
    - data: List of dictionaries to group (can be tool.loop results with item/last_obs structure)
    - group_field: Field name to group by (supports dotted path like "item.properties.types")
    - count_field: Optional field to sum (supports dotted path)
    - return_item_details: Whether to include items in groups (default: False)
    - return_sum: Whether to include sum of count_field (default: False)

    Returns:
    - groups: Dictionary of {group_value: {count, sum?, items?}}
    """
    from collections import defaultdict

    def get_nested_value(obj: Dict, path: str):
        """Get value from nested dict using dot notation (e.g., "item.properties.types")"""
        keys = path.split(".")
        result = obj
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return None
        return result

    def extract_damage_class(pixel_values: Dict) -> int:
        """Extract damage class from pixel_values dict.
        For single-band masks: {"1": class_value} -> return class_value
        """
        if not pixel_values or not isinstance(pixel_values, dict):
            return None

        # For single-band building damage mask: {"1": 2} means class 2
        if len(pixel_values) == 1:
            for key, value in pixel_values.items():
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return None

        # For multi-band case, find the class with maximum count
        max_class = None
        max_count = -1
        for cls_str, count in pixel_values.items():
            try:
                cls = int(cls_str)
                if count > max_count:
                    max_count = count
                    max_class = cls
            except (ValueError, TypeError):
                continue
        return max_class

    if not data:
        return {
            "tool": "poi.group_by",
            "group_field": group_field,
            "count_field": count_field,
            "total_items": 0,
            "total_groups": 0,
            "groups": {}
        }

    groups = defaultdict(lambda: {"count": 0, "sum": 0, "items": []})

    for item in data:
        # Support nested field paths
        key = get_nested_value(item, group_field)

        # Handle special damage_class extraction from last_obs.pixel_values
        damage_class = None
        if count_field == "damage_class":
            last_obs = item.get("last_obs")
            if last_obs and isinstance(last_obs, dict):
                pixel_values = last_obs.get("pixel_values")
                damage_class = extract_damage_class(pixel_values)
        elif count_field:
            damage_class = get_nested_value(item, count_field)

        if key is not None:
            # Convert types field if it contains comma-separated values (e.g., "restaurant, food, ...")
            if isinstance(key, str) and "," in key:
                key = key.split(",")[0].strip()

            groups[key]["count"] += 1
            if damage_class is not None:
                groups[key]["sum"] += float(damage_class)
            groups[key]["items"].append(item)

    # Convert to regular dict
    result_groups = {}
    for k, v in groups.items():
        group_data = {"count": v["count"]}
        if return_sum and v["sum"]:
            group_data["sum"] = round(v["sum"], 2)
        if return_item_details:
            group_data["items"] = v["items"]
        result_groups[str(k)] = group_data

    return {
        "tool": "poi.group_by",
        "group_field": group_field,
        "count_field": count_field,
        "return_item_details": return_item_details,
        "return_sum": return_sum,
        "total_items": len(data),
        "total_groups": len(result_groups),
        "groups": result_groups
    }


def poi_filter_by_damage(data: ListDict, min_damage_class: int = 2) -> Dict[str, Any]:
    """
    Filter POI results to only include those in damaged buildings.

    Parameters:
    - data: List of tool.loop results with item/last_obs structure
    - min_damage_class: Minimum damage class to include (default: 2 for partially damaged or worse)

    Returns:
    - filtered_results: List of results where damage_class >= min_damage_class
    - total_input: Total number of input items
    - total_filtered: Number of items after filtering
    - total_damaged: Number of damaged items
    """
    def extract_damage_class(pixel_values: Dict) -> int:
        """Extract damage class from pixel_values dict.
        For single-band masks: {"1": class_value} -> return class_value
        """
        if not pixel_values or not isinstance(pixel_values, dict):
            return None

        # For single-band building damage mask: {"1": 2} means class 2
        if len(pixel_values) == 1:
            for key, value in pixel_values.items():
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return None

        # For multi-band case, find the class with maximum count
        max_class = None
        max_count = -1
        for cls_str, count in pixel_values.items():
            try:
                cls = int(cls_str)
                if count > max_count:
                    max_count = count
                    max_class = cls
            except (ValueError, TypeError):
                continue
        return max_class

    if not data:
        return {
            "tool": "poi.filter_by_damage",
            "total_input": 0,
            "total_filtered": 0,
            "total_damaged": 0,
            "filtered_results": []
        }

    filtered_results = []
    total_damaged = 0
    for item in data:
        last_obs = item.get("last_obs")
        if last_obs and isinstance(last_obs, dict):
            pixel_values = last_obs.get("pixel_values")
            damage_class = extract_damage_class(pixel_values)
            if damage_class is not None and damage_class >= min_damage_class:
                filtered_results.append(item)
                total_damaged += 1

    return {
        "tool": "poi.filter_by_damage",
        "total_input": len(data),
        "total_filtered": len(filtered_results),
        "total_damaged": total_damaged,
        "filtered_results": filtered_results
    }


def poi_filter_by_intact(data: ListDict) -> Dict[str, Any]:
    """
    Filter POI results to only include those in intact buildings (damage_class == 1).

    Parameters:
    - data: List of tool.loop results with item/last_obs structure

    Returns:
    - filtered_results: List of results where damage_class == 1
    - total_input: Total number of input items
    - total_intact: Number of intact items
    """
    def extract_damage_class(pixel_values: Dict) -> int:
        """Extract damage class from pixel_values dict.
        Supports two formats:
        1. From ras.sample_point: {"1": class_value} -> return class_value
        2. From damage mask: {"class_value": pixel_count} -> return class_value
        """
        if not pixel_values or not isinstance(pixel_values, dict):
            return None

        # Format 1: ras.sample_point returns {1: class_value}
        # e.g., {"1": 3} means class 3
        if len(pixel_values) == 1:
            for key, value in pixel_values.items():
                try:
                    key_int = int(key)
                    if 1 <= key_int <= 10:
                        return int(value)
                except (ValueError, TypeError):
                    pass

        # Format 2: damage mask format {"class_value": count}
        # e.g., {"3": 5} means class 3 with 5 pixels
        max_class = None
        max_count = -1
        for cls_str, count in pixel_values.items():
            try:
                cls = int(cls_str)
                if count > max_count:
                    max_count = count
                    max_class = cls
            except (ValueError, TypeError):
                continue
        return max_class

    if not data:
        return {
            "tool": "poi.filter_by_intact",
            "total_input": 0,
            "total_intact": 0,
            "filtered_results": []
        }

    filtered_results = []
    total_intact = 0

    for item in data:
        last_obs = item.get("last_obs")
        if last_obs and isinstance(last_obs, dict):
            pixel_values = last_obs.get("pixel_values")
            damage_class = extract_damage_class(pixel_values)
            if damage_class is not None and damage_class == 1:
                filtered_results.append(item)
                total_intact += 1

    return {
        "tool": "poi.filter_by_intact",
        "total_input": len(data),
        "total_intact": total_intact,
        "filtered_results": filtered_results
    }


def poi_world_to_pixel(geojson_path: str, raster_path: str, pixel_size_m: str = 0.8) -> Dict[str, Any]:
    """
    Convert POI geographic coordinates to image pixel coordinates.
    """
    if not GEO_DEPS_OK:
        return _stub_result("poi.world_to_pixel", error="geopandas/shapely is not installed")
    if not RASTERIO_OK:
        return _stub_result("poi.world_to_pixel", error="rasterio is not installed")
    if not PYPROJ_OK:
        return _stub_result("poi.world_to_pixel", error="pyproj is not installed")

    with rasterio.open(raster_path) as src:
        img_bounds = src.bounds
        img_transform = src.transform
        img_crs = src.crs
        img_width = src.width
        img_height = src.height

    gdf = _read_geojson(geojson_path)

    if len(gdf) == 0:
        return _stub_result("poi.world_to_pixel", note="Empty GeoJSON input", geojson_path=geojson_path)

    transformer = Transformer.from_crs("EPSG:4326", img_crs, always_xy=True)

    valid_features = []
    total_points = len(gdf)

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        lon, lat = geom.x, geom.y

        try:
            utm_x, utm_y = transformer.transform(lon, lat)
        except Exception:
            continue

        pixel_row, pixel_col = _world_to_pixel(img_transform, utm_x, utm_y)

        if 0 <= pixel_row < img_height and 0 <= pixel_col < img_width:
            props = {k: v for k, v in row.items() if k != 'geometry'}
            props["pixel_row"] = int(pixel_row)
            props["pixel_col"] = int(pixel_col)
            props["latitude"] = float(lat)
            props["longitude"] = float(lon)
            props["utm_x"] = float(utm_x)
            props["utm_y"] = float(utm_y)
            props["pixel_size_m"] = float(pixel_size_m)

            valid_features.append({
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(pixel_col), float(pixel_row)]
                }
            })

    valid_count = len(valid_features)
    filtered_count = total_points - valid_count

    output_data = {
        "type": "FeatureCollection",
        "features": valid_features,
        "metadata": {
            "source_geojson": geojson_path,
            "source_raster": raster_path,
            "source_crs": "EPSG:4326",
            "target_crs": str(img_crs) if img_crs else None,
            "image_bounds": {
                "left": float(img_bounds.left),
                "right": float(img_bounds.right),
                "bottom": float(img_bounds.bottom),
                "top": float(img_bounds.top)
            },
            "image_size": [int(img_width), int(img_height)],
            "total_input_points": total_points,
            "valid_points": valid_count,
            "filtered_points": filtered_count
        }
    }

    output_name = _short_name(f"poi_pixel_coords_{Path(geojson_path).stem}.geojson")
    output_path = TEMP_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return {
        "tool": "poi.world_to_pixel",
        "features": valid_features,
        "output_path": str(output_path),
        "total_points": total_points,
        "valid_points": valid_count,
        "filtered_points": filtered_count,
        "image_bounds": {
            "left": float(img_bounds.left),
            "right": float(img_bounds.right),
            "bottom": float(img_bounds.bottom),
            "top": float(img_bounds.top)
        },
        "image_size": [int(img_width), int(img_height)],
        "source_crs": "EPSG:4326",
        "target_crs": str(img_crs) if img_crs else None
    }


@mcp.tool(name="poi.search_by_name", description='''
Search for POI (Points of Interest) by name in a GeoJSON file.
Supports both exact and fuzzy matching.

Parameters:
- geojson_path (str): Path to GeoJSON file containing POI data
- name (str): Name to search for (case-insensitive fuzzy matching)
- exact (bool): Whether to use exact matching (default: false)

Returns:
- search_name (str): The search term used
- features (list): The found POI features
- total_found (int): Number of POIs matching the search
- results (list): List of matching POIs with name, coordinates, and properties
''')
@_tool_guard("poi.search_by_name")
def poi_search_by_name_tool(geojson_path: str, name: str, exact: bool = False) -> str:
    result = poi_search_by_name(geojson_path, name, exact)
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(name="poi.to_vector", description='''
Convert POI (Points of Interest) from GeoJSON/Shapefile to a vector GeoJSON file.
Optionally filter by name.

Parameters:
- geojson_path (str): Path to input GeoJSON or Shapefile
- name (str, optional): Filter POI by name (fuzzy matching)
- exact (bool): Whether to use exact name matching (default: false)

Returns:
- vector_path (str): Path to output vector GeoJSON
- count (int): Number of features in output
- name_filter (str): The name filter used
''')
@_tool_guard("poi.to_vector")
def poi_to_vector_tool(geojson_path: str, name: str = None, exact: bool = False) -> str:
    result = poi_to_vector(geojson_path, name, exact)
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(name="poi.group_by", description='''
Group a list of dictionaries by a field and calculate statistics.

Parameters:
- data (list[dict]): List of dictionaries to group
- group_field (str): Field name to group by
- count_field (str, optional): Field to sum (for weighted counts)
- return_item_details (bool, optional): Whether to include items in groups (default: False)
- return_sum (bool, optional): Whether to include sum of count_field (default: False)

Returns:
- group_field (str): The field used for grouping
- total_items (int): Total number of input items
- total_groups (int): Number of unique groups
- groups (dict): Dictionary of {group_value: {count}} or {count, sum, items} based on options
''')
@_tool_guard("poi.group_by")
def poi_group_by_tool(data: ListDict, group_field: str, count_field: str = None, return_item_details: bool = False, return_sum: bool = False) -> str:
    result = poi_group_by(data, group_field, count_field, return_item_details, return_sum)
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(name="poi.world_to_pixel", description='''
Transform GeoJSON POI coordinates from geographic CRS (WGS84) to the pixel coordinate system of a reference raster image.
Points that fall outside the image bounds are filtered out.
The output GeoJSON contains the original properties plus pixel coordinates (pixel_row, pixel_col, utm_x, utm_y).

Parameters:
- geojson_path (str): Path to input GeoJSON file with Point geometries (WGS84)
- raster_path (str): Path to reference raster image (GeoTIFF) to define the target pixel coordinate system

Returns:
- output_path (str): Path to transformed GeoJSON
- total_points (int): Total input POI count
- valid_points (int): Number of POIs within image bounds
- filtered_points (int): Number of POIs filtered out (outside image)
- image_bounds (dict): Image bounds (left, right, bottom, top)
- image_size (list): [width, height] of the reference image
''')
@_tool_guard("poi.world_to_pixel")
def poi_world_to_pixel_tool(geojson_path: str, raster_path: str) -> str:
    result = poi_world_to_pixel(geojson_path, raster_path)
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(name="poi.filter_by_damage", description='''
Filter POI results to only include those located in damaged buildings.
Takes the output from tool.loop and filters items where damage_class >= 2 (partially damaged or totally destroyed).

Parameters:
- data (list[dict]): List of tool.loop results containing item and last_obs
- min_damage_class (int, optional): Minimum damage class to include (default: 2)

Returns:
- total_input (int): Total number of input items
- total_filtered (int): Number of items after filtering
- total_damaged (int): Number of damaged items found
- filtered_results (list): Filtered results with damage_class >= min_damage_class
''')
@_tool_guard("poi.filter_by_damage")
def poi_filter_by_damage_tool(data: ListDict, min_damage_class: int = 2) -> str:
    result = poi_filter_by_damage(data, min_damage_class)
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(name="poi.filter_by_intact", description='''
Filter POI results to only include those located in intact buildings.
Takes the output from tool.loop and keeps items where damage_class == 1.

Parameters:
- data (list[dict]): List of tool.loop results containing item and last_obs

Returns:
- total_input (int): Total number of input items
- total_intact (int): Number of intact items found
- filtered_results (list): Filtered results with damage_class == 1
''')
@_tool_guard("poi.filter_by_intact")
def poi_filter_by_intact_tool(data: ListDict) -> str:
    result = poi_filter_by_intact(data)
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


def poi_filter_by_attribute(data: ListDict, attribute_name: str, allowed_values: ListStr) -> Dict[str, Any]:
    """
    Filter POI results by attribute values (e.g., POI types like restaurant, cafe, bank).

    Parameters:
    - data: List of tool.loop results with item/last_obs structure
    - attribute_name: The attribute field to filter on (e.g., "types" for POI types)
    - allowed_values: List of allowed values to keep (e.g., ["restaurant", "cafe", "bank", "grocery_or_supermarket"])

    Returns:
    - filtered_results: List of results where the attribute matches allowed_values
    - total_input: Total number of input items
    - total_filtered: Number of items after filtering
    """
    if not data:
        return {
            "tool": "poi.filter_by_attribute",
            "total_input": 0,
            "total_filtered": 0,
            "attribute_name": attribute_name,
            "filtered_results": []
        }

    filtered_results = []

    for item in data:
        item_data = item.get("item", {})
        if isinstance(item_data, dict):
            # Handle both direct properties and nested properties
            props = item_data.get("properties", item_data)

            # Get the attribute value (can be a string or comma-separated list)
            attr_value = props.get(attribute_name, "")

            # Convert to list if comma-separated
            if isinstance(attr_value, str):
                attr_list = [v.strip() for v in attr_value.split(",")]
            elif isinstance(attr_value, list):
                attr_list = attr_value
            else:
                attr_list = [str(attr_value)]

            # Check if any of the attribute values match allowed_values
            for val in attr_list:
                if val in allowed_values:
                    filtered_results.append(item)
                    break

    return {
        "tool": "poi.filter_by_attribute",
        "total_input": len(data),
        "total_filtered": len(filtered_results),
        "attribute_name": attribute_name,
        "allowed_values": allowed_values,
        "filtered_results": filtered_results
    }


@mcp.tool(name="poi.filter_by_attribute", description='''
Filter POI results by attribute values.
Useful for keeping only specific POI types (e.g., commercial facilities like restaurants, cafes, banks).

Parameters:
- data (list[dict]): List of tool.loop results containing item and last_obs
- attribute_name (str): The attribute field to filter on (e.g., "types" for POI types)
- allowed_values (list[str]): List of allowed values to keep (e.g., ["restaurant", "cafe", "bank"])

Returns:
- total_input (int): Total number of input items
- total_filtered (int): Number of items after filtering
- attribute_name (str): The attribute field used for filtering
- allowed_values (list): The allowed values used for filtering
- filtered_results (list): Filtered results matching the allowed values
''')
@_tool_guard("poi.filter_by_attribute")
def poi_filter_by_attribute_tool(data: ListDict, attribute_name: str, allowed_values: ListStr) -> str:
    result = poi_filter_by_attribute(data, attribute_name, allowed_values)
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


def poi_damage_population_exposure(
    damage_feature_stats: ListDict,
    population_feature_stats: ListDict,
    damage_threshold: float = 2.0,
    damage_field: str = "max",
    population_field: str = "max",
    join_field: str = "name",
    allowed_types: OptListStr = None,
) -> Dict[str, Any]:
    population_by_key = {}
    for item in population_feature_stats or []:
        key = item.get(join_field)
        if key is not None:
            population_by_key[str(key)] = item

    allowed_set = {str(x).strip() for x in (allowed_types or []) if str(x).strip()}
    exposed = []

    for item in damage_feature_stats or []:
        key = item.get(join_field)
        if key is None:
            continue
        pop_item = population_by_key.get(str(key))
        if not pop_item:
            continue

        damage_value = item.get(damage_field)
        population_value = pop_item.get(population_field)
        if damage_value is None or population_value is None:
            continue

        types_value = item.get("type") or item.get("types") or pop_item.get("type") or pop_item.get("types")
        if allowed_set:
            type_tokens = {t.strip() for t in str(types_value or "").split(",") if t.strip()}
            if not (type_tokens & allowed_set):
                continue

        if float(damage_value) < float(damage_threshold):
            continue

        exposed.append({
            "name": str(key),
            "types": types_value,
            "damage_value": float(damage_value),
            "population_value": float(population_value),
            "damage_feature_idx": item.get("feature_idx"),
            "population_feature_idx": pop_item.get("feature_idx"),
        })

    exposed.sort(key=lambda x: (-x["population_value"], -x["damage_value"], x["name"]))
    top_item = exposed[0] if exposed else None

    return {
        "tool": "poi.damage_population_exposure",
        "join_field": join_field,
        "damage_field": damage_field,
        "population_field": population_field,
        "damage_threshold": float(damage_threshold),
        "allowed_types": sorted(allowed_set),
        "total_damage_features": len(damage_feature_stats or []),
        "total_population_features": len(population_feature_stats or []),
        "total_exposed": len(exposed),
        "exposed_pois": exposed,
        "top_exposed_poi": top_item,
    }


@mcp.tool(name="poi.damage_population_exposure", description='''
Join damage and population feature statistics for POIs and rank exposed locations.
Useful for identifying high-population POIs whose damage score exceeds a threshold.

Parameters:
- damage_feature_stats (list[dict]): Damage stats per POI
- population_feature_stats (list[dict]): Population stats per POI
- damage_threshold (float, optional): Minimum damage value to keep
- damage_field (str, optional): Field name for damage values
- population_field (str, optional): Field name for population values
- join_field (str, optional): Shared key used to join the two tables
- allowed_types (list[str], optional): Restrict to POI types if provided

Returns:
- total_exposed (int): Number of POIs above the damage threshold
- exposed_pois (list): Ranked exposed POIs with damage and population values
- top_exposed_poi (dict|None): Highest-priority exposed POI
''')
@_tool_guard("poi.damage_population_exposure")
def poi_damage_population_exposure_tool(
    damage_feature_stats: ListDict,
    population_feature_stats: ListDict,
    damage_threshold: float = 2.0,
    damage_field: str = "max",
    population_field: str = "max",
    join_field: str = "name",
    allowed_types: OptListStr = None,
) -> str:
    result = poi_damage_population_exposure(
        damage_feature_stats=damage_feature_stats,
        population_feature_stats=population_feature_stats,
        damage_threshold=damage_threshold,
        damage_field=damage_field,
        population_field=population_field,
        join_field=join_field,
        allowed_types=allowed_types,
    )
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(show_banner=False)
