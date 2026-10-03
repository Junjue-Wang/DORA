"""Shared coercers for tool parameters.

Some models (e.g. Qwen3.5, gemini) serialize list/dict args as JSON strings
instead of native types. Pydantic then rejects them with `list_type` /
`dict_type` / `model_attributes_type` validation errors. We apply a
BeforeValidator that parses such strings back to native structures, while
passing native values through unchanged. Schema is preserved so other
models are unaffected.
"""
import json as _json
from typing import Annotated, Any, Dict, List, Optional, Union

from pydantic import BeforeValidator


def _coerce_json_if_str(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if (s.startswith("[") and s.endswith("]")) or (
            s.startswith("{") and s.endswith("}")
        ):
            try:
                return _json.loads(s)
            except (ValueError, TypeError):
                pass
    return v


_BV = BeforeValidator(_coerce_json_if_str)

# --- Required (not Optional) list/dict types ---
ListInt = Annotated[List[int], _BV]
ListStr = Annotated[List[str], _BV]
ListFloat = Annotated[List[float], _BV]
ListAny = Annotated[List[Any], _BV]
ListDict = Annotated[List[Dict[str, Any]], _BV]          # bare List[Dict] or List[Dict[str,Any]]
ListDictStrAny = Annotated[List[Dict[str, Any]], _BV]    # alias
ListDictStrInt = Annotated[List[Dict[str, int]], _BV]
ListListInt = Annotated[List[List[int]], _BV]

DictStrStr = Annotated[Dict[str, str], _BV]
DictStrAny = Annotated[Dict[str, Any], _BV]
DictStrFloat = Annotated[Dict[str, float], _BV]
DictAnyStr = Annotated[Dict[Any, str], _BV]
DictAnyAny = Annotated[Dict[Any, Any], _BV]

ListOrDictStrAny = Annotated[Union[List[Dict[str, Any]], Dict[str, Any]], _BV]

# --- Optional variants ---
OptListInt = Annotated[Optional[List[int]], _BV]
OptListStr = Annotated[Optional[List[str]], _BV]
OptListFloat = Annotated[Optional[List[float]], _BV]
OptListAny = Annotated[Optional[List[Any]], _BV]
OptListDict = Annotated[Optional[List[Dict[str, Any]]], _BV]
OptListDictStrAny = Annotated[Optional[List[Dict[str, Any]]], _BV]
OptListListInt = Annotated[Optional[List[List[int]]], _BV]

OptDictStrStr = Annotated[Optional[Dict[str, str]], _BV]
OptDictStrAny = Annotated[Optional[Dict[str, Any]], _BV]
OptDictStrFloat = Annotated[Optional[Dict[str, float]], _BV]
OptDictAnyStr = Annotated[Optional[Dict[Any, str]], _BV]
OptDictAnyAny = Annotated[Optional[Dict[Any, Any]], _BV]
