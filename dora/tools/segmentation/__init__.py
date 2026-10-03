"""Perception model zoo: architectures + released checkpoints behind the ``seg.*`` tools."""
from functools import partial
from pathlib import Path

import torch
from safetensors.torch import load_model as _load_safetensors

from .dpt import DPT, BitemporalDPT
from .hrnet import HRNetSeg
from .inference import LARGE_IMAGE_PIXELS, predict_labels_out_of_core, predict_probs, sliding_window
from .segformer import WeTr
from .swin_upernet import SwinUperNet

# name -> (architecture builder, checkpoint file)
MODEL_ZOO = {
    "building_damage":        (partial(BitemporalDPT, num_classes=4), "seg_building_damage.safetensors"),
    "building_damage_opt_sar": (partial(BitemporalDPT, num_classes=4), "seg_building_damage_opt_sar.safetensors"),
    "road_damage":            (partial(BitemporalDPT, num_classes=4), "seg_road_damage.safetensors"),
    "isaid":                  (partial(DPT, num_classes=16), "seg_isaid.safetensors"),
    "landcover":              (partial(DPT, num_classes=8), "seg_landcover.safetensors"),
    "water":                  (partial(DPT, num_classes=2), "seg_water.safetensors"),
    "aerial":                 (partial(DPT, num_classes=12), "seg_aerial.safetensors"),
    "landslide_ms":           (partial(WeTr, backbone="mit_b0", num_classes=2, in_channels=14), "seg_landslide_ms.safetensors"),
    "landslide_bitemporal":   (partial(SwinUperNet, in_channels=6, num_classes=2), "seg_landslide_bitemporal.safetensors"),
    "building_lowres":        (partial(HRNetSeg, num_classes=2), "seg_building_lowres.safetensors"),
    "lava":                   (partial(BitemporalDPT, num_classes=2), "seg_lava.safetensors"),
    "reconstruction_building": (partial(BitemporalDPT, num_classes=2), "seg_reconstruction_building.safetensors"),
    "reservoir":              (partial(WeTr, backbone="mit_b0", num_classes=2, in_channels=14), "seg_reservoir.safetensors"),
}


def load_model(name: str, checkpoint_dir: Path) -> torch.nn.Module:
    """Build ``name`` on CPU, load its released weights (strict) and return it in eval mode."""
    build, filename = MODEL_ZOO[name]
    path = Path(checkpoint_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Run `python scripts/prepare_data.py` first.")
    model = build()
    _load_safetensors(model, str(path), strict=True)
    return model.eval()


__all__ = ["MODEL_ZOO", "LARGE_IMAGE_PIXELS", "load_model", "predict_probs", "predict_labels_out_of_core",
           "sliding_window"]
