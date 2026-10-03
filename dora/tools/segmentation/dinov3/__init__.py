"""Vision Transformer backbone vendored from Meta's DINOv3 (github.com/facebookresearch/dinov3).

Only the modules needed to build the ViT-L/16 encoder are kept. This code is
distributed under the DINOv3 License Agreement; see ``LICENSE`` in this folder.
"""

from .vision_transformer import DinoVisionTransformer, vit_large

__all__ = ["DinoVisionTransformer", "vit_large"]
