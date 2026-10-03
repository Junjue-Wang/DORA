"""SegFormer (MiT-B0) for 14-band Landslide4Sense inputs (inference-only wrapper)."""
import torch.nn as nn
import torch.nn.functional as F

from . import mix_transformer
from .segformer_head import SegFormerHead


class WeTr(nn.Module):
    def __init__(self, backbone="mit_b0", num_classes=2, embedding_dim=256, in_channels=14):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.input_channels = in_channels

        self.encoder = getattr(mix_transformer, backbone)(in_chans=in_channels)
        self.in_channels = self.encoder.embed_dims
        self.decoder = SegFormerHead(
            feature_strides=self.feature_strides,
            in_channels=self.in_channels,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
        )
        # Auxiliary CAM classifier from training; unused at inference but part of the checkpoint.
        self.classifier = nn.Conv2d(self.in_channels[-1], self.num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.decoder(self.encoder(x))
        return F.interpolate(out, size=x.size()[2:], mode="bilinear", align_corners=False)
