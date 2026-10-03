"""Swin-T + UPerNet for bi-temporal landslide change detection (pre/post stacked, 6 channels)."""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models
from torchvision.models.swin_transformer import SwinTransformerBlock


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for scale in pool_scales
        ])
        concat_channels = in_channels + len(pool_scales) * out_channels
        self.bottleneck = ConvBNReLU(concat_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        h, w = x.shape[2:]
        pooled = [x] + [F.interpolate(stage(x), size=(h, w), mode="bilinear", align_corners=False)
                        for stage in self.stages]
        return self.bottleneck(torch.cat(pooled, dim=1))


class UPerNetDecoder(nn.Module):
    def __init__(self, in_channels_list, fpn_channels=256, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.ppm = PyramidPoolingModule(in_channels_list[-1], fpn_channels, pool_scales)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in in_channels_list[:-1]:
            self.lateral_convs.append(ConvBNReLU(in_channels, fpn_channels, kernel_size=1))
            self.fpn_convs.append(ConvBNReLU(fpn_channels, fpn_channels, kernel_size=3, padding=1))
        self.fusion = ConvBNReLU(fpn_channels * len(in_channels_list), fpn_channels, kernel_size=3, padding=1)

    def forward(self, features):
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features[:-1])]
        prev = self.ppm(features[-1])
        fpn_results = [prev]
        for lateral, fpn_conv in zip(reversed(laterals), reversed(self.fpn_convs)):
            prev = F.interpolate(prev, size=lateral.shape[2:], mode="bilinear", align_corners=False)
            prev = fpn_conv(lateral + prev)
            fpn_results.insert(0, prev)
        target_size = fpn_results[0].shape[2:]
        resized = [fpn_results[0]] + [F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
                                      for f in fpn_results[1:]]
        return self.fusion(torch.cat(resized, dim=1))


class SwinTransformerFeatureExtractor(nn.Module):
    """Collect the NCHW output of every Swin stage of a torchvision Swin model."""

    def __init__(self, swin_model: nn.Module):
        super().__init__()
        self.swin_model = swin_model

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for layer in self.swin_model.features:
            x = layer(x)
            if isinstance(layer, nn.Sequential) and any(isinstance(b, SwinTransformerBlock) for b in layer):
                features.append(x.permute(0, 3, 1, 2).contiguous())
        return features


class SwinUperNet(nn.Module):
    def __init__(self, in_channels=6, num_classes=2, backbone="swin_t", fpn_channels=256, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        # Architecture only: every weight is overwritten by the released checkpoint.
        self.backbone_model = getattr(tv_models, backbone)(weights=None)
        if in_channels != 3:
            self._adapt_patch_embed(self.backbone_model, in_channels)
        self.feature_extractor = SwinTransformerFeatureExtractor(self.backbone_model)

        with torch.no_grad():
            self.feature_extractor.eval()
            sample = self.feature_extractor(torch.zeros(1, in_channels, 224, 224))
            self.feature_extractor.train()
        self.decoder = UPerNetDecoder([f.shape[1] for f in sample], fpn_channels, pool_scales)
        self.classifier = nn.Sequential(
            ConvBNReLU(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.Conv2d(fpn_channels, num_classes, kernel_size=1),
        )

    @staticmethod
    def _adapt_patch_embed(backbone: nn.Module, in_channels: int) -> None:
        patch_embed = backbone.features[0]
        conv = patch_embed[0]
        patch_embed[0] = nn.Conv2d(in_channels, conv.out_channels, kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, bias=conv.bias is not None)

    def forward(self, x):
        size = x.shape[2:]
        logits = self.classifier(self.decoder(self.feature_extractor(x)))
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
