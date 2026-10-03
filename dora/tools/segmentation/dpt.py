"""DINOv3 ViT-L/16 + DPT segmentation heads used by the DORA perception tools.

``BitemporalDPT`` fuses pre/post tokens for change-style tasks (building and road
damage); ``DPT`` is the single-image variant (iSAID objects, land cover, water,
aerial damage). Module/attribute names mirror the training code so the released
checkpoints load with ``strict=True``.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .dinov3 import vit_large

ENCODER_OUT_INDICES = (5, 11, 17, 23)
DPT_CHANNELS = [96, 192, 384, 768]


def build_encoder() -> nn.Module:
    encoder = vit_large(
        img_size=518,
        patch_size=16,
        layerscale_init=1e-5,
        mask_k_bias=True,
        norm_layer="layernorm",
        ffn_layer="mlp",
    )
    encoder.init_weights()
    return encoder


# --------------------------------------------------------------------------- DPT blocks


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.activation = activation

    def forward(self, x):
        out = self.conv1(self.activation(x))
        out = self.conv2(self.activation(out))
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation, align_corners=True):
        super().__init__()
        self.align_corners = align_corners
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)
        self.resConfUnit1 = ResidualConvUnit(features, activation)
        self.resConfUnit2 = ResidualConvUnit(features, activation)

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            if output.shape[-2:] != res.shape[-2:]:
                output = F.interpolate(output, size=res.shape[-2:], mode="bilinear", align_corners=self.align_corners)
            output = output + res
        output = self.resConfUnit2(output)
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        return self.out_conv(output)


class DPTHead(nn.Module):
    def __init__(self, num_classes, features=256, in_shape=DPT_CHANNELS):
        super().__init__()
        self.refinenet1 = FeatureFusionBlock(features, nn.ReLU(False))
        self.refinenet2 = FeatureFusionBlock(features, nn.ReLU(False))
        self.refinenet3 = FeatureFusionBlock(features, nn.ReLU(False))
        self.refinenet4 = FeatureFusionBlock(features, nn.ReLU(False))
        self.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )
        self.scratch = nn.Module()
        for i, channels in enumerate(in_shape, start=1):
            setattr(self.scratch, f"layer{i}_rn",
                    nn.Conv2d(channels, features, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x_list):
        layer_1, layer_2, layer_3, layer_4 = x_list
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)
        return self.output_conv(path_1)


# ------------------------------------------------------------------ token reassembly


class ProjectReadout(nn.Module):
    """Fuse the CLS token into every patch token."""

    def __init__(self, in_features, start_index=1):
        super().__init__()
        self.start_index = start_index
        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index:])
        return self.project(torch.cat((x[:, self.start_index:], readout), dim=-1))


class ReassembleLayer(nn.Module):
    """ViT tokens [B, 1+N, D] -> feature map [B, C, H', W'] at a given scale."""

    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.readout_oper = ProjectReadout(in_channels)
        self.conv_project = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        if scale_factor == 4.0:
            self.resize = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=4, padding=0)
        elif scale_factor == 2.0:
            self.resize = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0)
        elif scale_factor == 1.0:
            self.resize = nn.Identity()
        elif scale_factor == 0.5:
            self.resize = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError(f"Unsupported scale_factor: {scale_factor}")

    def forward(self, x):
        x = self.readout_oper(x).transpose(1, 2)
        b, c, n = x.shape
        h = w = math.isqrt(n)
        assert h * w == n, f"Number of patches {n} is not a perfect square"
        return self.resize(self.conv_project(x.reshape(b, c, h, w)))


class FeatureProcessor(nn.Module):
    def __init__(self, in_channels, out_channels=DPT_CHANNELS):
        super().__init__()
        self.reassemble_layers = nn.ModuleList([
            ReassembleLayer(in_channels, out_channels[i], scale)
            for i, scale in enumerate((4.0, 2.0, 1.0, 0.5))
        ])

    def forward(self, features):
        return [layer(feat) for layer, feat in zip(self.reassemble_layers, features)]


class DPTDecoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.feature_processor = FeatureProcessor(in_channels)
        self.dpt = DPTHead(num_classes)

    def forward(self, tokens):
        return self.dpt(self.feature_processor(tokens))


def _tokens_with_cls(layer_output):
    patch_tokens, cls_token = layer_output
    return torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)


# -------------------------------------------------------------------------- models


class BitemporalDPT(nn.Module):
    """Pre/post images stacked on channels (B, 6, H, W) -> per-pixel class probabilities."""

    def __init__(self, num_classes, embed_dim=1024):
        super().__init__()
        self.encoder = build_encoder()
        self.decoder = DPTDecoder(embed_dim, num_classes)
        self.temporal_fuse_list = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim * 4, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(True))
            for _ in range(4)
        ])

    def forward(self, x):
        x = rearrange(x, "b (t c) h w -> (b t) c h w", t=2)
        h, w = x.shape[-2:]
        layers = self.encoder.get_intermediate_layers(
            x=x, n=ENCODER_OUT_INDICES, reshape=False, return_class_token=True)
        fused = []
        for fuse, layer in zip(self.temporal_fuse_list, layers):
            t1, t2 = rearrange(_tokens_with_cls(layer), "(b t) s c -> t b s c", t=2)
            fused.append(fuse(torch.cat([torch.abs(t1 - t2), t1 * t2, t1, t2], dim=2)))
        out = F.interpolate(self.decoder(fused), size=(h, w), mode="bilinear", align_corners=True)
        return out.softmax(dim=1)


class DPT(nn.Module):
    """Single image (B, 3, H, W) -> per-pixel class probabilities."""

    def __init__(self, num_classes, neck_dim=512, embed_dim=1024):
        super().__init__()
        self.encoder = build_encoder()
        self.decoder = DPTDecoder(neck_dim, num_classes)
        self.neck_fuse_list = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, neck_dim), nn.LayerNorm(neck_dim), nn.ReLU(True))
            for _ in range(4)
        ])

    def forward(self, x):
        h, w = x.shape[-2:]
        layers = self.encoder.get_intermediate_layers(
            x=x, n=ENCODER_OUT_INDICES, reshape=False, return_class_token=True)
        neck = [fuse(_tokens_with_cls(layer)) for fuse, layer in zip(self.neck_fuse_list, layers)]
        out = F.interpolate(self.decoder(neck), size=(h, w), mode="bilinear", align_corners=True)
        return out.softmax(dim=1)
