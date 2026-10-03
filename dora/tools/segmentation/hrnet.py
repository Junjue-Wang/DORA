"""HRNet-W48 segmentation model reproducing the PaddleSeg HRNet of the SpaceNet-7 1st-place solution.

Backbone: HRNet-W48 (timm). Head: the four branch outputs are upsampled to 1/4 scale,
concatenated (720 channels) and projected to ``num_classes``; logits are returned at
input resolution. Weights are converted from the competition PaddlePaddle release.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class HRNetSeg(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        hrnet = timm.create_model("hrnet_w48", pretrained=pretrained)

        # Stem (stride-4 after two stride-2 convs)
        self.conv1 = hrnet.conv1
        self.bn1 = hrnet.bn1
        self.act1 = hrnet.act1
        self.conv2 = hrnet.conv2
        self.bn2 = hrnet.bn2
        self.act2 = hrnet.act2

        # Stage 1 bottleneck (keeps stride-4 resolution)
        self.layer1 = hrnet.layer1

        # Multi-resolution stages
        self.transition1 = hrnet.transition1  # 64ch -> [48ch@/4, 96ch@/8]
        self.stage2 = hrnet.stage2
        self.transition2 = hrnet.transition2  # -> [48, 96, 192]ch
        self.stage3 = hrnet.stage3
        self.transition3 = hrnet.transition3  # -> [48, 96, 192, 384]ch
        self.stage4 = hrnet.stage4

        # Segmentation head matching PaddleSeg HRNet:
        #   conv-2: Conv(720→720, 1x1) + BN + ReLU
        #   conv-1: Conv(720→num_classes, 1x1, no bias)
        self.seg_head = nn.Sequential(
            nn.Conv2d(720, 720, kernel_size=1, bias=False),   # conv-2
            nn.BatchNorm2d(720),
            nn.ReLU(inplace=True),
            nn.Conv2d(720, num_classes, kernel_size=1, bias=False),  # conv-1
        )

    @staticmethod
    def _apply_transition(xl, transition):
        """Apply a ModuleList of transition blocks.

        New branches are created from xl[-1]; existing branches use Identity
        (pass through xl[i] unchanged).
        """
        return [
            t(xl[-1]) if not isinstance(t, nn.Identity) else xl[i]
            for i, t in enumerate(transition)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]

        # Stem
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        # Stage 2: [x] -> transition1 -> [48@/4, 96@/8]
        xl = self._apply_transition([x], self.transition1)
        for module in self.stage2:
            xl = module(xl)

        # Stage 3: -> [48@/4, 96@/8, 192@/16]
        xl = self._apply_transition(xl, self.transition2)
        for module in self.stage3:
            xl = module(xl)

        # Stage 4: -> [48@/4, 96@/8, 192@/16, 384@/32]
        xl = self._apply_transition(xl, self.transition3)
        for module in self.stage4:
            xl = module(xl)

        # Upsample all branches to 1/4 resolution and concatenate
        target = xl[0].shape[2:]
        feats = [xl[0]] + [
            F.interpolate(f, size=target, mode="bilinear", align_corners=True)
            for f in xl[1:]
        ]
        x = torch.cat(feats, dim=1)  # (N, 720, H/4, W/4)

        x = self.seg_head(x)

        # Upsample to original resolution
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)
