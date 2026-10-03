"""Sliding-window inference for large rasters.

Numerically equivalent to ``AutoPatchSegm`` / ``AutoPatchMapping`` from the EVER
library (github.com/Z-Zheng/ever) that produced the paper results: identical window
grid and batch order, per-pixel averaging of overlapping patch predictions, and an
out-of-core (memmap) variant for very large scenes.
"""
import math
import tempfile
from typing import Callable

import numpy as np
import torch

# Scenes above this many pixels are merged out-of-core to bound host memory.
LARGE_IMAGE_PIXELS = 4000 * 4000


def sliding_window(input_size, kernel_size, stride):
    """Boxes ``[xmin, ymin, xmax, ymax]`` covering an image, last row/col snapped to the border."""
    ih, iw = input_size
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    assert ih > 0 and iw > 0 and kh > 0 and kw > 0 and sh > 0 and sw > 0

    kh = min(kh, ih)
    kw = min(kw, iw)
    num_rows = math.ceil((ih - kh) / sh) if math.ceil((ih - kh) / sh) * sh + kh >= ih else math.ceil((ih - kh) / sh) + 1
    num_cols = math.ceil((iw - kw) / sw) if math.ceil((iw - kw) / sw) * sw + kw >= iw else math.ceil((iw - kw) / sw) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = (x * sw).ravel()
    ymin = (y * sh).ravel()
    xmin_offset = np.where(xmin + kw > iw, iw - xmin - kw, np.zeros_like(xmin))
    ymin_offset = np.where(ymin + kh > ih, ih - ymin - kh, np.zeros_like(ymin))
    return np.stack([xmin + xmin_offset, ymin + ymin_offset,
                     np.minimum(xmin + kw, iw), np.minimum(ymin + kh, ih)], axis=1)


def _patch_outputs(model, image, kernel_size, stride, batch_size, preprocess, device):
    """Yield ``(probabilities [C, h, w] on CPU, box)`` for every window, in grid order."""
    boxes = sliding_window(image.shape[:2], kernel_size, stride)
    for start in range(0, len(boxes), batch_size):
        batch_boxes = boxes[start:start + batch_size]
        patches = torch.stack([
            torch.as_tensor(np.asarray(image[y0:y1, x0:x1])) for x0, y0, x1, y1 in batch_boxes
        ])
        out = model(preprocess(patches.to(device))).cpu()
        yield from zip(out, batch_boxes)


@torch.no_grad()
def predict_probs(model: torch.nn.Module, image, kernel_size, stride, *,
                  preprocess: Callable = lambda x: x.permute(0, 3, 1, 2).float(),
                  batch_size: int = 2, device: str = "cuda") -> torch.Tensor:
    """Average overlapping window predictions of an ``H x W x C`` image into a ``[K, H, W]`` tensor."""
    h, w = image.shape[:2]
    probs = counts = None
    for out, (x0, y0, x1, y1) in _patch_outputs(model, image, kernel_size, stride, batch_size, preprocess, device):
        if probs is None:
            probs = torch.zeros(out.size(0), h, w, dtype=torch.float32)
            counts = torch.zeros(h, w, dtype=torch.float32)
        counts[y0:y1, x0:x1] += 1
        probs[:, y0:y1, x0:x1] += out
    return probs / counts.unsqueeze_(0)


@torch.no_grad()
def predict_labels_out_of_core(model: torch.nn.Module, image, kernel_size, stride, *,
                               preprocess: Callable = lambda x: x.permute(0, 3, 1, 2).float(),
                               batch_size: int = 2, device: str = "cuda") -> np.ndarray:
    """Same as ``predict_probs(...).argmax(0)`` but accumulates on disk; returns a ``uint8`` label map."""
    h, w = image.shape[:2]
    with tempfile.TemporaryFile() as prob_file, tempfile.TemporaryFile() as count_file:
        probs = counts = None
        for out, (x0, y0, x1, y1) in _patch_outputs(model, image, kernel_size, stride, batch_size, preprocess, device):
            out = out.permute(1, 2, 0).numpy()
            if probs is None:
                probs = np.memmap(prob_file, shape=(h, w, out.shape[2]), dtype=np.float32, mode="w+")
                counts = np.memmap(count_file, shape=(h, w, 1), dtype=out.dtype, mode="w+")
            probs[y0:y1, x0:x1] += out
            counts[y0:y1, x0:x1] += 1.

        labels = np.empty((h, w), dtype=np.uint8)
        for x0, y0, x1, y1 in sliding_window((h, w), 1024, 1024):
            avg = np.asarray(probs[y0:y1, x0:x1], dtype=np.float32) / np.asarray(counts[y0:y1, x0:x1])
            if avg.shape[2] == 1:
                labels[y0:y1, x0:x1] = (avg > 0.5).squeeze(2)
            else:
                labels[y0:y1, x0:x1] = avg.argmax(axis=2)
        del probs, counts
    return labels
