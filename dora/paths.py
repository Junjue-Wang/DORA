"""Filesystem layout of a prepared DORA data root.

    <DORA_DATA>/
    ├── tasks/          t1_*.json ... t5_*.json   (queries, gold trajectories, answers)
    ├── images/         source rasters and vector layers referenced by the tasks
    └── checkpoints/    perception model weights (*.safetensors)

The root defaults to ``<repo>/data`` and can be overridden with the ``DORA_DATA``
environment variable (inherited by the MCP tool servers).
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DORA_DATA", REPO_ROOT / "data")).expanduser().resolve()
TASKS_DIR = DATA_ROOT / "tasks"
IMAGES_DIR = DATA_ROOT / "images"
CHECKPOINTS_DIR = DATA_ROOT / "checkpoints"
