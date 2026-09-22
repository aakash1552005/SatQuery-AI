"""
=============================================================================
SATQUERY AI -- VRSBENCH BENCHMARK ADAPTER & COORDINATE NORMALIZER
=============================================================================
Provides:
  - normalize_grounding_coordinates(): converts between VRSBench tokenized
    0-100 coordinates, normalized float [0.0, 1.0], and absolute pixel coordinates.
  - VRSBenchManifest: parses official VRSBench evaluation JSON packages
    (VRSBench_EVAL_vqa.json, VRSBench_EVAL_referring.json, VRSBench_EVAL_Cap.json).
  - Strict governance: asserts evaluation-only role (training forbidden).
=============================================================================
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any


def normalize_grounding_coordinates(
    coords: Union[str, List[float], Tuple[float, ...]],
    source_format: Optional[str] = None,
    target_format: str = "normalized_float",
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    dataset_version: str = "v1.0"
) -> Dict[str, Any]:
    """
    Normalizes and converts grounding coordinates between VRSBench tokenized integer format,
    normalized floating-point coordinates, and pixel coordinates.

    Supported source formats:
      - 'token_0_100': String formatted as '{<ymin><xmin><ymax><xmax>}' with integers in [0, 100].
      - 'normalized_float': List of 4 floats [ymin, xmin, ymax, xmax] in [0.0, 1.0].
      - 'obj_corner': List of 8 floats [x0, y0, x1, y1, x2, y2, x3, y3] representing 4 corners.

    Supported target formats:
      - 'normalized_float': [ymin, xmin, ymax, xmax] with values in [0.0, 1.0].
      - 'token_0_100': '{<ymin><xmin><ymax><xmax>}' with values in [0, 100].
      - 'pixel_xyxy': [xmin, ymin, xmax, ymax] in absolute integer pixels (requires image_width/height).
      - 'pixel_corners': 4 corner points in pixel coordinates (requires image_width/height).
    """
    detected_source = source_format

    # Auto-detect source format if not explicitly provided
    if isinstance(coords, str):
        match = re.findall(r"<(\d+)>", coords)
        if len(match) == 4:
            detected_source = "token_0_100"
            raw_box = [int(v) for v in match]
            norm_box = [
                float(raw_box[0]) / 100.0,
                float(raw_box[1]) / 100.0,
                float(raw_box[2]) / 100.0,
                float(raw_box[3]) / 100.0,
            ]
        else:
            raise ValueError(f"Unrecognized coordinate token string format: {coords}")
    elif isinstance(coords, (list, tuple)):
        if len(coords) == 4:
            detected_source = "normalized_float"
            norm_box = [float(v) for v in coords]
        elif len(coords) == 8:
            detected_source = "obj_corner"
            # 8 floats: x0, y0, x1, y1, x2, y2, x3, y3
            xs = [float(coords[i]) for i in range(0, 8, 2)]
            ys = [float(coords[i]) for i in range(1, 8, 2)]
            norm_box = [min(ys), min(xs), max(ys), max(xs)]
        else:
            raise ValueError(f"Unexpected coordinate sequence length: {len(coords)} (expected 4 or 8)")
    else:
        raise TypeError(f"Unsupported coordinate type: {type(coords)}")

    # Clamp normalized box to [0.0, 1.0]
    ymin, xmin, ymax, xmax = [max(0.0, min(1.0, v)) for v in norm_box]

    # Convert to target format
    if target_format == "normalized_float":
        converted = [round(ymin, 6), round(xmin, 6), round(ymax, 6), round(xmax, 6)]
    elif target_format == "token_0_100":
        t_ymin = int(round(ymin * 100))
        t_xmin = int(round(xmin * 100))
        t_ymax = int(round(ymax * 100))
        t_xmax = int(round(xmax * 100))
        converted = f"{{<{t_ymin}><{t_xmin}><{t_ymax}><{t_xmax}>}}"
    elif target_format == "pixel_xyxy":
        if image_width is None or image_height is None:
            raise ValueError("Target format 'pixel_xyxy' requires image_width and image_height")
        px_xmin = int(round(xmin * image_width))
        px_ymin = int(round(ymin * image_height))
        px_xmax = int(round(xmax * image_width))
        px_ymax = int(round(ymax * image_height))
        converted = [px_xmin, px_ymin, px_xmax, px_ymax]
    else:
        raise ValueError(f"Unsupported target format: {target_format}")

    return {
        "source_coordinate_format": detected_source,
        "target_coordinate_format": target_format,
        "dataset_version": dataset_version,
        "normalized_bbox_ymin_xmin_ymax_xmax": [round(ymin, 6), round(xmin, 6), round(ymax, 6), round(xmax, 6)],
        "converted": converted
    }


class VRSBenchManifest:
    """
    Parser and validator for official VRSBench benchmark packages.
    Enforces evaluation-only policy.
    """
    def __init__(self, annotations_dir: Union[str, Path]):
        self.annotations_dir = Path(annotations_dir)
        self.role = "public_evaluation"
        self.training_allowed = False
        self.evaluation_allowed = True

    def validate_evaluation_policy(self):
        """Asserts that VRSBench is strictly used for evaluation, never training."""
        assert self.training_allowed is False, "VRSBench cannot be used for training or fine-tuning."
        assert self.evaluation_allowed is True, "VRSBench is approved for public evaluation."
        return True

    def load_summary(self) -> Dict[str, Any]:
        """Loads and validates summary of available VRSBench evaluation files."""
        summary = {
            "dataset_name": "VRSBench",
            "role": self.role,
            "training_allowed": self.training_allowed,
            "evaluation_allowed": self.evaluation_allowed,
            "annotations_dir": str(self.annotations_dir),
            "files": {},
            "counts": {}
        }

        vqa_path = self.annotations_dir / "VRSBench_EVAL_vqa.json"
        if vqa_path.exists():
            with open(vqa_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            summary["files"]["eval_vqa"] = str(vqa_path)
            summary["counts"]["eval_vqa"] = len(data)

        ref_path = self.annotations_dir / "VRSBench_EVAL_referring.json"
        if ref_path.exists():
            with open(ref_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            summary["files"]["eval_referring"] = str(ref_path)
            summary["counts"]["eval_referring"] = len(data)

        cap_path = self.annotations_dir / "VRSBench_EVAL_Cap.json"
        if cap_path.exists():
            with open(cap_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            summary["files"]["eval_captioning"] = str(cap_path)
            summary["counts"]["eval_captioning"] = len(data)

        train_path = self.annotations_dir / "VRSBench_train.json"
        if train_path.exists():
            summary["files"]["upstream_train_catalog"] = str(train_path)

        return summary
